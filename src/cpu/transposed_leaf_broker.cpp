#include "transposed_leaf_broker.hpp"

#include "cuda_edge_elimination/cuda_device_affinity.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cudaee::detail {

namespace {

// 全活跃 worker 栅栏会让已到达的 target 等待无关的主机控制流。这里只
// 等待两个同步请求；执行 GPU 批次期间新到请求会自然聚合成下一批，
// 既保留跨目标合并，又避免全局栅栏和长尾的单请求启动。
constexpr std::size_t kDispatchRequestThreshold = 2U;

} // namespace

class TransposedLeafBroker::Impl {
public:
  Impl(const GraphSnapshot& graph, const KOptSnapshotBinding& snapshot_binding,
       KOptSearchOptions options, const std::size_t worker_count, const int device_ordinal)
      : graph_(graph), snapshot_binding_(snapshot_binding), options_(std::move(options)),
        active_workers_(CheckedWorkerCount(worker_count)), device_ordinal_(device_ordinal),
        dispatcher_([this] { Dispatch(); }) {}

  ~Impl() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    dispatcher_.join();
  }

  [[nodiscard]] TransposedLeafBatchResponse Evaluate(std::vector<NormalizedPathSystem> states,
                                                     const NodeEdge required_edge) {
    if (states.empty()) {
      throw std::invalid_argument("transposed leaf broker 不接受空请求");
    }
    Request request{.states = std::move(states),
                    .required_edge = required_edge,
                    .response = {},
                    .failure = nullptr,
                    .done = false};
    std::unique_lock<std::mutex> lock(mutex_);
    if (fatal_failure_ != nullptr) {
      std::rethrow_exception(fatal_failure_);
    }
    if (stopping_ || active_workers_ == 0U) {
      throw std::logic_error("transposed leaf broker 已停止接收请求");
    }
    pending_.push_back(&request);
    condition_.notify_all();
    condition_.wait(lock, [&] { return request.done || fatal_failure_ != nullptr; });
    if (request.failure != nullptr) {
      std::rethrow_exception(request.failure);
    }
    if (!request.done) {
      std::rethrow_exception(fatal_failure_);
    }
    return std::move(request.response);
  }

  void FinishWorkers(const std::size_t count) noexcept {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      active_workers_ = count >= active_workers_ ? 0U : active_workers_ - count;
    }
    condition_.notify_all();
  }

  [[nodiscard]] std::uint32_t SuggestedSpeculationWidth() {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (active_workers_ >= 16U) {
      return 1U;
    }
    if (active_workers_ >= 8U) {
      return 2U;
    }
    if (active_workers_ >= 4U) {
      return 4U;
    }
    return 8U;
  }

private:
  static std::size_t CheckedWorkerCount(const std::size_t worker_count) {
    if (worker_count == 0U) {
      throw std::invalid_argument("transposed leaf broker 至少需要一个 worker");
    }
    return worker_count;
  }

  struct Request {
    std::vector<NormalizedPathSystem> states;
    NodeEdge required_edge;
    TransposedLeafBatchResponse response;
    std::exception_ptr failure;
    bool done{false};
  };

  [[nodiscard]] std::vector<TransposedLeafBatchResponse>
  EvaluateBatch(const std::vector<Request*>& requests) const {
    if (requests.empty()) {
      throw std::logic_error("transposed leaf broker 生成了空物理 batch");
    }
    std::size_t total_states = 0U;
    for (const Request* const request : requests) {
      if (request == nullptr || request->states.empty() ||
          request->states.size() > std::numeric_limits<std::size_t>::max() - total_states) {
        throw std::overflow_error("transposed leaf broker 状态数量非法或溢出");
      }
      total_states += request->states.size();
    }

    std::vector<NormalizedPathSystem> states;
    std::vector<std::optional<NodeEdge>> required_edges;
    states.reserve(total_states);
    required_edges.reserve(total_states);
    for (Request* const request : requests) {
      states.insert(states.end(), std::make_move_iterator(request->states.begin()),
                    std::make_move_iterator(request->states.end()));
      required_edges.insert(required_edges.end(), request->states.size(), request->required_edge);
    }

    PathSystemKOptBatchResult aggregate = ProvePathSystemsByKOptCandidateOnlyBoundToSnapshot(
        graph_, states, required_edges, snapshot_binding_, options_);
    if (!aggregate.cpu_verified || aggregate.proofs.size() != total_states) {
      throw std::logic_error("transposed leaf broker 未返回完整 CPU 认证结果");
    }

    const std::string backend = aggregate.cost_backend;
    const int selected_device = aggregate.selected_device;
    const bool cpu_verified = aggregate.cpu_verified;
    std::vector<PathSystemKOptProof> proofs = std::move(aggregate.proofs);
    aggregate.proofs.clear();

    std::vector<TransposedLeafBatchResponse> responses(requests.size());
    std::size_t proof_offset = 0U;
    for (std::size_t request_index = 0U; request_index < requests.size(); ++request_index) {
      const std::size_t proof_count = requests[request_index]->states.size();
      PathSystemKOptBatchResult& destination = responses[request_index].batch;
      if (request_index == 0U) {
        destination = std::move(aggregate);
        responses[request_index].physical_request_count =
            static_cast<std::uint64_t>(requests.size());
        responses[request_index].physical_state_count = static_cast<std::uint64_t>(total_states);
        responses[request_index].owns_physical_metrics = true;
      } else {
        destination.cost_backend = backend;
        destination.selected_device = selected_device;
        destination.cpu_verified = cpu_verified;
      }
      destination.proofs.reserve(proof_count);
      for (std::size_t local = 0U; local < proof_count; ++local) {
        destination.proofs.push_back(std::move(proofs[proof_offset + local]));
      }
      proof_offset += proof_count;
    }
    return responses;
  }

  void Fail(const std::exception_ptr failure) noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    fatal_failure_ = failure;
    for (Request* const request : pending_) {
      request->failure = failure;
      request->done = true;
    }
    pending_.clear();
    condition_.notify_all();
  }

  void Dispatch() noexcept {
    if (device_ordinal_ >= 0) {
      std::string reason;
      if (!SetCudaDevicePreferenceForCurrentThread(device_ordinal_, &reason)) {
        try {
          throw std::runtime_error("transposed leaf broker 无法绑定 CUDA device " +
                                   std::to_string(device_ordinal_) + ": " + reason);
        } catch (...) {
          Fail(std::current_exception());
        }
        return;
      }
    }

    while (true) {
      std::vector<Request*> requests;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
          const std::size_t ready_threshold = std::min(active_workers_, kDispatchRequestThreshold);
          return stopping_ || fatal_failure_ != nullptr ||
                 (!pending_.empty() && pending_.size() >= ready_threshold);
        });
        if (stopping_ || fatal_failure_ != nullptr) {
          return;
        }
        requests.swap(pending_);
      }

      try {
        std::vector<TransposedLeafBatchResponse> responses = EvaluateBatch(requests);
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          for (std::size_t index = 0U; index < requests.size(); ++index) {
            requests[index]->response = std::move(responses[index]);
            requests[index]->done = true;
          }
        }
        condition_.notify_all();
      } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          fatal_failure_ = failure;
          for (Request* const request : requests) {
            request->failure = failure;
            request->done = true;
          }
          for (Request* const request : pending_) {
            request->failure = failure;
            request->done = true;
          }
          pending_.clear();
        }
        condition_.notify_all();
        return;
      }
    }
  }

  const GraphSnapshot& graph_;
  const KOptSnapshotBinding& snapshot_binding_;
  KOptSearchOptions options_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<Request*> pending_;
  std::exception_ptr fatal_failure_;
  std::size_t active_workers_{};
  int device_ordinal_{-1};
  bool stopping_{false};
  std::jthread dispatcher_;
};

TransposedLeafBroker::TransposedLeafBroker(const GraphSnapshot& graph,
                                           const KOptSnapshotBinding& snapshot_binding,
                                           KOptSearchOptions options,
                                           const std::size_t worker_count, const int device_ordinal)
    : impl_(std::make_unique<Impl>(graph, snapshot_binding, std::move(options), worker_count,
                                   device_ordinal)) {}

TransposedLeafBroker::~TransposedLeafBroker() = default;

TransposedLeafBatchResponse TransposedLeafBroker::Evaluate(std::vector<NormalizedPathSystem> states,
                                                           const NodeEdge required_edge) {
  return impl_->Evaluate(std::move(states), required_edge);
}

std::uint32_t TransposedLeafBroker::SuggestedSpeculationWidth() {
  return impl_->SuggestedSpeculationWidth();
}

void TransposedLeafBroker::FinishWorkers(const std::size_t count) noexcept {
  impl_->FinishWorkers(count);
}

} // namespace cudaee::detail
