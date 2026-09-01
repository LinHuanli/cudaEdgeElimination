#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaee::detail {
namespace {

void CheckCuda(const cudaError_t status, const char* const operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (count_ != 0U) {
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(HT path-append)");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() { return data_; }
  [[nodiscard]] const T* get() const { return data_; }

  void CopyFromHost(const T* const source) {
    if (count_ != 0U) {
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count_, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D(HT path-append)");
    }
  }

  void CopyToHost(T* const destination) const {
    if (count_ != 0U) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(HT path-append)");
    }
  }

private:
  T* data_{nullptr};
  std::size_t count_{};
};

int SelectDevice(std::string* const reason) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    if (reason != nullptr) {
      *reason =
          count_status == cudaSuccess ? "没有可见 CUDA 设备" : cudaGetErrorString(count_status);
    }
    return -1;
  }
  int best_device = -1;
  std::size_t best_free_bytes = 0U;
  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        (best_device < 0 || free_bytes > best_free_bytes)) {
      best_device = device;
      best_free_bytes = free_bytes;
    }
  }
  if (best_device < 0) {
    if (reason != nullptr) {
      *reason = "所有可见 GPU 均无法查询显存";
    }
    return -1;
  }
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(HT path-append)");
  return best_device;
}

__global__ void EvaluateHtPathAppendsKernel(const HtPathStateSpan* const states,
                                            const HtPathNodeRecord* const nodes,
                                            const HtPathAppendTask* const tasks,
                                            const std::size_t task_count,
                                            std::uint8_t* const feasible) {
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count) {
    return;
  }
  const HtPathAppendTask task = tasks[task_index];
  const HtPathStateSpan state = states[task.parent_index];
  constexpr std::uint32_t kAbsentComponent = UINT32_MAX;
  std::uint32_t first_component = kAbsentComponent;
  std::uint32_t second_component = kAbsentComponent;
  std::uint8_t first_degree = 0U;
  std::uint8_t center_degree = 0U;
  std::uint8_t second_degree = 0U;
  for (std::uint32_t offset = 0U; offset < state.node_count; ++offset) {
    const HtPathNodeRecord record = nodes[state.node_begin + offset];
    if (record.node == task.first) {
      first_component = record.component;
      first_degree = record.degree;
    }
    if (record.node == task.center) {
      center_degree = record.degree;
    }
    if (record.node == task.second) {
      second_component = record.component;
      second_degree = record.degree;
    }
  }

  bool valid = false;
  if (task.kind == HtPathAppendKind::kPoint) {
    // center 是新节点；两个旧端点均至多再接受一条边，且不能闭合已有同一条链。
    valid = center_degree == 0U && first_degree < 2U && second_degree < 2U &&
            (first_component == kAbsentComponent || second_component == kAbsentComponent ||
             first_component != second_component);
  } else {
    // end move 必须从现有端点出发，并只能连接新节点或另一条链的端点。
    valid = first_degree == 1U && second_degree < 2U &&
            (second_component == kAbsentComponent || first_component != second_component);
  }
  feasible[task_index] = static_cast<std::uint8_t>(valid);
}

} // namespace

bool HtPathAppendCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::uint8_t> EvaluateHtPathAppendsCuda(const std::int32_t dimension,
                                                    const std::vector<HtPathStateSpan>& states,
                                                    const std::vector<HtPathNodeRecord>& nodes,
                                                    const std::vector<HtPathAppendTask>& tasks,
                                                    int* const selected_device) {
  if (dimension <= 0 || states.empty() || tasks.empty() ||
      states.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("CUDA HT path-append 数组规模非法");
  }

  std::size_t expected_node = 0U;
  for (const HtPathStateSpan& state : states) {
    if (state.node_begin != expected_node || state.node_count > nodes.size() - expected_node) {
      throw std::invalid_argument("CUDA HT path-append state span 非法");
    }
    for (std::uint32_t offset = 0U; offset < state.node_count; ++offset) {
      const HtPathNodeRecord& record = nodes[expected_node + offset];
      if (record.node < 0 || record.node >= dimension || record.component >= state.node_count ||
          (record.degree != 1U && record.degree != 2U)) {
        throw std::invalid_argument("CUDA HT path-append node record 非法");
      }
      for (std::uint32_t prior = 0U; prior < offset; ++prior) {
        if (nodes[state.node_begin + prior].node == record.node) {
          throw std::invalid_argument("CUDA HT path-append state 含重复节点");
        }
      }
    }
    expected_node += state.node_count;
  }
  if (expected_node != nodes.size()) {
    throw std::invalid_argument("CUDA HT path-append 含未引用 node record");
  }
  for (const HtPathAppendTask& task : tasks) {
    if (task.parent_index >= states.size() || task.first < 0 || task.first >= dimension ||
        task.second < 0 || task.second >= dimension || task.first == task.second ||
        (task.kind == HtPathAppendKind::kPoint &&
         (task.center < 0 || task.center >= dimension || task.center == task.first ||
          task.center == task.second)) ||
        (task.kind == HtPathAppendKind::kEnd && task.center != -1) ||
        (task.kind != HtPathAppendKind::kPoint && task.kind != HtPathAppendKind::kEnd)) {
      throw std::invalid_argument("CUDA HT path-append task 非法");
    }
  }

  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA HT path-append 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }

  constexpr std::size_t kThreads = 256U;
  const std::size_t blocks = (tasks.size() + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA HT path-append 网格过大");
  }

  DeviceBuffer<HtPathStateSpan> device_states(states.size());
  DeviceBuffer<HtPathNodeRecord> device_nodes(nodes.size());
  DeviceBuffer<HtPathAppendTask> device_tasks(tasks.size());
  DeviceBuffer<std::uint8_t> device_feasible(tasks.size());
  device_states.CopyFromHost(states.data());
  device_nodes.CopyFromHost(nodes.data());
  device_tasks.CopyFromHost(tasks.data());

  EvaluateHtPathAppendsKernel<<<static_cast<unsigned int>(blocks),
                                static_cast<unsigned int>(kThreads)>>>(
      device_states.get(), device_nodes.get(), device_tasks.get(), tasks.size(),
      device_feasible.get());
  CheckCuda(cudaGetLastError(), "EvaluateHtPathAppendsKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "EvaluateHtPathAppendsKernel synchronize");

  std::vector<std::uint8_t> feasible(tasks.size());
  device_feasible.CopyToHost(feasible.data());
  return feasible;
}

} // namespace cudaee::detail
