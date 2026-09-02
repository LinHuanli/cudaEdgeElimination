#include "cuda_edge_elimination/elimination.hpp"

#include "cuda_edge_elimination/cuda_device_affinity.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::uint64_t kMaxHtScanTargets = 1000000U;
constexpr std::size_t kMaxHtTargetWorkers = 32U;

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

const char* StatusName(const HtSearchStatus status) {
  switch (status) {
  case HtSearchStatus::kProven:
    return "proven";
  case HtSearchStatus::kUnresolved:
    return "unresolved";
  case HtSearchStatus::kInvalid:
    return "invalid";
  }
  return "unknown";
}

void ValidateTargetDevices(const std::vector<int>& devices) {
  if (devices.empty()) {
    return;
  }
  if (devices.size() > kMaxHtTargetWorkers) {
    throw std::invalid_argument("HT scan target_devices 最多允许 32 个 worker");
  }
  for (std::size_t index = 0U; index < devices.size(); ++index) {
    if (devices[index] < 0) {
      throw std::invalid_argument("HT scan target device ordinal 不得为负数");
    }
    const auto current = devices.begin() + static_cast<std::ptrdiff_t>(index);
    if (std::find(devices.begin(), current, devices[index]) != current) {
      throw std::invalid_argument("HT scan target_devices 不得包含重复 ordinal");
    }
  }

  std::string reason;
  const int visible_devices = detail::VisibleCudaDeviceCount(&reason);
  if (visible_devices <= 0) {
    throw std::runtime_error("HT scan target worker 无可用 CUDA 设备: " + reason);
  }
  for (const int device : devices) {
    if (device >= visible_devices) {
      throw std::invalid_argument("HT scan target device ordinal " + std::to_string(device) +
                                  " 超出当前可见范围 [0," + std::to_string(visible_devices - 1) +
                                  "]");
    }
  }
}

std::size_t ResolveTargetWorkerCount(const HtScanOptions& options,
                                     const std::uint64_t attempt_count) {
  if (attempt_count == 0U) {
    return 0U;
  }
  const std::size_t requested = options.target_workers != 0U
                                    ? static_cast<std::size_t>(options.target_workers)
                                    : std::max<std::size_t>(1U, options.target_devices.size());
  return std::min(requested, static_cast<std::size_t>(attempt_count));
}

struct HtTargetEvaluation {
  HtScanAttempt attempt;
  std::optional<HtRecursiveProof> proven_proof;
  std::optional<HtShortCircuitTrace> short_circuit_trace;
};

HtTargetEvaluation
EvaluateHtTarget(const GraphSnapshot& graph, const std::int32_t edge_id, const int assigned_device,
                 const HtWavefrontOptions& options,
                 const detail::KOptSnapshotBinding& snapshot_binding,
                 const detail::HtGraphValidationBinding& graph_validation_binding,
                 const std::uint64_t snapshot_hash) {
  const Edge& edge = graph.edges[static_cast<std::size_t>(edge_id)];
  const auto search_start = std::chrono::steady_clock::now();
  HtWavefrontResult wavefront =
      options.scheduler == HtScheduler::kTransposed
          ? detail::ProveEdgeByTransposedHtBoundToSnapshot(
                graph, {edge.u, edge.v}, options, snapshot_binding, graph_validation_binding)
          : detail::ProveEdgeByWavefrontHtBoundToSnapshot(
                graph, {edge.u, edge.v}, options, snapshot_binding, graph_validation_binding);

  HtTargetEvaluation evaluation;
  HtScanAttempt& attempt = evaluation.attempt;
  attempt.edge_id = edge_id;
  attempt.target_edge = {edge.u, edge.v};
  attempt.status = wavefront.status;
  attempt.states_expanded = wavefront.proof.states_expanded;
  attempt.replies_expanded = wavefront.proof.replies_expanded;
  attempt.leaf_calls = wavefront.proof.leaf_calls;
  attempt.moves_generated = wavefront.moves_generated;
  attempt.peak_frontier = wavefront.peak_frontier;
  attempt.assigned_device = assigned_device;
  attempt.propagation_backend = wavefront.propagation_backend;
  attempt.selected_device = wavefront.selected_device;
  attempt.propagation_blocks = wavefront.propagation_blocks;
  attempt.propagation_cooperative = wavefront.propagation_cooperative;
  attempt.propagation_cpu_verified = wavefront.cpu_verified;
  attempt.leaf_cost_backend = wavefront.leaf_cost_backend;
  attempt.leaf_cost_selected_device = wavefront.leaf_cost_selected_device;
  attempt.leaf_cpu_verified = wavefront.leaf_cpu_verified;
  attempt.leaf_frontier_batches = wavefront.leaf_frontier_batches;
  attempt.leaf_frontier_states = wavefront.leaf_frontier_states;
  attempt.leaf_bucket_count = wavefront.leaf_bucket_count;
  attempt.peak_leaf_frontier_batch = wavefront.peak_leaf_frontier_batch;
  attempt.leaf_cost_batches = wavefront.leaf_cost_batches;
  attempt.leaf_cost_tasks = wavefront.leaf_cost_tasks;
  attempt.leaf_cost_cells = wavefront.leaf_cost_cells;
  attempt.leaf_cursor_searches_started = wavefront.leaf_cursor_searches_started;
  attempt.leaf_cuda_cost_batches = wavefront.leaf_cuda_cost_batches;
  attempt.leaf_cpu_long_tail_cells = wavefront.leaf_cpu_long_tail_cells;
  attempt.leaf_cost_rows_consumed = wavefront.leaf_cost_rows_consumed;
  attempt.leaf_candidate_templates_rechecked = wavefront.leaf_candidate_templates_rechecked;
  attempt.leaf_cpu_completeness_rows = wavefront.leaf_cpu_completeness_rows;
  attempt.leaf_cpu_completeness_templates = wavefront.leaf_cpu_completeness_templates;
  attempt.leaf_cpu_certified_cost_cells = wavefront.leaf_cpu_certified_cost_cells;
  attempt.leaf_cpu_cost_rows_scored = wavefront.leaf_cpu_cost_rows_scored;
  attempt.leaf_cpu_cost_rows_reused = wavefront.leaf_cpu_cost_rows_reused;
  attempt.leaf_cpu_parallel_cost_batches = wavefront.leaf_cpu_parallel_cost_batches;
  attempt.leaf_cpu_parallel_cost_cells = wavefront.leaf_cpu_parallel_cost_cells;
  attempt.peak_leaf_cpu_cost_threads = wavefront.peak_leaf_cpu_cost_threads;
  attempt.peak_leaf_device_cache_bytes = wavefront.peak_leaf_device_cache_bytes;
  attempt.path_append_selected_device = wavefront.path_append_selected_device;
  attempt.path_append_tasks = wavefront.path_append_tasks;
  attempt.root_child_normalizations = wavefront.root_child_normalizations;
  attempt.point_candidate_scans = wavefront.point_candidate_scans;
  attempt.point_candidate_nodes_checked = wavefront.point_candidate_nodes_checked;
  attempt.point_candidate_nodes_ranked = wavefront.point_candidate_nodes_ranked;
  attempt.point_candidate_nodes_selected = wavefront.point_candidate_nodes_selected;
  attempt.hamilton_reply_selected_device = wavefront.hamilton_reply_selected_device;
  attempt.hamilton_reply_batches = wavefront.hamilton_reply_batches;
  attempt.hamilton_reply_centers = wavefront.hamilton_reply_centers;
  attempt.hamilton_reply_unique_centers = wavefront.hamilton_reply_unique_centers;
  attempt.hamilton_reply_neighbor_pairs_tested = wavefront.hamilton_reply_neighbor_pairs_tested;
  attempt.hamilton_replies_generated = wavefront.hamilton_replies_generated;
  attempt.reply_cuda_batches = wavefront.reply_cuda_batches;
  attempt.reply_cuda_tasks_submitted = wavefront.reply_cuda_tasks_submitted;
  attempt.reply_cuda_graph_cache_hits = wavefront.reply_cuda_graph_cache_hits;
  attempt.reply_cuda_workspace_cache_hits = wavefront.reply_cuda_workspace_cache_hits;
  attempt.peak_reply_device_cache_bytes = wavefront.peak_reply_device_cache_bytes;
  attempt.end_reply_selected_device = wavefront.end_reply_selected_device;
  attempt.end_reply_batches = wavefront.end_reply_batches;
  attempt.end_reply_tasks = wavefront.end_reply_tasks;
  attempt.end_reply_unique_tasks = wavefront.end_reply_unique_tasks;
  attempt.end_replies_generated = wavefront.end_replies_generated;
  attempt.candidate_ms = wavefront.candidate_ms;
  attempt.work_graph_ms = wavefront.work_graph_ms;
  attempt.root_child_normalize_ms = wavefront.root_child_normalize_ms;
  attempt.point_candidate_scan_ms = wavefront.point_candidate_scan_ms;
  attempt.point_candidate_sort_ms = wavefront.point_candidate_sort_ms;
  attempt.leaf_ms = wavefront.leaf_ms;
  attempt.leaf_setup_ms = wavefront.leaf_setup_ms;
  attempt.leaf_proof_initialize_ms = wavefront.leaf_proof_initialize_ms;
  attempt.leaf_coverage_scan_ms = wavefront.leaf_coverage_scan_ms;
  attempt.leaf_cursor_construct_ms = wavefront.leaf_cursor_construct_ms;
  attempt.leaf_cursor_prepare_ms = wavefront.leaf_cursor_prepare_ms;
  attempt.leaf_cost_evaluate_ms = wavefront.leaf_cost_evaluate_ms;
  attempt.leaf_cost_cpu_certify_ms = wavefront.leaf_cost_cpu_certify_ms;
  attempt.leaf_cost_scatter_ms = wavefront.leaf_cost_scatter_ms;
  attempt.leaf_cursor_consume_ms = wavefront.leaf_cursor_consume_ms;
  attempt.leaf_candidate_recheck_ms = wavefront.leaf_candidate_recheck_ms;
  attempt.leaf_completeness_fallback_ms = wavefront.leaf_completeness_fallback_ms;
  attempt.leaf_scalar_search_ms = wavefront.leaf_scalar_search_ms;
  attempt.leaf_apply_ms = wavefront.leaf_apply_ms;
  attempt.leaf_proof_verify_ms = wavefront.leaf_proof_verify_ms;
  attempt.path_append_ms = wavefront.path_append_ms;
  attempt.path_append_parent_prepare_ms = wavefront.path_append_parent_prepare_ms;
  attempt.path_append_child_normalize_ms = wavefront.path_append_child_normalize_ms;
  attempt.path_append_child_edges_ms = wavefront.path_append_child_edges_ms;
  attempt.path_append_cuda_evaluate_ms = wavefront.path_append_cuda_evaluate_ms;
  attempt.path_append_cuda_compare_ms = wavefront.path_append_cuda_compare_ms;
  attempt.hamilton_reply_ms = wavefront.hamilton_reply_ms;
  attempt.hamilton_reply_validation_ms = wavefront.hamilton_reply_validation_ms;
  attempt.hamilton_reply_cpu_enumerate_ms = wavefront.hamilton_reply_cpu_enumerate_ms;
  attempt.hamilton_reply_cuda_evaluate_ms = wavefront.hamilton_reply_cuda_evaluate_ms;
  attempt.hamilton_reply_cuda_compare_ms = wavefront.hamilton_reply_cuda_compare_ms;
  attempt.end_reply_ms = wavefront.end_reply_ms;
  attempt.propagation_ms = wavefront.propagation_ms;
  attempt.proof_extract_ms = wavefront.proof_extract_ms;
  attempt.proof_verify_ms = wavefront.proof_verify_ms;
  attempt.search_ms = ElapsedMilliseconds(search_start);
  attempt.reason = wavefront.proof.reason;

  if (wavefront.status == HtSearchStatus::kInvalid) {
    throw std::runtime_error("HT scan 目标 " + std::to_string(edge.u) + '-' +
                             std::to_string(edge.v) + " 返回 " + StatusName(wavefront.status) +
                             ": " + wavefront.proof.reason);
  }
  if (wavefront.status == HtSearchStatus::kProven) {
    evaluation.proven_proof.emplace(std::move(wavefront.proof));
  }
  if (!wavefront.short_circuit_trace.nodes.empty()) {
    evaluation.short_circuit_trace.emplace(std::move(wavefront.short_circuit_trace));
  }
  if (graph.ContentHash() != snapshot_hash) {
    throw std::logic_error("HT scan 搜索阶段修改了不可变快照");
  }
  return evaluation;
}

void AccumulateHtAttempt(HtScanResult* const scan, const HtScanAttempt& attempt) {
  scan->search_ms += attempt.search_ms;
  scan->states_expanded += attempt.states_expanded;
  scan->replies_expanded += attempt.replies_expanded;
  scan->leaf_calls += attempt.leaf_calls;
  scan->moves_generated += attempt.moves_generated;
  scan->leaf_frontier_batches += attempt.leaf_frontier_batches;
  scan->leaf_frontier_states += attempt.leaf_frontier_states;
  scan->leaf_bucket_count += attempt.leaf_bucket_count;
  scan->peak_leaf_frontier_batch =
      std::max(scan->peak_leaf_frontier_batch, attempt.peak_leaf_frontier_batch);
  scan->leaf_cost_batches += attempt.leaf_cost_batches;
  scan->leaf_cost_tasks += attempt.leaf_cost_tasks;
  scan->leaf_cost_cells += attempt.leaf_cost_cells;
  scan->leaf_cursor_searches_started += attempt.leaf_cursor_searches_started;
  scan->leaf_cuda_cost_batches += attempt.leaf_cuda_cost_batches;
  scan->leaf_cpu_long_tail_cells += attempt.leaf_cpu_long_tail_cells;
  scan->leaf_cost_rows_consumed += attempt.leaf_cost_rows_consumed;
  scan->leaf_candidate_templates_rechecked += attempt.leaf_candidate_templates_rechecked;
  scan->leaf_cpu_completeness_rows += attempt.leaf_cpu_completeness_rows;
  scan->leaf_cpu_completeness_templates += attempt.leaf_cpu_completeness_templates;
  scan->leaf_cpu_certified_cost_cells += attempt.leaf_cpu_certified_cost_cells;
  scan->leaf_cpu_cost_rows_scored += attempt.leaf_cpu_cost_rows_scored;
  scan->leaf_cpu_cost_rows_reused += attempt.leaf_cpu_cost_rows_reused;
  scan->leaf_cpu_parallel_cost_batches += attempt.leaf_cpu_parallel_cost_batches;
  scan->leaf_cpu_parallel_cost_cells += attempt.leaf_cpu_parallel_cost_cells;
  scan->peak_leaf_cpu_cost_threads =
      std::max(scan->peak_leaf_cpu_cost_threads, attempt.peak_leaf_cpu_cost_threads);
  scan->peak_leaf_device_cache_bytes =
      std::max(scan->peak_leaf_device_cache_bytes, attempt.peak_leaf_device_cache_bytes);
  scan->root_child_normalizations += attempt.root_child_normalizations;
  scan->point_candidate_scans += attempt.point_candidate_scans;
  scan->point_candidate_nodes_checked += attempt.point_candidate_nodes_checked;
  scan->point_candidate_nodes_ranked += attempt.point_candidate_nodes_ranked;
  scan->point_candidate_nodes_selected += attempt.point_candidate_nodes_selected;
  scan->hamilton_reply_batches += attempt.hamilton_reply_batches;
  scan->hamilton_reply_centers += attempt.hamilton_reply_centers;
  scan->hamilton_reply_unique_centers += attempt.hamilton_reply_unique_centers;
  scan->hamilton_reply_neighbor_pairs_tested += attempt.hamilton_reply_neighbor_pairs_tested;
  scan->hamilton_replies_generated += attempt.hamilton_replies_generated;
  scan->reply_cuda_batches += attempt.reply_cuda_batches;
  scan->reply_cuda_tasks_submitted += attempt.reply_cuda_tasks_submitted;
  scan->reply_cuda_graph_cache_hits += attempt.reply_cuda_graph_cache_hits;
  scan->reply_cuda_workspace_cache_hits += attempt.reply_cuda_workspace_cache_hits;
  scan->peak_reply_device_cache_bytes =
      std::max(scan->peak_reply_device_cache_bytes, attempt.peak_reply_device_cache_bytes);
  scan->end_reply_batches += attempt.end_reply_batches;
  scan->end_reply_tasks += attempt.end_reply_tasks;
  scan->end_reply_unique_tasks += attempt.end_reply_unique_tasks;
  scan->end_replies_generated += attempt.end_replies_generated;
  scan->candidate_ms += attempt.candidate_ms;
  scan->work_graph_ms += attempt.work_graph_ms;
  scan->root_child_normalize_ms += attempt.root_child_normalize_ms;
  scan->point_candidate_scan_ms += attempt.point_candidate_scan_ms;
  scan->point_candidate_sort_ms += attempt.point_candidate_sort_ms;
  scan->leaf_ms += attempt.leaf_ms;
  scan->leaf_setup_ms += attempt.leaf_setup_ms;
  scan->leaf_proof_initialize_ms += attempt.leaf_proof_initialize_ms;
  scan->leaf_coverage_scan_ms += attempt.leaf_coverage_scan_ms;
  scan->leaf_cursor_construct_ms += attempt.leaf_cursor_construct_ms;
  scan->leaf_cursor_prepare_ms += attempt.leaf_cursor_prepare_ms;
  scan->leaf_cost_evaluate_ms += attempt.leaf_cost_evaluate_ms;
  scan->leaf_cost_cpu_certify_ms += attempt.leaf_cost_cpu_certify_ms;
  scan->leaf_cost_scatter_ms += attempt.leaf_cost_scatter_ms;
  scan->leaf_cursor_consume_ms += attempt.leaf_cursor_consume_ms;
  scan->leaf_candidate_recheck_ms += attempt.leaf_candidate_recheck_ms;
  scan->leaf_completeness_fallback_ms += attempt.leaf_completeness_fallback_ms;
  scan->leaf_scalar_search_ms += attempt.leaf_scalar_search_ms;
  scan->leaf_apply_ms += attempt.leaf_apply_ms;
  scan->leaf_proof_verify_ms += attempt.leaf_proof_verify_ms;
  scan->path_append_ms += attempt.path_append_ms;
  scan->path_append_parent_prepare_ms += attempt.path_append_parent_prepare_ms;
  scan->path_append_child_normalize_ms += attempt.path_append_child_normalize_ms;
  scan->path_append_child_edges_ms += attempt.path_append_child_edges_ms;
  scan->path_append_cuda_evaluate_ms += attempt.path_append_cuda_evaluate_ms;
  scan->path_append_cuda_compare_ms += attempt.path_append_cuda_compare_ms;
  scan->hamilton_reply_ms += attempt.hamilton_reply_ms;
  scan->hamilton_reply_validation_ms += attempt.hamilton_reply_validation_ms;
  scan->hamilton_reply_cpu_enumerate_ms += attempt.hamilton_reply_cpu_enumerate_ms;
  scan->hamilton_reply_cuda_evaluate_ms += attempt.hamilton_reply_cuda_evaluate_ms;
  scan->hamilton_reply_cuda_compare_ms += attempt.hamilton_reply_cuda_compare_ms;
  scan->end_reply_ms += attempt.end_reply_ms;
  scan->propagation_ms += attempt.propagation_ms;
  scan->proof_extract_ms += attempt.proof_extract_ms;
  scan->proof_verify_ms += attempt.proof_verify_ms;
}

void ConsumeHtTargetEvaluation(HtTargetEvaluation evaluation, HtScanResult* const scan,
                               std::vector<HtRecursiveProof>* const proven) {
  HtScanAttempt& attempt = evaluation.attempt;
  AccumulateHtAttempt(scan, attempt);
  if (evaluation.proven_proof.has_value()) {
    // 此时仍是候选 sidecar；CommitHtProofEpoch 会在任何图修改前整批精确重放一次。
    ++scan->proven_targets;
    proven->push_back(std::move(*evaluation.proven_proof));
  } else {
    ++scan->unresolved_targets;
  }
  if (evaluation.short_circuit_trace.has_value()) {
    scan->short_circuit_traces.traces.push_back(std::move(*evaluation.short_circuit_trace));
  }
  scan->attempts.push_back(std::move(attempt));
}

} // namespace

std::vector<std::int32_t> SelectHtTargetEdgeIds(const GraphSnapshot& graph,
                                                const HtTargetOrder order) {
  if (order != HtTargetOrder::kCanonical && order != HtTargetOrder::kWeightDescending) {
    throw std::invalid_argument("未知 HT 目标排序策略");
  }
  if (graph.edges.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("HT scan 边数超过稳定 edge id 范围");
  }

  std::vector<std::int32_t> targets;
  targets.reserve(graph.ActiveEdgeCount());
  for (std::size_t edge_id = 0U; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    // 提交器必然拒绝把任一端点降至二度；搜索这些边只会浪费有界预算。
    if (edge.active && graph.Degree(edge.u) > 2 && graph.Degree(edge.v) > 2) {
      targets.push_back(static_cast<std::int32_t>(edge_id));
    }
  }

  std::sort(targets.begin(), targets.end(),
            [&](const std::int32_t lhs_id, const std::int32_t rhs_id) {
              const Edge& lhs = graph.edges[static_cast<std::size_t>(lhs_id)];
              const Edge& rhs = graph.edges[static_cast<std::size_t>(rhs_id)];
              if (order == HtTargetOrder::kWeightDescending && lhs.weight != rhs.weight) {
                return lhs.weight > rhs.weight;
              }
              return std::tie(lhs.u, lhs.v, lhs_id) < std::tie(rhs.u, rhs.v, rhs_id);
            });
  return targets;
}

HtScanResult RunHtScanEpoch(GraphSnapshot* const graph, const HtScanOptions& options) {
  if (graph == nullptr) {
    throw std::invalid_argument("HT scan 的图不能为空");
  }
  if (options.max_targets == 0U || options.max_targets > kMaxHtScanTargets) {
    throw std::invalid_argument("HT scan max_targets 必须位于 [1,1000000]");
  }
  if (options.target_workers > kMaxHtTargetWorkers) {
    throw std::invalid_argument("HT scan target_workers 最多允许 32 个 worker");
  }

  const auto total_start = std::chrono::steady_clock::now();
  ValidateTargetDevices(options.target_devices);
  if (options.wavefront_options.scheduler == HtScheduler::kTransposed &&
      options.target_devices.size() > 1U) {
    throw std::invalid_argument("转置 HT 当前只支持单 GPU，target_devices 最多一个 ordinal");
  }
  // 整个 target 切片在 commit 前只读；两个强类型 binding 可被只读 worker 共享。
  const detail::KOptSnapshotBinding snapshot_binding(*graph);
  const detail::HtGraphValidationBinding graph_validation_binding(*graph);
  const std::uint64_t snapshot_hash = snapshot_binding.snapshot_hash();
  const auto selection_start = std::chrono::steady_clock::now();
  const std::vector<std::int32_t> targets = SelectHtTargetEdgeIds(*graph, options.target_order);
  if (options.target_offset > targets.size()) {
    throw std::invalid_argument("HT scan target_offset 超过 eligible target 数量");
  }

  HtScanResult scan;
  scan.target_selection_ms = ElapsedMilliseconds(selection_start);
  scan.eligible_targets = targets.size();
  scan.target_offset = options.target_offset;
  const std::uint64_t remaining = scan.eligible_targets - options.target_offset;
  const std::uint64_t attempt_count = std::min(options.max_targets, remaining);
  const std::size_t worker_count = ResolveTargetWorkerCount(options, attempt_count);
  scan.target_workers = static_cast<std::uint32_t>(worker_count);
  scan.target_parallel = worker_count > 1U;
  scan.attempts.reserve(static_cast<std::size_t>(attempt_count));
  std::vector<HtRecursiveProof> proven;
  proven.reserve(static_cast<std::size_t>(attempt_count));

  const auto target_execution_start = std::chrono::steady_clock::now();
  if (worker_count <= 1U && options.target_devices.empty()) {
    // 默认路径保持历史 target 顺序和当前线程 CUDA 驻留缓存语义。
    for (std::uint64_t relative = 0U; relative < attempt_count; ++relative) {
      const std::uint64_t target_index = options.target_offset + relative;
      const std::int32_t edge_id = targets[static_cast<std::size_t>(target_index)];
      ConsumeHtTargetEvaluation(EvaluateHtTarget(*graph, edge_id, -1, options.wavefront_options,
                                                 snapshot_binding, graph_validation_binding,
                                                 snapshot_hash),
                                &scan, &proven);
    }
  } else if (attempt_count == 0U) {
    scan.target_workers = 0U;
  } else {
    std::vector<std::unique_ptr<HtTargetEvaluation>> evaluations(
        static_cast<std::size_t>(attempt_count));
    std::vector<std::exception_ptr> failures(static_cast<std::size_t>(attempt_count));
    // jthread 让线程构造中途失败时也能先 join 已启动 worker，避免 std::terminate。
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0U; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker] {
        // 连设备绑定和错误消息分配也纳入捕获，线程入口不得让异常逃逸并触发 terminate。
        std::size_t failure_slot = worker;
        try {
          const int device = options.target_devices.empty()
                                 ? -1
                                 : options.target_devices[worker % options.target_devices.size()];
          if (device >= 0) {
            std::string reason;
            if (!detail::SetCudaDevicePreferenceForCurrentThread(device, &reason)) {
              throw std::runtime_error("HT target worker 无法绑定 CUDA device " +
                                       std::to_string(device) + ": " + reason);
            }
          }
          for (std::size_t relative = worker; relative < evaluations.size();
               relative += worker_count) {
            failure_slot = relative;
            const std::uint64_t target_index =
                options.target_offset + static_cast<std::uint64_t>(relative);
            const std::int32_t edge_id = targets[static_cast<std::size_t>(target_index)];
            evaluations[relative] = std::make_unique<HtTargetEvaluation>(
                EvaluateHtTarget(*graph, edge_id, device, options.wavefront_options,
                                 snapshot_binding, graph_validation_binding, snapshot_hash));
          }
        } catch (...) {
          failures[failure_slot] = std::current_exception();
          // 同一 worker 后续 target 不再运行；整批仍等待其他只读 worker 安全结束。
        }
      });
    }
    for (std::jthread& worker : workers) {
      worker.join();
    }
    for (const std::exception_ptr& failure : failures) {
      if (failure != nullptr) {
        std::rethrow_exception(failure);
      }
    }
    for (std::unique_ptr<HtTargetEvaluation>& evaluation : evaluations) {
      if (evaluation == nullptr) {
        throw std::logic_error("HT target worker 未返回完整的静态切片结果");
      }
      ConsumeHtTargetEvaluation(std::move(*evaluation), &scan, &proven);
      evaluation.reset();
    }
  }
  scan.target_execution_ms = ElapsedMilliseconds(target_execution_start);

  if (graph->ContentHash() != snapshot_hash) {
    throw std::logic_error("HT scan 提交前快照发生变化");
  }
  const auto commit_start = std::chrono::steady_clock::now();
  scan.elimination = CommitHtProofEpoch(graph, proven);
  scan.commit_ms = ElapsedMilliseconds(commit_start);
  scan.elimination.backend = options.wavefront_options.scheduler == HtScheduler::kTransposed
                                 ? "ht-transposed-scan-cpu-verified"
                                 : "ht-wavefront-scan-cpu-verified";
  scan.total_ms = ElapsedMilliseconds(total_start);
  return scan;
}

} // namespace cudaee
