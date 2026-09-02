#include "cuda_edge_elimination/elimination.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::uint64_t kMaxHtScanTargets = 1000000U;
constexpr std::size_t kMaxFusedCandidateTargets = 4096U;

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

  const auto total_start = std::chrono::steady_clock::now();
  // 整个 target 切片在 commit 前只读；两个强类型 binding 只跨该同步 scan 复用。
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
  scan.attempts.reserve(static_cast<std::size_t>(attempt_count));
  std::vector<HtRecursiveProof> proven;
  proven.reserve(static_cast<std::size_t>(attempt_count));

  std::vector<NodeEdge> target_edges;
  target_edges.reserve(static_cast<std::size_t>(attempt_count));
  for (std::uint64_t relative = 0U; relative < attempt_count; ++relative) {
    const std::uint64_t target_index = options.target_offset + relative;
    const Edge& edge =
        graph->edges[static_cast<std::size_t>(targets[static_cast<std::size_t>(target_index)])];
    target_edges.push_back({edge.u, edge.v});
  }

  std::optional<HtCdTargetBatchResult> prepared_candidates;
  if (options.fuse_target_candidates && target_edges.size() > 1U &&
      target_edges.size() <= kMaxFusedCandidateTargets) {
    const auto candidate_batch_start = std::chrono::steady_clock::now();
    ++scan.target_candidate_batches;
    scan.target_candidate_targets = target_edges.size();
    try {
      prepared_candidates = detail::EvaluateHtCdCandidatesForTargetsBoundToValidatedGraph(
          *graph, target_edges, options.wavefront_options.search_options.root_options,
          graph_validation_binding);
    } catch (const std::bad_alloc&) {
      prepared_candidates.reset();
    } catch (const std::length_error&) {
      prepared_candidates.reset();
    } catch (const std::exception&) {
      if (options.wavefront_options.search_options.root_options.candidate_backend ==
          PathCompatibilityBackend::kAuto) {
        HtShallowOptions cpu_options = options.wavefront_options.search_options.root_options;
        cpu_options.candidate_backend = PathCompatibilityBackend::kCpu;
        prepared_candidates = detail::EvaluateHtCdCandidatesForTargetsBoundToValidatedGraph(
            *graph, target_edges, cpu_options, graph_validation_binding);
      } else if (options.wavefront_options.search_options.root_options.candidate_backend !=
                 PathCompatibilityBackend::kCuda) {
        throw;
      }
      // 显式 CUDA 批次失败时保留原有逐目标 unresolved 语义，由下方旧路径逐项处理。
    }
    scan.target_candidate_batch_ms = ElapsedMilliseconds(candidate_batch_start);
    if (prepared_candidates.has_value()) {
      if (prepared_candidates->targets.size() != target_edges.size() ||
          !prepared_candidates->cpu_verified) {
        throw std::logic_error("跨目标 HT c,d 批次返回数量或 CPU 认证状态错误");
      }
      scan.target_candidate_screen_tasks = prepared_candidates->screen_tasks;
      scan.target_candidate_backend = prepared_candidates->backend;
      scan.target_candidate_selected_device = prepared_candidates->selected_device;
      scan.target_candidate_cpu_verified = prepared_candidates->cpu_verified;
    } else {
      scan.target_candidate_backend = "failed-sequential";
    }
    if (graph->ContentHash() != snapshot_hash) {
      throw std::logic_error("跨目标 HT c,d 候选批次修改了不可变快照");
    }
  }

  for (std::uint64_t relative = 0U; relative < attempt_count; ++relative) {
    const std::uint64_t target_index = options.target_offset + relative;
    const std::int32_t edge_id = targets[static_cast<std::size_t>(target_index)];
    const Edge& edge = graph->edges[static_cast<std::size_t>(edge_id)];
    const auto search_start = std::chrono::steady_clock::now();
    HtWavefrontResult wavefront;
    if (prepared_candidates.has_value()) {
      wavefront = detail::ProveEdgeByWavefrontHtWithPreparedCdCandidates(
          *graph, {edge.u, edge.v}, options.wavefront_options, snapshot_binding,
          graph_validation_binding,
          prepared_candidates->targets[static_cast<std::size_t>(relative)]);
    } else {
      wavefront = detail::ProveEdgeByWavefrontHtBoundToSnapshot(
          *graph, {edge.u, edge.v}, options.wavefront_options, snapshot_binding,
          graph_validation_binding);
    }
    // scan-wide 前置工作归入首个目标，使 attempts 的 candidate/search 总和继续等于 scan 总量。
    const double shared_candidate_ms = relative == 0U ? scan.target_candidate_batch_ms : 0.0;
    wavefront.candidate_ms += shared_candidate_ms;

    HtScanAttempt attempt;
    attempt.edge_id = edge_id;
    attempt.target_edge = {edge.u, edge.v};
    attempt.status = wavefront.status;
    attempt.states_expanded = wavefront.proof.states_expanded;
    attempt.replies_expanded = wavefront.proof.replies_expanded;
    attempt.leaf_calls = wavefront.proof.leaf_calls;
    attempt.moves_generated = wavefront.moves_generated;
    attempt.peak_frontier = wavefront.peak_frontier;
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
    attempt.path_append_tasks = wavefront.path_append_tasks;
    attempt.root_child_normalizations = wavefront.root_child_normalizations;
    attempt.point_candidate_scans = wavefront.point_candidate_scans;
    attempt.point_candidate_nodes_checked = wavefront.point_candidate_nodes_checked;
    attempt.point_candidate_nodes_ranked = wavefront.point_candidate_nodes_ranked;
    attempt.point_candidate_nodes_selected = wavefront.point_candidate_nodes_selected;
    attempt.hamilton_reply_batches = wavefront.hamilton_reply_batches;
    attempt.hamilton_reply_centers = wavefront.hamilton_reply_centers;
    attempt.hamilton_reply_unique_centers = wavefront.hamilton_reply_unique_centers;
    attempt.hamilton_reply_neighbor_pairs_tested = wavefront.hamilton_reply_neighbor_pairs_tested;
    attempt.hamilton_replies_generated = wavefront.hamilton_replies_generated;
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
    attempt.search_ms = ElapsedMilliseconds(search_start) + shared_candidate_ms;
    attempt.reason = wavefront.proof.reason;
    scan.search_ms += attempt.search_ms;
    scan.states_expanded += attempt.states_expanded;
    scan.replies_expanded += attempt.replies_expanded;
    scan.leaf_calls += attempt.leaf_calls;
    scan.moves_generated += attempt.moves_generated;
    scan.leaf_frontier_batches += attempt.leaf_frontier_batches;
    scan.leaf_frontier_states += attempt.leaf_frontier_states;
    scan.leaf_bucket_count += attempt.leaf_bucket_count;
    scan.peak_leaf_frontier_batch =
        std::max(scan.peak_leaf_frontier_batch, attempt.peak_leaf_frontier_batch);
    scan.leaf_cost_batches += attempt.leaf_cost_batches;
    scan.leaf_cost_tasks += attempt.leaf_cost_tasks;
    scan.leaf_cost_cells += attempt.leaf_cost_cells;
    scan.leaf_cursor_searches_started += attempt.leaf_cursor_searches_started;
    scan.leaf_cuda_cost_batches += attempt.leaf_cuda_cost_batches;
    scan.leaf_cpu_long_tail_cells += attempt.leaf_cpu_long_tail_cells;
    scan.leaf_cost_rows_consumed += attempt.leaf_cost_rows_consumed;
    scan.leaf_candidate_templates_rechecked += attempt.leaf_candidate_templates_rechecked;
    scan.leaf_cpu_completeness_rows += attempt.leaf_cpu_completeness_rows;
    scan.leaf_cpu_completeness_templates += attempt.leaf_cpu_completeness_templates;
    scan.leaf_cpu_certified_cost_cells += attempt.leaf_cpu_certified_cost_cells;
    scan.leaf_cpu_cost_rows_scored += attempt.leaf_cpu_cost_rows_scored;
    scan.leaf_cpu_cost_rows_reused += attempt.leaf_cpu_cost_rows_reused;
    scan.leaf_cpu_parallel_cost_batches += attempt.leaf_cpu_parallel_cost_batches;
    scan.leaf_cpu_parallel_cost_cells += attempt.leaf_cpu_parallel_cost_cells;
    scan.peak_leaf_cpu_cost_threads =
        std::max(scan.peak_leaf_cpu_cost_threads, attempt.peak_leaf_cpu_cost_threads);
    scan.peak_leaf_device_cache_bytes =
        std::max(scan.peak_leaf_device_cache_bytes, attempt.peak_leaf_device_cache_bytes);
    scan.root_child_normalizations += attempt.root_child_normalizations;
    scan.point_candidate_scans += attempt.point_candidate_scans;
    scan.point_candidate_nodes_checked += attempt.point_candidate_nodes_checked;
    scan.point_candidate_nodes_ranked += attempt.point_candidate_nodes_ranked;
    scan.point_candidate_nodes_selected += attempt.point_candidate_nodes_selected;
    scan.hamilton_reply_batches += attempt.hamilton_reply_batches;
    scan.hamilton_reply_centers += attempt.hamilton_reply_centers;
    scan.hamilton_reply_unique_centers += attempt.hamilton_reply_unique_centers;
    scan.hamilton_reply_neighbor_pairs_tested += attempt.hamilton_reply_neighbor_pairs_tested;
    scan.hamilton_replies_generated += attempt.hamilton_replies_generated;
    scan.candidate_ms += attempt.candidate_ms;
    scan.work_graph_ms += attempt.work_graph_ms;
    scan.root_child_normalize_ms += attempt.root_child_normalize_ms;
    scan.point_candidate_scan_ms += attempt.point_candidate_scan_ms;
    scan.point_candidate_sort_ms += attempt.point_candidate_sort_ms;
    scan.leaf_ms += attempt.leaf_ms;
    scan.leaf_setup_ms += attempt.leaf_setup_ms;
    scan.leaf_proof_initialize_ms += attempt.leaf_proof_initialize_ms;
    scan.leaf_coverage_scan_ms += attempt.leaf_coverage_scan_ms;
    scan.leaf_cursor_construct_ms += attempt.leaf_cursor_construct_ms;
    scan.leaf_cursor_prepare_ms += attempt.leaf_cursor_prepare_ms;
    scan.leaf_cost_evaluate_ms += attempt.leaf_cost_evaluate_ms;
    scan.leaf_cost_cpu_certify_ms += attempt.leaf_cost_cpu_certify_ms;
    scan.leaf_cost_scatter_ms += attempt.leaf_cost_scatter_ms;
    scan.leaf_cursor_consume_ms += attempt.leaf_cursor_consume_ms;
    scan.leaf_candidate_recheck_ms += attempt.leaf_candidate_recheck_ms;
    scan.leaf_completeness_fallback_ms += attempt.leaf_completeness_fallback_ms;
    scan.leaf_scalar_search_ms += attempt.leaf_scalar_search_ms;
    scan.leaf_apply_ms += attempt.leaf_apply_ms;
    scan.leaf_proof_verify_ms += attempt.leaf_proof_verify_ms;
    scan.path_append_ms += attempt.path_append_ms;
    scan.path_append_parent_prepare_ms += attempt.path_append_parent_prepare_ms;
    scan.path_append_child_normalize_ms += attempt.path_append_child_normalize_ms;
    scan.path_append_child_edges_ms += attempt.path_append_child_edges_ms;
    scan.path_append_cuda_evaluate_ms += attempt.path_append_cuda_evaluate_ms;
    scan.path_append_cuda_compare_ms += attempt.path_append_cuda_compare_ms;
    scan.hamilton_reply_ms += attempt.hamilton_reply_ms;
    scan.hamilton_reply_validation_ms += attempt.hamilton_reply_validation_ms;
    scan.hamilton_reply_cpu_enumerate_ms += attempt.hamilton_reply_cpu_enumerate_ms;
    scan.hamilton_reply_cuda_evaluate_ms += attempt.hamilton_reply_cuda_evaluate_ms;
    scan.hamilton_reply_cuda_compare_ms += attempt.hamilton_reply_cuda_compare_ms;
    scan.end_reply_ms += attempt.end_reply_ms;
    scan.propagation_ms += attempt.propagation_ms;
    scan.proof_extract_ms += attempt.proof_extract_ms;
    scan.proof_verify_ms += attempt.proof_verify_ms;

    if (wavefront.status == HtSearchStatus::kInvalid) {
      // 此时所有搜索都只读原图，尚未进入 commit；invalid 必须整批失败关闭。
      throw std::runtime_error("HT scan 目标 " + std::to_string(edge.u) + '-' +
                               std::to_string(edge.v) + " 返回 " + StatusName(wavefront.status) +
                               ": " + wavefront.proof.reason);
    }
    if (wavefront.status == HtSearchStatus::kProven) {
      std::string reason;
      const auto verify_start = std::chrono::steady_clock::now();
      const bool verified = VerifyHtRecursiveProof(*graph, wavefront.proof, &reason);
      attempt.immediate_verify_ms = ElapsedMilliseconds(verify_start);
      scan.immediate_verify_ms += attempt.immediate_verify_ms;
      if (!verified) {
        throw std::runtime_error("HT scan 成功 proof 即时 CPU 复核失败: " + reason);
      }
      ++scan.proven_targets;
      proven.push_back(std::move(wavefront.proof));
    } else {
      ++scan.unresolved_targets;
    }
    scan.attempts.push_back(std::move(attempt));

    if (graph->ContentHash() != snapshot_hash) {
      throw std::logic_error("HT scan 搜索阶段修改了不可变快照");
    }
  }

  const auto commit_start = std::chrono::steady_clock::now();
  scan.elimination = CommitHtProofEpoch(graph, proven);
  scan.commit_ms = ElapsedMilliseconds(commit_start);
  scan.elimination.backend = "ht-wavefront-scan-cpu-verified";
  scan.total_ms = ElapsedMilliseconds(total_start);
  return scan;
}

} // namespace cudaee
