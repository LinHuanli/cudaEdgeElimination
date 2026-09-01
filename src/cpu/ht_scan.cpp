#include "cuda_edge_elimination/elimination.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::uint64_t kMaxHtScanTargets = 1000000U;

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

  const std::uint64_t snapshot_hash = graph->ContentHash();
  const std::vector<std::int32_t> targets = SelectHtTargetEdgeIds(*graph, options.target_order);
  if (options.target_offset > targets.size()) {
    throw std::invalid_argument("HT scan target_offset 超过 eligible target 数量");
  }

  HtScanResult scan;
  scan.eligible_targets = targets.size();
  scan.target_offset = options.target_offset;
  const std::uint64_t remaining = scan.eligible_targets - options.target_offset;
  const std::uint64_t attempt_count = std::min(options.max_targets, remaining);
  scan.attempts.reserve(static_cast<std::size_t>(attempt_count));
  std::vector<HtRecursiveProof> proven;
  proven.reserve(static_cast<std::size_t>(attempt_count));

  for (std::uint64_t relative = 0U; relative < attempt_count; ++relative) {
    const std::uint64_t target_index = options.target_offset + relative;
    const std::int32_t edge_id = targets[static_cast<std::size_t>(target_index)];
    const Edge& edge = graph->edges[static_cast<std::size_t>(edge_id)];
    const auto search_start = std::chrono::steady_clock::now();
    HtWavefrontResult wavefront =
        ProveEdgeByWavefrontHt(*graph, {edge.u, edge.v}, options.wavefront_options);

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
    attempt.leaf_cost_cells = wavefront.leaf_cost_cells;
    attempt.leaf_cuda_cost_batches = wavefront.leaf_cuda_cost_batches;
    attempt.leaf_cpu_long_tail_cells = wavefront.leaf_cpu_long_tail_cells;
    attempt.peak_leaf_device_cache_bytes = wavefront.peak_leaf_device_cache_bytes;
    attempt.path_append_tasks = wavefront.path_append_tasks;
    attempt.hamilton_replies_generated = wavefront.hamilton_replies_generated;
    attempt.end_replies_generated = wavefront.end_replies_generated;
    attempt.search_ms = ElapsedMilliseconds(search_start);
    attempt.reason = wavefront.proof.reason;
    scan.search_ms += attempt.search_ms;
    scan.states_expanded += attempt.states_expanded;
    scan.replies_expanded += attempt.replies_expanded;
    scan.leaf_calls += attempt.leaf_calls;
    scan.moves_generated += attempt.moves_generated;
    scan.leaf_cost_cells += attempt.leaf_cost_cells;
    scan.leaf_cuda_cost_batches += attempt.leaf_cuda_cost_batches;
    scan.leaf_cpu_long_tail_cells += attempt.leaf_cpu_long_tail_cells;
    scan.peak_leaf_device_cache_bytes =
        std::max(scan.peak_leaf_device_cache_bytes, attempt.peak_leaf_device_cache_bytes);

    if (wavefront.status == HtSearchStatus::kInvalid) {
      // 此时所有搜索都只读原图，尚未进入 commit；invalid 必须整批失败关闭。
      throw std::runtime_error("HT scan 目标 " + std::to_string(edge.u) + '-' +
                               std::to_string(edge.v) + " 返回 " + StatusName(wavefront.status) +
                               ": " + wavefront.proof.reason);
    }
    if (wavefront.status == HtSearchStatus::kProven) {
      std::string reason;
      if (!VerifyHtRecursiveProof(*graph, wavefront.proof, &reason)) {
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

  scan.elimination = CommitHtProofEpoch(graph, proven);
  scan.elimination.backend = "ht-wavefront-scan-cpu-verified";
  return scan;
}

} // namespace cudaee
