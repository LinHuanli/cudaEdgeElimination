#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

using SteadyClock = std::chrono::steady_clock;

double ElapsedMilliseconds(const SteadyClock::time_point begin) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - begin).count();
}

NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? NodeEdge{first, second} : NodeEdge{second, first};
}

NormalizedPathSystem AddPaths(const NormalizedPathSystem& state, const std::vector<Path>& additions,
                              const std::int32_t dimension) {
  std::vector<Path> raw = state.paths;
  raw.insert(raw.end(), additions.begin(), additions.end());
  return NormalizePathSystem(raw, dimension);
}

bool ValidateTarget(const GraphSnapshot& graph, const NodeEdge target, std::string* const reason) {
  if (target.u < 0 || target.v >= graph.dimension || target.u >= target.v ||
      !graph.HasActiveEdge(target.u, target.v)) {
    if (reason != nullptr) {
      *reason = "转置 HT 目标必须是规范的活动边";
    }
    return false;
  }
  return true;
}

std::vector<NodeEdge> EnumerateEndReplies(const GraphSnapshot& graph, const std::int32_t endpoint,
                                          const std::int32_t internal_neighbor) {
  std::vector<NodeEdge> replies;
  const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(endpoint)];
  const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(endpoint) + 1U];
  for (std::int32_t offset = begin; offset < end; ++offset) {
    const std::int32_t neighbor = graph.neighbors[static_cast<std::size_t>(offset)];
    if (neighbor != internal_neighbor) {
      replies.push_back(CanonicalEdge(endpoint, neighbor));
    }
  }
  return replies;
}

struct PointCandidate {
  std::int32_t node{-1};
  std::vector<HtNeighborPair> replies;
};

struct EndCandidate {
  std::int32_t endpoint{-1};
  std::int32_t internal_neighbor{-1};
  std::vector<NodeEdge> replies;
};

struct PointCandidateOrderCache {
  std::vector<std::int32_t> ordered_nodes;
  std::vector<std::uint8_t> eligible;
  std::vector<std::uint32_t> state_marks;
  std::uint32_t generation{};
  bool initialized{false};
};

struct PointCandidateSelection {
  std::vector<std::int32_t> nodes;
  std::uint64_t nodes_checked{};
  std::uint64_t nodes_ranked{};
  double scan_ms{};
  double sort_ms{};
};

PointCandidateSelection SelectPointCandidateNodes(const GraphSnapshot& graph, const NodeEdge target,
                                                  const NormalizedPathSystem& state,
                                                  const HtRecursiveOptions& options,
                                                  PointCandidateOrderCache* const cache) {
  struct RankedNode {
    std::int32_t node{};
    __int128 midpoint_score{};
  };
  PointCandidateSelection selection;
  if (!cache->initialized) {
    PointCandidateOrderCache prepared;
    const std::size_t dimension = static_cast<std::size_t>(graph.dimension);
    prepared.eligible.assign(dimension, 0U);
    prepared.state_marks.assign(dimension, 0U);
    std::vector<RankedNode> ranked;
    ranked.reserve(dimension);
    const SteadyClock::time_point scan_begin = SteadyClock::now();
    for (std::int32_t node = 0; node < graph.dimension; ++node) {
      if (node == target.u || node == target.v ||
          (options.root_options.max_candidate_degree != 0U &&
           static_cast<std::uint32_t>(graph.Degree(node)) >
               options.root_options.max_candidate_degree)) {
        continue;
      }
      const __int128 dx =
          static_cast<__int128>(2) * graph.points[static_cast<std::size_t>(node)].integer_x -
          graph.points[static_cast<std::size_t>(target.u)].integer_x -
          graph.points[static_cast<std::size_t>(target.v)].integer_x;
      const __int128 dy =
          static_cast<__int128>(2) * graph.points[static_cast<std::size_t>(node)].integer_y -
          graph.points[static_cast<std::size_t>(target.u)].integer_y -
          graph.points[static_cast<std::size_t>(target.v)].integer_y;
      ranked.push_back({node, dx * dx + dy * dy});
      prepared.eligible[static_cast<std::size_t>(node)] = 1U;
    }
    selection.scan_ms += ElapsedMilliseconds(scan_begin);
    const SteadyClock::time_point sort_begin = SteadyClock::now();
    std::sort(ranked.begin(), ranked.end(), [](const RankedNode& lhs, const RankedNode& rhs) {
      return std::tie(lhs.midpoint_score, lhs.node) < std::tie(rhs.midpoint_score, rhs.node);
    });
    prepared.ordered_nodes.reserve(ranked.size());
    for (const RankedNode& node : ranked) {
      prepared.ordered_nodes.push_back(node.node);
    }
    prepared.initialized = true;
    selection.sort_ms += ElapsedMilliseconds(sort_begin);
    *cache = std::move(prepared);
  }

  selection.nodes_checked = static_cast<std::uint64_t>(graph.dimension);
  if (cache->generation == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(cache->state_marks.begin(), cache->state_marks.end(), 0U);
    cache->generation = 1U;
  } else {
    ++cache->generation;
  }
  const SteadyClock::time_point scan_begin = SteadyClock::now();
  std::uint64_t excluded = 0U;
  for (const Path& path : state.paths) {
    for (const std::int32_t node : path) {
      const std::size_t index = static_cast<std::size_t>(node);
      if (cache->state_marks[index] != cache->generation) {
        cache->state_marks[index] = cache->generation;
        excluded += cache->eligible[index];
      }
    }
  }
  if (excluded > cache->ordered_nodes.size()) {
    throw std::logic_error("转置 HT point candidate 过滤计数非法");
  }
  selection.nodes_ranked = static_cast<std::uint64_t>(cache->ordered_nodes.size()) - excluded;
  selection.scan_ms += ElapsedMilliseconds(scan_begin);
  const SteadyClock::time_point select_begin = SteadyClock::now();
  const std::size_t limit =
      options.root_options.max_neighborhood == 0U
          ? cache->ordered_nodes.size()
          : std::min(cache->ordered_nodes.size(),
                     static_cast<std::size_t>(options.root_options.max_neighborhood));
  selection.nodes.reserve(limit);
  for (const std::int32_t node : cache->ordered_nodes) {
    if (cache->state_marks[static_cast<std::size_t>(node)] == cache->generation) {
      continue;
    }
    selection.nodes.push_back(node);
    if (options.root_options.max_neighborhood != 0U && selection.nodes.size() == limit) {
      break;
    }
  }
  selection.sort_ms += ElapsedMilliseconds(select_begin);
  return selection;
}

std::vector<PointCandidate>
BuildPointCandidates(const GraphSnapshot& graph, const NodeEdge target,
                     const NormalizedPathSystem& state, const HtRecursiveOptions& options,
                     PointCandidateOrderCache* const order_cache,
                     std::vector<std::optional<std::vector<HtNeighborPair>>>* reply_cache,
                     HtWavefrontResult* const metrics) {
  const PointCandidateSelection selection =
      SelectPointCandidateNodes(graph, target, state, options, order_cache);
  ++metrics->point_candidate_scans;
  metrics->point_candidate_nodes_checked += selection.nodes_checked;
  metrics->point_candidate_nodes_ranked += selection.nodes_ranked;
  metrics->point_candidate_nodes_selected += selection.nodes.size();
  metrics->point_candidate_scan_ms += selection.scan_ms;
  metrics->point_candidate_sort_ms += selection.sort_ms;
  std::vector<PointCandidate> candidates;
  candidates.reserve(selection.nodes.size());
  for (const std::int32_t node : selection.nodes) {
    std::optional<std::vector<HtNeighborPair>>& cached =
        reply_cache->at(static_cast<std::size_t>(node));
    if (!cached.has_value()) {
      cached = EnumerateHtHamiltonReplies(graph, target, node);
    }
    candidates.push_back({node, *cached});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const PointCandidate& lhs, const PointCandidate& rhs) {
              return std::tuple{lhs.replies.size(), lhs.node} <
                     std::tuple{rhs.replies.size(), rhs.node};
            });
  if (options.max_point_candidates != 0U && candidates.size() > options.max_point_candidates) {
    candidates.resize(options.max_point_candidates);
  }
  return candidates;
}

std::vector<EndCandidate>
BuildEndCandidates(const GraphSnapshot& graph, const NormalizedPathSystem& state,
                   const HtRecursiveOptions& options,
                   std::unordered_map<std::uint64_t, std::vector<NodeEdge>>* reply_cache) {
  std::vector<EndCandidate> candidates;
  candidates.reserve(2U * state.paths.size());
  const auto replies_for =
      [&](const std::int32_t endpoint,
          const std::int32_t internal_neighbor) -> const std::vector<NodeEdge>& {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(endpoint)) << 32U) |
        static_cast<std::uint32_t>(internal_neighbor);
    const auto [iterator, inserted] = reply_cache->try_emplace(key);
    if (inserted) {
      iterator->second = EnumerateEndReplies(graph, endpoint, internal_neighbor);
    }
    return iterator->second;
  };
  for (const Path& path : state.paths) {
    candidates.push_back({path.front(), path[1], replies_for(path.front(), path[1])});
    candidates.push_back(
        {path.back(), path[path.size() - 2U], replies_for(path.back(), path[path.size() - 2U])});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const EndCandidate& lhs, const EndCandidate& rhs) {
              return std::tuple{lhs.replies.size(), lhs.endpoint, lhs.internal_neighbor} <
                     std::tuple{rhs.replies.size(), rhs.endpoint, rhs.internal_neighbor};
            });
  if (options.max_end_candidates != 0U && candidates.size() > options.max_end_candidates) {
    candidates.resize(options.max_end_candidates);
  }
  return candidates;
}

struct TransposedContext {
  const GraphSnapshot* graph{};
  NodeEdge target;
  const HtRecursiveOptions* options{};
  const detail::KOptSnapshotBinding* snapshot_binding{};
  HtWavefrontResult* result{};
  std::uint32_t speculation_width{1U};
  PointCandidateOrderCache point_candidate_order;
  std::vector<std::optional<std::vector<HtNeighborPair>>> point_reply_cache;
  std::unordered_map<std::uint64_t, std::vector<NodeEdge>> end_reply_cache;
  bool budget_exhausted{false};
};

bool MoveReplyCountAllowed(const TransposedContext& context, const std::uint64_t count) {
  return context.options->max_replies_per_move == 0U ||
         count <= context.options->max_replies_per_move;
}

bool ConsumeReply(TransposedContext* const context) {
  HtRecursiveProof& proof = context->result->proof;
  if (context->options->max_total_replies != 0U &&
      proof.replies_expanded >= context->options->max_total_replies) {
    context->budget_exhausted = true;
    return false;
  }
  ++proof.replies_expanded;
  return true;
}

void AddMetric(std::uint64_t* const destination, const std::uint64_t value,
               const char* const description) {
  if (value > std::numeric_limits<std::uint64_t>::max() - *destination) {
    throw std::overflow_error(std::string("转置 HT ") + description + " 计数溢出");
  }
  *destination += value;
}

void RecordLeafBatch(TransposedContext* const context, const PathSystemKOptBatchResult& batch,
                     const std::size_t state_count) {
  HtWavefrontResult& result = *context->result;
  AddMetric(&result.leaf_frontier_batches, 1U, "leaf batch");
  AddMetric(&result.leaf_frontier_states, state_count, "leaf states");
  result.peak_leaf_frontier_batch =
      std::max(result.peak_leaf_frontier_batch, static_cast<std::uint64_t>(state_count));
  AddMetric(&result.leaf_cost_batches, batch.cost_batches, "cost batches");
  AddMetric(&result.leaf_cost_tasks, batch.cost_tasks, "cost tasks");
  AddMetric(&result.leaf_cost_cells, batch.cost_cells, "cost cells");
  AddMetric(&result.leaf_scalar_searches, batch.scalar_searches, "scalar searches");
  AddMetric(&result.leaf_cursor_searches_started, batch.cursor_searches_started, "cursor searches");
  AddMetric(&result.leaf_cuda_cost_batches, batch.cuda_cost_batches, "CUDA cost batches");
  AddMetric(&result.leaf_snapshot_cache_hits, batch.snapshot_cache_hits, "snapshot cache hits");
  AddMetric(&result.leaf_template_cache_hits, batch.template_cache_hits, "template cache hits");
  AddMetric(&result.leaf_workspace_cache_hits, batch.workspace_cache_hits, "workspace cache hits");
  AddMetric(&result.leaf_cpu_long_tail_batches, batch.cpu_long_tail_batches,
            "CPU long-tail batches");
  AddMetric(&result.leaf_cpu_long_tail_tasks, batch.cpu_long_tail_tasks, "CPU long-tail tasks");
  AddMetric(&result.leaf_cpu_long_tail_cells, batch.cpu_long_tail_cells, "CPU long-tail cells");
  AddMetric(&result.leaf_cost_rows_consumed, batch.cost_rows_consumed, "cost rows");
  AddMetric(&result.leaf_candidate_templates_rechecked, batch.candidate_templates_rechecked,
            "candidate templates");
  AddMetric(&result.leaf_cpu_completeness_rows, batch.cpu_completeness_rows, "completeness rows");
  AddMetric(&result.leaf_cpu_completeness_templates, batch.cpu_completeness_templates,
            "completeness templates");
  AddMetric(&result.leaf_cpu_certified_cost_cells, batch.cpu_certified_cost_cells,
            "certified cells");
  AddMetric(&result.leaf_cpu_cost_rows_scored, batch.cpu_cost_rows_scored, "CPU rows scored");
  AddMetric(&result.leaf_cpu_cost_rows_reused, batch.cpu_cost_rows_reused, "CPU rows reused");
  AddMetric(&result.leaf_cpu_parallel_cost_batches, batch.cpu_parallel_cost_batches,
            "parallel batches");
  AddMetric(&result.leaf_cpu_parallel_cost_cells, batch.cpu_parallel_cost_cells, "parallel cells");
  result.peak_leaf_cpu_cost_threads =
      std::max(result.peak_leaf_cpu_cost_threads, batch.peak_cpu_cost_threads);
  result.peak_leaf_device_cache_bytes =
      std::max(result.peak_leaf_device_cache_bytes, batch.peak_device_cache_bytes);
  result.leaf_setup_ms += batch.setup_ms;
  result.leaf_proof_initialize_ms += batch.proof_initialize_ms;
  result.leaf_coverage_scan_ms += batch.coverage_scan_ms;
  result.leaf_cursor_construct_ms += batch.cursor_construct_ms;
  result.leaf_cursor_prepare_ms += batch.cursor_prepare_ms;
  result.leaf_cost_evaluate_ms += batch.cost_evaluate_ms;
  result.leaf_cost_cpu_certify_ms += batch.cost_cpu_certify_ms;
  result.leaf_cost_scatter_ms += batch.cost_scatter_ms;
  result.leaf_cursor_consume_ms += batch.cursor_consume_ms;
  result.leaf_candidate_recheck_ms += batch.candidate_recheck_ms;
  result.leaf_completeness_fallback_ms += batch.completeness_fallback_ms;
  result.leaf_scalar_search_ms += batch.scalar_search_ms;
  result.leaf_apply_ms += batch.apply_ms;
  result.leaf_proof_verify_ms += batch.proof_verify_ms;
  result.leaf_cpu_verified = result.leaf_frontier_batches == 1U
                                 ? batch.cpu_verified
                                 : result.leaf_cpu_verified && batch.cpu_verified;
  if (result.leaf_cost_backend == "none") {
    result.leaf_cost_backend = batch.cost_backend;
  } else if (result.leaf_cost_backend != batch.cost_backend) {
    result.leaf_cost_backend = "mixed";
  }
  if (batch.selected_device >= 0) {
    result.leaf_cost_selected_device = batch.selected_device;
  }
}

std::vector<PathSystemKOptProof>
EvaluateLeafWindow(TransposedContext* const context,
                   const std::vector<NormalizedPathSystem>& states) {
  if (states.empty()) {
    return {};
  }
  const SteadyClock::time_point begin = SteadyClock::now();
  const PathSystemKOptBatchResult batch = detail::ProvePathSystemsByKOptBoundToSnapshot(
      *context->graph, states, context->target, *context->snapshot_binding,
      context->options->root_options.leaf_options);
  context->result->leaf_ms += ElapsedMilliseconds(begin);
  if (batch.proofs.size() != states.size() || !batch.cpu_verified) {
    throw std::logic_error("转置 HT leaf batch 未返回完整 CPU 认证结果");
  }
  RecordLeafBatch(context, batch, states.size());
  return batch.proofs;
}

std::optional<std::uint32_t> ProvePreparedState(TransposedContext* context,
                                                NormalizedPathSystem state, std::uint32_t depth,
                                                PathSystemKOptProof leaf);

std::uint64_t CountUnusedValid(const std::vector<std::optional<std::size_t>>& proof_indices,
                               const std::size_t begin) {
  return static_cast<std::uint64_t>(
      std::count_if(proof_indices.begin() + static_cast<std::ptrdiff_t>(begin), proof_indices.end(),
                    [](const std::optional<std::size_t>& value) { return value.has_value(); }));
}

bool TryPointMove(TransposedContext* const context, const std::uint32_t node_index,
                  const NormalizedPathSystem& state, const PointCandidate& candidate,
                  const std::uint32_t depth) {
  if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
    return false;
  }
  AddMetric(&context->result->moves_generated, 1U, "point moves");
  const std::size_t checkpoint = context->result->proof.nodes.size();
  std::vector<HtTreeReply> records;
  records.reserve(candidate.replies.size());
  for (std::size_t begin = 0U; begin < candidate.replies.size();
       begin += context->speculation_width) {
    const std::size_t end =
        begin + std::min<std::size_t>(context->speculation_width, candidate.replies.size() - begin);
    std::vector<NormalizedPathSystem> valid_states;
    std::vector<std::optional<std::size_t>> proof_indices(end - begin);
    std::vector<NormalizedPathSystem> normalized(end - begin);
    for (std::size_t index = begin; index < end; ++index) {
      const HtNeighborPair& reply = candidate.replies[index];
      normalized[index - begin] =
          AddPaths(state, {{reply.first, reply.center, reply.second}}, context->graph->dimension);
      if (normalized[index - begin].valid) {
        proof_indices[index - begin] = valid_states.size();
        valid_states.push_back(normalized[index - begin]);
      }
    }
    std::vector<PathSystemKOptProof> leaf_proofs = EvaluateLeafWindow(context, valid_states);
    for (std::size_t index = begin; index < end; ++index) {
      if (!ConsumeReply(context)) {
        AddMetric(&context->result->speculative_leaf_tasks,
                  CountUnusedValid(proof_indices, index - begin), "speculative leaves");
        context->result->proof.nodes.resize(checkpoint);
        return false;
      }
      HtTreeReply record;
      record.first_pair = candidate.replies[index];
      const std::optional<std::size_t> proof_index = proof_indices[index - begin];
      if (!proof_index.has_value()) {
        record.path_infeasible = true;
        records.push_back(std::move(record));
        continue;
      }
      const std::optional<std::uint32_t> child =
          ProvePreparedState(context, std::move(normalized[index - begin]), depth + 1U,
                             std::move(leaf_proofs[*proof_index]));
      if (!child.has_value()) {
        AddMetric(&context->result->speculative_leaf_tasks,
                  CountUnusedValid(proof_indices, index - begin + 1U), "speculative leaves");
        if (index + 1U < candidate.replies.size()) {
          AddMetric(&context->result->short_circuits, 1U, "short circuits");
        }
        context->result->proof.nodes.resize(checkpoint);
        return false;
      }
      record.child_index = *child;
      records.push_back(std::move(record));
    }
  }
  HtTreeNode& node = context->result->proof.nodes[node_index];
  node.move_type = HtMoveType::kPoint;
  node.move_first = candidate.node;
  node.move_second = -1;
  node.leaf_proof = {};
  node.replies = std::move(records);
  return true;
}

bool TryEndMove(TransposedContext* const context, const std::uint32_t node_index,
                const NormalizedPathSystem& state, const EndCandidate& candidate,
                const std::uint32_t depth) {
  if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
    return false;
  }
  AddMetric(&context->result->moves_generated, 1U, "end moves");
  const std::size_t checkpoint = context->result->proof.nodes.size();
  std::vector<HtTreeReply> records;
  records.reserve(candidate.replies.size());
  for (std::size_t begin = 0U; begin < candidate.replies.size();
       begin += context->speculation_width) {
    const std::size_t end =
        begin + std::min<std::size_t>(context->speculation_width, candidate.replies.size() - begin);
    std::vector<NormalizedPathSystem> valid_states;
    std::vector<std::optional<std::size_t>> proof_indices(end - begin);
    std::vector<NormalizedPathSystem> normalized(end - begin);
    for (std::size_t index = begin; index < end; ++index) {
      const NodeEdge edge = candidate.replies[index];
      const std::int32_t neighbor = edge.u == candidate.endpoint ? edge.v : edge.u;
      normalized[index - begin] =
          AddPaths(state, {{candidate.endpoint, neighbor}}, context->graph->dimension);
      if (normalized[index - begin].valid) {
        proof_indices[index - begin] = valid_states.size();
        valid_states.push_back(normalized[index - begin]);
      }
    }
    std::vector<PathSystemKOptProof> leaf_proofs = EvaluateLeafWindow(context, valid_states);
    for (std::size_t index = begin; index < end; ++index) {
      if (!ConsumeReply(context)) {
        AddMetric(&context->result->speculative_leaf_tasks,
                  CountUnusedValid(proof_indices, index - begin), "speculative leaves");
        context->result->proof.nodes.resize(checkpoint);
        return false;
      }
      HtTreeReply record;
      record.edge = candidate.replies[index];
      const std::optional<std::size_t> proof_index = proof_indices[index - begin];
      if (!proof_index.has_value()) {
        record.path_infeasible = true;
        records.push_back(std::move(record));
        continue;
      }
      const std::optional<std::uint32_t> child =
          ProvePreparedState(context, std::move(normalized[index - begin]), depth + 1U,
                             std::move(leaf_proofs[*proof_index]));
      if (!child.has_value()) {
        AddMetric(&context->result->speculative_leaf_tasks,
                  CountUnusedValid(proof_indices, index - begin + 1U), "speculative leaves");
        if (index + 1U < candidate.replies.size()) {
          AddMetric(&context->result->short_circuits, 1U, "short circuits");
        }
        context->result->proof.nodes.resize(checkpoint);
        return false;
      }
      record.child_index = *child;
      records.push_back(std::move(record));
    }
  }
  HtTreeNode& node = context->result->proof.nodes[node_index];
  node.move_type = HtMoveType::kEnd;
  node.move_first = candidate.endpoint;
  node.move_second = candidate.internal_neighbor;
  node.leaf_proof = {};
  node.replies = std::move(records);
  return true;
}

std::optional<std::uint32_t> ProvePreparedState(TransposedContext* const context,
                                                NormalizedPathSystem state,
                                                const std::uint32_t depth,
                                                PathSystemKOptProof leaf) {
  HtRecursiveProof& proof = context->result->proof;
  if ((context->options->max_states != 0U &&
       proof.states_expanded >= context->options->max_states) ||
      proof.nodes.size() >= std::numeric_limits<std::uint32_t>::max()) {
    context->budget_exhausted = true;
    return std::nullopt;
  }
  ++proof.states_expanded;
  ++proof.leaf_calls;
  AddMetric(&context->result->continuations_created, 1U, "continuations");
  const std::uint32_t node_index = static_cast<std::uint32_t>(proof.nodes.size());
  HtTreeNode node;
  node.paths = std::move(state);
  proof.nodes.push_back(std::move(node));
  if (leaf.proven) {
    proof.nodes[node_index].move_type = HtMoveType::kLeaf;
    proof.nodes[node_index].leaf_proof = std::move(leaf);
    return node_index;
  }
  if (depth >= context->options->max_depth) {
    proof.nodes.resize(node_index);
    return std::nullopt;
  }

  const NormalizedPathSystem state_copy = proof.nodes[node_index].paths;
  if (context->options->enable_point_moves) {
    for (const PointCandidate& candidate : BuildPointCandidates(
             *context->graph, context->target, state_copy, *context->options,
             &context->point_candidate_order, &context->point_reply_cache, context->result)) {
      if (TryPointMove(context, node_index, state_copy, candidate, depth)) {
        return node_index;
      }
      if (context->budget_exhausted) {
        proof.nodes.resize(node_index);
        return std::nullopt;
      }
    }
  }
  if (context->options->enable_end_moves) {
    for (const EndCandidate& candidate : BuildEndCandidates(
             *context->graph, state_copy, *context->options, &context->end_reply_cache)) {
      if (TryEndMove(context, node_index, state_copy, candidate, depth)) {
        return node_index;
      }
      if (context->budget_exhausted) {
        proof.nodes.resize(node_index);
        return std::nullopt;
      }
    }
  }
  proof.nodes.resize(node_index);
  return std::nullopt;
}

bool TryRootMove(TransposedContext* const context, const HtCdCandidate& candidate,
                 const NormalizedPathSystem& root_state) {
  const std::vector<HtNeighborPair> first_replies =
      EnumerateHtHamiltonReplies(*context->graph, context->target, candidate.c);
  const std::vector<HtNeighborPair> second_replies =
      EnumerateHtHamiltonReplies(*context->graph, context->target, candidate.d);
  if (second_replies.size() != 0U &&
      first_replies.size() > std::numeric_limits<std::uint64_t>::max() / second_replies.size()) {
    return false;
  }
  const std::uint64_t reply_count =
      static_cast<std::uint64_t>(first_replies.size()) * second_replies.size();
  if ((context->options->root_options.max_reply_combinations != 0U &&
       reply_count > context->options->root_options.max_reply_combinations) ||
      !MoveReplyCountAllowed(*context, reply_count)) {
    return false;
  }
  AddMetric(&context->result->moves_generated, 1U, "root moves");

  struct RootReply {
    HtNeighborPair first;
    HtNeighborPair second;
  };
  std::vector<RootReply> replies;
  replies.reserve(static_cast<std::size_t>(reply_count));
  for (const HtNeighborPair& first : first_replies) {
    for (const HtNeighborPair& second : second_replies) {
      replies.push_back({first, second});
    }
  }
  const std::size_t checkpoint = context->result->proof.nodes.size();
  std::vector<HtTreeReply> records;
  records.reserve(replies.size());
  for (std::size_t begin = 0U; begin < replies.size(); begin += context->speculation_width) {
    const std::size_t end =
        begin + std::min<std::size_t>(context->speculation_width, replies.size() - begin);
    std::vector<NormalizedPathSystem> valid_states;
    std::vector<std::optional<std::size_t>> proof_indices(end - begin);
    std::vector<NormalizedPathSystem> normalized(end - begin);
    for (std::size_t index = begin; index < end; ++index) {
      const RootReply& reply = replies[index];
      normalized[index - begin] =
          AddPaths(root_state,
                   {{reply.first.first, reply.first.center, reply.first.second},
                    {reply.second.first, reply.second.center, reply.second.second}},
                   context->graph->dimension);
      if (normalized[index - begin].valid) {
        proof_indices[index - begin] = valid_states.size();
        valid_states.push_back(normalized[index - begin]);
      }
    }
    std::vector<PathSystemKOptProof> leaf_proofs = EvaluateLeafWindow(context, valid_states);
    for (std::size_t index = begin; index < end; ++index) {
      if (!ConsumeReply(context)) {
        AddMetric(&context->result->speculative_leaf_tasks,
                  CountUnusedValid(proof_indices, index - begin), "speculative leaves");
        context->result->proof.nodes.resize(checkpoint);
        return false;
      }
      HtTreeReply record;
      record.first_pair = replies[index].first;
      record.second_pair = replies[index].second;
      const std::optional<std::size_t> proof_index = proof_indices[index - begin];
      if (!proof_index.has_value()) {
        record.path_infeasible = true;
        records.push_back(std::move(record));
        continue;
      }
      const std::optional<std::uint32_t> child = ProvePreparedState(
          context, std::move(normalized[index - begin]), 0U, std::move(leaf_proofs[*proof_index]));
      if (!child.has_value()) {
        AddMetric(&context->result->speculative_leaf_tasks,
                  CountUnusedValid(proof_indices, index - begin + 1U), "speculative leaves");
        if (index + 1U < replies.size()) {
          AddMetric(&context->result->short_circuits, 1U, "short circuits");
        }
        context->result->proof.nodes.resize(checkpoint);
        return false;
      }
      record.child_index = *child;
      records.push_back(std::move(record));
    }
  }
  HtTreeNode& root = context->result->proof.nodes.front();
  root.move_type = HtMoveType::kCd;
  root.move_first = candidate.c;
  root.move_second = candidate.d;
  root.replies = std::move(records);
  return true;
}

HtWavefrontResult
ProveEdgeByTransposedHtImpl(const GraphSnapshot& graph, const NodeEdge raw_target,
                            const HtWavefrontOptions& options,
                            const detail::KOptSnapshotBinding* snapshot_binding,
                            const detail::HtGraphValidationBinding* graph_validation_binding,
                            const bool verify_extracted_proof) {
  HtWavefrontResult result;
  result.scheduler = "transposed";
  result.propagation_backend = "cpu-short-circuit";
  result.path_append_backend = "cpu";
  result.hamilton_reply_backend = "cpu";
  result.end_reply_backend = "cpu";
  result.path_append_cpu_verified = true;
  result.hamilton_reply_cpu_verified = true;
  result.end_reply_cpu_verified = true;
  if (options.scheduler != HtScheduler::kTransposed) {
    result.proof.reason = "转置 HT 收到非 transposed scheduler 选项";
    return result;
  }
  if (options.speculation_width > 256U) {
    result.proof.reason = "转置 HT speculation width 必须位于 [1,256] 或使用 auto";
    return result;
  }
  if (options.path_append_backend == PathCompatibilityBackend::kCuda ||
      options.hamilton_reply_backend == PathCompatibilityBackend::kCuda) {
    result.status = HtSearchStatus::kUnresolved;
    result.proof.reason = "转置 HT 当前只支持 CPU path-append/reply；GPU leaf 仍可使用";
    return result;
  }
  // 当前 host-window 原型尚未跨 target 汇聚 leaf。d15112 宽度扫描显示大于 1
  // 只会扩大投机空洞，因此 auto 保持严格短路；真正的跨目标 broker 落地后再调参。
  result.speculation_width = options.speculation_width == 0U ? 1U : options.speculation_width;

  std::optional<detail::KOptSnapshotBinding> owned_snapshot;
  if (snapshot_binding == nullptr) {
    owned_snapshot.emplace(graph);
    snapshot_binding = &*owned_snapshot;
  }
  std::optional<detail::HtGraphValidationBinding> owned_validation;
  if (graph_validation_binding == nullptr) {
    owned_validation.emplace(graph);
    graph_validation_binding = &*owned_validation;
  }
  HtRecursiveProof& proof = result.proof;
  proof.snapshot_hash = snapshot_binding->snapshot_hash();
  proof.target_edge = CanonicalEdge(raw_target.u, raw_target.v);
  proof.cd_mode = options.search_options.root_options.cd_mode;
  std::string reason;
  if (!ValidateTarget(graph, proof.target_edge, &reason)) {
    proof.reason = reason;
    return result;
  }

  HtCdBatchResult candidates;
  const SteadyClock::time_point candidate_begin = SteadyClock::now();
  try {
    candidates = detail::EvaluateHtCdCandidatesBoundToValidatedGraph(
        graph, proof.target_edge, options.search_options.root_options, *graph_validation_binding);
  } catch (const std::exception& error) {
    result.candidate_ms = ElapsedMilliseconds(candidate_begin);
    result.status =
        options.search_options.root_options.candidate_backend == PathCompatibilityBackend::kCuda
            ? HtSearchStatus::kUnresolved
            : HtSearchStatus::kInvalid;
    proof.reason = std::string("转置 HT c,d 候选失败: ") + error.what();
    return result;
  }
  result.candidate_ms = ElapsedMilliseconds(candidate_begin);
  if (candidates.candidates.empty()) {
    result.status = HtSearchStatus::kUnresolved;
    proof.reason = "转置 HT 没有可用 c,d 根 move";
    return result;
  }
  const NormalizedPathSystem root_paths =
      NormalizePathSystem({{proof.target_edge.u, proof.target_edge.v}}, graph.dimension);
  if (!root_paths.valid) {
    proof.reason = "转置 HT 无法建立目标边根路径";
    return result;
  }

  TransposedContext context{.graph = &graph,
                            .target = proof.target_edge,
                            .options = &options.search_options,
                            .snapshot_binding = snapshot_binding,
                            .result = &result,
                            .speculation_width = result.speculation_width,
                            .point_candidate_order = {},
                            .point_reply_cache =
                                std::vector<std::optional<std::vector<HtNeighborPair>>>(
                                    static_cast<std::size_t>(graph.dimension)),
                            .end_reply_cache = {},
                            .budget_exhausted = false};
  try {
    for (const HtCdCandidate& candidate : candidates.candidates) {
      ++proof.cd_candidates_tested;
      proof.nodes.clear();
      HtTreeNode root;
      root.paths = root_paths;
      proof.nodes.push_back(std::move(root));
      const SteadyClock::time_point work_begin = SteadyClock::now();
      const bool proven = TryRootMove(&context, candidate, root_paths);
      result.work_graph_ms += ElapsedMilliseconds(work_begin);
      if (proven) {
        proof.proven = true;
        // 规范 proof 与递归 DFS 保持逐字节一致；调度差异只记录在 result metrics。
        proof.reason = "递归 HT 的一个 c,d 根 move 已完成全部 AND replies";
        if (verify_extracted_proof) {
          const SteadyClock::time_point verify_begin = SteadyClock::now();
          const bool verified = VerifyHtRecursiveProof(graph, proof, &reason);
          result.proof_verify_ms += ElapsedMilliseconds(verify_begin);
          if (!verified) {
            proof.proven = false;
            proof.nodes.clear();
            proof.reason = "内部转置 HT 复核失败: " + reason;
            result.status = HtSearchStatus::kInvalid;
            return result;
          }
        }
        result.cpu_verified = true;
        result.status = HtSearchStatus::kProven;
        return result;
      }
      proof.nodes.clear();
      if (context.budget_exhausted) {
        break;
      }
    }
  } catch (const std::bad_alloc&) {
    context.budget_exhausted = true;
  } catch (const std::exception& error) {
    result.status = options.search_options.root_options.leaf_options.cost_backend ==
                            PathCompatibilityBackend::kCuda
                        ? HtSearchStatus::kUnresolved
                        : HtSearchStatus::kInvalid;
    proof.nodes.clear();
    proof.reason = std::string("转置 HT leaf/continuation 失败: ") + error.what();
    return result;
  }
  result.status = HtSearchStatus::kUnresolved;
  proof.nodes.clear();
  proof.reason =
      context.budget_exhausted ? "递归 HT 资源预算耗尽" : "递归 HT 的全部根 moves 均未解决";
  result.cpu_verified = true;
  return result;
}

} // namespace

HtWavefrontResult ProveEdgeByTransposedHt(const GraphSnapshot& graph, const NodeEdge target_edge,
                                          const HtWavefrontOptions& options) {
  return ProveEdgeByTransposedHtImpl(graph, target_edge, options, nullptr, nullptr, true);
}

HtWavefrontResult detail::ProveEdgeByTransposedHtBoundToSnapshot(
    const GraphSnapshot& graph, const NodeEdge target_edge, const HtWavefrontOptions& options,
    const KOptSnapshotBinding& snapshot_binding,
    const HtGraphValidationBinding& graph_validation_binding) {
  if (!snapshot_binding.Matches(graph) || !graph_validation_binding.Matches(graph)) {
    throw std::invalid_argument("转置 HT snapshot binding 与图对象不一致");
  }
  // 与 wavefront 一致，scan 中只在原子 epoch 提交前执行一次完整 CPU verifier。
  return ProveEdgeByTransposedHtImpl(graph, target_edge, options, &snapshot_binding,
                                     &graph_validation_binding, false);
}

} // namespace cudaee
