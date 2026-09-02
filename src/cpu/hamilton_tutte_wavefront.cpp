#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::size_t kHardWavefrontStates = 1000000U;
constexpr std::size_t kHardWavefrontMoves = 1000000U;
constexpr std::uint64_t kHardWavefrontReplies = 1000000U;

using SteadyClock = std::chrono::steady_clock;

double ElapsedMilliseconds(const SteadyClock::time_point begin) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - begin).count();
}

// 后端调用抛错时也要保留已消耗时间；target 始终属于外层仍存活的 result。
class ScopedPhaseTimer {
public:
  explicit ScopedPhaseTimer(double* const target) : target_(target), begin_(SteadyClock::now()) {}

  ScopedPhaseTimer(const ScopedPhaseTimer&) = delete;
  ScopedPhaseTimer& operator=(const ScopedPhaseTimer&) = delete;

  ~ScopedPhaseTimer() { *target_ += ElapsedMilliseconds(begin_); }

private:
  double* target_;
  SteadyClock::time_point begin_;
};

NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? NodeEdge{first, second} : NodeEdge{second, first};
}

NormalizedPathSystem AddPaths(const NormalizedPathSystem& state, const std::vector<Path>& additions,
                              const std::int32_t dimension) {
  std::vector<Path> raw = state.paths;
  raw.insert(raw.end(), additions.begin(), additions.end());
  return NormalizePathSystem(raw, dimension);
}

struct PointCandidate {
  std::int32_t node{-1};
  std::vector<HtNeighborPair> replies;
  std::size_t append_begin{};
};

struct EndCandidate {
  std::int32_t endpoint{-1};
  std::int32_t internal_neighbor{-1};
  std::vector<NodeEdge> replies;
  std::size_t append_begin{};
};

struct PreparedStateCandidates {
  std::size_t state_index{};
  std::vector<PointCandidate> point;
  std::vector<EndCandidate> end;
};

struct PreparedFrontierChunk {
  std::vector<PreparedStateCandidates> states;
  HtPathAppendBatchResult point_appends;
  HtPathAppendBatchResult end_appends;
};

struct PointCandidateSelection {
  std::vector<std::int32_t> nodes;
  std::uint64_t nodes_checked{};
  std::uint64_t nodes_ranked{};
  double scan_ms{};
  double sort_ms{};
};

struct PointCandidateOrderCache {
  // 中点评分、度数门禁和严格次序只依赖不可变 graph/target；state 仅过滤已有路径节点。
  std::vector<std::int32_t> ordered_nodes;
  std::vector<std::uint8_t> eligible;
  std::vector<std::uint32_t> state_marks;
  std::uint32_t generation{};
  bool initialized{false};
};

PointCandidateSelection BuildPointCandidateNodes(const GraphSnapshot& graph, const NodeEdge target,
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
    const auto dimension = static_cast<std::size_t>(graph.dimension);
    prepared.eligible.assign(dimension, 0U);
    prepared.state_marks.assign(dimension, 0U);
    std::vector<RankedNode> ranked_nodes;
    ranked_nodes.reserve(dimension);

    const auto prepare_scan_begin = SteadyClock::now();
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
      ranked_nodes.push_back({node, dx * dx + dy * dy});
      prepared.eligible[static_cast<std::size_t>(node)] = 1U;
    }
    selection.scan_ms += ElapsedMilliseconds(prepare_scan_begin);

    const auto prepare_sort_begin = SteadyClock::now();
    std::sort(
        ranked_nodes.begin(), ranked_nodes.end(), [](const RankedNode& lhs, const RankedNode& rhs) {
          return std::tie(lhs.midpoint_score, lhs.node) < std::tie(rhs.midpoint_score, rhs.node);
        });
    prepared.ordered_nodes.reserve(ranked_nodes.size());
    for (const RankedNode& ranked : ranked_nodes) {
      prepared.ordered_nodes.push_back(ranked.node);
    }
    prepared.initialized = true;
    selection.sort_ms += ElapsedMilliseconds(prepare_sort_begin);
    *cache = std::move(prepared);
  }

  selection.nodes_checked = static_cast<std::uint64_t>(graph.dimension);
  // generation mark 避免每个 frontier state 都清零完整维度的成员位图。
  if (cache->generation == std::numeric_limits<std::uint32_t>::max()) {
    std::fill(cache->state_marks.begin(), cache->state_marks.end(), 0U);
    cache->generation = 1U;
  } else {
    ++cache->generation;
  }

  const auto state_scan_begin = SteadyClock::now();
  std::uint64_t excluded_eligible = 0U;
  for (const Path& path : state.paths) {
    for (const std::int32_t node : path) {
      if (node < 0 || node >= graph.dimension) {
        continue;
      }
      const auto index = static_cast<std::size_t>(node);
      if (cache->state_marks[index] == cache->generation) {
        continue;
      }
      cache->state_marks[index] = cache->generation;
      excluded_eligible += cache->eligible[index];
    }
  }
  if (excluded_eligible > cache->ordered_nodes.size()) {
    throw std::logic_error("HT point candidate 状态过滤计数非法");
  }
  // 保留旧实现完整扫描得到的 ranked 规范计数，不能只统计实际访问的有序前缀。
  selection.nodes_ranked =
      static_cast<std::uint64_t>(cache->ordered_nodes.size()) - excluded_eligible;
  selection.scan_ms += ElapsedMilliseconds(state_scan_begin);

  const auto select_begin = SteadyClock::now();
  const std::size_t selection_limit =
      options.root_options.max_neighborhood == 0U
          ? cache->ordered_nodes.size()
          : std::min(cache->ordered_nodes.size(),
                     static_cast<std::size_t>(options.root_options.max_neighborhood));
  selection.nodes.reserve(selection_limit);
  for (const std::int32_t node : cache->ordered_nodes) {
    if (cache->state_marks[static_cast<std::size_t>(node)] == cache->generation) {
      continue;
    }
    selection.nodes.push_back(node);
    if (options.root_options.max_neighborhood != 0U &&
        selection.nodes.size() == options.root_options.max_neighborhood) {
      break;
    }
  }
  selection.sort_ms += ElapsedMilliseconds(select_begin);
  return selection;
}

void AddPointCandidateMetric(std::uint64_t* const total, const std::uint64_t value,
                             const char* const name) {
  if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
    throw std::overflow_error(std::string("HT point candidate 指标溢出: ") + name);
  }
  *total += value;
}

std::vector<HtEndReplyTask> BuildEndReplyTasks(const NormalizedPathSystem& state) {
  std::vector<HtEndReplyTask> tasks;
  tasks.reserve(2U * state.paths.size());
  for (const Path& path : state.paths) {
    tasks.push_back({path.front(), path[1]});
    tasks.push_back({path.back(), path[path.size() - 2U]});
  }
  return tasks;
}

struct WaveMove {
  HtMoveType type{HtMoveType::kLeaf};
  std::int32_t first{-1};
  std::int32_t second{-1};
  std::vector<HtTreeReply> replies;
};

struct WaveState {
  NormalizedPathSystem paths;
  std::uint32_t depth{};
  std::uint64_t incoming_reply_count{};
  PathSystemKOptProof leaf_proof;
  std::vector<WaveMove> moves;
};

struct WaveBuildContext {
  const GraphSnapshot* graph{};
  NodeEdge target;
  const HtRecursiveOptions* options{};
  PathCompatibilityBackend path_append_backend{PathCompatibilityBackend::kAuto};
  PathCompatibilityBackend hamilton_reply_backend{PathCompatibilityBackend::kAuto};
  std::uint32_t reply_frontier_batch_states{256};
  std::uint32_t leaf_frontier_batch_states{256};
  bool fuse_leaf_buckets{false};
  PointCandidateOrderCache point_candidate_order;
  const detail::KOptSnapshotBinding* leaf_snapshot_binding{};
  const detail::HtGraphValidationBinding* graph_validation_binding{};
  HtWavefrontResult* result{};
  bool budget_exhausted{false};
  bool path_append_failed{false};
  bool path_append_invalid{false};
  std::string path_append_reason;
  bool hamilton_reply_failed{false};
  bool hamilton_reply_invalid{false};
  std::string hamilton_reply_reason;
  bool end_reply_failed{false};
  bool end_reply_invalid{false};
  std::string end_reply_reason;
};

bool MoveReplyCountAllowed(const WaveBuildContext& context, const std::uint64_t count) {
  return context.options->max_replies_per_move == 0U ||
         count <= context.options->max_replies_per_move;
}

bool ConsumeReply(WaveBuildContext* const context) {
  HtRecursiveProof& proof = context->result->proof;
  if (proof.replies_expanded >= kHardWavefrontReplies ||
      (context->options->max_total_replies != 0U &&
       proof.replies_expanded >= context->options->max_total_replies)) {
    context->budget_exhausted = true;
    return false;
  }
  ++proof.replies_expanded;
  return true;
}

bool ReserveReplyBatch(WaveBuildContext* const context, const std::size_t count) {
  HtRecursiveProof& proof = context->result->proof;
  const std::uint64_t count64 = static_cast<std::uint64_t>(count);
  if (proof.replies_expanded > kHardWavefrontReplies ||
      count64 > kHardWavefrontReplies - proof.replies_expanded ||
      (context->options->max_total_replies != 0U &&
       (proof.replies_expanded > context->options->max_total_replies ||
        count64 > context->options->max_total_replies - proof.replies_expanded))) {
    context->budget_exhausted = true;
    return false;
  }
  proof.replies_expanded += count64;
  return true;
}

bool ReplyPlanFits(const WaveBuildContext& context, const std::uint64_t already_planned,
                   const std::size_t additional) {
  const std::uint64_t additional64 = static_cast<std::uint64_t>(additional);
  const std::uint64_t expanded = context.result->proof.replies_expanded;
  if (expanded > kHardWavefrontReplies || already_planned > kHardWavefrontReplies - expanded ||
      additional64 > kHardWavefrontReplies - expanded - already_planned) {
    return false;
  }
  if (context.options->max_total_replies == 0U) {
    return true;
  }
  const std::uint64_t limit = context.options->max_total_replies;
  return expanded <= limit && already_planned <= limit - expanded &&
         additional64 <= limit - expanded - already_planned;
}

void RecordPathAppendBatch(WaveBuildContext* const context, const HtPathAppendBatchResult& batch) {
  if (batch.feasible.empty()) {
    return;
  }
  HtWavefrontResult& result = *context->result;
  ++result.path_append_batches;
  result.path_append_tasks += batch.feasible.size();
  result.path_append_cpu_verified = result.path_append_batches == 1U
                                        ? batch.cpu_verified
                                        : result.path_append_cpu_verified && batch.cpu_verified;
  result.path_append_device_children_verified =
      result.path_append_batches == 1U
          ? batch.device_children_verified
          : result.path_append_device_children_verified && batch.device_children_verified;
  result.path_append_child_edges += batch.child_edges.size();
  result.path_append_parent_prepare_ms += batch.parent_prepare_ms;
  result.path_append_child_normalize_ms += batch.child_normalize_ms;
  result.path_append_child_edges_ms += batch.child_edges_ms;
  result.path_append_cuda_evaluate_ms += batch.cuda_evaluate_ms;
  result.path_append_cuda_compare_ms += batch.cuda_compare_ms;
  if (result.path_append_backend == "none") {
    result.path_append_backend = batch.backend;
  } else if (result.path_append_backend != batch.backend) {
    result.path_append_backend = "mixed";
  }
  if (batch.selected_device >= 0) {
    result.path_append_selected_device = batch.selected_device;
  }
}

std::optional<HtPathAppendBatchResult>
EvaluatePathAppendBatch(WaveBuildContext* const context,
                        const std::vector<NormalizedPathSystem>& parents,
                        const std::vector<HtPathAppendTask>& tasks) {
  ScopedPhaseTimer timer(&context->result->path_append_ms);
  try {
    HtPathAppendBatchResult batch = EvaluateHtPathAppends(context->graph->dimension, parents, tasks,
                                                          context->path_append_backend);
    RecordPathAppendBatch(context, batch);
    return batch;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::logic_error& error) {
    context->path_append_invalid = true;
    context->path_append_reason = error.what();
  } catch (const std::exception& error) {
    context->path_append_failed = true;
    context->path_append_reason = error.what();
  }
  return std::nullopt;
}

void RecordHamiltonReplyBatch(WaveBuildContext* const context,
                              const HtHamiltonReplyBatchResult& batch,
                              const std::size_t center_count) {
  if (center_count == 0U) {
    return;
  }
  HtWavefrontResult& result = *context->result;
  if (result.hamilton_reply_batches == std::numeric_limits<std::uint64_t>::max() ||
      center_count > std::numeric_limits<std::uint64_t>::max() - result.hamilton_reply_centers ||
      batch.unique_centers >
          std::numeric_limits<std::uint64_t>::max() - result.hamilton_reply_unique_centers ||
      batch.neighbor_pairs_tested >
          std::numeric_limits<std::uint64_t>::max() - result.hamilton_reply_neighbor_pairs_tested ||
      batch.replies.size() >
          std::numeric_limits<std::uint64_t>::max() - result.hamilton_replies_generated) {
    throw std::overflow_error("HT Hamilton reply 批处理统计溢出");
  }
  ++result.hamilton_reply_batches;
  result.hamilton_reply_centers += static_cast<std::uint64_t>(center_count);
  result.hamilton_reply_unique_centers += batch.unique_centers;
  result.hamilton_reply_neighbor_pairs_tested += batch.neighbor_pairs_tested;
  result.hamilton_replies_generated += static_cast<std::uint64_t>(batch.replies.size());
  result.hamilton_reply_validation_ms += batch.validation_ms;
  result.hamilton_reply_cpu_enumerate_ms += batch.cpu_enumerate_ms;
  result.hamilton_reply_cuda_evaluate_ms += batch.cuda_evaluate_ms;
  result.hamilton_reply_cuda_compare_ms += batch.cuda_compare_ms;
  result.hamilton_reply_cpu_verified =
      result.hamilton_reply_batches == 1U
          ? batch.cpu_verified
          : result.hamilton_reply_cpu_verified && batch.cpu_verified;
  if (result.hamilton_reply_backend == "none") {
    result.hamilton_reply_backend = batch.backend;
  } else if (result.hamilton_reply_backend != batch.backend) {
    result.hamilton_reply_backend = "mixed";
  }
  if (batch.selected_device >= 0) {
    result.hamilton_reply_selected_device = batch.selected_device;
  }
}

std::optional<HtHamiltonReplyBatchResult>
EvaluateHamiltonReplyBatch(WaveBuildContext* const context,
                           const std::vector<std::int32_t>& centers) {
  ScopedPhaseTimer timer(&context->result->hamilton_reply_ms);
  try {
    if (context->graph_validation_binding == nullptr) {
      throw std::logic_error("HT wavefront 缺少 graph validation binding");
    }
    HtHamiltonReplyBatchResult batch = detail::EvaluateHtHamiltonRepliesBoundToValidatedGraph(
        *context->graph, context->target, centers, *context->graph_validation_binding,
        context->hamilton_reply_backend);
    RecordHamiltonReplyBatch(context, batch, centers.size());
    return batch;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::logic_error& error) {
    context->hamilton_reply_invalid = true;
    context->hamilton_reply_reason = error.what();
  } catch (const std::exception& error) {
    context->hamilton_reply_failed = true;
    context->hamilton_reply_reason = error.what();
  }
  return std::nullopt;
}

std::vector<HtNeighborPair> CopyHamiltonReplySlice(const HtHamiltonReplyBatchResult& batch,
                                                   const std::size_t center_index) {
  if (center_index + 1U >= batch.offsets.size() ||
      batch.offsets[center_index] > batch.offsets[center_index + 1U] ||
      batch.offsets[center_index + 1U] > batch.replies.size()) {
    throw std::logic_error("HT Hamilton reply batch offset 非法");
  }
  const auto begin =
      batch.replies.begin() + static_cast<std::ptrdiff_t>(batch.offsets[center_index]);
  const auto end =
      batch.replies.begin() + static_cast<std::ptrdiff_t>(batch.offsets[center_index + 1U]);
  return {begin, end};
}

void RecordEndReplyBatch(WaveBuildContext* const context, const HtEndReplyBatchResult& batch,
                         const std::size_t task_count) {
  if (task_count == 0U) {
    return;
  }
  HtWavefrontResult& result = *context->result;
  if (result.end_reply_batches == std::numeric_limits<std::uint64_t>::max() ||
      task_count > std::numeric_limits<std::uint64_t>::max() - result.end_reply_tasks ||
      batch.replies.size() >
          std::numeric_limits<std::uint64_t>::max() - result.end_replies_generated) {
    throw std::overflow_error("HT end reply 批处理统计溢出");
  }
  ++result.end_reply_batches;
  result.end_reply_tasks += static_cast<std::uint64_t>(task_count);
  result.end_replies_generated += static_cast<std::uint64_t>(batch.replies.size());
  result.end_reply_cpu_verified = result.end_reply_batches == 1U
                                      ? batch.cpu_verified
                                      : result.end_reply_cpu_verified && batch.cpu_verified;
  if (result.end_reply_backend == "none") {
    result.end_reply_backend = batch.backend;
  } else if (result.end_reply_backend != batch.backend) {
    result.end_reply_backend = "mixed";
  }
  if (batch.selected_device >= 0) {
    result.end_reply_selected_device = batch.selected_device;
  }
}

std::optional<HtEndReplyBatchResult>
EvaluateEndReplyBatch(WaveBuildContext* const context, const std::vector<HtEndReplyTask>& tasks) {
  ScopedPhaseTimer timer(&context->result->end_reply_ms);
  try {
    if (context->graph_validation_binding == nullptr) {
      throw std::logic_error("HT wavefront 缺少 graph validation binding");
    }
    HtEndReplyBatchResult batch = detail::EvaluateHtEndRepliesBoundToValidatedGraph(
        *context->graph, tasks, *context->graph_validation_binding,
        context->hamilton_reply_backend);
    RecordEndReplyBatch(context, batch, tasks.size());
    return batch;
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::logic_error& error) {
    context->end_reply_invalid = true;
    context->end_reply_reason = error.what();
  } catch (const std::exception& error) {
    context->end_reply_failed = true;
    context->end_reply_reason = error.what();
  }
  return std::nullopt;
}

std::vector<NodeEdge> CopyEndReplySlice(const HtEndReplyBatchResult& batch,
                                        const std::size_t task_index) {
  if (task_index + 1U >= batch.offsets.size() ||
      batch.offsets[task_index] > batch.offsets[task_index + 1U] ||
      batch.offsets[task_index + 1U] > batch.replies.size()) {
    throw std::logic_error("HT end reply batch offset 非法");
  }
  const auto begin = batch.replies.begin() + static_cast<std::ptrdiff_t>(batch.offsets[task_index]);
  const auto end =
      batch.replies.begin() + static_cast<std::ptrdiff_t>(batch.offsets[task_index + 1U]);
  return {begin, end};
}

void RecordReplyFrontierBatch(WaveBuildContext* const context, const std::size_t state_count) {
  if (state_count == 0U) {
    return;
  }
  HtWavefrontResult& result = *context->result;
  if (result.reply_frontier_batches == std::numeric_limits<std::uint64_t>::max() ||
      state_count > std::numeric_limits<std::uint64_t>::max() - result.reply_frontier_states) {
    throw std::overflow_error("HT reply frontier 批处理统计溢出");
  }
  ++result.reply_frontier_batches;
  result.reply_frontier_states += static_cast<std::uint64_t>(state_count);
  result.peak_reply_frontier_batch =
      std::max(result.peak_reply_frontier_batch, static_cast<std::uint64_t>(state_count));
}

std::optional<PreparedFrontierChunk> PrepareFrontierCandidates(WaveBuildContext* const context,
                                                               const std::vector<WaveState>& states,
                                                               const std::size_t begin,
                                                               const std::size_t end) {
  struct CandidateInput {
    std::size_t state_index{};
    std::size_t point_begin{};
    std::vector<std::int32_t> point_nodes;
  };

  std::vector<CandidateInput> inputs;
  std::vector<std::int32_t> centers;
  inputs.reserve(end - begin);
  for (std::size_t state_index = begin; state_index < end; ++state_index) {
    const WaveState& state = states.at(state_index);
    if (state.leaf_proof.proven || state.depth >= context->options->max_depth) {
      continue;
    }
    CandidateInput input;
    input.state_index = state_index;
    input.point_begin = centers.size();
    if (context->options->enable_point_moves) {
      PointCandidateSelection selection =
          BuildPointCandidateNodes(*context->graph, context->target, state.paths, *context->options,
                                   &context->point_candidate_order);
      HtWavefrontResult& result = *context->result;
      AddPointCandidateMetric(&result.point_candidate_scans, 1U, "scans");
      AddPointCandidateMetric(&result.point_candidate_nodes_checked, selection.nodes_checked,
                              "nodes checked");
      AddPointCandidateMetric(&result.point_candidate_nodes_ranked, selection.nodes_ranked,
                              "nodes ranked");
      AddPointCandidateMetric(&result.point_candidate_nodes_selected, selection.nodes.size(),
                              "nodes selected");
      result.point_candidate_scan_ms += selection.scan_ms;
      result.point_candidate_sort_ms += selection.sort_ms;
      input.point_nodes = std::move(selection.nodes);
      centers.insert(centers.end(), input.point_nodes.begin(), input.point_nodes.end());
    }
    inputs.push_back(std::move(input));
  }

  std::optional<HtHamiltonReplyBatchResult> point_batch;
  if (!centers.empty()) {
    point_batch = EvaluateHamiltonReplyBatch(context, centers);
    if (!point_batch.has_value()) {
      return std::nullopt;
    }
  }

  PreparedFrontierChunk chunk;
  chunk.states.reserve(inputs.size());
  for (const CandidateInput& input : inputs) {
    PreparedStateCandidates state_candidates;
    state_candidates.state_index = input.state_index;
    state_candidates.point.reserve(input.point_nodes.size());
    for (std::size_t index = 0; index < input.point_nodes.size(); ++index) {
      state_candidates.point.push_back(
          {input.point_nodes[index],
           CopyHamiltonReplySlice(*point_batch, input.point_begin + index)});
    }
    std::sort(state_candidates.point.begin(), state_candidates.point.end(),
              [](const PointCandidate& lhs, const PointCandidate& rhs) {
                return std::tuple{lhs.replies.size(), lhs.node} <
                       std::tuple{rhs.replies.size(), rhs.node};
              });
    if (context->options->max_point_candidates != 0U &&
        state_candidates.point.size() > context->options->max_point_candidates) {
      state_candidates.point.resize(context->options->max_point_candidates);
    }
    chunk.states.push_back(std::move(state_candidates));
  }
  if (chunk.states.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("HT path-append frontier 父状态索引溢出");
  }

  std::vector<NormalizedPathSystem> parents;
  std::vector<HtPathAppendTask> point_append_tasks;
  parents.reserve(chunk.states.size());
  for (std::size_t parent_index = 0; parent_index < chunk.states.size(); ++parent_index) {
    PreparedStateCandidates& state_candidates = chunk.states[parent_index];
    parents.push_back(states.at(state_candidates.state_index).paths);
    for (PointCandidate& candidate : state_candidates.point) {
      if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
        continue;
      }
      candidate.append_begin = point_append_tasks.size();
      for (const HtNeighborPair& pair : candidate.replies) {
        point_append_tasks.push_back({static_cast<std::uint32_t>(parent_index),
                                      HtPathAppendKind::kPoint, pair.first, pair.center,
                                      pair.second});
      }
    }
  }
  if (!point_append_tasks.empty()) {
    std::optional<HtPathAppendBatchResult> batch =
        EvaluatePathAppendBatch(context, parents, point_append_tasks);
    if (!batch.has_value()) {
      return std::nullopt;
    }
    chunk.point_appends = std::move(*batch);
  }

  const auto has_vacuous_point_move = [&](const PreparedStateCandidates& state_candidates) {
    if (!context->options->enable_point_moves) {
      return false;
    }
    for (const PointCandidate& candidate : state_candidates.point) {
      if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
        continue;
      }
      if (candidate.append_begin > chunk.point_appends.feasible.size() ||
          candidate.replies.size() > chunk.point_appends.feasible.size() - candidate.append_begin) {
        throw std::logic_error("HT point path-append batch 区间非法");
      }
      bool all_infeasible = true;
      for (std::size_t reply = 0; reply < candidate.replies.size(); ++reply) {
        all_infeasible =
            all_infeasible && chunk.point_appends.feasible[candidate.append_begin + reply] == 0U;
      }
      if (all_infeasible) {
        return true;
      }
    }
    return false;
  };

  struct EndInput {
    std::size_t prepared_index{};
    std::size_t task_begin{};
    std::vector<HtEndReplyTask> tasks;
  };
  std::vector<EndInput> end_inputs;
  std::vector<HtEndReplyTask> end_tasks;
  if (context->options->enable_end_moves) {
    for (std::size_t prepared_index = 0; prepared_index < chunk.states.size(); ++prepared_index) {
      const PreparedStateCandidates& state_candidates = chunk.states[prepared_index];
      if (has_vacuous_point_move(state_candidates)) {
        continue;
      }
      EndInput input;
      input.prepared_index = prepared_index;
      input.task_begin = end_tasks.size();
      input.tasks = BuildEndReplyTasks(states.at(state_candidates.state_index).paths);
      end_tasks.insert(end_tasks.end(), input.tasks.begin(), input.tasks.end());
      end_inputs.push_back(std::move(input));
    }
  }
  std::optional<HtEndReplyBatchResult> end_reply_batch;
  if (!end_tasks.empty()) {
    end_reply_batch = EvaluateEndReplyBatch(context, end_tasks);
    if (!end_reply_batch.has_value()) {
      return std::nullopt;
    }
  }
  for (const EndInput& input : end_inputs) {
    std::vector<EndCandidate>& candidates = chunk.states[input.prepared_index].end;
    candidates.reserve(input.tasks.size());
    for (std::size_t index = 0; index < input.tasks.size(); ++index) {
      const HtEndReplyTask& task = input.tasks[index];
      candidates.push_back({task.endpoint, task.internal_neighbor,
                            CopyEndReplySlice(*end_reply_batch, input.task_begin + index)});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const EndCandidate& lhs, const EndCandidate& rhs) {
                return std::tuple{lhs.replies.size(), lhs.endpoint, lhs.internal_neighbor} <
                       std::tuple{rhs.replies.size(), rhs.endpoint, rhs.internal_neighbor};
              });
    if (context->options->max_end_candidates != 0U &&
        candidates.size() > context->options->max_end_candidates) {
      candidates.resize(context->options->max_end_candidates);
    }
  }

  std::vector<HtPathAppendTask> end_append_tasks;
  for (std::size_t parent_index = 0; parent_index < chunk.states.size(); ++parent_index) {
    for (EndCandidate& candidate : chunk.states[parent_index].end) {
      if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
        continue;
      }
      candidate.append_begin = end_append_tasks.size();
      for (const NodeEdge edge : candidate.replies) {
        const std::int32_t neighbor = edge.u == candidate.endpoint ? edge.v : edge.u;
        end_append_tasks.push_back({static_cast<std::uint32_t>(parent_index),
                                    HtPathAppendKind::kEnd, candidate.endpoint, -1, neighbor});
      }
    }
  }
  if (!end_append_tasks.empty()) {
    std::optional<HtPathAppendBatchResult> batch =
        EvaluatePathAppendBatch(context, parents, end_append_tasks);
    if (!batch.has_value()) {
      return std::nullopt;
    }
    chunk.end_appends = std::move(*batch);
  }

  RecordReplyFrontierBatch(context, chunk.states.size());
  return chunk;
}

bool AppendNormalizedChild(WaveBuildContext* const context, NormalizedPathSystem child,
                           const std::uint32_t child_depth,
                           const std::uint64_t incoming_reply_count, HtTreeReply* const reply,
                           std::vector<WaveState>* const states) {
  if (!child.valid) {
    reply->path_infeasible = true;
    return true;
  }
  HtRecursiveProof& proof = context->result->proof;
  if (states->size() >= kHardWavefrontStates ||
      states->size() >= std::numeric_limits<std::uint32_t>::max() ||
      (context->options->max_states != 0U &&
       proof.states_expanded >= context->options->max_states)) {
    context->budget_exhausted = true;
    return false;
  }
  ++proof.states_expanded;
  reply->child_index = static_cast<std::uint32_t>(states->size());
  WaveState state;
  state.paths = std::move(child);
  state.depth = child_depth;
  state.incoming_reply_count = incoming_reply_count;
  states->push_back(std::move(state));
  return true;
}

bool AppendChild(WaveBuildContext* const context, const NormalizedPathSystem& parent,
                 const std::vector<Path>& additions, const std::uint32_t child_depth,
                 const std::uint64_t incoming_reply_count, HtTreeReply* const reply,
                 std::vector<WaveState>* const states) {
  if (context->result->root_child_normalizations == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("HT 根 child 规范化计数溢出");
  }
  const auto normalize_begin = SteadyClock::now();
  NormalizedPathSystem child = AddPaths(parent, additions, context->graph->dimension);
  context->result->root_child_normalize_ms += ElapsedMilliseconds(normalize_begin);
  ++context->result->root_child_normalizations;
  return AppendNormalizedChild(context, std::move(child), child_depth, incoming_reply_count, reply,
                               states);
}

bool RecordMove(WaveBuildContext* const context, const std::uint32_t state_index, WaveMove move,
                std::vector<WaveState>* const states) {
  if (context->result->moves_generated >= kHardWavefrontMoves) {
    context->budget_exhausted = true;
    return false;
  }
  ++context->result->moves_generated;
  states->at(state_index).moves.push_back(std::move(move));
  return true;
}

bool GeneratePointMoves(WaveBuildContext* const context, const std::uint32_t state_index,
                        const std::vector<PointCandidate>& candidates,
                        const HtPathAppendBatchResult& batch,
                        std::vector<WaveState>* const states) {
  const std::uint32_t child_depth = states->at(state_index).depth + 1U;
  std::vector<const PointCandidate*> plans;
  std::uint64_t planned_replies = 0U;
  bool budget_blocked = false;
  for (const PointCandidate& candidate : candidates) {
    if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
      continue;
    }
    if (!ReplyPlanFits(*context, planned_replies, candidate.replies.size())) {
      budget_blocked = true;
      break;
    }
    plans.push_back(&candidate);
    planned_replies += static_cast<std::uint64_t>(candidate.replies.size());
  }
  if (plans.empty()) {
    if (budget_blocked) {
      context->budget_exhausted = true;
      return false;
    }
    return true;
  }
  for (const PointCandidate* const candidate_pointer : plans) {
    const PointCandidate& candidate = *candidate_pointer;
    WaveMove move;
    move.type = HtMoveType::kPoint;
    move.first = candidate.node;
    move.replies.reserve(candidate.replies.size());
    if (!ReserveReplyBatch(context, candidate.replies.size())) {
      return false;
    }
    bool all_infeasible = true;
    for (std::size_t reply_index = 0; reply_index < candidate.replies.size(); ++reply_index) {
      const HtNeighborPair& pair = candidate.replies[reply_index];
      HtTreeReply reply;
      reply.first_pair = pair;
      if (!AppendNormalizedChild(context, batch.children.at(candidate.append_begin + reply_index),
                                 child_depth, candidate.replies.size(), &reply, states)) {
        return false;
      }
      all_infeasible = all_infeasible && reply.path_infeasible;
      move.replies.push_back(reply);
    }
    if (!RecordMove(context, state_index, std::move(move), states)) {
      return false;
    }
    if (all_infeasible) {
      return true;
    }
  }
  if (budget_blocked) {
    context->budget_exhausted = true;
    return false;
  }
  return true;
}

bool GenerateEndMoves(WaveBuildContext* const context, const std::uint32_t state_index,
                      const std::vector<EndCandidate>& candidates,
                      const HtPathAppendBatchResult& batch, std::vector<WaveState>* const states) {
  const std::uint32_t child_depth = states->at(state_index).depth + 1U;
  std::vector<const EndCandidate*> plans;
  std::uint64_t planned_replies = 0U;
  bool budget_blocked = false;
  for (const EndCandidate& candidate : candidates) {
    if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
      continue;
    }
    if (!ReplyPlanFits(*context, planned_replies, candidate.replies.size())) {
      budget_blocked = true;
      break;
    }
    plans.push_back(&candidate);
    planned_replies += static_cast<std::uint64_t>(candidate.replies.size());
  }
  if (plans.empty()) {
    if (budget_blocked) {
      context->budget_exhausted = true;
      return false;
    }
    return true;
  }
  for (const EndCandidate* const candidate_pointer : plans) {
    const EndCandidate& candidate = *candidate_pointer;
    WaveMove move;
    move.type = HtMoveType::kEnd;
    move.first = candidate.endpoint;
    move.second = candidate.internal_neighbor;
    move.replies.reserve(candidate.replies.size());
    if (!ReserveReplyBatch(context, candidate.replies.size())) {
      return false;
    }
    bool all_infeasible = true;
    for (std::size_t reply_index = 0; reply_index < candidate.replies.size(); ++reply_index) {
      const NodeEdge edge = candidate.replies[reply_index];
      HtTreeReply reply;
      reply.edge = edge;
      if (!AppendNormalizedChild(context, batch.children.at(candidate.append_begin + reply_index),
                                 child_depth, candidate.replies.size(), &reply, states)) {
        return false;
      }
      all_infeasible = all_infeasible && reply.path_infeasible;
      move.replies.push_back(reply);
    }
    if (!RecordMove(context, state_index, std::move(move), states)) {
      return false;
    }
    if (all_infeasible) {
      return true;
    }
  }
  if (budget_blocked) {
    context->budget_exhausted = true;
    return false;
  }
  return true;
}

enum class RootBuildStatus : std::uint8_t {
  kBuilt,
  kSkipped,
  kBudget,
  kBackend,
};

RootBuildStatus BuildRootMove(WaveBuildContext* const context, const HtCdCandidate& candidate,
                              std::vector<WaveState>* const states) {
  if ((context->options->root_options.max_reply_combinations != 0U &&
       candidate.reply_product > context->options->root_options.max_reply_combinations) ||
      !MoveReplyCountAllowed(*context, candidate.reply_product)) {
    return RootBuildStatus::kSkipped;
  }
  if (candidate.reply_product > kHardWavefrontReplies) {
    context->budget_exhausted = true;
    return RootBuildStatus::kBudget;
  }
  const std::optional<HtHamiltonReplyBatchResult> batch =
      EvaluateHamiltonReplyBatch(context, {candidate.c, candidate.d});
  if (!batch.has_value()) {
    return RootBuildStatus::kBackend;
  }
  const std::vector<HtNeighborPair> first_replies = CopyHamiltonReplySlice(*batch, 0U);
  const std::vector<HtNeighborPair> second_replies = CopyHamiltonReplySlice(*batch, 1U);
  if (second_replies.size() != 0U &&
      first_replies.size() > std::numeric_limits<std::uint64_t>::max() / second_replies.size()) {
    return RootBuildStatus::kSkipped;
  }
  const std::uint64_t reply_count =
      static_cast<std::uint64_t>(first_replies.size()) * second_replies.size();
  if (reply_count != candidate.reply_product) {
    context->hamilton_reply_invalid = true;
    context->hamilton_reply_reason = "c,d 候选 reply_product 与批量枚举不一致";
    return RootBuildStatus::kBackend;
  }

  WaveMove move;
  move.type = HtMoveType::kCd;
  move.first = candidate.c;
  move.second = candidate.d;
  move.replies.reserve(static_cast<std::size_t>(reply_count));
  const NormalizedPathSystem parent = states->front().paths;
  for (const HtNeighborPair& first : first_replies) {
    for (const HtNeighborPair& second : second_replies) {
      if (!ConsumeReply(context)) {
        return RootBuildStatus::kBudget;
      }
      HtTreeReply reply;
      reply.first_pair = first;
      reply.second_pair = second;
      if (!AppendChild(context, parent,
                       {{first.first, first.center, first.second},
                        {second.first, second.center, second.second}},
                       0U, reply_count, &reply, states)) {
        return RootBuildStatus::kBudget;
      }
      move.replies.push_back(reply);
    }
  }
  return RecordMove(context, 0U, std::move(move), states) ? RootBuildStatus::kBuilt
                                                          : RootBuildStatus::kBudget;
}

struct LeafBucketKey {
  std::uint32_t depth{};
  std::uint32_t path_count{};
  std::uint32_t node_count{};
  std::uint32_t max_k{};
  std::uint32_t reply_bucket{};

  bool operator<(const LeafBucketKey& other) const {
    return std::tie(depth, path_count, node_count, max_k, reply_bucket) <
           std::tie(other.depth, other.path_count, other.node_count, other.max_k,
                    other.reply_bucket);
  }
};

std::uint32_t ReplyCountBucket(std::uint64_t reply_count) {
  std::uint32_t bucket = 0U;
  while (reply_count > 1U) {
    reply_count >>= 1U;
    ++bucket;
  }
  return bucket;
}

LeafBucketKey BuildLeafBucketKey(const WaveState& state, const KOptSearchOptions& options) {
  std::size_t node_count = 0U;
  for (const Path& path : state.paths.paths) {
    if (path.size() > std::numeric_limits<std::size_t>::max() - node_count) {
      throw std::overflow_error("HT leaf bucket 节点数溢出");
    }
    node_count += path.size();
  }
  if (state.paths.paths.size() > std::numeric_limits<std::uint32_t>::max() ||
      node_count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("HT leaf bucket 复杂度键溢出");
  }
  return {.depth = state.depth,
          .path_count = static_cast<std::uint32_t>(state.paths.paths.size()),
          .node_count = static_cast<std::uint32_t>(node_count),
          .max_k = options.max_k,
          .reply_bucket = ReplyCountBucket(state.incoming_reply_count)};
}

void AddLeafMetric(std::uint64_t* const total, const std::uint64_t value, const char* const name) {
  if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
    throw std::overflow_error(std::string("HT leaf 指标溢出: ") + name);
  }
  *total += value;
}

void RecordLeafBatch(WaveBuildContext* const context, const PathSystemKOptBatchResult& batch,
                     const std::size_t state_count) {
  HtWavefrontResult& result = *context->result;
  AddLeafMetric(&result.leaf_frontier_batches, 1U, "frontier batches");
  AddLeafMetric(&result.leaf_frontier_states, state_count, "frontier states");
  result.peak_leaf_frontier_batch =
      std::max(result.peak_leaf_frontier_batch, static_cast<std::uint64_t>(state_count));
  result.leaf_cpu_verified = result.leaf_frontier_batches == 1U
                                 ? batch.cpu_verified
                                 : result.leaf_cpu_verified && batch.cpu_verified;
  AddLeafMetric(&result.leaf_cost_batches, batch.cost_batches, "cost batches");
  AddLeafMetric(&result.leaf_cost_tasks, batch.cost_tasks, "cost tasks");
  AddLeafMetric(&result.leaf_cost_cells, batch.cost_cells, "cost cells");
  AddLeafMetric(&result.leaf_scalar_searches, batch.scalar_searches, "scalar searches");
  AddLeafMetric(&result.leaf_cursor_searches_started, batch.cursor_searches_started,
                "cursor searches started");
  AddLeafMetric(&result.leaf_cuda_cost_batches, batch.cuda_cost_batches, "CUDA cost batches");
  AddLeafMetric(&result.leaf_snapshot_cache_hits, batch.snapshot_cache_hits, "snapshot cache hits");
  AddLeafMetric(&result.leaf_template_cache_hits, batch.template_cache_hits, "template cache hits");
  AddLeafMetric(&result.leaf_workspace_cache_hits, batch.workspace_cache_hits,
                "workspace cache hits");
  result.peak_leaf_device_cache_bytes =
      std::max(result.peak_leaf_device_cache_bytes, batch.peak_device_cache_bytes);
  AddLeafMetric(&result.leaf_cpu_long_tail_batches, batch.cpu_long_tail_batches,
                "CPU long-tail batches");
  AddLeafMetric(&result.leaf_cpu_long_tail_tasks, batch.cpu_long_tail_tasks, "CPU long-tail tasks");
  AddLeafMetric(&result.leaf_cpu_long_tail_cells, batch.cpu_long_tail_cells, "CPU long-tail cells");
  AddLeafMetric(&result.leaf_cost_rows_consumed, batch.cost_rows_consumed, "cost rows consumed");
  AddLeafMetric(&result.leaf_candidate_templates_rechecked, batch.candidate_templates_rechecked,
                "candidate templates rechecked");
  AddLeafMetric(&result.leaf_cpu_completeness_rows, batch.cpu_completeness_rows,
                "CPU completeness rows");
  AddLeafMetric(&result.leaf_cpu_completeness_templates, batch.cpu_completeness_templates,
                "CPU completeness templates");
  AddLeafMetric(&result.leaf_cpu_certified_cost_cells, batch.cpu_certified_cost_cells,
                "CPU certified cost cells");
  AddLeafMetric(&result.leaf_cpu_cost_rows_scored, batch.cpu_cost_rows_scored,
                "CPU cost rows scored");
  AddLeafMetric(&result.leaf_cpu_cost_rows_reused, batch.cpu_cost_rows_reused,
                "CPU cost rows reused");
  AddLeafMetric(&result.leaf_cpu_parallel_cost_batches, batch.cpu_parallel_cost_batches,
                "CPU parallel cost batches");
  AddLeafMetric(&result.leaf_cpu_parallel_cost_cells, batch.cpu_parallel_cost_cells,
                "CPU parallel cost cells");
  result.peak_leaf_cpu_cost_threads =
      std::max(result.peak_leaf_cpu_cost_threads, batch.peak_cpu_cost_threads);
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
  if (batch.cost_backend != "none") {
    if (result.leaf_cost_backend == "none") {
      result.leaf_cost_backend = batch.cost_backend;
    } else if (result.leaf_cost_backend != batch.cost_backend) {
      result.leaf_cost_backend = "mixed";
    }
  }
  if (batch.selected_device >= 0) {
    result.leaf_cost_selected_device = batch.selected_device;
  }
}

void EvaluateLeafStateIndices(WaveBuildContext* const context, std::vector<WaveState>* const states,
                              const std::vector<std::size_t>& state_indices) {
  const std::size_t max_batch_states =
      context->leaf_frontier_batch_states == 0U
          ? state_indices.size()
          : static_cast<std::size_t>(context->leaf_frontier_batch_states);
  for (std::size_t begin = 0U; begin < state_indices.size();) {
    const std::size_t end = begin + std::min(max_batch_states, state_indices.size() - begin);
    std::vector<NormalizedPathSystem> path_systems;
    path_systems.reserve(end - begin);
    for (std::size_t offset = begin; offset < end; ++offset) {
      path_systems.push_back(states->at(state_indices[offset]).paths);
    }
    AddLeafMetric(&context->result->proof.leaf_calls, path_systems.size(), "leaf calls");
    PathSystemKOptBatchResult batch;
    {
      ScopedPhaseTimer timer(&context->result->leaf_ms);
      if (context->leaf_snapshot_binding == nullptr) {
        throw std::logic_error("HT leaf 缺少不可变 snapshot binding");
      }
      batch = detail::ProvePathSystemsByKOptBoundToSnapshot(
          *context->graph, path_systems, context->target, *context->leaf_snapshot_binding,
          context->options->root_options.leaf_options);
    }
    if (batch.proofs.size() != path_systems.size()) {
      throw std::logic_error("HT leaf batch 返回的 proof 数量不一致");
    }
    RecordLeafBatch(context, batch, path_systems.size());
    for (std::size_t offset = begin; offset < end; ++offset) {
      states->at(state_indices[offset]).leaf_proof = std::move(batch.proofs[offset - begin]);
    }
    begin = end;
  }
}

void EvaluateLeafFrontierChunk(WaveBuildContext* const context,
                               std::vector<WaveState>* const states,
                               const std::size_t frontier_begin, const std::size_t frontier_end) {
  std::map<LeafBucketKey, std::vector<std::size_t>> buckets;
  for (std::size_t state_index = frontier_begin; state_index < frontier_end; ++state_index) {
    buckets[BuildLeafBucketKey(states->at(state_index),
                               context->options->root_options.leaf_options)]
        .push_back(state_index);
  }
  AddLeafMetric(&context->result->leaf_bucket_count, buckets.size(), "bucket count");
  if (context->fuse_leaf_buckets) {
    std::vector<std::size_t> fused_state_indices;
    fused_state_indices.reserve(frontier_end - frontier_begin);
    // 保留既有规范桶序与桶内 state index 顺序，只消除批调用边界。
    for (const auto& [key, state_indices] : buckets) {
      static_cast<void>(key);
      fused_state_indices.insert(fused_state_indices.end(), state_indices.begin(),
                                 state_indices.end());
    }
    EvaluateLeafStateIndices(context, states, fused_state_indices);
    return;
  }
  for (const auto& [key, state_indices] : buckets) {
    static_cast<void>(key);
    EvaluateLeafStateIndices(context, states, state_indices);
  }
}

bool ExpandWavefront(WaveBuildContext* const context, std::vector<WaveState>* const states,
                     std::vector<std::uint32_t>* const level_offsets) {
  level_offsets->clear();
  level_offsets->push_back(0U);
  level_offsets->push_back(1U); // 根 c,d 单独占一层。
  std::size_t frontier_begin = 1U;
  std::size_t frontier_end = states->size();
  if (frontier_end > frontier_begin) {
    level_offsets->push_back(static_cast<std::uint32_t>(frontier_end));
  }
  while (frontier_begin < frontier_end) {
    context->result->peak_frontier = std::max(
        context->result->peak_frontier, static_cast<std::uint64_t>(frontier_end - frontier_begin));
    const std::size_t max_batch_states =
        context->reply_frontier_batch_states == 0U
            ? frontier_end - frontier_begin
            : static_cast<std::size_t>(context->reply_frontier_batch_states);
    for (std::size_t batch_begin = frontier_begin; batch_begin < frontier_end;) {
      const std::size_t batch_end =
          batch_begin + std::min(max_batch_states, frontier_end - batch_begin);
      EvaluateLeafFrontierChunk(context, states, batch_begin, batch_end);
      if (!context->options->enable_point_moves && !context->options->enable_end_moves) {
        batch_begin = batch_end;
        continue;
      }
      const std::optional<PreparedFrontierChunk> prepared =
          PrepareFrontierCandidates(context, *states, batch_begin, batch_end);
      if (!prepared.has_value()) {
        return false;
      }
      for (const PreparedStateCandidates& candidates : prepared->states) {
        const auto state_index = static_cast<std::uint32_t>(candidates.state_index);
        if (context->options->enable_point_moves &&
            !GeneratePointMoves(context, state_index, candidates.point, prepared->point_appends,
                                states)) {
          return false;
        }
        if (context->budget_exhausted) {
          return false;
        }
        // 已知成功的 point move 不再生成更晚的 OR 候选。
        const bool point_shortcut =
            !states->at(candidates.state_index).moves.empty() &&
            std::all_of(states->at(candidates.state_index).moves.back().replies.begin(),
                        states->at(candidates.state_index).moves.back().replies.end(),
                        [](const HtTreeReply& reply) { return reply.path_infeasible; });
        if (!point_shortcut && context->options->enable_end_moves &&
            !GenerateEndMoves(context, state_index, candidates.end, prepared->end_appends,
                              states)) {
          return false;
        }
        if (context->budget_exhausted) {
          return false;
        }
      }
      batch_begin = batch_end;
    }
    frontier_begin = frontier_end;
    frontier_end = states->size();
    if (frontier_end > frontier_begin) {
      level_offsets->push_back(static_cast<std::uint32_t>(frontier_end));
    }
  }
  return true;
}

bool MoveProven(const WaveMove& move, const std::vector<std::uint8_t>& status) {
  return std::all_of(move.replies.begin(), move.replies.end(), [&](const HtTreeReply& reply) {
    return reply.path_infeasible ||
           (reply.child_index < status.size() && status[reply.child_index] != 0U);
  });
}

std::vector<std::uint8_t> EvaluateWavefrontCpu(const std::vector<WaveState>& states) {
  std::vector<std::uint8_t> status(states.size(), 0U);
  for (std::size_t reverse = states.size(); reverse > 0U; --reverse) {
    const std::size_t index = reverse - 1U;
    const WaveState& state = states[index];
    const bool proven = state.leaf_proof.proven ||
                        std::any_of(state.moves.begin(), state.moves.end(),
                                    [&](const WaveMove& move) { return MoveProven(move, status); });
    status[index] = static_cast<std::uint8_t>(proven);
  }
  return status;
}

void FlattenWavefront(const std::vector<WaveState>& states,
                      std::vector<HtWavefrontStateTask>* const state_tasks,
                      std::vector<HtWavefrontMoveTask>* const move_tasks,
                      std::vector<HtWavefrontReplyTask>* const reply_tasks) {
  state_tasks->clear();
  move_tasks->clear();
  reply_tasks->clear();
  state_tasks->resize(states.size());
  std::vector<std::uint32_t> parent_moves(states.size(), kNoHtChild);
  for (std::size_t state_index = 0; state_index < states.size(); ++state_index) {
    const WaveState& state = states[state_index];
    HtWavefrontStateTask state_task;
    state_task.move_begin = static_cast<std::uint32_t>(move_tasks->size());
    state_task.move_count = static_cast<std::uint32_t>(state.moves.size());
    state_task.leaf_proven = static_cast<std::uint8_t>(state.leaf_proof.proven);
    for (const WaveMove& move : state.moves) {
      HtWavefrontMoveTask move_task;
      move_task.parent_state = static_cast<std::uint32_t>(state_index);
      move_task.reply_begin = static_cast<std::uint32_t>(reply_tasks->size());
      move_task.reply_count = static_cast<std::uint32_t>(move.replies.size());
      const std::uint32_t move_index = static_cast<std::uint32_t>(move_tasks->size());
      for (const HtTreeReply& reply : move.replies) {
        if (!reply.path_infeasible) {
          if (reply.child_index >= states.size() || reply.child_index == 0U ||
              parent_moves[reply.child_index] != kNoHtChild) {
            throw std::logic_error("HT wavefront child 没有唯一 parent move");
          }
          parent_moves[reply.child_index] = move_index;
          ++move_task.child_count;
        }
        reply_tasks->push_back(
            {reply.child_index, static_cast<std::uint8_t>(reply.path_infeasible)});
      }
      move_tasks->push_back(move_task);
    }
    state_tasks->at(state_index) = state_task;
  }
  for (std::size_t state_index = 1; state_index < states.size(); ++state_index) {
    if (parent_moves[state_index] == kNoHtChild) {
      throw std::logic_error("HT wavefront 非根状态缺少 parent move");
    }
    state_tasks->at(state_index).parent_move = parent_moves[state_index];
  }
}

enum class PropagationStatus : std::uint8_t {
  kOk,
  kUnavailable,
  kMismatch,
};

PropagationStatus
PropagateWavefront(const HtWavefrontOptions& options, const std::vector<WaveState>& states,
                   const std::vector<std::uint32_t>& level_offsets, HtWavefrontResult* const result,
                   std::vector<std::uint8_t>* const status, std::string* const reason) {
  const std::vector<std::uint8_t> cpu_status = EvaluateWavefrontCpu(states);
  const PathCompatibilityBackend backend = options.propagation_backend;
  if (backend == PathCompatibilityBackend::kCpu) {
    *status = cpu_status;
    result->propagation_backend = "cpu";
    result->cpu_verified = true;
    return PropagationStatus::kOk;
  }
  if (backend != PathCompatibilityBackend::kAuto && backend != PathCompatibilityBackend::kCuda) {
    *reason = "未知 HT wavefront propagation 后端";
    return PropagationStatus::kMismatch;
  }

  std::string cuda_reason;
  if (!detail::HtWavefrontCudaAvailable(&cuda_reason)) {
    if (backend == PathCompatibilityBackend::kCuda) {
      *reason = "CUDA HT wavefront 后端不可用: " + cuda_reason;
      return PropagationStatus::kUnavailable;
    }
    *status = cpu_status;
    result->propagation_backend = "cpu";
    result->cpu_verified = true;
    return PropagationStatus::kOk;
  }

  std::vector<HtWavefrontStateTask> state_tasks;
  std::vector<HtWavefrontMoveTask> move_tasks;
  std::vector<HtWavefrontReplyTask> reply_tasks;
  FlattenWavefront(states, &state_tasks, &move_tasks, &reply_tasks);
  try {
    detail::HtWavefrontDeviceResult device_result =
        detail::EvaluateHtWavefrontCuda(state_tasks, move_tasks, reply_tasks, level_offsets,
                                        options.propagation_blocks, &result->selected_device);
    *status = std::move(device_result.status);
    result->propagation_blocks = device_result.launched_blocks;
    result->propagation_cooperative = device_result.cooperative;
  } catch (const std::exception& error) {
    if (backend == PathCompatibilityBackend::kCuda) {
      *reason = std::string("CUDA HT wavefront propagation 失败: ") + error.what();
      return PropagationStatus::kUnavailable;
    }
    *status = cpu_status;
    result->propagation_backend = "cpu";
    result->selected_device = -1;
    result->propagation_blocks = 0U;
    result->propagation_cooperative = false;
    result->cpu_verified = true;
    return PropagationStatus::kOk;
  }
  if (*status != cpu_status) {
    *reason = "CUDA HT wavefront 状态与 CPU 规范传播不一致";
    return PropagationStatus::kMismatch;
  }
  result->propagation_backend = "cuda";
  result->cpu_verified = true;
  return PropagationStatus::kOk;
}

std::uint32_t CopySuccessfulState(const std::uint32_t source_index,
                                  const std::vector<WaveState>& states,
                                  const std::vector<std::uint8_t>& status,
                                  std::vector<HtTreeNode>* const nodes) {
  const std::uint32_t destination_index = static_cast<std::uint32_t>(nodes->size());
  nodes->emplace_back();
  const WaveState& source = states[source_index];
  HtTreeNode destination;
  destination.paths = source.paths;
  if (source.leaf_proof.proven) {
    destination.move_type = HtMoveType::kLeaf;
    destination.leaf_proof = source.leaf_proof;
    nodes->at(destination_index) = std::move(destination);
    return destination_index;
  }

  const auto selected =
      std::find_if(source.moves.begin(), source.moves.end(),
                   [&](const WaveMove& move) { return MoveProven(move, status); });
  if (selected == source.moves.end()) {
    throw std::logic_error("HT wavefront 成功状态没有成功 move");
  }
  destination.move_type = selected->type;
  destination.move_first = selected->first;
  destination.move_second = selected->second;
  destination.replies.reserve(selected->replies.size());
  for (const HtTreeReply& source_reply : selected->replies) {
    HtTreeReply destination_reply = source_reply;
    if (!source_reply.path_infeasible) {
      destination_reply.child_index =
          CopySuccessfulState(source_reply.child_index, states, status, nodes);
    }
    destination.replies.push_back(destination_reply);
  }
  nodes->at(destination_index) = std::move(destination);
  return destination_index;
}

} // namespace

namespace {

HtWavefrontResult
ProveEdgeByWavefrontHtImpl(const GraphSnapshot& graph, const NodeEdge raw_target,
                           const HtWavefrontOptions& options,
                           const detail::KOptSnapshotBinding* snapshot_binding,
                           const detail::HtGraphValidationBinding* graph_validation_binding) {
  HtWavefrontResult result;
  HtRecursiveProof& proof = result.proof;
  std::optional<detail::KOptSnapshotBinding> owned_snapshot_binding;
  if (snapshot_binding == nullptr) {
    owned_snapshot_binding.emplace(graph);
    snapshot_binding = &*owned_snapshot_binding;
  }
  proof.snapshot_hash = snapshot_binding->snapshot_hash();
  proof.target_edge = CanonicalEdge(raw_target.u, raw_target.v);
  proof.cd_mode = options.search_options.root_options.cd_mode;
  if (options.propagation_backend != PathCompatibilityBackend::kAuto &&
      options.propagation_backend != PathCompatibilityBackend::kCpu &&
      options.propagation_backend != PathCompatibilityBackend::kCuda) {
    proof.reason = "未知 HT wavefront propagation 后端";
    return result;
  }
  if (options.path_append_backend != PathCompatibilityBackend::kAuto &&
      options.path_append_backend != PathCompatibilityBackend::kCpu &&
      options.path_append_backend != PathCompatibilityBackend::kCuda) {
    proof.reason = "未知 HT path-append 后端";
    return result;
  }
  if (options.hamilton_reply_backend != PathCompatibilityBackend::kAuto &&
      options.hamilton_reply_backend != PathCompatibilityBackend::kCpu &&
      options.hamilton_reply_backend != PathCompatibilityBackend::kCuda) {
    proof.reason = "未知 HT Hamilton reply 后端";
    return result;
  }

  std::optional<detail::HtGraphValidationBinding> owned_graph_validation_binding;
  HtCdBatchResult candidates;
  const SteadyClock::time_point candidate_begin = SteadyClock::now();
  try {
    if (graph_validation_binding == nullptr) {
      owned_graph_validation_binding.emplace(graph);
      graph_validation_binding = &*owned_graph_validation_binding;
    }
    candidates = detail::EvaluateHtCdCandidatesBoundToValidatedGraph(
        graph, proof.target_edge, options.search_options.root_options, *graph_validation_binding);
  } catch (const std::exception& error) {
    if (options.search_options.root_options.candidate_backend == PathCompatibilityBackend::kAuto) {
      try {
        HtShallowOptions cpu_options = options.search_options.root_options;
        cpu_options.candidate_backend = PathCompatibilityBackend::kCpu;
        if (graph_validation_binding == nullptr) {
          owned_graph_validation_binding.emplace(graph);
          graph_validation_binding = &*owned_graph_validation_binding;
        }
        candidates = detail::EvaluateHtCdCandidatesBoundToValidatedGraph(
            graph, proof.target_edge, cpu_options, *graph_validation_binding);
      } catch (const std::exception& cpu_error) {
        result.candidate_ms += ElapsedMilliseconds(candidate_begin);
        proof.reason = cpu_error.what();
        return result;
      }
    } else if (options.search_options.root_options.candidate_backend ==
               PathCompatibilityBackend::kCuda) {
      result.candidate_ms += ElapsedMilliseconds(candidate_begin);
      result.status = HtSearchStatus::kUnresolved;
      proof.reason = std::string("CUDA HT wavefront c,d 筛选失败: ") + error.what();
      return result;
    } else {
      result.candidate_ms += ElapsedMilliseconds(candidate_begin);
      proof.reason = error.what();
      return result;
    }
  }
  result.candidate_ms += ElapsedMilliseconds(candidate_begin);
  if (candidates.candidates.empty()) {
    result.status = HtSearchStatus::kUnresolved;
    proof.reason = "HT wavefront 没有可用 c,d 根 move";
    return result;
  }

  const NormalizedPathSystem root_paths =
      NormalizePathSystem({{proof.target_edge.u, proof.target_edge.v}}, graph.dimension);
  if (!root_paths.valid) {
    proof.reason = "HT wavefront 无法建立目标边根路径";
    return result;
  }

  WaveBuildContext context{.graph = &graph,
                           .target = proof.target_edge,
                           .options = &options.search_options,
                           .path_append_backend = options.path_append_backend,
                           .hamilton_reply_backend = options.hamilton_reply_backend,
                           .reply_frontier_batch_states = options.reply_frontier_batch_states,
                           .leaf_frontier_batch_states = options.leaf_frontier_batch_states,
                           .fuse_leaf_buckets = options.fuse_leaf_buckets,
                           .point_candidate_order = {},
                           .leaf_snapshot_binding = snapshot_binding,
                           .graph_validation_binding = graph_validation_binding,
                           .result = &result,
                           .budget_exhausted = false,
                           .path_append_failed = false,
                           .path_append_invalid = false,
                           .path_append_reason = {},
                           .hamilton_reply_failed = false,
                           .hamilton_reply_invalid = false,
                           .hamilton_reply_reason = {},
                           .end_reply_failed = false,
                           .end_reply_invalid = false,
                           .end_reply_reason = {}};
  try {
    for (const HtCdCandidate& candidate : candidates.candidates) {
      ++proof.cd_candidates_tested;
      std::vector<WaveState> states;
      WaveState root;
      root.paths = root_paths;
      states.push_back(std::move(root));
      RootBuildStatus root_status;
      {
        ScopedPhaseTimer timer(&result.work_graph_ms);
        root_status = BuildRootMove(&context, candidate, &states);
      }
      if (root_status == RootBuildStatus::kSkipped) {
        continue;
      }
      if (root_status == RootBuildStatus::kBudget) {
        break;
      }
      if (root_status == RootBuildStatus::kBackend) {
        result.status =
            context.hamilton_reply_invalid ? HtSearchStatus::kInvalid : HtSearchStatus::kUnresolved;
        proof.reason = context.hamilton_reply_invalid
                           ? "HT Hamilton reply CPU/CUDA 复核失败: " + context.hamilton_reply_reason
                           : "HT Hamilton reply 后端失败: " + context.hamilton_reply_reason;
        return result;
      }

      std::vector<std::uint32_t> level_offsets;
      bool expanded = false;
      {
        ScopedPhaseTimer timer(&result.work_graph_ms);
        expanded = ExpandWavefront(&context, &states, &level_offsets);
      }
      if (!expanded) {
        if (context.path_append_invalid) {
          result.status = HtSearchStatus::kInvalid;
          proof.reason = "HT path-append CPU/CUDA 复核失败: " + context.path_append_reason;
          return result;
        }
        if (context.path_append_failed) {
          result.status = HtSearchStatus::kUnresolved;
          proof.reason = "HT path-append 后端失败: " + context.path_append_reason;
          return result;
        }
        if (context.hamilton_reply_invalid) {
          result.status = HtSearchStatus::kInvalid;
          proof.reason = "HT Hamilton reply CPU/CUDA 复核失败: " + context.hamilton_reply_reason;
          return result;
        }
        if (context.hamilton_reply_failed) {
          result.status = HtSearchStatus::kUnresolved;
          proof.reason = "HT Hamilton reply 后端失败: " + context.hamilton_reply_reason;
          return result;
        }
        if (context.end_reply_invalid) {
          result.status = HtSearchStatus::kInvalid;
          proof.reason = "HT end reply CPU/CUDA 复核失败: " + context.end_reply_reason;
          return result;
        }
        if (context.end_reply_failed) {
          result.status = HtSearchStatus::kUnresolved;
          proof.reason = "HT end reply 后端失败: " + context.end_reply_reason;
          return result;
        }
        break;
      }
      std::vector<std::uint8_t> status;
      std::string propagation_reason;
      PropagationStatus propagated;
      {
        ScopedPhaseTimer timer(&result.propagation_ms);
        propagated = PropagateWavefront(options, states, level_offsets, &result, &status,
                                        &propagation_reason);
      }
      if (propagated == PropagationStatus::kUnavailable) {
        result.status = HtSearchStatus::kUnresolved;
        proof.reason = propagation_reason;
        return result;
      }
      if (propagated == PropagationStatus::kMismatch) {
        result.status = HtSearchStatus::kInvalid;
        proof.reason = propagation_reason;
        return result;
      }
      if (status.empty() || status.front() == 0U) {
        continue;
      }

      proof.nodes.clear();
      {
        ScopedPhaseTimer timer(&result.proof_extract_ms);
        CopySuccessfulState(0U, states, status, &proof.nodes);
      }
      proof.proven = true;
      proof.reason = "HT wavefront 的一个 c,d 根 move 已完成全部 AND replies";
      std::string verify_reason;
      bool verified = false;
      {
        ScopedPhaseTimer timer(&result.proof_verify_ms);
        verified = VerifyHtRecursiveProof(graph, proof, &verify_reason);
      }
      if (!verified) {
        proof.proven = false;
        proof.nodes.clear();
        proof.reason = "内部 HT wavefront 复核失败: " + verify_reason;
        result.status = HtSearchStatus::kInvalid;
        return result;
      }
      result.status = HtSearchStatus::kProven;
      return result;
    }
  } catch (const std::bad_alloc&) {
    context.budget_exhausted = true;
  }

  result.status = HtSearchStatus::kUnresolved;
  proof.nodes.clear();
  proof.reason = context.budget_exhausted ? "HT wavefront 资源预算耗尽"
                                          : "HT wavefront 的全部根 moves 均未解决";
  return result;
}

} // namespace

HtWavefrontResult ProveEdgeByWavefrontHt(const GraphSnapshot& graph, const NodeEdge raw_target,
                                         const HtWavefrontOptions& options) {
  return ProveEdgeByWavefrontHtImpl(graph, raw_target, options, nullptr, nullptr);
}

HtWavefrontResult detail::ProveEdgeByWavefrontHtBoundToSnapshot(
    const GraphSnapshot& graph, const NodeEdge target_edge, const HtWavefrontOptions& options,
    const KOptSnapshotBinding& snapshot_binding,
    const HtGraphValidationBinding& graph_validation_binding) {
  if (!snapshot_binding.Matches(graph) || !graph_validation_binding.Matches(graph)) {
    throw std::invalid_argument("HT wavefront snapshot binding 与图对象不一致");
  }
  return ProveEdgeByWavefrontHtImpl(graph, target_edge, options, &snapshot_binding,
                                    &graph_validation_binding);
}

} // namespace cudaee
