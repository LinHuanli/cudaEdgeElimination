#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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

NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? NodeEdge{first, second} : NodeEdge{second, first};
}

bool ContainsNode(const NormalizedPathSystem& paths, const std::int32_t needle) {
  for (const Path& path : paths.paths) {
    if (std::find(path.begin(), path.end(), needle) != path.end()) {
      return true;
    }
  }
  return false;
}

NormalizedPathSystem AddPaths(const NormalizedPathSystem& state, const std::vector<Path>& additions,
                              const std::int32_t dimension) {
  std::vector<Path> raw = state.paths;
  raw.insert(raw.end(), additions.begin(), additions.end());
  return NormalizePathSystem(raw, dimension);
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

std::vector<PointCandidate> BuildPointCandidates(const GraphSnapshot& graph, const NodeEdge target,
                                                 const NormalizedPathSystem& state,
                                                 const HtRecursiveOptions& options) {
  struct RankedNode {
    std::int32_t node{};
    __int128 midpoint_score{};
  };
  std::vector<RankedNode> neighborhood;
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    if (node == target.u || node == target.v || ContainsNode(state, node) ||
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
    neighborhood.push_back({node, dx * dx + dy * dy});
  }
  std::sort(
      neighborhood.begin(), neighborhood.end(), [](const RankedNode& lhs, const RankedNode& rhs) {
        return std::tie(lhs.midpoint_score, lhs.node) < std::tie(rhs.midpoint_score, rhs.node);
      });
  if (options.root_options.max_neighborhood != 0U &&
      neighborhood.size() > options.root_options.max_neighborhood) {
    neighborhood.resize(options.root_options.max_neighborhood);
  }

  std::vector<PointCandidate> candidates;
  candidates.reserve(neighborhood.size());
  for (const RankedNode& ranked : neighborhood) {
    candidates.push_back({ranked.node, EnumerateHtHamiltonReplies(graph, target, ranked.node)});
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

std::vector<EndCandidate> BuildEndCandidates(const GraphSnapshot& graph,
                                             const NormalizedPathSystem& state,
                                             const HtRecursiveOptions& options) {
  std::vector<EndCandidate> candidates;
  candidates.reserve(2U * state.paths.size());
  for (const Path& path : state.paths) {
    candidates.push_back(
        {path.front(), path[1], EnumerateEndReplies(graph, path.front(), path[1])});
    candidates.push_back({path.back(), path[path.size() - 2U],
                          EnumerateEndReplies(graph, path.back(), path[path.size() - 2U])});
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

struct WaveMove {
  HtMoveType type{HtMoveType::kLeaf};
  std::int32_t first{-1};
  std::int32_t second{-1};
  std::vector<HtTreeReply> replies;
};

struct WaveState {
  NormalizedPathSystem paths;
  std::uint32_t depth{};
  PathSystemKOptProof leaf_proof;
  std::vector<WaveMove> moves;
};

struct WaveBuildContext {
  const GraphSnapshot* graph{};
  NodeEdge target;
  const HtRecursiveOptions* options{};
  HtWavefrontResult* result{};
  bool budget_exhausted{false};
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

bool AppendChild(WaveBuildContext* const context, const NormalizedPathSystem& parent,
                 const std::vector<Path>& additions, const std::uint32_t child_depth,
                 HtTreeReply* const reply, std::vector<WaveState>* const states) {
  NormalizedPathSystem child = AddPaths(parent, additions, context->graph->dimension);
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
  states->push_back(std::move(state));
  return true;
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
                        std::vector<WaveState>* const states) {
  const NormalizedPathSystem parent = states->at(state_index).paths;
  const std::uint32_t child_depth = states->at(state_index).depth + 1U;
  for (const PointCandidate& candidate :
       BuildPointCandidates(*context->graph, context->target, parent, *context->options)) {
    if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
      continue;
    }
    WaveMove move;
    move.type = HtMoveType::kPoint;
    move.first = candidate.node;
    move.replies.reserve(candidate.replies.size());
    bool all_infeasible = true;
    for (const HtNeighborPair& pair : candidate.replies) {
      if (!ConsumeReply(context)) {
        return false;
      }
      HtTreeReply reply;
      reply.first_pair = pair;
      if (!AppendChild(context, parent, {{pair.first, pair.center, pair.second}}, child_depth,
                       &reply, states)) {
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
  return true;
}

bool GenerateEndMoves(WaveBuildContext* const context, const std::uint32_t state_index,
                      std::vector<WaveState>* const states) {
  const NormalizedPathSystem parent = states->at(state_index).paths;
  const std::uint32_t child_depth = states->at(state_index).depth + 1U;
  for (const EndCandidate& candidate :
       BuildEndCandidates(*context->graph, parent, *context->options)) {
    if (!MoveReplyCountAllowed(*context, candidate.replies.size())) {
      continue;
    }
    WaveMove move;
    move.type = HtMoveType::kEnd;
    move.first = candidate.endpoint;
    move.second = candidate.internal_neighbor;
    move.replies.reserve(candidate.replies.size());
    bool all_infeasible = true;
    for (const NodeEdge edge : candidate.replies) {
      if (!ConsumeReply(context)) {
        return false;
      }
      HtTreeReply reply;
      reply.edge = edge;
      const std::int32_t neighbor = edge.u == candidate.endpoint ? edge.v : edge.u;
      if (!AppendChild(context, parent, {{candidate.endpoint, neighbor}}, child_depth, &reply,
                       states)) {
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
  return true;
}

enum class RootBuildStatus : std::uint8_t {
  kBuilt,
  kSkipped,
  kBudget,
};

RootBuildStatus BuildRootMove(WaveBuildContext* const context, const HtCdCandidate& candidate,
                              std::vector<WaveState>* const states) {
  const std::vector<HtNeighborPair> first_replies =
      EnumerateHtHamiltonReplies(*context->graph, context->target, candidate.c);
  const std::vector<HtNeighborPair> second_replies =
      EnumerateHtHamiltonReplies(*context->graph, context->target, candidate.d);
  if (second_replies.size() != 0U &&
      first_replies.size() > std::numeric_limits<std::uint64_t>::max() / second_replies.size()) {
    return RootBuildStatus::kSkipped;
  }
  const std::uint64_t reply_count =
      static_cast<std::uint64_t>(first_replies.size()) * second_replies.size();
  if ((context->options->root_options.max_reply_combinations != 0U &&
       reply_count > context->options->root_options.max_reply_combinations) ||
      !MoveReplyCountAllowed(*context, reply_count)) {
    return RootBuildStatus::kSkipped;
  }
  if (reply_count > kHardWavefrontReplies) {
    context->budget_exhausted = true;
    return RootBuildStatus::kBudget;
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
                       0U, &reply, states)) {
        return RootBuildStatus::kBudget;
      }
      move.replies.push_back(reply);
    }
  }
  return RecordMove(context, 0U, std::move(move), states) ? RootBuildStatus::kBuilt
                                                          : RootBuildStatus::kBudget;
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
    for (std::size_t index = frontier_begin; index < frontier_end; ++index) {
      ++context->result->proof.leaf_calls;
      WaveState& state = states->at(index);
      state.leaf_proof = ProvePathSystemByKOpt(*context->graph, state.paths, context->target,
                                               context->options->root_options.leaf_options);
      if (state.leaf_proof.proven || state.depth >= context->options->max_depth) {
        continue;
      }
      const std::uint32_t state_index = static_cast<std::uint32_t>(index);
      if (context->options->enable_point_moves &&
          !GeneratePointMoves(context, state_index, states)) {
        return false;
      }
      if (context->budget_exhausted) {
        return false;
      }
      // 已知成功的 point move 不再生成更晚的 OR 候选。
      const bool point_shortcut =
          !states->at(index).moves.empty() &&
          std::all_of(states->at(index).moves.back().replies.begin(),
                      states->at(index).moves.back().replies.end(),
                      [](const HtTreeReply& reply) { return reply.path_infeasible; });
      if (!point_shortcut && context->options->enable_end_moves &&
          !GenerateEndMoves(context, state_index, states)) {
        return false;
      }
      if (context->budget_exhausted) {
        return false;
      }
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
  state_tasks->reserve(states.size());
  for (const WaveState& state : states) {
    HtWavefrontStateTask state_task;
    state_task.move_begin = static_cast<std::uint32_t>(move_tasks->size());
    state_task.move_count = static_cast<std::uint32_t>(state.moves.size());
    state_task.leaf_proven = static_cast<std::uint8_t>(state.leaf_proof.proven);
    for (const WaveMove& move : state.moves) {
      HtWavefrontMoveTask move_task;
      move_task.reply_begin = static_cast<std::uint32_t>(reply_tasks->size());
      move_task.reply_count = static_cast<std::uint32_t>(move.replies.size());
      for (const HtTreeReply& reply : move.replies) {
        reply_tasks->push_back(
            {reply.child_index, static_cast<std::uint8_t>(reply.path_infeasible)});
      }
      move_tasks->push_back(move_task);
    }
    state_tasks->push_back(state_task);
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
    *status = detail::EvaluateHtWavefrontCuda(state_tasks, move_tasks, reply_tasks, level_offsets,
                                              &result->selected_device);
  } catch (const std::exception& error) {
    if (backend == PathCompatibilityBackend::kCuda) {
      *reason = std::string("CUDA HT wavefront propagation 失败: ") + error.what();
      return PropagationStatus::kUnavailable;
    }
    *status = cpu_status;
    result->propagation_backend = "cpu";
    result->selected_device = -1;
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

HtWavefrontResult ProveEdgeByWavefrontHt(const GraphSnapshot& graph, const NodeEdge raw_target,
                                         const HtWavefrontOptions& options) {
  HtWavefrontResult result;
  HtRecursiveProof& proof = result.proof;
  proof.snapshot_hash = graph.ContentHash();
  proof.target_edge = CanonicalEdge(raw_target.u, raw_target.v);
  proof.cd_mode = options.search_options.root_options.cd_mode;
  if (options.propagation_backend != PathCompatibilityBackend::kAuto &&
      options.propagation_backend != PathCompatibilityBackend::kCpu &&
      options.propagation_backend != PathCompatibilityBackend::kCuda) {
    proof.reason = "未知 HT wavefront propagation 后端";
    return result;
  }

  HtCdBatchResult candidates;
  try {
    candidates =
        EvaluateHtCdCandidates(graph, proof.target_edge, options.search_options.root_options);
  } catch (const std::exception& error) {
    if (options.search_options.root_options.candidate_backend == PathCompatibilityBackend::kAuto) {
      try {
        HtShallowOptions cpu_options = options.search_options.root_options;
        cpu_options.candidate_backend = PathCompatibilityBackend::kCpu;
        candidates = EvaluateHtCdCandidates(graph, proof.target_edge, cpu_options);
      } catch (const std::exception& cpu_error) {
        proof.reason = cpu_error.what();
        return result;
      }
    } else if (options.search_options.root_options.candidate_backend ==
               PathCompatibilityBackend::kCuda) {
      result.status = HtSearchStatus::kUnresolved;
      proof.reason = std::string("CUDA HT wavefront c,d 筛选失败: ") + error.what();
      return result;
    } else {
      proof.reason = error.what();
      return result;
    }
  }
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

  WaveBuildContext context{&graph, proof.target_edge, &options.search_options, &result, false};
  try {
    for (const HtCdCandidate& candidate : candidates.candidates) {
      ++proof.cd_candidates_tested;
      std::vector<WaveState> states;
      WaveState root;
      root.paths = root_paths;
      states.push_back(std::move(root));
      const RootBuildStatus root_status = BuildRootMove(&context, candidate, &states);
      if (root_status == RootBuildStatus::kSkipped) {
        continue;
      }
      if (root_status == RootBuildStatus::kBudget) {
        break;
      }

      std::vector<std::uint32_t> level_offsets;
      if (!ExpandWavefront(&context, &states, &level_offsets)) {
        break;
      }
      std::vector<std::uint8_t> status;
      std::string propagation_reason;
      const PropagationStatus propagated =
          PropagateWavefront(options, states, level_offsets, &result, &status, &propagation_reason);
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
      CopySuccessfulState(0U, states, status, &proof.nodes);
      proof.proven = true;
      proof.reason = "HT wavefront 的一个 c,d 根 move 已完成全部 AND replies";
      std::string verify_reason;
      if (!VerifyHtRecursiveProof(graph, proof, &verify_reason)) {
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

} // namespace cudaee
