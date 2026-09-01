#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? NodeEdge{first, second} : NodeEdge{second, first};
}

void SetReason(std::string* const reason, std::string value) {
  if (reason != nullptr) {
    *reason = std::move(value);
  }
}

bool SamePaths(const NormalizedPathSystem& first, const NormalizedPathSystem& second) {
  return first.valid && second.valid && first.edge_count == second.edge_count &&
         first.paths == second.paths;
}

bool ContainsNode(const NormalizedPathSystem& paths, const std::int32_t needle) {
  for (const Path& path : paths.paths) {
    if (std::find(path.begin(), path.end(), needle) != path.end()) {
      return true;
    }
  }
  return false;
}

bool ValidateGraph(const GraphSnapshot& graph, std::string* const reason) {
  if (!graph.integer_coordinates || !graph.integer_distance_safe || graph.dimension < 4 ||
      graph.points.size() != static_cast<std::size_t>(graph.dimension) ||
      graph.row_offsets.size() != static_cast<std::size_t>(graph.dimension) + 1U ||
      graph.row_offsets.front() != 0 || graph.row_offsets.back() < 0 ||
      static_cast<std::size_t>(graph.row_offsets.back()) != graph.neighbors.size() ||
      graph.neighbors.size() != 2U * graph.ActiveEdgeCount()) {
    SetReason(reason, "递归 HT 需要有效的整数距离 CSR 快照");
    return false;
  }
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(node)];
    const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(node) + 1U];
    if (begin < 0 || end < begin || static_cast<std::size_t>(end) > graph.neighbors.size()) {
      SetReason(reason, "递归 HT CSR row offset 非法");
      return false;
    }
    std::int32_t previous = -1;
    for (std::int32_t offset = begin; offset < end; ++offset) {
      const std::int32_t neighbor = graph.neighbors[static_cast<std::size_t>(offset)];
      if (neighbor <= previous || neighbor < 0 || neighbor >= graph.dimension || neighbor == node ||
          !graph.HasActiveEdge(neighbor, node)) {
        SetReason(reason, "递归 HT CSR 邻接表未排序、越界或不对称");
        return false;
      }
      previous = neighbor;
    }
  }
  return true;
}

bool ValidateTarget(const GraphSnapshot& graph, const NodeEdge target, std::string* const reason) {
  if (target.u < 0 || target.v >= graph.dimension || target.u >= target.v ||
      !graph.HasActiveEdge(target.u, target.v)) {
    SetReason(reason, "递归 HT 目标必须是规范的活动边");
    return false;
  }
  return true;
}

bool EdgesSurviveTwoOpt(const GraphSnapshot& graph, const NodeEdge target, const NodeEdge other) {
  const __int128 original =
      static_cast<__int128>(graph.Distance(target.u, target.v)) + graph.Distance(other.u, other.v);
  const __int128 orientation0 =
      static_cast<__int128>(graph.Distance(target.u, other.v)) + graph.Distance(target.v, other.u);
  const __int128 orientation1 =
      static_cast<__int128>(graph.Distance(target.u, other.u)) + graph.Distance(target.v, other.v);
  return orientation0 >= original || orientation1 >= original;
}

bool CdMoveAdmissible(const GraphSnapshot& graph, const NodeEdge target, const std::int32_t c,
                      const std::int32_t d, const HtCdMode mode) {
  if ((mode != HtCdMode::kActiveIncompatible && mode != HtCdMode::kMissingOrIncompatible) ||
      c < 0 || d < 0 || c >= graph.dimension || d >= graph.dimension || c == d || c == target.u ||
      c == target.v || d == target.u || d == target.v) {
    return false;
  }
  const bool active = graph.HasActiveEdge(c, d);
  const bool incompatible = !EdgesSurviveTwoOpt(graph, target, CanonicalEdge(c, d));
  return mode == HtCdMode::kActiveIncompatible ? active && incompatible : !active || incompatible;
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
        (options.root_options.max_candidate_degree != 0 &&
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
  if (options.root_options.max_neighborhood != 0 &&
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
  if (options.max_point_candidates != 0 && candidates.size() > options.max_point_candidates) {
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
  if (options.max_end_candidates != 0 && candidates.size() > options.max_end_candidates) {
    candidates.resize(options.max_end_candidates);
  }
  return candidates;
}

struct SearchContext {
  const GraphSnapshot* graph{};
  NodeEdge target;
  const HtRecursiveOptions* options{};
  HtRecursiveProof* proof{};
  bool budget_exhausted{false};
};

bool ConsumeReply(SearchContext* const context) {
  if (context->options->max_total_replies != 0 &&
      context->proof->replies_expanded >= context->options->max_total_replies) {
    context->budget_exhausted = true;
    return false;
  }
  ++context->proof->replies_expanded;
  return true;
}

bool MoveReplyCountAllowed(SearchContext* const context, const std::uint64_t count) {
  return context->options->max_replies_per_move == 0 ||
         count <= context->options->max_replies_per_move;
}

std::optional<std::uint32_t> ProveState(SearchContext* context, NormalizedPathSystem state,
                                        std::uint32_t depth);

bool TryPointMove(SearchContext* const context, const std::uint32_t node_index,
                  const NormalizedPathSystem& state, const PointCandidate& candidate,
                  const std::uint32_t depth) {
  if (!MoveReplyCountAllowed(context, candidate.replies.size())) {
    return false;
  }
  const std::size_t checkpoint = context->proof->nodes.size();
  std::vector<HtTreeReply> records;
  records.reserve(candidate.replies.size());
  for (const HtNeighborPair& reply : candidate.replies) {
    if (!ConsumeReply(context)) {
      context->proof->nodes.resize(checkpoint);
      return false;
    }
    HtTreeReply record;
    record.first_pair = reply;
    NormalizedPathSystem child =
        AddPaths(state, {{reply.first, reply.center, reply.second}}, context->graph->dimension);
    if (!child.valid) {
      record.path_infeasible = true;
      records.push_back(std::move(record));
      continue;
    }
    const std::optional<std::uint32_t> child_index =
        ProveState(context, std::move(child), depth + 1U);
    if (!child_index.has_value()) {
      context->proof->nodes.resize(checkpoint);
      return false;
    }
    record.child_index = *child_index;
    records.push_back(std::move(record));
  }
  HtTreeNode& node = context->proof->nodes[node_index];
  node.move_type = HtMoveType::kPoint;
  node.move_first = candidate.node;
  node.move_second = -1;
  node.leaf_proof = {};
  node.replies = std::move(records);
  return true;
}

bool TryEndMove(SearchContext* const context, const std::uint32_t node_index,
                const NormalizedPathSystem& state, const EndCandidate& candidate,
                const std::uint32_t depth) {
  if (!MoveReplyCountAllowed(context, candidate.replies.size())) {
    return false;
  }
  const std::size_t checkpoint = context->proof->nodes.size();
  std::vector<HtTreeReply> records;
  records.reserve(candidate.replies.size());
  for (const NodeEdge edge : candidate.replies) {
    if (!ConsumeReply(context)) {
      context->proof->nodes.resize(checkpoint);
      return false;
    }
    HtTreeReply record;
    record.edge = edge;
    const std::int32_t neighbor = edge.u == candidate.endpoint ? edge.v : edge.u;
    NormalizedPathSystem child =
        AddPaths(state, {{candidate.endpoint, neighbor}}, context->graph->dimension);
    if (!child.valid) {
      record.path_infeasible = true;
      records.push_back(std::move(record));
      continue;
    }
    const std::optional<std::uint32_t> child_index =
        ProveState(context, std::move(child), depth + 1U);
    if (!child_index.has_value()) {
      context->proof->nodes.resize(checkpoint);
      return false;
    }
    record.child_index = *child_index;
    records.push_back(std::move(record));
  }
  HtTreeNode& node = context->proof->nodes[node_index];
  node.move_type = HtMoveType::kEnd;
  node.move_first = candidate.endpoint;
  node.move_second = candidate.internal_neighbor;
  node.leaf_proof = {};
  node.replies = std::move(records);
  return true;
}

std::optional<std::uint32_t> ProveState(SearchContext* const context, NormalizedPathSystem state,
                                        const std::uint32_t depth) {
  if ((context->options->max_states != 0 &&
       context->proof->states_expanded >= context->options->max_states) ||
      context->proof->nodes.size() >= std::numeric_limits<std::uint32_t>::max()) {
    context->budget_exhausted = true;
    return std::nullopt;
  }
  ++context->proof->states_expanded;
  const auto node_index = static_cast<std::uint32_t>(context->proof->nodes.size());
  HtTreeNode node;
  node.paths = std::move(state);
  context->proof->nodes.push_back(std::move(node));

  ++context->proof->leaf_calls;
  PathSystemKOptProof leaf =
      ProvePathSystemByKOpt(*context->graph, context->proof->nodes[node_index].paths,
                            context->target, context->options->root_options.leaf_options);
  if (leaf.proven) {
    context->proof->nodes[node_index].move_type = HtMoveType::kLeaf;
    context->proof->nodes[node_index].leaf_proof = std::move(leaf);
    return node_index;
  }
  if (depth >= context->options->max_depth) {
    context->proof->nodes.resize(node_index);
    return std::nullopt;
  }

  const NormalizedPathSystem state_copy = context->proof->nodes[node_index].paths;
  if (context->options->enable_point_moves) {
    for (const PointCandidate& candidate :
         BuildPointCandidates(*context->graph, context->target, state_copy, *context->options)) {
      if (TryPointMove(context, node_index, state_copy, candidate, depth)) {
        return node_index;
      }
      if (context->budget_exhausted) {
        context->proof->nodes.resize(node_index);
        return std::nullopt;
      }
    }
  }
  if (context->options->enable_end_moves) {
    for (const EndCandidate& candidate :
         BuildEndCandidates(*context->graph, state_copy, *context->options)) {
      if (TryEndMove(context, node_index, state_copy, candidate, depth)) {
        return node_index;
      }
      if (context->budget_exhausted) {
        context->proof->nodes.resize(node_index);
        return std::nullopt;
      }
    }
  }
  context->proof->nodes.resize(node_index);
  return std::nullopt;
}

bool TryCdRootMove(SearchContext* const context, const HtCdCandidate& candidate) {
  const std::vector<HtNeighborPair> c_replies =
      EnumerateHtHamiltonReplies(*context->graph, context->target, candidate.c);
  const std::vector<HtNeighborPair> d_replies =
      EnumerateHtHamiltonReplies(*context->graph, context->target, candidate.d);
  if (d_replies.size() != 0 &&
      c_replies.size() > std::numeric_limits<std::uint64_t>::max() / d_replies.size()) {
    return false;
  }
  const std::uint64_t reply_count = static_cast<std::uint64_t>(c_replies.size()) * d_replies.size();
  if ((context->options->root_options.max_reply_combinations != 0 &&
       reply_count > context->options->root_options.max_reply_combinations) ||
      !MoveReplyCountAllowed(context, reply_count)) {
    return false;
  }

  const NormalizedPathSystem root_state = context->proof->nodes.front().paths;
  const std::size_t checkpoint = context->proof->nodes.size();
  std::vector<HtTreeReply> records;
  records.reserve(static_cast<std::size_t>(reply_count));
  for (const HtNeighborPair& c_reply : c_replies) {
    for (const HtNeighborPair& d_reply : d_replies) {
      if (!ConsumeReply(context)) {
        context->proof->nodes.resize(checkpoint);
        return false;
      }
      HtTreeReply record;
      record.first_pair = c_reply;
      record.second_pair = d_reply;
      NormalizedPathSystem child = AddPaths(root_state,
                                            {{c_reply.first, c_reply.center, c_reply.second},
                                             {d_reply.first, d_reply.center, d_reply.second}},
                                            context->graph->dimension);
      if (!child.valid) {
        record.path_infeasible = true;
        records.push_back(std::move(record));
        continue;
      }
      const std::optional<std::uint32_t> child_index = ProveState(context, std::move(child), 0);
      if (!child_index.has_value()) {
        context->proof->nodes.resize(checkpoint);
        return false;
      }
      record.child_index = *child_index;
      records.push_back(std::move(record));
    }
  }
  HtTreeNode& root = context->proof->nodes.front();
  root.move_type = HtMoveType::kCd;
  root.move_first = candidate.c;
  root.move_second = candidate.d;
  root.replies = std::move(records);
  return true;
}

std::vector<HtNeighborPair> EnumeratePointRepliesForVerifier(const GraphSnapshot& graph,
                                                             const NodeEdge target,
                                                             const std::int32_t center) {
  std::vector<HtNeighborPair> replies;
  const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(center)];
  const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(center) + 1U];
  for (std::int32_t first_offset = begin; first_offset < end; ++first_offset) {
    const std::int32_t first = graph.neighbors[static_cast<std::size_t>(first_offset)];
    for (std::int32_t second_offset = first_offset + 1; second_offset < end; ++second_offset) {
      const std::int32_t second = graph.neighbors[static_cast<std::size_t>(second_offset)];
      if (CanonicalEdge(first, second) == target ||
          !EdgesSurviveTwoOpt(graph, target, CanonicalEdge(center, first)) ||
          !EdgesSurviveTwoOpt(graph, target, CanonicalEdge(center, second))) {
        continue;
      }
      const __int128 original = static_cast<__int128>(graph.Distance(target.u, target.v)) +
                                graph.Distance(center, first) + graph.Distance(center, second);
      const __int128 replacement = static_cast<__int128>(graph.Distance(first, second)) +
                                   graph.Distance(target.u, center) +
                                   graph.Distance(target.v, center);
      if (original <= replacement) {
        replies.push_back({center, first, second});
      }
    }
  }
  return replies;
}

bool IsDefaultPair(const HtNeighborPair& pair) {
  return pair.center == -1 && pair.first == -1 && pair.second == -1;
}

bool IsDefaultEdge(const NodeEdge edge) { return edge.u == -1 && edge.v == -1; }

struct VerifyWork {
  std::uint32_t node_index{};
  NormalizedPathSystem expected_paths;
};

bool QueueChild(const GraphSnapshot& graph, const HtTreeReply& record,
                const NormalizedPathSystem& parent, const std::vector<Path>& additions,
                const std::uint32_t parent_index, const std::size_t node_count,
                std::vector<VerifyWork>* const work, std::string* const reason) {
  NormalizedPathSystem child = AddPaths(parent, additions, graph.dimension);
  if (!child.valid) {
    if (!record.path_infeasible || record.child_index != kNoHtChild) {
      SetReason(reason, "递归 HT 无效 reply 的状态标记非法");
      return false;
    }
    return true;
  }
  if (record.path_infeasible || record.child_index == kNoHtChild ||
      record.child_index <= parent_index || record.child_index >= node_count) {
    SetReason(reason, "递归 HT 合法 reply 的 child index 非法");
    return false;
  }
  work->push_back({record.child_index, std::move(child)});
  return true;
}

} // namespace

HtRecursiveResult ProveEdgeByRecursiveHt(const GraphSnapshot& graph, const NodeEdge raw_target,
                                         const HtRecursiveOptions& options) {
  HtRecursiveResult result;
  HtRecursiveProof& proof = result.proof;
  proof.snapshot_hash = graph.ContentHash();
  proof.target_edge = CanonicalEdge(raw_target.u, raw_target.v);
  proof.cd_mode = options.root_options.cd_mode;

  std::string reason;
  if (!ValidateGraph(graph, &reason) || !ValidateTarget(graph, proof.target_edge, &reason)) {
    proof.reason = reason;
    return result;
  }
  HtCdBatchResult batch;
  try {
    batch = EvaluateHtCdCandidates(graph, proof.target_edge, options.root_options);
  } catch (const std::exception& error) {
    if (options.root_options.candidate_backend == PathCompatibilityBackend::kAuto) {
      try {
        HtShallowOptions cpu_options = options.root_options;
        cpu_options.candidate_backend = PathCompatibilityBackend::kCpu;
        batch = EvaluateHtCdCandidates(graph, proof.target_edge, cpu_options);
      } catch (const std::exception& cpu_error) {
        proof.reason = cpu_error.what();
        return result;
      }
    } else if (options.root_options.candidate_backend == PathCompatibilityBackend::kCuda) {
      result.status = HtSearchStatus::kUnresolved;
      proof.reason = std::string("CUDA 递归 HT c,d 筛选失败: ") + error.what();
      return result;
    } else {
      proof.reason = error.what();
      return result;
    }
  }
  if (batch.candidates.empty()) {
    result.status = HtSearchStatus::kUnresolved;
    proof.reason = "递归 HT 没有可用 c,d 根 move";
    return result;
  }

  const NormalizedPathSystem root_paths =
      NormalizePathSystem({{proof.target_edge.u, proof.target_edge.v}}, graph.dimension);
  if (!root_paths.valid) {
    proof.reason = "递归 HT 无法建立目标边根路径";
    return result;
  }
  SearchContext context{&graph, proof.target_edge, &options, &proof, false};
  for (const HtCdCandidate& candidate : batch.candidates) {
    ++proof.cd_candidates_tested;
    proof.nodes.clear();
    HtTreeNode root;
    root.paths = root_paths;
    proof.nodes.push_back(std::move(root));
    if (TryCdRootMove(&context, candidate)) {
      proof.proven = true;
      proof.reason = "递归 HT 的一个 c,d 根 move 已完成全部 AND replies";
      std::string verify_reason;
      if (!VerifyHtRecursiveProof(graph, proof, &verify_reason)) {
        proof.proven = false;
        proof.reason = "内部递归 HT 复核失败: " + verify_reason;
        result.status = HtSearchStatus::kInvalid;
        return result;
      }
      result.status = HtSearchStatus::kProven;
      return result;
    }
    proof.nodes.clear();
    if (context.budget_exhausted) {
      break;
    }
  }
  result.status = HtSearchStatus::kUnresolved;
  proof.reason =
      context.budget_exhausted ? "递归 HT 资源预算耗尽" : "递归 HT 的全部根 moves 均未解决";
  return result;
}

bool VerifyHtRecursiveProof(const GraphSnapshot& graph, const HtRecursiveProof& proof,
                            std::string* const reason) {
  std::string graph_reason;
  if (!proof.proven || proof.snapshot_hash != graph.ContentHash() || proof.nodes.empty() ||
      proof.nodes.size() > 1000000U || !ValidateGraph(graph, &graph_reason) ||
      !ValidateTarget(graph, proof.target_edge, &graph_reason)) {
    SetReason(reason, graph_reason.empty() ? "递归 HT proof 的状态、快照或根非法" : graph_reason);
    return false;
  }
  const NormalizedPathSystem root_paths =
      NormalizePathSystem({{proof.target_edge.u, proof.target_edge.v}}, graph.dimension);
  std::vector<VerifyWork> work = {{0, root_paths}};
  std::vector<bool> visited(proof.nodes.size(), false);
  while (!work.empty()) {
    VerifyWork current = std::move(work.back());
    work.pop_back();
    if (current.node_index >= proof.nodes.size() || visited[current.node_index]) {
      SetReason(reason, "递归 HT arena 含环、共享 child 或越界索引");
      return false;
    }
    visited[current.node_index] = true;
    const HtTreeNode& node = proof.nodes[current.node_index];
    if (!SamePaths(node.paths, current.expected_paths)) {
      SetReason(reason, "递归 HT 节点路径系统与父 reply 不一致");
      return false;
    }

    if (node.move_type == HtMoveType::kLeaf) {
      if (node.move_first != -1 || node.move_second != -1 || !node.replies.empty()) {
        SetReason(reason, "递归 HT leaf 含多余 move 数据");
        return false;
      }
      std::string leaf_reason;
      if (!VerifyPathSystemKOptProof(graph, node.paths, proof.target_edge, node.leaf_proof,
                                     &leaf_reason)) {
        SetReason(reason, "递归 HT leaf 失败: " + leaf_reason);
        return false;
      }
      continue;
    }

    if (node.leaf_proof.proven) {
      SetReason(reason, "递归 HT 非 leaf 节点携带已授权 leaf proof");
      return false;
    }
    if (node.move_type == HtMoveType::kCd) {
      if (current.node_index != 0 || !CdMoveAdmissible(graph, proof.target_edge, node.move_first,
                                                       node.move_second, proof.cd_mode)) {
        SetReason(reason, "递归 HT c,d 根 move 非法");
        return false;
      }
      const std::vector<HtNeighborPair> first_replies =
          EnumeratePointRepliesForVerifier(graph, proof.target_edge, node.move_first);
      const std::vector<HtNeighborPair> second_replies =
          EnumeratePointRepliesForVerifier(graph, proof.target_edge, node.move_second);
      if (second_replies.size() != 0 &&
          first_replies.size() > std::numeric_limits<std::size_t>::max() / second_replies.size()) {
        SetReason(reason, "递归 HT c,d reply 数量溢出");
        return false;
      }
      const std::size_t expected_count = first_replies.size() * second_replies.size();
      if (node.replies.size() != expected_count) {
        SetReason(reason, "递归 HT c,d 根未覆盖完整 reply 笛卡尔积");
        return false;
      }
      std::size_t record_index = 0;
      for (const HtNeighborPair& first_reply : first_replies) {
        for (const HtNeighborPair& second_reply : second_replies) {
          const HtTreeReply& record = node.replies[record_index++];
          if (record.first_pair != first_reply || record.second_pair != second_reply ||
              !IsDefaultEdge(record.edge)) {
            SetReason(reason, "递归 HT c,d reply 内容非法");
            return false;
          }
          if (!QueueChild(graph, record, node.paths,
                          {{first_reply.first, first_reply.center, first_reply.second},
                           {second_reply.first, second_reply.center, second_reply.second}},
                          current.node_index, proof.nodes.size(), &work, reason)) {
            return false;
          }
        }
      }
      continue;
    }

    if (current.node_index == 0) {
      SetReason(reason, "递归 HT 根节点不是 c,d move");
      return false;
    }
    if (node.move_type == HtMoveType::kPoint) {
      if (node.move_second != -1 || node.move_first < 0 || node.move_first >= graph.dimension ||
          ContainsNode(node.paths, node.move_first)) {
        SetReason(reason, "递归 HT point move 非法");
        return false;
      }
      const std::vector<HtNeighborPair> replies =
          EnumeratePointRepliesForVerifier(graph, proof.target_edge, node.move_first);
      if (node.replies.size() != replies.size()) {
        SetReason(reason, "递归 HT point move 未覆盖全部 replies");
        return false;
      }
      for (std::size_t index = 0; index < replies.size(); ++index) {
        const HtTreeReply& record = node.replies[index];
        if (record.first_pair != replies[index] || !IsDefaultPair(record.second_pair) ||
            !IsDefaultEdge(record.edge)) {
          SetReason(reason, "递归 HT point reply 内容非法");
          return false;
        }
        if (!QueueChild(graph, record, node.paths,
                        {{replies[index].first, replies[index].center, replies[index].second}},
                        current.node_index, proof.nodes.size(), &work, reason)) {
          return false;
        }
      }
      continue;
    }
    if (node.move_type == HtMoveType::kEnd) {
      bool endpoint_found = false;
      for (const Path& path : node.paths.paths) {
        endpoint_found =
            endpoint_found || (path.front() == node.move_first && path[1] == node.move_second) ||
            (path.back() == node.move_first && path[path.size() - 2U] == node.move_second);
      }
      if (!endpoint_found) {
        SetReason(reason, "递归 HT end move 不是当前路径端点");
        return false;
      }
      const std::vector<NodeEdge> replies =
          EnumerateEndReplies(graph, node.move_first, node.move_second);
      if (node.replies.size() != replies.size()) {
        SetReason(reason, "递归 HT end move 未覆盖全部 replies");
        return false;
      }
      for (std::size_t index = 0; index < replies.size(); ++index) {
        const HtTreeReply& record = node.replies[index];
        const std::int32_t neighbor =
            replies[index].u == node.move_first ? replies[index].v : replies[index].u;
        if (record.edge != replies[index] || !IsDefaultPair(record.first_pair) ||
            !IsDefaultPair(record.second_pair)) {
          SetReason(reason, "递归 HT end reply 内容非法");
          return false;
        }
        if (!QueueChild(graph, record, node.paths, {{node.move_first, neighbor}},
                        current.node_index, proof.nodes.size(), &work, reason)) {
          return false;
        }
      }
      continue;
    }
    SetReason(reason, "递归 HT move type 未知");
    return false;
  }

  if (std::find(visited.begin(), visited.end(), false) != visited.end()) {
    SetReason(reason, "递归 HT arena 含未引用节点");
    return false;
  }
  SetReason(reason, "OK");
  return true;
}

} // namespace cudaee
