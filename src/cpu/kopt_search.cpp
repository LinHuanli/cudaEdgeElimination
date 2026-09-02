#include "cuda_edge_elimination/local_search.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef CUDAEE_HAS_OPENMP
#include <omp.h>
#endif

namespace cudaee {
namespace {

using EdgeSet = std::set<NodeEdge>;
constexpr std::array<std::size_t, 3U> kKOptTemplateCounts = {4U, 25U, 208U};
constexpr std::size_t kKOptCostPortCapacity = KOptCostTask{}.port_nodes.size();
constexpr std::size_t kKOptCpuParallelMinCells = 8192U;
constexpr std::size_t kKOptCpuDistanceCacheMaxNodes = 512U;
constexpr std::size_t kKOptCpuDistanceCacheMaxGraphNodes = 1048576U;
constexpr int kKOptCpuMaxThreads = 8;

using SteadyClock = std::chrono::steady_clock;

double ElapsedMilliseconds(const SteadyClock::time_point begin) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - begin).count();
}

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

void SetReason(std::string* const reason, std::string value) {
  if (reason != nullptr) {
    *reason = std::move(value);
  }
}

void HashByte(std::uint64_t* const hash, const std::uint8_t value) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  *hash ^= value;
  *hash *= kFnvPrime;
}

void HashUint32(std::uint64_t* const hash, const std::uint32_t value) {
  for (std::uint32_t shift = 0; shift < 32; shift += 8) {
    HashByte(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void HashUint64(std::uint64_t* const hash, const std::uint64_t value) {
  for (std::uint32_t shift = 0; shift < 64; shift += 8) {
    HashByte(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

class DisjointSet {
public:
  explicit DisjointSet(const std::size_t count) : parent_(count), rank_(count, 0) {
    for (std::size_t index = 0; index < count; ++index) {
      parent_[index] = index;
    }
  }

  std::size_t Find(const std::size_t value) {
    if (parent_[value] != value) {
      parent_[value] = Find(parent_[value]);
    }
    return parent_[value];
  }

  void Unite(const std::size_t first, const std::size_t second) {
    std::size_t first_root = Find(first);
    std::size_t second_root = Find(second);
    if (first_root == second_root) {
      return;
    }
    if (rank_[first_root] < rank_[second_root]) {
      std::swap(first_root, second_root);
    }
    parent_[second_root] = first_root;
    if (rank_[first_root] == rank_[second_root]) {
      ++rank_[first_root];
    }
  }

private:
  std::vector<std::size_t> parent_;
  std::vector<std::uint8_t> rank_;
};

struct TourContext {
  std::vector<std::int32_t> nodes;
  std::vector<std::int32_t> tour;
  std::vector<std::int32_t> endpoint_nodes;
  EdgeSet path_edges;
  EdgeSet outside_edges;
  EdgeSet all_edges;
  std::vector<std::size_t> selectable_positions;
  // 非路径边保持 -1；cost task 只读取 selectable positions 对应的精确整数成本。
  std::vector<std::int64_t> path_edge_costs_by_tour_position;
};

// batch 生成器只需为同一个 path 对象认证一次；公开 verifier 仍独立走 dense 规范化。
class KOptPathValidationBinding {
public:
  KOptPathValidationBinding(const GraphSnapshot& graph, const NormalizedPathSystem& paths)
      : graph_(&graph), paths_(&paths) {
    if (!graph.integer_coordinates || !graph.integer_distance_safe || graph.dimension <= 0 ||
        graph.points.size() != static_cast<std::size_t>(graph.dimension)) {
      reason_ = "k-opt 证明只支持平方距离安全的整数坐标图";
      return;
    }
    if (!paths.valid || paths.paths.empty() || paths.paths.size() > kMaxTestablePathCount) {
      reason_ = "输入不是可测试的规范路径系统";
      return;
    }
    const NormalizedPathSystem checked =
        detail::NormalizeSparsePathSystem(paths.paths, graph.dimension);
    if (!checked.valid || checked.paths != paths.paths || checked.edge_count != paths.edge_count) {
      reason_ = "路径系统未通过独立规范化复核";
      return;
    }
    valid_ = true;
  }

  [[nodiscard]] bool Matches(const GraphSnapshot& graph, const NormalizedPathSystem& paths) const {
    return graph_ == &graph && paths_ == &paths;
  }

  [[nodiscard]] bool valid() const { return valid_; }
  [[nodiscard]] const std::string& reason() const { return reason_; }

private:
  const GraphSnapshot* graph_{};
  const NormalizedPathSystem* paths_{};
  bool valid_{false};
  std::string reason_;
};

bool BuildCycle(const std::int32_t dimension, const std::vector<std::int32_t>& nodes,
                const EdgeSet& edges, std::vector<std::int32_t>* const tour,
                std::string* const reason) {
  if (nodes.size() < 3 || edges.size() != nodes.size()) {
    SetReason(reason, "局部边集不是简单巡回的规模");
    return false;
  }
  std::vector<bool> in_nodes(static_cast<std::size_t>(dimension), false);
  std::vector<std::array<std::int32_t, 2>> adjacency(static_cast<std::size_t>(dimension));
  std::vector<std::uint8_t> degrees(static_cast<std::size_t>(dimension), 0);
  for (const std::int32_t node : nodes) {
    if (node < 0 || node >= dimension || in_nodes[static_cast<std::size_t>(node)]) {
      SetReason(reason, "局部节点集合非法");
      return false;
    }
    in_nodes[static_cast<std::size_t>(node)] = true;
  }
  for (const NodeEdge& edge : edges) {
    if (edge.u < 0 || edge.v < 0 || edge.u >= dimension || edge.v >= dimension ||
        edge.u >= edge.v || !in_nodes[static_cast<std::size_t>(edge.u)] ||
        !in_nodes[static_cast<std::size_t>(edge.v)]) {
      SetReason(reason, "局部巡回包含非法边");
      return false;
    }
    for (const auto [from, to] : {std::pair{edge.u, edge.v}, std::pair{edge.v, edge.u}}) {
      auto& degree = degrees[static_cast<std::size_t>(from)];
      if (degree >= 2) {
        SetReason(reason, "局部巡回存在度数大于 2 的节点");
        return false;
      }
      adjacency[static_cast<std::size_t>(from)][degree++] = to;
    }
  }
  for (const std::int32_t node : nodes) {
    if (degrees[static_cast<std::size_t>(node)] != 2) {
      SetReason(reason, "局部巡回存在非 2 度节点");
      return false;
    }
  }

  const std::int32_t start = *std::min_element(nodes.begin(), nodes.end());
  std::vector<bool> visited(static_cast<std::size_t>(dimension), false);
  tour->clear();
  tour->reserve(nodes.size());
  std::int32_t previous = -1;
  std::int32_t current = start;
  for (std::size_t step = 0; step < nodes.size(); ++step) {
    if (visited[static_cast<std::size_t>(current)]) {
      SetReason(reason, "局部巡回提前闭合");
      return false;
    }
    visited[static_cast<std::size_t>(current)] = true;
    tour->push_back(current);
    const auto& neighbors = adjacency[static_cast<std::size_t>(current)];
    std::int32_t next = -1;
    if (previous == -1) {
      next = std::min(neighbors[0], neighbors[1]);
    } else {
      next = neighbors[0] == previous ? neighbors[1] : neighbors[0];
    }
    previous = current;
    current = next;
  }
  if (current != start) {
    SetReason(reason, "局部边集含多个回路或开放链");
    return false;
  }
  return true;
}

bool BuildTourContext(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                      const EndpointMatching& outside, const std::optional<NodeEdge>& required_edge,
                      TourContext* const context, std::string* const reason,
                      const KOptPathValidationBinding* const validation_binding = nullptr) {
  if (validation_binding != nullptr) {
    if (!validation_binding->Matches(graph, paths)) {
      throw std::logic_error("k-opt path validation binding 与输入对象不一致");
    }
    if (!validation_binding->valid()) {
      SetReason(reason, validation_binding->reason());
      return false;
    }
  } else {
    if (!graph.integer_coordinates || !graph.integer_distance_safe || graph.dimension <= 0 ||
        graph.points.size() != static_cast<std::size_t>(graph.dimension)) {
      SetReason(reason, "k-opt 证明只支持平方距离安全的整数坐标图");
      return false;
    }
    if (!paths.valid || paths.paths.empty() || paths.paths.size() > kMaxTestablePathCount) {
      SetReason(reason, "输入不是可测试的规范路径系统");
      return false;
    }
    const NormalizedPathSystem checked = NormalizePathSystem(paths.paths, graph.dimension);
    if (!checked.valid || checked.paths != paths.paths || checked.edge_count != paths.edge_count) {
      SetReason(reason, "路径系统未通过独立规范化复核");
      return false;
    }
  }
  const auto path_count = static_cast<std::uint32_t>(paths.paths.size());
  if (!IsPerfectEndpointMatching(outside, path_count)) {
    SetReason(reason, "outside matching 非法");
    return false;
  }

  context->nodes.clear();
  context->tour.clear();
  context->endpoint_nodes.assign(std::size_t{2} * path_count, -1);
  context->path_edges.clear();
  context->outside_edges.clear();
  context->all_edges.clear();
  context->selectable_positions.clear();
  context->path_edge_costs_by_tour_position.clear();
  for (std::uint32_t path_index = 0; path_index < path_count; ++path_index) {
    const Path& path = paths.paths[path_index];
    const std::size_t first_endpoint = std::size_t{2} * path_index;
    context->endpoint_nodes[first_endpoint] = path.front();
    context->endpoint_nodes[first_endpoint + 1] = path.back();
    context->nodes.insert(context->nodes.end(), path.begin(), path.end());
    for (std::size_t index = 1; index < path.size(); ++index) {
      const NodeEdge edge = CanonicalEdge(path[index - 1], path[index]);
      context->path_edges.insert(edge);
      context->all_edges.insert(edge);
    }
  }

  for (std::uint32_t endpoint = 0; endpoint < 2U * path_count; ++endpoint) {
    const std::uint32_t partner = outside.mate[endpoint];
    if (endpoint >= partner) {
      continue;
    }
    const NodeEdge edge =
        CanonicalEdge(context->endpoint_nodes[endpoint], context->endpoint_nodes[partner]);
    if (edge.u == edge.v || !context->outside_edges.insert(edge).second ||
        !context->all_edges.insert(edge).second) {
      SetReason(reason, "outside matching 与路径边形成重复边或自环");
      return false;
    }
  }
  if (!BuildCycle(graph.dimension, context->nodes, context->all_edges, &context->tour, reason)) {
    return false;
  }

  context->path_edge_costs_by_tour_position.assign(context->tour.size(), -1);
  for (std::size_t position = 0; position < context->tour.size(); ++position) {
    const NodeEdge edge = CanonicalEdge(context->tour[position],
                                        context->tour[(position + 1) % context->tour.size()]);
    if (context->path_edges.contains(edge)) {
      const std::int64_t distance = graph.Distance(edge.u, edge.v);
      if (distance < 0) {
        SetReason(reason, "路径边距离出现负值");
        return false;
      }
      context->path_edge_costs_by_tour_position[position] = distance;
      context->selectable_positions.push_back(position);
    }
  }
  if (context->selectable_positions.size() != paths.edge_count) {
    SetReason(reason, "巡回中的路径边数量不一致");
    return false;
  }

  if (required_edge.has_value()) {
    const NodeEdge required = CanonicalEdge(required_edge->u, required_edge->v);
    const auto iterator = std::find_if(
        context->selectable_positions.begin(), context->selectable_positions.end(),
        [&](const std::size_t position) {
          return CanonicalEdge(context->tour[position],
                               context->tour[(position + 1) % context->tour.size()]) == required;
        });
    if (iterator == context->selectable_positions.end()) {
      SetReason(reason, "required edge 不在路径系统中");
      return false;
    }
    std::rotate(context->selectable_positions.begin(), iterator,
                context->selectable_positions.end());
  }
  return true;
}

EndpointMatching EmptyMatching(const std::uint32_t pair_count) {
  EndpointMatching matching;
  matching.endpoint_count = static_cast<std::uint8_t>(2U * pair_count);
  matching.mate.fill(kUnmatchedEndpoint);
  return matching;
}

bool PairEndpoints(EndpointMatching* const matching, const std::uint32_t first,
                   const std::uint32_t second) {
  if (first == second || first >= matching->endpoint_count || second >= matching->endpoint_count ||
      matching->mate[first] != kUnmatchedEndpoint || matching->mate[second] != kUnmatchedEndpoint) {
    return false;
  }
  matching->mate[first] = static_cast<std::uint8_t>(second);
  matching->mate[second] = static_cast<std::uint8_t>(first);
  return true;
}

bool BuildComponentMatching(const TourContext& context,
                            const std::span<const std::size_t> deleted_positions,
                            std::vector<std::int32_t>* const port_nodes,
                            EndpointMatching* const components, EdgeSet* const remaining_edges,
                            std::string* const reason) {
  const std::size_t node_count = context.tour.size();
  const auto k = static_cast<std::uint32_t>(deleted_positions.size());
  if (node_count < 3 || k < 3) {
    SetReason(reason, "k-opt 分量构造的节点数或 k 非法");
    return false;
  }
  std::vector<bool> deleted(node_count, false);
  for (const std::size_t position : deleted_positions) {
    if (position >= node_count || deleted[position]) {
      SetReason(reason, "删除位置重复或越界");
      return false;
    }
    deleted[position] = true;
  }

  DisjointSet sets(node_count);
  *remaining_edges = context.all_edges;
  for (std::size_t position = 0; position < node_count; ++position) {
    if (deleted[position]) {
      remaining_edges->erase(
          CanonicalEdge(context.tour[position], context.tour[(position + 1) % node_count]));
    } else {
      sets.Unite(position, (position + 1) % node_count);
    }
  }

  port_nodes->clear();
  port_nodes->reserve(2U * deleted_positions.size());
  std::map<std::size_t, std::vector<std::uint32_t>> component_ports;
  for (const std::size_t position : deleted_positions) {
    const std::uint32_t first_port = static_cast<std::uint32_t>(port_nodes->size());
    port_nodes->push_back(context.tour[position]);
    component_ports[sets.Find(position)].push_back(first_port);
    const std::uint32_t second_port = static_cast<std::uint32_t>(port_nodes->size());
    port_nodes->push_back(context.tour[(position + 1) % node_count]);
    component_ports[sets.Find((position + 1) % node_count)].push_back(second_port);
  }
  if (component_ports.size() != k) {
    SetReason(reason, "删除 k 条巡回边后没有得到 k 条路径分量");
    return false;
  }
  *components = EmptyMatching(k);
  for (const auto& [root, ports] : component_ports) {
    static_cast<void>(root);
    if (ports.size() != 2 || !PairEndpoints(components, ports[0], ports[1])) {
      SetReason(reason, "路径分量端口数量不是 2");
      return false;
    }
  }
  return true;
}

bool SumEdgeCosts(const GraphSnapshot& graph, const std::vector<NodeEdge>& edges,
                  std::int64_t* const result, std::string* const reason) {
  __int128 total = 0;
  for (const NodeEdge& edge : edges) {
    const std::int64_t distance = graph.Distance(edge.u, edge.v);
    if (distance < 0) {
      SetReason(reason, "距离出现负值");
      return false;
    }
    total += distance;
  }
  if (total > std::numeric_limits<std::int64_t>::max()) {
    SetReason(reason, "k-opt 成本求和溢出 int64_t");
    return false;
  }
  *result = static_cast<std::int64_t>(total);
  return true;
}

bool ExtractInsideMatching(const TourContext& context, const EndpointMatching& outside,
                           const std::vector<std::int32_t>& improved_tour,
                           EndpointMatching* const inside, std::string* const reason) {
  const auto path_count = static_cast<std::uint32_t>(context.endpoint_nodes.size() / 2U);
  std::vector<std::int32_t> endpoint_name(
      static_cast<std::size_t>(*std::max_element(context.nodes.begin(), context.nodes.end()) + 1),
      -1);
  for (std::uint32_t endpoint = 0; endpoint < context.endpoint_nodes.size(); ++endpoint) {
    endpoint_name[static_cast<std::size_t>(context.endpoint_nodes[endpoint])] =
        static_cast<std::int32_t>(endpoint);
  }
  std::vector<std::uint32_t> endpoint_order;
  const std::size_t endpoint_count = std::size_t{2} * path_count;
  endpoint_order.reserve(endpoint_count);
  for (const std::int32_t node : improved_tour) {
    if (node < static_cast<std::int32_t>(endpoint_name.size()) &&
        endpoint_name[static_cast<std::size_t>(node)] >= 0) {
      endpoint_order.push_back(
          static_cast<std::uint32_t>(endpoint_name[static_cast<std::size_t>(node)]));
    }
  }
  if (endpoint_order.size() != endpoint_count) {
    SetReason(reason, "改善巡回没有恰好访问全部路径端点");
    return false;
  }

  *inside = EmptyMatching(path_count);
  if (outside.mate[endpoint_order[0]] == endpoint_order[1]) {
    for (std::uint32_t index = 1; index + 1 < endpoint_order.size(); index += 2) {
      if (!PairEndpoints(inside, endpoint_order[index], endpoint_order[index + 1])) {
        SetReason(reason, "无法从改善巡回提取 inside matching");
        return false;
      }
    }
    if (!PairEndpoints(inside, endpoint_order.back(), endpoint_order.front())) {
      SetReason(reason, "无法闭合 inside matching");
      return false;
    }
  } else {
    for (std::uint32_t index = 0; index < endpoint_order.size(); index += 2) {
      if (!PairEndpoints(inside, endpoint_order[index], endpoint_order[index + 1])) {
        SetReason(reason, "无法从改善巡回提取 inside matching");
        return false;
      }
    }
  }
  if (!IsAlternatingHamiltonianCycle(outside, *inside, path_count)) {
    SetReason(reason, "提取的 inside matching 不覆盖源 outside matching");
    return false;
  }
  return true;
}

struct ReconnectAttempt {
  std::optional<KOptWitness> witness;
  bool fatal{false};
  std::string reason;
  std::uint64_t matchings_tested{};
  std::uint64_t candidate_templates_rechecked{};
  std::uint64_t completeness_templates_tested{};
  bool used_completeness_fallback{false};
  double candidate_recheck_ms{};
  double completeness_fallback_ms{};
};

struct ExactTourBlock {
  std::int32_t first{-1};
  std::int32_t second{-1};
  bool paired{false};
};

std::int32_t BlockEntry(const ExactTourBlock& block, const std::uint32_t orientation) {
  return orientation == 0 ? block.first : block.second;
}

std::int32_t BlockExit(const ExactTourBlock& block, const std::uint32_t orientation) {
  if (!block.paired) {
    return block.first;
  }
  return orientation == 0 ? block.second : block.first;
}

std::uint32_t BlockOrientationCount(const ExactTourBlock& block) { return block.paired ? 2U : 1U; }

std::vector<ExactTourBlock> BuildExactTourBlocks(const TourContext& context,
                                                 const EndpointMatching& outside) {
  std::vector<ExactTourBlock> blocks;
  std::vector<bool> is_endpoint(
      static_cast<std::size_t>(*std::max_element(context.nodes.begin(), context.nodes.end()) + 1),
      false);
  for (std::uint32_t endpoint = 0; endpoint < outside.endpoint_count; ++endpoint) {
    const std::uint32_t partner = outside.mate[endpoint];
    if (endpoint >= partner) {
      continue;
    }
    const std::int32_t first = context.endpoint_nodes[endpoint];
    const std::int32_t second = context.endpoint_nodes[partner];
    blocks.push_back({first, second, true});
    is_endpoint[static_cast<std::size_t>(first)] = true;
    is_endpoint[static_cast<std::size_t>(second)] = true;
  }
  std::vector<std::int32_t> singleton_nodes;
  for (const std::int32_t node : context.nodes) {
    if (!is_endpoint[static_cast<std::size_t>(node)]) {
      singleton_nodes.push_back(node);
    }
  }
  std::sort(singleton_nodes.begin(), singleton_nodes.end());
  for (const std::int32_t node : singleton_nodes) {
    blocks.push_back({node, node, false});
  }
  return blocks;
}

KOptCostTask BuildKOptCostTask(const TourContext& context,
                               const std::span<const std::size_t> deleted_positions) {
  KOptCostTask task;
  if (deleted_positions.size() < 3U ||
      deleted_positions.size() > task.port_nodes.size() / 2U) {
    throw std::runtime_error("无法构造 k-opt cost task: 删除位置数量非法");
  }
  __int128 deleted_cost = 0;
  for (std::size_t edge = 0; edge < deleted_positions.size(); ++edge) {
    const std::size_t position = deleted_positions[edge];
    if (position >= context.tour.size() ||
        position >= context.path_edge_costs_by_tour_position.size() ||
        context.path_edge_costs_by_tour_position[position] < 0) {
      throw std::runtime_error("无法构造 k-opt cost task: 删除位置或缓存成本非法");
    }
    task.port_nodes[2U * edge] = context.tour[position];
    task.port_nodes[2U * edge + 1U] = context.tour[(position + 1) % context.tour.size()];
    deleted_cost += context.path_edge_costs_by_tour_position[position];
  }
  if (deleted_cost > std::numeric_limits<std::int64_t>::max()) {
    throw std::runtime_error("无法构造 k-opt cost task: k-opt 成本求和溢出 int64_t");
  }
  task.deleted_cost = static_cast<std::int64_t>(deleted_cost);
  return task;
}

ReconnectAttempt TryReconnect(const GraphSnapshot& graph, const TourContext& context,
                              const EndpointMatching& outside,
                              const std::span<const std::size_t> deleted_positions,
                              const std::vector<EndpointMatching>& reconnect_matchings) {
  ReconnectAttempt attempt;
  std::vector<std::int32_t> port_nodes;
  EndpointMatching components;
  EdgeSet remaining_edges;
  if (!BuildComponentMatching(context, deleted_positions, &port_nodes, &components,
                              &remaining_edges, &attempt.reason)) {
    attempt.fatal = true;
    return attempt;
  }

  std::vector<NodeEdge> deleted_edges;
  deleted_edges.reserve(deleted_positions.size());
  for (const std::size_t position : deleted_positions) {
    deleted_edges.push_back(
        CanonicalEdge(context.tour[position], context.tour[(position + 1) % context.tour.size()]));
  }
  std::sort(deleted_edges.begin(), deleted_edges.end());
  const EdgeSet deleted_edge_set(deleted_edges.begin(), deleted_edges.end());
  std::int64_t deleted_cost = 0;
  if (!SumEdgeCosts(graph, deleted_edges, &deleted_cost, &attempt.reason)) {
    attempt.fatal = true;
    return attempt;
  }

  const auto k = static_cast<std::uint32_t>(deleted_positions.size());
  for (const EndpointMatching& reconnect : reconnect_matchings) {
    ++attempt.matchings_tested;
    if (!IsAlternatingHamiltonianCycle(components, reconnect, k)) {
      continue;
    }
    std::vector<NodeEdge> added_edges;
    EdgeSet unique_added;
    bool valid = true;
    for (std::uint32_t port = 0; port < reconnect.endpoint_count; ++port) {
      const std::uint32_t partner = reconnect.mate[port];
      if (port >= partner) {
        continue;
      }
      const NodeEdge edge = CanonicalEdge(port_nodes[port], port_nodes[partner]);
      // proper k-opt 必须真正替换全部删除边；这也与参考 reconnect template 的语义一致。
      if (edge.u == edge.v || remaining_edges.contains(edge) || deleted_edge_set.contains(edge) ||
          !unique_added.insert(edge).second) {
        valid = false;
        break;
      }
      added_edges.push_back(edge);
    }
    if (!valid || added_edges.size() != k) {
      continue;
    }
    std::sort(added_edges.begin(), added_edges.end());
    std::int64_t added_cost = 0;
    if (!SumEdgeCosts(graph, added_edges, &added_cost, &attempt.reason)) {
      attempt.fatal = true;
      return attempt;
    }
    if (added_cost >= deleted_cost) {
      continue;
    }

    EdgeSet improved_edges = remaining_edges;
    improved_edges.insert(added_edges.begin(), added_edges.end());
    std::vector<std::int32_t> improved_tour;
    std::string cycle_reason;
    if (!BuildCycle(graph.dimension, context.nodes, improved_edges, &improved_tour,
                    &cycle_reason)) {
      continue;
    }
    EndpointMatching inside;
    if (!ExtractInsideMatching(context, outside, improved_tour, &inside, &attempt.reason)) {
      attempt.fatal = true;
      return attempt;
    }

    KOptWitness witness;
    witness.k = k;
    witness.deleted_cost = deleted_cost;
    witness.added_cost = added_cost;
    witness.deleted_edges = std::move(deleted_edges);
    witness.added_edges = std::move(added_edges);
    witness.inside_matching = inside;
    attempt.witness = std::move(witness);
    return attempt;
  }
  return attempt;
}

ReconnectAttempt TryReconnectFromCostRow(const GraphSnapshot& graph, const TourContext& context,
                                         const EndpointMatching& outside,
                                         const std::span<const std::size_t> deleted_positions,
                                         const KOptCostTask& task,
                                         const std::vector<EndpointMatching>& reconnect_templates,
                                         const std::span<const std::int64_t> added_costs,
                                         const std::string_view cost_backend,
                                         const bool cost_matrix_cpu_verified) {
  if (added_costs.size() != reconnect_templates.size() ||
      (cost_backend != "cpu" && cost_backend != "cuda") ||
      (cost_backend == "cpu" && !cost_matrix_cpu_verified)) {
    ReconnectAttempt invalid;
    invalid.fatal = true;
    invalid.reason = "k-opt cost row 或后端非法";
    return invalid;
  }
  const SteadyClock::time_point candidate_begin = SteadyClock::now();
  std::uint64_t candidate_templates_rechecked = 0U;
  for (std::size_t template_index = 0U; template_index < reconnect_templates.size();
       ++template_index) {
    if (added_costs[template_index] >= task.deleted_cost) {
      continue;
    }
    ++candidate_templates_rechecked;
    const std::vector<EndpointMatching> preferred = {reconnect_templates[template_index]};
    ReconnectAttempt attempt = TryReconnect(graph, context, outside, deleted_positions, preferred);
    if (attempt.fatal) {
      attempt.candidate_templates_rechecked = candidate_templates_rechecked;
      attempt.candidate_recheck_ms = ElapsedMilliseconds(candidate_begin);
      return attempt;
    }
    if (attempt.witness.has_value()) {
      // 指标记录规范 CPU 枚举到该模板的位置，而不是 GPU 预筛选的额外工作量。
      attempt.matchings_tested = static_cast<std::uint64_t>(template_index + 1U);
      attempt.candidate_templates_rechecked = candidate_templates_rechecked;
      attempt.candidate_recheck_ms = ElapsedMilliseconds(candidate_begin);
      return attempt;
    }
  }
  const double candidate_recheck_ms = ElapsedMilliseconds(candidate_begin);
  if (cost_backend == "cuda" && !cost_matrix_cpu_verified) {
    // CUDA 仍只是候选器；没有 CPU 接受的候选时完整枚举，保证 completeness。
    const SteadyClock::time_point completeness_begin = SteadyClock::now();
    ReconnectAttempt fallback =
        TryReconnect(graph, context, outside, deleted_positions, reconnect_templates);
    fallback.candidate_templates_rechecked = candidate_templates_rechecked;
    fallback.completeness_templates_tested = fallback.matchings_tested;
    fallback.used_completeness_fallback = true;
    fallback.candidate_recheck_ms = candidate_recheck_ms;
    fallback.completeness_fallback_ms = ElapsedMilliseconds(completeness_begin);
    return fallback;
  }
  ReconnectAttempt exhausted;
  exhausted.matchings_tested = static_cast<std::uint64_t>(reconnect_templates.size());
  exhausted.candidate_templates_rechecked = candidate_templates_rechecked;
  exhausted.candidate_recheck_ms = candidate_recheck_ms;
  return exhausted;
}

bool AdvanceCombination(std::vector<std::size_t>* const combination, const std::size_t item_count) {
  if (combination->size() <= 1) {
    return false;
  }
  std::size_t position = combination->size() - 1;
  while (position >= 1 &&
         (*combination)[position] == item_count - (combination->size() - position)) {
    if (position == 1) {
      return false;
    }
    --position;
  }
  ++(*combination)[position];
  for (std::size_t next = position + 1; next < combination->size(); ++next) {
    (*combination)[next] = (*combination)[next - 1] + 1;
  }
  return true;
}

bool AddWithoutOverflow(std::uint64_t* const total, const std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
    return false;
  }
  *total += value;
  return true;
}

std::size_t KOptCostCellCount(const std::uint32_t k, const std::size_t task_count) {
  if (k < 3U || k > 5U) {
    throw std::invalid_argument("k-opt cost cell 计数的 k 非法");
  }
  const std::size_t template_count = kKOptTemplateCounts[static_cast<std::size_t>(k - 3U)];
  if (task_count > std::numeric_limits<std::size_t>::max() / template_count) {
    throw std::overflow_error("k-opt cost cell 计数溢出");
  }
  return task_count * template_count;
}

bool IsCpuLongTail(const KOptSearchOptions& options, const std::size_t cell_count) {
  return options.cost_backend == PathCompatibilityBackend::kAuto &&
         options.cuda_min_cost_cells != 0U && cell_count < options.cuda_min_cost_cells;
}

PathCompatibilityBackend SelectKOptCostBackend(const KOptSearchOptions& options,
                                               const std::size_t cell_count) {
  return IsCpuLongTail(options, cell_count) ? PathCompatibilityBackend::kCpu : options.cost_backend;
}

void ValidateKOptCostTask(const GraphSnapshot& graph, const std::uint32_t k,
                          const KOptCostTask& task) {
  if (task.deleted_cost < 0) {
    throw std::invalid_argument("k-opt cost task 的 deleted_cost 为负");
  }
  // k<=5，固定数组线性查重比为每个 cost row 构造红黑树更小且保持原检查顺序。
  std::array<NodeEdge, 5U> deleted_edges{};
  std::uint32_t deleted_count = 0U;
  for (std::uint32_t edge = 0; edge < k; ++edge) {
    const std::size_t first_port = std::size_t{2} * edge;
    const std::int32_t first = task.port_nodes[first_port];
    const std::int32_t second = task.port_nodes[first_port + 1];
    if (first < 0 || second < 0 || first >= graph.dimension || second >= graph.dimension ||
        first == second) {
      throw std::invalid_argument("k-opt cost task 包含非法或重复删除边");
    }
    const NodeEdge canonical = CanonicalEdge(first, second);
    if (std::find(deleted_edges.begin(), deleted_edges.begin() + deleted_count, canonical) !=
        deleted_edges.begin() + deleted_count) {
      throw std::invalid_argument("k-opt cost task 包含非法或重复删除边");
    }
    deleted_edges[deleted_count++] = canonical;
  }
}

struct KOptCpuReconnectPlan {
  std::array<std::uint8_t, 5U> pair_indices{};
};

std::vector<KOptCpuReconnectPlan>
BuildKOptCpuReconnectPlans(const std::uint32_t k,
                           const std::vector<EndpointMatching>& reconnect_templates) {
  std::vector<KOptCpuReconnectPlan> plans;
  plans.reserve(reconnect_templates.size());
  for (const EndpointMatching& reconnect_template : reconnect_templates) {
    if (reconnect_template.endpoint_count != 2U * k) {
      throw std::logic_error("CPU k-opt reconnect plan 的端点数非法");
    }
    KOptCpuReconnectPlan plan;
    std::uint32_t pair_count = 0U;
    for (std::uint32_t port = 0U; port < reconnect_template.endpoint_count; ++port) {
      const std::uint32_t partner = reconnect_template.mate[port];
      if (port >= partner) {
        continue;
      }
      if (partner >= reconnect_template.endpoint_count || pair_count >= k) {
        throw std::logic_error("CPU k-opt reconnect plan 包含非法匹配");
      }
      plan.pair_indices[pair_count++] = static_cast<std::uint8_t>(
          static_cast<std::size_t>(port) * kKOptCostPortCapacity + partner);
    }
    if (pair_count != k) {
      throw std::logic_error("CPU k-opt reconnect plan 的匹配数非法");
    }
    plans.push_back(plan);
  }
  return plans;
}

const std::vector<KOptCpuReconnectPlan>&
CachedKOptCpuReconnectPlans(const std::uint32_t k,
                            const std::vector<EndpointMatching>& reconnect_templates) {
  const auto build = [&] { return BuildKOptCpuReconnectPlans(k, reconnect_templates); };
  switch (k) {
  case 3U: {
    static const std::vector<KOptCpuReconnectPlan> plans = build();
    return plans;
  }
  case 4U: {
    static const std::vector<KOptCpuReconnectPlan> plans = build();
    return plans;
  }
  case 5U: {
    static const std::vector<KOptCpuReconnectPlan> plans = build();
    return plans;
  }
  default:
    throw std::invalid_argument("CPU k-opt reconnect plan 的 k 非法");
  }
}

class KOptBatchDistanceCache {
public:
  KOptBatchDistanceCache(const GraphSnapshot& graph, std::vector<std::int32_t> nodes,
                         std::vector<std::int32_t> node_to_local)
      : node_count_(nodes.size()), node_to_local_(std::move(node_to_local)),
        distances_(node_count_ * node_count_, 0) {
    for (std::size_t first = 0U; first < node_count_; ++first) {
      for (std::size_t second = first + 1U; second < node_count_; ++second) {
        const std::int64_t distance = graph.Distance(nodes[first], nodes[second]);
        distances_[first * node_count_ + second] = distance;
        distances_[second * node_count_ + first] = distance;
      }
    }
  }

  [[nodiscard]] std::size_t node_count() const { return node_count_; }

  [[nodiscard]] std::int32_t LocalIndex(const std::int32_t node) const {
    if (node < 0 || static_cast<std::size_t>(node) >= node_to_local_.size()) {
      return -1;
    }
    return node_to_local_[static_cast<std::size_t>(node)];
  }

  [[nodiscard]] std::int64_t Distance(const std::int32_t first, const std::int32_t second) const {
    if (first < 0 || second < 0 || static_cast<std::size_t>(first) >= node_count_ ||
        static_cast<std::size_t>(second) >= node_count_) {
      throw std::logic_error("CPU k-opt batch 距离缓存索引越界");
    }
    return distances_[static_cast<std::size_t>(first) * node_count_ +
                      static_cast<std::size_t>(second)];
  }

private:
  std::size_t node_count_{};
  std::vector<std::int32_t> node_to_local_;
  std::vector<std::int64_t> distances_;
};

std::optional<KOptBatchDistanceCache> BuildKOptBatchDistanceCache(
    const GraphSnapshot& graph, const std::uint32_t k, const std::vector<KOptCostTask>& tasks,
    const std::vector<KOptCpuReconnectPlan>& reconnect_plans, const std::size_t cell_count) {
  if (cell_count < kKOptCpuParallelMinCells || tasks.empty()) {
    return std::nullopt;
  }
  if (static_cast<std::size_t>(graph.dimension) > kKOptCpuDistanceCacheMaxGraphNodes) {
    return std::nullopt;
  }
  const std::size_t active_port_count = std::size_t{2} * k;
  if (tasks.size() > std::numeric_limits<std::size_t>::max() / active_port_count) {
    return std::nullopt;
  }
  std::vector<std::int32_t> nodes;
  nodes.reserve(std::min(tasks.size() * active_port_count, kKOptCpuDistanceCacheMaxNodes));
  std::vector<std::int32_t> node_to_local(static_cast<std::size_t>(graph.dimension), -1);
  for (const KOptCostTask& task : tasks) {
    for (std::size_t port = 0U; port < active_port_count; ++port) {
      const std::int32_t node = task.port_nodes[port];
      std::int32_t& local = node_to_local[static_cast<std::size_t>(node)];
      if (local >= 0) {
        continue;
      }
      if (nodes.size() >= kKOptCpuDistanceCacheMaxNodes) {
        return std::nullopt;
      }
      local = static_cast<std::int32_t>(nodes.size());
      nodes.push_back(node);
    }
  }
  if (nodes.empty()) {
    return std::nullopt;
  }

  std::array<bool, kKOptCostPortCapacity * kKOptCostPortCapacity> used_pairs{};
  std::size_t used_pair_count = 0U;
  for (const KOptCpuReconnectPlan& plan : reconnect_plans) {
    for (std::uint32_t pair = 0U; pair < k; ++pair) {
      const std::uint8_t flat_index = plan.pair_indices[pair];
      if (!used_pairs[flat_index]) {
        used_pairs[flat_index] = true;
        ++used_pair_count;
      }
    }
  }
  const std::size_t full_pair_count = nodes.size() * (nodes.size() - 1U) / 2U;
  if (used_pair_count == 0U ||
      tasks.size() > std::numeric_limits<std::size_t>::max() / used_pair_count) {
    return std::nullopt;
  }
  // 只有理论距离调用至少可减少一半时才建表；否则保留 task-local 惰性路径。
  const std::size_t task_pair_upper_bound = tasks.size() * used_pair_count;
  if (full_pair_count > task_pair_upper_bound / 2U) {
    return std::nullopt;
  }
  return KOptBatchDistanceCache(graph, std::move(nodes), std::move(node_to_local));
}

class KOptCostTaskCpuScorer {
public:
  KOptCostTaskCpuScorer(const GraphSnapshot& graph, const std::uint32_t k, const KOptCostTask& task,
                        const KOptBatchDistanceCache* const distance_cache)
      : graph_(&graph), k_(k), task_(&task) {
    for (std::uint32_t edge = 0U; edge < k_; ++edge) {
      const std::size_t first_port = std::size_t{2} * edge;
      deleted_edges_[edge] =
          CanonicalEdge(task_->port_nodes[first_port], task_->port_nodes[first_port + 1U]);
    }
    if (distance_cache != nullptr) {
      distance_cache_ = distance_cache;
      for (std::size_t port = 0U; port < std::size_t{2} * k_; ++port) {
        local_ports_[port] = distance_cache_->LocalIndex(task_->port_nodes[port]);
        if (local_ports_[port] < 0) {
          throw std::logic_error("CPU k-opt task 端口不在 batch 距离缓存中");
        }
      }
    }
  }

  [[nodiscard]] std::int64_t Score(const KOptCpuReconnectPlan& plan) {
    std::array<NodeEdge, 5U> added_edges{};
    std::uint32_t added_count = 0U;
    std::int64_t total = 0;
    for (std::uint32_t pair = 0U; pair < k_; ++pair) {
      const PairCost& pair_cost = ResolvePairCost(plan.pair_indices[pair]);
      if (pair_cost.distance == kInvalidKOptTemplateCost ||
          std::find(added_edges.begin(), added_edges.begin() + added_count, pair_cost.edge) !=
              added_edges.begin() + added_count) {
        return kInvalidKOptTemplateCost;
      }
      if (pair_cost.distance < 0 ||
          total > std::numeric_limits<std::int64_t>::max() - pair_cost.distance) {
        return kInvalidKOptTemplateCost;
      }
      added_edges[added_count++] = pair_cost.edge;
      total += pair_cost.distance;
    }
    return added_count == k_ ? total : kInvalidKOptTemplateCost;
  }

private:
  struct PairCost {
    NodeEdge edge;
    std::int64_t distance{-1};
  };

  [[nodiscard]] const PairCost& ResolvePairCost(const std::uint8_t flat_index) {
    PairCost& pair_cost = pair_costs_[flat_index];
    if (pair_cost.distance != -1) {
      return pair_cost;
    }
    const std::size_t port_count = task_->port_nodes.size();
    const std::size_t first_port = flat_index / port_count;
    const std::size_t second_port = flat_index % port_count;
    pair_cost.edge = CanonicalEdge(task_->port_nodes[first_port], task_->port_nodes[second_port]);
    if (pair_cost.edge.u == pair_cost.edge.v ||
        std::find(deleted_edges_.begin(), deleted_edges_.begin() + k_, pair_cost.edge) !=
            deleted_edges_.begin() + k_) {
      pair_cost.distance = kInvalidKOptTemplateCost;
      return pair_cost;
    }
    pair_cost.distance =
        distance_cache_ == nullptr
            ? graph_->Distance(pair_cost.edge.u, pair_cost.edge.v)
            : distance_cache_->Distance(local_ports_[first_port], local_ports_[second_port]);
    if (pair_cost.distance < 0) {
      pair_cost.distance = kInvalidKOptTemplateCost;
    }
    return pair_cost;
  }

  const GraphSnapshot* graph_{};
  const KOptBatchDistanceCache* distance_cache_{};
  std::uint32_t k_{};
  const KOptCostTask* task_{};
  std::array<NodeEdge, 5U> deleted_edges_{};
  std::array<std::int32_t, kKOptCostPortCapacity> local_ports_{};
  std::array<PairCost, kKOptCostPortCapacity * kKOptCostPortCapacity> pair_costs_{};
};

void EvaluateKOptTemplateCostsCpuInto(
    const GraphSnapshot& graph, const std::uint32_t k, const std::vector<KOptCostTask>& tasks,
    const std::vector<KOptCpuReconnectPlan>& reconnect_plans, std::uint32_t* const threads_used,
    std::uint32_t* const distance_cache_nodes, const std::span<std::int64_t> costs) {
  const std::size_t template_count = reconnect_plans.size();
  const std::size_t cell_count = tasks.size() * template_count;
  if (costs.size() != cell_count) {
    throw std::logic_error("CPU k-opt cost 输出矩阵规模错误");
  }
  const std::optional<KOptBatchDistanceCache> distance_cache =
      BuildKOptBatchDistanceCache(graph, k, tasks, reconnect_plans, cell_count);
  *distance_cache_nodes =
      distance_cache.has_value() ? static_cast<std::uint32_t>(distance_cache->node_count()) : 0U;
  *threads_used = 1U;
  const auto score_task = [&](const std::size_t task_index) {
    KOptCostTaskCpuScorer scorer(graph, k, tasks[task_index],
                                 distance_cache.has_value() ? &*distance_cache : nullptr);
    const std::size_t row_begin = task_index * template_count;
    for (std::size_t template_index = 0U; template_index < template_count; ++template_index) {
      // 每个 task 独占连续 row；静态调度只改变计算时序，不改变规范矩阵布局。
      costs[row_begin + template_index] = scorer.Score(reconnect_plans[template_index]);
    }
  };
#ifdef CUDAEE_HAS_OPENMP
  const int thread_limit =
      std::min({kKOptCpuMaxThreads, omp_get_max_threads(), omp_get_num_procs()});
  if (thread_limit > 1 && cell_count >= kKOptCpuParallelMinCells) {
#pragma omp parallel num_threads(thread_limit)
    {
#pragma omp single
      {
        *threads_used = static_cast<std::uint32_t>(omp_get_num_threads());
      }
#pragma omp for schedule(static)
      for (std::size_t task_index = 0U; task_index < tasks.size(); ++task_index) {
        score_task(task_index);
      }
    }
    return;
  }
#endif
  for (std::size_t task_index = 0U; task_index < tasks.size(); ++task_index) {
    score_task(task_index);
  }
}

std::vector<std::int64_t> EvaluateKOptTemplateCostsCpu(
    const GraphSnapshot& graph, const std::uint32_t k, const std::vector<KOptCostTask>& tasks,
    const std::vector<KOptCpuReconnectPlan>& reconnect_plans, std::uint32_t* const threads_used,
    std::uint32_t* const distance_cache_nodes) {
  std::vector<std::int64_t> costs(tasks.size() * reconnect_plans.size());
  EvaluateKOptTemplateCostsCpuInto(graph, k, tasks, reconnect_plans, threads_used,
                                   distance_cache_nodes, costs);
  return costs;
}

void RecordKOptCpuParallelism(PathSystemKOptBatchResult* const result,
                              const KOptCostBatchResult& costs,
                              const std::size_t cost_cell_count) {
  result->peak_cpu_cost_threads = std::max(result->peak_cpu_cost_threads, costs.cpu_threads_used);
  if (costs.cpu_threads_used <= 1U) {
    return;
  }
  if (!AddWithoutOverflow(&result->cpu_parallel_cost_batches, 1U) ||
      !AddWithoutOverflow(&result->cpu_parallel_cost_cells, cost_cell_count)) {
    throw std::overflow_error("path-system k-opt CPU 并行统计溢出");
  }
}

void ExpectToken(std::istream* const input, const std::string_view expected) {
  std::string token;
  if (!(*input >> token) || token != expected) {
    throw std::runtime_error("path k-opt proof 缺少字段: " + std::string(expected));
  }
}

std::uint64_t ReadHexHash(std::istream* const input, const std::string_view field) {
  std::string value;
  if (!(*input >> value) || value.size() != 16 ||
      !std::all_of(value.begin(), value.end(),
                   [](const unsigned char character) { return std::isxdigit(character) != 0; })) {
    throw std::runtime_error("path k-opt proof 的 " + std::string(field) + " 非法");
  }
  std::size_t consumed = 0;
  std::uint64_t parsed = 0;
  try {
    parsed = std::stoull(value, &consumed, 16);
  } catch (const std::exception&) {
    throw std::runtime_error("path k-opt proof 的 " + std::string(field) + " 非十六进制");
  }
  if (consumed != value.size()) {
    throw std::runtime_error("path k-opt proof 的 " + std::string(field) + " 含多余字符");
  }
  return parsed;
}

} // namespace

KOptReconnectTable BuildKOptReconnectTable(const std::uint32_t k) {
  if (k < 3 || k > 5) {
    throw std::invalid_argument("reconnect template 的 k 必须位于 [3,5]");
  }
  EndpointMatching components = EmptyMatching(k);
  for (std::uint32_t edge = 0; edge + 1 < k; ++edge) {
    if (!PairEndpoints(&components, 2U * edge + 1U, 2U * edge + 2U)) {
      throw std::logic_error("无法生成 k-opt 分量匹配");
    }
  }
  if (!PairEndpoints(&components, 2U * k - 1U, 0)) {
    throw std::logic_error("无法闭合 k-opt 分量匹配");
  }

  KOptReconnectTable table;
  table.k = k;
  for (const EndpointMatching& candidate : EnumerateInsideMatchings(k)) {
    if (!IsAlternatingHamiltonianCycle(components, candidate, k)) {
      continue;
    }
    bool reuses_deleted_edge = false;
    for (std::uint32_t edge = 0; edge < k; ++edge) {
      const std::size_t first_endpoint = std::size_t{2} * edge;
      if (candidate.mate[first_endpoint] == first_endpoint + 1) {
        reuses_deleted_edge = true;
        break;
      }
    }
    if (!reuses_deleted_edge) {
      table.templates.push_back(candidate);
    }
  }
  if (table.templates.size() != kKOptTemplateCounts[static_cast<std::size_t>(k - 3U)]) {
    throw std::logic_error("proper k-opt reconnect template 数量错误");
  }

  std::uint64_t hash = 14695981039346656037ULL;
  constexpr std::string_view kDomain = "CUDAEE_KOPT_RECONNECT_TEMPLATES_V1";
  for (const char value : kDomain) {
    HashByte(&hash, static_cast<std::uint8_t>(value));
  }
  HashUint32(&hash, table.k);
  HashUint32(&hash, static_cast<std::uint32_t>(table.templates.size()));
  for (const EndpointMatching& matching : table.templates) {
    for (std::uint32_t endpoint = 0; endpoint < matching.endpoint_count; ++endpoint) {
      HashByte(&hash, matching.mate[endpoint]);
    }
  }
  table.generator_hash = hash;
  return table;
}

namespace {

const KOptReconnectTable& CachedKOptReconnectTable(const std::uint32_t k) {
  // proper templates 只由 k 决定；函数内静态对象由 C++ 保证线程安全地延迟初始化。
  switch (k) {
  case 3U: {
    static const KOptReconnectTable table = BuildKOptReconnectTable(3U);
    return table;
  }
  case 4U: {
    static const KOptReconnectTable table = BuildKOptReconnectTable(4U);
    return table;
  }
  case 5U: {
    static const KOptReconnectTable table = BuildKOptReconnectTable(5U);
    return table;
  }
  default:
    throw std::invalid_argument("reconnect template 缓存的 k 必须位于 [3,5]");
  }
}

struct PathMatchingCatalog {
  std::vector<EndpointMatching> outside;
  std::vector<EndpointMatching> inside;
  std::optional<PathCompatibilityTable> table;
};

PathMatchingCatalog BuildPathMatchingCatalog(const std::uint32_t path_count) {
  PathMatchingCatalog catalog;
  catalog.outside = EnumerateOutsideMatchings(path_count);
  catalog.inside = EnumerateInsideMatchings(path_count);
  if (path_count <= kMaxGpuPathCount) {
    catalog.table = BuildPathCompatibilityTable(path_count);
  }
  return catalog;
}

const PathMatchingCatalog& CachedPathMatchingCatalog(const std::uint32_t path_count) {
  // outside/inside 规范顺序和兼容表都只由 path_count 决定；调用方只能持有 const 引用。
  switch (path_count) {
  case 1U: {
    static const PathMatchingCatalog catalog = BuildPathMatchingCatalog(1U);
    return catalog;
  }
  case 2U: {
    static const PathMatchingCatalog catalog = BuildPathMatchingCatalog(2U);
    return catalog;
  }
  case 3U: {
    static const PathMatchingCatalog catalog = BuildPathMatchingCatalog(3U);
    return catalog;
  }
  case 4U: {
    static const PathMatchingCatalog catalog = BuildPathMatchingCatalog(4U);
    return catalog;
  }
  case 5U: {
    static const PathMatchingCatalog catalog = BuildPathMatchingCatalog(5U);
    return catalog;
  }
  case 6U: {
    static const PathMatchingCatalog catalog = BuildPathMatchingCatalog(6U);
    return catalog;
  }
  case 7U: {
    static const PathMatchingCatalog catalog = BuildPathMatchingCatalog(7U);
    return catalog;
  }
  default:
    throw std::invalid_argument("path matching 缓存的路径数必须位于 [1,7]");
  }
}

} // namespace

namespace {

struct KOptCostEvaluationPlan {
  const KOptReconnectTable& table;
  const std::vector<KOptCpuReconnectPlan>& cpu_plans;
  std::size_t cell_count{};
};

KOptCostEvaluationPlan PrepareKOptCostEvaluation(
    const GraphSnapshot& graph, const std::uint32_t k,
    const std::vector<KOptCostTask>& tasks) {
  if (k < 3U || k > 5U) {
    throw std::invalid_argument("k-opt cost 的 k 必须位于 [3,5]");
  }
  if (!graph.integer_coordinates || !graph.integer_distance_safe || graph.dimension <= 0 ||
      graph.points.size() != static_cast<std::size_t>(graph.dimension)) {
    throw std::invalid_argument("k-opt cost 只支持平方距离安全的整数坐标图");
  }
  for (const KOptCostTask& task : tasks) {
    ValidateKOptCostTask(graph, k, task);
  }
  const KOptReconnectTable& table = CachedKOptReconnectTable(k);
  const std::vector<KOptCpuReconnectPlan>& cpu_plans =
      CachedKOptCpuReconnectPlans(k, table.templates);
  if (cpu_plans.size() != table.templates.size()) {
    throw std::logic_error("CPU k-opt reconnect plan 与规范模板数量不一致");
  }
  if (!tasks.empty() &&
      table.templates.size() > std::numeric_limits<std::size_t>::max() / tasks.size()) {
    throw std::overflow_error("k-opt cost 矩阵规模溢出");
  }
  return {table, cpu_plans, tasks.size() * table.templates.size()};
}

class KOptCpuCostWorkspace {
public:
  [[nodiscard]] std::span<std::int64_t> Prepare(const std::size_t cell_count) {
    if (cell_count == 0U) {
      return {};
    }
    if (cell_count > capacity_) {
      // scorer 会在任何读取前覆盖全部 cells，无需先把大矩阵清零。
      storage_ = std::make_unique_for_overwrite<std::int64_t[]>(cell_count);
      capacity_ = cell_count;
    }
    return {storage_.get(), cell_count};
  }

private:
  std::unique_ptr<std::int64_t[]> storage_;
  std::size_t capacity_{};
};

KOptCostBatchResult EvaluateKOptTemplateCostsCpuWithWorkspace(
    const GraphSnapshot& graph, const std::uint32_t k, const std::vector<KOptCostTask>& tasks,
    KOptCpuCostWorkspace* const workspace, std::span<const std::int64_t>* const cost_values) {
  if (workspace == nullptr || cost_values == nullptr) {
    throw std::invalid_argument("CPU k-opt cost workspace 或输出 view 为空");
  }
  const KOptCostEvaluationPlan plan = PrepareKOptCostEvaluation(graph, k, tasks);
  KOptCostBatchResult result;
  result.k = k;
  result.template_count = static_cast<std::uint32_t>(plan.table.templates.size());
  const SteadyClock::time_point cpu_begin = SteadyClock::now();
  const std::span<std::int64_t> output = workspace->Prepare(plan.cell_count);
#ifndef NDEBUG
  // Debug 门禁确保无初始化 workspace 的每个逻辑 cell 都在返回前被 scorer 覆盖。
  std::fill(output.begin(), output.end(), std::int64_t{-1});
#endif
  EvaluateKOptTemplateCostsCpuInto(graph, k, tasks, plan.cpu_plans, &result.cpu_threads_used,
                                   &result.cpu_distance_cache_nodes, output);
#ifndef NDEBUG
  if (std::find(output.begin(), output.end(), std::int64_t{-1}) != output.end()) {
    throw std::logic_error("CPU k-opt cost workspace 存在未覆盖 cell");
  }
#endif
  result.cpu_certify_ms = ElapsedMilliseconds(cpu_begin);
  result.backend = "cpu";
  result.cpu_verified = true;
  *cost_values = output;
  return result;
}

} // namespace

KOptCostBatchResult EvaluateKOptTemplateCosts(const GraphSnapshot& graph, const std::uint32_t k,
                                              const std::vector<KOptCostTask>& tasks,
                                              const PathCompatibilityBackend backend) {
  const KOptCostEvaluationPlan plan = PrepareKOptCostEvaluation(graph, k, tasks);
  const KOptReconnectTable& table = plan.table;
  const std::vector<KOptCpuReconnectPlan>& cpu_plans = plan.cpu_plans;

  bool use_cuda = backend == PathCompatibilityBackend::kCuda;
  if (backend == PathCompatibilityBackend::kAuto) {
    std::string reason;
    use_cuda = detail::KOptCostCudaAvailable(&reason);
  } else if (backend != PathCompatibilityBackend::kCpu &&
             backend != PathCompatibilityBackend::kCuda) {
    throw std::invalid_argument("未知 k-opt cost 后端");
  }

  KOptCostBatchResult result;
  result.k = k;
  result.template_count = static_cast<std::uint32_t>(table.templates.size());
  if (use_cuda) {
    std::string reason;
    if (!detail::KOptCostCudaAvailable(&reason)) {
      throw std::runtime_error("CUDA k-opt cost 后端不可用: " + reason);
    }
    result.added_costs = detail::EvaluateKOptTemplateCostsCuda(
        graph, table, tasks, &result.selected_device, &result.cuda_cache);
    const SteadyClock::time_point cpu_begin = SteadyClock::now();
    const std::vector<std::int64_t> cpu_costs = EvaluateKOptTemplateCostsCpu(
        graph, k, tasks, cpu_plans, &result.cpu_threads_used, &result.cpu_distance_cache_nodes);
    const auto mismatch = std::mismatch(result.added_costs.begin(), result.added_costs.end(),
                                        cpu_costs.begin(), cpu_costs.end());
    result.cpu_certify_ms = ElapsedMilliseconds(cpu_begin);
    if (mismatch.first != result.added_costs.end() || mismatch.second != cpu_costs.end()) {
      const std::size_t cell =
          static_cast<std::size_t>(mismatch.first - result.added_costs.begin());
      throw std::runtime_error("CUDA k-opt cost 未通过 CPU 精确矩阵认证，cell=" +
                               std::to_string(cell));
    }
    result.backend = "cuda";
    result.cpu_verified = true;
  } else {
    const SteadyClock::time_point cpu_begin = SteadyClock::now();
    result.added_costs = EvaluateKOptTemplateCostsCpu(
        graph, k, tasks, cpu_plans, &result.cpu_threads_used, &result.cpu_distance_cache_nodes);
    result.cpu_certify_ms = ElapsedMilliseconds(cpu_begin);
    result.backend = "cpu";
    result.cpu_verified = true;
  }
  if (result.added_costs.size() != plan.cell_count) {
    throw std::logic_error("k-opt cost 后端返回矩阵规模错误");
  }
  return result;
}

std::uint64_t ComputePathSystemHash(const NormalizedPathSystem& paths) {
  std::uint64_t hash = 14695981039346656037ULL;
  constexpr std::string_view kDomain = "CUDAEE_NORMALIZED_PATH_SYSTEM_V1";
  for (const char value : kDomain) {
    HashByte(&hash, static_cast<std::uint8_t>(value));
  }
  HashUint32(&hash, static_cast<std::uint32_t>(paths.paths.size()));
  HashUint64(&hash, static_cast<std::uint64_t>(paths.edge_count));
  for (const Path& path : paths.paths) {
    HashUint32(&hash, static_cast<std::uint32_t>(path.size()));
    for (const std::int32_t node : path) {
      HashUint32(&hash, static_cast<std::uint32_t>(node));
    }
  }
  return hash;
}

namespace {

KOptSearchResult FindKOptWitnessImpl(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                                     const EndpointMatching& outside,
                                     const std::optional<NodeEdge>& required_edge,
                                     const KOptSearchOptions& options) {
  KOptSearchResult result;
  if (options.max_k < 3 || options.max_k > 5 || options.cost_batch_size == 0) {
    result.reason = "max_k 必须位于 [3,5] 且 cost_batch_size 必须为正数";
    return result;
  }
  if (options.cost_backend != PathCompatibilityBackend::kAuto &&
      options.cost_backend != PathCompatibilityBackend::kCpu &&
      options.cost_backend != PathCompatibilityBackend::kCuda) {
    result.reason = "未知 k-opt cost 后端";
    return result;
  }
  TourContext context;
  if (!BuildTourContext(graph, paths, outside, required_edge, &context, &result.reason)) {
    return result;
  }

  const auto consume_attempt = [&](ReconnectAttempt attempt) {
    if (!AddWithoutOverflow(&result.reconnect_matchings_tested, attempt.matchings_tested)) {
      result.status = KOptSearchStatus::kUnresolved;
      result.reason = "k-opt 重连计数溢出";
      return true;
    }
    if (attempt.fatal) {
      result.status = KOptSearchStatus::kInvalid;
      result.reason = attempt.reason;
      return true;
    }
    if (!attempt.witness.has_value()) {
      return false;
    }
    result.status = KOptSearchStatus::kImproved;
    result.reason = "找到严格改善的 k-opt witness";
    result.witness = std::move(*attempt.witness);
    std::string verify_reason;
    if (!VerifyKOptWitness(graph, paths, outside, required_edge, result.witness, &verify_reason)) {
      result.status = KOptSearchStatus::kInvalid;
      result.reason = "内部 witness 复核失败: " + verify_reason;
    }
    return true;
  };

  const std::size_t selectable_count = context.selectable_positions.size();
  for (std::uint32_t k = 3; k <= options.max_k && k < selectable_count; ++k) {
    const KOptReconnectTable& reconnect_table = CachedKOptReconnectTable(k);
    std::vector<std::size_t> combination(k);
    for (std::size_t index = 0; index < combination.size(); ++index) {
      combination[index] = index;
    }
    if (options.cost_backend == PathCompatibilityBackend::kCpu) {
      do {
        if (options.max_deletion_sets != 0 &&
            result.deletion_sets_tested >= options.max_deletion_sets) {
          result.status = KOptSearchStatus::kUnresolved;
          result.reason = "k-opt 删除集合预算耗尽";
          return result;
        }
        ++result.deletion_sets_tested;
        std::vector<std::size_t> deleted_positions;
        deleted_positions.reserve(k);
        for (const std::size_t selected : combination) {
          deleted_positions.push_back(context.selectable_positions[selected]);
        }
        // 端口编号必须按原巡回次序，才能直接使用固定的 proper reconnect templates。
        std::sort(deleted_positions.begin(), deleted_positions.end());
        if (consume_attempt(TryReconnect(graph, context, outside, deleted_positions,
                                         reconnect_table.templates))) {
          return result;
        }
      } while (AdvanceCombination(&combination, selectable_count));
      continue;
    }

    struct CostWork {
      std::vector<std::size_t> deleted_positions;
      KOptCostTask task;
    };
    bool has_combination = true;
    while (has_combination) {
      std::vector<CostWork> works;
      works.reserve(options.cost_batch_size);
      bool budget_blocked = false;
      while (has_combination && works.size() < options.cost_batch_size) {
        if (options.max_deletion_sets != 0 &&
            (result.deletion_sets_tested >= options.max_deletion_sets ||
             works.size() >= options.max_deletion_sets - result.deletion_sets_tested)) {
          budget_blocked = true;
          break;
        }
        CostWork work;
        work.deleted_positions.reserve(k);
        for (const std::size_t selected : combination) {
          work.deleted_positions.push_back(context.selectable_positions[selected]);
        }
        std::sort(work.deleted_positions.begin(), work.deleted_positions.end());
        work.task = BuildKOptCostTask(context, work.deleted_positions);
        works.push_back(std::move(work));
        has_combination = AdvanceCombination(&combination, selectable_count);
      }

      if (!works.empty()) {
        std::vector<KOptCostTask> tasks;
        tasks.reserve(works.size());
        for (const CostWork& work : works) {
          tasks.push_back(work.task);
        }
        const PathCompatibilityBackend selected_cost_backend =
            SelectKOptCostBackend(options, KOptCostCellCount(k, tasks.size()));
        KOptCostBatchResult costs;
        try {
          costs = EvaluateKOptTemplateCosts(graph, k, tasks, selected_cost_backend);
        } catch (const std::exception& error) {
          if (selected_cost_backend != PathCompatibilityBackend::kAuto) {
            result.status = KOptSearchStatus::kUnresolved;
            result.reason = std::string("k-opt cost 失败: ") + error.what();
            return result;
          }
          costs = EvaluateKOptTemplateCosts(graph, k, tasks, PathCompatibilityBackend::kCpu);
        }
        if (costs.added_costs.size() != tasks.size() * reconnect_table.templates.size()) {
          result.status = KOptSearchStatus::kUnresolved;
          result.reason = "k-opt cost 缓存矩阵规模错误";
          return result;
        }

        for (std::size_t work_index = 0; work_index < works.size(); ++work_index) {
          // block 尾部可能因更早的 witness 不会被测试；proof 只记录实际消费的删除集合。
          if (!AddWithoutOverflow(&result.deletion_sets_tested, 1U)) {
            result.status = KOptSearchStatus::kUnresolved;
            result.reason = "k-opt 删除集合计数溢出";
            return result;
          }
          const std::size_t row_offset = work_index * reconnect_table.templates.size();
          const std::span<const std::int64_t> row(costs.added_costs.data() + row_offset,
                                                  reconnect_table.templates.size());
          if (consume_attempt(TryReconnectFromCostRow(
                  graph, context, outside, works[work_index].deleted_positions,
                  works[work_index].task, reconnect_table.templates, row, costs.backend,
                  costs.cpu_verified))) {
            return result;
          }
        }
      }
      if (budget_blocked) {
        result.status = KOptSearchStatus::kUnresolved;
        result.reason = "k-opt 删除集合预算耗尽";
        return result;
      }
    }
  }
  result.status = KOptSearchStatus::kNoImprovement;
  result.reason = "已穷举允许的 k-opt 重连但未找到严格改善";
  return result;
}

} // namespace

KOptSearchResult FindKOptWitness(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                                 const EndpointMatching& outside,
                                 const std::optional<NodeEdge>& required_edge,
                                 const KOptSearchOptions& options) {
  return FindKOptWitnessImpl(graph, paths, outside, required_edge, options);
}

KOptSearchResult FindExactTourWitness(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                                      const EndpointMatching& outside,
                                      const std::optional<NodeEdge>& required_edge,
                                      const std::uint32_t max_blocks) {
  KOptSearchResult result;
  if (max_blocks == 0 || max_blocks > 18) {
    result.status = KOptSearchStatus::kUnresolved;
    result.reason = "exact fallback 的 max_blocks 必须位于 [1,18]";
    return result;
  }
  TourContext context;
  if (!BuildTourContext(graph, paths, outside, required_edge, &context, &result.reason)) {
    return result;
  }
  const std::vector<ExactTourBlock> blocks = BuildExactTourBlocks(context, outside);
  const std::size_t block_count = blocks.size();
  if (block_count < 2) {
    result.reason = "收缩 outside matching 后不足两个 block";
    return result;
  }
  if (block_count > max_blocks) {
    result.status = KOptSearchStatus::kUnresolved;
    result.reason = "exact fallback 超过 block 上限";
    return result;
  }

  const std::size_t state_count = 2U * block_count;
  const std::size_t mask_count = std::size_t{1} << block_count;
  if (state_count > std::numeric_limits<std::size_t>::max() / mask_count) {
    result.status = KOptSearchStatus::kUnresolved;
    result.reason = "exact fallback DP 规模溢出";
    return result;
  }
  const std::size_t cell_count = state_count * mask_count;
  const std::int64_t infinity = std::numeric_limits<std::int64_t>::max();
  std::vector<std::int64_t> cost;
  std::vector<std::int16_t> predecessor;
  try {
    cost.assign(cell_count, infinity);
    predecessor.assign(cell_count, -1);
  } catch (const std::bad_alloc&) {
    result.status = KOptSearchStatus::kUnresolved;
    result.reason = "exact fallback DP 内存不足";
    return result;
  }
  const auto cell = [state_count](const std::size_t mask, const std::size_t state) {
    return mask * state_count + state;
  };

  // 固定第 0 个 forced-edge block 的方向；反转整个无向巡回可覆盖另一方向。
  constexpr std::size_t kStartState = 0;
  constexpr std::size_t kStartMask = 1;
  cost[cell(kStartMask, kStartState)] = 0;
  const std::optional<NodeEdge> forbidden =
      required_edge.has_value()
          ? std::optional<NodeEdge>(CanonicalEdge(required_edge->u, required_edge->v))
          : std::nullopt;

  for (std::size_t mask = 1; mask < mask_count; ++mask) {
    if ((mask & kStartMask) == 0) {
      continue;
    }
    for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
      if ((mask & (std::size_t{1} << block_index)) == 0) {
        continue;
      }
      for (std::uint32_t orientation = 0; orientation < BlockOrientationCount(blocks[block_index]);
           ++orientation) {
        const std::size_t state = 2U * block_index + orientation;
        const std::int64_t current_cost = cost[cell(mask, state)];
        if (current_cost == infinity) {
          continue;
        }
        ++result.exact_states_tested;
        const std::int32_t from = BlockExit(blocks[block_index], orientation);
        for (std::size_t next_block = 0; next_block < block_count; ++next_block) {
          const std::size_t next_bit = std::size_t{1} << next_block;
          if ((mask & next_bit) != 0) {
            continue;
          }
          for (std::uint32_t next_orientation = 0;
               next_orientation < BlockOrientationCount(blocks[next_block]); ++next_orientation) {
            const std::int32_t to = BlockEntry(blocks[next_block], next_orientation);
            if (forbidden.has_value() && CanonicalEdge(from, to) == *forbidden) {
              continue;
            }
            const std::int64_t transition = graph.Distance(from, to);
            if (transition < 0 || current_cost > infinity - transition) {
              continue;
            }
            const std::size_t next_mask = mask | next_bit;
            const std::size_t next_state = 2U * next_block + next_orientation;
            const std::int64_t candidate = current_cost + transition;
            const std::size_t next_cell = cell(next_mask, next_state);
            if (candidate < cost[next_cell]) {
              cost[next_cell] = candidate;
              predecessor[next_cell] = static_cast<std::int16_t>(state);
            }
          }
        }
      }
    }
  }

  const std::size_t full_mask = mask_count - 1;
  std::int64_t best_cost = infinity;
  std::size_t best_state = state_count;
  const std::int32_t start_entry = BlockEntry(blocks.front(), 0);
  for (std::size_t block_index = 1; block_index < block_count; ++block_index) {
    for (std::uint32_t orientation = 0; orientation < BlockOrientationCount(blocks[block_index]);
         ++orientation) {
      const std::size_t state = 2U * block_index + orientation;
      const std::int64_t path_cost = cost[cell(full_mask, state)];
      if (path_cost == infinity) {
        continue;
      }
      const std::int32_t from = BlockExit(blocks[block_index], orientation);
      if (forbidden.has_value() && CanonicalEdge(from, start_entry) == *forbidden) {
        continue;
      }
      const std::int64_t closing = graph.Distance(from, start_entry);
      if (closing < 0 || path_cost > infinity - closing) {
        continue;
      }
      const std::int64_t candidate = path_cost + closing;
      if (candidate < best_cost) {
        best_cost = candidate;
        best_state = state;
      }
    }
  }
  if (best_state == state_count) {
    result.status = KOptSearchStatus::kNoImprovement;
    result.reason = "不存在满足 required-edge 约束的局部巡回";
    return result;
  }

  std::vector<NodeEdge> original_path_edges(context.path_edges.begin(), context.path_edges.end());
  std::int64_t original_path_cost = 0;
  std::string cost_reason;
  if (!SumEdgeCosts(graph, original_path_edges, &original_path_cost, &cost_reason)) {
    result.reason = cost_reason;
    return result;
  }
  if (best_cost >= original_path_cost) {
    result.status = KOptSearchStatus::kNoImprovement;
    result.reason = "精确 DP 证明不存在严格更短的受约束局部巡回";
    return result;
  }

  std::vector<std::size_t> state_order;
  std::size_t mask = full_mask;
  std::size_t state = best_state;
  while (state != kStartState) {
    state_order.push_back(state);
    const std::int16_t previous = predecessor[cell(mask, state)];
    if (previous < 0) {
      result.reason = "exact fallback predecessor 链损坏";
      return result;
    }
    mask ^= std::size_t{1} << (state / 2U);
    state = static_cast<std::size_t>(previous);
  }
  state_order.push_back(kStartState);
  std::reverse(state_order.begin(), state_order.end());
  if (state_order.size() != block_count) {
    result.reason = "exact fallback 未重建全部 block";
    return result;
  }

  EdgeSet improved_edges = context.outside_edges;
  for (std::size_t order_index = 0; order_index < state_order.size(); ++order_index) {
    const std::size_t current_state = state_order[order_index];
    const std::size_t next_state = state_order[(order_index + 1) % state_order.size()];
    const std::size_t current_block = current_state / 2U;
    const std::size_t next_block = next_state / 2U;
    const auto current_orientation = static_cast<std::uint32_t>(current_state % 2U);
    const auto next_orientation = static_cast<std::uint32_t>(next_state % 2U);
    const NodeEdge transition = CanonicalEdge(BlockExit(blocks[current_block], current_orientation),
                                              BlockEntry(blocks[next_block], next_orientation));
    if (transition.u == transition.v || !improved_edges.insert(transition).second) {
      result.reason = "exact fallback 重建了自环或重复边";
      return result;
    }
  }

  std::vector<std::int32_t> improved_tour;
  if (!BuildCycle(graph.dimension, context.nodes, improved_edges, &improved_tour, &result.reason)) {
    return result;
  }
  std::vector<NodeEdge> deleted_edges;
  std::vector<NodeEdge> added_edges;
  std::set_difference(context.all_edges.begin(), context.all_edges.end(), improved_edges.begin(),
                      improved_edges.end(), std::back_inserter(deleted_edges));
  std::set_difference(improved_edges.begin(), improved_edges.end(), context.all_edges.begin(),
                      context.all_edges.end(), std::back_inserter(added_edges));
  if (deleted_edges.size() < 2 || deleted_edges.size() != added_edges.size() ||
      deleted_edges.size() > std::numeric_limits<std::uint32_t>::max()) {
    result.reason = "exact fallback 的交换边集规模非法";
    return result;
  }

  KOptWitness witness;
  witness.k = static_cast<std::uint32_t>(deleted_edges.size());
  witness.deleted_edges = std::move(deleted_edges);
  witness.added_edges = std::move(added_edges);
  if (!SumEdgeCosts(graph, witness.deleted_edges, &witness.deleted_cost, &result.reason) ||
      !SumEdgeCosts(graph, witness.added_edges, &witness.added_cost, &result.reason) ||
      witness.added_cost >= witness.deleted_cost) {
    if (result.reason.empty()) {
      result.reason = "exact fallback 交换成本不构成严格改善";
    }
    return result;
  }
  if (!ExtractInsideMatching(context, outside, improved_tour, &witness.inside_matching,
                             &result.reason)) {
    return result;
  }
  std::string verify_reason;
  if (!VerifyKOptWitness(graph, paths, outside, required_edge, witness, &verify_reason)) {
    result.reason = "exact fallback witness 复核失败: " + verify_reason;
    return result;
  }
  result.status = KOptSearchStatus::kImproved;
  result.reason = "精确 DP 找到严格改善的局部巡回";
  result.witness = std::move(witness);
  return result;
}

bool VerifyKOptWitness(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                       const EndpointMatching& outside,
                       const std::optional<NodeEdge>& required_edge, const KOptWitness& witness,
                       std::string* const reason) {
  TourContext context;
  if (!BuildTourContext(graph, paths, outside, required_edge, &context, reason)) {
    return false;
  }
  if (witness.k < 2 || witness.k > context.path_edges.size() ||
      witness.deleted_edges.size() != witness.k || witness.added_edges.size() != witness.k) {
    SetReason(reason, "witness 的 k 或边数非法");
    return false;
  }

  EdgeSet deleted;
  for (const NodeEdge& edge : witness.deleted_edges) {
    if (edge.u >= edge.v || !context.path_edges.contains(edge) || !deleted.insert(edge).second) {
      SetReason(reason, "witness 删除了非路径边、非规范边或重复边");
      return false;
    }
  }
  if (required_edge.has_value() &&
      !deleted.contains(CanonicalEdge(required_edge->u, required_edge->v))) {
    SetReason(reason, "witness 未删除 required edge");
    return false;
  }

  EdgeSet remaining = context.all_edges;
  for (const NodeEdge& edge : deleted) {
    if (remaining.erase(edge) != 1) {
      SetReason(reason, "witness 删除边不在原巡回中");
      return false;
    }
  }
  EdgeSet added;
  for (const NodeEdge& edge : witness.added_edges) {
    if (edge.u < 0 || edge.v >= graph.dimension || edge.u >= edge.v || remaining.contains(edge) ||
        deleted.contains(edge) || !added.insert(edge).second) {
      SetReason(reason, "witness 添加了非法边、现存边或重复边");
      return false;
    }
  }

  std::vector<NodeEdge> deleted_vector(deleted.begin(), deleted.end());
  std::vector<NodeEdge> added_vector(added.begin(), added.end());
  std::int64_t deleted_cost = 0;
  std::int64_t added_cost = 0;
  if (!SumEdgeCosts(graph, deleted_vector, &deleted_cost, reason) ||
      !SumEdgeCosts(graph, added_vector, &added_cost, reason)) {
    return false;
  }
  if (deleted_cost != witness.deleted_cost || added_cost != witness.added_cost ||
      added_cost >= deleted_cost) {
    SetReason(reason, "witness 成本或严格改善不成立");
    return false;
  }

  remaining.insert(added.begin(), added.end());
  std::vector<std::int32_t> improved_tour;
  if (!BuildCycle(graph.dimension, context.nodes, remaining, &improved_tour, reason)) {
    return false;
  }
  EndpointMatching inside;
  if (!ExtractInsideMatching(context, outside, improved_tour, &inside, reason)) {
    return false;
  }
  if (inside != witness.inside_matching) {
    SetReason(reason, "witness inside matching 与改善巡回不一致");
    return false;
  }
  SetReason(reason, "OK");
  return true;
}

PathSystemKOptProof ProvePathSystemByKOpt(const GraphSnapshot& graph,
                                          const NormalizedPathSystem& paths,
                                          const std::optional<NodeEdge>& required_edge,
                                          const KOptSearchOptions& options) {
  PathSystemKOptProof proof;
  proof.snapshot_hash = graph.ContentHash();
  proof.path_system_hash = ComputePathSystemHash(paths);
  proof.path_count = static_cast<std::uint32_t>(paths.paths.size());
  if (proof.path_count == 0 || proof.path_count > kMaxTestablePathCount) {
    proof.reason = "路径数不在 [1,7]";
    return proof;
  }
  const PathMatchingCatalog& catalog = CachedPathMatchingCatalog(proof.path_count);
  const std::vector<EndpointMatching>& outside = catalog.outside;
  const std::vector<EndpointMatching>& inside = catalog.inside;
  proof.outside_count = static_cast<std::uint32_t>(outside.size());
  if (catalog.table.has_value()) {
    proof.compatibility_table_hash = catalog.table->generator_hash;
  }

  std::vector<bool> covered(outside.size(), false);
  for (std::uint32_t source = 0; source < outside.size(); ++source) {
    if (covered[source]) {
      continue;
    }
    KOptSearchResult search =
        FindKOptWitness(graph, paths, outside[source], required_edge, options);
    if (!AddWithoutOverflow(&proof.deletion_sets_tested, search.deletion_sets_tested) ||
        !AddWithoutOverflow(&proof.reconnect_matchings_tested, search.reconnect_matchings_tested)) {
      proof.reason = "path-system k-opt 统计计数溢出";
      return proof;
    }
    if (search.status != KOptSearchStatus::kImproved && options.exact_fallback_max_blocks != 0) {
      KOptSearchResult exact = FindExactTourWitness(graph, paths, outside[source], required_edge,
                                                    options.exact_fallback_max_blocks);
      if (!AddWithoutOverflow(&proof.exact_states_tested, exact.exact_states_tested)) {
        proof.reason = "path-system exact DP 状态计数溢出";
        return proof;
      }
      if (exact.status == KOptSearchStatus::kImproved) {
        search = std::move(exact);
      } else {
        search.reason += "; exact fallback: " + exact.reason;
        search.status = exact.status;
      }
    }
    if (search.status != KOptSearchStatus::kImproved) {
      proof.reason = "outside " + std::to_string(source) + " unresolved: " + search.reason;
      return proof;
    }
    std::string verify_reason;
    if (!VerifyKOptWitness(graph, paths, outside[source], required_edge, search.witness,
                           &verify_reason)) {
      proof.reason = "outside witness 复核失败: " + verify_reason;
      return proof;
    }
    const auto inside_iterator =
        std::find(inside.begin(), inside.end(), search.witness.inside_matching);
    if (inside_iterator == inside.end()) {
      proof.reason = "witness inside matching 不在规范枚举中";
      return proof;
    }
    const auto inside_index = static_cast<std::uint32_t>(inside_iterator - inside.begin());
    for (std::uint32_t outside_index = 0; outside_index < outside.size(); ++outside_index) {
      const bool compatible =
          catalog.table.has_value()
              ? catalog.table->Covers(outside_index, inside_index)
              : IsAlternatingHamiltonianCycle(outside[outside_index],
                                              search.witness.inside_matching, proof.path_count);
      covered[outside_index] = covered[outside_index] || compatible;
    }
    if (!covered[source]) {
      proof.reason = "witness 未覆盖其源 outside matching";
      return proof;
    }
    proof.records.push_back({source, std::move(search.witness)});
  }
  proof.proven =
      std::all_of(covered.begin(), covered.end(), [](const bool value) { return value; });
  proof.reason =
      proof.proven ? "全部 outside matching 均有严格改善 witness" : "存在未覆盖 outside matching";
  return proof;
}

namespace {

struct KOptCursorWork {
  std::array<std::size_t, 5U> deleted_positions{};
  KOptCostTask task;
};

struct KOptCursorBlock {
  std::uint32_t k{};
  std::vector<KOptCursorWork> works;
  bool budget_blocked{false};
};

struct KOptCursorConsumeStats {
  std::uint64_t cost_rows{};
  std::uint64_t candidate_templates_rechecked{};
  std::uint64_t cpu_completeness_rows{};
  std::uint64_t cpu_completeness_templates{};
  double candidate_recheck_ms{};
  double completeness_fallback_ms{};
};

// 只借用一次融合 cost matrix 的连续子区间；调用方必须保证矩阵覆盖整个消费过程。
struct KOptCostBlockView {
  std::uint32_t k{};
  std::uint32_t template_count{};
  std::span<const std::int64_t> added_costs;
  std::string_view backend;
  bool cpu_verified{false};
};

class KOptSearchCursor {
public:
  KOptSearchCursor(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                   const EndpointMatching& outside, const std::optional<NodeEdge>& required_edge,
                   const KOptSearchOptions& options,
                   const KOptPathValidationBinding& validation_binding)
      : graph_(&graph), paths_(&paths), outside_(outside), required_edge_(required_edge),
        options_(options) {
    if (options_.max_k < 3U || options_.max_k > 5U || options_.cost_batch_size == 0U) {
      result_.reason = "max_k 必须位于 [3,5] 且 cost_batch_size 必须为正数";
      finished_ = true;
      return;
    }
    if (options_.cost_backend != PathCompatibilityBackend::kAuto &&
        options_.cost_backend != PathCompatibilityBackend::kCpu &&
        options_.cost_backend != PathCompatibilityBackend::kCuda) {
      result_.reason = "k-opt cost cursor 收到未知后端";
      finished_ = true;
      return;
    }
    if (!BuildTourContext(graph, paths, outside_, required_edge_, &context_, &result_.reason,
                          &validation_binding)) {
      finished_ = true;
    }
  }

  [[nodiscard]] bool Finished() const { return finished_; }

  [[nodiscard]] const KOptCursorBlock* PrepareNextBlock() {
    if (finished_) {
      return nullptr;
    }
    if (pending_.has_value()) {
      return &*pending_;
    }
    while (true) {
      if (!k_ready_ && !PrepareNextK()) {
        return nullptr;
      }

      KOptCursorBlock block;
      block.k = current_k_;
      block.works.reserve(options_.cost_batch_size);
      while (has_combination_ && block.works.size() < options_.cost_batch_size) {
        if (options_.max_deletion_sets != 0U &&
            (result_.deletion_sets_tested >= options_.max_deletion_sets ||
             block.works.size() >= options_.max_deletion_sets - result_.deletion_sets_tested)) {
          block.budget_blocked = true;
          break;
        }
        KOptCursorWork work;
        for (std::size_t index = 0U; index < combination_.size(); ++index) {
          work.deleted_positions[index] = context_.selectable_positions[combination_[index]];
        }
        const std::span<std::size_t> deleted_positions(work.deleted_positions.data(), current_k_);
        std::sort(deleted_positions.begin(), deleted_positions.end());
        work.task = BuildKOptCostTask(context_, deleted_positions);
        block.works.push_back(std::move(work));
        has_combination_ = AdvanceCombination(&combination_, context_.selectable_positions.size());
      }
      if (!block.works.empty()) {
        pending_ = std::move(block);
        return &*pending_;
      }
      if (block.budget_blocked) {
        result_.status = KOptSearchStatus::kUnresolved;
        result_.reason = "k-opt 删除集合预算耗尽";
        finished_ = true;
        return nullptr;
      }
      k_ready_ = false;
    }
  }

  KOptCursorConsumeStats ConsumeBlock(const KOptCostBlockView& costs) {
    KOptCursorConsumeStats stats;
    if (finished_ || !pending_.has_value()) {
      throw std::logic_error("k-opt cost cursor 没有待消费 block");
    }
    const KOptCursorBlock& block = *pending_;
    const KOptReconnectTable& reconnect_table = CachedKOptReconnectTable(block.k);
    if (costs.k != block.k || costs.template_count != reconnect_table.templates.size() ||
        costs.added_costs.size() != block.works.size() * reconnect_table.templates.size()) {
      result_.status = KOptSearchStatus::kUnresolved;
      result_.reason = "k-opt cost cursor 矩阵规模错误";
      pending_.reset();
      finished_ = true;
      return stats;
    }
    for (std::size_t work_index = 0U; work_index < block.works.size(); ++work_index) {
      // cost block 可投机生成尾部 rows；规范 proof 只计入真正消费到的删除集合。
      if (!AddWithoutOverflow(&result_.deletion_sets_tested, 1U)) {
        result_.status = KOptSearchStatus::kUnresolved;
        result_.reason = "k-opt 删除集合计数溢出";
        pending_.reset();
        finished_ = true;
        return stats;
      }
      ++stats.cost_rows;
      const std::size_t row_offset = work_index * reconnect_table.templates.size();
      const std::span<const std::int64_t> row =
          costs.added_costs.subspan(row_offset, reconnect_table.templates.size());
      const std::span<const std::size_t> deleted_positions(
          block.works[work_index].deleted_positions.data(), block.k);
      ReconnectAttempt attempt = TryReconnectFromCostRow(
          *graph_, context_, outside_, deleted_positions, block.works[work_index].task,
          reconnect_table.templates, row, costs.backend, costs.cpu_verified);
      stats.candidate_templates_rechecked += attempt.candidate_templates_rechecked;
      stats.candidate_recheck_ms += attempt.candidate_recheck_ms;
      stats.completeness_fallback_ms += attempt.completeness_fallback_ms;
      if (attempt.used_completeness_fallback) {
        ++stats.cpu_completeness_rows;
        stats.cpu_completeness_templates += attempt.completeness_templates_tested;
      }
      if (ConsumeAttempt(std::move(attempt))) {
        pending_.reset();
        return stats;
      }
    }
    const bool budget_blocked = block.budget_blocked;
    pending_.reset();
    if (budget_blocked) {
      result_.status = KOptSearchStatus::kUnresolved;
      result_.reason = "k-opt 删除集合预算耗尽";
      finished_ = true;
      return stats;
    }
    if (!has_combination_) {
      k_ready_ = false;
    }
    return stats;
  }

  void FailCost(std::string reason) {
    if (finished_) {
      return;
    }
    pending_.reset();
    result_.status = KOptSearchStatus::kUnresolved;
    result_.reason = "k-opt cost 失败: " + std::move(reason);
    finished_ = true;
  }

  [[nodiscard]] KOptSearchResult TakeResult() {
    if (!finished_) {
      throw std::logic_error("k-opt cost cursor 尚未完成");
    }
    return std::move(result_);
  }

private:
  bool PrepareNextK() {
    while (++current_k_ <= options_.max_k) {
      if (current_k_ >= context_.selectable_positions.size()) {
        break;
      }
      combination_.resize(current_k_);
      for (std::size_t index = 0U; index < combination_.size(); ++index) {
        combination_[index] = index;
      }
      has_combination_ = true;
      k_ready_ = true;
      return true;
    }
    result_.status = KOptSearchStatus::kNoImprovement;
    result_.reason = "已穷举允许的 k-opt 重连但未找到严格改善";
    finished_ = true;
    return false;
  }

  bool ConsumeAttempt(ReconnectAttempt attempt) {
    if (!AddWithoutOverflow(&result_.reconnect_matchings_tested, attempt.matchings_tested)) {
      result_.status = KOptSearchStatus::kUnresolved;
      result_.reason = "k-opt 重连计数溢出";
      finished_ = true;
      return true;
    }
    if (attempt.fatal) {
      result_.status = KOptSearchStatus::kInvalid;
      result_.reason = attempt.reason;
      finished_ = true;
      return true;
    }
    if (!attempt.witness.has_value()) {
      return false;
    }
    result_.status = KOptSearchStatus::kImproved;
    result_.reason = "找到严格改善的 k-opt witness";
    result_.witness = std::move(*attempt.witness);
    std::string verify_reason;
    if (!VerifyKOptWitness(*graph_, *paths_, outside_, required_edge_, result_.witness,
                           &verify_reason)) {
      result_.status = KOptSearchStatus::kInvalid;
      result_.reason = "内部 witness 复核失败: " + verify_reason;
    }
    finished_ = true;
    return true;
  }

  const GraphSnapshot* graph_{};
  const NormalizedPathSystem* paths_{};
  EndpointMatching outside_;
  std::optional<NodeEdge> required_edge_;
  KOptSearchOptions options_;
  TourContext context_;
  KOptSearchResult result_;
  std::uint32_t current_k_{2U};
  bool k_ready_{false};
  bool has_combination_{false};
  bool finished_{false};
  std::vector<std::size_t> combination_;
  std::optional<KOptCursorBlock> pending_;
};

struct BatchedPathProofWork {
  PathSystemKOptProof proof;
  const PathMatchingCatalog* catalog{};
  std::optional<KOptPathValidationBinding> path_validation;
  std::vector<bool> covered;
  bool finished{false};
};

BatchedPathProofWork InitializeBatchedPathProof(const GraphSnapshot& graph,
                                                const NormalizedPathSystem& paths,
                                                const std::uint64_t snapshot_hash) {
  BatchedPathProofWork work;
  // 同一 batch 绑定同一不可变快照；主调方只计算一次全图哈希，避免每个 leaf 重扫 CSR。
  work.proof.snapshot_hash = snapshot_hash;
  work.proof.path_system_hash = ComputePathSystemHash(paths);
  work.proof.path_count = static_cast<std::uint32_t>(paths.paths.size());
  if (work.proof.path_count == 0U || work.proof.path_count > kMaxTestablePathCount) {
    work.proof.reason = "路径数不在 [1,7]";
    work.finished = true;
    return work;
  }
  work.catalog = &CachedPathMatchingCatalog(work.proof.path_count);
  work.proof.outside_count = static_cast<std::uint32_t>(work.catalog->outside.size());
  if (work.catalog->table.has_value()) {
    work.proof.compatibility_table_hash = work.catalog->table->generator_hash;
  }
  work.path_validation.emplace(graph, paths);
  work.covered.assign(work.catalog->outside.size(), false);
  return work;
}

std::optional<std::uint32_t> FirstUncoveredOutside(const BatchedPathProofWork& work) {
  for (std::size_t index = 0U; index < work.covered.size(); ++index) {
    if (!work.covered[index]) {
      return static_cast<std::uint32_t>(index);
    }
  }
  return std::nullopt;
}

void FinishBatchedPathProofIfCovered(BatchedPathProofWork* const work) {
  if (FirstUncoveredOutside(*work).has_value()) {
    return;
  }
  work->proof.proven = true;
  work->proof.reason = "全部 outside matching 均有严格改善 witness";
  work->finished = true;
}

void ApplyBatchedKOptSearch(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                            const std::optional<NodeEdge>& required_edge,
                            const KOptSearchOptions& options, const std::uint32_t source,
                            KOptSearchResult search, BatchedPathProofWork* const work) {
  if (!AddWithoutOverflow(&work->proof.deletion_sets_tested, search.deletion_sets_tested) ||
      !AddWithoutOverflow(&work->proof.reconnect_matchings_tested,
                          search.reconnect_matchings_tested)) {
    work->proof.reason = "path-system k-opt 统计计数溢出";
    work->finished = true;
    return;
  }
  if (search.status != KOptSearchStatus::kImproved && options.exact_fallback_max_blocks != 0U) {
    KOptSearchResult exact = FindExactTourWitness(graph, paths, work->catalog->outside[source],
                                                  required_edge, options.exact_fallback_max_blocks);
    if (!AddWithoutOverflow(&work->proof.exact_states_tested, exact.exact_states_tested)) {
      work->proof.reason = "path-system exact DP 状态计数溢出";
      work->finished = true;
      return;
    }
    if (exact.status == KOptSearchStatus::kImproved) {
      search = std::move(exact);
    } else {
      search.reason += "; exact fallback: " + exact.reason;
      search.status = exact.status;
    }
  }
  if (search.status != KOptSearchStatus::kImproved) {
    work->proof.reason = "outside " + std::to_string(source) + " unresolved: " + search.reason;
    work->finished = true;
    return;
  }
  std::string verify_reason;
  if (!VerifyKOptWitness(graph, paths, work->catalog->outside[source], required_edge,
                         search.witness, &verify_reason)) {
    work->proof.reason = "outside witness 复核失败: " + verify_reason;
    work->finished = true;
    return;
  }
  const auto inside_iterator = std::find(work->catalog->inside.begin(), work->catalog->inside.end(),
                                         search.witness.inside_matching);
  if (inside_iterator == work->catalog->inside.end()) {
    work->proof.reason = "witness inside matching 不在规范枚举中";
    work->finished = true;
    return;
  }
  const auto inside_index =
      static_cast<std::uint32_t>(inside_iterator - work->catalog->inside.begin());
  for (std::uint32_t outside_index = 0U; outside_index < work->catalog->outside.size();
       ++outside_index) {
    const bool compatible =
        work->catalog->table.has_value()
            ? work->catalog->table->Covers(outside_index, inside_index)
            : IsAlternatingHamiltonianCycle(work->catalog->outside[outside_index],
                                            search.witness.inside_matching, work->proof.path_count);
    work->covered[outside_index] = work->covered[outside_index] || compatible;
  }
  if (!work->covered[source]) {
    work->proof.reason = "witness 未覆盖其源 outside matching";
    work->finished = true;
    return;
  }
  work->proof.records.push_back({source, std::move(search.witness)});
  FinishBatchedPathProofIfCovered(work);
}

bool VerifyPathSystemKOptProofBoundToSnapshot(const GraphSnapshot& graph,
                                              const NormalizedPathSystem& paths,
                                              const std::optional<NodeEdge>& required_edge,
                                              const PathSystemKOptProof& proof,
                                              const std::uint64_t snapshot_hash,
                                              std::string* const reason) {
  if (!proof.proven || proof.snapshot_hash != snapshot_hash ||
      proof.path_system_hash != ComputePathSystemHash(paths) ||
      proof.path_count != paths.paths.size() || proof.path_count == 0 ||
      proof.path_count > kMaxTestablePathCount) {
    SetReason(reason, "path-system proof 的状态或绑定哈希不一致");
    return false;
  }
  const PathMatchingCatalog& catalog = CachedPathMatchingCatalog(proof.path_count);
  const std::vector<EndpointMatching>& outside = catalog.outside;
  if (proof.outside_count != outside.size()) {
    SetReason(reason, "path-system proof 的 outside 数量不一致");
    return false;
  }
  if (catalog.table.has_value()) {
    if (proof.compatibility_table_hash != catalog.table->generator_hash) {
      SetReason(reason, "path-system proof 的兼容表哈希不一致");
      return false;
    }
  } else if (proof.compatibility_table_hash != 0) {
    SetReason(reason, "m>5 proof 不应绑定完整兼容表");
    return false;
  }

  std::vector<bool> covered(outside.size(), false);
  for (const OutsideKOptWitness& record : proof.records) {
    if (record.source_outside_index >= outside.size() || covered[record.source_outside_index]) {
      SetReason(reason, "path-system proof 的 source outside 顺序非法");
      return false;
    }
    std::string witness_reason;
    if (!VerifyKOptWitness(graph, paths, outside[record.source_outside_index], required_edge,
                           record.witness, &witness_reason)) {
      SetReason(reason, "path-system witness 失败: " + witness_reason);
      return false;
    }
    for (std::size_t outside_index = 0; outside_index < outside.size(); ++outside_index) {
      if (IsAlternatingHamiltonianCycle(outside[outside_index], record.witness.inside_matching,
                                        proof.path_count)) {
        covered[outside_index] = true;
      }
    }
    if (!covered[record.source_outside_index]) {
      SetReason(reason, "path-system witness 不覆盖其 source outside");
      return false;
    }
  }
  if (!std::all_of(covered.begin(), covered.end(), [](const bool value) { return value; })) {
    SetReason(reason, "path-system proof 未覆盖全部 outside matching");
    return false;
  }
  SetReason(reason, "OK");
  return true;
}

void RecordKOptBatchBackend(PathSystemKOptBatchResult* const result, const std::string& backend,
                            const int selected_device) {
  if (result->cost_backend == "none") {
    result->cost_backend = backend;
  } else if (result->cost_backend != backend) {
    result->cost_backend = "mixed";
  }
  if (selected_device >= 0) {
    result->selected_device = selected_device;
  }
}

void RecordKOptCudaCache(PathSystemKOptBatchResult* const result,
                         const KOptCostBatchResult& costs) {
  if (costs.backend != "cuda") {
    return;
  }
  if (!AddWithoutOverflow(&result->cuda_cost_batches, 1U) ||
      (costs.cuda_cache.snapshot_hit && !AddWithoutOverflow(&result->snapshot_cache_hits, 1U)) ||
      (costs.cuda_cache.template_hit && !AddWithoutOverflow(&result->template_cache_hits, 1U)) ||
      (costs.cuda_cache.workspace_hit && !AddWithoutOverflow(&result->workspace_cache_hits, 1U))) {
    throw std::overflow_error("path-system k-opt CUDA cache 统计溢出");
  }
  result->peak_device_cache_bytes =
      std::max(result->peak_device_cache_bytes, costs.cuda_cache.resident_bytes);
}

void RecordKOptCpuLongTail(PathSystemKOptBatchResult* const result, const std::size_t task_count,
                           const std::size_t cell_count) {
  if (!AddWithoutOverflow(&result->cpu_long_tail_batches, 1U) ||
      !AddWithoutOverflow(&result->cpu_long_tail_tasks, task_count) ||
      !AddWithoutOverflow(&result->cpu_long_tail_cells, cell_count)) {
    throw std::overflow_error("path-system k-opt CPU long-tail 统计溢出");
  }
}

} // namespace

detail::KOptSnapshotBinding::KOptSnapshotBinding(const GraphSnapshot& graph)
    : graph_(&graph), snapshot_hash_(graph.ContentHash()) {}

namespace {

PathSystemKOptBatchResult ProvePathSystemsByKOptImpl(
    const GraphSnapshot& graph, const std::vector<NormalizedPathSystem>& path_systems,
    const std::optional<NodeEdge>& required_edge, const KOptSearchOptions& options,
    const std::optional<std::uint64_t> bound_snapshot_hash) {
  PathSystemKOptBatchResult result;
  if (path_systems.empty()) {
    result.cpu_verified = true;
    return result;
  }

  std::vector<BatchedPathProofWork> works;
  std::uint64_t snapshot_hash = 0U;
  {
    ScopedPhaseTimer timer(&result.setup_ms);
    ScopedPhaseTimer initialize_timer(&result.proof_initialize_ms);
    snapshot_hash = bound_snapshot_hash.has_value() ? *bound_snapshot_hash : graph.ContentHash();
    works.reserve(path_systems.size());
    for (const NormalizedPathSystem& paths : path_systems) {
      works.push_back(InitializeBatchedPathProof(graph, paths, snapshot_hash));
    }
  }
  const bool can_batch_costs = options.max_k >= 3U && options.max_k <= 5U &&
                               options.cost_batch_size != 0U &&
                               (options.cost_backend == PathCompatibilityBackend::kAuto ||
                                options.cost_backend == PathCompatibilityBackend::kCpu ||
                                options.cost_backend == PathCompatibilityBackend::kCuda);
  // 单一 workspace 只保存待立即消费的成本；不绑定图状态，并在本次 proof batch 结束时释放。
  KOptCpuCostWorkspace cpu_cost_workspace;

  struct ActiveSearch {
    std::size_t path_index{};
    std::uint32_t source{};
    std::optional<KOptSearchCursor> cursor;
  };
  while (true) {
    std::vector<ActiveSearch> active;
    {
      ScopedPhaseTimer timer(&result.setup_ms);
      for (std::size_t path_index = 0U; path_index < works.size(); ++path_index) {
        BatchedPathProofWork& work = works[path_index];
        if (work.finished) {
          continue;
        }
        std::uint32_t source = 0U;
        {
          ScopedPhaseTimer coverage_timer(&result.coverage_scan_ms);
          FinishBatchedPathProofIfCovered(&work);
          if (!work.finished) {
            source = *FirstUncoveredOutside(work);
          }
        }
        if (work.finished) {
          continue;
        }
        ActiveSearch search{.path_index = path_index, .source = source, .cursor = std::nullopt};
        if (can_batch_costs) {
          if (!work.path_validation.has_value()) {
            throw std::logic_error("path-system k-opt 缺少批内 path validation binding");
          }
          {
            ScopedPhaseTimer cursor_timer(&result.cursor_construct_ms);
            search.cursor.emplace(graph, path_systems[path_index], work.catalog->outside[source],
                                  required_edge, options, *work.path_validation);
          }
          if (!AddWithoutOverflow(&result.cursor_searches_started, 1U)) {
            throw std::overflow_error("path-system k-opt cursor 构造计数溢出");
          }
        }
        active.push_back(std::move(search));
      }
    }
    if (active.empty()) {
      break;
    }

    struct CursorSlice {
      std::size_t active_index{};
      std::size_t row_begin{};
      std::size_t row_count{};
    };
    while (true) {
      std::array<std::vector<KOptCostTask>, 3U> tasks_by_k;
      std::array<std::vector<CursorSlice>, 3U> slices_by_k;
      bool has_pending_block = false;
      for (std::size_t active_index = 0U; active_index < active.size(); ++active_index) {
        if (!active[active_index].cursor.has_value()) {
          continue;
        }
        const KOptCursorBlock* block = nullptr;
        {
          ScopedPhaseTimer timer(&result.cursor_prepare_ms);
          block = active[active_index].cursor->PrepareNextBlock();
        }
        if (block == nullptr) {
          continue;
        }
        if (block->k < 3U || block->k > 5U || block->works.empty()) {
          throw std::logic_error("path-system k-opt cursor 返回非法 block");
        }
        const std::size_t bucket = static_cast<std::size_t>(block->k - 3U);
        CursorSlice slice{.active_index = active_index,
                          .row_begin = tasks_by_k[bucket].size(),
                          .row_count = block->works.size()};
        for (const KOptCursorWork& work : block->works) {
          tasks_by_k[bucket].push_back(work.task);
        }
        slices_by_k[bucket].push_back(slice);
        has_pending_block = true;
      }
      if (!has_pending_block) {
        break;
      }

      for (std::uint32_t k = 3U; k <= 5U; ++k) {
        const std::size_t bucket = static_cast<std::size_t>(k - 3U);
        std::vector<KOptCostTask>& cost_tasks = tasks_by_k[bucket];
        if (cost_tasks.empty()) {
          continue;
        }
        if (!AddWithoutOverflow(&result.cost_batches, 1U) ||
            !AddWithoutOverflow(&result.cost_tasks, cost_tasks.size())) {
          throw std::overflow_error("path-system k-opt batch 统计溢出");
        }
        const std::size_t requested_cells = KOptCostCellCount(k, cost_tasks.size());
        const bool cpu_long_tail = IsCpuLongTail(options, requested_cells);
        const PathCompatibilityBackend selected_cost_backend =
            SelectKOptCostBackend(options, requested_cells);
        KOptCostBatchResult costs;
        std::span<const std::int64_t> cost_values;
        {
          ScopedPhaseTimer timer(&result.cost_evaluate_ms);
          try {
            if (selected_cost_backend == PathCompatibilityBackend::kCpu) {
              costs = EvaluateKOptTemplateCostsCpuWithWorkspace(
                  graph, k, cost_tasks, &cpu_cost_workspace, &cost_values);
            } else {
              costs = EvaluateKOptTemplateCosts(graph, k, cost_tasks, selected_cost_backend);
              cost_values = costs.added_costs;
            }
          } catch (const std::exception& error) {
            if (selected_cost_backend != PathCompatibilityBackend::kAuto) {
              for (const CursorSlice& slice : slices_by_k[bucket]) {
                active[slice.active_index].cursor->FailCost(error.what());
              }
              RecordKOptBatchBackend(&result,
                                     selected_cost_backend == PathCompatibilityBackend::kCuda
                                         ? "cuda-error"
                                         : "cpu-error",
                                     -1);
              continue;
            }
            costs = EvaluateKOptTemplateCostsCpuWithWorkspace(
                graph, k, cost_tasks, &cpu_cost_workspace, &cost_values);
          }
        }
        if (!AddWithoutOverflow(&result.cost_cells, cost_values.size())) {
          throw std::overflow_error("path-system k-opt batch cost cell 统计溢出");
        }
        if (!costs.cpu_verified) {
          throw std::logic_error("path-system k-opt cost 矩阵未通过 CPU 完整认证");
        }
        if (!AddWithoutOverflow(&result.cpu_certified_cost_cells, cost_values.size())) {
          throw std::overflow_error("path-system k-opt CPU 认证 cell 统计溢出");
        }
        result.cost_cpu_certify_ms += costs.cpu_certify_ms;
        RecordKOptCpuParallelism(&result, costs, cost_values.size());
        RecordKOptBatchBackend(&result, costs.backend, costs.selected_device);
        RecordKOptCudaCache(&result, costs);
        if (cpu_long_tail) {
          RecordKOptCpuLongTail(&result, cost_tasks.size(), cost_values.size());
        }
        if (costs.template_count == 0U ||
            cost_values.size() != cost_tasks.size() * costs.template_count) {
          throw std::logic_error("path-system k-opt batch cost 矩阵规模错误");
        }
        for (const CursorSlice& slice : slices_by_k[bucket]) {
          const std::size_t cell_begin = slice.row_begin * costs.template_count;
          const std::size_t cell_count = slice.row_count * costs.template_count;
          KOptCostBlockView cursor_costs;
          {
            ScopedPhaseTimer timer(&result.cost_scatter_ms);
            cursor_costs.k = k;
            cursor_costs.template_count = costs.template_count;
            cursor_costs.added_costs = cost_values.subspan(cell_begin, cell_count);
            cursor_costs.backend = costs.backend;
            cursor_costs.cpu_verified = costs.cpu_verified;
          }
          {
            ScopedPhaseTimer timer(&result.cursor_consume_ms);
            const KOptCursorConsumeStats stats =
                active[slice.active_index].cursor->ConsumeBlock(cursor_costs);
            if (!AddWithoutOverflow(&result.cost_rows_consumed, stats.cost_rows) ||
                !AddWithoutOverflow(&result.candidate_templates_rechecked,
                                    stats.candidate_templates_rechecked) ||
                !AddWithoutOverflow(&result.cpu_completeness_rows, stats.cpu_completeness_rows) ||
                !AddWithoutOverflow(&result.cpu_completeness_templates,
                                    stats.cpu_completeness_templates)) {
              throw std::overflow_error("path-system k-opt cursor consume 统计溢出");
            }
            result.candidate_recheck_ms += stats.candidate_recheck_ms;
            result.completeness_fallback_ms += stats.completeness_fallback_ms;
          }
        }
      }
    }

    for (ActiveSearch& search : active) {
      KOptSearchResult search_result;
      if (search.cursor.has_value()) {
        search_result = search.cursor->TakeResult();
      } else {
        if (!AddWithoutOverflow(&result.scalar_searches, 1U)) {
          throw std::overflow_error("path-system k-opt scalar 统计溢出");
        }
        {
          ScopedPhaseTimer timer(&result.scalar_search_ms);
          search_result = FindKOptWitness(graph, path_systems[search.path_index],
                                          works[search.path_index].catalog->outside[search.source],
                                          required_edge, options);
        }
      }
      {
        ScopedPhaseTimer timer(&result.apply_ms);
        ApplyBatchedKOptSearch(graph, path_systems[search.path_index], required_edge, options,
                               search.source, std::move(search_result), &works[search.path_index]);
      }
    }
  }

  result.proofs.reserve(works.size());
  result.cpu_verified = true;
  for (std::size_t index = 0U; index < works.size(); ++index) {
    if (works[index].proof.proven) {
      std::string reason;
      bool verified = false;
      {
        ScopedPhaseTimer timer(&result.proof_verify_ms);
        // graph 在整个同步 batch 内只读；沿用入口哈希，逐 proof 仍完整复核绑定与 witness。
        verified = VerifyPathSystemKOptProofBoundToSnapshot(
            graph, path_systems[index], required_edge, works[index].proof, snapshot_hash, &reason);
      }
      if (!verified) {
        throw std::logic_error("批量 path-system proof CPU 复核失败: " + reason);
      }
    }
    result.proofs.push_back(std::move(works[index].proof));
  }
  if (result.cost_backend == "none" && result.scalar_searches != 0U) {
    result.cost_backend =
        options.cost_backend == PathCompatibilityBackend::kCpu ? "cpu-scalar" : "scalar";
  }
  return result;
}

} // namespace

PathSystemKOptBatchResult ProvePathSystemsByKOpt(
    const GraphSnapshot& graph, const std::vector<NormalizedPathSystem>& path_systems,
    const std::optional<NodeEdge>& required_edge, const KOptSearchOptions& options) {
  return ProvePathSystemsByKOptImpl(graph, path_systems, required_edge, options, std::nullopt);
}

PathSystemKOptBatchResult detail::ProvePathSystemsByKOptBoundToSnapshot(
    const GraphSnapshot& graph, const std::vector<NormalizedPathSystem>& path_systems,
    const std::optional<NodeEdge>& required_edge, const KOptSnapshotBinding& binding,
    const KOptSearchOptions& options) {
  if (!binding.Matches(graph)) {
    throw std::invalid_argument("k-opt snapshot binding 与图对象不一致");
  }
  return ProvePathSystemsByKOptImpl(graph, path_systems, required_edge, options,
                                    binding.snapshot_hash());
}

bool VerifyPathSystemKOptProof(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                               const std::optional<NodeEdge>& required_edge,
                               const PathSystemKOptProof& proof, std::string* const reason) {
  // 独立 verifier 必须自行绑定调用时的完整图；只有同一同步 batch 的内部复核可复用哈希。
  return VerifyPathSystemKOptProofBoundToSnapshot(graph, paths, required_edge, proof,
                                                  graph.ContentHash(), reason);
}

namespace {

void WritePathSystemKOptProofStream(std::ostream* const output, const PathSystemKOptProof& proof) {
  *output << "CUDAEE_PATH_KOPT_PROOF_V1\n";
  *output << "proven " << (proof.proven ? 1 : 0) << '\n';
  *output << "reason " << std::quoted(proof.reason) << '\n';
  *output << "snapshot_hash " << std::hex << std::setfill('0') << std::setw(16)
          << proof.snapshot_hash << '\n';
  *output << "path_system_hash " << std::setw(16) << proof.path_system_hash << '\n';
  *output << "compatibility_table_hash " << std::setw(16) << proof.compatibility_table_hash
          << std::dec << '\n';
  *output << "path_count " << proof.path_count << '\n';
  *output << "outside_count " << proof.outside_count << '\n';
  *output << "deletion_sets_tested " << proof.deletion_sets_tested << '\n';
  *output << "reconnect_matchings_tested " << proof.reconnect_matchings_tested << '\n';
  *output << "record_count " << proof.records.size() << '\n';
  for (const OutsideKOptWitness& record : proof.records) {
    const KOptWitness& witness = record.witness;
    *output << "record " << record.source_outside_index << ' ' << witness.k << ' '
            << witness.deleted_cost << ' ' << witness.added_cost << '\n';
    *output << "deleted " << witness.deleted_edges.size();
    for (const NodeEdge& edge : witness.deleted_edges) {
      *output << ' ' << edge.u << ' ' << edge.v;
    }
    *output << '\n';
    *output << "added " << witness.added_edges.size();
    for (const NodeEdge& edge : witness.added_edges) {
      *output << ' ' << edge.u << ' ' << edge.v;
    }
    *output << '\n';
    *output << "inside " << static_cast<std::uint32_t>(witness.inside_matching.endpoint_count);
    for (std::uint32_t endpoint = 0; endpoint < witness.inside_matching.endpoint_count;
         ++endpoint) {
      *output << ' ' << static_cast<std::uint32_t>(witness.inside_matching.mate[endpoint]);
    }
    *output << "\nendrecord\n";
  }
  *output << "END\n";
}

PathSystemKOptProof ReadPathSystemKOptProofStream(std::istream* const input) {
  ExpectToken(input, "CUDAEE_PATH_KOPT_PROOF_V1");
  PathSystemKOptProof proof;
  int proven = 0;
  ExpectToken(input, "proven");
  if (!(*input >> proven) || (proven != 0 && proven != 1)) {
    throw std::runtime_error("path k-opt proof 的 proven 非法");
  }
  proof.proven = proven == 1;
  ExpectToken(input, "reason");
  if (!(*input >> std::quoted(proof.reason)) || proof.reason.size() > 4096) {
    throw std::runtime_error("path k-opt proof 的 reason 非法");
  }
  ExpectToken(input, "snapshot_hash");
  proof.snapshot_hash = ReadHexHash(input, "snapshot_hash");
  ExpectToken(input, "path_system_hash");
  proof.path_system_hash = ReadHexHash(input, "path_system_hash");
  ExpectToken(input, "compatibility_table_hash");
  proof.compatibility_table_hash = ReadHexHash(input, "compatibility_table_hash");
  ExpectToken(input, "path_count");
  if (!(*input >> proof.path_count) || proof.path_count == 0 ||
      proof.path_count > kMaxTestablePathCount) {
    throw std::runtime_error("path k-opt proof 的 path_count 非法");
  }
  ExpectToken(input, "outside_count");
  if (!(*input >> proof.outside_count) ||
      proof.outside_count != ExpectedOutsideMatchingCount(proof.path_count)) {
    throw std::runtime_error("path k-opt proof 的 outside_count 非法");
  }
  ExpectToken(input, "deletion_sets_tested");
  if (!(*input >> proof.deletion_sets_tested)) {
    throw std::runtime_error("path k-opt proof 的 deletion_sets_tested 非法");
  }
  ExpectToken(input, "reconnect_matchings_tested");
  if (!(*input >> proof.reconnect_matchings_tested)) {
    throw std::runtime_error("path k-opt proof 的 reconnect_matchings_tested 非法");
  }
  std::size_t record_count = 0;
  ExpectToken(input, "record_count");
  if (!(*input >> record_count) || record_count > proof.outside_count) {
    throw std::runtime_error("path k-opt proof 的 record_count 非法");
  }
  proof.records.reserve(record_count);
  for (std::size_t record_index = 0; record_index < record_count; ++record_index) {
    OutsideKOptWitness record;
    ExpectToken(input, "record");
    if (!(*input >> record.source_outside_index >> record.witness.k >>
          record.witness.deleted_cost >> record.witness.added_cost) ||
        record.source_outside_index >= proof.outside_count || record.witness.k < 2 ||
        record.witness.k > 200 || record.witness.deleted_cost < 0 ||
        record.witness.added_cost < 0) {
      throw std::runtime_error("path k-opt proof 的 record 头非法");
    }

    std::size_t edge_count = 0;
    ExpectToken(input, "deleted");
    if (!(*input >> edge_count) || edge_count != record.witness.k) {
      throw std::runtime_error("path k-opt proof 的 deleted 数量非法");
    }
    record.witness.deleted_edges.resize(edge_count);
    for (NodeEdge& edge : record.witness.deleted_edges) {
      if (!(*input >> edge.u >> edge.v)) {
        throw std::runtime_error("path k-opt proof 的 deleted 边非法");
      }
    }
    ExpectToken(input, "added");
    if (!(*input >> edge_count) || edge_count != record.witness.k) {
      throw std::runtime_error("path k-opt proof 的 added 数量非法");
    }
    record.witness.added_edges.resize(edge_count);
    for (NodeEdge& edge : record.witness.added_edges) {
      if (!(*input >> edge.u >> edge.v)) {
        throw std::runtime_error("path k-opt proof 的 added 边非法");
      }
    }

    std::uint32_t endpoint_count = 0;
    ExpectToken(input, "inside");
    if (!(*input >> endpoint_count) || endpoint_count != 2U * proof.path_count) {
      throw std::runtime_error("path k-opt proof 的 inside 端点数非法");
    }
    record.witness.inside_matching.endpoint_count = static_cast<std::uint8_t>(endpoint_count);
    record.witness.inside_matching.mate.fill(kUnmatchedEndpoint);
    for (std::uint32_t endpoint = 0; endpoint < endpoint_count; ++endpoint) {
      std::uint32_t partner = 0;
      if (!(*input >> partner) || partner >= endpoint_count) {
        throw std::runtime_error("path k-opt proof 的 inside mate 非法");
      }
      record.witness.inside_matching.mate[endpoint] = static_cast<std::uint8_t>(partner);
    }
    if (!IsPerfectEndpointMatching(record.witness.inside_matching, proof.path_count)) {
      throw std::runtime_error("path k-opt proof 的 inside 不是完美匹配");
    }
    ExpectToken(input, "endrecord");
    proof.records.push_back(std::move(record));
  }
  ExpectToken(input, "END");
  std::string trailing;
  if (*input >> trailing) {
    throw std::runtime_error("path k-opt proof 的 END 后存在多余字段");
  }
  return proof;
}

} // namespace

std::string SerializePathSystemKOptProof(const PathSystemKOptProof& proof) {
  std::ostringstream output;
  WritePathSystemKOptProofStream(&output, proof);
  if (!output) {
    throw std::runtime_error("序列化 path k-opt proof 失败");
  }
  return output.str();
}

PathSystemKOptProof ParsePathSystemKOptProof(const std::string_view serialized) {
  std::istringstream input{std::string(serialized)};
  return ReadPathSystemKOptProofStream(&input);
}

void WritePathSystemKOptProof(const std::filesystem::path& path, const PathSystemKOptProof& proof) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 path k-opt proof: " + path.string());
  }
  WritePathSystemKOptProofStream(&output, proof);
  if (!output) {
    throw std::runtime_error("写入 path k-opt proof 失败: " + path.string());
  }
}

PathSystemKOptProof ReadPathSystemKOptProof(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法打开 path k-opt proof: " + path.string());
  }
  return ReadPathSystemKOptProofStream(&input);
}

} // namespace cudaee
