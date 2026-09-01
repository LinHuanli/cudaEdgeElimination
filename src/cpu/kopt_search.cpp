#include "cuda_edge_elimination/local_search.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

using EdgeSet = std::set<NodeEdge>;

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
                      TourContext* const context, std::string* const reason) {
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

  for (std::size_t position = 0; position < context->tour.size(); ++position) {
    const NodeEdge edge = CanonicalEdge(context->tour[position],
                                        context->tour[(position + 1) % context->tour.size()]);
    if (context->path_edges.contains(edge)) {
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
                            const std::vector<std::size_t>& deleted_positions,
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

KOptCostTask BuildKOptCostTask(const GraphSnapshot& graph, const TourContext& context,
                               const std::vector<std::size_t>& deleted_positions) {
  KOptCostTask task;
  std::vector<NodeEdge> deleted_edges;
  deleted_edges.reserve(deleted_positions.size());
  for (std::size_t edge = 0; edge < deleted_positions.size(); ++edge) {
    const std::size_t position = deleted_positions[edge];
    task.port_nodes[2U * edge] = context.tour[position];
    task.port_nodes[2U * edge + 1U] = context.tour[(position + 1) % context.tour.size()];
    deleted_edges.push_back(
        CanonicalEdge(task.port_nodes[2U * edge], task.port_nodes[2U * edge + 1U]));
  }
  std::string reason;
  if (!SumEdgeCosts(graph, deleted_edges, &task.deleted_cost, &reason)) {
    throw std::runtime_error("无法构造 k-opt cost task: " + reason);
  }
  return task;
}

ReconnectAttempt TryReconnect(const GraphSnapshot& graph, const TourContext& context,
                              const EndpointMatching& outside,
                              const std::vector<std::size_t>& deleted_positions,
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

void ValidateKOptCostTask(const GraphSnapshot& graph, const std::uint32_t k,
                          const KOptCostTask& task) {
  if (task.deleted_cost < 0) {
    throw std::invalid_argument("k-opt cost task 的 deleted_cost 为负");
  }
  EdgeSet deleted_edges;
  for (std::uint32_t edge = 0; edge < k; ++edge) {
    const std::size_t first_port = std::size_t{2} * edge;
    const std::int32_t first = task.port_nodes[first_port];
    const std::int32_t second = task.port_nodes[first_port + 1];
    if (first < 0 || second < 0 || first >= graph.dimension || second >= graph.dimension ||
        first == second || !deleted_edges.insert(CanonicalEdge(first, second)).second) {
      throw std::invalid_argument("k-opt cost task 包含非法或重复删除边");
    }
  }
}

std::int64_t ScoreKOptTemplateCpu(const GraphSnapshot& graph, const std::uint32_t k,
                                  const KOptCostTask& task,
                                  const EndpointMatching& reconnect_template) {
  EdgeSet deleted_edges;
  for (std::uint32_t edge = 0; edge < k; ++edge) {
    const std::size_t first_port = std::size_t{2} * edge;
    deleted_edges.insert(
        CanonicalEdge(task.port_nodes[first_port], task.port_nodes[first_port + 1]));
  }
  EdgeSet added_edges;
  for (std::uint32_t port = 0; port < reconnect_template.endpoint_count; ++port) {
    const std::uint32_t partner = reconnect_template.mate[port];
    if (port >= partner) {
      continue;
    }
    const NodeEdge edge = CanonicalEdge(task.port_nodes[port], task.port_nodes[partner]);
    if (edge.u == edge.v || deleted_edges.contains(edge) || !added_edges.insert(edge).second) {
      return kInvalidKOptTemplateCost;
    }
  }
  if (added_edges.size() != k) {
    return kInvalidKOptTemplateCost;
  }
  __int128 total = 0;
  for (const NodeEdge& edge : added_edges) {
    const std::int64_t distance = graph.Distance(edge.u, edge.v);
    if (distance < 0) {
      return kInvalidKOptTemplateCost;
    }
    total += distance;
  }
  if (total > std::numeric_limits<std::int64_t>::max()) {
    return kInvalidKOptTemplateCost;
  }
  return static_cast<std::int64_t>(total);
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
  constexpr std::array<std::size_t, 3> kExpectedCounts = {4, 25, 208};
  if (table.templates.size() != kExpectedCounts[static_cast<std::size_t>(k - 3U)]) {
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

KOptCostBatchResult EvaluateKOptTemplateCosts(const GraphSnapshot& graph, const std::uint32_t k,
                                              const std::vector<KOptCostTask>& tasks,
                                              const PathCompatibilityBackend backend) {
  if (k < 3 || k > 5) {
    throw std::invalid_argument("k-opt cost 的 k 必须位于 [3,5]");
  }
  if (!graph.integer_coordinates || !graph.integer_distance_safe || graph.dimension <= 0 ||
      graph.points.size() != static_cast<std::size_t>(graph.dimension)) {
    throw std::invalid_argument("k-opt cost 只支持平方距离安全的整数坐标图");
  }
  for (const KOptCostTask& task : tasks) {
    ValidateKOptCostTask(graph, k, task);
  }
  const KOptReconnectTable table = BuildKOptReconnectTable(k);
  if (!tasks.empty() &&
      table.templates.size() > std::numeric_limits<std::size_t>::max() / tasks.size()) {
    throw std::overflow_error("k-opt cost 矩阵规模溢出");
  }

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
    result.added_costs =
        detail::EvaluateKOptTemplateCostsCuda(graph, table, tasks, &result.selected_device);
    result.backend = "cuda";
  } else {
    result.added_costs.reserve(tasks.size() * table.templates.size());
    for (const KOptCostTask& task : tasks) {
      for (const EndpointMatching& reconnect_template : table.templates) {
        result.added_costs.push_back(ScoreKOptTemplateCpu(graph, k, task, reconnect_template));
      }
    }
    result.backend = "cpu";
  }
  if (result.added_costs.size() != tasks.size() * table.templates.size()) {
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

KOptSearchResult FindKOptWitness(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
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
    const KOptReconnectTable reconnect_table = BuildKOptReconnectTable(k);
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
            result.deletion_sets_tested >= options.max_deletion_sets) {
          budget_blocked = true;
          break;
        }
        ++result.deletion_sets_tested;
        CostWork work;
        work.deleted_positions.reserve(k);
        for (const std::size_t selected : combination) {
          work.deleted_positions.push_back(context.selectable_positions[selected]);
        }
        std::sort(work.deleted_positions.begin(), work.deleted_positions.end());
        work.task = BuildKOptCostTask(graph, context, work.deleted_positions);
        works.push_back(std::move(work));
        has_combination = AdvanceCombination(&combination, selectable_count);
      }

      if (!works.empty()) {
        std::vector<KOptCostTask> tasks;
        tasks.reserve(works.size());
        for (const CostWork& work : works) {
          tasks.push_back(work.task);
        }
        KOptCostBatchResult costs;
        try {
          costs = EvaluateKOptTemplateCosts(graph, k, tasks, options.cost_backend);
        } catch (const std::exception& error) {
          if (options.cost_backend != PathCompatibilityBackend::kAuto) {
            result.status = KOptSearchStatus::kUnresolved;
            result.reason = std::string("CUDA k-opt cost 失败: ") + error.what();
            return result;
          }
          costs = EvaluateKOptTemplateCosts(graph, k, tasks, PathCompatibilityBackend::kCpu);
        }
        if (!AddWithoutOverflow(&result.reconnect_matchings_tested,
                                static_cast<std::uint64_t>(costs.added_costs.size()))) {
          result.status = KOptSearchStatus::kUnresolved;
          result.reason = "k-opt cost 单元计数溢出";
          return result;
        }

        for (std::size_t work_index = 0; work_index < works.size(); ++work_index) {
          const std::size_t row_offset = work_index * reconnect_table.templates.size();
          for (std::size_t template_index = 0; template_index < reconnect_table.templates.size();
               ++template_index) {
            if (costs.added_costs[row_offset + template_index] >=
                works[work_index].task.deleted_cost) {
              continue;
            }
            const std::vector<EndpointMatching> preferred = {
                reconnect_table.templates[template_index]};
            if (consume_attempt(TryReconnect(graph, context, outside,
                                             works[work_index].deleted_positions, preferred))) {
              return result;
            }
          }
          // GPU 是候选器：任何未命中或坏候选都必须由 CPU 全模板穷举兜底。
          if (costs.backend == "cuda" &&
              consume_attempt(TryReconnect(graph, context, outside,
                                           works[work_index].deleted_positions,
                                           reconnect_table.templates))) {
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
  const std::vector<EndpointMatching> outside = EnumerateOutsideMatchings(proof.path_count);
  const std::vector<EndpointMatching> inside = EnumerateInsideMatchings(proof.path_count);
  proof.outside_count = static_cast<std::uint32_t>(outside.size());
  std::optional<PathCompatibilityTable> table;
  if (proof.path_count <= kMaxGpuPathCount) {
    table = BuildPathCompatibilityTable(proof.path_count);
    proof.compatibility_table_hash = table->generator_hash;
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
          table.has_value()
              ? table->Covers(outside_index, inside_index)
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

bool VerifyPathSystemKOptProof(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                               const std::optional<NodeEdge>& required_edge,
                               const PathSystemKOptProof& proof, std::string* const reason) {
  if (!proof.proven || proof.snapshot_hash != graph.ContentHash() ||
      proof.path_system_hash != ComputePathSystemHash(paths) ||
      proof.path_count != paths.paths.size() || proof.path_count == 0 ||
      proof.path_count > kMaxTestablePathCount) {
    SetReason(reason, "path-system proof 的状态或绑定哈希不一致");
    return false;
  }
  const std::vector<EndpointMatching> outside = EnumerateOutsideMatchings(proof.path_count);
  if (proof.outside_count != outside.size()) {
    SetReason(reason, "path-system proof 的 outside 数量不一致");
    return false;
  }
  if (proof.path_count <= kMaxGpuPathCount) {
    if (proof.compatibility_table_hash !=
        BuildPathCompatibilityTable(proof.path_count).generator_hash) {
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
