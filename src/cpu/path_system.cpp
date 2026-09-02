#include "cuda_edge_elimination/path_system.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

[[nodiscard]] NormalizedPathSystem InvalidPathSystem(std::string reason) {
  NormalizedPathSystem result;
  result.reason = std::move(reason);
  return result;
}

void ValidatePathCount(const std::uint32_t path_count) {
  if (path_count == 0 || path_count > kMaxTestablePathCount) {
    throw std::invalid_argument("路径数必须位于 [1,7]");
  }
}

void AddPair(EndpointMatching* const matching, const std::uint32_t first,
             const std::uint32_t second) {
  if (first == second || first >= matching->endpoint_count || second >= matching->endpoint_count ||
      matching->mate[first] != kUnmatchedEndpoint || matching->mate[second] != kUnmatchedEndpoint) {
    throw std::logic_error("生成了非法端点匹配");
  }
  matching->mate[first] = static_cast<std::uint8_t>(second);
  matching->mate[second] = static_cast<std::uint8_t>(first);
}

EndpointMatching EmptyMatching(const std::uint32_t path_count) {
  EndpointMatching matching;
  matching.endpoint_count = static_cast<std::uint8_t>(2U * path_count);
  matching.mate.fill(kUnmatchedEndpoint);
  return matching;
}

void EnumerateInsideRecursive(EndpointMatching* const matching,
                              std::vector<EndpointMatching>* const result) {
  std::uint32_t first = matching->endpoint_count;
  for (std::uint32_t endpoint = 0; endpoint < matching->endpoint_count; ++endpoint) {
    if (matching->mate[endpoint] == kUnmatchedEndpoint) {
      first = endpoint;
      break;
    }
  }
  if (first == matching->endpoint_count) {
    result->push_back(*matching);
    return;
  }

  for (std::uint32_t second = first + 1; second < matching->endpoint_count; ++second) {
    if (matching->mate[second] != kUnmatchedEndpoint) {
      continue;
    }
    matching->mate[first] = static_cast<std::uint8_t>(second);
    matching->mate[second] = static_cast<std::uint8_t>(first);
    EnumerateInsideRecursive(matching, result);
    matching->mate[first] = kUnmatchedEndpoint;
    matching->mate[second] = kUnmatchedEndpoint;
  }
}

void HashByte(std::uint64_t* const hash, const std::uint8_t value) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  *hash ^= value;
  *hash *= kFnvPrime;
}

void HashUint32(std::uint64_t* const hash, const std::uint32_t value) {
  // 明确使用小端字节序，避免生成器哈希依赖主机 ABI。
  for (std::uint32_t shift = 0; shift < 32; shift += 8) {
    HashByte(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void HashUint64(std::uint64_t* const hash, const std::uint64_t value) {
  for (std::uint32_t shift = 0; shift < 64; shift += 8) {
    HashByte(hash, static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

std::uint64_t ComputeGeneratorHash(const PathCompatibilityTable& table) {
  std::uint64_t hash = 14695981039346656037ULL;
  constexpr std::string_view kDomain = "CUDAEE_PATH_COMPATIBILITY_V1";
  for (const char value : kDomain) {
    HashByte(&hash, static_cast<std::uint8_t>(value));
  }
  HashUint32(&hash, table.path_count);
  HashUint32(&hash, table.outside_count);
  HashUint32(&hash, table.inside_count);
  HashUint32(&hash, table.words_per_inside);
  for (const std::uint64_t word : table.coverage) {
    HashUint64(&hash, word);
  }
  return hash;
}

} // namespace

NormalizedPathSystem NormalizePathSystem(const std::vector<Path>& paths,
                                         const std::int32_t node_count) {
  if (node_count <= 0) {
    return InvalidPathSystem("节点数必须为正数");
  }
  if (paths.empty()) {
    return InvalidPathSystem("路径系统不能为空");
  }

  const auto node_size = static_cast<std::size_t>(node_count);
  std::vector<std::vector<std::int32_t>> adjacency(node_size);
  std::set<std::pair<std::int32_t, std::int32_t>> edges;
  std::vector<bool> seen_in_path(node_size, false);

  for (const Path& path : paths) {
    if (path.size() < 2) {
      return InvalidPathSystem("每条路径至少需要两个节点");
    }
    std::vector<std::int32_t> touched;
    touched.reserve(path.size());
    for (const std::int32_t node : path) {
      if (node < 0 || node >= node_count) {
        return InvalidPathSystem("路径包含越界节点");
      }
      const auto index = static_cast<std::size_t>(node);
      if (seen_in_path[index]) {
        return InvalidPathSystem("单条路径内出现重复节点");
      }
      seen_in_path[index] = true;
      touched.push_back(node);
    }

    for (std::size_t index = 1; index < path.size(); ++index) {
      const std::int32_t raw_u = path[index - 1];
      const std::int32_t raw_v = path[index];
      const std::int32_t u = std::min(raw_u, raw_v);
      const std::int32_t v = std::max(raw_u, raw_v);
      if (!edges.emplace(u, v).second) {
        return InvalidPathSystem("路径系统包含重复边");
      }
      adjacency[static_cast<std::size_t>(raw_u)].push_back(raw_v);
      adjacency[static_cast<std::size_t>(raw_v)].push_back(raw_u);
      if (adjacency[static_cast<std::size_t>(raw_u)].size() > 2 ||
          adjacency[static_cast<std::size_t>(raw_v)].size() > 2) {
        return InvalidPathSystem("路径并集存在度数大于 2 的节点");
      }
    }
    for (const std::int32_t node : touched) {
      seen_in_path[static_cast<std::size_t>(node)] = false;
    }
  }

  NormalizedPathSystem result;
  result.edge_count = edges.size();
  std::vector<bool> visited(node_size, false);
  for (std::int32_t start = 0; start < node_count; ++start) {
    const auto start_index = static_cast<std::size_t>(start);
    if (visited[start_index] || adjacency[start_index].size() != 1) {
      continue;
    }

    Path merged;
    std::int32_t previous = -1;
    std::int32_t current = start;
    while (true) {
      const auto current_index = static_cast<std::size_t>(current);
      if (visited[current_index]) {
        return InvalidPathSystem("路径并集包含回路");
      }
      visited[current_index] = true;
      merged.push_back(current);

      std::int32_t next = -1;
      for (const std::int32_t neighbor : adjacency[current_index]) {
        if (neighbor != previous) {
          if (next != -1) {
            return InvalidPathSystem("路径并集不是简单链");
          }
          next = neighbor;
        }
      }
      if (next == -1) {
        break;
      }
      previous = current;
      current = next;
    }
    result.paths.push_back(std::move(merged));
  }

  for (std::int32_t node = 0; node < node_count; ++node) {
    const auto index = static_cast<std::size_t>(node);
    if (!adjacency[index].empty() && !visited[index]) {
      return InvalidPathSystem("路径并集包含回路");
    }
  }

  std::sort(result.paths.begin(), result.paths.end());
  std::size_t reconstructed_edges = 0;
  for (const Path& path : result.paths) {
    reconstructed_edges += path.size() - 1;
  }
  if (reconstructed_edges != result.edge_count) {
    return InvalidPathSystem("规范化路径未能保持边集");
  }
  result.valid = true;
  return result;
}

NormalizedPathSystem detail::NormalizeSparsePathSystem(const std::vector<Path>& paths,
                                                       const std::int32_t node_count) {
  if (node_count <= 0) {
    return InvalidPathSystem("节点数必须为正数");
  }
  if (paths.empty()) {
    return InvalidPathSystem("路径系统不能为空");
  }

  struct SparseNode {
    std::vector<std::int32_t> neighbors;
    bool visited{false};
  };
  std::map<std::int32_t, SparseNode> adjacency;
  std::set<std::pair<std::int32_t, std::int32_t>> edges;
  for (const Path& path : paths) {
    if (path.size() < 2U) {
      return InvalidPathSystem("每条路径至少需要两个节点");
    }
    std::set<std::int32_t> seen_in_path;
    for (const std::int32_t node : path) {
      if (node < 0 || node >= node_count) {
        return InvalidPathSystem("路径包含越界节点");
      }
      if (!seen_in_path.insert(node).second) {
        return InvalidPathSystem("单条路径内出现重复节点");
      }
    }

    for (std::size_t index = 1U; index < path.size(); ++index) {
      const std::int32_t raw_u = path[index - 1U];
      const std::int32_t raw_v = path[index];
      const std::int32_t u = std::min(raw_u, raw_v);
      const std::int32_t v = std::max(raw_u, raw_v);
      if (!edges.emplace(u, v).second) {
        return InvalidPathSystem("路径系统包含重复边");
      }
      SparseNode& u_node = adjacency[raw_u];
      SparseNode& v_node = adjacency[raw_v];
      u_node.neighbors.push_back(raw_v);
      v_node.neighbors.push_back(raw_u);
      if (u_node.neighbors.size() > 2U || v_node.neighbors.size() > 2U) {
        return InvalidPathSystem("路径并集存在度数大于 2 的节点");
      }
    }
  }

  NormalizedPathSystem result;
  result.edge_count = edges.size();
  for (auto& [start, start_node] : adjacency) {
    if (start_node.visited || start_node.neighbors.size() != 1U) {
      continue;
    }

    Path merged;
    std::int32_t previous = -1;
    std::int32_t current = start;
    while (true) {
      SparseNode& current_node = adjacency.at(current);
      if (current_node.visited) {
        return InvalidPathSystem("路径并集包含回路");
      }
      current_node.visited = true;
      merged.push_back(current);

      std::int32_t next = -1;
      for (const std::int32_t neighbor : current_node.neighbors) {
        if (neighbor != previous) {
          if (next != -1) {
            return InvalidPathSystem("路径并集不是简单链");
          }
          next = neighbor;
        }
      }
      if (next == -1) {
        break;
      }
      previous = current;
      current = next;
    }
    result.paths.push_back(std::move(merged));
  }

  for (const auto& [node, state] : adjacency) {
    static_cast<void>(node);
    if (!state.neighbors.empty() && !state.visited) {
      return InvalidPathSystem("路径并集包含回路");
    }
  }

  std::sort(result.paths.begin(), result.paths.end());
  std::size_t reconstructed_edges = 0U;
  for (const Path& path : result.paths) {
    reconstructed_edges += path.size() - 1U;
  }
  if (reconstructed_edges != result.edge_count) {
    return InvalidPathSystem("规范化路径未能保持边集");
  }
  result.valid = true;
  return result;
}

std::size_t ExpectedOutsideMatchingCount(const std::uint32_t path_count) {
  ValidatePathCount(path_count);
  std::size_t count = std::size_t{1} << (path_count - 1U);
  for (std::uint32_t factor = 2; factor < path_count; ++factor) {
    count *= factor;
  }
  return count;
}

std::size_t ExpectedInsideMatchingCount(const std::uint32_t path_count) {
  ValidatePathCount(path_count);
  std::size_t count = 1;
  for (std::uint32_t factor = 1; factor < 2U * path_count; factor += 2) {
    count *= factor;
  }
  return count;
}

std::vector<EndpointMatching> EnumerateOutsideMatchings(const std::uint32_t path_count) {
  ValidatePathCount(path_count);
  std::vector<EndpointMatching> result;
  result.reserve(ExpectedOutsideMatchingCount(path_count));
  if (path_count == 1) {
    EndpointMatching matching = EmptyMatching(path_count);
    AddPair(&matching, 0, 1);
    result.push_back(matching);
    return result;
  }

  std::vector<std::uint32_t> permutation;
  for (std::uint32_t path = 1; path < path_count; ++path) {
    permutation.push_back(path);
  }
  do {
    const std::uint32_t orientation_count = 1U << (path_count - 1U);
    for (std::uint32_t orientations = 0; orientations < orientation_count; ++orientations) {
      EndpointMatching matching = EmptyMatching(path_count);
      std::uint32_t outgoing = 1; // 固定第 0 条路径为正向，从其末端离开。
      for (std::uint32_t position = 0; position < permutation.size(); ++position) {
        const std::uint32_t path = permutation[position];
        const bool reversed = (orientations & (1U << position)) != 0;
        const std::uint32_t incoming = 2U * path + (reversed ? 1U : 0U);
        AddPair(&matching, outgoing, incoming);
        outgoing = 2U * path + (reversed ? 0U : 1U);
      }
      AddPair(&matching, outgoing, 0);
      result.push_back(matching);
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  if (result.size() != ExpectedOutsideMatchingCount(path_count)) {
    throw std::logic_error("outside matching 数量与公式不一致");
  }
  return result;
}

std::vector<EndpointMatching> EnumerateInsideMatchings(const std::uint32_t path_count) {
  ValidatePathCount(path_count);
  std::vector<EndpointMatching> result;
  result.reserve(ExpectedInsideMatchingCount(path_count));
  EndpointMatching matching = EmptyMatching(path_count);
  EnumerateInsideRecursive(&matching, &result);
  if (result.size() != ExpectedInsideMatchingCount(path_count)) {
    throw std::logic_error("inside matching 数量与公式不一致");
  }
  return result;
}

bool IsPerfectEndpointMatching(const EndpointMatching& matching, const std::uint32_t path_count) {
  if (path_count == 0 || path_count > kMaxTestablePathCount ||
      matching.endpoint_count != 2U * path_count) {
    return false;
  }
  for (std::uint32_t endpoint = 0; endpoint < matching.endpoint_count; ++endpoint) {
    const std::uint8_t partner = matching.mate[endpoint];
    if (partner >= matching.endpoint_count || partner == endpoint ||
        matching.mate[partner] != endpoint) {
      return false;
    }
  }
  return true;
}

bool IsAlternatingHamiltonianCycle(const EndpointMatching& outside, const EndpointMatching& inside,
                                   const std::uint32_t path_count) {
  if (!IsPerfectEndpointMatching(outside, path_count) ||
      !IsPerfectEndpointMatching(inside, path_count)) {
    return false;
  }

  const std::uint32_t endpoint_count = 2U * path_count;
  std::array<bool, kMaxPathEndpoints> visited{};
  std::uint32_t current = 0;
  std::uint32_t steps = 0;
  while (!visited[current]) {
    visited[current] = true;
    current = (steps % 2U == 0U) ? inside.mate[current] : outside.mate[current];
    ++steps;
  }
  return steps == endpoint_count && current == 0;
}

bool PathCompatibilityTable::Covers(const std::uint32_t outside_index,
                                    const std::uint32_t inside_index) const {
  if (outside_index >= outside_count || inside_index >= inside_count) {
    throw std::out_of_range("路径兼容表索引越界");
  }
  const std::size_t word_index =
      static_cast<std::size_t>(inside_index) * words_per_inside + outside_index / 64U;
  if (word_index >= coverage.size()) {
    throw std::logic_error("路径兼容表布局损坏");
  }
  return (coverage[word_index] & (std::uint64_t{1} << (outside_index % 64U))) != 0;
}

PathCompatibilityTable BuildPathCompatibilityTable(const std::uint32_t path_count) {
  ValidatePathCount(path_count);
  if (path_count > kMaxGpuPathCount) {
    throw std::invalid_argument("完整兼容表仅支持 m<=5；m=6,7 必须 CPU 直接判定");
  }
  const std::vector<EndpointMatching> outside = EnumerateOutsideMatchings(path_count);
  const std::vector<EndpointMatching> inside = EnumerateInsideMatchings(path_count);
  if (outside.size() > std::numeric_limits<std::uint32_t>::max() ||
      inside.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("路径兼容表计数超出 uint32_t");
  }

  PathCompatibilityTable table;
  table.path_count = path_count;
  table.outside_count = static_cast<std::uint32_t>(outside.size());
  table.inside_count = static_cast<std::uint32_t>(inside.size());
  table.words_per_inside = (table.outside_count + 63U) / 64U;
  table.coverage.assign(static_cast<std::size_t>(table.inside_count) * table.words_per_inside, 0);

  for (std::uint32_t inside_index = 0; inside_index < table.inside_count; ++inside_index) {
    for (std::uint32_t outside_index = 0; outside_index < table.outside_count; ++outside_index) {
      if (IsAlternatingHamiltonianCycle(outside[outside_index], inside[inside_index], path_count)) {
        const std::size_t word_index =
            static_cast<std::size_t>(inside_index) * table.words_per_inside + outside_index / 64U;
        table.coverage[word_index] |= std::uint64_t{1} << (outside_index % 64U);
      }
    }
  }
  table.generator_hash = ComputeGeneratorHash(table);
  return table;
}

PathCompatibilityBatchResult
EvaluatePathCompatibility(const std::uint32_t path_count,
                          const std::vector<PathCompatibilityQuery>& queries,
                          const PathCompatibilityBackend backend) {
  ValidatePathCount(path_count);
  const std::vector<EndpointMatching> outside = EnumerateOutsideMatchings(path_count);
  const std::vector<EndpointMatching> inside = EnumerateInsideMatchings(path_count);
  for (const PathCompatibilityQuery& query : queries) {
    if (query.outside_index >= outside.size() || query.inside_index >= inside.size()) {
      throw std::out_of_range("路径兼容查询索引越界");
    }
  }

  PathCompatibilityBatchResult result;
  if (path_count > kMaxGpuPathCount) {
    result.backend = backend == PathCompatibilityBackend::kCpu ? "cpu" : "cpu-fallback-m>5";
    result.compatible.reserve(queries.size());
    for (const PathCompatibilityQuery& query : queries) {
      result.compatible.push_back(static_cast<std::uint8_t>(IsAlternatingHamiltonianCycle(
          outside[query.outside_index], inside[query.inside_index], path_count)));
    }
    result.cpu_verified = true;
    return result;
  }

  const PathCompatibilityTable table = BuildPathCompatibilityTable(path_count);
  result.generator_hash = table.generator_hash;
  bool use_cuda = backend == PathCompatibilityBackend::kCuda;
  if (backend == PathCompatibilityBackend::kAuto) {
    std::string reason;
    use_cuda = detail::PathCompatibilityCudaAvailable(&reason);
  }
  if (use_cuda) {
    std::string reason;
    if (!detail::PathCompatibilityCudaAvailable(&reason)) {
      throw std::runtime_error("CUDA 路径兼容后端不可用: " + reason);
    }
    result.compatible =
        detail::LookupPathCompatibilityCuda(table, queries, &result.selected_device);
    result.backend = "cuda";
  } else {
    result.compatible.reserve(queries.size());
    for (const PathCompatibilityQuery& query : queries) {
      result.compatible.push_back(
          static_cast<std::uint8_t>(table.Covers(query.outside_index, query.inside_index)));
    }
    result.backend = "cpu";
  }

  if (result.compatible.size() != queries.size()) {
    throw std::logic_error("路径兼容后端返回数量错误");
  }
  for (std::size_t index = 0; index < queries.size(); ++index) {
    const PathCompatibilityQuery& query = queries[index];
    const bool expected = IsAlternatingHamiltonianCycle(outside[query.outside_index],
                                                        inside[query.inside_index], path_count);
    if ((result.compatible[index] != 0) != expected) {
      throw std::runtime_error("路径兼容后端未通过独立 CPU 复核");
    }
  }
  result.cpu_verified = true;
  return result;
}

} // namespace cudaee
