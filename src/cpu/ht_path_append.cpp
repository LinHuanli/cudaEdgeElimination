#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

bool SameCanonicalPathSystem(const NormalizedPathSystem& first,
                             const NormalizedPathSystem& second) {
  return first.valid == second.valid && first.edge_count == second.edge_count &&
         first.paths == second.paths;
}

NormalizedPathSystem InvalidPathSystem(std::string reason) {
  NormalizedPathSystem result;
  result.reason = std::move(reason);
  return result;
}

struct ParentNodeLocation {
  std::int32_t node{-1};
  std::size_t component{};
  std::size_t offset{};
  bool endpoint{false};
};

using ParentNodeIndex = std::vector<ParentNodeLocation>;

const ParentNodeLocation* FindParentNode(const ParentNodeIndex& index,
                                         const std::int32_t node) {
  const auto found = std::lower_bound(
      index.begin(), index.end(), node,
      [](const ParentNodeLocation& location, const std::int32_t value) {
        return location.node < value;
      });
  return found != index.end() && found->node == node ? &*found : nullptr;
}

// HT 每个 child 只含少量实际节点。这里保持 dense 规范化器的全部规则和确定顺序，
// 但不再按完整 TSP 维度分配邻接表；最终 proof 仍由 dense 实现独立重放。
NormalizedPathSystem NormalizeSparsePathSystem(const std::vector<Path>& paths,
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

std::vector<NodeEdge> BuildCanonicalEdges(const NormalizedPathSystem& paths) {
  std::vector<NodeEdge> edges;
  edges.reserve(paths.edge_count);
  for (const Path& path : paths.paths) {
    for (std::size_t offset = 1U; offset < path.size(); ++offset) {
      const std::int32_t first = path[offset - 1U];
      const std::int32_t second = path[offset];
      edges.push_back({std::min(first, second), std::max(first, second)});
    }
  }
  std::sort(edges.begin(), edges.end());
  if (edges.size() != paths.edge_count ||
      std::adjacent_find(edges.begin(), edges.end()) != edges.end()) {
    throw std::logic_error("规范路径系统的边计数或唯一性失效");
  }
  return edges;
}

struct ResolvedAppendTask {
  const ParentNodeLocation* first{};
  const ParentNodeLocation* second{};
};

ResolvedAppendTask ValidateTask(const std::int32_t dimension,
                                const std::vector<NormalizedPathSystem>& parents,
                                const std::vector<ParentNodeIndex>& parent_indices,
                                const HtPathAppendTask& task) {
  if (task.parent_index >= parents.size() || task.first < 0 || task.first >= dimension ||
      task.second < 0 || task.second >= dimension || task.first == task.second) {
    throw std::invalid_argument("HT path-append task 的父状态或端点非法");
  }
  const ParentNodeIndex& index = parent_indices.at(task.parent_index);
  const ParentNodeLocation* const first = FindParentNode(index, task.first);
  const ParentNodeLocation* const second = FindParentNode(index, task.second);
  if (task.kind == HtPathAppendKind::kPoint) {
    if (task.center < 0 || task.center >= dimension || task.center == task.first ||
        task.center == task.second || FindParentNode(index, task.center) != nullptr) {
      throw std::invalid_argument("HT point append 的中心点必须合法且尚未出现在父状态中");
    }
    return {first, second};
  }
  if (task.kind != HtPathAppendKind::kEnd || task.center != -1 || first == nullptr ||
      !first->endpoint) {
    throw std::invalid_argument("HT end append 必须从父状态的路径端点出发");
  }
  return {first, second};
}

bool LocationsShareEdge(const ParentNodeLocation& first, const ParentNodeLocation& second) {
  return first.component == second.component &&
         (first.offset + 1U == second.offset || second.offset + 1U == first.offset);
}

void AppendOrientedComponent(Path* const merged, const NormalizedPathSystem& parent,
                             const ParentNodeLocation* const location,
                             const std::int32_t absent_node, const bool node_at_back) {
  if (location == nullptr) {
    merged->push_back(absent_node);
    return;
  }
  const Path& component = parent.paths.at(location->component);
  if (!location->endpoint) {
    throw std::logic_error("HT path-append 尝试连接父路径内部节点");
  }
  const bool forward = node_at_back ? location->offset + 1U == component.size()
                                    : location->offset == 0U;
  if (forward) {
    merged->insert(merged->end(), component.begin(), component.end());
  } else {
    merged->insert(merged->end(), component.rbegin(), component.rend());
  }
}

// parent 已在 batch 入口通过通用 sparse 规范器认证。每个 task 只增加一条或两条边，
// 因此直接合并至多两个规范链，避免为每个 child 重建 map/set 邻接；失败原因顺序
// 仍与 NormalizePathSystem 完全一致，并由单元测试中的独立 dense 规范器逐项差分。
NormalizedPathSystem AppendToNormalizedPathSystem(const NormalizedPathSystem& parent,
                                                  const HtPathAppendTask& task,
                                                  const ResolvedAppendTask& resolved) {
  if (task.kind == HtPathAppendKind::kPoint) {
    if ((resolved.first != nullptr && !resolved.first->endpoint) ||
        (resolved.second != nullptr && !resolved.second->endpoint)) {
      return InvalidPathSystem("路径并集存在度数大于 2 的节点");
    }
  } else {
    if (resolved.second != nullptr &&
        LocationsShareEdge(*resolved.first, *resolved.second)) {
      return InvalidPathSystem("路径系统包含重复边");
    }
    if (resolved.second != nullptr && !resolved.second->endpoint) {
      return InvalidPathSystem("路径并集存在度数大于 2 的节点");
    }
  }

  if (resolved.first != nullptr && resolved.second != nullptr &&
      resolved.first->component == resolved.second->component) {
    return InvalidPathSystem("路径并集包含回路");
  }

  const std::size_t first_size =
      resolved.first == nullptr ? 1U : parent.paths.at(resolved.first->component).size();
  const std::size_t second_size =
      resolved.second == nullptr ? 1U : parent.paths.at(resolved.second->component).size();
  const std::size_t center_size = task.kind == HtPathAppendKind::kPoint ? 1U : 0U;
  if (first_size > std::numeric_limits<std::size_t>::max() - second_size ||
      first_size + second_size > std::numeric_limits<std::size_t>::max() - center_size) {
    throw std::overflow_error("HT path-append 合并路径长度溢出");
  }

  Path merged;
  merged.reserve(first_size + second_size + center_size);
  AppendOrientedComponent(&merged, parent, resolved.first, task.first, true);
  if (task.kind == HtPathAppendKind::kPoint) {
    merged.push_back(task.center);
  }
  AppendOrientedComponent(&merged, parent, resolved.second, task.second, false);
  if (merged.front() > merged.back()) {
    std::reverse(merged.begin(), merged.end());
  }

  const std::size_t added_edges = task.kind == HtPathAppendKind::kPoint ? 2U : 1U;
  if (parent.edge_count > std::numeric_limits<std::size_t>::max() - added_edges) {
    throw std::overflow_error("HT path-append child 边数溢出");
  }
  NormalizedPathSystem child;
  child.valid = true;
  child.edge_count = parent.edge_count + added_edges;
  child.paths.reserve(parent.paths.size() + 1U);
  for (std::size_t component = 0U; component < parent.paths.size(); ++component) {
    if ((resolved.first != nullptr && component == resolved.first->component) ||
        (resolved.second != nullptr && component == resolved.second->component)) {
      continue;
    }
    child.paths.push_back(parent.paths[component]);
  }
  child.paths.push_back(std::move(merged));
  std::sort(child.paths.begin(), child.paths.end());
  return child;
}

} // namespace

HtPathAppendBatchResult EvaluateHtPathAppends(const std::int32_t dimension,
                                              const std::vector<NormalizedPathSystem>& parents,
                                              const std::vector<HtPathAppendTask>& tasks,
                                              const PathCompatibilityBackend backend) {
  if (dimension <= 0 || parents.empty()) {
    throw std::invalid_argument("HT path-append 需要正节点数和非空父状态");
  }
  if (backend != PathCompatibilityBackend::kAuto && backend != PathCompatibilityBackend::kCpu &&
      backend != PathCompatibilityBackend::kCuda) {
    throw std::invalid_argument("未知 HT path-append 后端");
  }

  std::vector<detail::HtPathStateSpan> state_spans;
  std::vector<detail::HtPathNodeRecord> node_records;
  std::vector<NodeEdge> parent_edges;
  std::vector<ParentNodeIndex> parent_indices;
  state_spans.reserve(parents.size());
  parent_indices.reserve(parents.size());
  const auto parent_prepare_begin = std::chrono::steady_clock::now();
  for (const NormalizedPathSystem& parent : parents) {
    const NormalizedPathSystem canonical = NormalizeSparsePathSystem(parent.paths, dimension);
    if (!parent.valid || !canonical.valid || !SameCanonicalPathSystem(parent, canonical)) {
      throw std::invalid_argument("HT path-append 父状态不是规范路径系统");
    }
    std::size_t parent_node_count = 0U;
    for (const Path& path : parent.paths) {
      if (path.size() > std::numeric_limits<std::size_t>::max() - parent_node_count) {
        throw std::overflow_error("HT path-append 父状态节点数溢出");
      }
      parent_node_count += path.size();
    }
    constexpr std::size_t kMaxRecords = std::numeric_limits<std::uint32_t>::max();
    if (parent_node_count > kMaxRecords || node_records.size() > kMaxRecords - parent_node_count) {
      throw std::overflow_error("HT path-append 节点记录过多");
    }
    const std::vector<NodeEdge> canonical_edges = BuildCanonicalEdges(parent);
    if (canonical_edges.size() > kMaxRecords ||
        parent_edges.size() > kMaxRecords - canonical_edges.size()) {
      throw std::overflow_error("HT path-append 父边记录过多");
    }
    detail::HtPathStateSpan span;
    span.node_begin = static_cast<std::uint32_t>(node_records.size());
    span.edge_begin = static_cast<std::uint32_t>(parent_edges.size());
    ParentNodeIndex parent_index;
    parent_index.reserve(parent_node_count);
    for (std::size_t component = 0; component < parent.paths.size(); ++component) {
      const Path& path = parent.paths[component];
      if (component > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("HT path-append 分量编号溢出");
      }
      for (std::size_t offset = 0; offset < path.size(); ++offset) {
        const bool endpoint = offset == 0U || offset + 1U == path.size();
        node_records.push_back({path[offset], static_cast<std::uint32_t>(component),
                                static_cast<std::uint8_t>(endpoint ? 1U : 2U)});
        parent_index.push_back({path[offset], component, offset, endpoint});
      }
    }
    std::sort(parent_index.begin(), parent_index.end(),
              [](const ParentNodeLocation& first, const ParentNodeLocation& second) {
                return first.node < second.node;
              });
    if (std::adjacent_find(parent_index.begin(), parent_index.end(),
                           [](const ParentNodeLocation& first,
                              const ParentNodeLocation& second) {
                             return first.node == second.node;
                           }) != parent_index.end()) {
      throw std::logic_error("HT path-append 规范父状态含重复节点");
    }
    span.node_count = static_cast<std::uint32_t>(parent_node_count);
    span.edge_count = static_cast<std::uint32_t>(canonical_edges.size());
    parent_edges.insert(parent_edges.end(), canonical_edges.begin(), canonical_edges.end());
    state_spans.push_back(span);
    parent_indices.push_back(std::move(parent_index));
  }

  HtPathAppendBatchResult result;
  result.parent_prepare_ms = ElapsedMilliseconds(parent_prepare_begin);
  result.feasible.reserve(tasks.size());
  result.children.reserve(tasks.size());
  result.child_edge_offsets.reserve(tasks.size() + 1U);
  result.child_edge_offsets.push_back(0U);
  for (const HtPathAppendTask& task : tasks) {
    const auto child_normalize_begin = std::chrono::steady_clock::now();
    const ResolvedAppendTask resolved = ValidateTask(dimension, parents, parent_indices, task);
    result.children.push_back(
        AppendToNormalizedPathSystem(parents[task.parent_index], task, resolved));
    result.child_normalize_ms += ElapsedMilliseconds(child_normalize_begin);

    const auto child_edges_begin = std::chrono::steady_clock::now();
    result.feasible.push_back(static_cast<std::uint8_t>(result.children.back().valid));
    if (result.children.back().valid) {
      const std::vector<NodeEdge> child_edges = BuildCanonicalEdges(result.children.back());
      if (child_edges.size() >
          std::numeric_limits<std::uint64_t>::max() - result.child_edges.size()) {
        throw std::overflow_error("HT path-append child 边记录过多");
      }
      result.child_edges.insert(result.child_edges.end(), child_edges.begin(), child_edges.end());
    }
    result.child_edge_offsets.push_back(static_cast<std::uint64_t>(result.child_edges.size()));
    result.child_edges_ms += ElapsedMilliseconds(child_edges_begin);
  }
  result.cpu_verified = true;

  if (backend == PathCompatibilityBackend::kCpu || tasks.empty()) {
    result.backend = "cpu";
    return result;
  }

  std::string availability_reason;
  if (!detail::HtPathAppendCudaAvailable(&availability_reason)) {
    if (backend == PathCompatibilityBackend::kCuda) {
      throw std::runtime_error("CUDA HT path-append 后端不可用: " + availability_reason);
    }
    result.backend = "cpu-fallback";
    return result;
  }

  detail::HtPathAppendDeviceBatch cuda_batch;
  const auto cuda_evaluate_begin = std::chrono::steady_clock::now();
  try {
    cuda_batch = detail::EvaluateHtPathAppendsCuda(dimension, state_spans, node_records,
                                                   parent_edges, tasks, &result.selected_device);
  } catch (const std::exception&) {
    result.cuda_evaluate_ms = ElapsedMilliseconds(cuda_evaluate_begin);
    if (backend == PathCompatibilityBackend::kCuda) {
      throw;
    }
    result.selected_device = -1;
    result.backend = "cpu-fallback";
    return result;
  }
  result.cuda_evaluate_ms = ElapsedMilliseconds(cuda_evaluate_begin);
  const auto cuda_compare_begin = std::chrono::steady_clock::now();
  if (cuda_batch.feasible != result.feasible ||
      cuda_batch.child_edge_offsets != result.child_edge_offsets ||
      cuda_batch.child_edges != result.child_edges) {
    throw std::logic_error("CUDA HT path-append child SoA 与 CPU 规范化结果不一致");
  }
  result.child_edge_offsets = std::move(cuda_batch.child_edge_offsets);
  result.child_edges = std::move(cuda_batch.child_edges);
  result.device_children_verified = true;
  result.backend = "cuda";
  result.cuda_compare_ms = ElapsedMilliseconds(cuda_compare_begin);
  return result;
}

} // namespace cudaee
