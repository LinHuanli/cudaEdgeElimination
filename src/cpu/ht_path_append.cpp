#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
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

bool ContainsNode(const NormalizedPathSystem& paths, const std::int32_t node) {
  return std::any_of(paths.paths.begin(), paths.paths.end(), [&](const Path& path) {
    return std::find(path.begin(), path.end(), node) != path.end();
  });
}

bool IsEndpoint(const NormalizedPathSystem& paths, const std::int32_t node) {
  return std::any_of(paths.paths.begin(), paths.paths.end(),
                     [&](const Path& path) { return path.front() == node || path.back() == node; });
}

std::vector<Path> BuildRawChild(const NormalizedPathSystem& parent, const HtPathAppendTask& task) {
  std::vector<Path> raw = parent.paths;
  if (task.kind == HtPathAppendKind::kPoint) {
    raw.push_back({task.first, task.center, task.second});
  } else {
    raw.push_back({task.first, task.second});
  }
  return raw;
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

void ValidateTask(const std::int32_t dimension, const std::vector<NormalizedPathSystem>& parents,
                  const HtPathAppendTask& task) {
  if (task.parent_index >= parents.size() || task.first < 0 || task.first >= dimension ||
      task.second < 0 || task.second >= dimension || task.first == task.second) {
    throw std::invalid_argument("HT path-append task 的父状态或端点非法");
  }
  const NormalizedPathSystem& parent = parents[task.parent_index];
  if (task.kind == HtPathAppendKind::kPoint) {
    if (task.center < 0 || task.center >= dimension || task.center == task.first ||
        task.center == task.second || ContainsNode(parent, task.center)) {
      throw std::invalid_argument("HT point append 的中心点必须合法且尚未出现在父状态中");
    }
    return;
  }
  if (task.kind != HtPathAppendKind::kEnd || task.center != -1 || !IsEndpoint(parent, task.first)) {
    throw std::invalid_argument("HT end append 必须从父状态的路径端点出发");
  }
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
  state_spans.reserve(parents.size());
  const auto parent_prepare_begin = std::chrono::steady_clock::now();
  for (const NormalizedPathSystem& parent : parents) {
    const NormalizedPathSystem canonical = NormalizePathSystem(parent.paths, dimension);
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
    for (std::size_t component = 0; component < parent.paths.size(); ++component) {
      const Path& path = parent.paths[component];
      if (component > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("HT path-append 分量编号溢出");
      }
      for (std::size_t offset = 0; offset < path.size(); ++offset) {
        const bool endpoint = offset == 0U || offset + 1U == path.size();
        node_records.push_back({path[offset], static_cast<std::uint32_t>(component),
                                static_cast<std::uint8_t>(endpoint ? 1U : 2U)});
      }
    }
    span.node_count = static_cast<std::uint32_t>(parent_node_count);
    span.edge_count = static_cast<std::uint32_t>(canonical_edges.size());
    parent_edges.insert(parent_edges.end(), canonical_edges.begin(), canonical_edges.end());
    state_spans.push_back(span);
  }

  HtPathAppendBatchResult result;
  result.parent_prepare_ms = ElapsedMilliseconds(parent_prepare_begin);
  result.feasible.reserve(tasks.size());
  result.children.reserve(tasks.size());
  result.child_edge_offsets.reserve(tasks.size() + 1U);
  result.child_edge_offsets.push_back(0U);
  for (const HtPathAppendTask& task : tasks) {
    const auto child_normalize_begin = std::chrono::steady_clock::now();
    ValidateTask(dimension, parents, task);
    result.children.push_back(
        NormalizePathSystem(BuildRawChild(parents[task.parent_index], task), dimension));
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
