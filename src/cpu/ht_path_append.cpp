#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

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
  state_spans.reserve(parents.size());
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
    constexpr std::size_t kMaxNodeRecords = std::numeric_limits<std::uint32_t>::max();
    if (parent_node_count > kMaxNodeRecords ||
        node_records.size() > kMaxNodeRecords - parent_node_count) {
      throw std::overflow_error("HT path-append 节点记录过多");
    }
    detail::HtPathStateSpan span;
    span.node_begin = static_cast<std::uint32_t>(node_records.size());
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
    state_spans.push_back(span);
  }

  HtPathAppendBatchResult result;
  result.feasible.reserve(tasks.size());
  result.children.reserve(tasks.size());
  for (const HtPathAppendTask& task : tasks) {
    ValidateTask(dimension, parents, task);
    result.children.push_back(
        NormalizePathSystem(BuildRawChild(parents[task.parent_index], task), dimension));
    result.feasible.push_back(static_cast<std::uint8_t>(result.children.back().valid));
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

  std::vector<std::uint8_t> cuda_flags;
  try {
    cuda_flags = detail::EvaluateHtPathAppendsCuda(dimension, state_spans, node_records, tasks,
                                                   &result.selected_device);
  } catch (const std::exception&) {
    if (backend == PathCompatibilityBackend::kCuda) {
      throw;
    }
    result.selected_device = -1;
    result.backend = "cpu-fallback";
    return result;
  }
  if (cuda_flags != result.feasible) {
    throw std::logic_error("CUDA HT path-append flags 与 CPU 规范化结果不一致");
  }
  result.backend = "cuda";
  return result;
}

} // namespace cudaee
