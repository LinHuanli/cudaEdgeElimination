#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaee::detail {
namespace {

void CheckCuda(const cudaError_t status, const char* const operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (count_ != 0U) {
      if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("CUDA HT path-append 设备缓冲区大小溢出");
      }
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(HT path-append)");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() { return data_; }
  [[nodiscard]] const T* get() const { return data_; }

  void CopyFromHost(const T* const source) {
    if (count_ != 0U) {
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count_, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D(HT path-append)");
    }
  }

  void CopyToHost(T* const destination) const {
    if (count_ != 0U) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(HT path-append)");
    }
  }

private:
  T* data_{nullptr};
  std::size_t count_{};
};

int SelectDevice(std::string* const reason) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    if (reason != nullptr) {
      *reason =
          count_status == cudaSuccess ? "没有可见 CUDA 设备" : cudaGetErrorString(count_status);
    }
    return -1;
  }
  int best_device = -1;
  std::size_t best_free_bytes = 0U;
  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        (best_device < 0 || free_bytes > best_free_bytes)) {
      best_device = device;
      best_free_bytes = free_bytes;
    }
  }
  if (best_device < 0) {
    if (reason != nullptr) {
      *reason = "所有可见 GPU 均无法查询显存";
    }
    return -1;
  }
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(HT path-append)");
  return best_device;
}

__device__ bool IsHtPathAppendFeasible(const HtPathStateSpan state,
                                       const HtPathNodeRecord* const nodes,
                                       const HtPathAppendTask task) {
  constexpr std::uint32_t kAbsentComponent = UINT32_MAX;
  std::uint32_t first_component = kAbsentComponent;
  std::uint32_t second_component = kAbsentComponent;
  std::uint8_t first_degree = 0U;
  std::uint8_t center_degree = 0U;
  std::uint8_t second_degree = 0U;
  for (std::uint32_t offset = 0U; offset < state.node_count; ++offset) {
    const HtPathNodeRecord record = nodes[state.node_begin + offset];
    if (record.node == task.first) {
      first_component = record.component;
      first_degree = record.degree;
    }
    if (record.node == task.center) {
      center_degree = record.degree;
    }
    if (record.node == task.second) {
      second_component = record.component;
      second_degree = record.degree;
    }
  }

  if (task.kind == HtPathAppendKind::kPoint) {
    // center 是新节点；两个旧端点均至多再接受一条边，且不能闭合已有同一条链。
    return center_degree == 0U && first_degree < 2U && second_degree < 2U &&
           (first_component == kAbsentComponent || second_component == kAbsentComponent ||
            first_component != second_component);
  }
  // end move 必须从现有端点出发，并只能连接新节点或另一条链的端点。
  return first_degree == 1U && second_degree < 2U &&
         (second_component == kAbsentComponent || first_component != second_component);
}

__global__ void CountHtPathAppendEdgesKernel(const HtPathStateSpan* const states,
                                             const HtPathNodeRecord* const nodes,
                                             const HtPathAppendTask* const tasks,
                                             const std::size_t task_count,
                                             std::uint8_t* const feasible,
                                             std::uint32_t* const child_edge_counts) {
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count) {
    return;
  }
  const HtPathAppendTask task = tasks[task_index];
  const HtPathStateSpan state = states[task.parent_index];
  const bool valid = IsHtPathAppendFeasible(state, nodes, task);
  feasible[task_index] = static_cast<std::uint8_t>(valid);
  child_edge_counts[task_index] =
      valid ? state.edge_count + (task.kind == HtPathAppendKind::kPoint ? 2U : 1U) : 0U;
}

__device__ NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? NodeEdge{first, second} : NodeEdge{second, first};
}

__device__ bool EdgeLess(const NodeEdge first, const NodeEdge second) {
  return first.u < second.u || (first.u == second.u && first.v < second.v);
}

__device__ bool SameEdge(const NodeEdge first, const NodeEdge second) {
  return first.u == second.u && first.v == second.v;
}

__global__ void WriteHtPathAppendEdgesKernel(
    const HtPathStateSpan* const states, const NodeEdge* const parent_edges,
    const HtPathAppendTask* const tasks, const std::uint8_t* const feasible,
    const std::uint64_t* const child_edge_offsets, const std::size_t task_count,
    NodeEdge* const child_edges, std::uint32_t* const write_error) {
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count) {
    return;
  }
  const std::uint64_t begin = child_edge_offsets[task_index];
  const std::uint64_t end = child_edge_offsets[task_index + 1U];
  if (feasible[task_index] == 0U) {
    if (begin != end) {
      atomicExch(write_error, 1U);
    }
    return;
  }

  const HtPathAppendTask task = tasks[task_index];
  const HtPathStateSpan state = states[task.parent_index];
  const std::uint32_t added_count = task.kind == HtPathAppendKind::kPoint ? 2U : 1U;
  if (end - begin != static_cast<std::uint64_t>(state.edge_count) + added_count) {
    atomicExch(write_error, 1U);
    return;
  }

  std::uint64_t output = begin;
  for (std::uint32_t offset = 0U; offset < state.edge_count; ++offset) {
    child_edges[output++] = parent_edges[state.edge_begin + offset];
  }
  if (task.kind == HtPathAppendKind::kPoint) {
    // point 的两条边是 first-center 与 center-second；写入后统一排序，避免方向分支。
    child_edges[output++] = CanonicalEdge(task.first, task.center);
    child_edges[output++] = CanonicalEdge(task.center, task.second);
  } else {
    child_edges[output++] = CanonicalEdge(task.first, task.second);
  }
  if (output != end) {
    atomicExch(write_error, 1U);
    return;
  }

  // 每个线程只修改自己的 CSR slice；插入排序适合当前短路径，并保持完全确定性。
  for (std::uint64_t current = begin + 1U; current < end; ++current) {
    const NodeEdge key = child_edges[current];
    std::uint64_t insertion = current;
    while (insertion > begin && EdgeLess(key, child_edges[insertion - 1U])) {
      child_edges[insertion] = child_edges[insertion - 1U];
      --insertion;
    }
    child_edges[insertion] = key;
  }
  for (std::uint64_t current = begin + 1U; current < end; ++current) {
    if (SameEdge(child_edges[current - 1U], child_edges[current])) {
      atomicExch(write_error, 1U);
      return;
    }
  }
}

} // namespace

bool HtPathAppendCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

HtPathAppendDeviceBatch EvaluateHtPathAppendsCuda(const std::int32_t dimension,
                                                  const std::vector<HtPathStateSpan>& states,
                                                  const std::vector<HtPathNodeRecord>& nodes,
                                                  const std::vector<NodeEdge>& parent_edges,
                                                  const std::vector<HtPathAppendTask>& tasks,
                                                  int* const selected_device) {
  if (dimension <= 0 || states.empty() || tasks.empty() ||
      states.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("CUDA HT path-append 数组规模非法");
  }

  const auto edge_less = [](const NodeEdge& first, const NodeEdge& second) {
    return first.u < second.u || (first.u == second.u && first.v < second.v);
  };
  std::size_t expected_node = 0U;
  std::size_t expected_edge = 0U;
  for (const HtPathStateSpan& state : states) {
    if (state.node_begin != expected_node || expected_node > nodes.size() ||
        state.node_count > nodes.size() - expected_node || state.edge_begin != expected_edge ||
        expected_edge > parent_edges.size() ||
        state.edge_count > parent_edges.size() - expected_edge || state.node_count < 2U ||
        state.edge_count == 0U || state.edge_count >= state.node_count) {
      throw std::invalid_argument("CUDA HT path-append state span 非法");
    }
    for (std::uint32_t offset = 0U; offset < state.node_count; ++offset) {
      const HtPathNodeRecord& record = nodes[expected_node + offset];
      if (record.node < 0 || record.node >= dimension || record.component >= state.node_count ||
          (record.degree != 1U && record.degree != 2U)) {
        throw std::invalid_argument("CUDA HT path-append node record 非法");
      }
      for (std::uint32_t prior = 0U; prior < offset; ++prior) {
        if (nodes[state.node_begin + prior].node == record.node) {
          throw std::invalid_argument("CUDA HT path-append state 含重复节点");
        }
      }
    }
    for (std::uint32_t offset = 0U; offset < state.edge_count; ++offset) {
      const NodeEdge& edge = parent_edges[state.edge_begin + offset];
      if (edge.u < 0 || edge.u >= dimension || edge.v <= edge.u || edge.v >= dimension ||
          (offset != 0U && !edge_less(parent_edges[state.edge_begin + offset - 1U], edge))) {
        throw std::invalid_argument("CUDA HT path-append parent edge 非法或未严格排序");
      }
      const HtPathNodeRecord* first_record = nullptr;
      const HtPathNodeRecord* second_record = nullptr;
      for (std::uint32_t node_offset = 0U; node_offset < state.node_count; ++node_offset) {
        const HtPathNodeRecord& record = nodes[state.node_begin + node_offset];
        if (record.node == edge.u) {
          first_record = &record;
        }
        if (record.node == edge.v) {
          second_record = &record;
        }
      }
      if (first_record == nullptr || second_record == nullptr ||
          first_record->component != second_record->component) {
        throw std::invalid_argument("CUDA HT path-append parent edge 与 node records 不一致");
      }
    }
    for (std::uint32_t node_offset = 0U; node_offset < state.node_count; ++node_offset) {
      const HtPathNodeRecord& record = nodes[state.node_begin + node_offset];
      std::uint32_t edge_degree = 0U;
      for (std::uint32_t edge_offset = 0U; edge_offset < state.edge_count; ++edge_offset) {
        const NodeEdge& edge = parent_edges[state.edge_begin + edge_offset];
        edge_degree += static_cast<std::uint32_t>(edge.u == record.node || edge.v == record.node);
      }
      if (edge_degree != record.degree) {
        throw std::invalid_argument("CUDA HT path-append parent edge度数与 node records 不一致");
      }
    }
    expected_node += state.node_count;
    expected_edge += state.edge_count;
  }
  if (expected_node != nodes.size() || expected_edge != parent_edges.size()) {
    throw std::invalid_argument("CUDA HT path-append 含未引用 parent record");
  }
  for (const HtPathAppendTask& task : tasks) {
    if (task.parent_index >= states.size() || task.first < 0 || task.first >= dimension ||
        task.second < 0 || task.second >= dimension || task.first == task.second ||
        (task.kind == HtPathAppendKind::kPoint &&
         (task.center < 0 || task.center >= dimension || task.center == task.first ||
          task.center == task.second)) ||
        (task.kind == HtPathAppendKind::kEnd && task.center != -1) ||
        (task.kind != HtPathAppendKind::kPoint && task.kind != HtPathAppendKind::kEnd)) {
      throw std::invalid_argument("CUDA HT path-append task 非法");
    }
    bool center_present = false;
    bool first_is_endpoint = false;
    const HtPathStateSpan& state = states[task.parent_index];
    for (std::uint32_t offset = 0U; offset < state.node_count; ++offset) {
      const HtPathNodeRecord& record = nodes[state.node_begin + offset];
      center_present = center_present || record.node == task.center;
      first_is_endpoint = first_is_endpoint || (record.node == task.first && record.degree == 1U);
    }
    if ((task.kind == HtPathAppendKind::kPoint && center_present) ||
        (task.kind == HtPathAppendKind::kEnd && !first_is_endpoint)) {
      throw std::invalid_argument("CUDA HT path-append task 不满足 point/end 输入约束");
    }
  }

  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA HT path-append 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }

  constexpr std::size_t kThreads = 256U;
  const std::size_t blocks = (tasks.size() + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA HT path-append 网格过大");
  }

  DeviceBuffer<HtPathStateSpan> device_states(states.size());
  DeviceBuffer<HtPathNodeRecord> device_nodes(nodes.size());
  DeviceBuffer<HtPathAppendTask> device_tasks(tasks.size());
  DeviceBuffer<std::uint8_t> device_feasible(tasks.size());
  DeviceBuffer<std::uint32_t> device_child_edge_counts(tasks.size());
  device_states.CopyFromHost(states.data());
  device_nodes.CopyFromHost(nodes.data());
  device_tasks.CopyFromHost(tasks.data());

  CountHtPathAppendEdgesKernel<<<static_cast<unsigned int>(blocks),
                                 static_cast<unsigned int>(kThreads)>>>(
      device_states.get(), device_nodes.get(), device_tasks.get(), tasks.size(),
      device_feasible.get(), device_child_edge_counts.get());
  CheckCuda(cudaGetLastError(), "CountHtPathAppendEdgesKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "CountHtPathAppendEdgesKernel synchronize");

  HtPathAppendDeviceBatch result;
  result.feasible.resize(tasks.size());
  std::vector<std::uint32_t> child_edge_counts(tasks.size());
  device_feasible.CopyToHost(result.feasible.data());
  device_child_edge_counts.CopyToHost(child_edge_counts.data());
  result.child_edge_offsets.resize(tasks.size() + 1U, 0U);
  for (std::size_t index = 0U; index < tasks.size(); ++index) {
    if (child_edge_counts[index] >
        std::numeric_limits<std::uint64_t>::max() - result.child_edge_offsets[index]) {
      throw std::overflow_error("CUDA HT path-append child 边前缀和溢出");
    }
    result.child_edge_offsets[index + 1U] =
        result.child_edge_offsets[index] + child_edge_counts[index];
  }
  const std::uint64_t total_edges = result.child_edge_offsets.back();
  if (total_edges > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("CUDA HT path-append child 边数组超过主机地址空间");
  }
  result.child_edges.resize(static_cast<std::size_t>(total_edges));

  DeviceBuffer<NodeEdge> device_parent_edges(parent_edges.size());
  DeviceBuffer<std::uint64_t> device_child_edge_offsets(result.child_edge_offsets.size());
  DeviceBuffer<NodeEdge> device_child_edges(result.child_edges.size());
  DeviceBuffer<std::uint32_t> device_write_error(1U);
  device_parent_edges.CopyFromHost(parent_edges.data());
  device_child_edge_offsets.CopyFromHost(result.child_edge_offsets.data());
  std::uint32_t write_error = 0U;
  device_write_error.CopyFromHost(&write_error);

  WriteHtPathAppendEdgesKernel<<<static_cast<unsigned int>(blocks),
                                 static_cast<unsigned int>(kThreads)>>>(
      device_states.get(), device_parent_edges.get(), device_tasks.get(), device_feasible.get(),
      device_child_edge_offsets.get(), tasks.size(), device_child_edges.get(),
      device_write_error.get());
  CheckCuda(cudaGetLastError(), "WriteHtPathAppendEdgesKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "WriteHtPathAppendEdgesKernel synchronize");
  device_write_error.CopyToHost(&write_error);
  if (write_error != 0U) {
    throw std::logic_error("CUDA HT path-append child 边写出违反 CSR 或唯一性约束");
  }
  device_child_edges.CopyToHost(result.child_edges.data());
  return result;
}

} // namespace cudaee::detail
