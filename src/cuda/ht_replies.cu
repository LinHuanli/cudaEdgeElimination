#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(HT replies)");
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
                "cudaMemcpy H2D(HT replies)");
    }
  }

  void CopyToHost(T* const destination) const {
    if (count_ != 0U) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(HT replies)");
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
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(HT replies)");
  return best_device;
}

__device__ std::uint64_t IntegerSqrtFloorDevice(const std::uint64_t value) {
  std::uint64_t remainder = value;
  std::uint64_t root = 0U;
  std::uint64_t bit = std::uint64_t{1} << 62;
  while (bit > remainder) {
    bit >>= 2;
  }
  while (bit != 0U) {
    if (remainder >= root + bit) {
      remainder -= root + bit;
      root = (root >> 1) + bit;
    } else {
      root >>= 1;
    }
    bit >>= 2;
  }
  return root;
}

__device__ std::uint64_t AbsoluteDifference(const std::int64_t first, const std::int64_t second) {
  return first >= second ? static_cast<std::uint64_t>(first - second)
                         : static_cast<std::uint64_t>(second - first);
}

__device__ std::uint64_t ExactDistanceDevice(const std::int32_t first, const std::int32_t second,
                                             const std::int64_t* const x,
                                             const std::int64_t* const y,
                                             const std::uint8_t distance_type) {
  const std::uint64_t dx = AbsoluteDifference(x[first], x[second]);
  const std::uint64_t dy = AbsoluteDifference(y[first], y[second]);
  const std::uint64_t squared = dx * dx + dy * dy;
  const std::uint64_t root = IntegerSqrtFloorDevice(squared);
  std::uint64_t rounded = root;
  if (distance_type == static_cast<std::uint8_t>(DistanceType::kEuc2D)) {
    rounded += squared - root * root > root ? 1U : 0U;
  } else {
    rounded += root * root != squared ? 1U : 0U;
  }
  return rounded;
}

__device__ bool EdgesSurviveTwoOptDevice(const NodeEdge target, const std::int32_t center,
                                         const std::int32_t neighbor, const std::int64_t* const x,
                                         const std::int64_t* const y,
                                         const std::uint8_t distance_type) {
  const std::uint64_t original = ExactDistanceDevice(target.u, target.v, x, y, distance_type) +
                                 ExactDistanceDevice(center, neighbor, x, y, distance_type);
  const std::uint64_t orientation0 = ExactDistanceDevice(target.u, neighbor, x, y, distance_type) +
                                     ExactDistanceDevice(target.v, center, x, y, distance_type);
  const std::uint64_t orientation1 = ExactDistanceDevice(target.u, center, x, y, distance_type) +
                                     ExactDistanceDevice(target.v, neighbor, x, y, distance_type);
  return orientation0 >= original || orientation1 >= original;
}

__device__ bool ReplySurvivesDevice(const NodeEdge target, const std::int32_t center,
                                    const std::int32_t first, const std::int32_t second,
                                    const std::int64_t* const x, const std::int64_t* const y,
                                    const std::uint8_t distance_type) {
  if ((first == target.u && second == target.v) || (first == target.v && second == target.u) ||
      !EdgesSurviveTwoOptDevice(target, center, first, x, y, distance_type) ||
      !EdgesSurviveTwoOptDevice(target, center, second, x, y, distance_type)) {
    return false;
  }
  const std::uint64_t original = ExactDistanceDevice(target.u, target.v, x, y, distance_type) +
                                 ExactDistanceDevice(center, first, x, y, distance_type) +
                                 ExactDistanceDevice(center, second, x, y, distance_type);
  const std::uint64_t replacement = ExactDistanceDevice(first, second, x, y, distance_type) +
                                    ExactDistanceDevice(target.u, center, x, y, distance_type) +
                                    ExactDistanceDevice(target.v, center, x, y, distance_type);
  return original <= replacement;
}

__global__ void CountHtRepliesKernel(const NodeEdge target, const std::int32_t* const centers,
                                     const std::size_t center_count,
                                     const std::int32_t* const row_offsets,
                                     const std::int32_t* const neighbors,
                                     const std::int64_t* const x, const std::int64_t* const y,
                                     const std::uint8_t distance_type,
                                     std::uint64_t* const counts) {
  const std::size_t center_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center_index >= center_count) {
    return;
  }
  const std::int32_t center = centers[center_index];
  const std::int32_t begin = row_offsets[center];
  const std::int32_t end = row_offsets[center + 1];
  std::uint64_t count = 0U;
  for (std::int32_t first_offset = begin; first_offset < end; ++first_offset) {
    const std::int32_t first = neighbors[first_offset];
    for (std::int32_t second_offset = first_offset + 1; second_offset < end; ++second_offset) {
      const std::int32_t second = neighbors[second_offset];
      count += static_cast<std::uint64_t>(
          ReplySurvivesDevice(target, center, first, second, x, y, distance_type));
    }
  }
  counts[center_index] = count;
}

__global__ void
WriteHtRepliesKernel(const NodeEdge target, const std::int32_t* const centers,
                     const std::size_t center_count, const std::int32_t* const row_offsets,
                     const std::int32_t* const neighbors, const std::int64_t* const x,
                     const std::int64_t* const y, const std::uint8_t distance_type,
                     const std::uint64_t* const offsets, const std::uint64_t total_replies,
                     HtNeighborPair* const replies, std::uint32_t* const error_code) {
  const std::size_t center_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center_index >= center_count) {
    return;
  }
  const std::int32_t center = centers[center_index];
  const std::int32_t begin = row_offsets[center];
  const std::int32_t end = row_offsets[center + 1];
  std::uint64_t output = offsets[center_index];
  for (std::int32_t first_offset = begin; first_offset < end; ++first_offset) {
    const std::int32_t first = neighbors[first_offset];
    for (std::int32_t second_offset = first_offset + 1; second_offset < end; ++second_offset) {
      const std::int32_t second = neighbors[second_offset];
      if (!ReplySurvivesDevice(target, center, first, second, x, y, distance_type)) {
        continue;
      }
      if (output >= total_replies) {
        atomicCAS(error_code, 0U, 1U);
        return;
      }
      replies[output++] = {center, first, second};
    }
  }
  if (output != offsets[center_index + 1U]) {
    atomicCAS(error_code, 0U, 2U);
  }
}

__global__ void
CountEndRepliesKernel(const HtEndReplyTask* const tasks, const std::size_t task_count,
                      const std::int32_t* const row_offsets, const std::int32_t* const neighbors,
                      std::uint64_t* const counts, std::uint32_t* const error_code) {
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count) {
    return;
  }
  const HtEndReplyTask task = tasks[task_index];
  const std::int32_t begin = row_offsets[task.endpoint];
  const std::int32_t end = row_offsets[task.endpoint + 1];
  std::uint64_t count = 0U;
  bool found_internal = false;
  for (std::int32_t offset = begin; offset < end; ++offset) {
    const std::int32_t neighbor = neighbors[offset];
    found_internal = found_internal || neighbor == task.internal_neighbor;
    count += static_cast<std::uint64_t>(neighbor != task.internal_neighbor);
  }
  if (!found_internal) {
    atomicCAS(error_code, 0U, 1U);
    return;
  }
  counts[task_index] = count;
}

__global__ void
WriteEndRepliesKernel(const HtEndReplyTask* const tasks, const std::size_t task_count,
                      const std::int32_t* const row_offsets, const std::int32_t* const neighbors,
                      const std::uint64_t* const offsets, const std::uint64_t total_replies,
                      NodeEdge* const replies, std::uint32_t* const error_code) {
  const std::size_t task_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (task_index >= task_count) {
    return;
  }
  const HtEndReplyTask task = tasks[task_index];
  const std::int32_t begin = row_offsets[task.endpoint];
  const std::int32_t end = row_offsets[task.endpoint + 1];
  std::uint64_t output = offsets[task_index];
  for (std::int32_t offset = begin; offset < end; ++offset) {
    const std::int32_t neighbor = neighbors[offset];
    if (neighbor == task.internal_neighbor) {
      continue;
    }
    if (output >= total_replies) {
      atomicCAS(error_code, 0U, 2U);
      return;
    }
    replies[output++] = task.endpoint < neighbor ? NodeEdge{task.endpoint, neighbor}
                                                 : NodeEdge{neighbor, task.endpoint};
  }
  if (output != offsets[task_index + 1U]) {
    atomicCAS(error_code, 0U, 3U);
  }
}

bool ValidateGraphShape(const GraphSnapshot& graph) {
  if ((graph.distance_type != DistanceType::kEuc2D &&
       graph.distance_type != DistanceType::kCeil2D) ||
      !graph.integer_coordinates || !graph.integer_distance_safe || graph.dimension < 4 ||
      graph.points.size() != static_cast<std::size_t>(graph.dimension) ||
      graph.row_offsets.size() != static_cast<std::size_t>(graph.dimension) + 1U ||
      graph.row_offsets.front() != 0 || graph.row_offsets.back() < 0 ||
      static_cast<std::size_t>(graph.row_offsets.back()) != graph.neighbors.size() ||
      graph.neighbors.size() != 2U * graph.ActiveEdgeCount()) {
    return false;
  }
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(node)];
    const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(node) + 1U];
    if (begin < 0 || end < begin || static_cast<std::size_t>(end) > graph.neighbors.size()) {
      return false;
    }
  }
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(node)];
    const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(node) + 1U];
    std::int32_t previous = -1;
    for (std::int32_t offset = begin; offset < end; ++offset) {
      const std::int32_t neighbor = graph.neighbors[static_cast<std::size_t>(offset)];
      if (neighbor <= previous || neighbor < 0 || neighbor >= graph.dimension || neighbor == node ||
          !graph.HasActiveEdge(neighbor, node)) {
        return false;
      }
      previous = neighbor;
    }
  }
  return true;
}

void ValidateInputs(const GraphSnapshot& graph, const NodeEdge target,
                    const std::vector<std::int32_t>& centers) {
  if (!ValidateGraphShape(graph) || target.u < 0 || target.v >= graph.dimension ||
      target.u >= target.v || !graph.HasActiveEdge(target.u, target.v)) {
    throw std::invalid_argument("CUDA HT reply 图或目标边非法");
  }
  for (const std::int32_t center : centers) {
    if (center < 0 || center >= graph.dimension || center == target.u || center == target.v) {
      throw std::invalid_argument("CUDA HT reply 中心点非法");
    }
  }
}

void ValidateEndInputs(const GraphSnapshot& graph, const std::vector<HtEndReplyTask>& tasks) {
  if (!ValidateGraphShape(graph)) {
    throw std::invalid_argument("CUDA HT end reply 图非法");
  }
  for (const HtEndReplyTask& task : tasks) {
    if (task.endpoint < 0 || task.endpoint >= graph.dimension || task.internal_neighbor < 0 ||
        task.internal_neighbor >= graph.dimension || task.endpoint == task.internal_neighbor ||
        !graph.HasActiveEdge(task.endpoint, task.internal_neighbor)) {
      throw std::invalid_argument("CUDA HT end reply task 非法");
    }
  }
}

} // namespace

bool HtHamiltonReplyCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

HtHamiltonReplyDeviceBatch EvaluateHtHamiltonRepliesCuda(const GraphSnapshot& graph,
                                                         const NodeEdge target_edge,
                                                         const std::vector<std::int32_t>& centers,
                                                         int* const selected_device) {
  ValidateInputs(graph, target_edge, centers);
  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA HT reply 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  HtHamiltonReplyDeviceBatch result;
  result.offsets.assign(centers.size() + 1U, 0U);
  if (centers.empty()) {
    return result;
  }

  constexpr std::size_t kThreads = 256U;
  const std::size_t blocks = (centers.size() + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA HT reply 网格过大");
  }

  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  for (std::size_t index = 0; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }
  DeviceBuffer<std::int32_t> device_centers(centers.size());
  DeviceBuffer<std::int32_t> device_row_offsets(graph.row_offsets.size());
  DeviceBuffer<std::int32_t> device_neighbors(graph.neighbors.size());
  DeviceBuffer<std::int64_t> device_x(host_x.size());
  DeviceBuffer<std::int64_t> device_y(host_y.size());
  DeviceBuffer<std::uint64_t> device_counts(centers.size());
  device_centers.CopyFromHost(centers.data());
  device_row_offsets.CopyFromHost(graph.row_offsets.data());
  device_neighbors.CopyFromHost(graph.neighbors.data());
  device_x.CopyFromHost(host_x.data());
  device_y.CopyFromHost(host_y.data());

  CountHtRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      target_edge, device_centers.get(), centers.size(), device_row_offsets.get(),
      device_neighbors.get(), device_x.get(), device_y.get(),
      static_cast<std::uint8_t>(graph.distance_type), device_counts.get());
  CheckCuda(cudaGetLastError(), "CountHtRepliesKernel launch");
  std::vector<std::uint64_t> counts(centers.size());
  device_counts.CopyToHost(counts.data());
  for (std::size_t index = 0; index < counts.size(); ++index) {
    if (counts[index] > std::numeric_limits<std::uint64_t>::max() - result.offsets[index]) {
      throw std::overflow_error("CUDA HT reply 总数溢出");
    }
    result.offsets[index + 1U] = result.offsets[index] + counts[index];
  }
  if (result.offsets.back() > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("CUDA HT reply 输出超过主机索引范围");
  }
  const std::size_t reply_count = static_cast<std::size_t>(result.offsets.back());
  result.replies.resize(reply_count);

  DeviceBuffer<std::uint64_t> device_offsets(result.offsets.size());
  DeviceBuffer<HtNeighborPair> device_replies(reply_count);
  DeviceBuffer<std::uint32_t> device_error_code(1U);
  device_offsets.CopyFromHost(result.offsets.data());
  CheckCuda(cudaMemset(device_error_code.get(), 0, sizeof(std::uint32_t)),
            "cudaMemset(HT reply error_code)");
  WriteHtRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      target_edge, device_centers.get(), centers.size(), device_row_offsets.get(),
      device_neighbors.get(), device_x.get(), device_y.get(),
      static_cast<std::uint8_t>(graph.distance_type), device_offsets.get(), result.offsets.back(),
      device_replies.get(), device_error_code.get());
  CheckCuda(cudaGetLastError(), "WriteHtRepliesKernel launch");
  std::uint32_t error_code = 0U;
  device_error_code.CopyToHost(&error_code);
  if (error_code == 1U) {
    throw std::runtime_error("CUDA HT reply 写出越界");
  }
  if (error_code == 2U) {
    throw std::runtime_error("CUDA HT reply count/write 不一致");
  }
  device_replies.CopyToHost(result.replies.data());
  return result;
}

bool HtEndReplyCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

HtEndReplyDeviceBatch EvaluateHtEndRepliesCuda(const GraphSnapshot& graph,
                                               const std::vector<HtEndReplyTask>& tasks,
                                               int* const selected_device) {
  ValidateEndInputs(graph, tasks);
  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA HT end reply 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  HtEndReplyDeviceBatch result;
  result.offsets.assign(tasks.size() + 1U, 0U);
  if (tasks.empty()) {
    return result;
  }

  constexpr std::size_t kThreads = 256U;
  const std::size_t blocks = (tasks.size() + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA HT end reply 网格过大");
  }

  DeviceBuffer<HtEndReplyTask> device_tasks(tasks.size());
  DeviceBuffer<std::int32_t> device_row_offsets(graph.row_offsets.size());
  DeviceBuffer<std::int32_t> device_neighbors(graph.neighbors.size());
  DeviceBuffer<std::uint64_t> device_counts(tasks.size());
  DeviceBuffer<std::uint32_t> device_error_code(1U);
  device_tasks.CopyFromHost(tasks.data());
  device_row_offsets.CopyFromHost(graph.row_offsets.data());
  device_neighbors.CopyFromHost(graph.neighbors.data());
  CheckCuda(cudaMemset(device_error_code.get(), 0, sizeof(std::uint32_t)),
            "cudaMemset(HT end reply count error_code)");

  CountEndRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      device_tasks.get(), tasks.size(), device_row_offsets.get(), device_neighbors.get(),
      device_counts.get(), device_error_code.get());
  CheckCuda(cudaGetLastError(), "CountEndRepliesKernel launch");
  std::vector<std::uint64_t> counts(tasks.size());
  device_counts.CopyToHost(counts.data());
  std::uint32_t error_code = 0U;
  device_error_code.CopyToHost(&error_code);
  if (error_code != 0U) {
    throw std::runtime_error("CUDA HT end reply 未找到路径内部邻点");
  }
  for (std::size_t index = 0; index < counts.size(); ++index) {
    if (counts[index] > std::numeric_limits<std::uint64_t>::max() - result.offsets[index]) {
      throw std::overflow_error("CUDA HT end reply 总数溢出");
    }
    result.offsets[index + 1U] = result.offsets[index] + counts[index];
  }
  if (result.offsets.back() > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("CUDA HT end reply 输出超过主机索引范围");
  }
  const std::size_t reply_count = static_cast<std::size_t>(result.offsets.back());
  result.replies.resize(reply_count);

  DeviceBuffer<std::uint64_t> device_offsets(result.offsets.size());
  DeviceBuffer<NodeEdge> device_replies(reply_count);
  device_offsets.CopyFromHost(result.offsets.data());
  CheckCuda(cudaMemset(device_error_code.get(), 0, sizeof(std::uint32_t)),
            "cudaMemset(HT end reply write error_code)");
  WriteEndRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      device_tasks.get(), tasks.size(), device_row_offsets.get(), device_neighbors.get(),
      device_offsets.get(), result.offsets.back(), device_replies.get(), device_error_code.get());
  CheckCuda(cudaGetLastError(), "WriteEndRepliesKernel launch");
  device_error_code.CopyToHost(&error_code);
  if (error_code == 2U) {
    throw std::runtime_error("CUDA HT end reply 写出越界");
  }
  if (error_code == 3U) {
    throw std::runtime_error("CUDA HT end reply count/write 不一致");
  }
  if (error_code != 0U) {
    throw std::runtime_error("CUDA HT end reply 未知设备错误");
  }
  device_replies.CopyToHost(result.replies.data());
  return result;
}

} // namespace cudaee::detail
