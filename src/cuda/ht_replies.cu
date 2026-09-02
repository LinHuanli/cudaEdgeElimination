#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
  DeviceBuffer() = default;

  DeviceBuffer(const std::size_t count, const int device) : device_(device), count_(count) {
    if (count_ != 0U) {
      if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("CUDA HT reply buffer 字节数溢出");
      }
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(HT reply buffer allocate)");
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(HT replies)");
    }
  }

  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { MoveFrom(&other); }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      MoveFrom(&other);
    }
    return *this;
  }

  [[nodiscard]] T* get() { return data_; }
  [[nodiscard]] const T* get() const { return data_; }
  [[nodiscard]] std::size_t count() const { return count_; }
  [[nodiscard]] std::uint64_t bytes() const {
    return static_cast<std::uint64_t>(count_) * sizeof(T);
  }

  void CopyFromHost(const T* const source, const std::size_t count) {
    if (count > count_) {
      throw std::logic_error("CUDA HT reply H2D 超出驻留 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(HT reply H2D)");
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D(HT replies)");
    }
  }

  void CopyToHost(T* const destination, const std::size_t count) const {
    if (count > count_) {
      throw std::logic_error("CUDA HT reply D2H 超出驻留 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(HT reply D2H)");
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(HT replies)");
    }
  }

private:
  void Reset() noexcept {
    if (data_ != nullptr) {
      // 析构不能抛异常；owner device 防止释放另一个 CUDA context 的指针。
      static_cast<void>(cudaSetDevice(device_));
      static_cast<void>(cudaFree(data_));
    }
    data_ = nullptr;
    device_ = -1;
    count_ = 0U;
  }

  void MoveFrom(DeviceBuffer* const other) noexcept {
    data_ = other->data_;
    device_ = other->device_;
    count_ = other->count_;
    other->data_ = nullptr;
    other->device_ = -1;
    other->count_ = 0U;
  }

  T* data_{nullptr};
  int device_{-1};
  std::size_t count_{};
};

struct HtReplyDeviceCache {
  explicit HtReplyDeviceCache(const int selected_device) : device(selected_device) {}

  int device{-1};
  std::int32_t dimension{};
  DistanceType distance_type{DistanceType::kEuc2D};
  std::vector<std::int64_t> host_x;
  std::vector<std::int64_t> host_y;
  std::vector<std::int32_t> host_row_offsets;
  std::vector<std::int32_t> host_neighbors;
  DeviceBuffer<std::int64_t> device_x;
  DeviceBuffer<std::int64_t> device_y;
  DeviceBuffer<std::int32_t> device_row_offsets;
  DeviceBuffer<std::int32_t> device_neighbors;
  DeviceBuffer<std::int32_t> device_centers;
  DeviceBuffer<HtEndReplyTask> device_end_tasks;
  DeviceBuffer<std::uint64_t> device_counts;
  DeviceBuffer<std::uint64_t> device_offsets;
  DeviceBuffer<HtNeighborPair> device_hamilton_replies;
  DeviceBuffer<NodeEdge> device_end_replies;
  DeviceBuffer<std::uint32_t> device_error_code;
};

thread_local std::vector<std::unique_ptr<HtReplyDeviceCache>> g_reply_device_caches;
thread_local int g_reply_preferred_device = -1;

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
  if (g_reply_preferred_device >= 0 && g_reply_preferred_device < device_count &&
      cudaSetDevice(g_reply_preferred_device) == cudaSuccess) {
    return g_reply_preferred_device;
  }
  g_reply_preferred_device = -1;
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
  g_reply_preferred_device = best_device;
  return best_device;
}

HtReplyDeviceCache& CacheForDevice(const int device) {
  const auto iterator = std::find_if(g_reply_device_caches.begin(), g_reply_device_caches.end(),
                                     [device](const std::unique_ptr<HtReplyDeviceCache>& cache) {
                                       return cache->device == device;
                                     });
  if (iterator != g_reply_device_caches.end()) {
    return **iterator;
  }
  g_reply_device_caches.push_back(std::make_unique<HtReplyDeviceCache>(device));
  return *g_reply_device_caches.back();
}

bool GraphMatches(const HtReplyDeviceCache& cache, const GraphSnapshot& graph) {
  if (cache.dimension != graph.dimension || cache.distance_type != graph.distance_type ||
      cache.host_x.size() != graph.points.size() || cache.host_y.size() != graph.points.size() ||
      cache.host_row_offsets != graph.row_offsets || cache.host_neighbors != graph.neighbors) {
    return false;
  }
  // 逐项比较 kernel 的完整输入，避免只凭对象地址或哈希复用过期活动图。
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    if (cache.host_x[index] != graph.points[index].integer_x ||
        cache.host_y[index] != graph.points[index].integer_y) {
      return false;
    }
  }
  return true;
}

bool PrepareGraph(HtReplyDeviceCache* const cache, const GraphSnapshot& graph) {
  if (GraphMatches(*cache, graph)) {
    return true;
  }

  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }
  std::vector<std::int32_t> host_row_offsets = graph.row_offsets;
  std::vector<std::int32_t> host_neighbors = graph.neighbors;
  DeviceBuffer<std::int64_t> device_x(host_x.size(), cache->device);
  DeviceBuffer<std::int64_t> device_y(host_y.size(), cache->device);
  DeviceBuffer<std::int32_t> device_row_offsets(host_row_offsets.size(), cache->device);
  DeviceBuffer<std::int32_t> device_neighbors(host_neighbors.size(), cache->device);
  device_x.CopyFromHost(host_x.data(), host_x.size());
  device_y.CopyFromHost(host_y.data(), host_y.size());
  device_row_offsets.CopyFromHost(host_row_offsets.data(), host_row_offsets.size());
  device_neighbors.CopyFromHost(host_neighbors.data(), host_neighbors.size());

  cache->device_x = std::move(device_x);
  cache->device_y = std::move(device_y);
  cache->device_row_offsets = std::move(device_row_offsets);
  cache->device_neighbors = std::move(device_neighbors);
  cache->host_x = std::move(host_x);
  cache->host_y = std::move(host_y);
  cache->host_row_offsets = std::move(host_row_offsets);
  cache->host_neighbors = std::move(host_neighbors);
  cache->dimension = graph.dimension;
  cache->distance_type = graph.distance_type;
  return false;
}

std::size_t GrowthCapacity(const std::size_t current, const std::size_t required) {
  if (current >= required) {
    return current;
  }
  if (current == 0U || current > std::numeric_limits<std::size_t>::max() / 2U) {
    return required;
  }
  return std::max(required, current * 2U);
}

bool PrepareCommonWorkspace(HtReplyDeviceCache* const cache, const std::size_t task_count) {
  if (task_count == std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("CUDA HT reply offsets 数量溢出");
  }
  const bool hit = cache->device_counts.count() >= task_count &&
                   cache->device_offsets.count() >= task_count + 1U &&
                   cache->device_error_code.count() >= 1U;
  if (cache->device_counts.count() < task_count) {
    cache->device_counts = DeviceBuffer<std::uint64_t>(
        GrowthCapacity(cache->device_counts.count(), task_count), cache->device);
  }
  if (cache->device_offsets.count() < task_count + 1U) {
    cache->device_offsets = DeviceBuffer<std::uint64_t>(
        GrowthCapacity(cache->device_offsets.count(), task_count + 1U), cache->device);
  }
  if (cache->device_error_code.count() < 1U) {
    cache->device_error_code = DeviceBuffer<std::uint32_t>(1U, cache->device);
  }
  return hit;
}

bool PrepareHamiltonInput(HtReplyDeviceCache* const cache, const std::size_t center_count) {
  const bool hit =
      PrepareCommonWorkspace(cache, center_count) && cache->device_centers.count() >= center_count;
  if (cache->device_centers.count() < center_count) {
    cache->device_centers = DeviceBuffer<std::int32_t>(
        GrowthCapacity(cache->device_centers.count(), center_count), cache->device);
  }
  return hit;
}

bool PrepareEndInput(HtReplyDeviceCache* const cache, const std::size_t task_count) {
  const bool hit =
      PrepareCommonWorkspace(cache, task_count) && cache->device_end_tasks.count() >= task_count;
  if (cache->device_end_tasks.count() < task_count) {
    cache->device_end_tasks = DeviceBuffer<HtEndReplyTask>(
        GrowthCapacity(cache->device_end_tasks.count(), task_count), cache->device);
  }
  return hit;
}

bool PrepareHamiltonOutput(HtReplyDeviceCache* const cache, const std::size_t reply_count) {
  const bool hit = cache->device_hamilton_replies.count() >= reply_count;
  if (!hit) {
    cache->device_hamilton_replies = DeviceBuffer<HtNeighborPair>(
        GrowthCapacity(cache->device_hamilton_replies.count(), reply_count), cache->device);
  }
  return hit;
}

bool PrepareEndOutput(HtReplyDeviceCache* const cache, const std::size_t reply_count) {
  const bool hit = cache->device_end_replies.count() >= reply_count;
  if (!hit) {
    cache->device_end_replies = DeviceBuffer<NodeEdge>(
        GrowthCapacity(cache->device_end_replies.count(), reply_count), cache->device);
  }
  return hit;
}

std::uint64_t ResidentBytes(const HtReplyDeviceCache& cache) {
  return cache.device_x.bytes() + cache.device_y.bytes() + cache.device_row_offsets.bytes() +
         cache.device_neighbors.bytes() + cache.device_centers.bytes() +
         cache.device_end_tasks.bytes() + cache.device_counts.bytes() +
         cache.device_offsets.bytes() + cache.device_hamilton_replies.bytes() +
         cache.device_end_replies.bytes() + cache.device_error_code.bytes();
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
                                                         int* const selected_device,
                                                         HtReplyCudaCacheUsage* const cache_usage) {
  if (cache_usage != nullptr) {
    *cache_usage = {};
  }
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

  HtReplyDeviceCache& cache = CacheForDevice(device);
  const bool graph_hit = PrepareGraph(&cache, graph);
  bool workspace_hit = PrepareHamiltonInput(&cache, centers.size());
  cache.device_centers.CopyFromHost(centers.data(), centers.size());

  CountHtRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      target_edge, cache.device_centers.get(), centers.size(), cache.device_row_offsets.get(),
      cache.device_neighbors.get(), cache.device_x.get(), cache.device_y.get(),
      static_cast<std::uint8_t>(graph.distance_type), cache.device_counts.get());
  CheckCuda(cudaGetLastError(), "CountHtRepliesKernel launch");
  std::vector<std::uint64_t> counts(centers.size());
  cache.device_counts.CopyToHost(counts.data(), counts.size());
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

  workspace_hit = PrepareHamiltonOutput(&cache, reply_count) && workspace_hit;
  cache.device_offsets.CopyFromHost(result.offsets.data(), result.offsets.size());
  CheckCuda(cudaMemset(cache.device_error_code.get(), 0, sizeof(std::uint32_t)),
            "cudaMemset(HT reply error_code)");
  WriteHtRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      target_edge, cache.device_centers.get(), centers.size(), cache.device_row_offsets.get(),
      cache.device_neighbors.get(), cache.device_x.get(), cache.device_y.get(),
      static_cast<std::uint8_t>(graph.distance_type), cache.device_offsets.get(),
      result.offsets.back(), cache.device_hamilton_replies.get(), cache.device_error_code.get());
  CheckCuda(cudaGetLastError(), "WriteHtRepliesKernel launch");
  std::uint32_t error_code = 0U;
  cache.device_error_code.CopyToHost(&error_code, 1U);
  if (error_code == 1U) {
    throw std::runtime_error("CUDA HT reply 写出越界");
  }
  if (error_code == 2U) {
    throw std::runtime_error("CUDA HT reply count/write 不一致");
  }
  cache.device_hamilton_replies.CopyToHost(result.replies.data(), result.replies.size());
  if (cache_usage != nullptr) {
    cache_usage->graph_hit = graph_hit;
    cache_usage->workspace_hit = workspace_hit;
    cache_usage->resident_bytes = ResidentBytes(cache);
  }
  return result;
}

bool HtEndReplyCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

HtEndReplyDeviceBatch EvaluateHtEndRepliesCuda(const GraphSnapshot& graph,
                                               const std::vector<HtEndReplyTask>& tasks,
                                               int* const selected_device,
                                               HtReplyCudaCacheUsage* const cache_usage) {
  if (cache_usage != nullptr) {
    *cache_usage = {};
  }
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

  HtReplyDeviceCache& cache = CacheForDevice(device);
  const bool graph_hit = PrepareGraph(&cache, graph);
  bool workspace_hit = PrepareEndInput(&cache, tasks.size());
  cache.device_end_tasks.CopyFromHost(tasks.data(), tasks.size());
  CheckCuda(cudaMemset(cache.device_error_code.get(), 0, sizeof(std::uint32_t)),
            "cudaMemset(HT end reply count error_code)");

  CountEndRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      cache.device_end_tasks.get(), tasks.size(), cache.device_row_offsets.get(),
      cache.device_neighbors.get(), cache.device_counts.get(), cache.device_error_code.get());
  CheckCuda(cudaGetLastError(), "CountEndRepliesKernel launch");
  std::vector<std::uint64_t> counts(tasks.size());
  cache.device_counts.CopyToHost(counts.data(), counts.size());
  std::uint32_t error_code = 0U;
  cache.device_error_code.CopyToHost(&error_code, 1U);
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

  workspace_hit = PrepareEndOutput(&cache, reply_count) && workspace_hit;
  cache.device_offsets.CopyFromHost(result.offsets.data(), result.offsets.size());
  CheckCuda(cudaMemset(cache.device_error_code.get(), 0, sizeof(std::uint32_t)),
            "cudaMemset(HT end reply write error_code)");
  WriteEndRepliesKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      cache.device_end_tasks.get(), tasks.size(), cache.device_row_offsets.get(),
      cache.device_neighbors.get(), cache.device_offsets.get(), result.offsets.back(),
      cache.device_end_replies.get(), cache.device_error_code.get());
  CheckCuda(cudaGetLastError(), "WriteEndRepliesKernel launch");
  cache.device_error_code.CopyToHost(&error_code, 1U);
  if (error_code == 2U) {
    throw std::runtime_error("CUDA HT end reply 写出越界");
  }
  if (error_code == 3U) {
    throw std::runtime_error("CUDA HT end reply count/write 不一致");
  }
  if (error_code != 0U) {
    throw std::runtime_error("CUDA HT end reply 未知设备错误");
  }
  cache.device_end_replies.CopyToHost(result.replies.data(), result.replies.size());
  if (cache_usage != nullptr) {
    cache_usage->graph_hit = graph_hit;
    cache_usage->workspace_hit = workspace_hit;
    cache_usage->resident_bytes = ResidentBytes(cache);
  }
  return result;
}

void ClearHtReplyCudaCache() {
  g_reply_device_caches.clear();
  g_reply_preferred_device = -1;
}

} // namespace cudaee::detail
