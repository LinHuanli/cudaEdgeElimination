#include "cuda_edge_elimination/cuda_device_affinity.hpp"
#include "cuda_edge_elimination/local_search.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
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

constexpr std::uint32_t kExactInfinity = UINT32_MAX;

void CheckCuda(const cudaError_t status, const char* const operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const std::size_t count, const int device) : count_(count), device_(device) {
    if (count_ != 0U) {
      if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("CUDA exact DP buffer 字节数溢出");
      }
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(exact DP allocate)");
      CheckCuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc(exact DP)");
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
  void CopyFromHost(const T* source, const std::size_t count) {
    if (count > count_) {
      throw std::logic_error("CUDA exact DP H2D 超出 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(exact DP H2D)");
      CheckCuda(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy H2D(exact DP)");
    }
  }
  void CopyToHost(T* destination, const std::size_t count) const {
    if (count > count_) {
      throw std::logic_error("CUDA exact DP D2H 超出 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(exact DP D2H)");
      CheckCuda(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(exact DP)");
    }
  }

private:
  void Reset() noexcept {
    if (data_ != nullptr) {
      static_cast<void>(cudaSetDevice(device_));
      static_cast<void>(cudaFree(data_));
    }
    data_ = nullptr;
    count_ = 0U;
    device_ = -1;
  }
  void MoveFrom(DeviceBuffer* other) noexcept {
    data_ = other->data_;
    count_ = other->count_;
    device_ = other->device_;
    other->data_ = nullptr;
    other->count_ = 0U;
    other->device_ = -1;
  }
  T* data_{nullptr};
  std::size_t count_{};
  int device_{-1};
};

struct MaskTableCache {
  std::uint32_t block_count{};
  std::vector<std::uint16_t> masks;
  std::vector<std::uint16_t> ranks;
  std::array<std::uint32_t, kExactTourCudaMaxBlocks + 2U> layer_offsets{};
  std::size_t max_layer_cells{};
  DeviceBuffer<std::uint16_t> device_masks;
  DeviceBuffer<std::uint16_t> device_ranks;
  DeviceBuffer<std::uint32_t> device_layer_offsets;
};

struct ExactDpDeviceCache {
  explicit ExactDpDeviceCache(const int ordinal) : device(ordinal) {}
  int device{-1};
  std::int32_t dimension{};
  DistanceType distance_type{DistanceType::kEuc2D};
  std::vector<std::int64_t> host_x;
  std::vector<std::int64_t> host_y;
  DeviceBuffer<std::int64_t> device_x;
  DeviceBuffer<std::int64_t> device_y;
  std::array<MaskTableCache, kExactTourCudaMaxBlocks + 1U> mask_tables;
  DeviceBuffer<ExactTourCostTask> device_tasks;
  DeviceBuffer<std::uint32_t> device_costs;
};

thread_local std::vector<std::unique_ptr<ExactDpDeviceCache>> g_exact_dp_caches;
thread_local int g_exact_preferred_device = -1;

int SelectDevice(std::string* const reason) {
  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status != cudaSuccess || count == 0) {
    if (reason != nullptr) {
      *reason =
          count_status == cudaSuccess ? "没有可见 CUDA 设备" : cudaGetErrorString(count_status);
    }
    return -1;
  }
  const int forced = CudaDevicePreferenceForCurrentThread();
  if (forced >= 0) {
    if (forced >= count || cudaSetDevice(forced) != cudaSuccess) {
      if (reason != nullptr) {
        *reason = "exact DP 强制 CUDA device ordinal 不可用";
      }
      return -1;
    }
    g_exact_preferred_device = forced;
    return forced;
  }
  if (g_exact_preferred_device >= 0 && g_exact_preferred_device < count &&
      cudaSetDevice(g_exact_preferred_device) == cudaSuccess) {
    return g_exact_preferred_device;
  }
  int best = -1;
  std::size_t best_free = 0U;
  for (int device = 0; device < count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        (best < 0 || free_bytes > best_free)) {
      best = device;
      best_free = free_bytes;
    }
  }
  if (best < 0) {
    if (reason != nullptr) {
      *reason = "所有可见 GPU 均无法查询显存";
    }
    return -1;
  }
  CheckCuda(cudaSetDevice(best), "cudaSetDevice(exact DP)");
  g_exact_preferred_device = best;
  return best;
}

ExactDpDeviceCache& CacheForDevice(const int device) {
  const auto found = std::find_if(g_exact_dp_caches.begin(), g_exact_dp_caches.end(),
                                  [device](const std::unique_ptr<ExactDpDeviceCache>& cache) {
                                    return cache->device == device;
                                  });
  if (found != g_exact_dp_caches.end()) {
    return **found;
  }
  g_exact_dp_caches.push_back(std::make_unique<ExactDpDeviceCache>(device));
  return *g_exact_dp_caches.back();
}

bool SnapshotMatches(const ExactDpDeviceCache& cache, const GraphSnapshot& graph) {
  if (cache.dimension != graph.dimension || cache.distance_type != graph.distance_type ||
      cache.host_x.size() != graph.points.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    if (cache.host_x[index] != graph.points[index].integer_x ||
        cache.host_y[index] != graph.points[index].integer_y) {
      return false;
    }
  }
  return true;
}

void PrepareSnapshot(ExactDpDeviceCache* const cache, const GraphSnapshot& graph) {
  if (SnapshotMatches(*cache, graph)) {
    return;
  }
  std::vector<std::int64_t> x(graph.points.size());
  std::vector<std::int64_t> y(graph.points.size());
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    x[index] = graph.points[index].integer_x;
    y[index] = graph.points[index].integer_y;
  }
  DeviceBuffer<std::int64_t> device_x(x.size(), cache->device);
  DeviceBuffer<std::int64_t> device_y(y.size(), cache->device);
  device_x.CopyFromHost(x.data(), x.size());
  device_y.CopyFromHost(y.data(), y.size());
  cache->host_x = std::move(x);
  cache->host_y = std::move(y);
  cache->device_x = std::move(device_x);
  cache->device_y = std::move(device_y);
  cache->dimension = graph.dimension;
  cache->distance_type = graph.distance_type;
}

void PrepareMaskTable(ExactDpDeviceCache* const cache, const std::uint32_t block_count) {
  MaskTableCache& table = cache->mask_tables[block_count];
  if (table.block_count == block_count) {
    return;
  }
  const std::uint32_t mask_count = 1U << block_count;
  table.ranks.assign(mask_count, UINT16_MAX);
  table.masks.clear();
  table.max_layer_cells = 1U;
  for (std::uint32_t layer = 1U; layer <= block_count; ++layer) {
    table.layer_offsets[layer] = static_cast<std::uint32_t>(table.masks.size());
    std::uint32_t rank = 0U;
    for (std::uint32_t mask = 1U; mask < mask_count; mask += 2U) {
      if (std::popcount(mask) != static_cast<int>(layer)) {
        continue;
      }
      if (rank > UINT16_MAX || mask > UINT16_MAX) {
        throw std::overflow_error("CUDA exact DP mask/rank 超过 uint16 范围");
      }
      table.ranks[mask] = static_cast<std::uint16_t>(rank++);
      table.masks.push_back(static_cast<std::uint16_t>(mask));
    }
    const std::size_t cells = layer == 1U ? 1U : static_cast<std::size_t>(rank) * 2U * (layer - 1U);
    table.max_layer_cells = std::max(table.max_layer_cells, cells);
  }
  table.layer_offsets[block_count + 1U] = static_cast<std::uint32_t>(table.masks.size());
  DeviceBuffer<std::uint16_t> masks(table.masks.size(), cache->device);
  DeviceBuffer<std::uint16_t> ranks(table.ranks.size(), cache->device);
  DeviceBuffer<std::uint32_t> offsets(table.layer_offsets.size(), cache->device);
  masks.CopyFromHost(table.masks.data(), table.masks.size());
  ranks.CopyFromHost(table.ranks.data(), table.ranks.size());
  offsets.CopyFromHost(table.layer_offsets.data(), table.layer_offsets.size());
  table.device_masks = std::move(masks);
  table.device_ranks = std::move(ranks);
  table.device_layer_offsets = std::move(offsets);
  table.block_count = block_count;
}

void PrepareWorkspace(ExactDpDeviceCache* const cache, const std::size_t task_count) {
  if (cache->device_tasks.count() < task_count) {
    cache->device_tasks = DeviceBuffer<ExactTourCostTask>(task_count, cache->device);
  }
  if (cache->device_costs.count() < task_count) {
    cache->device_costs = DeviceBuffer<std::uint32_t>(task_count, cache->device);
  }
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
      root = (root >> 1U) + bit;
    } else {
      root >>= 1U;
    }
    bit >>= 2U;
  }
  return root;
}

__device__ std::uint64_t AbsoluteDifference(const std::int64_t first, const std::int64_t second) {
  return first >= second ? static_cast<std::uint64_t>(first - second)
                         : static_cast<std::uint64_t>(second - first);
}

__device__ std::uint32_t ExactDistanceDevice(const std::int32_t first, const std::int32_t second,
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
  return rounded >= kExactInfinity ? kExactInfinity : static_cast<std::uint32_t>(rounded);
}

__device__ bool IsForbidden(const ExactTourCostTask& task, const std::int32_t first,
                            const std::int32_t second) {
  if (task.forbidden.u < 0) {
    return false;
  }
  const std::int32_t u = min(first, second);
  const std::int32_t v = max(first, second);
  return u == task.forbidden.u && v == task.forbidden.v;
}

__device__ std::int32_t BlockEntry(const ExactTourCostTask& task, const std::uint32_t block,
                                   const std::uint32_t orientation) {
  return orientation == 0U ? task.first[block] : task.second[block];
}

__device__ std::int32_t BlockExit(const ExactTourCostTask& task, const std::uint32_t block,
                                  const std::uint32_t orientation) {
  if (task.paired[block] == 0U) {
    return task.first[block];
  }
  return orientation == 0U ? task.second[block] : task.first[block];
}

__device__ std::uint32_t NthBlock(const std::uint32_t mask, const std::uint32_t ordinal) {
  std::uint32_t seen = 0U;
  for (std::uint32_t block = 1U; block < kExactTourCudaMaxBlocks; ++block) {
    if ((mask & (1U << block)) == 0U) {
      continue;
    }
    if (seen++ == ordinal) {
      return block;
    }
  }
  return kExactTourCudaMaxBlocks;
}

__global__ void ExactTourValueKernel(const ExactTourCostTask* const tasks,
                                     const std::uint16_t* const masks,
                                     const std::uint16_t* const ranks,
                                     const std::uint32_t* const layer_offsets,
                                     const std::size_t max_layer_cells, const std::int64_t* const x,
                                     const std::int64_t* const y, const std::uint8_t distance_type,
                                     std::uint32_t* const output) {
  const std::size_t task_index = blockIdx.x;
  const ExactTourCostTask task = tasks[task_index];
  extern __shared__ std::uint32_t values[];
  std::uint32_t* current = values;
  std::uint32_t* next = values + max_layer_cells;
  if (threadIdx.x == 0U) {
    current[0] = 0U;
  }
  __syncthreads();

  for (std::uint32_t layer = 2U; layer <= task.block_count; ++layer) {
    const std::uint32_t mask_begin = layer_offsets[layer];
    const std::uint32_t mask_end = layer_offsets[layer + 1U];
    const std::uint32_t cells_per_mask = 2U * (layer - 1U);
    const std::size_t cell_count = static_cast<std::size_t>(mask_end - mask_begin) * cells_per_mask;
    for (std::size_t flat = threadIdx.x; flat < cell_count; flat += blockDim.x) {
      const std::uint32_t mask_rank = static_cast<std::uint32_t>(flat / cells_per_mask);
      const std::uint32_t local = static_cast<std::uint32_t>(flat % cells_per_mask);
      const std::uint32_t end_ordinal = local / 2U;
      const std::uint32_t orientation = local % 2U;
      const std::uint32_t mask = masks[mask_begin + mask_rank];
      const std::uint32_t end_block = NthBlock(mask, end_ordinal);
      std::uint32_t best = kExactInfinity;
      if (end_block < task.block_count && (orientation == 0U || task.paired[end_block] != 0U)) {
        const std::int32_t to = BlockEntry(task, end_block, orientation);
        const std::uint32_t previous_mask = mask ^ (1U << end_block);
        if (layer == 2U) {
          const std::int32_t from = BlockExit(task, 0U, 0U);
          if (!IsForbidden(task, from, to)) {
            best = ExactDistanceDevice(from, to, x, y, distance_type);
          }
        } else {
          const std::uint32_t previous_rank = ranks[previous_mask];
          const std::uint32_t previous_cells_per_mask = 2U * (layer - 2U);
          std::uint32_t previous_ordinal = 0U;
          for (std::uint32_t previous_block = 1U; previous_block < task.block_count;
               ++previous_block) {
            if ((previous_mask & (1U << previous_block)) == 0U) {
              continue;
            }
            const std::uint32_t orientation_count = task.paired[previous_block] != 0U ? 2U : 1U;
            for (std::uint32_t previous_orientation = 0U; previous_orientation < orientation_count;
                 ++previous_orientation) {
              const std::size_t previous_cell =
                  static_cast<std::size_t>(previous_rank) * previous_cells_per_mask +
                  2U * previous_ordinal + previous_orientation;
              const std::uint32_t prefix = current[previous_cell];
              const std::int32_t from = BlockExit(task, previous_block, previous_orientation);
              if (prefix == kExactInfinity || IsForbidden(task, from, to)) {
                continue;
              }
              const std::uint32_t distance = ExactDistanceDevice(from, to, x, y, distance_type);
              if (distance != kExactInfinity && prefix <= kExactInfinity - 1U - distance) {
                best = min(best, prefix + distance);
              }
            }
            ++previous_ordinal;
          }
        }
      }
      next[flat] = best;
    }
    __syncthreads();
    std::uint32_t* temporary = current;
    current = next;
    next = temporary;
    __syncthreads();
  }

  if (threadIdx.x == 0U) {
    std::uint32_t best = kExactInfinity;
    const std::uint32_t cells_per_mask = 2U * (task.block_count - 1U);
    const std::int32_t start = BlockEntry(task, 0U, 0U);
    for (std::uint32_t local = 0U; local < cells_per_mask; ++local) {
      const std::uint32_t end_ordinal = local / 2U;
      const std::uint32_t orientation = local % 2U;
      const std::uint32_t end_block = NthBlock((1U << task.block_count) - 1U, end_ordinal);
      if (orientation != 0U && task.paired[end_block] == 0U) {
        continue;
      }
      const std::uint32_t prefix = current[local];
      const std::int32_t from = BlockExit(task, end_block, orientation);
      if (prefix == kExactInfinity || IsForbidden(task, from, start)) {
        continue;
      }
      const std::uint32_t distance = ExactDistanceDevice(from, start, x, y, distance_type);
      if (distance != kExactInfinity && prefix <= kExactInfinity - 1U - distance) {
        best = min(best, prefix + distance);
      }
    }
    output[task_index] = best;
  }
}

} // namespace

bool ExactTourCostCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::int64_t> EvaluateExactTourCostsCuda(const GraphSnapshot& graph,
                                                     const std::vector<ExactTourCostTask>& tasks,
                                                     int* const selected_device,
                                                     std::uint64_t* const shared_memory_bytes) {
  if (tasks.empty()) {
    if (shared_memory_bytes != nullptr) {
      *shared_memory_bytes = 0U;
    }
    return {};
  }
  const std::uint32_t block_count = tasks.front().block_count;
  if (block_count < 2U || block_count > kExactTourCudaMaxBlocks ||
      std::any_of(tasks.begin(), tasks.end(), [block_count](const ExactTourCostTask& task) {
        return task.block_count != block_count;
      })) {
    throw std::invalid_argument("CUDA exact DP tasks 必须具有相同且位于 [2,13] 的 block_count");
  }
  if (graph.distance_type != DistanceType::kEuc2D && graph.distance_type != DistanceType::kCeil2D) {
    throw std::invalid_argument("CUDA exact DP 只支持 EUC_2D 与 CEIL_2D");
  }
  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA exact DP 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  if (tasks.size() > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA exact DP task 网格过大");
  }

  ExactDpDeviceCache& cache = CacheForDevice(device);
  PrepareSnapshot(&cache, graph);
  PrepareMaskTable(&cache, block_count);
  PrepareWorkspace(&cache, tasks.size());
  MaskTableCache& table = cache.mask_tables[block_count];
  const std::size_t shared_bytes = 2U * table.max_layer_cells * sizeof(std::uint32_t);
  int max_optin = 0;
  CheckCuda(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
            "cudaDeviceGetAttribute(max opt-in shared memory)");
  if (shared_bytes > static_cast<std::size_t>(max_optin)) {
    throw std::runtime_error("CUDA exact DP 所需 shared memory 超过设备 opt-in 上限");
  }
  if (shared_bytes > 48U * 1024U) {
    // function attribute 是 device 全局状态；共享同一 GPU 的 target workers 必须写入
    // 相同的设备上限，不能按各自 block_count 写不同值而产生 launch 竞态。
    CheckCuda(cudaFuncSetAttribute(ExactTourValueKernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin),
              "cudaFuncSetAttribute(exact DP shared memory)");
  }
  cache.device_tasks.CopyFromHost(tasks.data(), tasks.size());
  constexpr unsigned int kThreads = 256U;
  ExactTourValueKernel<<<static_cast<unsigned int>(tasks.size()), kThreads, shared_bytes>>>(
      cache.device_tasks.get(), table.device_masks.get(), table.device_ranks.get(),
      table.device_layer_offsets.get(), table.max_layer_cells, cache.device_x.get(),
      cache.device_y.get(), static_cast<std::uint8_t>(graph.distance_type),
      cache.device_costs.get());
  CheckCuda(cudaGetLastError(), "ExactTourValueKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "ExactTourValueKernel synchronize");

  std::vector<std::uint32_t> packed(tasks.size());
  cache.device_costs.CopyToHost(packed.data(), packed.size());
  std::vector<std::int64_t> costs(tasks.size(), std::numeric_limits<std::int64_t>::max());
  for (std::size_t index = 0U; index < packed.size(); ++index) {
    if (packed[index] != kExactInfinity) {
      costs[index] = packed[index];
    }
  }
  if (shared_memory_bytes != nullptr) {
    *shared_memory_bytes = shared_bytes;
  }
  return costs;
}

void ClearExactTourCostCudaCache() { g_exact_dp_caches.clear(); }

} // namespace cudaee::detail
