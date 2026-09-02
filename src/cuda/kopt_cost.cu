#include "cuda_edge_elimination/cuda_device_affinity.hpp"
#include "cuda_edge_elimination/local_search.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
        throw std::overflow_error("CUDA k-opt buffer 字节数溢出");
      }
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(k-opt buffer allocate)");
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(k-opt cost)");
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
      throw std::logic_error("CUDA k-opt H2D 超出驻留 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(k-opt H2D)");
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D(k-opt cost)");
    }
  }

  void CopyToHost(T* const destination, const std::size_t count) const {
    if (count > count_) {
      throw std::logic_error("CUDA k-opt D2H 超出驻留 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(k-opt D2H)");
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(k-opt cost)");
    }
  }

private:
  void Reset() noexcept {
    if (data_ != nullptr) {
      // 析构路径不能抛异常；owner device 防止释放另一个 CUDA context 的指针。
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

struct KOptTemplateDeviceCache {
  std::uint32_t k{};
  std::uint64_t generator_hash{};
  std::vector<EndpointMatching> host_templates;
  DeviceBuffer<EndpointMatching> device_templates;
};

struct KOptDeviceCache {
  explicit KOptDeviceCache(const int selected_device) : device(selected_device) {}

  int device{-1};
  std::int32_t dimension{};
  DistanceType distance_type{DistanceType::kEuc2D};
  std::vector<std::int64_t> host_x;
  std::vector<std::int64_t> host_y;
  DeviceBuffer<std::int64_t> device_x;
  DeviceBuffer<std::int64_t> device_y;
  std::array<KOptTemplateDeviceCache, 3U> templates;
  DeviceBuffer<KOptCostTask> device_tasks;
  DeviceBuffer<std::int64_t> device_costs;
};

thread_local std::vector<std::unique_ptr<KOptDeviceCache>> g_kopt_device_caches;
thread_local int g_kopt_preferred_device = -1;

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
  const int forced_device = CudaDevicePreferenceForCurrentThread();
  if (forced_device >= 0) {
    if (forced_device >= device_count) {
      if (reason != nullptr) {
        *reason = "k-opt 强制 CUDA device ordinal 超出当前可见范围";
      }
      return -1;
    }
    const cudaError_t select_status = cudaSetDevice(forced_device);
    if (select_status != cudaSuccess) {
      if (reason != nullptr) {
        *reason = std::string("cudaSetDevice(k-opt forced): ") + cudaGetErrorString(select_status);
      }
      return -1;
    }
    g_kopt_preferred_device = forced_device;
    return forced_device;
  }
  if (g_kopt_preferred_device >= 0 && g_kopt_preferred_device < device_count &&
      cudaSetDevice(g_kopt_preferred_device) == cudaSuccess) {
    return g_kopt_preferred_device;
  }
  g_kopt_preferred_device = -1;
  int best_device = -1;
  std::size_t best_free_bytes = 0;
  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
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
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(k-opt cost)");
  g_kopt_preferred_device = best_device;
  return best_device;
}

KOptDeviceCache& CacheForDevice(const int device) {
  const auto iterator = std::find_if(
      g_kopt_device_caches.begin(), g_kopt_device_caches.end(),
      [device](const std::unique_ptr<KOptDeviceCache>& cache) { return cache->device == device; });
  if (iterator != g_kopt_device_caches.end()) {
    return **iterator;
  }
  g_kopt_device_caches.push_back(std::make_unique<KOptDeviceCache>(device));
  return *g_kopt_device_caches.back();
}

bool SnapshotMatches(const KOptDeviceCache& cache, const GraphSnapshot& graph) {
  if (cache.dimension != graph.dimension || cache.distance_type != graph.distance_type ||
      cache.host_x.size() != graph.points.size() || cache.host_y.size() != graph.points.size()) {
    return false;
  }
  // kernel 不读取活动边；逐坐标比较其完整依赖既避免哈希碰撞，也不扫描无关边表。
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    if (cache.host_x[index] != graph.points[index].integer_x ||
        cache.host_y[index] != graph.points[index].integer_y) {
      return false;
    }
  }
  return true;
}

bool PrepareSnapshot(KOptDeviceCache* const cache, const GraphSnapshot& graph) {
  if (SnapshotMatches(*cache, graph)) {
    return true;
  }

  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }
  DeviceBuffer<std::int64_t> device_x(host_x.size(), cache->device);
  DeviceBuffer<std::int64_t> device_y(host_y.size(), cache->device);
  device_x.CopyFromHost(host_x.data(), host_x.size());
  device_y.CopyFromHost(host_y.data(), host_y.size());

  cache->device_x = std::move(device_x);
  cache->device_y = std::move(device_y);
  cache->host_x = std::move(host_x);
  cache->host_y = std::move(host_y);
  cache->dimension = graph.dimension;
  cache->distance_type = graph.distance_type;
  return false;
}

bool PrepareTemplates(KOptDeviceCache* const cache, const KOptReconnectTable& table) {
  KOptTemplateDeviceCache& entry = cache->templates[static_cast<std::size_t>(table.k - 3U)];
  if (entry.k == table.k && entry.generator_hash == table.generator_hash &&
      entry.host_templates == table.templates) {
    return true;
  }

  DeviceBuffer<EndpointMatching> device_templates(table.templates.size(), cache->device);
  device_templates.CopyFromHost(table.templates.data(), table.templates.size());
  entry.device_templates = std::move(device_templates);
  entry.host_templates = table.templates;
  entry.k = table.k;
  entry.generator_hash = table.generator_hash;
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

bool PrepareWorkspace(KOptDeviceCache* const cache, const std::size_t task_count,
                      const std::size_t cell_count) {
  const bool hit =
      cache->device_tasks.count() >= task_count && cache->device_costs.count() >= cell_count;
  if (cache->device_tasks.count() < task_count) {
    DeviceBuffer<KOptCostTask> tasks(GrowthCapacity(cache->device_tasks.count(), task_count),
                                     cache->device);
    cache->device_tasks = std::move(tasks);
  }
  if (cache->device_costs.count() < cell_count) {
    DeviceBuffer<std::int64_t> costs(GrowthCapacity(cache->device_costs.count(), cell_count),
                                     cache->device);
    cache->device_costs = std::move(costs);
  }
  return hit;
}

std::uint64_t ResidentBytes(const KOptDeviceCache& cache) {
  std::uint64_t bytes = cache.device_x.bytes() + cache.device_y.bytes() +
                        cache.device_tasks.bytes() + cache.device_costs.bytes();
  for (const KOptTemplateDeviceCache& entry : cache.templates) {
    bytes += entry.device_templates.bytes();
  }
  return bytes;
}

__device__ std::uint64_t IntegerSqrtFloorDevice(const std::uint64_t value) {
  std::uint64_t remainder = value;
  std::uint64_t root = 0;
  std::uint64_t bit = std::uint64_t{1} << 62;
  while (bit > remainder) {
    bit >>= 2;
  }
  while (bit != 0) {
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

__device__ std::int64_t ExactDistanceDevice(const std::int32_t first, const std::int32_t second,
                                            const std::int64_t* const x,
                                            const std::int64_t* const y,
                                            const std::uint8_t distance_type) {
  const std::uint64_t dx = AbsoluteDifference(x[first], x[second]);
  const std::uint64_t dy = AbsoluteDifference(y[first], y[second]);
  const std::uint64_t squared = dx * dx + dy * dy;
  const std::uint64_t root = IntegerSqrtFloorDevice(squared);
  std::uint64_t rounded = root;
  if (distance_type == static_cast<std::uint8_t>(DistanceType::kEuc2D)) {
    rounded += squared - root * root > root ? 1 : 0;
  } else {
    rounded += root * root != squared ? 1 : 0;
  }
  return static_cast<std::int64_t>(rounded);
}

__device__ void CanonicalEndpoints(const std::int32_t first, const std::int32_t second,
                                   std::int32_t* const u, std::int32_t* const v) {
  if (first < second) {
    *u = first;
    *v = second;
  } else {
    *u = second;
    *v = first;
  }
}

__global__ void KOptTemplateCostsKernel(const std::uint32_t k, const KOptCostTask* const tasks,
                                        const std::size_t task_count,
                                        const EndpointMatching* const templates,
                                        const std::uint32_t template_count,
                                        const std::int64_t* const x, const std::int64_t* const y,
                                        const std::uint8_t distance_type,
                                        std::int64_t* const costs) {
  const std::size_t flat_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t cell_count = task_count * template_count;
  if (flat_index >= cell_count) {
    return;
  }
  const std::size_t task_index = flat_index / template_count;
  const std::uint32_t template_index = static_cast<std::uint32_t>(flat_index % template_count);
  const KOptCostTask task = tasks[task_index];
  const EndpointMatching matching = templates[template_index];

  std::int32_t added_u[5];
  std::int32_t added_v[5];
  std::uint32_t added_count = 0;
  std::int64_t total = 0;
  for (std::uint32_t port = 0; port < 2U * k; ++port) {
    const std::uint32_t partner = matching.mate[port];
    if (port >= partner) {
      continue;
    }
    std::int32_t u = -1;
    std::int32_t v = -1;
    CanonicalEndpoints(task.port_nodes[port], task.port_nodes[partner], &u, &v);
    if (u == v) {
      costs[flat_index] = kInvalidKOptTemplateCost;
      return;
    }
    for (std::uint32_t edge = 0; edge < k; ++edge) {
      std::int32_t deleted_u = -1;
      std::int32_t deleted_v = -1;
      CanonicalEndpoints(task.port_nodes[2U * edge], task.port_nodes[2U * edge + 1U], &deleted_u,
                         &deleted_v);
      if (u == deleted_u && v == deleted_v) {
        costs[flat_index] = kInvalidKOptTemplateCost;
        return;
      }
    }
    for (std::uint32_t edge = 0; edge < added_count; ++edge) {
      if (u == added_u[edge] && v == added_v[edge]) {
        costs[flat_index] = kInvalidKOptTemplateCost;
        return;
      }
    }
    const std::int64_t distance = ExactDistanceDevice(u, v, x, y, distance_type);
    if (distance < 0 || total > LLONG_MAX - distance) {
      costs[flat_index] = kInvalidKOptTemplateCost;
      return;
    }
    added_u[added_count] = u;
    added_v[added_count] = v;
    ++added_count;
    total += distance;
  }
  costs[flat_index] = added_count == k ? total : kInvalidKOptTemplateCost;
}

} // namespace

bool KOptCostCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::int64_t> EvaluateKOptTemplateCostsCuda(const GraphSnapshot& graph,
                                                        const KOptReconnectTable& table,
                                                        const std::vector<KOptCostTask>& tasks,
                                                        int* const selected_device,
                                                        KOptCudaCacheUsage* const cache_usage) {
  if (cache_usage != nullptr) {
    *cache_usage = {};
  }
  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA k-opt cost 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  if (tasks.empty()) {
    return {};
  }
  if (table.k < 3U || table.k > 5U || table.templates.empty()) {
    throw std::invalid_argument("CUDA k-opt reconnect table 非法");
  }

  const std::size_t cell_count = tasks.size() * table.templates.size();
  constexpr std::size_t kThreads = 256;
  const std::size_t blocks = (cell_count + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA k-opt cost 网格过大");
  }

  KOptDeviceCache& cache = CacheForDevice(device);
  const bool snapshot_hit = PrepareSnapshot(&cache, graph);
  const bool template_hit = PrepareTemplates(&cache, table);
  const bool workspace_hit = PrepareWorkspace(&cache, tasks.size(), cell_count);
  cache.device_tasks.CopyFromHost(tasks.data(), tasks.size());

  KOptTemplateCostsKernel<<<static_cast<unsigned int>(blocks),
                            static_cast<unsigned int>(kThreads)>>>(
      table.k, cache.device_tasks.get(), tasks.size(),
      cache.templates[static_cast<std::size_t>(table.k - 3U)].device_templates.get(),
      static_cast<std::uint32_t>(table.templates.size()), cache.device_x.get(),
      cache.device_y.get(), static_cast<std::uint8_t>(graph.distance_type),
      cache.device_costs.get());
  CheckCuda(cudaGetLastError(), "KOptTemplateCostsKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "KOptTemplateCostsKernel synchronize");

  std::vector<std::int64_t> costs(cell_count);
  cache.device_costs.CopyToHost(costs.data(), costs.size());
  if (cache_usage != nullptr) {
    cache_usage->snapshot_hit = snapshot_hit;
    cache_usage->template_hit = template_hit;
    cache_usage->workspace_hit = workspace_hit;
    cache_usage->resident_bytes = ResidentBytes(cache);
  }
  return costs;
}

void ClearKOptCostCudaCache() { g_kopt_device_caches.clear(); }

} // namespace cudaee::detail
