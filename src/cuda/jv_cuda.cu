#include "cuda_edge_elimination/elimination.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr int kMaxCandidateNodes = 10;

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
        throw std::overflow_error("CUDA JV buffer 字节数溢出");
      }
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(JV buffer allocate)");
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc");
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
      throw std::logic_error("CUDA JV H2D 超出驻留 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(JV H2D)");
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D");
    }
  }

  void CopyToHost(T* const destination, const std::size_t count) const {
    if (count > count_) {
      throw std::logic_error("CUDA JV D2H 超出驻留 buffer");
    }
    if (count != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(JV D2H)");
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H");
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

struct JvDeviceCache {
  explicit JvDeviceCache(const int selected_device) : device(selected_device) {}

  int device{-1};
  std::int32_t dimension{};
  DistanceType distance_type{DistanceType::kEuc2D};
  std::vector<std::int32_t> host_edge_u;
  std::vector<std::int32_t> host_edge_v;
  std::vector<std::int64_t> host_edge_weight;
  std::vector<std::int64_t> host_x;
  std::vector<std::int64_t> host_y;
  std::vector<std::int32_t> host_edge_active;
  std::vector<std::int32_t> host_witnesses;
  DeviceBuffer<std::int32_t> device_edge_u;
  DeviceBuffer<std::int32_t> device_edge_v;
  DeviceBuffer<std::int64_t> device_edge_weight;
  DeviceBuffer<std::int64_t> device_x;
  DeviceBuffer<std::int64_t> device_y;
  DeviceBuffer<std::int32_t> device_edge_active;
  DeviceBuffer<std::int32_t> device_row_offsets;
  DeviceBuffer<std::int32_t> device_neighbors;
  DeviceBuffer<std::int32_t> device_csr_edge_ids;
  DeviceBuffer<std::int32_t> device_witnesses;
};

thread_local std::vector<std::unique_ptr<JvDeviceCache>> g_jv_device_caches;
thread_local int g_jv_preferred_device = -1;

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

__device__ std::uint64_t AbsoluteDifference(const std::int64_t lhs, const std::int64_t rhs) {
  return lhs >= rhs ? static_cast<std::uint64_t>(lhs - rhs) : static_cast<std::uint64_t>(rhs - lhs);
}

__device__ std::int64_t ExactDistanceDevice(const std::int32_t a, const std::int32_t b,
                                            const std::int64_t* const x,
                                            const std::int64_t* const y,
                                            const std::uint8_t distance_type) {
  const std::uint64_t dx = AbsoluteDifference(x[a], x[b]);
  const std::uint64_t dy = AbsoluteDifference(y[a], y[b]);
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

__device__ bool RowContains(const std::int32_t* const row_offsets,
                            const std::int32_t* const neighbors, const std::int32_t row,
                            const std::int32_t needle) {
  std::int32_t low = row_offsets[row];
  std::int32_t high = row_offsets[row + 1];
  while (low < high) {
    const std::int32_t middle = low + (high - low) / 2;
    if (neighbors[middle] < needle) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < row_offsets[row + 1] && neighbors[low] == needle;
}

__device__ bool ScoreLess(const std::uint64_t lhs_score, const std::int32_t lhs_node,
                          const std::uint64_t rhs_score, const std::int32_t rhs_node) {
  return lhs_score < rhs_score || (lhs_score == rhs_score && lhs_node < rhs_node);
}

__device__ void InsertCandidate(const std::int32_t node, const std::uint64_t score,
                                std::int32_t* const nodes, std::uint64_t* const scores,
                                std::int32_t* const count) {
  if (*count == kMaxCandidateNodes &&
      !ScoreLess(score, node, scores[kMaxCandidateNodes - 1], nodes[kMaxCandidateNodes - 1])) {
    return;
  }
  std::int32_t position = *count < kMaxCandidateNodes ? (*count)++ : kMaxCandidateNodes - 1;
  while (position > 0 && ScoreLess(score, node, scores[position - 1], nodes[position - 1])) {
    if (position < kMaxCandidateNodes) {
      scores[position] = scores[position - 1];
      nodes[position] = nodes[position - 1];
    }
    --position;
  }
  scores[position] = score;
  nodes[position] = node;
}

__global__ void
JvCandidatesKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                   const std::int32_t* const edge_v, const std::int64_t* const edge_weight,
                   const std::int32_t* const edge_active, const std::int32_t* const row_offsets,
                   const std::int32_t* const neighbors, const std::int32_t* const csr_edge_ids,
                   const std::int64_t* const x, const std::int64_t* const y,
                   const std::uint8_t distance_type, std::int32_t* const witnesses) {
  const std::int32_t edge_id = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge_id >= edge_count || edge_active[edge_id] == 0) {
    return;
  }
  const std::int32_t a = edge_u[edge_id];
  const std::int32_t b = edge_v[edge_id];
  if (row_offsets[a + 1] - row_offsets[a] <= 2 || row_offsets[b + 1] - row_offsets[b] <= 2) {
    return;
  }

  std::int32_t candidate_nodes[kMaxCandidateNodes];
  std::uint64_t candidate_scores[kMaxCandidateNodes];
  std::int32_t candidate_count = 0;
  for (int side = 0; side < 2; ++side) {
    const std::int32_t from = side == 0 ? a : b;
    const std::int32_t other = side == 0 ? b : a;
    for (std::int32_t offset = row_offsets[from]; offset < row_offsets[from + 1]; ++offset) {
      const std::int32_t candidate = neighbors[offset];
      if (candidate == a || candidate == b ||
          (side == 1 && RowContains(row_offsets, neighbors, a, candidate))) {
        continue;
      }
      const std::uint64_t score =
          static_cast<std::uint64_t>(edge_weight[csr_edge_ids[offset]]) +
          static_cast<std::uint64_t>(ExactDistanceDevice(candidate, other, x, y, distance_type));
      if (score <= static_cast<std::uint64_t>(LLONG_MAX)) {
        InsertCandidate(candidate, score, candidate_nodes, candidate_scores, &candidate_count);
      }
    }
  }

  const std::uint64_t cab = static_cast<std::uint64_t>(edge_weight[edge_id]);
  for (std::int32_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
    const std::int32_t c = candidate_nodes[candidate_index];
    const std::uint64_t cac =
        static_cast<std::uint64_t>(ExactDistanceDevice(a, c, x, y, distance_type));
    const std::uint64_t cbc =
        static_cast<std::uint64_t>(ExactDistanceDevice(b, c, x, y, distance_type));
    bool compatible = false;
    for (std::int32_t offset = row_offsets[c]; offset < row_offsets[c + 1]; ++offset) {
      const std::int32_t d = neighbors[offset];
      if (d == a || d == b) {
        continue;
      }
      const std::uint64_t left =
          cab + static_cast<std::uint64_t>(edge_weight[csr_edge_ids[offset]]);
      const std::uint64_t first =
          cac + static_cast<std::uint64_t>(ExactDistanceDevice(d, b, x, y, distance_type));
      const std::uint64_t second =
          static_cast<std::uint64_t>(ExactDistanceDevice(a, d, x, y, distance_type)) + cbc;
      if (left <= first || left <= second) {
        compatible = true;
        break;
      }
    }
    if (!compatible) {
      witnesses[edge_id] = c;
      return;
    }
  }
}

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
  if (g_jv_preferred_device >= 0 && g_jv_preferred_device < device_count &&
      cudaSetDevice(g_jv_preferred_device) == cudaSuccess) {
    return g_jv_preferred_device;
  }
  g_jv_preferred_device = -1;

  int best_device = -1;
  std::size_t best_free = 0;
  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        (best_device < 0 || free_bytes > best_free)) {
      best_device = device;
      best_free = free_bytes;
    }
  }
  if (best_device < 0) {
    if (reason != nullptr) {
      *reason = "所有可见 GPU 均无法查询显存";
    }
    return -1;
  }
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice");
  g_jv_preferred_device = best_device;
  return best_device;
}

JvDeviceCache& CacheForDevice(const int device) {
  const auto iterator = std::find_if(
      g_jv_device_caches.begin(), g_jv_device_caches.end(),
      [device](const std::unique_ptr<JvDeviceCache>& cache) { return cache->device == device; });
  if (iterator != g_jv_device_caches.end()) {
    return **iterator;
  }
  g_jv_device_caches.push_back(std::make_unique<JvDeviceCache>(device));
  return *g_jv_device_caches.back();
}

bool StaticGraphMatches(const JvDeviceCache& cache, const GraphSnapshot& graph) {
  if (cache.dimension != graph.dimension || cache.distance_type != graph.distance_type ||
      cache.host_x.size() != graph.points.size() || cache.host_y.size() != graph.points.size() ||
      cache.host_edge_u.size() != graph.edges.size() ||
      cache.host_edge_v.size() != graph.edges.size() ||
      cache.host_edge_weight.size() != graph.edges.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    if (cache.host_x[index] != graph.points[index].integer_x ||
        cache.host_y[index] != graph.points[index].integer_y) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < graph.edges.size(); ++index) {
    if (cache.host_edge_u[index] != graph.edges[index].u ||
        cache.host_edge_v[index] != graph.edges[index].v ||
        cache.host_edge_weight[index] != graph.edges[index].weight) {
      return false;
    }
  }
  return true;
}

bool PrepareStaticGraph(JvDeviceCache* const cache, const GraphSnapshot& graph) {
  // active 位和 CSR 每个 epoch 都会变化；这里只缓存 kernel 真正不变的完整依赖，
  // 并逐元素比较而非只信任哈希，防止跨实例错误复用设备数据。
  if (StaticGraphMatches(*cache, graph)) {
    return true;
  }

  std::vector<std::int32_t> host_edge_u(graph.edges.size());
  std::vector<std::int32_t> host_edge_v(graph.edges.size());
  std::vector<std::int64_t> host_edge_weight(graph.edges.size());
  for (std::size_t index = 0U; index < graph.edges.size(); ++index) {
    host_edge_u[index] = graph.edges[index].u;
    host_edge_v[index] = graph.edges[index].v;
    host_edge_weight[index] = graph.edges[index].weight;
  }
  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  for (std::size_t index = 0U; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }

  DeviceBuffer<std::int32_t> device_edge_u(host_edge_u.size(), cache->device);
  DeviceBuffer<std::int32_t> device_edge_v(host_edge_v.size(), cache->device);
  DeviceBuffer<std::int64_t> device_edge_weight(host_edge_weight.size(), cache->device);
  DeviceBuffer<std::int64_t> device_x(host_x.size(), cache->device);
  DeviceBuffer<std::int64_t> device_y(host_y.size(), cache->device);
  device_edge_u.CopyFromHost(host_edge_u.data(), host_edge_u.size());
  device_edge_v.CopyFromHost(host_edge_v.data(), host_edge_v.size());
  device_edge_weight.CopyFromHost(host_edge_weight.data(), host_edge_weight.size());
  device_x.CopyFromHost(host_x.data(), host_x.size());
  device_y.CopyFromHost(host_y.data(), host_y.size());

  cache->device_edge_u = std::move(device_edge_u);
  cache->device_edge_v = std::move(device_edge_v);
  cache->device_edge_weight = std::move(device_edge_weight);
  cache->device_x = std::move(device_x);
  cache->device_y = std::move(device_y);
  cache->host_edge_u = std::move(host_edge_u);
  cache->host_edge_v = std::move(host_edge_v);
  cache->host_edge_weight = std::move(host_edge_weight);
  cache->host_x = std::move(host_x);
  cache->host_y = std::move(host_y);
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

bool PrepareWorkspace(JvDeviceCache* const cache, const GraphSnapshot& graph) {
  const bool hit = cache->device_edge_active.count() >= graph.edges.size() &&
                   cache->device_row_offsets.count() >= graph.row_offsets.size() &&
                   cache->device_neighbors.count() >= graph.neighbors.size() &&
                   cache->device_csr_edge_ids.count() >= graph.csr_edge_ids.size() &&
                   cache->device_witnesses.count() >= graph.edges.size();
  if (cache->device_edge_active.count() < graph.edges.size()) {
    cache->device_edge_active = DeviceBuffer<std::int32_t>(
        GrowthCapacity(cache->device_edge_active.count(), graph.edges.size()), cache->device);
  }
  if (cache->device_row_offsets.count() < graph.row_offsets.size()) {
    cache->device_row_offsets = DeviceBuffer<std::int32_t>(
        GrowthCapacity(cache->device_row_offsets.count(), graph.row_offsets.size()), cache->device);
  }
  if (cache->device_neighbors.count() < graph.neighbors.size()) {
    cache->device_neighbors = DeviceBuffer<std::int32_t>(
        GrowthCapacity(cache->device_neighbors.count(), graph.neighbors.size()), cache->device);
  }
  if (cache->device_csr_edge_ids.count() < graph.csr_edge_ids.size()) {
    cache->device_csr_edge_ids = DeviceBuffer<std::int32_t>(
        GrowthCapacity(cache->device_csr_edge_ids.count(), graph.csr_edge_ids.size()),
        cache->device);
  }
  if (cache->device_witnesses.count() < graph.edges.size()) {
    cache->device_witnesses = DeviceBuffer<std::int32_t>(
        GrowthCapacity(cache->device_witnesses.count(), graph.edges.size()), cache->device);
  }
  return hit;
}

std::uint64_t ResidentBytes(const JvDeviceCache& cache) {
  return cache.device_edge_u.bytes() + cache.device_edge_v.bytes() +
         cache.device_edge_weight.bytes() + cache.device_x.bytes() + cache.device_y.bytes() +
         cache.device_edge_active.bytes() + cache.device_row_offsets.bytes() +
         cache.device_neighbors.bytes() + cache.device_csr_edge_ids.bytes() +
         cache.device_witnesses.bytes();
}

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

} // namespace

bool CudaBackendAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<Candidate> FindJvCandidatesCuda(const GraphSnapshot& graph, int* const selected_device,
                                            JvCudaCacheUsage* const cache_usage) {
  if (cache_usage != nullptr) {
    *cache_usage = {};
  }
  if (!graph.integer_coordinates || !graph.integer_distance_safe) {
    throw std::runtime_error("CUDA JV 仅支持平方距离不溢出的整数坐标");
  }
  if (graph.edges.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("CUDA JV 边数超过 int32 kernel 索引范围");
  }
  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA 设备不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  if (graph.edges.empty()) {
    return {};
  }

  JvDeviceCache& cache = CacheForDevice(device);
  const bool static_hit = PrepareStaticGraph(&cache, graph);
  const bool workspace_hit = PrepareWorkspace(&cache, graph);
  cache.host_edge_active.resize(graph.edges.size());
  for (std::size_t index = 0U; index < graph.edges.size(); ++index) {
    cache.host_edge_active[index] = graph.edges[index].active ? 1 : 0;
  }
  cache.host_witnesses.assign(graph.edges.size(), -1);

  // 动态输入每轮完整覆盖。CSR 只传稳定 edge id，权重从已驻留的静态表读取；
  // 这不会改变 snapshot 或 CPU verifier 的语义。
  const auto h2d_start = std::chrono::steady_clock::now();
  cache.device_edge_active.CopyFromHost(cache.host_edge_active.data(),
                                        cache.host_edge_active.size());
  cache.device_row_offsets.CopyFromHost(graph.row_offsets.data(), graph.row_offsets.size());
  cache.device_neighbors.CopyFromHost(graph.neighbors.data(), graph.neighbors.size());
  cache.device_csr_edge_ids.CopyFromHost(graph.csr_edge_ids.data(), graph.csr_edge_ids.size());
  cache.device_witnesses.CopyFromHost(cache.host_witnesses.data(), cache.host_witnesses.size());
  const double h2d_ms = ElapsedMilliseconds(h2d_start);
  if (cache_usage != nullptr) {
    cache_usage->static_hit = static_hit;
    cache_usage->workspace_hit = workspace_hit;
    cache_usage->resident_bytes = ResidentBytes(cache);
    cache_usage->h2d_ms = h2d_ms;
  }

  constexpr int kThreads = 128;
  const int blocks = (static_cast<int>(graph.edges.size()) + kThreads - 1) / kThreads;
  const auto kernel_start = std::chrono::steady_clock::now();
  JvCandidatesKernel<<<blocks, kThreads>>>(
      static_cast<std::int32_t>(graph.edges.size()), cache.device_edge_u.get(),
      cache.device_edge_v.get(), cache.device_edge_weight.get(), cache.device_edge_active.get(),
      cache.device_row_offsets.get(), cache.device_neighbors.get(),
      cache.device_csr_edge_ids.get(),
      cache.device_x.get(), cache.device_y.get(), static_cast<std::uint8_t>(graph.distance_type),
      cache.device_witnesses.get());
  CheckCuda(cudaGetLastError(), "JvCandidatesKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "JvCandidatesKernel synchronize");
  const double kernel_ms = ElapsedMilliseconds(kernel_start);
  const auto d2h_start = std::chrono::steady_clock::now();
  cache.device_witnesses.CopyToHost(cache.host_witnesses.data(), cache.host_witnesses.size());
  const double d2h_ms = ElapsedMilliseconds(d2h_start);
  if (cache_usage != nullptr) {
    cache_usage->kernel_ms = kernel_ms;
    cache_usage->d2h_ms = d2h_ms;
  }

  std::vector<Candidate> candidates;
  for (std::size_t edge_id = 0; edge_id < cache.host_witnesses.size(); ++edge_id) {
    if (cache.host_witnesses[edge_id] >= 0) {
      candidates.push_back({static_cast<std::int32_t>(edge_id), cache.host_witnesses[edge_id],
                            EliminationMethod::kJv});
    }
  }
  return candidates;
}

void ClearJvCudaCache() {
  g_jv_device_caches.clear();
  g_jv_preferred_device = -1;
}

} // namespace cudaee
