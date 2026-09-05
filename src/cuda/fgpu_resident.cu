#include "../fgpu/resident_backend.hpp"

#include "../fgpu/main_edge_predicate.hpp"
#include "../fgpu/permutation_catalog.hpp"
#include "../fgpu/quick_hs_predicate.hpp"
#include "../fgpu/resident_lp_model.hpp"
#include "../fgpu/sparse_pdhg.hpp"
#include "cuda_edge_elimination/cuda_device_affinity.hpp"
#include "main_edge_metric.cuh"
#include "resident_pdhg_quantize.cuh"
#include "resident_sec_replay.cuh"
#include "resident_transaction.cuh"
#include "signed128.cuh"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace cudaee::detail {
namespace {

using SteadyClock = std::chrono::steady_clock;

void CheckCuda(const cudaError_t status, const char* const operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

double ElapsedMilliseconds(const SteadyClock::time_point begin) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - begin).count();
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const std::size_t count, const int device) : count_(count), device_(device) {
    if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("resident CUDA buffer 字节数溢出");
    }
    if (count_ != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(resident allocation)");
      CheckCuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc(resident)");
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
  [[nodiscard]] std::uint64_t bytes() const {
    return static_cast<std::uint64_t>(count_) * sizeof(T);
  }

  void CopyFromHost(const T* const source, const std::size_t count) {
    if (count > count_) {
      throw std::logic_error("resident CUDA H2D 越界");
    }
    if (count != 0U) {
      CheckCuda(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy resident H2D");
    }
  }

  void CopyToHost(T* const destination, const std::size_t count) const {
    if (count > count_) {
      throw std::logic_error("resident CUDA D2H 越界");
    }
    if (count != 0U) {
      CheckCuda(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                "cudaMemcpy resident D2H");
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
  void MoveFrom(DeviceBuffer* const other) noexcept {
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

int SelectDevice(const int requested, std::string* const reason) {
  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status != cudaSuccess || count == 0) {
    if (reason != nullptr) {
      *reason =
          count_status == cudaSuccess ? "没有可见 CUDA 设备" : cudaGetErrorString(count_status);
    }
    return -1;
  }
  int selected = requested;
  if (selected < 0) {
    selected = CudaDevicePreferenceForCurrentThread();
  }
  if (selected < 0) {
    std::size_t best_free = 0U;
    for (int device = 0; device < count; ++device) {
      if (cudaSetDevice(device) != cudaSuccess) {
        continue;
      }
      std::size_t free_bytes = 0U;
      std::size_t total_bytes = 0U;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
          (selected < 0 || free_bytes > best_free)) {
        selected = device;
        best_free = free_bytes;
      }
    }
  }
  if (selected < 0 || selected >= count || cudaSetDevice(selected) != cudaSuccess) {
    if (reason != nullptr) {
      *reason = "无法选择 resident CUDA device";
    }
    return -1;
  }
  return selected;
}

__global__ void InitializeEdgeIdsKernel(const std::int32_t edge_count,
                                        std::int32_t* const edge_ids) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge < edge_count) {
    edge_ids[edge] = edge;
  }
}

// 完整图首次快照仅供 geometry/replay/事务检查使用，无需 CPU 排序 CSR。
// 行内按顶点编号确定性生成；几何阶段后再转成按 (cost,node) 排序的稀疏行。
__global__ void InitializeCompleteAdjacencyKernel(const int dimension, std::int64_t* rows,
                                                  int* neighbors, int* edge_ids) {
  const std::int64_t node = blockIdx.x;
  if (threadIdx.x == 0) {
    rows[node] = node * (dimension - 1LL);
    if (node == dimension - 1)
      rows[dimension] = static_cast<std::int64_t>(dimension) * (dimension - 1);
  }
  for (int slot = threadIdx.x; slot < dimension - 1; slot += blockDim.x) {
    const std::int64_t other = slot >= node ? slot + 1 : slot;
    const auto u = node < other ? node : other, v = node < other ? other : node;
    const auto index = node * (dimension - 1LL) + slot;
    neighbors[index] = static_cast<int>(other);
    edge_ids[index] = static_cast<int>(u * (2LL * dimension - u - 1) / 2 + v - u - 1);
  }
}

__global__ void CopyDegreeToOffsetsKernel(const std::int32_t dimension,
                                          const std::int32_t* const degree,
                                          std::int64_t* const offsets) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node < dimension) {
    offsets[node] = degree[node];
  } else if (node == dimension) {
    offsets[node] = 0;
  }
}

__global__ void BuildPairCountsKernel(const std::int32_t dimension,
                                      const std::int32_t* const degree,
                                      std::int64_t* const counts) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node < dimension) {
    const std::int64_t value = degree[node];
    counts[node] = value * (value - 1) / 2;
  } else if (node == dimension) {
    counts[node] = 0;
  }
}

__device__ bool DecodeTriangularPair(const std::int64_t degree, const std::int64_t ordinal,
                                     std::int64_t* const first, std::int64_t* const second) {
  std::int64_t low = 0;
  std::int64_t high = degree - 1;
  while (low < high) {
    const std::int64_t middle = low + (high - low + 1) / 2;
    const std::int64_t prefix = middle * (2 * degree - middle - 1) / 2;
    if (prefix <= ordinal) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  *first = low;
  const std::int64_t prefix = low * (2 * degree - low - 1) / 2;
  *second = low + 1 + ordinal - prefix;
  return *first >= 0 && *second < degree;
}

__global__ void CarryNonpairMaskKernel(
    const std::int32_t dimension, const std::int64_t* const old_row_offsets,
    const std::int32_t* const old_neighbor_edge_ids, const std::int64_t* const old_pair_offsets,
    const std::uint8_t* const old_nonpair_mask, const std::int64_t* const new_row_offsets,
    const std::int32_t* const new_neighbor_edge_ids, const std::int64_t* const new_pair_offsets,
    std::uint8_t* const new_nonpair_mask, std::int32_t* const invalid) {
  const std::int32_t center = static_cast<std::int32_t>(blockIdx.x);
  if (center >= dimension) {
    return;
  }
  const std::int64_t old_begin = old_row_offsets[center];
  const std::int64_t old_degree = old_row_offsets[center + 1] - old_begin;
  const std::int64_t new_begin = new_row_offsets[center];
  const std::int64_t new_degree = new_row_offsets[center + 1] - new_begin;
  const std::int64_t pair_count = new_degree * (new_degree - 1) / 2;
  for (std::int64_t ordinal = threadIdx.x; ordinal < pair_count; ordinal += blockDim.x) {
    std::int64_t new_first = -1;
    std::int64_t new_second = -1;
    if (!DecodeTriangularPair(new_degree, ordinal, &new_first, &new_second)) {
      atomicExch(invalid, 1);
      continue;
    }
    const std::int32_t first_edge = new_neighbor_edge_ids[new_begin + new_first];
    const std::int32_t second_edge = new_neighbor_edge_ids[new_begin + new_second];
    std::int64_t old_first = -1;
    std::int64_t old_second = -1;
    for (std::int64_t slot = 0; slot < old_degree; ++slot) {
      const std::int32_t edge = old_neighbor_edge_ids[old_begin + slot];
      old_first = edge == first_edge ? slot : old_first;
      old_second = edge == second_edge ? slot : old_second;
    }
    if (old_first < 0 || old_second < 0) {
      // 新 CSR 只能删除旧边；找不到 stable edge id 表示内部不一致。
      atomicExch(invalid, 1);
      continue;
    }
    if (old_first > old_second) {
      const std::int64_t temporary = old_first;
      old_first = old_second;
      old_second = temporary;
    }
    const std::int64_t old_local =
        old_first * (2 * old_degree - old_first - 1) / 2 + (old_second - old_first - 1);
    new_nonpair_mask[new_pair_offsets[center] + ordinal] =
        old_nonpair_mask[old_pair_offsets[center] + old_local];
  }
}

__global__ void CommitNonpairMaskKernel(const std::int64_t pair_count,
                                        const std::uint8_t* const authorized,
                                        std::uint8_t* const nonpair_mask,
                                        unsigned long long* const committed_count) {
  const std::int64_t pair = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair < pair_count && authorized[pair] != 0U && nonpair_mask[pair] == 0U) {
    nonpair_mask[pair] = 1U;
    atomicAdd(committed_count, 1ULL);
  }
}

__global__ void ScatterCompactAdjacencyKernel(const std::int32_t work_count,
                                              const std::int32_t* const active_edge_ids,
                                              const std::int32_t* const edge_u,
                                              const std::int32_t* const edge_v,
                                              unsigned long long* const cursor,
                                              std::int32_t* const neighbors,
                                              std::int32_t* const neighbor_edge_ids) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  const std::int32_t u = edge_u[edge];
  const std::int32_t v = edge_v[edge];
  const std::uint64_t u_slot = atomicAdd(&cursor[u], 1ULL);
  const std::uint64_t v_slot = atomicAdd(&cursor[v], 1ULL);
  neighbors[u_slot] = v;
  neighbor_edge_ids[u_slot] = edge;
  neighbors[v_slot] = u;
  neighbor_edge_ids[v_slot] = edge;
}

__global__ void SortCompactAdjacencyKernel(const std::int32_t dimension,
                                           const std::int64_t* const row_offsets,
                                           const std::int64_t* const edge_weight,
                                           std::int32_t* const neighbors,
                                           std::int32_t* const neighbor_edge_ids) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x);
  if (node >= dimension || threadIdx.x != 0U) {
    return;
  }
  const std::int64_t begin = row_offsets[node];
  const std::int64_t end = row_offsets[node + 1];
  for (std::int64_t slot = begin + 1; slot < end; ++slot) {
    const std::int32_t neighbor = neighbors[slot];
    const std::int32_t edge = neighbor_edge_ids[slot];
    const std::int64_t weight = edge_weight[edge];
    std::int64_t position = slot;
    while (position > begin) {
      const std::int32_t previous_edge = neighbor_edge_ids[position - 1];
      const std::int32_t previous_neighbor = neighbors[position - 1];
      const std::int64_t previous_weight = edge_weight[previous_edge];
      if (previous_weight < weight || (previous_weight == weight && previous_neighbor < neighbor) ||
          (previous_weight == weight && previous_neighbor == neighbor && previous_edge < edge)) {
        break;
      }
      neighbors[position] = previous_neighbor;
      neighbor_edge_ids[position] = previous_edge;
      --position;
    }
    neighbors[position] = neighbor;
    neighbor_edge_ids[position] = edge;
  }
}

__global__ void
ValidateCompactAdjacencyKernel(const std::int32_t dimension, const std::int64_t slot_count,
                               const std::int32_t edge_count, const std::int64_t* const row_offsets,
                               const std::int32_t* const neighbors,
                               const std::int32_t* const neighbor_edge_ids,
                               const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                               const std::uint8_t* const edge_active, std::int32_t* const invalid) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node >= dimension) {
    return;
  }
  const std::int64_t begin = row_offsets[node];
  const std::int64_t end = row_offsets[node + 1];
  if (begin < 0 || end < begin || end > slot_count) {
    atomicExch(invalid, 1);
    return;
  }
  for (std::int64_t slot = begin; slot < end; ++slot) {
    const std::int32_t neighbor = neighbors[slot];
    const std::int32_t edge = neighbor_edge_ids[slot];
    if (neighbor < 0 || neighbor >= dimension || edge < 0 || edge >= edge_count ||
        edge_active[edge] == 0U ||
        !((edge_u[edge] == node && edge_v[edge] == neighbor) ||
          (edge_v[edge] == node && edge_u[edge] == neighbor))) {
      atomicExch(invalid, 1);
      return;
    }
  }
}

__global__ void ExpandDirtyVerticesKernel(const quick_hs::GraphView graph,
                                          const std::uint32_t* const dirty,
                                          std::uint32_t* const expanded) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node >= graph.dimension || dirty[node] == 0U) {
    return;
  }
  atomicExch(&expanded[node], 1U);
  for (std::int64_t slot = quick_hs::NeighborBegin(graph, node);
       slot < quick_hs::NeighborEnd(graph, node); ++slot) {
    if (quick_hs::NeighborActive(graph, slot)) {
      atomicExch(&expanded[quick_hs::Neighbor(graph, node, slot)], 1U);
    }
  }
}

__global__ void
BuildDirtyRootFlagsKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                          const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                          const std::uint32_t* const dirty, std::uint8_t* const flags) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work < work_count) {
    const std::int32_t edge = active_edge_ids[work];
    flags[work] = dirty[edge_u[edge]] != 0U || dirty[edge_v[edge]] != 0U ? 1U : 0U;
  }
}

__global__ void ValidateMetricKernel(const std::int32_t edge_count,
                                     const std::int32_t* const edge_u,
                                     const std::int32_t* const edge_v,
                                     const std::int64_t* const edge_weight,
                                     const quick_hs::GraphView graph, std::int32_t* const invalid) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge < edge_count &&
      quick_hs::Distance(graph, edge_u[edge], edge_v[edge]) != edge_weight[edge]) {
    atomicExch(invalid, 1);
  }
}

constexpr std::int32_t kMaxGeometryPotential = 32;
constexpr std::int32_t kGeometryKdStackCapacity = 32;
constexpr std::int32_t kLocalSecFamilies = 4;
constexpr std::int32_t kConnectivitySupportLevels = 3;
// 每条边最多跨 16 个静态 local windows；每个 support 阈值下，它只会
// 跨两个 endpoint components，因此三层 separator 再增加至多 6 个 incidence。
constexpr std::int32_t kMaxLocalSecIncidence = 16 + 2 * kConnectivitySupportLevels;

struct LocalSecLayout {
  std::int32_t dimension{};
  std::int32_t cut_count{};
  std::int32_t offset[kLocalSecFamilies + 1]{};
  std::int32_t size[kLocalSecFamilies]{16, 24, 32, 48};
  std::int32_t stride[kLocalSecFamilies]{8, 12, 16, 24};
};

LocalSecLayout BuildLocalSecLayout(const std::int32_t dimension) {
  LocalSecLayout layout;
  layout.dimension = dimension;
  for (std::int32_t family = 0; family < kLocalSecFamilies; ++family) {
    layout.offset[family] = layout.cut_count;
    if (dimension - layout.size[family] >= 2) {
      layout.cut_count += (dimension - layout.size[family]) / layout.stride[family] + 1;
    }
  }
  layout.offset[kLocalSecFamilies] = layout.cut_count;
  return layout;
}

struct GeometryKdNode {
  std::int32_t point{-1};
  std::int32_t left{-1};
  std::int32_t right{-1};
  std::int64_t min_x{};
  std::int64_t max_x{};
  std::int64_t min_y{};
  std::int64_t max_y{};
};

struct DeviceInterval {
  double lower{};
  double upper{};
};

struct IntervalGeometryPotential {
  std::int32_t node{-1};
  DeviceInterval min_p;
  DeviceInterval min_q;
};

__device__ DeviceInterval IntervalInteger(const std::int64_t value) {
  return {__ll2double_rd(value), __ll2double_ru(value)};
}

__device__ DeviceInterval IntervalAdd(const DeviceInterval lhs, const DeviceInterval rhs) {
  return {__dadd_rd(lhs.lower, rhs.lower), __dadd_ru(lhs.upper, rhs.upper)};
}

__device__ DeviceInterval IntervalNegate(const DeviceInterval value) {
  return {-value.upper, -value.lower};
}

__device__ DeviceInterval IntervalSubtract(const DeviceInterval lhs, const DeviceInterval rhs) {
  return IntervalAdd(lhs, IntervalNegate(rhs));
}

__device__ DeviceInterval IntervalMultiply(const DeviceInterval lhs, const DeviceInterval rhs) {
  const double lower0 = __dmul_rd(lhs.lower, rhs.lower);
  const double lower1 = __dmul_rd(lhs.lower, rhs.upper);
  const double lower2 = __dmul_rd(lhs.upper, rhs.lower);
  const double lower3 = __dmul_rd(lhs.upper, rhs.upper);
  const double upper0 = __dmul_ru(lhs.lower, rhs.lower);
  const double upper1 = __dmul_ru(lhs.lower, rhs.upper);
  const double upper2 = __dmul_ru(lhs.upper, rhs.lower);
  const double upper3 = __dmul_ru(lhs.upper, rhs.upper);
  return {fmin(fmin(lower0, lower1), fmin(lower2, lower3)),
          fmax(fmax(upper0, upper1), fmax(upper2, upper3))};
}

__device__ DeviceInterval IntervalSquare(const DeviceInterval value) {
  if (value.lower <= 0.0 && value.upper >= 0.0) {
    return {0.0, fmax(__dmul_ru(value.lower, value.lower), __dmul_ru(value.upper, value.upper))};
  }
  const double lower0 = __dmul_rd(value.lower, value.lower);
  const double lower1 = __dmul_rd(value.upper, value.upper);
  const double upper0 = __dmul_ru(value.lower, value.lower);
  const double upper1 = __dmul_ru(value.upper, value.upper);
  return {fmin(lower0, lower1), fmax(upper0, upper1)};
}

__device__ bool IntervalContainsZero(const DeviceInterval value) {
  return value.lower <= 0.0 && value.upper >= 0.0;
}

__device__ DeviceInterval IntervalDivide(const DeviceInterval numerator,
                                         const DeviceInterval denominator) {
  if (IntervalContainsZero(denominator)) {
    return {-CUDART_INF, CUDART_INF};
  }
  DeviceInterval reciprocal{__ddiv_rd(1.0, denominator.upper), __ddiv_ru(1.0, denominator.lower)};
  if (reciprocal.lower > reciprocal.upper) {
    const double temporary = reciprocal.lower;
    reciprocal.lower = reciprocal.upper;
    reciprocal.upper = temporary;
  }
  return IntervalMultiply(numerator, reciprocal);
}

__device__ DeviceInterval IntervalSqrt(const DeviceInterval value) {
  if (value.lower < 0.0) {
    return {-CUDART_INF, CUDART_INF};
  }
  return {__dsqrt_rd(value.lower), __dsqrt_ru(value.upper)};
}

__device__ bool IntervalPositive(const DeviceInterval value) { return value.lower > 0.0; }

__device__ bool IntervalGreater(const DeviceInterval lhs, const DeviceInterval rhs) {
  return lhs.lower > rhs.upper;
}

__device__ bool IntervalGreaterEqual(const DeviceInterval lhs, const DeviceInterval rhs) {
  return lhs.lower >= rhs.upper;
}

__device__ bool IntervalCosine(const DeviceInterval value) {
  return value.lower >= -1.0 && value.upper <= 1.0;
}

__device__ DeviceInterval CoordinateDistanceInterval(const std::int32_t lhs, const std::int32_t rhs,
                                                     const std::int64_t* const x,
                                                     const std::int64_t* const y,
                                                     const std::uint32_t denominator) {
  const DeviceInterval dx = IntervalSubtract(IntervalInteger(x[lhs]), IntervalInteger(x[rhs]));
  const DeviceInterval dy = IntervalSubtract(IntervalInteger(y[lhs]), IntervalInteger(y[rhs]));
  return IntervalDivide(IntervalSqrt(IntervalAdd(IntervalSquare(dx), IntervalSquare(dy))),
                        IntervalInteger(denominator));
}

__device__ DeviceInterval IntervalCosineSum(const DeviceInterval first,
                                            const DeviceInterval second) {
  const DeviceInterval one = IntervalInteger(1);
  const DeviceInterval sine_first = IntervalSqrt(IntervalSubtract(one, IntervalSquare(first)));
  const DeviceInterval sine_second = IntervalSqrt(IntervalSubtract(one, IntervalSquare(second)));
  return IntervalSubtract(IntervalMultiply(first, second),
                          IntervalMultiply(sine_first, sine_second));
}

__device__ bool IntervalPotentialBounds(const quick_hs::GraphView& graph, const std::int32_t p,
                                        const std::int32_t q, const std::int32_t r,
                                        const std::int64_t nearest, const std::int64_t* const x,
                                        const std::int64_t* const y,
                                        IntervalGeometryPotential* const output) {
  const DeviceInterval half{0.5, 0.5};
  const DeviceInterval one = IntervalInteger(1);
  const DeviceInterval two = IntervalInteger(2);
  const DeviceInterval delta = IntervalSubtract(IntervalInteger(nearest), half);
  const DeviceInterval lpq = IntervalInteger(quick_hs::Distance(graph, p, q));
  const DeviceInterval lpr = IntervalInteger(quick_hs::Distance(graph, p, r));
  const DeviceInterval lqr = IntervalInteger(quick_hs::Distance(graph, q, r));
  const DeviceInterval dpq = CoordinateDistanceInterval(p, q, x, y, graph.coordinate_denominator);
  const DeviceInterval dpr = CoordinateDistanceInterval(p, r, x, y, graph.coordinate_denominator);
  const DeviceInterval dqr = CoordinateDistanceInterval(q, r, x, y, graph.coordinate_denominator);
  if (!IntervalPositive(delta) || !IntervalPositive(dpq) || !IntervalPositive(dpr) ||
      !IntervalPositive(dqr)) {
    return false;
  }
  const DeviceInterval length_p =
      IntervalSubtract(IntervalSubtract(IntervalAdd(delta, lpq), lqr), one);
  const DeviceInterval length_q =
      IntervalSubtract(IntervalSubtract(IntervalAdd(delta, lpq), lpr), one);
  if (!IntervalPositive(length_p) || !IntervalPositive(length_q) ||
      !IntervalGreaterEqual(IntervalAdd(length_p, length_q), IntervalSubtract(lpq, half))) {
    return false;
  }

  const DeviceInterval gamma_numerator =
      IntervalAdd(IntervalSubtract(IntervalAdd(length_p, length_q), lpq), half);
  const DeviceInterval cos_gamma =
      IntervalSubtract(one, IntervalDivide(IntervalSquare(gamma_numerator),
                                           IntervalMultiply(two, IntervalSquare(delta))));
  const DeviceInterval cos_alpha_p_half = IntervalDivide(
      IntervalSubtract(IntervalSubtract(IntervalSquare(length_q), IntervalSquare(delta)),
                       IntervalSquare(dqr)),
      IntervalMultiply(two, IntervalMultiply(delta, dqr)));
  const DeviceInterval cos_alpha_q_half = IntervalDivide(
      IntervalSubtract(IntervalSubtract(IntervalSquare(length_p), IntervalSquare(delta)),
                       IntervalSquare(dpr)),
      IntervalMultiply(two, IntervalMultiply(delta, dpr)));
  if (!IntervalCosine(cos_gamma) || !IntervalCosine(cos_alpha_p_half) ||
      !IntervalCosine(cos_alpha_q_half) || !IntervalPositive(cos_alpha_p_half) ||
      !IntervalPositive(cos_alpha_q_half)) {
    return false;
  }
  const DeviceInterval cos_alpha_p =
      IntervalSubtract(IntervalMultiply(two, IntervalSquare(cos_alpha_p_half)), one);
  const DeviceInterval cos_alpha_q =
      IntervalSubtract(IntervalMultiply(two, IntervalSquare(cos_alpha_q_half)), one);
  if (!IntervalGreater(cos_alpha_p, cos_gamma) || !IntervalGreater(cos_alpha_q, cos_gamma)) {
    return false;
  }

  const DeviceInterval cos_e_p = IntervalDivide(
      IntervalSubtract(IntervalAdd(IntervalSquare(dpq), IntervalSquare(dpr)), IntervalSquare(dqr)),
      IntervalMultiply(two, IntervalMultiply(dpq, dpr)));
  const DeviceInterval cos_e_q = IntervalDivide(
      IntervalSubtract(IntervalAdd(IntervalSquare(dpq), IntervalSquare(dqr)), IntervalSquare(dpr)),
      IntervalMultiply(two, IntervalMultiply(dpq, dqr)));
  const DeviceInterval dqr_delta = IntervalAdd(dqr, delta);
  const DeviceInterval dpr_delta = IntervalAdd(dpr, delta);
  const DeviceInterval left_p =
      IntervalDivide(IntervalSubtract(IntervalAdd(IntervalSquare(dqr_delta), IntervalSquare(dpq)),
                                      IntervalSquare(length_p)),
                     IntervalMultiply(two, IntervalMultiply(dqr_delta, dpq)));
  const DeviceInterval left_q =
      IntervalDivide(IntervalSubtract(IntervalAdd(IntervalSquare(dpr_delta), IntervalSquare(dpq)),
                                      IntervalSquare(length_q)),
                     IntervalMultiply(two, IntervalMultiply(dpr_delta, dpq)));
  if (!IntervalCosine(cos_e_p) || !IntervalCosine(cos_e_q) ||
      !IntervalGreaterEqual(cos_e_q, left_p) || !IntervalGreaterEqual(cos_e_p, left_q)) {
    return false;
  }

  const DeviceInterval cos_t_p =
      IntervalDivide(IntervalSubtract(IntervalAdd(IntervalSquare(length_p), IntervalSquare(dpr)),
                                      IntervalSquare(delta)),
                     IntervalMultiply(two, IntervalMultiply(length_p, dpr)));
  const DeviceInterval cos_t_q =
      IntervalDivide(IntervalSubtract(IntervalAdd(IntervalSquare(length_q), IntervalSquare(dqr)),
                                      IntervalSquare(delta)),
                     IntervalMultiply(two, IntervalMultiply(length_q, dqr)));
  const DeviceInterval zero = IntervalInteger(0);
  if (!IntervalCosine(cos_t_p) || !IntervalCosine(cos_t_q) ||
      !IntervalGreaterEqual(IntervalAdd(cos_e_p, cos_t_p), zero) ||
      !IntervalGreaterEqual(IntervalAdd(cos_e_q, cos_t_q), zero)) {
    return false;
  }
  const DeviceInterval cos_sum_q = IntervalCosineSum(cos_e_q, cos_t_q);
  const DeviceInterval cos_sum_p = IntervalCosineSum(cos_e_p, cos_t_p);
  const DeviceInterval max_p_squared = IntervalSubtract(
      IntervalAdd(IntervalSquare(dpq), IntervalSquare(length_q)),
      IntervalMultiply(two, IntervalMultiply(IntervalMultiply(dpq, length_q), cos_sum_q)));
  const DeviceInterval max_q_squared = IntervalSubtract(
      IntervalAdd(IntervalSquare(dpq), IntervalSquare(length_p)),
      IntervalMultiply(two, IntervalMultiply(IntervalMultiply(dpq, length_p), cos_sum_p)));
  if (max_p_squared.lower < 0.0 || max_q_squared.lower < 0.0) {
    return false;
  }
  output->node = r;
  output->min_p = IntervalSubtract(IntervalSubtract(delta, one), IntervalSqrt(max_p_squared));
  output->min_q = IntervalSubtract(IntervalSubtract(delta, one), IntervalSqrt(max_q_squared));
  return isfinite(output->min_p.lower) && isfinite(output->min_p.upper) &&
         isfinite(output->min_q.lower) && isfinite(output->min_q.upper);
}

__device__ bool GeometryCandidateLess(const double lhs_score, const std::int32_t lhs_node,
                                      const double rhs_score, const std::int32_t rhs_node) {
  return lhs_score < rhs_score || (lhs_score == rhs_score && lhs_node < rhs_node);
}

__device__ void InsertGeometryCandidate(const std::int32_t node, const double score,
                                        const std::int32_t capacity, std::int32_t* const nodes,
                                        double* const scores, std::int32_t* const count) {
  if (*count == capacity &&
      !GeometryCandidateLess(score, node, scores[capacity - 1], nodes[capacity - 1])) {
    return;
  }
  std::int32_t position = *count < capacity ? (*count)++ : capacity - 1;
  while (position > 0 &&
         GeometryCandidateLess(score, node, scores[position - 1], nodes[position - 1])) {
    scores[position] = scores[position - 1];
    nodes[position] = nodes[position - 1];
    --position;
  }
  scores[position] = score;
  nodes[position] = node;
}

__device__ double GeometryMidpointScore(const std::int32_t p, const std::int32_t q,
                                        const std::int32_t node, const std::int64_t* const x,
                                        const std::int64_t* const y) {
  const double dx =
      2.0 * static_cast<double>(x[node]) - static_cast<double>(x[p]) - static_cast<double>(x[q]);
  const double dy =
      2.0 * static_cast<double>(y[node]) - static_cast<double>(y[p]) - static_cast<double>(y[q]);
  return dx * dx + dy * dy;
}

__device__ double GeometryKdLowerBound(const GeometryKdNode& node, const std::int32_t p,
                                       const std::int32_t q, const std::int64_t* const x,
                                       const std::int64_t* const y) {
  const double midpoint_x = static_cast<double>(x[p]) + static_cast<double>(x[q]);
  const double midpoint_y = static_cast<double>(y[p]) + static_cast<double>(y[q]);
  const double minimum_x = 2.0 * static_cast<double>(node.min_x);
  const double maximum_x = 2.0 * static_cast<double>(node.max_x);
  const double minimum_y = 2.0 * static_cast<double>(node.min_y);
  const double maximum_y = 2.0 * static_cast<double>(node.max_y);
  const double dx = midpoint_x < minimum_x   ? minimum_x - midpoint_x
                    : midpoint_x > maximum_x ? midpoint_x - maximum_x
                                             : 0.0;
  const double dy = midpoint_y < minimum_y   ? minimum_y - midpoint_y
                    : midpoint_y > maximum_y ? midpoint_y - maximum_y
                                             : 0.0;
  // 下界使用向下舍入；即使输入接近 FP64 边界，也只会少剪枝，不会漏掉候选。
  return __dadd_rd(__dmul_rd(dx, dx), __dmul_rd(dy, dy));
}

__device__ bool SelectGeometryCandidates(const std::int32_t p, const std::int32_t q,
                                         const GeometryKdNode* const kd_nodes,
                                         const std::int32_t kd_root, const std::int64_t* const x,
                                         const std::int64_t* const y, const std::int32_t capacity,
                                         std::int32_t* const nodes, double* const scores,
                                         std::int32_t* const selected) {
  std::int32_t stack[kGeometryKdStackCapacity]{};
  std::int32_t stack_size = 0;
  stack[stack_size++] = kd_root;
  while (stack_size != 0) {
    const std::int32_t tree_index = stack[--stack_size];
    const GeometryKdNode tree_node = kd_nodes[tree_index];
    if (*selected == capacity &&
        GeometryKdLowerBound(tree_node, p, q, x, y) > scores[capacity - 1]) {
      continue;
    }
    if (tree_node.point != p && tree_node.point != q) {
      InsertGeometryCandidate(tree_node.point, GeometryMidpointScore(p, q, tree_node.point, x, y),
                              capacity, nodes, scores, selected);
    }

    // 先访问 AABB 下界较小的子树。栈溢出时不能漏点，直接放弃该边的几何删除。
    const std::int32_t first = tree_node.left;
    const std::int32_t second = tree_node.right;
    if (first >= 0 && second >= 0) {
      const double first_bound = GeometryKdLowerBound(kd_nodes[first], p, q, x, y);
      const double second_bound = GeometryKdLowerBound(kd_nodes[second], p, q, x, y);
      const std::int32_t near_child =
          first_bound < second_bound || (first_bound == second_bound && first < second) ? first
                                                                                        : second;
      const std::int32_t far_child = near_child == first ? second : first;
      if (stack_size + 2 > kGeometryKdStackCapacity) {
        return false;
      }
      stack[stack_size++] = far_child;
      stack[stack_size++] = near_child;
    } else if (first >= 0 || second >= 0) {
      if (stack_size + 1 > kGeometryKdStackCapacity) {
        return false;
      }
      stack[stack_size++] = first >= 0 ? first : second;
    }
  }
  return true;
}

__device__ double GeometryPositionScore(const std::int32_t p, const std::int32_t q,
                                        const std::int32_t node,
                                        const std::int32_t position_numerator,
                                        const std::int32_t position_denominator,
                                        const std::int64_t* const x, const std::int64_t* const y) {
  const double numerator_x =
      static_cast<double>(position_denominator - position_numerator) * static_cast<double>(x[p]) +
      static_cast<double>(position_numerator) * static_cast<double>(x[q]);
  const double numerator_y =
      static_cast<double>(position_denominator - position_numerator) * static_cast<double>(y[p]) +
      static_cast<double>(position_numerator) * static_cast<double>(y[q]);
  const double dx =
      static_cast<double>(position_denominator) * static_cast<double>(x[node]) - numerator_x;
  const double dy =
      static_cast<double>(position_denominator) * static_cast<double>(y[node]) - numerator_y;
  return dx * dx + dy * dy;
}

__device__ double GeometryKdPositionLowerBound(const GeometryKdNode& node, const std::int32_t p,
                                               const std::int32_t q,
                                               const std::int32_t position_numerator,
                                               const std::int32_t position_denominator,
                                               const std::int64_t* const x,
                                               const std::int64_t* const y) {
  const double target_x =
      static_cast<double>(position_denominator - position_numerator) * static_cast<double>(x[p]) +
      static_cast<double>(position_numerator) * static_cast<double>(x[q]);
  const double target_y =
      static_cast<double>(position_denominator - position_numerator) * static_cast<double>(y[p]) +
      static_cast<double>(position_numerator) * static_cast<double>(y[q]);
  const double minimum_x =
      static_cast<double>(position_denominator) * static_cast<double>(node.min_x);
  const double maximum_x =
      static_cast<double>(position_denominator) * static_cast<double>(node.max_x);
  const double minimum_y =
      static_cast<double>(position_denominator) * static_cast<double>(node.min_y);
  const double maximum_y =
      static_cast<double>(position_denominator) * static_cast<double>(node.max_y);
  const double dx = target_x < minimum_x   ? minimum_x - target_x
                    : target_x > maximum_x ? target_x - maximum_x
                                           : 0.0;
  const double dy = target_y < minimum_y   ? minimum_y - target_y
                    : target_y > maximum_y ? target_y - maximum_y
                                           : 0.0;
  return __dadd_rd(__dmul_rd(dx, dx), __dmul_rd(dy, dy));
}

// Step 2 会在一条边的多个等距位置查询邻近点。单独保留该查询，避免改变
// Step 1 已锁定的 midpoint 候选顺序与历史输出哈希。
__device__ bool SelectGeometryCandidatesAtPosition(
    const std::int32_t p, const std::int32_t q, const GeometryKdNode* const kd_nodes,
    const std::int32_t kd_root, const std::int64_t* const x, const std::int64_t* const y,
    const std::int32_t position_numerator, const std::int32_t position_denominator,
    const std::int32_t capacity, std::int32_t* const nodes, double* const scores,
    std::int32_t* const selected) {
  std::int32_t stack[kGeometryKdStackCapacity]{};
  std::int32_t stack_size = 0;
  stack[stack_size++] = kd_root;
  while (stack_size != 0) {
    const std::int32_t tree_index = stack[--stack_size];
    const GeometryKdNode tree_node = kd_nodes[tree_index];
    if (*selected == capacity &&
        GeometryKdPositionLowerBound(tree_node, p, q, position_numerator, position_denominator, x,
                                     y) > scores[capacity - 1]) {
      continue;
    }
    if (tree_node.point != p && tree_node.point != q) {
      InsertGeometryCandidate(tree_node.point,
                              GeometryPositionScore(p, q, tree_node.point, position_numerator,
                                                    position_denominator, x, y),
                              capacity, nodes, scores, selected);
    }

    const std::int32_t first = tree_node.left;
    const std::int32_t second = tree_node.right;
    if (first >= 0 && second >= 0) {
      const double first_bound = GeometryKdPositionLowerBound(
          kd_nodes[first], p, q, position_numerator, position_denominator, x, y);
      const double second_bound = GeometryKdPositionLowerBound(
          kd_nodes[second], p, q, position_numerator, position_denominator, x, y);
      const std::int32_t near_child =
          first_bound < second_bound || (first_bound == second_bound && first < second) ? first
                                                                                        : second;
      const std::int32_t far_child = near_child == first ? second : first;
      if (stack_size + 2 > kGeometryKdStackCapacity) {
        return false;
      }
      stack[stack_size++] = far_child;
      stack[stack_size++] = near_child;
    } else if (first >= 0 || second >= 0) {
      if (stack_size + 1 > kGeometryKdStackCapacity) {
        return false;
      }
      stack[stack_size++] = first >= 0 ? first : second;
    }
  }
  return true;
}

__global__ void PointNearPriorityKernel(const int dimension, const GeometryKdNode* const tree,
                                        const std::int64_t* const x, const std::int64_t* const y,
                                        std::int32_t* const priority, std::uint32_t* const mask) {
  const int center = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (center >= dimension)
    return;
  std::int32_t nodes[quick_hs::kMaxPotentialNodes]{};
  double scores[quick_hs::kMaxPotentialNodes]{};
  std::int32_t selected = 0;
  if (!SelectGeometryCandidatesAtPosition(center, center, tree, 0, x, y, 1, 2,
                                          quick_hs::kMaxPotentialNodes, nodes, scores, &selected)) {
    selected = 0; // 排序不可用时完整扫描全部点，不缩小搜索域。
  }
  const std::size_t row_words = (static_cast<std::size_t>(dimension) + 31U) / 32U;
  for (int rank = 0; rank < quick_hs::kMaxPotentialNodes; ++rank) {
    priority[static_cast<std::size_t>(center) * quick_hs::kMaxPotentialNodes + rank] =
        rank < selected ? nodes[rank] : -1;
    if (rank < selected) {
      mask[static_cast<std::size_t>(center) * row_words + nodes[rank] / 32] |=
          1U << (nodes[rank] % 32);
    }
  }
}

__global__ void NearestDistanceKernel(const quick_hs::GraphView graph,
                                      std::int64_t* const nearest) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node >= graph.dimension) {
    return;
  }
  std::int64_t best = INT64_MAX;
  for (std::int32_t other = 0; other < graph.dimension; ++other) {
    if (other != node) {
      best = min(best, quick_hs::Distance(graph, node, other));
    }
  }
  nearest[node] = best;
}

__global__ void
GeometryKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
               const std::int32_t* const edge_u, const std::int32_t* const edge_v,
               const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
               const quick_hs::GraphView graph, const std::int64_t* const x,
               const std::int64_t* const y, const GeometryKdNode* const kd_nodes,
               const std::int32_t kd_root, const std::int64_t* const nearest,
               const std::int32_t potential_count, std::uint8_t* const proposed,
               std::int32_t* const first_witness, std::int32_t* const second_witness) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U || protected_edge[edge] != 0U) {
    return;
  }
  const std::int32_t p = edge_u[edge];
  const std::int32_t q = edge_v[edge];
  if (graph.degree[p] <= 2 || graph.degree[q] <= 2) {
    return;
  }
  std::int32_t nodes[kMaxGeometryPotential]{};
  double scores[kMaxGeometryPotential]{};
  std::int32_t selected = 0;
  if (!SelectGeometryCandidates(p, q, kd_nodes, kd_root, x, y, potential_count, nodes, scores,
                                &selected)) {
    return;
  }
  IntervalGeometryPotential potential[kMaxGeometryPotential]{};
  std::int32_t valid = 0;
  for (std::int32_t index = 0; index < selected; ++index) {
    IntervalGeometryPotential bounds;
    // CUDA 有向舍入生成严格包含真值的 FP64 区间；不确定的边界样本直接保留。
    if (IntervalPotentialBounds(graph, p, q, nodes[index], nearest[nodes[index]], x, y, &bounds)) {
      potential[valid++] = bounds;
    }
  }
  for (std::int32_t first = 0; first < valid; ++first) {
    for (std::int32_t second = first + 1; second < valid; ++second) {
      const std::int32_t r = potential[first].node;
      const std::int32_t s = potential[second].node;
      const std::int64_t original =
          quick_hs::Distance(graph, p, q) + quick_hs::Distance(graph, r, s);
      if (quick_hs::Distance(graph, p, r) + quick_hs::Distance(graph, q, s) >= original ||
          quick_hs::Distance(graph, p, s) + quick_hs::Distance(graph, q, r) >= original) {
        continue;
      }
      const DeviceInterval lpq = IntervalInteger(quick_hs::Distance(graph, p, q));
      const DeviceInterval lrs = IntervalInteger(quick_hs::Distance(graph, r, s));
      const DeviceInterval first_bound = IntervalSubtract(
          IntervalAdd(IntervalAdd(lpq, potential[second].min_p), potential[first].min_q), lrs);
      const DeviceInterval second_bound = IntervalSubtract(
          IntervalAdd(IntervalAdd(lpq, potential[first].min_p), potential[second].min_q), lrs);
      if (IntervalPositive(first_bound) && IntervalPositive(second_bound)) {
        proposed[edge] = 1U;
        first_witness[edge] = r;
        second_witness[edge] = s;
        return;
      }
    }
  }
}

__device__ bool IsJvWitness(const quick_hs::GraphView& graph, const std::int32_t a,
                            const std::int32_t b, const std::int32_t c) {
  if (c == a || c == b) {
    return false;
  }
  const std::int64_t cab = quick_hs::Distance(graph, a, b);
  const std::int64_t cac = quick_hs::Distance(graph, a, c);
  const std::int64_t cbc = quick_hs::Distance(graph, b, c);
  for (std::int64_t slot = quick_hs::NeighborBegin(graph, c);
       slot < quick_hs::NeighborEnd(graph, c); ++slot) {
    if (!quick_hs::NeighborActive(graph, slot)) {
      continue;
    }
    const std::int32_t d = quick_hs::Neighbor(graph, c, slot);
    if (d == a || d == b) {
      continue;
    }
    const std::int64_t left = cab + quick_hs::Distance(graph, c, d);
    if (left <= cac + quick_hs::Distance(graph, d, b) ||
        left <= quick_hs::Distance(graph, a, d) + cbc) {
      return false;
    }
  }
  return true;
}

__device__ std::int32_t FindJvWitness(const quick_hs::GraphView& graph, const std::int32_t a,
                                      const std::int32_t b) {
  std::int32_t candidates[quick_hs::kMaxPotentialNodes]{};
  std::int64_t scores[quick_hs::kMaxPotentialNodes]{};
  std::int32_t count = 0;
  for (std::int32_t side = 0; side < 2; ++side) {
    const std::int32_t from = side == 0 ? a : b;
    const std::int32_t other = side == 0 ? b : a;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, from);
         slot < quick_hs::NeighborEnd(graph, from); ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t node = quick_hs::Neighbor(graph, from, slot);
      if (node == a || node == b) {
        continue;
      }
      bool duplicate = false;
      for (std::int32_t existing = 0; existing < count; ++existing) {
        duplicate = duplicate || candidates[existing] == node;
      }
      if (duplicate) {
        continue;
      }
      const std::int64_t score =
          quick_hs::Distance(graph, from, node) + quick_hs::Distance(graph, node, other);
      if (count == quick_hs::kMaxPotentialNodes &&
          (score > scores[quick_hs::kMaxPotentialNodes - 1] ||
           (score == scores[quick_hs::kMaxPotentialNodes - 1] &&
            node >= candidates[quick_hs::kMaxPotentialNodes - 1]))) {
        continue;
      }
      std::int32_t position =
          count < quick_hs::kMaxPotentialNodes ? count++ : quick_hs::kMaxPotentialNodes - 1;
      while (position > 0 && (score < scores[position - 1] ||
                              (score == scores[position - 1] && node < candidates[position - 1]))) {
        scores[position] = scores[position - 1];
        candidates[position] = candidates[position - 1];
        --position;
      }
      scores[position] = score;
      candidates[position] = node;
    }
  }
  for (std::int32_t index = 0; index < count; ++index) {
    if (IsJvWitness(graph, a, b, candidates[index])) {
      return candidates[index];
    }
  }
  for (std::int32_t c = 0; c < graph.dimension; ++c) {
    if (IsJvWitness(graph, a, b, c)) {
      return c;
    }
  }
  return -1;
}

__global__ void JvKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                         const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                         const std::uint8_t* const edge_active,
                         const std::uint8_t* const protected_edge, const quick_hs::GraphView graph,
                         std::uint8_t* const proposed, std::int32_t* const first_witness) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U || protected_edge[edge] != 0U) {
    return;
  }
  const std::int32_t a = edge_u[edge];
  const std::int32_t b = edge_v[edge];
  if (graph.degree[a] <= 2 || graph.degree[b] <= 2) {
    return;
  }
  const std::int32_t witness = FindJvWitness(graph, a, b);
  if (witness >= 0) {
    proposed[edge] = 1U;
    first_witness[edge] = witness;
  }
}

__device__ bool DecodeNeighborPair(const quick_hs::GraphView graph, const std::int32_t center,
                                   const std::uint64_t ordinal, std::int32_t* const first_node,
                                   std::int32_t* const second_node,
                                   std::int32_t* const first_edge = nullptr,
                                   std::int32_t* const second_edge = nullptr) {
  const std::int64_t begin = quick_hs::NeighborBegin(graph, center);
  const std::int64_t degree = quick_hs::NeighborEnd(graph, center) - begin;
  std::int64_t low = 0;
  std::int64_t high = degree - 1;
  while (low < high) {
    const std::int64_t middle = low + (high - low + 1) / 2;
    const std::uint64_t prefix = static_cast<std::uint64_t>(middle * (2 * degree - middle - 1) / 2);
    if (prefix <= ordinal) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  const std::int64_t first = low;
  const std::uint64_t prefix = static_cast<std::uint64_t>(first * (2 * degree - first - 1) / 2);
  const std::int64_t second = first + 1 + static_cast<std::int64_t>(ordinal - prefix);
  if (first < 0 || second >= degree) {
    return false;
  }
  const std::int64_t first_slot = begin + first;
  const std::int64_t second_slot = begin + second;
  if (!quick_hs::NeighborActive(graph, first_slot) ||
      !quick_hs::NeighborActive(graph, second_slot) ||
      quick_hs::PairForbiddenBySlots(graph, center, first_slot, second_slot)) {
    return false;
  }
  *first_node = quick_hs::Neighbor(graph, center, first_slot);
  *second_node = quick_hs::Neighbor(graph, center, second_slot);
  if (first_edge != nullptr) {
    *first_edge = graph.neighbor_edge_ids == nullptr ? -1 : graph.neighbor_edge_ids[first_slot];
  }
  if (second_edge != nullptr) {
    *second_edge = graph.neighbor_edge_ids == nullptr ? -1 : graph.neighbor_edge_ids[second_slot];
  }
  return true;
}

__device__ std::uint64_t CountSurvivingPairsCta(const quick_hs::GraphView graph,
                                                const std::int32_t a, const std::int32_t b,
                                                const std::int32_t center) {
  const std::uint64_t degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, center) -
                                                          quick_hs::NeighborBegin(graph, center));
  const std::uint64_t pair_count = degree < 2U ? 0U : degree * (degree - 1U) / 2U;
  std::uint64_t surviving = 0U;
  for (std::uint64_t window = 0U; window < pair_count; window += blockDim.x) {
    const std::uint64_t ordinal = window + threadIdx.x;
    bool allowed = false;
    if (ordinal < pair_count) {
      std::int32_t first = -1;
      std::int32_t second = -1;
      allowed = DecodeNeighborPair(graph, center, ordinal, &first, &second) &&
                main_edge::Compatible(graph, a, b, center, first) &&
                main_edge::Compatible(graph, a, b, center, second) &&
                main_edge::ThreeCompatible(graph, a, b, first, center, second);
    }
    surviving += static_cast<std::uint64_t>(__syncthreads_count(allowed ? 1 : 0));
  }
  return surviving;
}

constexpr std::int32_t kMaxQuickCandidatePairs =
    quick_hs::kMaxPotentialNodes * (quick_hs::kMaxPotentialNodes - 1) / 2;
constexpr std::int32_t kQuickPairSortCapacity = 512;
static_assert(kMaxQuickCandidatePairs <= kQuickPairSortCapacity);

__device__ void DecodeQuickCandidatePair(const std::int32_t candidate_count,
                                         const std::int32_t ordinal, std::int32_t* const first,
                                         std::int32_t* const second) {
  std::int32_t remaining = ordinal;
  for (std::int32_t upper = 1; upper < candidate_count; ++upper) {
    if (remaining < upper) {
      *first = upper;
      *second = remaining;
      return;
    }
    remaining -= upper;
  }
  *first = -1;
  *second = -1;
}

__device__ bool QuickPairOrderAfter(const std::uint64_t lhs_product,
                                    const std::int64_t lhs_distance, const std::int32_t lhs_ordinal,
                                    const std::uint64_t rhs_product,
                                    const std::int64_t rhs_distance,
                                    const std::int32_t rhs_ordinal) {
  return lhs_product > rhs_product || (lhs_product == rhs_product && lhs_distance > rhs_distance) ||
         (lhs_product == rhs_product && lhs_distance == rhs_distance && lhs_ordinal > rhs_ordinal);
}

__device__ std::uint64_t SaturatingProduct(const std::uint64_t lhs, const std::uint64_t rhs) {
  return rhs != 0U && lhs > ULLONG_MAX / rhs ? ULLONG_MAX : lhs * rhs;
}

__device__ std::int32_t FindActiveEdgeId(const quick_hs::GraphView graph, const std::int32_t first,
                                         const std::int32_t second) {
  if (first == second) {
    return -1;
  }
  const std::int32_t u = first < second ? first : second;
  const std::int32_t v = first < second ? second : first;
  std::int64_t edge = 0;
  if (graph.complete_graph) {
    edge = static_cast<std::int64_t>(u) * (2LL * graph.dimension - u - 1) / 2 + (v - u - 1);
  } else {
    std::int64_t low = 0;
    std::int64_t high = graph.edge_count;
    while (low < high) {
      const std::int64_t middle = low + (high - low) / 2;
      if (graph.edge_u[middle] < u || (graph.edge_u[middle] == u && graph.edge_v[middle] < v)) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }
    edge = low;
  }
  return edge >= 0 && edge < graph.edge_count && graph.edge_u[edge] == u &&
                 graph.edge_v[edge] == v && graph.edge_active[edge] != 0U
             ? static_cast<std::int32_t>(edge)
             : -1;
}

__device__ bool LpClosesReplyWarp(const quick_hs::GraphView graph, const std::int32_t a,
                                  const std::int32_t b, const std::int32_t c1, const std::int32_t c,
                                  const std::int32_t c2, const std::int32_t d1,
                                  const std::int32_t d, const std::int32_t d2,
                                  const std::int64_t* const reduced_cost,
                                  const Signed128* const lower_bound,
                                  const std::int64_t incumbent_numerator,
                                  unsigned long long* const closed_replies) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  if (reduced_cost == nullptr || lower_bound == nullptr) {
    return false;
  }
  const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x & 31U);
  std::int32_t closed = 0;
  if (lane == 0) {
    const std::int32_t endpoint[5][2] = {{a, b}, {c1, c}, {c, c2}, {d1, d}, {d, d2}};
    std::int32_t edge_ids[5]{};
    std::int32_t unique_count = 0;
    bool complete = true;
    for (std::int32_t path_edge = 0; path_edge < 5; ++path_edge) {
      const std::int32_t edge =
          FindActiveEdgeId(graph, endpoint[path_edge][0], endpoint[path_edge][1]);
      if (edge < 0) {
        complete = false;
        break;
      }
      bool duplicate = false;
      for (std::int32_t previous = 0; previous < unique_count; ++previous) {
        duplicate = duplicate || edge_ids[previous] == edge;
      }
      if (!duplicate) {
        edge_ids[unique_count++] = edge;
      }
    }
    if (complete) {
      Signed128 forced = *lower_bound;
      for (std::int32_t index = 0; index < unique_count; ++index) {
        const std::int64_t reduced = reduced_cost[edge_ids[index]];
        forced = Signed128AddInt64(forced, reduced > 0 ? reduced : 0);
      }
      closed = Signed128GreaterThanInt64(forced, incumbent_numerator) ? 1 : 0;
      if (closed != 0 && closed_replies != nullptr) {
        atomicAdd(closed_replies, 1ULL);
      }
    }
  }
  return __shfl_sync(kFullWarp, closed, 0) != 0;
}

__device__ bool
LpClosesPointReplyWarp(const std::int32_t root_first_edge, const std::int32_t root_second_edge,
                       const std::int32_t point_first_edge, const std::int32_t point_second_edge,
                       const std::int64_t* const reduced_cost, const Signed128* const lower_bound,
                       const std::int64_t incumbent_numerator,
                       unsigned long long* const closed_replies) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  if (reduced_cost == nullptr || lower_bound == nullptr) {
    return false;
  }
  const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x & 31U);
  std::int32_t closed = 0;
  if (lane == 0) {
    const std::int32_t source_edge_ids[4] = {root_first_edge, root_second_edge, point_first_edge,
                                             point_second_edge};
    std::int32_t edge_ids[4]{};
    std::int32_t unique_count = 0;
    bool complete = true;
    for (std::int32_t path_edge = 0; path_edge < 4; ++path_edge) {
      const std::int32_t edge = source_edge_ids[path_edge];
      if (edge < 0) {
        complete = false;
        break;
      }
      bool duplicate = false;
      for (std::int32_t previous = 0; previous < unique_count; ++previous) {
        duplicate = duplicate || edge_ids[previous] == edge;
      }
      if (!duplicate) {
        edge_ids[unique_count++] = edge;
      }
    }
    if (complete) {
      Signed128 forced = *lower_bound;
      for (std::int32_t index = 0; index < unique_count; ++index) {
        const std::int64_t reduced = reduced_cost[edge_ids[index]];
        forced = Signed128AddInt64(forced, reduced > 0 ? reduced : 0);
      }
      closed = Signed128GreaterThanInt64(forced, incumbent_numerator) ? 1 : 0;
      if (closed != 0 && closed_replies != nullptr) {
        atomicAdd(closed_replies, 1ULL);
      }
    }
  }
  return __shfl_sync(kFullWarp, closed, 0) != 0;
}

__device__ bool LpClosesPointPathEndReplyWarp(
    const std::int32_t root_first_edge, const std::int32_t root_second_edge,
    const std::int32_t point_first_edge, const std::int32_t point_second_edge,
    const std::int32_t extension_edge, const std::int64_t* const reduced_cost,
    const Signed128* const lower_bound, const std::int64_t incumbent_numerator,
    unsigned long long* const closed_replies) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  if (reduced_cost == nullptr || lower_bound == nullptr) {
    return false;
  }
  const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x & 31U);
  std::int32_t closed = 0;
  if (lane == 0) {
    const std::int32_t source_edge_ids[5] = {root_first_edge, root_second_edge, point_first_edge,
                                             point_second_edge, extension_edge};
    std::int32_t edge_ids[5]{};
    std::int32_t unique_count = 0;
    bool complete = true;
    for (std::int32_t path_edge = 0; path_edge < 5; ++path_edge) {
      const std::int32_t edge = source_edge_ids[path_edge];
      if (edge < 0) {
        complete = false;
        break;
      }
      bool duplicate = false;
      for (std::int32_t previous = 0; previous < unique_count; ++previous) {
        duplicate = duplicate || edge_ids[previous] == edge;
      }
      if (!duplicate) {
        edge_ids[unique_count++] = edge;
      }
    }
    if (complete) {
      Signed128 forced = *lower_bound;
      for (std::int32_t index = 0; index < unique_count; ++index) {
        const std::int64_t reduced = reduced_cost[edge_ids[index]];
        forced = Signed128AddInt64(forced, reduced > 0 ? reduced : 0);
      }
      closed = Signed128GreaterThanInt64(forced, incumbent_numerator) ? 1 : 0;
      if (closed != 0 && closed_replies != nullptr) {
        atomicAdd(closed_replies, 1ULL);
      }
    }
  }
  return __shfl_sync(kFullWarp, closed, 0) != 0;
}

__device__ std::int32_t SmallFactorial(const std::int32_t value) {
  std::int32_t result = 1;
  for (std::int32_t factor = 2; factor <= value; ++factor) {
    result *= factor;
  }
  return result;
}

__device__ void UnrankSmallPermutation(std::int32_t rank, const std::int32_t count,
                                       std::uint8_t* const permutation) {
  std::uint8_t available[quick_hs::kMaxPathNodes - 2]{};
  for (std::int32_t index = 0; index < count; ++index) {
    available[index] = static_cast<std::uint8_t>(index);
  }
  std::int32_t remaining = count;
  for (std::int32_t position = 0; position < count; ++position) {
    const std::int32_t block = SmallFactorial(remaining - 1);
    const std::int32_t choice = block == 0 ? 0 : rank / block;
    rank = block == 0 ? 0 : rank % block;
    permutation[position] = available[choice];
    for (std::int32_t move = choice; move + 1 < remaining; ++move) {
      available[move] = available[move + 1];
    }
    --remaining;
  }
}

constexpr std::int32_t kMaximumWarpPathDistances =
    quick_hs::kMaxPathNodes * (quick_hs::kMaxPathNodes - 1) / 2;
constexpr std::int32_t kMaximumDynamicPathNodes = quick_hs::kMaxPathNodes - 2;
constexpr std::int32_t kMaximumWarpSubsetDpStates =
    (1 << kMaximumDynamicPathNodes) * kMaximumDynamicPathNodes;

__device__ std::int32_t WarpPathDistanceIndex(const std::int32_t node_count, std::int32_t first,
                                              std::int32_t second) {
  if (first > second) {
    const std::int32_t saved = first;
    first = second;
    second = saved;
  }
  return first * (2 * node_count - first - 1) / 2 + (second - first - 1);
}

__device__ std::int64_t WarpPathDistance(const std::int64_t* const cache,
                                         const std::int32_t node_count, const std::int32_t first,
                                         const std::int32_t second) {
  return first == second ? 0 : cache[WarpPathDistanceIndex(node_count, first, second)];
}

__device__ std::int64_t WarpPathTransitionCost(const std::int64_t* const cache,
                                               const std::int32_t node_count,
                                               const std::uint8_t* const fixed_after,
                                               const std::int32_t first, const std::int32_t second,
                                               const std::int64_t forced_cost) {
  const std::int32_t difference = first - second;
  if (difference == 1 || difference == -1) {
    const std::int32_t lower = first < second ? first : second;
    if (fixed_after[lower] != 0U) {
      return forced_cost;
    }
  }
  return WarpPathDistance(cache, node_count, first, second);
}

__device__ bool
PathOrderIsOptWarp(const quick_hs::GraphView graph, const quick_hs::SmallPath* const paths,
                   const std::int32_t path_count, const std::int32_t* const path_permutation,
                   const std::uint32_t reverse_mask, std::int64_t* const distance_cache,
                   std::int64_t* const subset_dp_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x & 31U);
  std::int32_t order[quick_hs::kMaxPathNodes]{};
  std::uint8_t fixed_after[quick_hs::kMaxPathNodes]{};
  std::int32_t total = 0;
  for (std::int32_t position = 0; position < path_count; ++position) {
    const quick_hs::SmallPath& path = paths[path_permutation[position]];
    const bool reverse = position > 0 && ((reverse_mask >> (position - 1)) & 1U) != 0U;
    for (std::int32_t index = 0; index < path.size; ++index) {
      order[total++] = path.node[reverse ? path.size - 1 - index : index];
    }
    fixed_after[total - 1] = 1U;
  }
  fixed_after[total - 1] = 0U;
  if (total < 3 || total > quick_hs::kMaxPathNodes) {
    return true;
  }
  std::int32_t distance_index = 0;
  for (std::int32_t first = 0; first < total; ++first) {
    for (std::int32_t second = first + 1; second < total; ++second, ++distance_index) {
      if ((distance_index & 31) == lane) {
        distance_cache[distance_index] = quick_hs::Distance(graph, order[first], order[second]);
      }
    }
  }
  __syncwarp(kFullWarp);
  const std::int32_t dynamic_nodes = total - 2;
  std::int64_t original = WarpPathDistance(distance_cache, total, 0, 1);
  std::int32_t outside_links = 0;
  for (std::int32_t position = 1; position < total - 1; ++position) {
    if (fixed_after[position] != 0U) {
      ++outside_links;
    } else {
      original += WarpPathDistance(distance_cache, total, position, position + 1);
    }
  }
  // L+1 严格支配所有真实内部路径成本；大坐标也必须保留全部外部连接。
  const std::int64_t forced_cost = -(original + 1);
  original += static_cast<std::int64_t>(outside_links) * forced_cost;

  if (subset_dp_cache != nullptr) {
    // -e2 的 10 点谓词若直接枚举会为每个 path orientation 扫描 8! 个
    // 顺序。这里由一个 warp 协作执行同一 Hamilton-path 最短路：状态
    // (mask,last) 按基数分层，层内无依赖，因此只需每层一次 warp 同步。
    // 负哨兵和严格小于比较均与 KH 的排列定义保持一致。
    constexpr std::int64_t kInfinity = LLONG_MAX / 4;
    const std::int32_t subset_count = 1 << dynamic_nodes;
    const std::int32_t state_count = subset_count * kMaximumDynamicPathNodes;
    for (std::int32_t state_index = lane; state_index < state_count; state_index += 32) {
      subset_dp_cache[state_index] = kInfinity;
    }
    __syncwarp(kFullWarp);
    if (lane < dynamic_nodes) {
      subset_dp_cache[(1 << lane) * kMaximumDynamicPathNodes + lane] =
          WarpPathDistance(distance_cache, total, lane + 1, 0);
    }
    __syncwarp(kFullWarp);

    for (std::int32_t cardinality = 2; cardinality <= dynamic_nodes; ++cardinality) {
      for (std::int32_t state_index = lane; state_index < state_count; state_index += 32) {
        const std::int32_t mask = state_index / kMaximumDynamicPathNodes;
        const std::int32_t last = state_index % kMaximumDynamicPathNodes;
        if (last >= dynamic_nodes || (mask & (1 << last)) == 0 ||
            __popc(static_cast<unsigned>(mask)) != cardinality) {
          continue;
        }
        const std::int32_t previous_mask = mask ^ (1 << last);
        std::int64_t best = kInfinity;
        for (std::int32_t previous = 0; previous < dynamic_nodes; ++previous) {
          if ((previous_mask & (1 << previous)) == 0) {
            continue;
          }
          const std::int64_t prefix =
              subset_dp_cache[previous_mask * kMaximumDynamicPathNodes + previous];
          const std::int64_t candidate =
              prefix + WarpPathTransitionCost(distance_cache, total, fixed_after, previous + 1,
                                              last + 1, forced_cost);
          best = candidate < best ? candidate : best;
        }
        subset_dp_cache[state_index] = best;
      }
      __syncwarp(kFullWarp);
    }

    const std::int32_t full_mask = subset_count - 1;
    std::int64_t best = kInfinity;
    if (lane < dynamic_nodes) {
      best = subset_dp_cache[full_mask * kMaximumDynamicPathNodes + lane] +
             WarpPathTransitionCost(distance_cache, total, fixed_after, lane + 1, total - 1,
                                    forced_cost);
    }
    for (std::int32_t offset = 16; offset > 0; offset >>= 1) {
      const std::int64_t other = __shfl_down_sync(kFullWarp, best, offset);
      best = other < best ? other : best;
    }
    best = __shfl_sync(kFullWarp, best, 0);
    const bool is_opt = best >= original;
    // 下一种 path orientation 会复用同一 warp 的 shared distance cache。
    // shuffle/vote 只同步寄存器通信，不能替代 shared-memory 的 WAR 屏障。
    __syncwarp(kFullWarp);
    return is_opt;
  }

  const std::int32_t permutation_count = SmallFactorial(dynamic_nodes);
  const auto permutation_offset = permutation_catalog::Offset(dynamic_nodes);
  for (std::int32_t window = 0; window < permutation_count; window += 32) {
    const std::int32_t rank = window + lane;
    bool improving = false;
    if (rank < permutation_count) {
      std::uint8_t dynamic_order[quick_hs::kMaxPathNodes - 2]{};
      if (graph.permutation_orders != nullptr &&
          dynamic_nodes <= permutation_catalog::kMaximumNodes) {
        // position-major 布局使同一位置的相邻 permutation rank 由 warp 合并读取。
        // 目录已在 bootstrap 独立检查；重放仍可从头计算，目录不缩减排列覆盖。
        for (int position = 0; position < dynamic_nodes; ++position)
          dynamic_order[position] =
              graph.permutation_orders[permutation_offset +
                                       static_cast<std::size_t>(position) * permutation_count +
                                       rank];
      } else {
        UnrankSmallPermutation(rank, dynamic_nodes, dynamic_order);
      }
      std::int64_t candidate = WarpPathDistance(distance_cache, total, dynamic_order[0] + 1, 0);
      for (std::int32_t position = 1; position < dynamic_nodes; ++position) {
        candidate += WarpPathTransitionCost(
            distance_cache, total, fixed_after,
            static_cast<std::int32_t>(dynamic_order[position - 1]) + 1,
            static_cast<std::int32_t>(dynamic_order[position]) + 1, forced_cost);
      }
      candidate += WarpPathTransitionCost(
          distance_cache, total, fixed_after,
          static_cast<std::int32_t>(dynamic_order[dynamic_nodes - 1]) + 1, total - 1, forced_cost);
      improving = candidate < original;
    }
    if (__any_sync(kFullWarp, improving)) {
      __syncwarp(kFullWarp);
      return false;
    }
  }
  __syncwarp(kFullWarp);
  return true;
}

__device__ bool PathSystemOptWarp(const quick_hs::GraphView graph,
                                  const quick_hs::SmallPath* const source_paths,
                                  const std::int32_t source_path_count, bool* const supported,
                                  std::int64_t* const distance_cache,
                                  std::int64_t* const subset_dp_cache = nullptr) {
  // 与 quick_hs::Opt/作者 opt 完全相同的 fixed-endpoint 合并。每个 lane
  // 复制并执行同一小数组状态机，随后共同进入距离缓存和 subset DP；这样
  // 度数为 2 或端点重合不再退化为 lane 0 的串行阶乘枚举。
  *supported = true;
  quick_hs::SmallPath paths[quick_hs::kMaxPathCount]{};
  if (source_path_count <= 0 || source_path_count > quick_hs::kMaxPathCount) {
    *supported = false;
    return true;
  }
  std::int32_t path_count = source_path_count;
  for (std::int32_t path = 0; path < path_count; ++path) {
    paths[path] = source_paths[path];
  }

  bool merged = true;
  while (merged) {
    merged = false;
    for (std::int32_t i = 1; i < path_count && !merged; ++i) {
      const std::int32_t b1 = paths[i].node[0];
      const std::int32_t b2 = paths[i].node[paths[i].size - 1];
      if ((b1 == b2 && paths[i].size > 1) ||
          (paths[i].size > 2 && quick_hs::Fixed(graph, b1, b2))) {
        return quick_hs::ClosedPathsMayCoverWholeGraph(graph, paths[i]);
      }
      for (std::int32_t j = 0; j < i && !merged; ++j) {
        const std::int32_t a1 = paths[j].node[0];
        const std::int32_t a2 = paths[j].node[paths[j].size - 1];
        if (j == 0 && ((a1 == a2 && paths[j].size > 1) ||
                       (paths[j].size > 2 && quick_hs::Fixed(graph, a1, a2)))) {
          return quick_hs::ClosedPathsMayCoverWholeGraph(graph, paths[j]);
        }

        std::int32_t first_direction = 1;
        std::int32_t second_direction = 1;
        bool join = false;
        bool shared = false;
        if (a2 == b1 || (a2 != b2 && b1 != a1 && quick_hs::Fixed(graph, a2, b1))) {
          if (a1 == b2) {
            return quick_hs::ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
          }
          join = true;
          shared = a2 == b1;
        } else if (a2 == b2 || (a2 != b1 && b2 != a1 && quick_hs::Fixed(graph, a2, b2))) {
          if (a1 == b1) {
            return quick_hs::ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
          }
          join = true;
          shared = a2 == b2;
          second_direction = -1;
        } else if (a1 == b1 || (a1 != b2 && b1 != a2 && quick_hs::Fixed(graph, a1, b1))) {
          if (a2 == b2) {
            return quick_hs::ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
          }
          join = true;
          shared = a1 == b1;
          first_direction = -1;
        } else if (a1 == b2 || (a1 != b1 && b2 != a2 && quick_hs::Fixed(graph, a1, b2))) {
          if (a2 == b1) {
            return quick_hs::ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
          }
          join = true;
          shared = a1 == b2;
          first_direction = -1;
          second_direction = -1;
        }
        if (!join) {
          continue;
        }

        quick_hs::SmallPath combined{};
        for (std::int32_t index = 0; index < paths[j].size; ++index) {
          combined.node[combined.size++] =
              paths[j].node[first_direction > 0 ? index : paths[j].size - 1 - index];
        }
        const std::int32_t start = shared ? 1 : 0;
        for (std::int32_t index = start; index < paths[i].size; ++index) {
          if (combined.size >= quick_hs::kMaxPathNodes) {
            return true;
          }
          combined.node[combined.size++] =
              paths[i].node[second_direction > 0 ? index : paths[i].size - 1 - index];
        }
        if (i != path_count - 1) {
          paths[i] = paths[path_count - 1];
        }
        paths[j] = combined;
        --path_count;
        merged = true;
      }
    }
  }

  if (quick_hs::HasAmbiguousPathOverlap(paths, path_count)) {
    // 未被 endpoint normalization 消除的内部重叠不属于 path-order
    // oracle 的支持域；保持 reply 开放，禁止以“unsupported”授权删除。
    return true;
  }

  std::int32_t endpoints[2 * quick_hs::kMaxPathCount]{};
  for (std::int32_t path = 0; path < path_count; ++path) {
    endpoints[2 * path] = paths[path].node[0];
    endpoints[2 * path + 1] = paths[path].node[paths[path].size - 1];
  }
  if (quick_hs::HasCycle(endpoints, path_count)) {
    return quick_hs::ClosedPathsMayCoverWholeGraph(graph, paths[0],
                                                   path_count > 1 ? &paths[1] : nullptr,
                                                   path_count > 2 ? &paths[2] : nullptr);
  }

  std::int32_t path_permutation[quick_hs::kMaxPathCount]{0, 1, 2};
  const std::int32_t permutation_count = path_count == 3 ? 2 : 1;
  const std::uint32_t orientation_count = 1U << (path_count > 0 ? path_count - 1 : 0);
  for (std::int32_t permutation = 0; permutation < permutation_count; ++permutation) {
    if (permutation == 1) {
      path_permutation[1] = 2;
      path_permutation[2] = 1;
    }
    for (std::uint32_t reverse = 0U; reverse < orientation_count; ++reverse) {
      if (PathOrderIsOptWarp(graph, paths, path_count, path_permutation, reverse, distance_cache,
                             subset_dp_cache)) {
        return true;
      }
    }
  }
  return false;
}

__device__ bool Opt243Warp(const quick_hs::GraphView graph, const std::int32_t a,
                           const std::int32_t b, const std::int32_t c1, const std::int32_t c,
                           const std::int32_t c2, const std::int32_t c3, const std::int32_t d1,
                           const std::int32_t d, const std::int32_t d2,
                           std::int64_t* const distance_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  // 与作者 opt243 相同的五组必要条件属于证明语义，不能因为后面使用
  // 精确 path-order DP 就省略。
  if (!quick_hs::Opt23(graph, c1, c, c, c2, c3, quick_hs::Distance(graph, c1, c),
                       quick_hs::Distance(graph, c, c2), quick_hs::Distance(graph, c2, c3)) ||
      !quick_hs::Opt23(graph, c1, c, d1, d, d2, quick_hs::Distance(graph, c1, c),
                       quick_hs::Distance(graph, d1, d), quick_hs::Distance(graph, d, d2)) ||
      !quick_hs::Opt23(graph, d1, d, c1, c, c2, quick_hs::Distance(graph, d1, d),
                       quick_hs::Distance(graph, c1, c), quick_hs::Distance(graph, c, c2)) ||
      !quick_hs::Opt23(graph, d, d2, c1, c, c2, quick_hs::Distance(graph, d, d2),
                       quick_hs::Distance(graph, c1, c), quick_hs::Distance(graph, c, c2)) ||
      !quick_hs::Opt23(graph, a, b, c1, c, c2, quick_hs::Distance(graph, a, b),
                       quick_hs::Distance(graph, c1, c), quick_hs::Distance(graph, c, c2))) {
    return false;
  }
  const quick_hs::SmallPath paths[3] = {
      {.size = 2, .node = {a, b}},
      {.size = 4, .node = {c1, c, c2, c3}},
      {.size = 3, .node = {d1, d, d2}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(graph, paths, 3, &supported, distance_cache);
  if (!supported) {
    result =
        (threadIdx.x & 31U) == 0U ? quick_hs::Opt243(graph, a, b, c1, c, c2, c3, d1, d, d2) : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

__device__ bool Opt333Warp(const quick_hs::GraphView graph, const std::int32_t a,
                           const std::int32_t b, const std::int32_t b2, const std::int32_t c1,
                           const std::int32_t c2, const std::int32_t c3, const std::int32_t d1,
                           const std::int32_t d2, const std::int32_t d3,
                           std::int64_t* const distance_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const quick_hs::SmallPath paths[3] = {
      {.size = 3, .node = {a, b, b2}},
      {.size = 3, .node = {c1, c2, c3}},
      {.size = 3, .node = {d1, d2, d3}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(graph, paths, 3, &supported, distance_cache);
  if (!supported) {
    result = (threadIdx.x & 31U) == 0U ? quick_hs::Opt333(graph, a, b, b2, c1, c2, c3, d1, d2, d3)
                                       : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

__device__ bool Opt244Warp(const quick_hs::GraphView graph, const std::int32_t a,
                           const std::int32_t b, const std::int32_t c1, const std::int32_t c2,
                           const std::int32_t c3, const std::int32_t c4, const std::int32_t d1,
                           const std::int32_t d2, const std::int32_t d3, const std::int32_t d4,
                           std::int64_t* const distance_cache,
                           std::int64_t* const subset_dp_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const std::int32_t roles[10] = {a, b, c1, c2, c3, c4, d1, d2, d3, d4};
  if (!quick_hs::AllNodesDistinct(roles, 10)) {
    return true;
  }
  if (!quick_hs::Opt23(graph, c1, c2, c2, c3, c4, quick_hs::Distance(graph, c1, c2),
                       quick_hs::Distance(graph, c2, c3), quick_hs::Distance(graph, c3, c4)) ||
      !quick_hs::Opt23(graph, c1, c2, d1, d2, d3, quick_hs::Distance(graph, c1, c2),
                       quick_hs::Distance(graph, d1, d2), quick_hs::Distance(graph, d2, d3)) ||
      !quick_hs::Opt23(graph, c1, c2, d2, d3, d4, quick_hs::Distance(graph, c1, c2),
                       quick_hs::Distance(graph, d2, d3), quick_hs::Distance(graph, d3, d4)) ||
      !quick_hs::Opt23(graph, d1, d2, c1, c2, c3, quick_hs::Distance(graph, d1, d2),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3)) ||
      !quick_hs::Opt23(graph, d2, d3, c1, c2, c3, quick_hs::Distance(graph, d2, d3),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3)) ||
      !quick_hs::Opt23(graph, d3, d4, c1, c2, c3, quick_hs::Distance(graph, d3, d4),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3)) ||
      !quick_hs::Opt23(graph, a, b, c1, c2, c3, quick_hs::Distance(graph, a, b),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3))) {
    return false;
  }
  const quick_hs::SmallPath paths[3] = {
      {.size = 2, .node = {a, b}},
      {.size = 4, .node = {c1, c2, c3, c4}},
      {.size = 4, .node = {d1, d2, d3, d4}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(graph, paths, 3, &supported, distance_cache, subset_dp_cache);
  if (!supported) {
    result = (threadIdx.x & 31U) == 0U
                 ? quick_hs::Opt244(graph, a, b, c1, c2, c3, c4, d1, d2, d3, d4)
                 : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

__device__ bool Opt253Warp(const quick_hs::GraphView graph, const std::int32_t a,
                           const std::int32_t b, const std::int32_t c1, const std::int32_t c2,
                           const std::int32_t c3, const std::int32_t c4, const std::int32_t c5,
                           const std::int32_t d1, const std::int32_t d2, const std::int32_t d3,
                           std::int64_t* const distance_cache,
                           std::int64_t* const subset_dp_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const std::int32_t roles[10] = {a, b, c1, c2, c3, c4, c5, d1, d2, d3};
  if (!quick_hs::AllNodesDistinct(roles, 10)) {
    return true;
  }
  if (!quick_hs::Opt23(graph, c1, c2, c2, c3, c4, quick_hs::Distance(graph, c1, c2),
                       quick_hs::Distance(graph, c2, c3), quick_hs::Distance(graph, c3, c4)) ||
      !quick_hs::Opt23(graph, c1, c2, c3, c4, c5, quick_hs::Distance(graph, c1, c2),
                       quick_hs::Distance(graph, c3, c4), quick_hs::Distance(graph, c4, c5)) ||
      !quick_hs::Opt23(graph, c1, c2, d1, d2, d3, quick_hs::Distance(graph, c1, c2),
                       quick_hs::Distance(graph, d1, d2), quick_hs::Distance(graph, d2, d3)) ||
      !quick_hs::Opt23(graph, d1, d2, c1, c2, c3, quick_hs::Distance(graph, d1, d2),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3)) ||
      !quick_hs::Opt23(graph, d2, d3, c1, c2, c3, quick_hs::Distance(graph, d2, d3),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3)) ||
      !quick_hs::Opt23(graph, a, b, c1, c2, c3, quick_hs::Distance(graph, a, b),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3))) {
    return false;
  }
  const quick_hs::SmallPath paths[3] = {
      {.size = 2, .node = {a, b}},
      {.size = 5, .node = {c1, c2, c3, c4, c5}},
      {.size = 3, .node = {d1, d2, d3}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(graph, paths, 3, &supported, distance_cache, subset_dp_cache);
  if (!supported) {
    result = (threadIdx.x & 31U) == 0U
                 ? quick_hs::Opt253(graph, a, b, c1, c2, c3, c4, c5, d1, d2, d3)
                 : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

__device__ bool Opt343Warp(const quick_hs::GraphView graph, const std::int32_t a1,
                           const std::int32_t a2, const std::int32_t a3, const std::int32_t b1,
                           const std::int32_t b2, const std::int32_t b3, const std::int32_t b4,
                           const std::int32_t c1, const std::int32_t c2, const std::int32_t c3,
                           std::int64_t* const distance_cache,
                           std::int64_t* const subset_dp_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const std::int32_t roles[10] = {a1, a2, a3, b1, b2, b3, b4, c1, c2, c3};
  if (!quick_hs::AllNodesDistinct(roles, 10)) {
    return true;
  }
  if (!quick_hs::Opt23(graph, b1, b2, b2, b3, b4, quick_hs::Distance(graph, b1, b2),
                       quick_hs::Distance(graph, b2, b3), quick_hs::Distance(graph, b3, b4)) ||
      !quick_hs::Opt23(graph, b1, b2, a1, a2, a3, quick_hs::Distance(graph, b1, b2),
                       quick_hs::Distance(graph, a1, a2), quick_hs::Distance(graph, a2, a3)) ||
      !quick_hs::Opt23(graph, b1, b2, c1, c2, c3, quick_hs::Distance(graph, b1, b2),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3)) ||
      !quick_hs::Opt23(graph, a1, a2, b1, b2, b3, quick_hs::Distance(graph, a1, a2),
                       quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
      !quick_hs::Opt23(graph, a2, a3, b1, b2, b3, quick_hs::Distance(graph, a2, a3),
                       quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
      !quick_hs::Opt23(graph, c1, c2, b1, b2, b3, quick_hs::Distance(graph, c1, c2),
                       quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
      !quick_hs::Opt23(graph, c2, c3, b1, b2, b3, quick_hs::Distance(graph, c2, c3),
                       quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
      !quick_hs::Opt23(graph, a2, a3, b2, b3, b4, quick_hs::Distance(graph, a2, a3),
                       quick_hs::Distance(graph, b2, b3), quick_hs::Distance(graph, b3, b4)) ||
      !quick_hs::Opt23(graph, a2, a3, c1, c2, c3, quick_hs::Distance(graph, a2, a3),
                       quick_hs::Distance(graph, c1, c2), quick_hs::Distance(graph, c2, c3))) {
    return false;
  }
  const quick_hs::SmallPath paths[3] = {
      {.size = 3, .node = {a1, a2, a3}},
      {.size = 4, .node = {b1, b2, b3, b4}},
      {.size = 3, .node = {c1, c2, c3}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(graph, paths, 3, &supported, distance_cache, subset_dp_cache);
  if (!supported) {
    result = (threadIdx.x & 31U) == 0U
                 ? quick_hs::Opt343(graph, a1, a2, a3, b1, b2, b3, b4, c1, c2, c3)
                 : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

__device__ bool Opt233Warp(const quick_hs::GraphView graph, const std::int32_t a,
                           const std::int32_t b, const std::int32_t c1, const std::int32_t c,
                           const std::int32_t c2, const std::int32_t d1, const std::int32_t d,
                           const std::int32_t d2, std::int64_t* const distance_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const quick_hs::SmallPath paths[3] = {
      {.size = 2, .node = {a, b}},
      {.size = 3, .node = {c1, c, c2}},
      {.size = 3, .node = {d1, d, d2}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(graph, paths, 3, &supported, distance_cache);
  if (!supported) {
    result =
        (threadIdx.x & 31U) == 0U ? quick_hs::Opt233(graph, a, b, c1, c, c2, d1, d, d2) : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

__device__ bool Opt33Warp(const quick_hs::GraphView graph, const std::int32_t a1,
                          const std::int32_t a2, const std::int32_t a3, const std::int32_t b1,
                          const std::int32_t b2, const std::int32_t b3,
                          std::int64_t* const distance_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const std::int32_t nodes[6] = {a1, a2, a3, b1, b2, b3};
  bool all_distinct = true;
  for (std::int32_t first = 0; first < 6; ++first) {
    for (std::int32_t second = first + 1; second < 6; ++second) {
      all_distinct = all_distinct && nodes[first] != nodes[second];
    }
  }
  if (all_distinct &&
      (!quick_hs::Opt23(graph, a1, a2, b1, b2, b3, quick_hs::Distance(graph, a1, a2),
                        quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
       !quick_hs::Opt23(graph, a2, a3, b1, b2, b3, quick_hs::Distance(graph, a2, a3),
                        quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
       !quick_hs::Opt23(graph, b1, b2, a1, a2, a3, quick_hs::Distance(graph, b1, b2),
                        quick_hs::Distance(graph, a1, a2), quick_hs::Distance(graph, a2, a3)) ||
       !quick_hs::Opt23(graph, b2, b3, a1, a2, a3, quick_hs::Distance(graph, b2, b3),
                        quick_hs::Distance(graph, a1, a2), quick_hs::Distance(graph, a2, a3)))) {
    return false;
  }
  // 通过作者门禁后，再由 cooperative warp-DP 精确求解 3+3 path
  // system；共享端点仍交给通用合并逻辑处理，避免改变重叠路径语义。
  const quick_hs::SmallPath paths[quick_hs::kMaxPathCount] = {
      {.size = 3, .node = {a1, a2, a3}},
      {.size = 3, .node = {b1, b2, b3}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(graph, paths, 2, &supported, distance_cache);
  if (!supported) {
    result = (threadIdx.x & 31U) == 0U ? quick_hs::Opt33(graph, a1, a2, a3, b1, b2, b3) : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

__device__ bool Opt34Warp(const quick_hs::GraphView graph, const std::int32_t a1,
                          const std::int32_t a2, const std::int32_t a3, const std::int32_t b1,
                          const std::int32_t b2, const std::int32_t b3, const std::int32_t b4,
                          std::int64_t* const distance_cache, std::int64_t* const subset_dp_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const std::int32_t nodes[7] = {a1, a2, a3, b1, b2, b3, b4};
  if (graph.point_leaf_kernel != PointLeafKernel::kPermutation &&
      quick_hs::AllNodesDistinct(nodes, 7) &&
      (!quick_hs::Opt23(graph, a1, a2, b1, b2, b3, quick_hs::Distance(graph, a1, a2),
                        quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
       !quick_hs::Opt23(graph, a1, a2, b2, b3, b4, quick_hs::Distance(graph, a1, a2),
                        quick_hs::Distance(graph, b2, b3), quick_hs::Distance(graph, b3, b4)) ||
       !quick_hs::Opt23(graph, a2, a3, b1, b2, b3, quick_hs::Distance(graph, a2, a3),
                        quick_hs::Distance(graph, b1, b2), quick_hs::Distance(graph, b2, b3)) ||
       !quick_hs::Opt23(graph, a2, a3, b2, b3, b4, quick_hs::Distance(graph, a2, a3),
                        quick_hs::Distance(graph, b2, b3), quick_hs::Distance(graph, b3, b4)) ||
       !quick_hs::Opt23(graph, b1, b2, a1, a2, a3, quick_hs::Distance(graph, b1, b2),
                        quick_hs::Distance(graph, a1, a2), quick_hs::Distance(graph, a2, a3)) ||
       !quick_hs::Opt23(graph, b2, b3, a1, a2, a3, quick_hs::Distance(graph, b2, b3),
                        quick_hs::Distance(graph, a1, a2), quick_hs::Distance(graph, a2, a3)) ||
       !quick_hs::Opt23(graph, b3, b4, a1, a2, a3, quick_hs::Distance(graph, b3, b4),
                        quick_hs::Distance(graph, a1, a2), quick_hs::Distance(graph, a2, a3)))) {
    return false;
  }
  const quick_hs::SmallPath paths[quick_hs::kMaxPathCount] = {
      {.size = 3, .node = {a1, a2, a3}},
      {.size = 4, .node = {b1, b2, b3, b4}},
  };
  bool supported = false;
  bool result = PathSystemOptWarp(
      graph, paths, 2, &supported, distance_cache,
      graph.point_leaf_kernel == PointLeafKernel::kPrescreenSubsetDp ? subset_dp_cache : nullptr);
  if (!supported) {
    result = (threadIdx.x & 31U) == 0U ? quick_hs::Opt34(graph, a1, a2, a3, b1, b2, b3, b4) : false;
    result = __shfl_sync(kFullWarp, result, 0);
  }
  return result;
}

// KH -e2 的第二层 endpoint reveal。state 保留原实现的规范顺序：第一层
// 已经处理过的 endpoint 不重复展开，后续 endpoint 仍保持完整 AND 语义。
__device__ bool ExtraExtraEdge243Warp(const quick_hs::GraphView graph, const std::int32_t a,
                                      const std::int32_t b, const std::int32_t c1,
                                      const std::int32_t c2, const std::int32_t c3,
                                      const std::int32_t c4, const std::int32_t d1,
                                      const std::int32_t d2, const std::int32_t d3,
                                      const std::int32_t state, std::int64_t* const distance_cache,
                                      std::int64_t* const subset_dp_cache) {
  std::int32_t current_state = 0;
  const std::int32_t c_path[2][4] = {{c1, c2, c3, c4}, {c4, c3, c2, c1}};
  for (std::int32_t orientation = 0; orientation < 2; ++orientation) {
    if (++current_state < state) {
      continue;
    }
    const std::int32_t endpoint = c_path[orientation][0];
    if (endpoint == a || endpoint == b || endpoint == d1 || endpoint == d3) {
      continue;
    }
    bool unresolved = false;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == c_path[orientation][1] || extension == c_path[orientation][2] ||
          extension == c_path[orientation][3] || extension == d2 ||
          quick_hs::PairForbidden(graph, endpoint, c_path[orientation][1], extension)) {
        continue;
      }
      if (Opt253Warp(graph, a, b, extension, c_path[orientation][0], c_path[orientation][1],
                     c_path[orientation][2], c_path[orientation][3], d1, d2, d3, distance_cache,
                     subset_dp_cache)) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }

  const std::int32_t d_path[2][3] = {{d1, d2, d3}, {d3, d2, d1}};
  for (std::int32_t orientation = 0; orientation < 2; ++orientation) {
    if (++current_state < state) {
      continue;
    }
    const std::int32_t endpoint = d_path[orientation][0];
    if (endpoint == a || endpoint == b || endpoint == c1 || endpoint == c4) {
      continue;
    }
    bool unresolved = false;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == d_path[orientation][1] || extension == d_path[orientation][2] ||
          extension == c2 || extension == c3 ||
          quick_hs::PairForbidden(graph, endpoint, d_path[orientation][1], extension)) {
        continue;
      }
      if (Opt244Warp(graph, a, b, extension, d_path[orientation][0], d_path[orientation][1],
                     d_path[orientation][2], c1, c2, c3, c4, distance_cache, subset_dp_cache)) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }

  const std::int32_t target[2][2] = {{a, b}, {b, a}};
  for (std::int32_t orientation = 0; orientation < 2; ++orientation) {
    if (++current_state < state) {
      continue;
    }
    const std::int32_t opposite = target[orientation][0];
    const std::int32_t endpoint = target[orientation][1];
    if (endpoint == c1 || endpoint == c4 || endpoint == d1 || endpoint == d3) {
      continue;
    }
    bool unresolved = false;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == opposite || extension == c2 || extension == c3 || extension == d2 ||
          quick_hs::PairForbidden(graph, endpoint, opposite, extension)) {
        continue;
      }
      if (Opt343Warp(graph, opposite, endpoint, extension, c1, c2, c3, c4, d1, d2, d3,
                     distance_cache, subset_dp_cache)) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }
  return true;
}

__device__ bool ExtraExtraEdge333Warp(const quick_hs::GraphView graph, const std::int32_t a1,
                                      const std::int32_t a2, const std::int32_t a3,
                                      const std::int32_t c1, const std::int32_t c2,
                                      const std::int32_t c3, const std::int32_t d1,
                                      const std::int32_t d2, const std::int32_t d3,
                                      const std::int32_t state, std::int64_t* const distance_cache,
                                      std::int64_t* const subset_dp_cache) {
  std::int32_t current_state = 0;
  const std::int32_t c_path[2][3] = {{c1, c2, c3}, {c3, c2, c1}};
  for (std::int32_t orientation = 0; orientation < 2; ++orientation) {
    if (++current_state < state) {
      continue;
    }
    const std::int32_t endpoint = c_path[orientation][0];
    if (endpoint == a1 || endpoint == a3 || endpoint == d1 || endpoint == d3) {
      continue;
    }
    bool unresolved = false;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == c_path[orientation][1] || extension == c_path[orientation][2] ||
          extension == a2 || extension == d2 ||
          quick_hs::PairForbidden(graph, endpoint, c_path[orientation][1], extension)) {
        continue;
      }
      if (Opt343Warp(graph, a1, a2, a3, extension, c_path[orientation][0], c_path[orientation][1],
                     c_path[orientation][2], d1, d2, d3, distance_cache, subset_dp_cache)) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }

  const std::int32_t d_path[2][3] = {{d1, d2, d3}, {d3, d2, d1}};
  for (std::int32_t orientation = 0; orientation < 2; ++orientation) {
    if (++current_state < state) {
      continue;
    }
    const std::int32_t endpoint = d_path[orientation][0];
    if (endpoint == a1 || endpoint == a3 || endpoint == c1 || endpoint == c3) {
      continue;
    }
    bool unresolved = false;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == d_path[orientation][1] || extension == d_path[orientation][2] ||
          extension == a2 || extension == c2 ||
          quick_hs::PairForbidden(graph, endpoint, d_path[orientation][1], extension)) {
        continue;
      }
      if (Opt343Warp(graph, a1, a2, a3, extension, d_path[orientation][0], d_path[orientation][1],
                     d_path[orientation][2], c1, c2, c3, distance_cache, subset_dp_cache)) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }

  const std::int32_t a_path[2][3] = {{a1, a2, a3}, {a3, a2, a1}};
  for (std::int32_t orientation = 0; orientation < 2; ++orientation) {
    if (++current_state < state) {
      continue;
    }
    const std::int32_t endpoint = a_path[orientation][0];
    if (endpoint == c1 || endpoint == c3 || endpoint == d1 || endpoint == d3) {
      continue;
    }
    bool unresolved = false;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == a_path[orientation][1] || extension == a_path[orientation][2] ||
          extension == c2 || extension == d2 ||
          quick_hs::PairForbidden(graph, endpoint, a_path[orientation][1], extension)) {
        continue;
      }
      if (Opt343Warp(graph, c1, c2, c3, extension, a_path[orientation][0], a_path[orientation][1],
                     a_path[orientation][2], d1, d2, d3, distance_cache, subset_dp_cache)) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }
  return true;
}

template <bool EnableExtraEdge2>
__device__ bool
ExtraEdgeOpt1Warp(const quick_hs::GraphView graph, const std::int32_t a, const std::int32_t b,
                  const std::int32_t c1, const std::int32_t c, const std::int32_t c2,
                  const std::int32_t d1, const std::int32_t d, const std::int32_t d2,
                  std::int64_t* const distance_cache, std::int64_t* const subset_dp_cache) {
  const std::int32_t path_endpoints[2][2] = {{c1, c2}, {d1, d2}};
  const std::int32_t path_centers[2] = {c, d};
  std::int32_t state = 0;
  for (std::int32_t path = 0; path < 2; ++path) {
    const std::int32_t other = 1 - path;
    for (std::int32_t side = 0; side < 2; ++side) {
      ++state;
      const std::int32_t endpoint = path_endpoints[path][side];
      const std::int32_t opposite = path_endpoints[path][1 - side];
      if (endpoint == a || endpoint == b || endpoint == path_endpoints[other][0] ||
          endpoint == path_endpoints[other][1]) {
        continue;
      }
      bool unresolved = false;
      const std::int64_t begin = quick_hs::NeighborBegin(graph, endpoint);
      const std::int64_t end = quick_hs::NeighborEnd(graph, endpoint);
      for (std::int64_t slot = begin; slot < end; ++slot) {
        if (!quick_hs::NeighborActive(graph, slot)) {
          continue;
        }
        const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
        if (extension == path_centers[path] || extension == opposite ||
            extension == path_centers[other] ||
            quick_hs::PairForbidden(graph, endpoint, path_centers[path], extension)) {
          continue;
        }
        bool admits = Opt243Warp(graph, a, b, extension, endpoint, path_centers[path], opposite,
                                 path_endpoints[other][0], path_centers[other],
                                 path_endpoints[other][1], distance_cache);
        if constexpr (EnableExtraEdge2) {
          admits = admits && ExtraExtraEdge243Warp(
                                 graph, a, b, extension, endpoint, path_centers[path], opposite,
                                 path_endpoints[other][0], path_centers[other],
                                 path_endpoints[other][1], state, distance_cache, subset_dp_cache);
        }
        if (admits) {
          unresolved = true;
          break;
        }
      }
      if (!unresolved) {
        return false;
      }
    }
  }

  const std::int32_t target_endpoints[2] = {b, a};
  const std::int32_t target_opposites[2] = {a, b};
  for (std::int32_t side = 0; side < 2; ++side) {
    ++state;
    const std::int32_t endpoint = target_endpoints[side];
    const std::int32_t opposite = target_opposites[side];
    if (endpoint == c1 || endpoint == c2 || endpoint == d1 || endpoint == d2) {
      continue;
    }
    bool unresolved = false;
    const std::int64_t begin = quick_hs::NeighborBegin(graph, endpoint);
    const std::int64_t end = quick_hs::NeighborEnd(graph, endpoint);
    for (std::int64_t slot = begin; slot < end; ++slot) {
      if (!quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == opposite || extension == c || extension == d ||
          quick_hs::PairForbidden(graph, endpoint, opposite, extension)) {
        continue;
      }
      bool admits =
          Opt333Warp(graph, opposite, endpoint, extension, c1, c, c2, d1, d, d2, distance_cache);
      if constexpr (EnableExtraEdge2) {
        admits = admits && ExtraExtraEdge333Warp(graph, opposite, endpoint, extension, c1, c, c2,
                                                 d1, d, d2, state, distance_cache, subset_dp_cache);
      }
      if (admits) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }
  return true;
}

// AND replies 在 CTA 内按 warp 并行：一个 warp 负责一个基础 reply，
// lane 再并行其六个 endpoint 的 extension 枚举。OR 候选仍严格顺序短路。
template <std::int32_t ExtraEdgeDepth>
__device__ bool
RepliesClosedCta(const quick_hs::GraphView graph, const std::int32_t a, const std::int32_t b,
                 const std::int32_t c, const std::int32_t d, const std::int64_t* const reduced_cost,
                 const Signed128* const lower_bound, const std::int64_t incumbent_numerator,
                 unsigned long long* const lp_closed_replies) {
  const std::uint64_t c_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, c) -
                                                            quick_hs::NeighborBegin(graph, c));
  const std::uint64_t d_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, d) -
                                                            quick_hs::NeighborBegin(graph, d));
  const std::uint64_t c_pairs = c_degree < 2U ? 0U : c_degree * (c_degree - 1U) / 2U;
  const std::uint64_t d_pairs = d_degree < 2U ? 0U : d_degree * (d_degree - 1U) / 2U;
  if (d_pairs != 0U && c_pairs > ULLONG_MAX / d_pairs) {
    // 极端高度数下不能完整编号 reply 空间，保守地不授权。
    return false;
  }
  const std::uint64_t reply_count = c_pairs * d_pairs;
  // 深度 2 每个活动 warp 需要 16 KiB 精确 subset-DP 工作区。保留两个
  // 活动 warp 后，连同候选排序共享内存仍低于默认 48 KiB 上限；其余线程
  // 继续参加 CTA barrier，但不领取 reply。
  constexpr std::uint32_t kWarpsPerBlock = ExtraEdgeDepth >= 2 ? 2U : 4U;
  constexpr std::int32_t kSubsetDpStatesPerBlock =
      ExtraEdgeDepth >= 2 ? static_cast<std::int32_t>(kWarpsPerBlock) * kMaximumWarpSubsetDpStates
                          : 1;
  __shared__ std::int64_t path_distance_cache[kWarpsPerBlock * kMaximumWarpPathDistances];
  __shared__ std::int64_t subset_dp_cache[kSubsetDpStatesPerBlock];
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t lane = threadIdx.x & 31U;
  for (std::uint64_t window = 0; window < reply_count; window += kWarpsPerBlock) {
    const std::uint64_t reply = warp < kWarpsPerBlock ? window + warp : reply_count;
    std::int32_t c1 = -1;
    std::int32_t c2 = -1;
    std::int32_t d1 = -1;
    std::int32_t d2 = -1;
    std::int32_t decoded = 0;
    if (lane == 0U && reply < reply_count) {
      const std::uint64_t c_ordinal = reply / d_pairs;
      const std::uint64_t d_ordinal = reply % d_pairs;
      decoded = DecodeNeighborPair(graph, c, c_ordinal, &c1, &c2) &&
                DecodeNeighborPair(graph, d, d_ordinal, &d1, &d2);
    }
    constexpr unsigned kFullWarp = 0xffffffffU;
    decoded = __shfl_sync(kFullWarp, decoded, 0);
    c1 = __shfl_sync(kFullWarp, c1, 0);
    c2 = __shfl_sync(kFullWarp, c2, 0);
    d1 = __shfl_sync(kFullWarp, d1, 0);
    d2 = __shfl_sync(kFullWarp, d2, 0);
    const bool passes_fast =
        decoded != 0 &&
        quick_hs::ReplyPassesFastFilters<(ExtraEdgeDepth >= 1)>(graph, a, b, c1, c, c2, d1, d, d2);
    const bool lp_closed =
        passes_fast && LpClosesReplyWarp(graph, a, b, c1, c, c2, d1, d, d2, reduced_cost,
                                         lower_bound, incumbent_numerator, lp_closed_replies);
    bool admits_tour = passes_fast && !lp_closed &&
                       Opt233Warp(graph, a, b, c1, c, c2, d1, d, d2,
                                  path_distance_cache + warp * kMaximumWarpPathDistances);
    if constexpr (ExtraEdgeDepth >= 1) {
      if (admits_tour) {
        std::int64_t* warp_subset_dp = nullptr;
        if constexpr (ExtraEdgeDepth >= 2) {
          warp_subset_dp = subset_dp_cache + warp * kMaximumWarpSubsetDpStates;
        }
        admits_tour = ExtraEdgeOpt1Warp<(ExtraEdgeDepth >= 2)>(
            graph, a, b, c1, c, c2, d1, d, d2,
            path_distance_cache + warp * kMaximumWarpPathDistances, warp_subset_dp);
      }
    }
    if (__syncthreads_or(lane == 0U && admits_tour ? 1 : 0) != 0) {
      return false;
    }
  }
  return true;
}

// 深层 -e1 使用 CTA-per-target continuation，避免旧实现由单线程串行
// 扫描 |R(c)|*|R(d)| 产生的长尾。
template <std::int32_t ExtraEdgeDepth>
__global__ void QuickHsContinuationKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
    const quick_hs::GraphView graph, std::uint8_t* const proposed,
    std::int32_t* const first_witness, std::int32_t* const second_witness,
    const std::int32_t candidate_limit, const std::int32_t pair_trial_limit,
    const bool include_two_hop, const std::int64_t* const reduced_cost,
    const Signed128* const lower_bound, const std::int64_t incumbent_numerator,
    unsigned long long* const lp_closed_replies) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x);
  if (work >= work_count) {
    return;
  }
  __shared__ std::int32_t candidates[quick_hs::kMaxPotentialNodes];
  __shared__ std::int64_t scores[quick_hs::kMaxPotentialNodes];
  __shared__ std::uint64_t surviving_pairs[quick_hs::kMaxPotentialNodes];
  __shared__ std::uint64_t pair_product[kQuickPairSortCapacity];
  __shared__ std::int64_t pair_distance[kQuickPairSortCapacity];
  __shared__ std::int32_t pair_ordinal[kQuickPairSortCapacity];
  __shared__ std::int32_t candidate_count;
  __shared__ std::int32_t candidate_pair_count;
  __shared__ std::int32_t enabled;
  __shared__ std::int32_t found;
  __shared__ std::int32_t candidate_eligible;

  const std::int32_t edge = active_edge_ids[work];
  const std::int32_t a = edge_u[edge];
  const std::int32_t b = edge_v[edge];
  if (threadIdx.x == 0U) {
    candidate_count = 0;
    found = 0;
    enabled = edge_active[edge] != 0U && protected_edge[edge] == 0U && graph.degree[a] > 2 &&
              graph.degree[b] > 2;
    if (enabled != 0) {
      for (std::int32_t side = 0; side < 2; ++side) {
        const std::int32_t from = side == 0 ? a : b;
        for (std::int64_t slot = quick_hs::NeighborBegin(graph, from);
             slot < quick_hs::NeighborEnd(graph, from); ++slot) {
          if (quick_hs::NeighborActive(graph, slot)) {
            quick_hs::InsertWitnessCandidate(graph, a, b, quick_hs::Neighbor(graph, from, slot),
                                             candidate_limit, candidates, scores, &candidate_count);
          }
        }
      }
      if (include_two_hop) {
        for (std::int32_t side = 0; side < 2; ++side) {
          const std::int32_t from = side == 0 ? a : b;
          for (std::int64_t first = quick_hs::NeighborBegin(graph, from);
               first < quick_hs::NeighborEnd(graph, from); ++first) {
            if (!quick_hs::NeighborActive(graph, first)) {
              continue;
            }
            const std::int32_t middle = quick_hs::Neighbor(graph, from, first);
            for (std::int64_t second = quick_hs::NeighborBegin(graph, middle);
                 second < quick_hs::NeighborEnd(graph, middle); ++second) {
              if (quick_hs::NeighborActive(graph, second)) {
                quick_hs::InsertWitnessCandidate(
                    graph, a, b, quick_hs::Neighbor(graph, middle, second), candidate_limit,
                    candidates, scores, &candidate_count);
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (enabled == 0) {
    return;
  }

  if (threadIdx.x == 0U) {
    candidate_pair_count = candidate_count * (candidate_count - 1) / 2;
  }
  __syncthreads();

  if (pair_trial_limit != 0) {
    // P6：预算受限时，先用 Direct/Main-Edge 快规则统计每个中心仍允许的
    // 邻边对数，再按 |R(c)|*|R(d)| 排序 OR 候选。全量 sweep 最终会访问
    // 所有失败候选，排序只增加物理工作，因此保留规范顺序。
    for (std::int32_t candidate = 0; candidate < candidate_count; ++candidate) {
      const std::uint64_t surviving = CountSurvivingPairsCta(graph, a, b, candidates[candidate]);
      if (threadIdx.x == 0U) {
        surviving_pairs[candidate] = surviving;
      }
    }
    for (std::int32_t slot = static_cast<std::int32_t>(threadIdx.x); slot < kQuickPairSortCapacity;
         slot += static_cast<std::int32_t>(blockDim.x)) {
      if (slot < candidate_pair_count) {
        std::int32_t first = -1;
        std::int32_t second = -1;
        DecodeQuickCandidatePair(candidate_count, slot, &first, &second);
        pair_product[slot] = SaturatingProduct(surviving_pairs[first], surviving_pairs[second]);
        pair_distance[slot] = scores[first] + scores[second];
        pair_ordinal[slot] = slot;
      } else {
        pair_product[slot] = ULLONG_MAX;
        pair_distance[slot] = INT64_MAX;
        pair_ordinal[slot] = INT_MAX;
      }
    }
    __syncthreads();

    // 固定 512 项 bitonic 网络避免 thread-0 插入排序成为每个 target 的串行瓶颈。
    for (std::int32_t width = 2; width <= kQuickPairSortCapacity; width <<= 1) {
      for (std::int32_t stride = width >> 1; stride > 0; stride >>= 1) {
        for (std::int32_t slot = static_cast<std::int32_t>(threadIdx.x);
             slot < kQuickPairSortCapacity; slot += static_cast<std::int32_t>(blockDim.x)) {
          const std::int32_t partner = slot ^ stride;
          if (partner <= slot) {
            continue;
          }
          const bool ascending = (slot & width) == 0;
          const bool lhs_after_rhs = QuickPairOrderAfter(
              pair_product[slot], pair_distance[slot], pair_ordinal[slot], pair_product[partner],
              pair_distance[partner], pair_ordinal[partner]);
          const bool rhs_after_lhs = QuickPairOrderAfter(
              pair_product[partner], pair_distance[partner], pair_ordinal[partner],
              pair_product[slot], pair_distance[slot], pair_ordinal[slot]);
          if ((ascending && lhs_after_rhs) || (!ascending && rhs_after_lhs)) {
            const std::uint64_t saved_product = pair_product[slot];
            const std::int64_t saved_distance = pair_distance[slot];
            const std::int32_t saved_ordinal = pair_ordinal[slot];
            pair_product[slot] = pair_product[partner];
            pair_distance[slot] = pair_distance[partner];
            pair_ordinal[slot] = pair_ordinal[partner];
            pair_product[partner] = saved_product;
            pair_distance[partner] = saved_distance;
            pair_ordinal[partner] = saved_ordinal;
          }
        }
        __syncthreads();
      }
    }
  }

  for (std::int32_t trial = 0; trial < candidate_pair_count; ++trial) {
    if (pair_trial_limit != 0 && trial >= pair_trial_limit) {
      return;
    }
    std::int32_t first = -1;
    std::int32_t second = -1;
    const std::int32_t ordinal = pair_trial_limit == 0 ? trial : pair_ordinal[trial];
    DecodeQuickCandidatePair(candidate_count, ordinal, &first, &second);
    const std::int32_t c = candidates[first];
    const std::int32_t d = candidates[second];
    if (threadIdx.x == 0U) {
      candidate_eligible =
          !quick_hs::Compatible(graph, a, b, c, d, quick_hs::Distance(graph, a, b));
    }
    __syncthreads();
    const bool closed = candidate_eligible != 0 && RepliesClosedCta<ExtraEdgeDepth>(
                                                       graph, a, b, c, d, reduced_cost, lower_bound,
                                                       incumbent_numerator, lp_closed_replies);
    if (threadIdx.x == 0U && closed) {
      proposed[edge] = 1U;
      first_witness[edge] = c;
      second_witness[edge] = d;
      found = 1;
    }
    __syncthreads();
    if (found != 0) {
      return;
    }
  }
}

__global__ void ActiveCostSummaryKernel(const std::int32_t work_count,
                                        const std::int32_t* const active_edge_ids,
                                        const std::int64_t* const edge_weight,
                                        const std::uint8_t* const edge_active,
                                        unsigned long long* const active_count,
                                        unsigned long long* const cost_sum) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work < work_count) {
    const std::int32_t edge = active_edge_ids[work];
    if (edge_active[edge] == 0U) {
      return;
    }
    atomicAdd(active_count, 1ULL);
    atomicAdd(cost_sum, static_cast<unsigned long long>(edge_weight[edge]));
  }
}

__device__ void InsertLocalSecIncidence(const LocalSecLayout layout, const std::int32_t family,
                                        const std::int32_t window, const std::int32_t rank_u,
                                        const std::int32_t rank_v, std::int32_t* const cut_ids,
                                        std::uint8_t* const count) {
  const std::int32_t family_count = layout.offset[family + 1] - layout.offset[family];
  if (window < 0 || window >= family_count) {
    return;
  }
  const std::int32_t begin = window * layout.stride[family];
  const std::int32_t end = begin + layout.size[family];
  const bool contains_u = rank_u >= begin && rank_u < end;
  const bool contains_v = rank_v >= begin && rank_v < end;
  if (contains_u == contains_v) {
    return;
  }
  const std::int32_t cut = layout.offset[family] + window;
  for (std::uint8_t index = 0U; index < *count; ++index) {
    if (cut_ids[index] == cut) {
      return;
    }
  }
  if (*count < kMaxLocalSecIncidence) {
    cut_ids[(*count)++] = cut;
  }
}

__global__ void BuildLocalSecIncidenceKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::int32_t* const geometry_rank, const LocalSecLayout layout,
    std::uint8_t* const incidence_count, std::int32_t* const incidence_ids) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  std::int32_t* const output =
      incidence_ids + static_cast<std::int64_t>(edge) * kMaxLocalSecIncidence;
  std::uint8_t count = 0U;
  const std::int32_t u = edge_u[edge];
  const std::int32_t v = edge_v[edge];
  const std::int32_t rank_u = geometry_rank[u];
  const std::int32_t rank_v = geometry_rank[v];
  for (std::int32_t family = 0; family < kLocalSecFamilies; ++family) {
    const std::int32_t stride = layout.stride[family];
    const std::int32_t u_window = rank_u / stride;
    const std::int32_t v_window = rank_v / stride;
    InsertLocalSecIncidence(layout, family, u_window, rank_u, rank_v, output, &count);
    InsertLocalSecIncidence(layout, family, u_window - 1, rank_u, rank_v, output, &count);
    InsertLocalSecIncidence(layout, family, v_window, rank_u, rank_v, output, &count);
    InsertLocalSecIncidence(layout, family, v_window - 1, rank_u, rank_v, output, &count);
  }
  incidence_count[edge] = count;
}

__global__ void InitializeSupportLabelsKernel(const std::int32_t dimension,
                                              std::int32_t* const label) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex < dimension) {
    label[vertex] = vertex;
  }
}

__global__ void RelaxSupportLabelsKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::uint8_t* const edge_active, const std::int64_t* const reduced_cost,
    const std::int64_t support_threshold, std::int32_t* const label, std::int32_t* const changed) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U || reduced_cost[edge] >= support_threshold) {
    return;
  }
  const std::int32_t u = edge_u[edge];
  const std::int32_t v = edge_v[edge];
  // label 同时被其他 support edge 的 atomicMin 更新；用原子零加读取，
  // 避免普通 load 与原子写混用造成 CUDA 内存模型中的数据竞争。
  const std::int32_t minimum = min(atomicAdd(&label[u], 0), atomicAdd(&label[v], 0));
  if (atomicMin(&label[u], minimum) > minimum) {
    atomicExch(changed, 1);
  }
  if (atomicMin(&label[v], minimum) > minimum) {
    atomicExch(changed, 1);
  }
}

__global__ void CompressSupportLabelsKernel(const std::int32_t dimension, std::int32_t* const label,
                                            std::int32_t* const changed) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex >= dimension) {
    return;
  }
  const std::int32_t old_label = atomicAdd(&label[vertex], 0);
  std::int32_t root = old_label;
  for (;;) {
    const std::int32_t parent = atomicAdd(&label[root], 0);
    if (parent >= root) {
      break;
    }
    root = parent;
  }
  if (root < old_label) {
    if (atomicMin(&label[vertex], root) > root) {
      atomicExch(changed, 1);
    }
  }
}

__global__ void CountSupportComponentsKernel(const std::int32_t dimension,
                                             const std::int32_t* const label,
                                             std::int32_t* const component_size) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex < dimension) {
    atomicAdd(&component_size[label[vertex]], 1);
  }
}

__global__ void ActivateConnectivityCutsKernel(const std::int32_t dimension,
                                               const std::int32_t dynamic_cut_offset,
                                               const std::int32_t* const label,
                                               const std::int32_t* const component_size,
                                               std::uint8_t* const cut_active,
                                               unsigned long long* const connectivity_cut_count) {
  const std::int32_t root = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (root >= dimension) {
    return;
  }
  const std::int32_t size = component_size[root];
  const bool active = label[root] == root && size >= 2 && size <= dimension - 2;
  cut_active[dynamic_cut_offset + root] = active ? 1U : 0U;
  if (active) {
    atomicAdd(connectivity_cut_count, 1ULL);
  }
}

__device__ void AppendCutIncidence(const std::int32_t cut, std::int32_t* const cut_ids,
                                   std::uint8_t* const count, std::int32_t* const invalid) {
  if (*count >= kMaxLocalSecIncidence) {
    atomicExch(invalid, 1);
    return;
  }
  cut_ids[(*count)++] = cut;
}

__global__ void AppendConnectivityCutIncidenceKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::uint8_t* const edge_active, const std::int32_t dynamic_cut_offset,
    const std::int32_t* const label, const std::uint8_t* const cut_active,
    std::uint8_t* const incidence_count, std::int32_t* const incidence_ids,
    std::int32_t* const invalid) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U) {
    return;
  }
  const std::int32_t first_root = label[edge_u[edge]];
  const std::int32_t second_root = label[edge_v[edge]];
  if (first_root == second_root) {
    return;
  }
  std::int32_t* const output =
      incidence_ids + static_cast<std::int64_t>(edge) * kMaxLocalSecIncidence;
  std::uint8_t count = incidence_count[edge];
  const std::int32_t first_cut = dynamic_cut_offset + first_root;
  const std::int32_t second_cut = dynamic_cut_offset + second_root;
  if (cut_active[first_cut] != 0U) {
    AppendCutIncidence(first_cut, output, &count, invalid);
  }
  if (cut_active[second_cut] != 0U) {
    AppendCutIncidence(second_cut, output, &count, invalid);
  }
  incidence_count[edge] = count;
}

__global__ void
SelectDegreeBoxKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                      const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                      const std::int64_t* const edge_weight, const std::uint8_t* const edge_active,
                      const double* const dual, std::int32_t* const selected_degree) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U) {
    return;
  }
  const std::int32_t u = edge_u[edge];
  const std::int32_t v = edge_v[edge];
  if (static_cast<double>(edge_weight[edge]) - dual[u] - dual[v] < 0.0) {
    atomicAdd(&selected_degree[u], 1);
    atomicAdd(&selected_degree[v], 1);
  }
}

__global__ void
SelectStrongBoxKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                      const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                      const std::int64_t* const edge_weight, const std::uint8_t* const edge_active,
                      const double* const vertex_dual, const double* const cut_dual,
                      const std::uint8_t* const incidence_count,
                      const std::int32_t* const incidence_ids, std::int32_t* const selected_degree,
                      std::int32_t* const selected_cut) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U) {
    return;
  }
  double reduced = static_cast<double>(edge_weight[edge]) - vertex_dual[edge_u[edge]] -
                   vertex_dual[edge_v[edge]];
  const std::int32_t* const cuts =
      incidence_ids + static_cast<std::int64_t>(edge) * kMaxLocalSecIncidence;
  for (std::uint8_t index = 0U; index < incidence_count[edge]; ++index) {
    reduced -= cut_dual[cuts[index]];
  }
  if (reduced < 0.0) {
    atomicAdd(&selected_degree[edge_u[edge]], 1);
    atomicAdd(&selected_degree[edge_v[edge]], 1);
    for (std::uint8_t index = 0U; index < incidence_count[edge]; ++index) {
      atomicAdd(&selected_cut[cuts[index]], 1);
    }
  }
}

__global__ void UpdateResidentDualKernel(const std::int32_t dimension,
                                         const std::int32_t* const selected_degree,
                                         const double step, const double dual_limit,
                                         double* const dual, double* const average) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex >= dimension) {
    return;
  }
  const double gradient = 2.0 - static_cast<double>(selected_degree[vertex]);
  dual[vertex] = fmin(dual_limit, fmax(-dual_limit, dual[vertex] + step * gradient));
  average[vertex] += dual[vertex];
}

__global__ void UpdateLocalSecDualKernel(const std::int32_t cut_count,
                                         const std::int32_t* const selected_cut,
                                         const std::uint8_t* const cut_active, const double step,
                                         const double dual_limit, double* const dual,
                                         double* const average) {
  const std::int32_t cut = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cut >= cut_count) {
    return;
  }
  if (cut_active[cut] == 0U) {
    dual[cut] = 0.0;
    average[cut] = 0.0;
    return;
  }
  const double gradient = 2.0 - static_cast<double>(selected_cut[cut]);
  dual[cut] = fmin(dual_limit, fmax(0.0, dual[cut] + step * gradient));
  average[cut] += dual[cut];
}

__global__ void QuantizeResidentDualKernel(const std::int32_t dimension,
                                           const double inverse_iterations,
                                           const double denominator, const double maximum_scaled,
                                           const double* const average,
                                           std::int64_t* const quantized,
                                           std::int32_t* const invalid) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex >= dimension) {
    return;
  }
  const double scaled = average[vertex] * inverse_iterations * denominator;
  if (!isfinite(scaled) || fabs(scaled) > maximum_scaled) {
    atomicExch(invalid, 1);
    quantized[vertex] = 0;
    return;
  }
  // std::round 的 ties-away-from-zero 语义，保持 CPU sidecar 完全一致。
  quantized[vertex] =
      static_cast<std::int64_t>(scaled >= 0.0 ? floor(scaled + 0.5) : ceil(scaled - 0.5));
}

__global__ void
QuantizeLocalSecDualKernel(const std::int32_t cut_count, const double inverse_iterations,
                           const double denominator, const double maximum_scaled,
                           const double* const average, const std::uint8_t* const cut_active,
                           std::int64_t* const quantized, std::int32_t* const invalid) {
  const std::int32_t cut = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cut >= cut_count) {
    return;
  }
  if (cut_active[cut] == 0U) {
    quantized[cut] = 0;
    return;
  }
  const double scaled = average[cut] * inverse_iterations * denominator;
  if (!isfinite(scaled) || scaled < 0.0 || scaled > maximum_scaled) {
    atomicExch(invalid, 1);
    quantized[cut] = 0;
    return;
  }
  quantized[cut] = static_cast<std::int64_t>(floor(scaled + 0.5));
}

__global__ void
ReducedCostKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                  const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                  const std::int64_t* const edge_weight, const std::uint8_t* const edge_active,
                  const std::int64_t denominator, const std::int64_t* const quantized,
                  const std::int64_t* const quantized_cut,
                  const std::uint8_t* const incidence_count,
                  const std::int32_t* const incidence_ids, std::int64_t* const reduced_cost) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U) {
    reduced_cost[edge] = 0;
    return;
  }
  std::int64_t reduced =
      edge_weight[edge] * denominator - quantized[edge_u[edge]] - quantized[edge_v[edge]];
  const std::int32_t* const cuts =
      incidence_ids + static_cast<std::int64_t>(edge) * kMaxLocalSecIncidence;
  for (std::uint8_t index = 0U; index < incidence_count[edge]; ++index) {
    reduced -= quantized_cut[cuts[index]];
  }
  reduced_cost[edge] = reduced;
}

template <std::int32_t Threads>
__global__ void
ExactBoxBoundPartialsKernel(const std::int32_t dimension, const std::int32_t work_count,
                            const std::int32_t* const active_edge_ids,
                            const std::uint8_t* const edge_active,
                            const std::int64_t* const quantized, const std::int32_t cut_count,
                            const std::int64_t* const quantized_cut,
                            const std::int64_t* const reduced_cost, Signed128* const partials) {
  __shared__ Signed128 reduction[Threads];
  const std::int64_t total = static_cast<std::int64_t>(dimension) + cut_count + work_count;
  const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  Signed128 local{};
  for (std::int64_t index = thread; index < total; index += stride) {
    if (index < dimension) {
      local = Signed128AddInt64(local, 2 * quantized[index]);
      continue;
    }
    const std::int64_t cut_index = index - dimension;
    if (cut_index < cut_count) {
      local = Signed128AddInt64(local, 2 * quantized_cut[cut_index]);
      continue;
    }
    const std::int32_t work = static_cast<std::int32_t>(cut_index - cut_count);
    const std::int32_t edge = active_edge_ids[work];
    if (edge_active[edge] != 0U && reduced_cost[edge] < 0) {
      local = Signed128AddInt64(local, reduced_cost[edge]);
    }
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (std::int32_t offset = Threads / 2; offset != 0; offset /= 2) {
    if (static_cast<std::int32_t>(threadIdx.x) < offset) {
      reduction[threadIdx.x] =
          Signed128Add(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    partials[blockIdx.x] = reduction[0];
  }
}

template <std::int32_t Threads>
__global__ void ExactBoxBoundFinalizeKernel(const std::int32_t partial_count,
                                            const Signed128* const partials,
                                            Signed128* const lower_bound) {
  __shared__ Signed128 reduction[Threads];
  Signed128 local{};
  for (std::int32_t index = static_cast<std::int32_t>(threadIdx.x); index < partial_count;
       index += Threads) {
    local = Signed128Add(local, partials[index]);
  }
  reduction[threadIdx.x] = local;
  __syncthreads();
  for (std::int32_t offset = Threads / 2; offset != 0; offset /= 2) {
    if (static_cast<std::int32_t>(threadIdx.x) < offset) {
      reduction[threadIdx.x] =
          Signed128Add(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U) {
    *lower_bound = reduction[0];
  }
}

__global__ void DecideBetterLpSnapshotKernel(const Signed128* const candidate_lower_bound,
                                             Signed128* const selected_lower_bound,
                                             const std::int32_t candidate_kind,
                                             const bool force_candidate,
                                             std::int32_t* const selected_kind,
                                             std::int32_t* const apply_candidate) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  const bool choose_candidate =
      force_candidate || Signed128Greater(*candidate_lower_bound, *selected_lower_bound);
  if (choose_candidate) {
    *selected_lower_bound = *candidate_lower_bound;
    *selected_kind = candidate_kind;
  }
  *apply_candidate = choose_candidate ? 1 : 0;
}

__global__ void ApplyBetterLpSnapshotKernel(
    const std::int32_t dimension, const std::int32_t work_count,
    const std::int32_t* const active_edge_ids, const std::int32_t cut_count,
    const std::int64_t* const candidate_quantized,
    const std::int64_t* const candidate_quantized_cut,
    const std::int64_t* const candidate_reduced_cost, std::int64_t* const selected_quantized,
    std::int64_t* const selected_quantized_cut, std::int64_t* const selected_reduced_cost,
    const std::int32_t* const apply_candidate) {
  if (*apply_candidate == 0) {
    return;
  }
  const std::int32_t thread = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t stride = static_cast<std::int32_t>(gridDim.x * blockDim.x);
  for (std::int32_t vertex = thread; vertex < dimension; vertex += stride) {
    selected_quantized[vertex] = candidate_quantized[vertex];
  }
  for (std::int32_t cut = thread; cut < cut_count; cut += stride) {
    selected_quantized_cut[cut] = candidate_quantized_cut[cut];
  }
  for (std::int32_t work = thread; work < work_count; work += stride) {
    const std::int32_t edge = active_edge_ids[work];
    selected_reduced_cost[edge] = candidate_reduced_cost[edge];
  }
}

__global__ void ValidateLpReplayBoundKernel(const Signed128* const proposed,
                                            const Signed128* const replayed,
                                            std::int32_t* const invalid) {
  if (blockIdx.x == 0U && threadIdx.x == 0U &&
      (proposed->high != replayed->high || proposed->low != replayed->low)) {
    *invalid = 1;
  }
}

__global__ void
LpForcedOneKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                  const std::int32_t* const active_edge_ids, const std::int32_t* const edge_v,
                  const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
                  const std::int32_t* const degree, const std::int64_t* const reduced_cost,
                  const Signed128* const lower_bound, const std::int64_t incumbent_numerator,
                  std::uint8_t* const proposed) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= edge_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U || protected_edge[edge] != 0U || degree[edge_u[edge]] <= 2 ||
      degree[edge_v[edge]] <= 2) {
    return;
  }
  const std::int64_t positive_reduced = reduced_cost[edge] > 0 ? reduced_cost[edge] : 0;
  if (Signed128GreaterThanInt64(Signed128AddInt64(*lower_bound, positive_reduced),
                                incumbent_numerator)) {
    proposed[edge] = 1U;
  }
}

__device__ void RecordReplayResult(bool proposed, bool valid, std::uint8_t* verified,
                                   std::int32_t edge, unsigned long long* replayed,
                                   unsigned long long* rejected);

__global__ void
LpForcedZeroKernel(const std::int32_t edge_count, const std::int32_t* const active_edge_ids,
                   const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
                   const std::int64_t* const reduced_cost, const Signed128* const lower_bound,
                   const std::int64_t incumbent_numerator, std::uint8_t* const proposed_fixed) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= edge_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] == 0U || protected_edge[edge] != 0U) {
    return;
  }
  const std::int64_t negative_reduced = reduced_cost[edge] < 0 ? -reduced_cost[edge] : 0;
  if (Signed128GreaterThanInt64(Signed128AddInt64(*lower_bound, negative_reduced),
                                incumbent_numerator)) {
    proposed_fixed[edge] = 1U;
  }
}

__device__ bool NonpairsForceEdgeAtEndpoint(const quick_hs::GraphView graph,
                                            const std::int32_t edge, const std::int32_t center) {
  if (graph.pair_offsets == nullptr || graph.nonpair_mask == nullptr || edge < 0 ||
      edge >= graph.edge_count || graph.edge_active[edge] == 0U) {
    return false;
  }
  const std::int32_t other = graph.edge_u[edge] == center ? graph.edge_v[edge] : graph.edge_u[edge];
  const std::int64_t begin = quick_hs::NeighborBegin(graph, center);
  const std::int64_t end = quick_hs::NeighborEnd(graph, center);
  std::int64_t edge_slot = -1;
  for (std::int64_t slot = begin; slot < end; ++slot) {
    if (quick_hs::Neighbor(graph, center, slot) == other) {
      edge_slot = slot;
      break;
    }
  }
  if (edge_slot < 0) {
    return false;
  }
  // 如果 center 的所有可行 Hamilton 邻边对都含该边，则任何
  // 最优 tour 都必须使用它。只有已授权 non-pair 可排除替代对。
  for (std::int64_t first = begin; first + 1 < end; ++first) {
    if (first == edge_slot) {
      continue;
    }
    for (std::int64_t second = first + 1; second < end; ++second) {
      if (second != edge_slot && !quick_hs::PairForbiddenBySlots(graph, center, first, second)) {
        return false;
      }
    }
  }
  return true;
}

__global__ void
NonpairImpliedFixKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                        const quick_hs::GraphView graph, const std::uint8_t* const protected_edge,
                        std::uint8_t* const proposed_fixed, std::uint8_t* const fixed_from_nonpair,
                        unsigned long long* const proposal_count) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (graph.edge_active[edge] == 0U || protected_edge[edge] != 0U || proposed_fixed[edge] != 0U) {
    return;
  }
  if (NonpairsForceEdgeAtEndpoint(graph, edge, graph.edge_u[edge]) ||
      NonpairsForceEdgeAtEndpoint(graph, edge, graph.edge_v[edge])) {
    proposed_fixed[edge] = 1U;
    fixed_from_nonpair[edge] = 1U;
    atomicAdd(proposal_count, 1ULL);
  }
}

constexpr std::uint8_t kFixedReasonLp = 0U;
constexpr std::uint8_t kFixedReasonNonpair = 1U;
constexpr std::uint8_t kFixedReasonDirect = 2U;
constexpr std::uint32_t kDirectFixWarpsPerBlock = 4U;

// 若 tour 不使用 edge=(a,b)，它必须分别在 a、b 选择一对不含 edge 的
// incident edges。这里完整枚举两个 endpoint pair 集的笛卡尔积；每个
// 3+3 path system 都被严格重连或 LP path bound 关闭时，edge 才可固定。
__device__ bool DirectFixProvesEdgeCta(const quick_hs::GraphView graph, const std::int32_t edge,
                                       const std::int64_t* const reduced_cost,
                                       const Signed128* const lower_bound,
                                       const std::int64_t incumbent_numerator,
                                       unsigned long long* const lp_closed_replies) {
  const std::int32_t a = graph.edge_u[edge];
  const std::int32_t b = graph.edge_v[edge];
  const std::uint64_t a_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, a) -
                                                            quick_hs::NeighborBegin(graph, a));
  const std::uint64_t b_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, b) -
                                                            quick_hs::NeighborBegin(graph, b));
  const std::uint64_t a_pairs = a_degree < 2U ? 0U : a_degree * (a_degree - 1U) / 2U;
  const std::uint64_t b_pairs = b_degree < 2U ? 0U : b_degree * (b_degree - 1U) / 2U;
  if (a_pairs == 0U || b_pairs == 0U || (b_pairs != 0U && a_pairs > ULLONG_MAX / b_pairs)) {
    return false;
  }
  const std::uint64_t reply_count = a_pairs * b_pairs;
  __shared__ std::int64_t distance_cache[kDirectFixWarpsPerBlock * kMaximumWarpPathDistances];
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t lane = threadIdx.x & 31U;
  for (std::uint64_t window = 0U; window < reply_count; window += kDirectFixWarpsPerBlock) {
    const std::uint64_t reply = window + warp;
    std::int32_t a1 = -1;
    std::int32_t a2 = -1;
    std::int32_t b1 = -1;
    std::int32_t b2 = -1;
    std::int32_t a1_edge = -1;
    std::int32_t a2_edge = -1;
    std::int32_t b1_edge = -1;
    std::int32_t b2_edge = -1;
    std::int32_t allowed = 0;
    if (lane == 0U && warp < kDirectFixWarpsPerBlock && reply < reply_count) {
      allowed = DecodeNeighborPair(graph, a, reply / b_pairs, &a1, &a2, &a1_edge, &a2_edge) &&
                DecodeNeighborPair(graph, b, reply % b_pairs, &b1, &b2, &b1_edge, &b2_edge) &&
                a1 != b && a2 != b && b1 != a && b2 != a;
    }
    constexpr unsigned kFullWarp = 0xffffffffU;
    allowed = __shfl_sync(kFullWarp, allowed, 0);
    a1 = __shfl_sync(kFullWarp, a1, 0);
    a2 = __shfl_sync(kFullWarp, a2, 0);
    b1 = __shfl_sync(kFullWarp, b1, 0);
    b2 = __shfl_sync(kFullWarp, b2, 0);
    a1_edge = __shfl_sync(kFullWarp, a1_edge, 0);
    a2_edge = __shfl_sync(kFullWarp, a2_edge, 0);
    b1_edge = __shfl_sync(kFullWarp, b1_edge, 0);
    b2_edge = __shfl_sync(kFullWarp, b2_edge, 0);
    const bool lp_closed =
        allowed != 0 && LpClosesPointReplyWarp(a1_edge, a2_edge, b1_edge, b2_edge, reduced_cost,
                                               lower_bound, incumbent_numerator, lp_closed_replies);
    const bool admits =
        allowed != 0 && !lp_closed &&
        Opt33Warp(graph, a1, a, a2, b1, b, b2, distance_cache + warp * kMaximumWarpPathDistances);
    if (__syncthreads_or(lane == 0U && admits ? 1 : 0) != 0) {
      return false;
    }
  }
  return true;
}

__global__ void
DirectFixKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                const quick_hs::GraphView graph, const std::uint8_t* const protected_edge,
                const std::int64_t* const reduced_cost, const Signed128* const lower_bound,
                const std::int64_t incumbent_numerator, std::uint8_t* const proposed_fixed,
                std::uint8_t* const fixed_reason, unsigned long long* const lp_closed_replies,
                unsigned long long* const proposal_count) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  __shared__ std::int32_t enabled;
  if (threadIdx.x == 0U) {
    enabled = graph.edge_active[edge] != 0U && graph.fixed_edge[edge] == 0U &&
                      protected_edge[edge] == 0U && proposed_fixed[edge] == 0U
                  ? 1
                  : 0;
  }
  __syncthreads();
  if (enabled == 0) {
    return;
  }
  const bool proved = DirectFixProvesEdgeCta(graph, edge, reduced_cost, lower_bound,
                                             incumbent_numerator, lp_closed_replies);
  if (threadIdx.x == 0U && proved) {
    proposed_fixed[edge] = 1U;
    fixed_reason[edge] = kFixedReasonDirect;
    atomicAdd(proposal_count, 1ULL);
  }
}

__global__ void
LpNonpairKernel(const quick_hs::GraphView graph, const std::int64_t* const reduced_cost,
                const Signed128* const lower_bound, const std::int64_t incumbent_numerator,
                const std::int64_t* const pair_offsets, const std::uint8_t* const existing_nonpair,
                std::uint8_t* const proposed_nonpair, unsigned long long* const proposal_count) {
  const std::int32_t center = static_cast<std::int32_t>(blockIdx.x);
  if (center >= graph.dimension) {
    return;
  }
  const std::int64_t begin = quick_hs::NeighborBegin(graph, center);
  const std::int64_t degree = quick_hs::NeighborEnd(graph, center) - begin;
  const std::int64_t pair_begin = pair_offsets[center];
  for (std::int64_t first = threadIdx.x; first + 1 < degree; first += blockDim.x) {
    const std::int32_t first_edge = graph.neighbor_edge_ids[begin + first];
    const std::int64_t first_reduced = reduced_cost[first_edge];
    for (std::int64_t second = first + 1; second < degree; ++second) {
      const std::int32_t second_edge = graph.neighbor_edge_ids[begin + second];
      const std::int64_t second_reduced = reduced_cost[second_edge];
      Signed128 forced = *lower_bound;
      forced = Signed128AddInt64(forced, first_reduced > 0 ? first_reduced : 0);
      forced = Signed128AddInt64(forced, second_reduced > 0 ? second_reduced : 0);
      const std::int64_t local = first * (2 * degree - first - 1) / 2 + (second - first - 1);
      const std::int64_t pair = pair_begin + local;
      if (existing_nonpair[pair] == 0U && Signed128GreaterThanInt64(forced, incumbent_numerator)) {
        proposed_nonpair[pair] = 1U;
        atomicAdd(proposal_count, 1ULL);
      }
    }
  }
}

__device__ bool
FixedAnchorProvesNonpair(const quick_hs::GraphView graph, const std::int32_t center,
                         const std::int64_t first_slot, const std::int64_t second_slot,
                         const std::int32_t fixed_edge, const std::int32_t* const edge_u,
                         const std::int32_t* const edge_v, const std::uint8_t* const edge_active,
                         const std::uint8_t* const fixed_bits) {
  if (fixed_edge < 0 || fixed_edge >= graph.edge_count || edge_active[fixed_edge] == 0U ||
      fixed_bits[fixed_edge] == 0U) {
    return false;
  }
  const std::int32_t first = quick_hs::Neighbor(graph, center, first_slot);
  const std::int32_t second = quick_hs::Neighbor(graph, center, second_slot);
  const std::int32_t p = edge_u[fixed_edge];
  const std::int32_t q = edge_v[fixed_edge];
  // opt23 的两条输入路径必须顶点不交；共享端点需要先规范化为另一类
  // path system，不能把该返回值直接解释成 non-pair 证明。
  if (p == center || p == first || p == second || q == center || q == first || q == second) {
    return false;
  }
  return !quick_hs::Opt23(graph, p, q, first, center, second, quick_hs::Distance(graph, p, q),
                          quick_hs::Distance(graph, first, center),
                          quick_hs::Distance(graph, center, second));
}

// 若固定边 pq 与二边路径 x-center-y 不可能同时属于最优 tour，则该邻边对
// 是严格 non-pair。每个输出保存 stable fixed-edge id，replay 只复核该 witness。
__global__ void FixedAnchorNonpairKernel(
    const quick_hs::GraphView graph, const std::int32_t* const edge_u,
    const std::int32_t* const edge_v, const std::uint8_t* const edge_active,
    const std::uint8_t* const fixed_bits, const std::int32_t* const fixed_edge_ids,
    const std::int32_t* const fixed_edge_count, const std::int64_t* const pair_offsets,
    const std::uint8_t* const existing_nonpair, std::uint8_t* const proposed_nonpair,
    std::int32_t* const fixed_witness, unsigned long long* const proposal_count) {
  const std::int32_t center = static_cast<std::int32_t>(blockIdx.x);
  if (center >= graph.dimension) {
    return;
  }
  const std::int64_t begin = quick_hs::NeighborBegin(graph, center);
  const std::int64_t degree = quick_hs::NeighborEnd(graph, center) - begin;
  const std::int64_t pair_begin = pair_offsets[center];
  for (std::int64_t first = threadIdx.x; first + 1 < degree; first += blockDim.x) {
    for (std::int64_t second = first + 1; second < degree; ++second) {
      const std::int64_t local = first * (2 * degree - first - 1) / 2 + (second - first - 1);
      const std::int64_t pair = pair_begin + local;
      if (existing_nonpair[pair] != 0U || proposed_nonpair[pair] != 0U) {
        continue;
      }
      for (std::int32_t fixed = 0; fixed < *fixed_edge_count; ++fixed) {
        const std::int32_t witness = fixed_edge_ids[fixed];
        if (FixedAnchorProvesNonpair(graph, center, begin + first, begin + second, witness, edge_u,
                                     edge_v, edge_active, fixed_bits)) {
          fixed_witness[pair] = witness;
          proposed_nonpair[pair] = 1U;
          atomicAdd(proposal_count, 1ULL);
          break;
        }
      }
    }
  }
}

constexpr std::uint32_t kNonpairPointWarpsPerBlock = 4U;
constexpr std::int32_t kPointPathEndSubsetDpStates = (1 << 5) * kMaximumDynamicPathNodes;

// 点分支留下的 3+3 reply 再尝试四个 path-end Tutte OR 分支。
// 每个端点的所有合法伙伴构成 AND；一个伙伴存活只否定当前端点，
// 四个端点都无法关闭时才保持根 reply。候选排序不缩小证明搜索域。
__device__ bool PointReplyAdmitsPathEndWarp(
    const quick_hs::GraphView graph, const std::int32_t first, const std::int32_t center,
    const std::int32_t second, const std::int32_t root_first_edge,
    const std::int32_t root_second_edge, const std::int32_t point_first, const std::int32_t point,
    const std::int32_t point_second, const std::int32_t point_first_edge,
    const std::int32_t point_second_edge, const std::int64_t* const reduced_cost,
    const Signed128* const lower_bound, const std::int64_t incumbent_numerator,
    unsigned long long* const lp_closed_replies, std::int64_t* const path_distance_cache,
    std::int64_t* const subset_dp_cache) {
  constexpr unsigned kFullWarp = 0xffffffffU;
  const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x & 31U);
  const std::int32_t roles[6] = {first, center, second, point_first, point, point_second};
  if (!quick_hs::AllNodesDistinct(roles, 6) || graph.neighbor_edge_ids == nullptr) {
    // 共享顶点的 3+3 系统应先走更一般的 normalization；本层暂不以
    // 未覆盖的 overlap 形态授权关闭。
    return true;
  }

  const std::int32_t endpoints[4] = {first, second, point_first, point_second};
  const std::int32_t inward_edges[4] = {root_first_edge, root_second_edge, point_first_edge,
                                        point_second_edge};
  const std::int32_t opposite_endpoints[4] = {second, first, point_second, point_first};
  // 前四个 lane 并行计数；四个 OR 分支只排序，不能只尝试最便宜的一个。
  std::int64_t own_slot = -1;
  std::int64_t own_count = 0;
  if (lane < 4) {
    const std::int32_t endpoint = endpoints[lane];
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (quick_hs::NeighborActive(graph, slot) &&
          graph.neighbor_edge_ids[slot] == inward_edges[lane]) {
        own_slot = slot;
        break;
      }
    }
    if (own_slot >= 0) {
      for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
           slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
        if (slot == own_slot || !quick_hs::NeighborActive(graph, slot)) {
          continue;
        }
        const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
        if (extension != center && extension != point &&
            !quick_hs::PairForbiddenBySlots(graph, endpoint, own_slot, slot)) {
          ++own_count;
        }
      }
    }
  }
  std::int64_t inward_slots[4], counts[4];
  for (int index = 0; index < 4; ++index) {
    inward_slots[index] = __shfl_sync(kFullWarp, own_slot, index);
    counts[index] = __shfl_sync(kFullWarp, own_count, index);
    if (inward_slots[index] < 0) {
      return true;
    }
  }
  for (int attempt = 0; attempt < 4; ++attempt) {
    int selected_endpoint = -1;
    for (int index = 0; index < 4; ++index) {
      if (counts[index] >= 0 &&
          (selected_endpoint < 0 || counts[index] < counts[selected_endpoint])) {
        selected_endpoint = index;
      }
    }
    counts[selected_endpoint] = -1;
    const std::int64_t selected_inward_slot = inward_slots[selected_endpoint];
    bool unresolved = false;

    const std::int32_t endpoint = endpoints[selected_endpoint];
    const std::int64_t begin = quick_hs::NeighborBegin(graph, endpoint);
    const std::int64_t end = quick_hs::NeighborEnd(graph, endpoint);
    for (std::int64_t slot = begin; slot < end; ++slot) {
      std::int32_t extension = -1;
      std::int32_t extension_edge = -1;
      std::int32_t allowed = 0;
      if (lane == 0 && slot != selected_inward_slot && quick_hs::NeighborActive(graph, slot)) {
        extension = quick_hs::Neighbor(graph, endpoint, slot);
        extension_edge = graph.neighbor_edge_ids[slot];
        allowed =
            extension != center && extension != point &&
                    !quick_hs::PairForbiddenBySlots(graph, endpoint, selected_inward_slot, slot)
                ? 1
                : 0;
      }
      extension = __shfl_sync(kFullWarp, extension, 0);
      extension_edge = __shfl_sync(kFullWarp, extension_edge, 0);
      allowed = __shfl_sync(kFullWarp, allowed, 0);
      if (allowed == 0) {
        continue;
      }
      if (extension == opposite_endpoints[selected_endpoint]) {
        // 同一路径两端相连会形成不覆盖全部图的真子环；六个角色互异保证
        // 图中至少还有另一条三点路径，因此该回复可直接关闭。
        continue;
      }
      if (LpClosesPointPathEndReplyWarp(root_first_edge, root_second_edge, point_first_edge,
                                        point_second_edge, extension_edge, reduced_cost,
                                        lower_bound, incumbent_numerator, lp_closed_replies)) {
        continue;
      }

      bool admits = false;
      if (selected_endpoint == 0) {
        admits = Opt34Warp(graph, point_first, point, point_second, extension, first, center,
                           second, path_distance_cache, subset_dp_cache);
      } else if (selected_endpoint == 1) {
        admits = Opt34Warp(graph, point_first, point, point_second, first, center, second,
                           extension, path_distance_cache, subset_dp_cache);
      } else if (selected_endpoint == 2) {
        admits = Opt34Warp(graph, first, center, second, extension, point_first, point,
                           point_second, path_distance_cache, subset_dp_cache);
      } else {
        admits = Opt34Warp(graph, first, center, second, point_first, point, point_second,
                           extension, path_distance_cache, subset_dp_cache);
      }
      if (admits) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }
  return true;
}

// Hamilton--Tutte point move：若某个路径外点 point 的每个可能
// Hamilton 邻边对都能与 first-center-second 一起被精确路径重连
// 排除，则根邻边对是 non-pair。这是一层完整 AND reply，不使用
// 启发式阴性结果授权。
__device__ bool PointProvesNonpairCta(const quick_hs::GraphView graph, const std::int32_t first,
                                      const std::int32_t center, const std::int32_t second,
                                      const std::int32_t root_first_edge,
                                      const std::int32_t root_second_edge, const std::int32_t point,
                                      const std::int64_t* const reduced_cost,
                                      const Signed128* const lower_bound,
                                      const std::int64_t incumbent_numerator,
                                      unsigned long long* const lp_closed_replies,
                                      unsigned long long* const point_path_end_closed_replies) {
  if (point < 0 || point >= graph.dimension || point == first || point == center ||
      point == second) {
    return false;
  }
  const std::int64_t point_begin = quick_hs::NeighborBegin(graph, point);
  const std::int64_t point_degree = quick_hs::NeighborEnd(graph, point) - point_begin;
  if (point_degree < 2) {
    return false;
  }
  const std::uint64_t reply_count =
      static_cast<std::uint64_t>(point_degree) * static_cast<std::uint64_t>(point_degree - 1) / 2U;
  __shared__ std::int64_t
      path_distance_cache[kNonpairPointWarpsPerBlock * kMaximumWarpPathDistances];
  __shared__ std::int64_t
      path_end_subset_dp[kNonpairPointWarpsPerBlock * kPointPathEndSubsetDpStates];
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t lane = threadIdx.x & 31U;
  for (std::uint64_t window = 0; window < reply_count; window += kNonpairPointWarpsPerBlock) {
    const std::uint64_t reply = window + warp;
    std::int32_t point_first = -1;
    std::int32_t point_second = -1;
    std::int32_t point_first_edge = -1;
    std::int32_t point_second_edge = -1;
    std::int32_t decoded = 0;
    if (lane == 0U && warp < kNonpairPointWarpsPerBlock && reply < reply_count) {
      decoded = DecodeNeighborPair(graph, point, reply, &point_first, &point_second,
                                   &point_first_edge, &point_second_edge);
    }
    constexpr unsigned kFullWarp = 0xffffffffU;
    decoded = __shfl_sync(kFullWarp, decoded, 0);
    point_first = __shfl_sync(kFullWarp, point_first, 0);
    point_second = __shfl_sync(kFullWarp, point_second, 0);
    point_first_edge = __shfl_sync(kFullWarp, point_first_edge, 0);
    point_second_edge = __shfl_sync(kFullWarp, point_second_edge, 0);
    // 两条二边路径在四节点实例上可能恰好拼成完整 Hamilton 环；通用
    // local path oracle 会把任何闭环视为局部无效，因此这里显式保留这个
    // 唯一不能按“真子环”关闭的边界情形。
    const bool full_four_cycle = decoded != 0 && graph.dimension == 4 &&
                                 ((point_first == first && point_second == second) ||
                                  (point_first == second && point_second == first));
    const bool lp_closed =
        decoded != 0 && !full_four_cycle &&
        LpClosesPointReplyWarp(root_first_edge, root_second_edge, point_first_edge,
                               point_second_edge, reduced_cost, lower_bound, incumbent_numerator,
                               lp_closed_replies);
    bool admits = decoded != 0 && !lp_closed &&
                  (full_four_cycle ||
                   Opt33Warp(graph, first, center, second, point_first, point, point_second,
                             path_distance_cache + warp * kMaximumWarpPathDistances));
    if (admits && !full_four_cycle) {
      admits = PointReplyAdmitsPathEndWarp(graph, first, center, second, root_first_edge,
                                           root_second_edge, point_first, point, point_second,
                                           point_first_edge, point_second_edge, reduced_cost,
                                           lower_bound, incumbent_numerator, lp_closed_replies,
                                           path_distance_cache + warp * kMaximumWarpPathDistances,
                                           path_end_subset_dp + warp * kPointPathEndSubsetDpStates);
      if (!admits && lane == 0U && point_path_end_closed_replies != nullptr) {
        atomicAdd(point_path_end_closed_replies, 1ULL);
      }
    }
    if (__syncthreads_or(lane == 0U && admits ? 1 : 0) != 0) {
      return false;
    }
  }
  return true;
}

__device__ bool
LpClosesPointReplyThread(const std::int32_t root_first_edge, const std::int32_t root_second_edge,
                         const std::int32_t point_first_edge, const std::int32_t point_second_edge,
                         const std::int64_t* const reduced_cost, const Signed128* const lower_bound,
                         const std::int64_t incumbent_numerator,
                         unsigned long long* const closed_replies) {
  if (reduced_cost == nullptr || lower_bound == nullptr) {
    return false;
  }
  const std::int32_t source_edge_ids[4] = {root_first_edge, root_second_edge, point_first_edge,
                                           point_second_edge};
  std::int32_t edge_ids[4]{};
  std::int32_t unique_count = 0;
  for (std::int32_t path_edge = 0; path_edge < 4; ++path_edge) {
    const std::int32_t edge = source_edge_ids[path_edge];
    if (edge < 0) {
      return false;
    }
    bool duplicate = false;
    for (std::int32_t previous = 0; previous < unique_count; ++previous) {
      duplicate = duplicate || edge_ids[previous] == edge;
    }
    if (!duplicate) {
      edge_ids[unique_count++] = edge;
    }
  }
  Signed128 forced = *lower_bound;
  for (std::int32_t index = 0; index < unique_count; ++index) {
    const std::int64_t reduced = reduced_cost[edge_ids[index]];
    forced = Signed128AddInt64(forced, reduced > 0 ? reduced : 0);
  }
  const bool closed = Signed128GreaterThanInt64(forced, incumbent_numerator);
  if (closed && closed_replies != nullptr) {
    atomicAdd(closed_replies, 1ULL);
  }
  return closed;
}

__device__ bool LpClosesPointPathEndReplyThread(
    const std::int32_t root_first_edge, const std::int32_t root_second_edge,
    const std::int32_t point_first_edge, const std::int32_t point_second_edge,
    const std::int32_t extension_edge, const std::int64_t* const reduced_cost,
    const Signed128* const lower_bound, const std::int64_t incumbent_numerator) {
  if (reduced_cost == nullptr || lower_bound == nullptr) {
    return false;
  }
  const std::int32_t source_edge_ids[5] = {root_first_edge, root_second_edge, point_first_edge,
                                           point_second_edge, extension_edge};
  std::int32_t edge_ids[5]{};
  std::int32_t unique_count = 0;
  for (std::int32_t path_edge = 0; path_edge < 5; ++path_edge) {
    const std::int32_t edge = source_edge_ids[path_edge];
    if (edge < 0) {
      return false;
    }
    bool duplicate = false;
    for (std::int32_t previous = 0; previous < unique_count; ++previous) {
      duplicate = duplicate || edge_ids[previous] == edge;
    }
    if (!duplicate) {
      edge_ids[unique_count++] = edge;
    }
  }
  Signed128 forced = *lower_bound;
  for (std::int32_t index = 0; index < unique_count; ++index) {
    const std::int64_t reduced = reduced_cost[edge_ids[index]];
    forced = Signed128AddInt64(forced, reduced > 0 ? reduced : 0);
  }
  return Signed128GreaterThanInt64(forced, incumbent_numerator);
}

// replay 采用单线程串行枚举端点伙伴，并调用 host/device 公用的 Opt34；
// 它不复用 proposer 的 warp cache、shuffle 或 subset DP。
__device__ bool PointReplyAdmitsPathEndThread(
    const quick_hs::GraphView graph, const std::int32_t first, const std::int32_t center,
    const std::int32_t second, const std::int32_t root_first_edge,
    const std::int32_t root_second_edge, const std::int32_t point_first, const std::int32_t point,
    const std::int32_t point_second, const std::int32_t point_first_edge,
    const std::int32_t point_second_edge, const std::int64_t* const reduced_cost,
    const Signed128* const lower_bound, const std::int64_t incumbent_numerator) {
  const std::int32_t roles[6] = {first, center, second, point_first, point, point_second};
  if (!quick_hs::AllNodesDistinct(roles, 6) || graph.neighbor_edge_ids == nullptr) {
    return true;
  }

  const std::int32_t endpoints[4] = {first, second, point_first, point_second};
  const std::int32_t inward_edges[4] = {root_first_edge, root_second_edge, point_first_edge,
                                        point_second_edge};
  const std::int32_t opposite_endpoints[4] = {second, first, point_second, point_first};
  // 独立 replay 按自然端点顺序搜索，与 proposer 的并行排序无关。
  for (int selected_endpoint = 0; selected_endpoint < 4; ++selected_endpoint) {
    std::int64_t selected_inward_slot = -1;
    const std::int32_t candidate = endpoints[selected_endpoint];
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, candidate);
         slot < quick_hs::NeighborEnd(graph, candidate); ++slot) {
      if (quick_hs::NeighborActive(graph, slot) &&
          graph.neighbor_edge_ids[slot] == inward_edges[selected_endpoint]) {
        selected_inward_slot = slot;
        break;
      }
    }
    if (selected_inward_slot < 0) {
      return true;
    }
    bool unresolved = false;

    const std::int32_t endpoint = endpoints[selected_endpoint];
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, endpoint);
         slot < quick_hs::NeighborEnd(graph, endpoint); ++slot) {
      if (slot == selected_inward_slot || !quick_hs::NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = quick_hs::Neighbor(graph, endpoint, slot);
      if (extension == center || extension == point ||
          quick_hs::PairForbiddenBySlots(graph, endpoint, selected_inward_slot, slot)) {
        continue;
      }
      if (extension == opposite_endpoints[selected_endpoint]) {
        continue;
      }
      const std::int32_t extension_edge = graph.neighbor_edge_ids[slot];
      if (LpClosesPointPathEndReplyThread(root_first_edge, root_second_edge, point_first_edge,
                                          point_second_edge, extension_edge, reduced_cost,
                                          lower_bound, incumbent_numerator)) {
        continue;
      }
      bool admits = false;
      if (selected_endpoint == 0) {
        admits = quick_hs::Opt34(graph, point_first, point, point_second, extension, first, center,
                                 second);
      } else if (selected_endpoint == 1) {
        admits = quick_hs::Opt34(graph, point_first, point, point_second, first, center, second,
                                 extension);
      } else if (selected_endpoint == 2) {
        admits = quick_hs::Opt34(graph, first, center, second, extension, point_first, point,
                                 point_second);
      } else {
        admits = quick_hs::Opt34(graph, first, center, second, point_first, point, point_second,
                                 extension);
      }
      if (admits) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }
  return true;
}

// replay 与 proposer 使用不同的物理枚举：proposer 一个 warp 协作判定
// 一个 reply；replay 则每线程独立判定一个 reply。两者都穷举完整 AND
// 空间，可在不复用执行轨迹的情况下捕获 warp-DP 或映射错误。
__device__ bool PointProvesNonpairReplayCta(
    const quick_hs::GraphView graph, const std::int32_t first, const std::int32_t center,
    const std::int32_t second, const std::int32_t root_first_edge,
    const std::int32_t root_second_edge, const std::int32_t point,
    const std::int64_t* const reduced_cost, const Signed128* const lower_bound,
    const std::int64_t incumbent_numerator, unsigned long long* const lp_closed_replies) {
  if (point < 0 || point >= graph.dimension || point == first || point == center ||
      point == second) {
    return false;
  }
  const std::int64_t degree =
      quick_hs::NeighborEnd(graph, point) - quick_hs::NeighborBegin(graph, point);
  if (degree < 2) {
    return false;
  }
  const std::uint64_t reply_count =
      static_cast<std::uint64_t>(degree) * static_cast<std::uint64_t>(degree - 1) / 2U;
  for (std::uint64_t window = 0; window < reply_count; window += blockDim.x) {
    const std::uint64_t reply = window + threadIdx.x;
    std::int32_t point_first = -1;
    std::int32_t point_second = -1;
    std::int32_t point_first_edge = -1;
    std::int32_t point_second_edge = -1;
    const bool decoded =
        reply < reply_count && DecodeNeighborPair(graph, point, reply, &point_first, &point_second,
                                                  &point_first_edge, &point_second_edge);
    const bool full_four_cycle = decoded && graph.dimension == 4 &&
                                 ((point_first == first && point_second == second) ||
                                  (point_first == second && point_second == first));
    const bool lp_closed =
        decoded && !full_four_cycle &&
        LpClosesPointReplyThread(root_first_edge, root_second_edge, point_first_edge,
                                 point_second_edge, reduced_cost, lower_bound, incumbent_numerator,
                                 lp_closed_replies);
    bool admits = decoded && !lp_closed &&
                  (full_four_cycle ||
                   quick_hs::Opt33(graph, first, center, second, point_first, point, point_second));
    if (admits && !full_four_cycle) {
      admits = PointReplyAdmitsPathEndThread(graph, first, center, second, root_first_edge,
                                             root_second_edge, point_first, point, point_second,
                                             point_first_edge, point_second_edge, reduced_cost,
                                             lower_bound, incumbent_numerator);
    }
    if (__syncthreads_or(admits ? 1 : 0) != 0) {
      return false;
    }
  }
  return true;
}

// direct fixing 的 replay 改用 thread-per-pair-product，并调用通用 Opt33，
// 与 proposer 的 warp distance cache/枚举映射相互独立。
__device__ bool DirectFixProvesEdgeReplayCta(const quick_hs::GraphView graph,
                                             const std::int32_t edge,
                                             const std::int64_t* const reduced_cost,
                                             const Signed128* const lower_bound,
                                             const std::int64_t incumbent_numerator) {
  const std::int32_t a = graph.edge_u[edge];
  const std::int32_t b = graph.edge_v[edge];
  const std::uint64_t a_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, a) -
                                                            quick_hs::NeighborBegin(graph, a));
  const std::uint64_t b_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, b) -
                                                            quick_hs::NeighborBegin(graph, b));
  const std::uint64_t a_pairs = a_degree < 2U ? 0U : a_degree * (a_degree - 1U) / 2U;
  const std::uint64_t b_pairs = b_degree < 2U ? 0U : b_degree * (b_degree - 1U) / 2U;
  if (a_pairs == 0U || b_pairs == 0U || (b_pairs != 0U && a_pairs > ULLONG_MAX / b_pairs)) {
    return false;
  }
  const std::uint64_t reply_count = a_pairs * b_pairs;
  for (std::uint64_t window = 0U; window < reply_count; window += blockDim.x) {
    const std::uint64_t reply = window + threadIdx.x;
    std::int32_t a1 = -1;
    std::int32_t a2 = -1;
    std::int32_t b1 = -1;
    std::int32_t b2 = -1;
    std::int32_t a1_edge = -1;
    std::int32_t a2_edge = -1;
    std::int32_t b1_edge = -1;
    std::int32_t b2_edge = -1;
    const bool allowed =
        reply < reply_count &&
        DecodeNeighborPair(graph, a, reply / b_pairs, &a1, &a2, &a1_edge, &a2_edge) &&
        DecodeNeighborPair(graph, b, reply % b_pairs, &b1, &b2, &b1_edge, &b2_edge) && a1 != b &&
        a2 != b && b1 != a && b2 != a;
    const bool lp_closed =
        allowed && LpClosesPointReplyThread(a1_edge, a2_edge, b1_edge, b2_edge, reduced_cost,
                                            lower_bound, incumbent_numerator, nullptr);
    const bool admits = allowed && !lp_closed && quick_hs::Opt33(graph, a1, a, a2, b1, b, b2);
    if (__syncthreads_or(admits ? 1 : 0) != 0) {
      return false;
    }
  }
  return true;
}

__global__ void
ReplayDirectFixKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                      const quick_hs::GraphView graph, const std::uint8_t* const protected_edge,
                      const std::uint8_t* const proposed_fixed,
                      const std::uint8_t* const fixed_reason,
                      const std::int64_t* const reduced_cost, const Signed128* const lower_bound,
                      const std::int64_t incumbent_numerator, std::uint8_t* const verified_fixed,
                      unsigned long long* const replayed, unsigned long long* const rejected) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  __shared__ std::int32_t enabled;
  if (threadIdx.x == 0U) {
    enabled = proposed_fixed[edge] != 0U && fixed_reason[edge] == kFixedReasonDirect &&
                      graph.edge_active[edge] != 0U && graph.fixed_edge[edge] == 0U &&
                      protected_edge[edge] == 0U
                  ? 1
                  : 0;
  }
  __syncthreads();
  if (enabled == 0) {
    return;
  }
  const bool valid =
      DirectFixProvesEdgeReplayCta(graph, edge, reduced_cost, lower_bound, incumbent_numerator);
  if (threadIdx.x == 0U) {
    RecordReplayResult(true, valid, verified_fixed, edge, replayed, rejected);
  }
}

__device__ bool DecodeGlobalNeighborPair(const quick_hs::GraphView graph,
                                         const std::int64_t* const pair_offsets,
                                         const std::int64_t pair, std::int32_t* const center,
                                         std::int32_t* const first, std::int32_t* const second,
                                         std::int32_t* const first_edge,
                                         std::int32_t* const second_edge) {
  std::int32_t low = 0;
  std::int32_t high = graph.dimension;
  while (low < high) {
    const std::int32_t middle = low + (high - low) / 2;
    if (pair_offsets[middle + 1] <= pair) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  if (low < 0 || low >= graph.dimension || pair < pair_offsets[low] ||
      pair >= pair_offsets[low + 1]) {
    return false;
  }
  *center = low;
  return DecodeNeighborPair(graph, low, static_cast<std::uint64_t>(pair - pair_offsets[low]), first,
                            second, first_edge, second_edge);
}

template <int MinBlocks>
__global__ __launch_bounds__(128, MinBlocks) void PointNonpairKernel(
    const std::int64_t pair_count, const quick_hs::GraphView graph,
    const std::int64_t* const pair_offsets, const std::uint8_t* const existing_nonpair,
    std::uint8_t* const proposed_nonpair, std::int32_t* const point_witness,
    const std::int64_t* const reduced_cost, const Signed128* const lower_bound,
    const std::int64_t incumbent_numerator, unsigned long long* const lp_closed_replies,
    unsigned long long* const point_path_end_closed_replies,
    unsigned long long* const proposal_count, const std::int32_t* const near_points,
    const std::uint32_t* const near_mask, const bool priority_only) {
  __shared__ std::int32_t center;
  __shared__ std::int32_t first;
  __shared__ std::int32_t second;
  __shared__ std::int32_t first_edge;
  __shared__ std::int32_t second_edge;
  __shared__ std::int32_t valid_root;
  __shared__ std::int32_t selected_point;
  for (std::int64_t pair = static_cast<std::int64_t>(blockIdx.x); pair < pair_count;
       pair += static_cast<std::int64_t>(gridDim.x)) {
    if (threadIdx.x == 0U) {
      valid_root = existing_nonpair[pair] == 0U && proposed_nonpair[pair] == 0U &&
                           DecodeGlobalNeighborPair(graph, pair_offsets, pair, &center, &first,
                                                    &second, &first_edge, &second_edge)
                       ? 1
                       : 0;
    }
    __syncthreads();
    if (valid_root != 0) {
      // 扫描全部路径外点，只在找到一个完整关闭所有 replies
      // 的 point 时提案。未找到始终保留 pair。
      if (threadIdx.x == 0U) {
        selected_point = -1;
      }
      __syncthreads();
      if (near_points != nullptr) {
        for (int rank = 0; rank < quick_hs::kMaxPotentialNodes; ++rank) {
          const int point =
              near_points[static_cast<std::size_t>(center) * quick_hs::kMaxPotentialNodes + rank];
          if (point < 0)
            break;
          const bool proved = PointProvesNonpairCta(
              graph, first, center, second, first_edge, second_edge, point, reduced_cost,
              lower_bound, incumbent_numerator, lp_closed_replies, point_path_end_closed_replies);
          if (threadIdx.x == 0U && proved)
            selected_point = point;
          __syncthreads();
          if (selected_point >= 0)
            break;
        }
      }
      // AND reply 数是 C(deg(point), 2)。按真实活动度数分桶后仍完整访问
      // 所有 point，但先尝试代价最低、最容易首成功的 witness；桶内保持
      // node-id 规范顺序，结果确定且不把启发式顺序当成完整性假设。
      // priority_only 仅用于首次预热：未关闭的根不作任何状态修改，编排层
      // 在结束前强制重跑完整 service。它不是正式 Point 域的点数上限。
      for (std::int32_t degree_bucket = 0;
           !priority_only && degree_bucket < 4 && selected_point < 0; ++degree_bucket) {
        for (std::int32_t point = 0; point < graph.dimension; ++point) {
          const std::int32_t point_degree = graph.degree[point];
          const std::int32_t point_bucket =
              point_degree <= 2 ? 0 : (point_degree <= 4 ? 1 : (point_degree <= 8 ? 2 : 3));
          const std::size_t near_row_words =
              (static_cast<std::size_t>(graph.dimension) + 31U) / 32U;
          const bool already_tried =
              near_mask != nullptr &&
              ((near_mask[static_cast<std::size_t>(center) * near_row_words + point / 32] >>
                (point % 32)) &
               1U) != 0U;
          const bool proved =
              !already_tried && point_bucket == degree_bucket &&
              PointProvesNonpairCta(graph, first, center, second, first_edge, second_edge, point,
                                    reduced_cost, lower_bound, incumbent_numerator,
                                    lp_closed_replies, point_path_end_closed_replies);
          if (threadIdx.x == 0U && proved) {
            selected_point = point;
          }
          __syncthreads();
          if (selected_point >= 0) {
            break;
          }
        }
      }
      if (threadIdx.x == 0U && selected_point >= 0) {
        point_witness[pair] = selected_point;
        proposed_nonpair[pair] = 1U;
        atomicAdd(proposal_count, 1ULL);
      }
    }
    __syncthreads();
  }
}

__global__ void ReplayPointNonpairKernel(
    const std::int64_t pair_count, const quick_hs::GraphView graph,
    const std::int64_t* const pair_offsets, const std::uint8_t* const proposed_nonpair,
    const std::int32_t* const point_witness, const std::int64_t* const reduced_cost,
    const Signed128* const lower_bound, const std::int64_t incumbent_numerator,
    std::uint8_t* const verified_nonpair, unsigned long long* const replayed,
    unsigned long long* const rejected) {
  __shared__ std::int32_t center;
  __shared__ std::int32_t first;
  __shared__ std::int32_t second;
  __shared__ std::int32_t first_edge;
  __shared__ std::int32_t second_edge;
  __shared__ std::int32_t witness;
  __shared__ std::int32_t valid_root;
  for (std::int64_t pair = static_cast<std::int64_t>(blockIdx.x); pair < pair_count;
       pair += static_cast<std::int64_t>(gridDim.x)) {
    if (threadIdx.x == 0U) {
      witness = point_witness[pair];
      valid_root = proposed_nonpair[pair] != 0U && witness >= 0 &&
                           DecodeGlobalNeighborPair(graph, pair_offsets, pair, &center, &first,
                                                    &second, &first_edge, &second_edge)
                       ? 1
                       : 0;
    }
    __syncthreads();
    if (valid_root != 0) {
      const bool valid = PointProvesNonpairReplayCta(graph, first, center, second, first_edge,
                                                     second_edge, witness, reduced_cost,
                                                     lower_bound, incumbent_numerator, nullptr);
      if (threadIdx.x == 0U) {
        atomicAdd(replayed, 1ULL);
        if (valid) {
          verified_nonpair[pair] = 1U;
        } else {
          atomicAdd(rejected, 1ULL);
        }
      }
    }
    __syncthreads();
  }
}

__device__ std::int64_t ReplayReducedCost(
    const std::int32_t edge, const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::int64_t* const edge_weight, const std::int64_t denominator,
    const std::int64_t* const quantized, const std::int64_t* const quantized_cut,
    const std::uint8_t* const incidence_count, const std::int32_t* const incidence_ids) {
  std::int64_t reduced =
      edge_weight[edge] * denominator - quantized[edge_u[edge]] - quantized[edge_v[edge]];
  const std::int32_t* const cuts =
      incidence_ids + static_cast<std::int64_t>(edge) * kMaxLocalSecIncidence;
  for (std::uint8_t index = 0U; index < incidence_count[edge]; ++index) {
    reduced -= quantized_cut[cuts[index]];
  }
  return reduced;
}

__global__ void ReplayLpNonpairKernel(
    const quick_hs::GraphView graph, const std::int32_t* const edge_u,
    const std::int32_t* const edge_v, const std::int64_t* const edge_weight,
    const std::int64_t denominator, const std::int64_t* const quantized,
    const std::int64_t* const quantized_cut, const std::uint8_t* const incidence_count,
    const std::int32_t* const incidence_ids, const Signed128* const lower_bound,
    const std::int64_t incumbent_numerator, const std::int64_t* const pair_offsets,
    const std::uint8_t* const proposed_nonpair, const std::int32_t* const fixed_witness,
    const std::int32_t* const point_witness, const std::uint8_t* const fixed_bits,
    std::uint8_t* const verified_nonpair, unsigned long long* const replayed,
    unsigned long long* const rejected) {
  const std::int32_t center = static_cast<std::int32_t>(blockIdx.x);
  if (center >= graph.dimension) {
    return;
  }
  const std::int64_t begin = quick_hs::NeighborBegin(graph, center);
  const std::int64_t degree = quick_hs::NeighborEnd(graph, center) - begin;
  const std::int64_t pair_begin = pair_offsets[center];
  for (std::int64_t first = threadIdx.x; first + 1 < degree; first += blockDim.x) {
    const std::int32_t first_edge = graph.neighbor_edge_ids[begin + first];
    for (std::int64_t second = first + 1; second < degree; ++second) {
      const std::int64_t local = first * (2 * degree - first - 1) / 2 + (second - first - 1);
      const std::int64_t pair = pair_begin + local;
      if (proposed_nonpair[pair] == 0U || point_witness[pair] >= 0) {
        continue;
      }
      const std::int32_t second_edge = graph.neighbor_edge_ids[begin + second];
      bool valid = false;
      if (fixed_witness[pair] >= 0) {
        valid = FixedAnchorProvesNonpair(graph, center, begin + first, begin + second,
                                         fixed_witness[pair], edge_u, edge_v, graph.edge_active,
                                         fixed_bits);
      } else if (lower_bound != nullptr) {
        const std::int64_t first_reduced =
            ReplayReducedCost(first_edge, edge_u, edge_v, edge_weight, denominator, quantized,
                              quantized_cut, incidence_count, incidence_ids);
        const std::int64_t second_reduced =
            ReplayReducedCost(second_edge, edge_u, edge_v, edge_weight, denominator, quantized,
                              quantized_cut, incidence_count, incidence_ids);
        Signed128 forced = *lower_bound;
        forced = Signed128AddInt64(forced, first_reduced > 0 ? first_reduced : 0);
        forced = Signed128AddInt64(forced, second_reduced > 0 ? second_reduced : 0);
        valid = Signed128GreaterThanInt64(forced, incumbent_numerator);
      }
      atomicAdd(replayed, 1ULL);
      if (valid) {
        verified_nonpair[pair] = 1U;
      } else {
        atomicAdd(rejected, 1ULL);
      }
    }
  }
}

__global__ void ReplayLpFixedKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const quick_hs::GraphView graph, const std::int32_t* const edge_u,
    const std::int32_t* const edge_v, const std::int64_t* const edge_weight,
    const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
    const std::int64_t denominator, const std::int64_t* const quantized,
    const std::int64_t* const quantized_cut, const std::uint8_t* const incidence_count,
    const std::int32_t* const incidence_ids, const Signed128* const lower_bound,
    const std::int64_t incumbent_numerator, const std::uint8_t* const proposed_fixed,
    const std::uint8_t* const fixed_reason, std::uint8_t* const verified_fixed,
    unsigned long long* const replayed, unsigned long long* const rejected) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  const std::uint8_t reason = fixed_reason[edge];
  const bool candidate = proposed_fixed[edge] != 0U && reason != kFixedReasonDirect;
  if (!candidate) {
    return;
  }
  const std::int64_t reduced =
      reason == kFixedReasonLp && lower_bound != nullptr
          ? ReplayReducedCost(edge, edge_u, edge_v, edge_weight, denominator, quantized,
                              quantized_cut, incidence_count, incidence_ids)
          : 0;
  const bool active_candidate = edge_active[edge] != 0U && protected_edge[edge] == 0U;
  const bool valid = active_candidate &&
                     (reason == kFixedReasonNonpair
                          ? (NonpairsForceEdgeAtEndpoint(graph, edge, edge_u[edge]) ||
                             NonpairsForceEdgeAtEndpoint(graph, edge, edge_v[edge]))
                          : reason == kFixedReasonLp && lower_bound != nullptr &&
                                Signed128GreaterThanInt64(
                                    Signed128AddInt64(*lower_bound, reduced < 0 ? -reduced : 0),
                                    incumbent_numerator));
  RecordReplayResult(true, valid, verified_fixed, edge, replayed, rejected);
}

__global__ void ApplyFixedKernel(const std::int32_t work_count,
                                 const std::int32_t* const active_edge_ids,
                                 const std::uint8_t* const proposed_delete,
                                 const std::uint8_t* const authorized_fixed,
                                 std::uint8_t* const protected_edge, std::uint8_t* const fixed_edge,
                                 unsigned long long* const fixed_count,
                                 std::int32_t* const invalid) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (authorized_fixed[edge] == 0U) {
    return;
  }
  if (proposed_delete[edge] != 0U) {
    atomicExch(invalid, 1);
    return;
  }
  fixed_edge[edge] = 1U;
  protected_edge[edge] = 1U;
  atomicAdd(fixed_count, 1ULL);
}

__global__ void
CountFixedDegreeKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                       const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                       const std::uint8_t* const edge_active, const std::uint8_t* const fixed_edge,
                       std::int32_t* const fixed_degree) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] != 0U && fixed_edge[edge] != 0U) {
    atomicAdd(&fixed_degree[edge_u[edge]], 1);
    atomicAdd(&fixed_degree[edge_v[edge]], 1);
  }
}

__global__ void ValidateFixedDegreeKernel(const std::int32_t dimension,
                                          const std::int32_t* const fixed_degree,
                                          std::int32_t* const invalid) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex < dimension && fixed_degree[vertex] > 2) {
    // bit 1：与普通 degree-floor（bit 0）分开，便于 fail-closed 诊断。
    atomicOr(invalid, 2);
  }
}

__global__ void
FixedPropagationKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                       const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                       const std::uint8_t* const edge_active,
                       const std::uint8_t* const protected_edge,
                       const std::int32_t* const fixed_degree, std::uint8_t* const proposed) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] != 0U && protected_edge[edge] == 0U &&
      (fixed_degree[edge_u[edge]] >= 2 || fixed_degree[edge_v[edge]] >= 2)) {
    proposed[edge] = 1U;
  }
}

__global__ void ReplayFixedPropagationKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
    const std::int32_t* const fixed_degree, const std::uint8_t* const proposed,
    std::uint8_t* const verified, unsigned long long* const replayed,
    unsigned long long* const rejected) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  const bool candidate = proposed[edge] != 0U;
  const bool valid = edge_active[edge] != 0U && protected_edge[edge] == 0U &&
                     (fixed_degree[edge_u[edge]] >= 2 || fixed_degree[edge_v[edge]] >= 2);
  RecordReplayResult(candidate, valid, verified, edge, replayed, rejected);
}

__device__ void RecordReplayResult(const bool proposed, const bool valid,
                                   std::uint8_t* const verified, const std::int32_t edge,
                                   unsigned long long* const replayed,
                                   unsigned long long* const rejected) {
  if (!proposed) {
    return;
  }
  atomicAdd(replayed, 1ULL);
  if (valid) {
    verified[edge] = 1U;
  } else {
    atomicAdd(rejected, 1ULL);
  }
}

__global__ void
ReplayJvKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
               const std::int32_t* const edge_u, const std::int32_t* const edge_v,
               const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
               const quick_hs::GraphView graph, const std::uint8_t* const proposed,
               const std::int32_t* const witness, std::uint8_t* const verified,
               unsigned long long* const replayed, unsigned long long* const rejected) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  const bool candidate = proposed[edge] != 0U;
  const std::int32_t c = witness[edge];
  const bool valid = edge_active[edge] != 0U && protected_edge[edge] == 0U && c >= 0 &&
                     c < graph.dimension && graph.degree[edge_u[edge]] > 2 &&
                     graph.degree[edge_v[edge]] > 2 &&
                     IsJvWitness(graph, edge_u[edge], edge_v[edge], c);
  RecordReplayResult(candidate, valid, verified, edge, replayed, rejected);
}

template <std::int32_t ExtraEdgeDepth>
__global__ void ReplayQuickHsContinuationKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
    const quick_hs::GraphView graph, const std::uint8_t* const proposed,
    const std::int32_t* const first_witness, const std::int32_t* const second_witness,
    std::uint8_t* const verified, unsigned long long* const replayed,
    unsigned long long* const rejected, const std::int64_t* const reduced_cost,
    const Signed128* const lower_bound, const std::int64_t incumbent_numerator) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x);
  if (work >= work_count) {
    return;
  }
  __shared__ std::int32_t enabled;
  const std::int32_t edge = active_edge_ids[work];
  const std::int32_t a = edge_u[edge];
  const std::int32_t b = edge_v[edge];
  const std::int32_t c = first_witness[edge];
  const std::int32_t d = second_witness[edge];
  if (threadIdx.x == 0U) {
    enabled = proposed[edge] != 0U && edge_active[edge] != 0U && protected_edge[edge] == 0U &&
              c >= 0 && d >= 0 && c < graph.dimension && d < graph.dimension && c != d &&
              graph.degree[a] > 2 && graph.degree[b] > 2;
    if (enabled != 0) {
      enabled = !quick_hs::Compatible(graph, a, b, c, d, quick_hs::Distance(graph, a, b));
    }
  }
  __syncthreads();
  const bool valid =
      enabled != 0 && RepliesClosedCta<ExtraEdgeDepth>(graph, a, b, c, d, reduced_cost, lower_bound,
                                                       incumbent_numerator, nullptr);
  if (threadIdx.x == 0U) {
    RecordReplayResult(proposed[edge] != 0U, valid, verified, edge, replayed, rejected);
  }
}

__global__ void
ReplayGeometryKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                     const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                     const std::uint8_t* const edge_active,
                     const std::uint8_t* const protected_edge, const quick_hs::GraphView graph,
                     const std::int64_t* const x, const std::int64_t* const y,
                     const std::int64_t* const nearest, const std::uint8_t* const proposed,
                     const std::int32_t* const first_witness,
                     const std::int32_t* const second_witness, std::uint8_t* const verified,
                     unsigned long long* const replayed, unsigned long long* const rejected) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  const bool candidate = proposed[edge] != 0U;
  const std::int32_t p = edge_u[edge];
  const std::int32_t q = edge_v[edge];
  const std::int32_t r = first_witness[edge];
  const std::int32_t s = second_witness[edge];
  bool valid = edge_active[edge] != 0U && protected_edge[edge] == 0U && r >= 0 && s >= 0 &&
               r < graph.dimension && s < graph.dimension && r != s && r != p && r != q && s != p &&
               s != q && graph.degree[p] > 2 && graph.degree[q] > 2;
  IntervalGeometryPotential first;
  IntervalGeometryPotential second;
  if (valid) {
    valid = IntervalPotentialBounds(graph, p, q, r, nearest[r], x, y, &first) &&
            IntervalPotentialBounds(graph, p, q, s, nearest[s], x, y, &second);
  }
  if (valid) {
    const std::int64_t original = quick_hs::Distance(graph, p, q) + quick_hs::Distance(graph, r, s);
    valid = quick_hs::Distance(graph, p, r) + quick_hs::Distance(graph, q, s) < original &&
            quick_hs::Distance(graph, p, s) + quick_hs::Distance(graph, q, r) < original;
    const DeviceInterval lpq = IntervalInteger(quick_hs::Distance(graph, p, q));
    const DeviceInterval lrs = IntervalInteger(quick_hs::Distance(graph, r, s));
    const DeviceInterval first_bound =
        IntervalSubtract(IntervalAdd(IntervalAdd(lpq, second.min_p), first.min_q), lrs);
    const DeviceInterval second_bound =
        IntervalSubtract(IntervalAdd(IntervalAdd(lpq, first.min_p), second.min_q), lrs);
    valid = valid && IntervalPositive(first_bound) && IntervalPositive(second_bound);
  }
  RecordReplayResult(candidate, valid, verified, edge, replayed, rejected);
}

__global__ void MaximumNeighborSpanKernel(const quick_hs::GraphView graph,
                                          unsigned long long* const maximum) {
  const int vertex = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex < graph.dimension) {
    // CSR 可以保留 inactive slot；必须按 slot 跨度分配，不能按当前活动度数分配。
    atomicMax(maximum, static_cast<unsigned long long>(quick_hs::NeighborEnd(graph, vertex) -
                                                       quick_hs::NeighborBegin(graph, vertex)));
  }
}

__device__ bool MainPotentialHasAllowedPairCta(const quick_hs::GraphView graph,
                                               const std::int32_t p, const std::int32_t q,
                                               const std::int32_t middle,
                                               const bool enable_strong_metric,
                                               std::uint32_t* const pair_cache,
                                               const bool full_metric) {
  const std::uint64_t degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, middle) -
                                                          quick_hs::NeighborBegin(graph, middle));
  const std::uint64_t pair_count = degree < 2U ? 0U : degree * (degree - 1U) / 2U;
  if (full_metric) {
    const std::uint64_t word_count = (pair_count + 31U) / 32U;
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x / 32U;
    bool any_allowed = false;
    // 每个 warp 独占一个 bitset word，避免非原子 word 写与其它 pair 的位发生竞争。
    for (std::uint64_t word = warp; word < word_count; word += blockDim.x / 32U) {
      std::uint32_t bits = 0U;
      for (unsigned bit = 0; bit < 32U && word * 32U + bit < pair_count; ++bit) {
        std::int32_t first = -1, second = -1;
        const bool decoded = DecodeNeighborPair(graph, middle, word * 32U + bit, &first, &second);
        if (decoded && main_metric::AllowedPairWarp(graph, p, q, first, middle, second)) {
          bits |= 1U << bit;
        }
      }
      if (lane == 0U)
        pair_cache[word] = bits;
      any_allowed |= bits != 0U;
    }
    return __syncthreads_or(any_allowed) != 0;
  }
  bool any_allowed = false;
  for (std::uint64_t window = 0; window < pair_count; window += blockDim.x) {
    const std::uint64_t ordinal = window + threadIdx.x;
    bool allowed = false;
    if (ordinal < pair_count) {
      std::int32_t first = -1;
      std::int32_t second = -1;
      allowed =
          DecodeNeighborPair(graph, middle, ordinal, &first, &second) &&
          (enable_strong_metric ? main_edge::AllowedPair(graph, p, q, middle, first, second)
                                : main_edge::BasicAllowedPair(graph, p, q, middle, first, second));
    }
    if (pair_cache != nullptr) {
      // 每个 warp 唯一写入一个 word；尾部不足 32 个 pair 的位显式置零。
      const auto mask = __ballot_sync(0xffffffffU, allowed);
      if ((threadIdx.x & 31U) == 0U && ordinal < pair_count) {
        pair_cache[ordinal / 32U] = mask;
      }
    }
    if (__syncthreads_or(allowed ? 1 : 0) != 0) {
      any_allowed = true;
      if (pair_cache == nullptr) {
        return true;
      }
    }
  }
  return any_allowed;
}

__device__ bool MainPotentialPairAdmitsTourCta(const quick_hs::GraphView graph,
                                               const std::int32_t p, const std::int32_t q,
                                               const std::int32_t r, const std::int32_t s,
                                               const bool enable_strong_metric,
                                               const std::uint32_t* const r_cache,
                                               const std::uint32_t* const s_cache) {
  const std::uint64_t r_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, r) -
                                                            quick_hs::NeighborBegin(graph, r));
  const std::uint64_t s_degree = static_cast<std::uint64_t>(quick_hs::NeighborEnd(graph, s) -
                                                            quick_hs::NeighborBegin(graph, s));
  const std::uint64_t r_pairs = r_degree < 2U ? 0U : r_degree * (r_degree - 1U) / 2U;
  const std::uint64_t s_pairs = s_degree < 2U ? 0U : s_degree * (s_degree - 1U) / 2U;
  if (s_pairs != 0U && r_pairs > ULLONG_MAX / s_pairs) {
    return true;
  }
  const std::uint64_t reply_count = r_pairs * s_pairs;
  for (std::uint64_t window = 0; window < reply_count; window += blockDim.x) {
    const std::uint64_t reply = window + threadIdx.x;
    bool admits = false;
    if (reply < reply_count) {
      std::int32_t r1 = -1;
      std::int32_t r2 = -1;
      std::int32_t s1 = -1;
      std::int32_t s2 = -1;
      const bool decoded = DecodeNeighborPair(graph, r, reply / s_pairs, &r1, &r2) &&
                           DecodeNeighborPair(graph, s, reply % s_pairs, &s1, &s2);
      const auto r_pair = reply / s_pairs;
      const auto s_pair = reply % s_pairs;
      const bool r_allowed =
          decoded &&
          (r_cache != nullptr
               ? ((r_cache[r_pair / 32U] >> (r_pair % 32U)) & 1U) != 0U
               : (enable_strong_metric ? main_edge::AllowedPair(graph, p, q, r, r1, r2)
                                       : main_edge::BasicAllowedPair(graph, p, q, r, r1, r2)));
      const bool s_allowed =
          decoded &&
          (s_cache != nullptr
               ? ((s_cache[s_pair / 32U] >> (s_pair % 32U)) & 1U) != 0U
               : (enable_strong_metric ? main_edge::AllowedPair(graph, p, q, s, s1, s2)
                                       : main_edge::BasicAllowedPair(graph, p, q, s, s1, s2)));
      admits = decoded && r_allowed && s_allowed &&
               !main_edge::MainEdgeEliminates(graph, p, q, r1, r, r2, s1, s, s2);
    }
    if (__syncthreads_or(admits ? 1 : 0) != 0) {
      return true;
    }
  }
  return false;
}

template <bool Replay>
__global__ void MainEdgeContinuationKernel(
    const std::int32_t work_count, const std::int32_t* const active_edge_ids,
    const std::int32_t* const edge_u, const std::int32_t* const edge_v,
    const std::uint8_t* const edge_active, const std::uint8_t* const protected_edge,
    const quick_hs::GraphView graph, const std::int64_t* const x, const std::int64_t* const y,
    const GeometryKdNode* const kd_nodes, const std::int32_t kd_root,
    const std::int32_t potential_count, const std::int32_t position_numerator,
    const std::int32_t position_denominator, const bool enable_strong_metric,
    const std::uint8_t* const replay_proposed, std::uint8_t* const output,
    std::int32_t* const first_witness, std::int32_t* const second_witness,
    unsigned long long* const replayed, unsigned long long* const rejected,
    std::uint32_t* const pair_cache, const std::size_t pair_cache_words, const bool full_metric) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x);
  if (work >= work_count) {
    return;
  }
  __shared__ std::int32_t potentials[kMaxGeometryPotential];
  __shared__ double scores[kMaxGeometryPotential];
  __shared__ std::int32_t selected;
  __shared__ std::int32_t enabled;
  __shared__ std::int32_t proven;
  __shared__ std::int32_t proof_first;
  __shared__ std::int32_t proof_second;

  const std::int32_t edge = active_edge_ids[work];
  const std::int32_t p = edge_u[edge];
  const std::int32_t q = edge_v[edge];
  // 生命周期仅限本次 kernel 的 (root, center, snapshot)，绝不写全局 nonpair。
  auto* const root_cache =
      pair_cache == nullptr
          ? nullptr
          : pair_cache + static_cast<std::size_t>(work) * potential_count * pair_cache_words;
  if (threadIdx.x == 0U) {
    selected = 0;
    proven = 0;
    proof_first = -1;
    proof_second = -1;
    const bool requested = !Replay || replay_proposed[edge] != 0U;
    enabled = requested && edge_active[edge] != 0U && protected_edge[edge] == 0U &&
              graph.degree[p] > 2 && graph.degree[q] > 2;
    if constexpr (Replay) {
      // replay 不重新执行几何排序和 OR 搜索：只验证成功的一个中心或一对中心。
      // 任意非端点都可以作合法见证；不需要证明它仍位于 proposer 的候选排名内。
      if (enabled) {
        const int first = first_witness[edge], second = second_witness[edge];
        enabled = first >= 0 && first < graph.dimension && first != p && first != q &&
                  (second == -1 || (second >= 0 && second < graph.dimension && second != first &&
                                    second != p && second != q));
        potentials[0] = first;
        potentials[1] = second;
        selected = second == -1 ? 1 : 2;
      }
    } else {
      enabled = enabled &&
                SelectGeometryCandidatesAtPosition(p, q, kd_nodes, kd_root, x, y,
                                                   position_numerator, position_denominator,
                                                   potential_count, potentials, scores, &selected);
    }
  }
  __syncthreads();
  if (enabled != 0) {
    for (std::int32_t index = 0; index < selected; ++index) {
      const bool allowed = MainPotentialHasAllowedPairCta(
          graph, p, q, potentials[index], enable_strong_metric,
          root_cache == nullptr ? nullptr : root_cache + index * pair_cache_words, full_metric);
      if (threadIdx.x == 0U && !allowed) {
        proven = 1;
        proof_first = potentials[index];
      }
      __syncthreads();
      if (proven != 0) {
        break;
      }
    }
    if (proven == 0) {
      for (std::int32_t first = 0; first < selected && proven == 0; ++first) {
        for (std::int32_t second = first + 1; second < selected; ++second) {
          const bool admits = MainPotentialPairAdmitsTourCta(
              graph, p, q, potentials[first], potentials[second], enable_strong_metric,
              root_cache == nullptr ? nullptr : root_cache + first * pair_cache_words,
              root_cache == nullptr ? nullptr : root_cache + second * pair_cache_words);
          if (threadIdx.x == 0U && !admits) {
            proven = 1;
            proof_first = potentials[first];
            proof_second = potentials[second];
          }
          __syncthreads();
          if (proven != 0) {
            break;
          }
        }
      }
    }
  }
  if (threadIdx.x == 0U) {
    if constexpr (Replay) {
      RecordReplayResult(replay_proposed[edge] != 0U, enabled != 0 && proven != 0, output, edge,
                         replayed, rejected);
    } else if (proven != 0) {
      output[edge] = 1U;
      first_witness[edge] = proof_first;
      second_witness[edge] = proof_second;
    }
  }
}

__global__ void
ReplayLpKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
               const std::int32_t* const edge_u, const std::int32_t* const edge_v,
               const std::int64_t* const edge_weight, const std::uint8_t* const edge_active,
               const std::uint8_t* const protected_edge, const std::int32_t* const degree,
               const std::int64_t denominator, const std::int64_t* const quantized,
               const std::int64_t* const quantized_cut, const std::uint8_t* const incidence_count,
               const std::int32_t* const incidence_ids, const Signed128* const lower_bound,
               const std::int64_t incumbent_numerator, const std::uint8_t* const proposed,
               std::uint8_t* const verified, unsigned long long* const replayed,
               unsigned long long* const rejected) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  const bool candidate = proposed[edge] != 0U;
  const std::int32_t u = edge_u[edge];
  const std::int32_t v = edge_v[edge];
  std::int64_t reduced = edge_weight[edge] * denominator - quantized[u] - quantized[v];
  const std::int32_t* const cuts =
      incidence_ids + static_cast<std::int64_t>(edge) * kMaxLocalSecIncidence;
  for (std::uint8_t index = 0U; index < incidence_count[edge]; ++index) {
    reduced -= quantized_cut[cuts[index]];
  }
  const bool valid =
      edge_active[edge] != 0U && protected_edge[edge] == 0U && degree[u] > 2 && degree[v] > 2 &&
      Signed128GreaterThanInt64(Signed128AddInt64(*lower_bound, reduced > 0 ? reduced : 0),
                                incumbent_numerator);
  RecordReplayResult(candidate, valid, verified, edge, replayed, rejected);
}

// stable edge id 定义了冲突优先级。对每个端点，只选中该点前
// degree-2 个候选；因此所有选中边可在下一个 kernel 并行提交，
// 且与 CTA 调度无关。被另一端点拒绝的边会在下一 epoch 重新竞争。
__global__ void SelectDegreeFloorCommitKernel(const std::int32_t work_count,
                                              const std::int32_t* const active_edge_ids,
                                              const quick_hs::GraphView graph,
                                              const std::uint8_t* const protected_edge,
                                              const std::uint8_t* const authorized,
                                              std::uint8_t* const committed) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (graph.edge_active[edge] == 0U || authorized[edge] == 0U || protected_edge[edge] != 0U) {
    return;
  }
  const std::int32_t endpoints[2]{graph.edge_u[edge], graph.edge_v[edge]};
  for (const std::int32_t node : endpoints) {
    const std::int32_t capacity = graph.degree[node] - 2;
    if (capacity <= 0) {
      return;
    }
    std::int32_t rank = 0;
    for (std::int64_t slot = quick_hs::NeighborBegin(graph, node);
         slot < quick_hs::NeighborEnd(graph, node); ++slot) {
      const std::int32_t candidate = graph.neighbor_edge_ids[slot];
      if (candidate >= edge) {
        continue;
      }
      if (graph.edge_active[candidate] != 0U && authorized[candidate] != 0U &&
          protected_edge[candidate] == 0U) {
        ++rank;
      }
    }
    if (rank >= capacity) {
      return;
    }
  }
  committed[edge] = 1U;
}

__global__ void SelectProtectedFloorCommitKernel(const std::int32_t work_count,
                                                 const std::int32_t* const active_edge_ids,
                                                 const std::uint8_t* const edge_active,
                                                 const std::uint8_t* const protected_edge,
                                                 const std::uint8_t* const authorized,
                                                 std::uint8_t* const committed) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] != 0U && protected_edge[edge] == 0U && authorized[edge] != 0U) {
    committed[edge] = 1U;
  }
}

__global__ void ApplyCommitKernel(const std::int32_t work_count,
                                  const std::int32_t* const active_edge_ids,
                                  const std::int32_t* const edge_u,
                                  const std::int32_t* const edge_v, std::uint8_t* const edge_active,
                                  const std::uint8_t* const committed, std::int32_t* const degree,
                                  unsigned long long* const committed_count,
                                  std::uint32_t* const dirty_vertex) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (committed[edge] == 0U) {
    return;
  }
  edge_active[edge] = 0U;
  atomicSub(&degree[edge_u[edge]], 1);
  atomicSub(&degree[edge_v[edge]], 1);
  atomicExch(&dirty_vertex[edge_u[edge]], 1U);
  atomicExch(&dirty_vertex[edge_v[edge]], 1U);
  atomicAdd(committed_count, 1ULL);
}

__global__ void
MarkDegreeTwoFixedKernel(const std::int32_t work_count, const std::int32_t* const active_edge_ids,
                         const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                         const std::uint8_t* const edge_active, const std::int32_t* const degree,
                         std::uint8_t* const protected_edge, std::uint8_t* const fixed_edge) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= work_count) {
    return;
  }
  const std::int32_t edge = active_edge_ids[work];
  if (edge_active[edge] != 0U && (degree[edge_u[edge]] == 2 || degree[edge_v[edge]] == 2)) {
    // 每条 stable edge 只有一个线程写，赋值幂等；显式 fixed state 随后
    // 被 propagation、non-pair 与所有 HT 谓词共同消费。
    fixed_edge[edge] = 1U;
    protected_edge[edge] = 1U;
  }
}

__global__ void ValidateDegreeFloorKernel(const std::int32_t dimension,
                                          const std::int32_t* const degree,
                                          std::int32_t* const invalid) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node < dimension && degree[node] < 2) {
    atomicOr(invalid, 1);
  }
}

constexpr std::int32_t kQuickHsDifferentialDimension = 12;
constexpr std::uint32_t kQuickHsDifferentialWarpsPerBlock = 2U;

__global__ void
QuickHsPathDifferentialKernel(const std::uint32_t sample_count, const std::int64_t* const distance,
                              const std::uint8_t* const active, const std::int32_t* const degree,
                              const std::int32_t* const neighbors,
                              const quick_hs::SmallPath* const paths, std::uint8_t* const result) {
  const std::uint32_t warp = threadIdx.x >> 5U;
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t sample = blockIdx.x * kQuickHsDifferentialWarpsPerBlock + warp;
  if (sample >= sample_count) {
    return;
  }

  constexpr std::size_t kMatrixSize =
      static_cast<std::size_t>(kQuickHsDifferentialDimension) * kQuickHsDifferentialDimension;
  quick_hs::GraphView graph{
      .dimension = kQuickHsDifferentialDimension,
      .degree = degree + static_cast<std::size_t>(sample) * kQuickHsDifferentialDimension,
      .neighbors = neighbors + static_cast<std::size_t>(sample) * kMatrixSize,
      .distance = distance + static_cast<std::size_t>(sample) * kMatrixSize,
      .active = active + static_cast<std::size_t>(sample) * kMatrixSize,
      .row_offsets = nullptr,
      .neighbor_edge_ids = nullptr,
      .pair_offsets = nullptr,
      .nonpair_mask = nullptr,
      .edge_u = nullptr,
      .edge_v = nullptr,
      .edge_active = nullptr,
      .fixed_edge = nullptr,
      .coordinate_x = nullptr,
      .coordinate_y = nullptr,
      .edge_count = 0,
      .distance_type = 0U,
      .complete_graph = true,
  };
  __shared__ std::int64_t
      distance_cache[kQuickHsDifferentialWarpsPerBlock * kMaximumWarpPathDistances];
  __shared__ std::int64_t
      subset_dp_cache[kQuickHsDifferentialWarpsPerBlock * kMaximumWarpSubsetDpStates];
  const auto* local_paths = paths + static_cast<std::size_t>(sample) * quick_hs::kMaxPathCount;
  if (local_paths[2].size == 0) {
    bool agrees = true;
    bool answer = false;
    for (int variant = 0; variant < 3; ++variant) {
      graph.point_leaf_kernel = static_cast<PointLeafKernel>(variant);
      const auto& a = local_paths[0];
      const auto& b = local_paths[1];
      const bool value =
          Opt34Warp(graph, a.node[0], a.node[1], a.node[2], b.node[0], b.node[1], b.node[2],
                    b.node[3], distance_cache + warp * kMaximumWarpPathDistances,
                    subset_dp_cache + warp * kMaximumWarpSubsetDpStates);
      if (variant > 0 && answer != value) {
        agrees = false;
      }
      answer = value;
    }
    if (lane == 0U) {
      result[sample] = agrees ? (answer ? 1U : 0U) : 3U;
    }
    return;
  }
  bool supported = false;
  const bool value = PathSystemOptWarp(
      graph, paths + static_cast<std::size_t>(sample) * quick_hs::kMaxPathCount,
      quick_hs::kMaxPathCount, &supported, distance_cache + warp * kMaximumWarpPathDistances,
      subset_dp_cache + warp * kMaximumWarpSubsetDpStates);
  if (lane == 0U) {
    // supported=false 意味 GPU 未完整覆盖该状态；测试将其记为不匹配，
    // 避免回退路径掩盖 warp-DP 的能力缺口。
    result[sample] = supported && value ? 1U : supported ? 0U : 2U;
  }
}

struct SparseAdjacency {
  std::vector<std::int64_t> row_offsets;
  std::vector<std::int32_t> neighbors;
  std::vector<std::int32_t> edge_ids;
};

SparseAdjacency BuildSparseAdjacency(const GraphSnapshot& graph) {
  SparseAdjacency result;
  result.row_offsets.assign(static_cast<std::size_t>(graph.dimension) + 1U, 0);
  for (const Edge& edge : graph.edges) {
    ++result.row_offsets[static_cast<std::size_t>(edge.u) + 1U];
    ++result.row_offsets[static_cast<std::size_t>(edge.v) + 1U];
  }
  for (std::int32_t node = 1; node <= graph.dimension; ++node) {
    result.row_offsets[static_cast<std::size_t>(node)] +=
        result.row_offsets[static_cast<std::size_t>(node - 1)];
  }
  const std::size_t slot_count = static_cast<std::size_t>(result.row_offsets.back());
  result.neighbors.resize(slot_count);
  result.edge_ids.resize(slot_count);
  std::vector<std::int64_t> cursor = result.row_offsets;
  for (std::size_t edge_id = 0; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    for (const auto [from, to] : {std::pair{edge.u, edge.v}, std::pair{edge.v, edge.u}}) {
      const std::size_t slot = static_cast<std::size_t>(cursor[static_cast<std::size_t>(from)]++);
      result.neighbors[slot] = to;
      result.edge_ids[slot] = static_cast<std::int32_t>(edge_id);
    }
  }
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    const std::size_t begin =
        static_cast<std::size_t>(result.row_offsets[static_cast<std::size_t>(node)]);
    const std::size_t end =
        static_cast<std::size_t>(result.row_offsets[static_cast<std::size_t>(node) + 1U]);
    std::vector<std::pair<std::int32_t, std::int32_t>> row;
    row.reserve(end - begin);
    for (std::size_t slot = begin; slot < end; ++slot) {
      row.emplace_back(result.neighbors[slot], result.edge_ids[slot]);
    }
    std::sort(row.begin(), row.end(), [&](const auto& lhs, const auto& rhs) {
      const Edge& lhs_edge = graph.edges[static_cast<std::size_t>(lhs.second)];
      const Edge& rhs_edge = graph.edges[static_cast<std::size_t>(rhs.second)];
      return std::tie(lhs_edge.weight, lhs.first, lhs.second) <
             std::tie(rhs_edge.weight, rhs.first, rhs.second);
    });
    for (std::size_t offset = 0; offset < row.size(); ++offset) {
      result.neighbors[begin + offset] = row[offset].first;
      result.edge_ids[begin + offset] = row[offset].second;
    }
  }
  return result;
}

bool IsCanonicalCompleteGraph(const GraphSnapshot& graph) {
  const std::int64_t dimension = graph.dimension;
  const std::int64_t expected = dimension * (dimension - 1) / 2;
  if (expected != static_cast<std::int64_t>(graph.edges.size())) {
    return false;
  }
  std::size_t edge_id = 0U;
  for (std::int32_t u = 0; u < graph.dimension; ++u) {
    for (std::int32_t v = u + 1; v < graph.dimension; ++v, ++edge_id) {
      if (graph.edges[edge_id].u != u || graph.edges[edge_id].v != v) {
        return false;
      }
    }
  }
  return true;
}

std::vector<GeometryKdNode> BuildGeometryKdTree(const GraphSnapshot& graph) {
  std::vector<std::int32_t> points(static_cast<std::size_t>(graph.dimension));
  std::iota(points.begin(), points.end(), 0);
  std::vector<GeometryKdNode> nodes;
  nodes.reserve(points.size());

  const auto coordinate = [&](const std::int32_t point, const bool split_x) {
    return split_x ? graph.points[static_cast<std::size_t>(point)].integer_x
                   : graph.points[static_cast<std::size_t>(point)].integer_y;
  };
  std::function<std::int32_t(std::size_t, std::size_t, bool)> build =
      [&](const std::size_t begin, const std::size_t end, const bool split_x) -> std::int32_t {
    if (begin == end) {
      return -1;
    }
    const std::size_t middle = begin + (end - begin) / 2U;
    std::nth_element(points.begin() + static_cast<std::ptrdiff_t>(begin),
                     points.begin() + static_cast<std::ptrdiff_t>(middle),
                     points.begin() + static_cast<std::ptrdiff_t>(end),
                     [&](const std::int32_t lhs, const std::int32_t rhs) {
                       const std::int64_t lhs_coordinate = coordinate(lhs, split_x);
                       const std::int64_t rhs_coordinate = coordinate(rhs, split_x);
                       return lhs_coordinate < rhs_coordinate ||
                              (lhs_coordinate == rhs_coordinate && lhs < rhs);
                     });
    const std::int32_t tree_index = static_cast<std::int32_t>(nodes.size());
    const std::int32_t point = points[middle];
    const auto& source = graph.points[static_cast<std::size_t>(point)];
    nodes.push_back({.point = point,
                     .left = -1,
                     .right = -1,
                     .min_x = source.integer_x,
                     .max_x = source.integer_x,
                     .min_y = source.integer_y,
                     .max_y = source.integer_y});
    const std::int32_t left = build(begin, middle, !split_x);
    const std::int32_t right = build(middle + 1U, end, !split_x);
    GeometryKdNode& output = nodes[static_cast<std::size_t>(tree_index)];
    output.left = left;
    output.right = right;
    for (const std::int32_t child : {left, right}) {
      if (child < 0) {
        continue;
      }
      const GeometryKdNode& child_node = nodes[static_cast<std::size_t>(child)];
      output.min_x = std::min(output.min_x, child_node.min_x);
      output.max_x = std::max(output.max_x, child_node.max_x);
      output.min_y = std::min(output.min_y, child_node.min_y);
      output.max_y = std::max(output.max_y, child_node.max_y);
    }
    return tree_index;
  };
  const std::int32_t root = build(0U, points.size(), true);
  if (root != 0 || nodes.size() != points.size()) {
    throw std::logic_error("resident geometry KD-tree 构造不完整");
  }
  return nodes;
}

} // namespace

bool ResidentEliminationCudaAvailable(std::string* const reason) {
  return SelectDevice(-1, reason) >= 0;
}

QuickHsPathDifferentialResult RunQuickHsPathDifferentialCuda(const int requested_device,
                                                             const std::uint32_t samples) {
  if (samples == 0U) {
    throw std::invalid_argument("Quick-HS GPU 差分测试的样本数必须大于 0");
  }
  std::string reason;
  const int device = SelectDevice(requested_device, &reason);
  if (device < 0) {
    throw std::runtime_error("Quick-HS GPU 差分测试无可用 CUDA: " + reason);
  }

  constexpr std::size_t kDimension = static_cast<std::size_t>(kQuickHsDifferentialDimension);
  constexpr std::size_t kMatrixSize = kDimension * kDimension;
  const std::size_t sample_count = static_cast<std::size_t>(samples);
  if (sample_count > std::numeric_limits<std::size_t>::max() / kMatrixSize) {
    throw std::overflow_error("Quick-HS GPU 差分测试的样本数溢出");
  }

  std::vector<std::int64_t> host_distance(sample_count * kMatrixSize, 0);
  std::vector<std::uint8_t> host_active(sample_count * kMatrixSize, 1U);
  std::vector<std::int32_t> host_degree(sample_count * kDimension,
                                        kQuickHsDifferentialDimension - 1);
  std::vector<std::int32_t> host_neighbors(sample_count * kMatrixSize, -1);
  std::vector<quick_hs::SmallPath> host_paths(sample_count * quick_hs::kMaxPathCount);
  std::vector<std::uint8_t> expected(sample_count, 0U);

  const auto fill_path = [](quick_hs::SmallPath* const path, const std::int32_t* const nodes,
                            const std::int32_t begin, const std::int32_t size) {
    *path = {};
    path->size = size;
    for (std::int32_t index = 0; index < size; ++index) {
      path->node[index] = nodes[begin + index];
    }
  };

  for (std::uint32_t sample = 0U; sample < samples; ++sample) {
    const std::size_t matrix_begin = static_cast<std::size_t>(sample) * kMatrixSize;
    for (std::int32_t first = 0; first < kQuickHsDifferentialDimension; ++first) {
      std::size_t slot = 0U;
      for (std::int32_t second = 0; second < kQuickHsDifferentialDimension; ++second) {
        if (second != first) {
          host_neighbors[matrix_begin + static_cast<std::size_t>(first) * kDimension + slot++] =
              second;
        }
      }
      host_active[matrix_begin + static_cast<std::size_t>(first) * kDimension +
                  static_cast<std::size_t>(first)] = 0U;
      for (std::int32_t second = first + 1; second < kQuickHsDifferentialDimension; ++second) {
        // 故意使用非度量对称权重：这里验证的是通用 path-order
        // 语义，不让欧氏三角不等式恰好遮蔽漏枚举。
        const std::uint64_t value = (static_cast<std::uint64_t>(sample + 1U) * 131U +
                                     static_cast<std::uint64_t>(first + 3) * 47U +
                                     static_cast<std::uint64_t>(second + 5) * 89U +
                                     static_cast<std::uint64_t>(first * second) * 17U +
                                     static_cast<std::uint64_t>(sample + 7U) *
                                         static_cast<std::uint64_t>(first + second + 1) * 13U) %
                                    997U;
        const std::int64_t scale = sample % 4U == 0U ? 1000000000LL : 1LL;
        const std::int64_t weight = static_cast<std::int64_t>(value + 1U) * scale;
        host_distance[matrix_begin + static_cast<std::size_t>(first) * kDimension +
                      static_cast<std::size_t>(second)] = weight;
        host_distance[matrix_begin + static_cast<std::size_t>(second) * kDimension +
                      static_cast<std::size_t>(first)] = weight;
      }
    }

    std::int32_t node[10]{};
    for (std::int32_t index = 0; index < 10; ++index) {
      node[index] =
          static_cast<std::int32_t>((static_cast<std::uint32_t>(index) * 5U + sample * 3U) %
                                    static_cast<std::uint32_t>(kQuickHsDifferentialDimension));
    }
    quick_hs::SmallPath* const paths =
        host_paths.data() + static_cast<std::size_t>(sample) * quick_hs::kMaxPathCount;
    switch (sample % 12U) {
    case 0U:
      fill_path(&paths[0], node, 0, 2);
      fill_path(&paths[1], node, 2, 4);
      fill_path(&paths[2], node, 6, 4);
      break;
    case 1U:
      fill_path(&paths[0], node, 0, 2);
      fill_path(&paths[1], node, 2, 5);
      fill_path(&paths[2], node, 7, 3);
      break;
    case 2U:
      fill_path(&paths[0], node, 0, 3);
      fill_path(&paths[1], node, 3, 4);
      fill_path(&paths[2], node, 7, 3);
      break;
    case 3U:
      fill_path(&paths[0], node, 0, 2);
      fill_path(&paths[1], node, 2, 3);
      fill_path(&paths[2], node, 5, 3);
      break;
    case 4U:
      fill_path(&paths[0], node, 0, 2);
      fill_path(&paths[1], node, 1, 4);
      fill_path(&paths[2], node, 5, 4);
      break;
    case 5U:
      fill_path(&paths[0], node, 0, 2);
      fill_path(&paths[1], node, 2, 4);
      fill_path(&paths[2], node, 6, 4);
      host_degree[static_cast<std::size_t>(sample) * kDimension +
                  static_cast<std::size_t>(node[1])] = 2;
      break;
    case 6U:
      fill_path(&paths[0], node, 0, 2);
      paths[1] = {.size = 3, .node = {node[1], node[2], node[3]}};
      paths[2] = {.size = 3, .node = {node[3], node[4], node[0]}};
      break;
    case 7U:
      fill_path(&paths[0], node, 0, 3);
      paths[1] = {.size = 3, .node = {node[3], node[4], node[0]}};
      paths[2] = {.size = 3, .node = {node[5], node[6], node[3]}};
      break;
    default:
      fill_path(&paths[0], node, 0, 3);
      fill_path(&paths[1], node, sample % 12U == 10U ? 2 : 3, 4);
      paths[2] = {};
      if (sample % 12U == 11U) {
        // 内部重叠必须保守地保持开放，不当作两个独立位置枚举。
        paths[1].node[1] = paths[0].node[1];
      }
      break;
    }

    quick_hs::GraphView host_graph{
        .dimension = kQuickHsDifferentialDimension,
        .degree = host_degree.data() + static_cast<std::size_t>(sample) * kDimension,
        .neighbors = host_neighbors.data() + matrix_begin,
        .distance = host_distance.data() + matrix_begin,
        .active = host_active.data() + matrix_begin,
        .row_offsets = nullptr,
        .neighbor_edge_ids = nullptr,
        .pair_offsets = nullptr,
        .nonpair_mask = nullptr,
        .edge_u = nullptr,
        .edge_v = nullptr,
        .edge_active = nullptr,
        .fixed_edge = nullptr,
        .coordinate_x = nullptr,
        .coordinate_y = nullptr,
        .edge_count = 0,
        .distance_type = 0U,
        .complete_graph = true,
    };
    expected[static_cast<std::size_t>(sample)] =
        quick_hs::Opt(host_graph, paths, paths[2].size == 0 ? 2 : quick_hs::kMaxPathCount) ? 1U
                                                                                           : 0U;
  }

  DeviceBuffer<std::int64_t> device_distance(host_distance.size(), device);
  DeviceBuffer<std::uint8_t> device_active(host_active.size(), device);
  DeviceBuffer<std::int32_t> device_degree(host_degree.size(), device);
  DeviceBuffer<std::int32_t> device_neighbors(host_neighbors.size(), device);
  DeviceBuffer<quick_hs::SmallPath> device_paths(host_paths.size(), device);
  DeviceBuffer<std::uint8_t> device_result(sample_count, device);
  device_distance.CopyFromHost(host_distance.data(), host_distance.size());
  device_active.CopyFromHost(host_active.data(), host_active.size());
  device_degree.CopyFromHost(host_degree.data(), host_degree.size());
  device_neighbors.CopyFromHost(host_neighbors.data(), host_neighbors.size());
  device_paths.CopyFromHost(host_paths.data(), host_paths.size());
  const std::uint32_t blocks =
      (samples + kQuickHsDifferentialWarpsPerBlock - 1U) / kQuickHsDifferentialWarpsPerBlock;
  QuickHsPathDifferentialKernel<<<blocks, kQuickHsDifferentialWarpsPerBlock * 32U>>>(
      samples, device_distance.get(), device_active.get(), device_degree.get(),
      device_neighbors.get(), device_paths.get(), device_result.get());
  CheckCuda(cudaGetLastError(), "QuickHsPathDifferentialKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "QuickHsPathDifferentialKernel synchronize");

  std::vector<std::uint8_t> actual(sample_count, 2U);
  device_result.CopyToHost(actual.data(), actual.size());
  QuickHsPathDifferentialResult result{.cases = sample_count};
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    result.mismatches += static_cast<std::size_t>(actual[sample] != expected[sample]);
  }
  return result;
}

ResidentGpuResult RunResidentEliminationCuda(const GraphSnapshot& graph,
                                             const std::vector<std::uint8_t>& protected_edges,
                                             const ResidentGpuOptions& options) {
  if (graph.dimension <= 0 || graph.edges.empty() ||
      (!graph.integer_coordinates && graph.integer_coordinate_denominator != 2U) ||
      (graph.integer_coordinate_denominator != 1U && graph.integer_coordinate_denominator != 2U) ||
      !graph.integer_distance_safe || protected_edges.size() != graph.edges.size() ||
      options.pdlp_iterations == 0U || options.potential_candidates < 2U ||
      options.potential_candidates > 32U || options.fractional_bits > 30U ||
      options.main_edge_potentials < 2U || options.main_edge_potentials > 32U ||
      options.main_edge_positions == 0U ||
      options.main_edge_positions >= static_cast<std::uint32_t>(INT32_MAX) ||
      options.quick_hs_candidates < 2U ||
      options.quick_hs_candidates > static_cast<std::uint32_t>(quick_hs::kMaxPotentialNodes) ||
      options.quick_hs_pair_trials > static_cast<std::uint32_t>(INT32_MAX) ||
      options.extra_edge_depth < 1U || options.extra_edge_depth > 2U ||
      (options.point_cta_blocks != 2U && options.point_cta_blocks != 4U) ||
      (options.enable_pdlp && options.incumbent_cost < 0) ||
      (options.full_metric && (!options.main_pair_cache || !options.enable_strong_metric)) ||
      (options.point_prime_near && (!options.point_adaptive_start || !options.point_near_first)) ||
      (options.collect_trace && !options.enable_pdlp &&
       (options.enable_point_nonpair || options.enable_direct_fix)) ||
      (!options.enable_quick_hs && !options.enable_jv && !options.enable_geometry &&
       !options.enable_pdlp && !options.enable_main_edge && !options.enable_extra_edge)) {
    throw std::invalid_argument("resident GPU 输入图、tour mask、阶段开关或预算非法");
  }
  if (graph.edges.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("resident GPU 的 int32 edge id 无法表示当前图");
  }
  std::string reason;
  const int device = SelectDevice(options.device, &reason);
  if (device < 0) {
    throw std::runtime_error("resident GPU CUDA 不可用: " + reason);
  }

  const SteadyClock::time_point solve_begin = SteadyClock::now();
  const std::size_t dimension = static_cast<std::size_t>(graph.dimension);
  const std::size_t edge_count = graph.edges.size();
  std::vector<std::int32_t> host_edge_u(edge_count);
  std::vector<std::int32_t> host_edge_v(edge_count);
  std::vector<std::int64_t> host_edge_weight(edge_count);
  std::vector<std::uint8_t> host_edge_active(edge_count);
  std::vector<std::int32_t> host_degree(dimension, 0);
  std::vector<std::int32_t> host_protected_degree(dimension, 0);
  for (std::size_t edge = 0; edge < edge_count; ++edge) {
    host_edge_u[edge] = graph.edges[edge].u;
    host_edge_v[edge] = graph.edges[edge].v;
    host_edge_weight[edge] = graph.edges[edge].weight;
    host_edge_active[edge] = graph.edges[edge].active ? 1U : 0U;
    if (graph.edges[edge].active) {
      ++host_degree[static_cast<std::size_t>(graph.edges[edge].u)];
      ++host_degree[static_cast<std::size_t>(graph.edges[edge].v)];
      if (protected_edges[edge] != 0U) {
        ++host_protected_degree[static_cast<std::size_t>(graph.edges[edge].u)];
        ++host_protected_degree[static_cast<std::size_t>(graph.edges[edge].v)];
      }
    }
  }
  const bool has_protected_degree_floor =
      std::all_of(host_protected_degree.begin(), host_protected_degree.end(),
                  [](const std::int32_t degree) { return degree >= 2; });
  const bool complete_graph = IsCanonicalCompleteGraph(graph);
  if (options.gpu_complete_graph && (!complete_graph || !options.enable_geometry)) {
    throw std::invalid_argument("GPU 初始完整 CSR 需要规范完整图及首个 geometry 阶段");
  }
  const SparseAdjacency host_adjacency =
      options.gpu_complete_graph ? SparseAdjacency{} : BuildSparseAdjacency(graph);
  const std::size_t initial_slots =
      options.gpu_complete_graph ? edge_count * 2U : host_adjacency.neighbors.size();
  const std::vector<GeometryKdNode> host_geometry_kd = BuildGeometryKdTree(graph);
  std::vector<std::int32_t> host_geometry_rank(dimension, -1);
  // KD 树按前序连续存储，同一子树在数组中连续。以该顺序构造重叠
  // 16--48 点局部 SEC，比原始 TSPLIB 编号窗口更贴近真实几何邻域。
  for (std::size_t rank = 0U; rank < host_geometry_kd.size(); ++rank) {
    host_geometry_rank[static_cast<std::size_t>(host_geometry_kd[rank].point)] =
        static_cast<std::int32_t>(rank);
  }
  std::vector<std::int64_t> host_x(dimension);
  std::vector<std::int64_t> host_y(dimension);
  for (std::size_t node = 0U; node < dimension; ++node) {
    host_x[node] = graph.points[node].integer_x;
    host_y[node] = graph.points[node].integer_y;
  }
  const std::int64_t denominator = std::int64_t{1} << options.fractional_bits;
  const LocalSecLayout local_sec = BuildLocalSecLayout(graph.dimension);
  const std::int64_t maximum_cut_count_wide =
      static_cast<std::int64_t>(local_sec.cut_count) +
      static_cast<std::int64_t>(kConnectivitySupportLevels) * graph.dimension;
  if (maximum_cut_count_wide > INT32_MAX) {
    throw std::overflow_error("resident GPU cut id 超出 int32 范围");
  }
  const std::int32_t maximum_cut_count = static_cast<std::int32_t>(maximum_cut_count_wide);
  const std::int64_t maximum_edge_weight =
      *std::max_element(host_edge_weight.begin(), host_edge_weight.end());
  const std::int64_t minimum_edge_weight =
      *std::min_element(host_edge_weight.begin(), host_edge_weight.end());
  // Quick-HS 最长局部路径至多累加 8 条边；额外留出 8 倍余量覆盖
  // compatibility/几何谓词和负哨兵，禁止任何有符号中间和溢出。
  if (minimum_edge_weight < 0 || maximum_edge_weight > INT64_MAX / 64) {
    throw std::overflow_error("resident GPU 距离不满足非负 int64 安全余量");
  }
  const double dual_limit =
      4.0 * static_cast<double>(std::max<std::int64_t>(maximum_edge_weight, 1));
  const long double maximum_quantized =
      static_cast<long double>(dual_limit) * static_cast<long double>(denominator);
  // 一条边最多同时跨越 kMaxLocalSecIncidence 个局部 SEC。reduced cost
  // 的 int64 门禁必须覆盖全部 cut multiplier，不能只按两个 degree dual 估计。
  const long double maximum_reduced =
      static_cast<long double>(maximum_edge_weight) * denominator +
      static_cast<long double>(2 + kMaxLocalSecIncidence) * maximum_quantized;
  if (options.enable_pdlp && (maximum_quantized > 1.0e15L ||
                              maximum_reduced > static_cast<long double>(INT64_MAX) / 2.0L ||
                              static_cast<long double>(options.incumbent_cost) * denominator >
                                  static_cast<long double>(INT64_MAX) / 2.0L)) {
    throw std::overflow_error("resident PDLP 单项 reduced cost 无法用 int64 精确表示");
  }

  DeviceBuffer<std::int32_t> device_edge_u(edge_count, device);
  DeviceBuffer<std::int32_t> device_edge_v(edge_count, device);
  DeviceBuffer<std::int64_t> device_edge_weight(edge_count, device);
  DeviceBuffer<std::uint8_t> device_edge_active(edge_count, device);
  DeviceBuffer<std::int32_t> device_all_edge_ids(edge_count, device);
  DeviceBuffer<std::int32_t> device_active_edge_ids(edge_count, device);
  DeviceBuffer<std::int32_t> device_active_edge_count(1U, device);
  DeviceBuffer<std::int32_t> device_dirty_root_edge_ids(edge_count, device);
  DeviceBuffer<std::int32_t> device_dirty_root_edge_count(1U, device);
  DeviceBuffer<std::uint8_t> device_dirty_root_flags(edge_count, device);
  DeviceBuffer<std::uint32_t> device_dirty_vertices(dimension, device);
  DeviceBuffer<std::uint32_t> device_dirty_vertices_next(dimension, device);
  DeviceBuffer<std::uint8_t> device_protected(protected_edges.size(), device);
  DeviceBuffer<std::int64_t> device_x(dimension, device);
  DeviceBuffer<std::int64_t> device_y(dimension, device);
  DeviceBuffer<std::int64_t> device_row_offsets(dimension + 1U, device);
  DeviceBuffer<std::int32_t> device_neighbors(initial_slots, device);
  DeviceBuffer<std::int32_t> device_neighbor_edge_ids(initial_slots, device);
  DeviceBuffer<std::int64_t> device_compact_row_offsets;
  DeviceBuffer<std::int64_t> device_compact_row_offsets_next;
  DeviceBuffer<std::int64_t> device_csr_counts;
  DeviceBuffer<std::int32_t> device_compact_neighbors;
  DeviceBuffer<std::int32_t> device_compact_neighbors_next;
  DeviceBuffer<std::int32_t> device_compact_neighbor_edge_ids;
  DeviceBuffer<std::int32_t> device_compact_neighbor_edge_ids_next;
  DeviceBuffer<unsigned long long> device_compact_cursor;
  DeviceBuffer<std::uint8_t> device_csr_scan_temp;
  DeviceBuffer<std::int64_t> device_pair_counts(dimension + 1U, device);
  DeviceBuffer<std::int64_t> device_pair_offsets(dimension + 1U, device);
  DeviceBuffer<std::int64_t> device_pair_offsets_next(dimension + 1U, device);
  DeviceBuffer<std::uint8_t> device_pair_scan_temp;
  DeviceBuffer<std::uint8_t> device_nonpair_mask;
  DeviceBuffer<std::uint8_t> device_nonpair_mask_next;
  DeviceBuffer<std::uint8_t> device_nonpair_proposed;
  DeviceBuffer<std::uint8_t> device_nonpair_verified;
  DeviceBuffer<std::int32_t> device_nonpair_fixed_witness;
  DeviceBuffer<std::int32_t> device_nonpair_point_witness;
  DeviceBuffer<unsigned long long> device_nonpair_proposal_counts(3U, device);
  DeviceBuffer<unsigned long long> device_nonpair_committed_count(1U, device);
  DeviceBuffer<GeometryKdNode> device_geometry_kd(host_geometry_kd.size(), device);
  DeviceBuffer<std::int32_t> device_geometry_rank(dimension, device);
  DeviceBuffer<std::int64_t> device_nearest(dimension, device);
  DeviceBuffer<std::int32_t> device_degree(dimension, device);
  DeviceBuffer<std::uint8_t> device_proposed(edge_count, device);
  DeviceBuffer<std::uint8_t> device_verified(edge_count, device);
  DeviceBuffer<std::uint8_t> device_fixed(edge_count, device);
  DeviceBuffer<std::uint8_t> device_pending_fixed(edge_count, device);
  DeviceBuffer<std::int32_t> device_pending_degree(dimension, device);
  DeviceBuffer<std::int32_t> device_fixed_parent(dimension, device);
  DeviceBuffer<std::int32_t> device_fixed_component_size(dimension, device);
  DeviceBuffer<std::int32_t> device_fixed_component_degree(dimension, device);
  DeviceBuffer<std::int32_t> device_fixed_edge_ids(edge_count, device);
  DeviceBuffer<std::int32_t> device_fixed_edge_count(1U, device);
  DeviceBuffer<std::uint8_t> device_fixed_proposed(edge_count, device);
  DeviceBuffer<std::uint8_t> device_fixed_verified(edge_count, device);
  DeviceBuffer<std::uint8_t> device_fixed_reason(edge_count, device);
  DeviceBuffer<std::uint8_t> device_committed(edge_count, device);
  DeviceBuffer<std::int32_t> device_first_witness(edge_count, device);
  DeviceBuffer<std::int32_t> device_second_witness(edge_count, device);
  DeviceBuffer<std::int32_t> device_point_priority;
  DeviceBuffer<std::uint32_t> device_point_priority_mask;
  DeviceBuffer<std::uint32_t> device_main_pair_cache;
  DeviceBuffer<unsigned long long> device_main_maximum_span(1U, device);
  std::size_t main_pair_cache_capacity{};
  DeviceBuffer<unsigned long long> device_committed_count(1U, device);
  DeviceBuffer<unsigned long long> device_replay_counters(2U, device);
  DeviceBuffer<unsigned long long> device_lp_path_closed_replies(1U, device);
  DeviceBuffer<unsigned long long> device_point_path_end_closed_replies(1U, device);
  DeviceBuffer<unsigned long long> device_fixed_count(1U, device);
  DeviceBuffer<unsigned long long> device_nonpair_fix_proposal_count(1U, device);
  DeviceBuffer<unsigned long long> device_direct_fix_proposal_count(1U, device);
  DeviceBuffer<std::int32_t> device_fixed_degree(dimension, device);
  DeviceBuffer<std::int32_t> device_selected_degree(dimension, device);
  DeviceBuffer<std::uint8_t> device_local_sec_incidence_count(edge_count, device);
  DeviceBuffer<std::int32_t> device_local_sec_incidence_ids(
      edge_count * static_cast<std::size_t>(kMaxLocalSecIncidence), device);
  DeviceBuffer<std::int32_t> device_selected_cut(static_cast<std::size_t>(maximum_cut_count),
                                                 device);
  DeviceBuffer<std::uint8_t> device_cut_active(static_cast<std::size_t>(maximum_cut_count), device);
  DeviceBuffer<std::uint8_t> device_cut_valid(static_cast<std::size_t>(maximum_cut_count), device);
  DeviceBuffer<std::int32_t> device_support_label(dimension, device);
  DeviceBuffer<std::int32_t> device_sec_membership(dimension * kConnectivitySupportLevels, device);
  DeviceBuffer<std::int32_t> device_sec_replay_sizes(dimension * kConnectivitySupportLevels,
                                                     device);
  DeviceBuffer<std::int32_t> device_component_size(dimension, device);
  DeviceBuffer<std::int32_t> device_component_changed(1U, device);
  DeviceBuffer<unsigned long long> device_connectivity_cut_count(1U, device);
  DeviceBuffer<double> device_dual(dimension, device);
  DeviceBuffer<double> device_degree_dual(dimension, device);
  DeviceBuffer<double> device_average(dimension, device);
  DeviceBuffer<double> device_cut_dual(static_cast<std::size_t>(maximum_cut_count), device);
  DeviceBuffer<double> device_cut_average(static_cast<std::size_t>(maximum_cut_count), device);
  DeviceBuffer<std::int64_t> device_quantized_dual(dimension, device);
  DeviceBuffer<std::int64_t> device_quantized_cut(static_cast<std::size_t>(maximum_cut_count),
                                                  device);
  DeviceBuffer<std::int64_t> device_reduced_cost(edge_count, device);
  // local SEC 次梯度并不保证在有限迭代内优于 degree-box。每个阈值
  // 独立产生候选快照，再由 device Signed128 精确竞赛，确保强化不倒退。
  DeviceBuffer<std::int64_t> device_degree_quantized_dual(dimension, device);
  DeviceBuffer<std::int64_t> device_degree_reduced_cost(edge_count, device);
  DeviceBuffer<Signed128> device_degree_lower_bound(1U, device);
  DeviceBuffer<std::int64_t> device_candidate_quantized_dual(dimension, device);
  DeviceBuffer<std::int64_t> device_candidate_quantized_cut(
      static_cast<std::size_t>(maximum_cut_count), device);
  DeviceBuffer<std::int64_t> device_candidate_reduced_cost(edge_count, device);
  DeviceBuffer<Signed128> device_candidate_lower_bound(1U, device);
  DeviceBuffer<std::int32_t> device_selected_lp_snapshot_kind(1U, device);
  DeviceBuffer<std::int32_t> device_apply_lp_snapshot(1U, device);
  // gpu-safe replay 用独立 reduced-cost 向量重新构造全局下界。
  DeviceBuffer<std::int64_t> device_replay_reduced_cost(edge_count, device);
  DeviceBuffer<Signed128> device_replay_lower_bound(1U, device);
  const std::size_t maximum_bound_terms =
      dimension + static_cast<std::size_t>(maximum_cut_count) + edge_count;
  constexpr std::size_t kBoundReductionThreads = 128U;
  const std::size_t bound_partial_capacity = std::max<std::size_t>(
      1U, std::min<std::size_t>(1024U, (maximum_bound_terms + kBoundReductionThreads - 1U) /
                                           kBoundReductionThreads));
  DeviceBuffer<Signed128> device_bound_partials(bound_partial_capacity, device);
  DeviceBuffer<unsigned long long> device_summary(2U, device);
  DeviceBuffer<Signed128> device_lower_bound(1U, device);
  DeviceBuffer<std::int32_t> device_invalid(1U, device);
  std::size_t active_select_temp_bytes = 0U;
  CheckCuda(cub::DeviceSelect::Flagged(nullptr, active_select_temp_bytes, device_all_edge_ids.get(),
                                       device_edge_active.get(), device_active_edge_ids.get(),
                                       device_active_edge_count.get(),
                                       static_cast<std::int32_t>(edge_count)),
            "cub::DeviceSelect::Flagged size query");
  DeviceBuffer<std::uint8_t> device_active_select_temp(active_select_temp_bytes, device);
  std::size_t pair_scan_temp_bytes = 0U;
  CheckCuda(cub::DeviceScan::ExclusiveSum(nullptr, pair_scan_temp_bytes, device_pair_counts.get(),
                                          device_pair_offsets.get(), graph.dimension + 1),
            "cub::DeviceScan::ExclusiveSum pair size query");
  device_pair_scan_temp = DeviceBuffer<std::uint8_t>(pair_scan_temp_bytes, device);

  ResidentGpuResult result;
  result.selected_device = device;
  result.backend = "cuda-fully-resident-geometry+pdlp+jv+quick-hs+extra-edge";
  result.initial_edges = graph.ActiveEdgeCount();
  if (options.enable_point_nonpair) {
    const auto point_kernel =
        options.point_cta_blocks == 4U ? PointNonpairKernel<4> : PointNonpairKernel<2>;
    cudaFuncAttributes attributes{};
    int active_blocks = 0;
    CheckCuda(cudaFuncGetAttributes(&attributes, point_kernel), "Point kernel 资源信息");
    CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, point_kernel, 128, 0),
              "Point kernel 理论驻留 CTA 数");
    result.lp.point_registers = static_cast<std::uint32_t>(attributes.numRegs);
    result.lp.point_active_blocks_per_sm = static_cast<std::uint32_t>(active_blocks);
    result.lp.point_local_bytes_per_thread = attributes.localSizeBytes;
  }
  result.resident_bytes =
      device_edge_u.bytes() + device_edge_v.bytes() + device_edge_weight.bytes() +
      device_edge_active.bytes() + device_all_edge_ids.bytes() + device_active_edge_ids.bytes() +
      device_active_edge_count.bytes() + device_active_select_temp.bytes() +
      device_dirty_root_edge_ids.bytes() + device_dirty_root_edge_count.bytes() +
      device_dirty_root_flags.bytes() + device_dirty_vertices.bytes() +
      device_dirty_vertices_next.bytes() + device_protected.bytes() + device_x.bytes() +
      device_y.bytes() + device_row_offsets.bytes() + device_neighbors.bytes() +
      device_neighbor_edge_ids.bytes() + device_geometry_kd.bytes() + device_geometry_rank.bytes() +
      device_nearest.bytes() + device_degree.bytes() + device_proposed.bytes() +
      device_verified.bytes() + device_fixed.bytes() + device_fixed_proposed.bytes() +
      device_fixed_verified.bytes() + device_fixed_reason.bytes() + device_committed.bytes() +
      device_first_witness.bytes() + device_second_witness.bytes() +
      device_committed_count.bytes() + device_replay_counters.bytes() +
      device_lp_path_closed_replies.bytes() + device_point_path_end_closed_replies.bytes() +
      device_fixed_count.bytes() + device_nonpair_fix_proposal_count.bytes() +
      device_direct_fix_proposal_count.bytes() + device_fixed_degree.bytes() +
      device_selected_degree.bytes() + device_local_sec_incidence_count.bytes() +
      device_local_sec_incidence_ids.bytes() + device_selected_cut.bytes() +
      device_cut_active.bytes() + device_cut_valid.bytes() + device_support_label.bytes() +
      device_sec_membership.bytes() + device_sec_replay_sizes.bytes() +
      device_component_size.bytes() + device_component_changed.bytes() +
      device_connectivity_cut_count.bytes() + device_dual.bytes() + device_degree_dual.bytes() +
      device_average.bytes() + device_cut_dual.bytes() + device_cut_average.bytes() +
      device_quantized_dual.bytes() + device_quantized_cut.bytes() + device_reduced_cost.bytes() +
      device_degree_quantized_dual.bytes() + device_degree_reduced_cost.bytes() +
      device_degree_lower_bound.bytes() + device_candidate_quantized_dual.bytes() +
      device_candidate_quantized_cut.bytes() + device_candidate_reduced_cost.bytes() +
      device_candidate_lower_bound.bytes() + device_selected_lp_snapshot_kind.bytes() +
      device_apply_lp_snapshot.bytes() + device_replay_reduced_cost.bytes() +
      device_replay_lower_bound.bytes() + device_bound_partials.bytes() + device_summary.bytes() +
      device_lower_bound.bytes() + device_invalid.bytes() + device_pair_counts.bytes() +
      device_pair_offsets.bytes() + device_pair_offsets_next.bytes() +
      device_pair_scan_temp.bytes() + device_nonpair_proposal_counts.bytes() +
      device_nonpair_committed_count.bytes() + device_fixed_edge_ids.bytes() +
      device_fixed_edge_count.bytes();

  const SteadyClock::time_point upload_begin = SteadyClock::now();
  device_edge_u.CopyFromHost(host_edge_u.data(), edge_count);
  device_edge_v.CopyFromHost(host_edge_v.data(), edge_count);
  device_edge_weight.CopyFromHost(host_edge_weight.data(), edge_count);
  device_edge_active.CopyFromHost(host_edge_active.data(), edge_count);
  device_geometry_rank.CopyFromHost(host_geometry_rank.data(), dimension);
  constexpr int kThreads = 128;
  const int all_edge_blocks =
      static_cast<int>((static_cast<std::int64_t>(edge_count) + kThreads - 1) / kThreads);
  InitializeEdgeIdsKernel<<<all_edge_blocks, kThreads>>>(static_cast<std::int32_t>(edge_count),
                                                         device_all_edge_ids.get());
  CheckCuda(cudaGetLastError(), "InitializeEdgeIdsKernel launch");
  CheckCuda(
      cudaMemset(device_local_sec_incidence_count.get(), 0, edge_count * sizeof(std::uint8_t)),
      "cudaMemset local SEC incidence count");
  if (local_sec.cut_count != 0) {
    BuildLocalSecIncidenceKernel<<<all_edge_blocks, kThreads>>>(
        static_cast<std::int32_t>(edge_count), device_all_edge_ids.get(), device_edge_u.get(),
        device_edge_v.get(), device_geometry_rank.get(), local_sec,
        device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get());
    CheckCuda(cudaGetLastError(), "BuildLocalSecIncidenceKernel launch");
  }
  device_protected.CopyFromHost(protected_edges.data(), protected_edges.size());
  device_x.CopyFromHost(host_x.data(), dimension);
  device_y.CopyFromHost(host_y.data(), dimension);
  device_row_offsets.CopyFromHost(host_adjacency.row_offsets.data(),
                                  host_adjacency.row_offsets.size());
  device_neighbors.CopyFromHost(host_adjacency.neighbors.data(), host_adjacency.neighbors.size());
  device_neighbor_edge_ids.CopyFromHost(host_adjacency.edge_ids.data(),
                                        host_adjacency.edge_ids.size());
  if (options.gpu_complete_graph) {
    InitializeCompleteAdjacencyKernel<<<graph.dimension, kThreads>>>(
        graph.dimension, device_row_offsets.get(), device_neighbors.get(),
        device_neighbor_edge_ids.get());
    CheckCuda(cudaGetLastError(), "InitializeCompleteAdjacencyKernel launch");
  }
  device_geometry_kd.CopyFromHost(host_geometry_kd.data(), host_geometry_kd.size());
  CheckCuda(cudaMemset(device_fixed.get(), 0, edge_count * sizeof(std::uint8_t)),
            "cudaMemset resident fixed");
  CheckCuda(cudaMemset(device_fixed_count.get(), 0, sizeof(unsigned long long)),
            "cudaMemset resident fixed count");
  CheckCuda(cudaMemset(device_dirty_vertices.get(), 0, dimension * sizeof(std::uint32_t)),
            "cudaMemset resident dirty vertices");
  CheckCuda(cudaMemset(device_dirty_vertices_next.get(), 0, dimension * sizeof(std::uint32_t)),
            "cudaMemset resident next dirty vertices");
  // Geometry/LP 只读度数而不读排序邻接表；直接上传初始度数可省掉完整图上
  // 最昂贵的一次 O(n^2) 邻接构建。后续提交会在 device 上同步维护它。
  device_degree.CopyFromHost(host_degree.data(), dimension);
  MarkDegreeTwoFixedKernel<<<all_edge_blocks, kThreads>>>(
      static_cast<std::int32_t>(edge_count), device_all_edge_ids.get(), device_edge_u.get(),
      device_edge_v.get(), device_edge_active.get(), device_degree.get(), device_protected.get(),
      device_fixed.get());
  CheckCuda(cudaGetLastError(), "MarkDegreeTwoFixedKernel initial launch");
  CheckCuda(cudaMemset(device_fixed_degree.get(), 0, dimension * sizeof(std::int32_t)),
            "cudaMemset resident initial fixed degree");
  CountFixedDegreeKernel<<<all_edge_blocks, kThreads>>>(
      static_cast<std::int32_t>(edge_count), device_all_edge_ids.get(), device_edge_u.get(),
      device_edge_v.get(), device_edge_active.get(), device_fixed.get(), device_fixed_degree.get());
  result.upload_ms = ElapsedMilliseconds(upload_begin);

  const int vertex_blocks =
      static_cast<int>((static_cast<std::int64_t>(graph.dimension) + kThreads - 1) / kThreads);
  const int offset_blocks =
      static_cast<int>((static_cast<std::int64_t>(graph.dimension) + 1 + kThreads - 1) / kThreads);
  quick_hs::GraphView view{.dimension = graph.dimension,
                           .degree = device_degree.get(),
                           .neighbors = device_neighbors.get(),
                           .distance = nullptr,
                           .active = nullptr,
                           .row_offsets = device_row_offsets.get(),
                           .neighbor_edge_ids = device_neighbor_edge_ids.get(),
                           .pair_offsets = nullptr,
                           .nonpair_mask = nullptr,
                           .edge_u = device_edge_u.get(),
                           .edge_v = device_edge_v.get(),
                           .edge_active = device_edge_active.get(),
                           .fixed_edge = device_fixed.get(),
                           .coordinate_x = device_x.get(),
                           .coordinate_y = device_y.get(),
                           .edge_count = static_cast<std::int64_t>(edge_count),
                           .distance_type = static_cast<std::uint8_t>(graph.distance_type),
                           .complete_graph = complete_graph,
                           .point_leaf_kernel = options.point_leaf_kernel,
                           .triangular_distance = options.triangular_distance,
                           .coordinate_denominator = graph.integer_coordinate_denominator,
                           .permutation_orders = options.permutation_orders};
  if (options.point_near_first) {
    const auto mask_words = dimension * ((dimension + 31U) / 32U);
    device_point_priority =
        DeviceBuffer<std::int32_t>(dimension * quick_hs::kMaxPotentialNodes, device);
    device_point_priority_mask = DeviceBuffer<std::uint32_t>(mask_words, device);
    CheckCuda(cudaMemset(device_point_priority_mask.get(), 0, device_point_priority_mask.bytes()),
              "clear Point priority mask");
    PointNearPriorityKernel<<<vertex_blocks, kThreads>>>(
        graph.dimension, device_geometry_kd.get(), device_x.get(), device_y.get(),
        device_point_priority.get(), device_point_priority_mask.get());
    CheckCuda(cudaGetLastError(), "PointNearPriorityKernel launch");
    result.resident_bytes += device_point_priority.bytes() + device_point_priority_mask.bytes();
  }
  CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
            "cudaMemset resident metric validation");
  ValidateMetricKernel<<<all_edge_blocks, kThreads>>>(
      static_cast<std::int32_t>(edge_count), device_edge_u.get(), device_edge_v.get(),
      device_edge_weight.get(), view, device_invalid.get());
  ValidateFixedDegreeKernel<<<vertex_blocks, kThreads>>>(graph.dimension, device_fixed_degree.get(),
                                                         device_invalid.get());
  CheckCuda(cudaGetLastError(), "ValidateMetricKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "resident metric validation synchronize");
  std::int32_t invalid_metric = 0;
  device_invalid.CopyToHost(&invalid_metric, 1U);
  if (invalid_metric != 0) {
    throw std::runtime_error("resident GPU 精确距离 oracle 与输入边权不一致");
  }
  std::size_t current_edges = result.initial_edges;
  std::int32_t active_edge_count = 0;
  // API 也接收含 inactive 稳定边 ID 的中间快照。初始 host CSR 包含这些
  // 槽位，而 pair offsets 只统计活动度数；在首个 pair 任务前必须先压缩。
  bool adjacency_dirty = result.initial_edges != edge_count;
  bool compact_csr_allocated = false;
  std::size_t pair_capacity = 0U;
  std::int64_t current_pair_count = 0;
  std::unique_ptr<ResidentSecModelCuda> primal_dual_model;
  std::unique_ptr<SparsePdhgCuda> primal_dual_solver;
  std::uint64_t primal_dual_workspace_bytes = 0U;
  if (options.enable_pdlp && options.enable_primal_dual_lp && !options.collect_trace) {
    primal_dual_model = std::make_unique<ResidentSecModelCuda>(device);
    primal_dual_solver = std::make_unique<SparsePdhgCuda>(device);
  }
  bool lp_snapshot_ready = false;
  bool point_prime_completed = false;
  std::uint64_t snapshot_sequence = 0U;
  result.resident_bytes += device_pending_fixed.bytes() + device_pending_degree.bytes() +
                           device_fixed_parent.bytes() + device_fixed_component_size.bytes() +
                           device_fixed_component_degree.bytes();
  std::vector<std::uint8_t> host_committed(options.collect_trace ? edge_count : 0U);
  std::vector<std::int32_t> host_first_witness(options.collect_trace ? edge_count : 0U);
  std::vector<std::int32_t> host_second_witness(options.collect_trace ? edge_count : 0U);
  std::vector<std::uint8_t> host_fixed_epoch(options.collect_trace ? edge_count : 0U);

  const auto compact_active_edges = [&] {
    const SteadyClock::time_point compact_begin = SteadyClock::now();
    CheckCuda(cub::DeviceSelect::Flagged(
                  device_active_select_temp.get(), active_select_temp_bytes,
                  device_all_edge_ids.get(), device_edge_active.get(), device_active_edge_ids.get(),
                  device_active_edge_count.get(), static_cast<std::int32_t>(edge_count)),
              "cub::DeviceSelect::Flagged active edges");
    device_active_edge_count.CopyToHost(&active_edge_count, 1U);
    if (active_edge_count < 0 || static_cast<std::size_t>(active_edge_count) != current_edges) {
      throw std::logic_error("resident GPU 活动 edge-id 压缩计数不一致");
    }
    result.compaction_ms += ElapsedMilliseconds(compact_begin);
  };
  // CUB 的 Flagged selection 保留输入次序，因此列表始终按 stable edge id 递增。
  compact_active_edges();

  const auto rebuild_compact_adjacency = [&] {
    if (!adjacency_dirty) {
      return;
    }
    const SteadyClock::time_point compact_begin = SteadyClock::now();
    if (!compact_csr_allocated) {
      if (edge_count > std::numeric_limits<std::size_t>::max() / 2U) {
        throw std::overflow_error("resident 紧凑 CSR slot 数溢出");
      }
      const std::size_t slot_capacity = edge_count * 2U;
      device_csr_counts = DeviceBuffer<std::int64_t>(dimension + 1U, device);
      device_compact_row_offsets = DeviceBuffer<std::int64_t>(dimension + 1U, device);
      device_compact_row_offsets_next = DeviceBuffer<std::int64_t>(dimension + 1U, device);
      device_compact_neighbors = DeviceBuffer<std::int32_t>(slot_capacity, device);
      device_compact_neighbors_next = DeviceBuffer<std::int32_t>(slot_capacity, device);
      device_compact_neighbor_edge_ids = DeviceBuffer<std::int32_t>(slot_capacity, device);
      device_compact_neighbor_edge_ids_next = DeviceBuffer<std::int32_t>(slot_capacity, device);
      device_compact_cursor = DeviceBuffer<unsigned long long>(dimension, device);

      std::size_t scan_temp_bytes = 0U;
      CheckCuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, device_csr_counts.get(),
                                              device_compact_row_offsets.get(),
                                              graph.dimension + 1),
                "cub::DeviceScan::ExclusiveSum size query");
      device_csr_scan_temp = DeviceBuffer<std::uint8_t>(scan_temp_bytes, device);
      result.resident_bytes +=
          device_csr_counts.bytes() + device_compact_row_offsets.bytes() +
          device_compact_row_offsets_next.bytes() + device_compact_neighbors.bytes() +
          device_compact_neighbors_next.bytes() + device_compact_neighbor_edge_ids.bytes() +
          device_compact_neighbor_edge_ids_next.bytes() + device_compact_cursor.bytes() +
          device_csr_scan_temp.bytes();
      compact_csr_allocated = true;
    }

    CopyDegreeToOffsetsKernel<<<offset_blocks, kThreads>>>(graph.dimension, device_degree.get(),
                                                           device_csr_counts.get());
    CheckCuda(cudaGetLastError(), "CopyDegreeToOffsetsKernel launch");
    std::size_t scan_temp_bytes = device_csr_scan_temp.bytes();
    CheckCuda(cub::DeviceScan::ExclusiveSum(
                  device_csr_scan_temp.get(), scan_temp_bytes, device_csr_counts.get(),
                  device_compact_row_offsets_next.get(), graph.dimension + 1),
              "cub::DeviceScan::ExclusiveSum compact CSR");

    std::int64_t compact_slots = 0;
    CheckCuda(cudaMemcpy(&compact_slots, device_compact_row_offsets_next.get() + graph.dimension,
                         sizeof(compact_slots), cudaMemcpyDeviceToHost),
              "cudaMemcpy compact CSR slot count");
    if (compact_slots < 0 || static_cast<std::uint64_t>(compact_slots) !=
                                 2ULL * static_cast<std::uint64_t>(current_edges)) {
      throw std::logic_error("resident 紧凑 CSR degree prefix sum 与活动边数不一致");
    }
    CheckCuda(cudaMemcpy(device_compact_cursor.get(), device_compact_row_offsets_next.get(),
                         dimension * sizeof(std::int64_t), cudaMemcpyDeviceToDevice),
              "cudaMemcpy compact CSR cursor");
    const int compact_blocks = (active_edge_count + kThreads - 1) / kThreads;
    ScatterCompactAdjacencyKernel<<<compact_blocks, kThreads>>>(
        active_edge_count, device_active_edge_ids.get(), device_edge_u.get(), device_edge_v.get(),
        device_compact_cursor.get(), device_compact_neighbors_next.get(),
        device_compact_neighbor_edge_ids_next.get());
    CheckCuda(cudaGetLastError(), "ScatterCompactAdjacencyKernel launch");
    SortCompactAdjacencyKernel<<<graph.dimension, 1>>>(
        graph.dimension, device_compact_row_offsets_next.get(), device_edge_weight.get(),
        device_compact_neighbors_next.get(), device_compact_neighbor_edge_ids_next.get());
    CheckCuda(cudaGetLastError(), "SortCompactAdjacencyKernel launch");
    CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
              "cudaMemset compact CSR validation");
    ValidateCompactAdjacencyKernel<<<vertex_blocks, kThreads>>>(
        graph.dimension, compact_slots, static_cast<std::int32_t>(edge_count),
        device_compact_row_offsets_next.get(), device_compact_neighbors_next.get(),
        device_compact_neighbor_edge_ids_next.get(), device_edge_u.get(), device_edge_v.get(),
        device_edge_active.get(), device_invalid.get());
    CheckCuda(cudaGetLastError(), "ValidateCompactAdjacencyKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "resident compact CSR synchronize");
    std::int32_t invalid_csr = 0;
    device_invalid.CopyToHost(&invalid_csr, 1U);
    if (invalid_csr != 0) {
      throw std::logic_error("resident 紧凑 CSR 内容校验失败");
    }

    if (view.pair_offsets != nullptr && view.nonpair_mask != nullptr) {
      BuildPairCountsKernel<<<offset_blocks, kThreads>>>(graph.dimension, device_degree.get(),
                                                         device_pair_counts.get());
      CheckCuda(cudaGetLastError(), "BuildPairCountsKernel compact launch");
      CheckCuda(cub::DeviceScan::ExclusiveSum(device_pair_scan_temp.get(), pair_scan_temp_bytes,
                                              device_pair_counts.get(),
                                              device_pair_offsets_next.get(), graph.dimension + 1),
                "cub::DeviceScan::ExclusiveSum compact pair offsets");
      std::int64_t next_pair_count = 0;
      CheckCuda(cudaMemcpy(&next_pair_count, device_pair_offsets_next.get() + graph.dimension,
                           sizeof(next_pair_count), cudaMemcpyDeviceToHost),
                "cudaMemcpy compact pair count");
      if (next_pair_count < 0 || static_cast<std::uint64_t>(next_pair_count) > pair_capacity) {
        throw std::logic_error("resident nonpair compact 容量或计数不一致");
      }
      if (next_pair_count != 0) {
        CheckCuda(cudaMemset(device_nonpair_mask_next.get(), 0,
                             static_cast<std::size_t>(next_pair_count) * sizeof(std::uint8_t)),
                  "cudaMemset compact nonpair mask");
        CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
                  "cudaMemset compact nonpair validation");
        CarryNonpairMaskKernel<<<graph.dimension, kThreads>>>(
            graph.dimension, view.row_offsets, view.neighbor_edge_ids, view.pair_offsets,
            view.nonpair_mask, device_compact_row_offsets_next.get(),
            device_compact_neighbor_edge_ids_next.get(), device_pair_offsets_next.get(),
            device_nonpair_mask_next.get(), device_invalid.get());
        CheckCuda(cudaGetLastError(), "CarryNonpairMaskKernel launch");
        CheckCuda(cudaDeviceSynchronize(), "resident nonpair compact synchronize");
        std::int32_t invalid_nonpair = 0;
        device_invalid.CopyToHost(&invalid_nonpair, 1U);
        if (invalid_nonpair != 0) {
          throw std::logic_error("resident nonpair stable edge 重映射失败");
        }
      }
      std::swap(device_pair_offsets, device_pair_offsets_next);
      std::swap(device_nonpair_mask, device_nonpair_mask_next);
      current_pair_count = next_pair_count;
    } else {
      current_pair_count = 0;
    }

    std::swap(device_compact_row_offsets, device_compact_row_offsets_next);
    std::swap(device_compact_neighbors, device_compact_neighbors_next);
    std::swap(device_compact_neighbor_edge_ids, device_compact_neighbor_edge_ids_next);
    view.row_offsets = device_compact_row_offsets.get();
    view.neighbors = device_compact_neighbors.get();
    view.neighbor_edge_ids = device_compact_neighbor_edge_ids.get();
    // non-pair 是固定点的一等单调状态；紧凑 CSR 改变 slot 后，用 stable
    // edge id 在 device 上重映射，不能因下一轮删边而遗失已证明的 pair。
    view.pair_offsets = current_pair_count == 0 ? nullptr : device_pair_offsets.get();
    view.nonpair_mask = current_pair_count == 0 ? nullptr : device_nonpair_mask.get();
    // 指针切换后释放初始 CSR，峰值内存已记入 resident_bytes。
    device_row_offsets = DeviceBuffer<std::int64_t>();
    device_neighbors = DeviceBuffer<std::int32_t>();
    device_neighbor_edge_ids = DeviceBuffer<std::int32_t>();
    adjacency_dirty = false;
    result.compaction_ms += ElapsedMilliseconds(compact_begin);
  };

  const auto clear_dirty_vertices = [&] {
    CheckCuda(cudaMemset(device_dirty_vertices.get(), 0, dimension * sizeof(std::uint32_t)),
              "cudaMemset resident dirty frontier");
  };

  const auto select_dirty_roots = [&](const std::uint32_t radius) -> std::int32_t {
    rebuild_compact_adjacency();
    for (std::uint32_t hop = 0U; hop < radius; ++hop) {
      CheckCuda(cudaMemcpy(device_dirty_vertices_next.get(), device_dirty_vertices.get(),
                           dimension * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice),
                "cudaMemcpy resident dirty frontier");
      ExpandDirtyVerticesKernel<<<vertex_blocks, kThreads>>>(view, device_dirty_vertices.get(),
                                                             device_dirty_vertices_next.get());
      CheckCuda(cudaGetLastError(), "ExpandDirtyVerticesKernel launch");
      std::swap(device_dirty_vertices, device_dirty_vertices_next);
    }
    const int root_blocks = (active_edge_count + kThreads - 1) / kThreads;
    BuildDirtyRootFlagsKernel<<<root_blocks, kThreads>>>(
        active_edge_count, device_active_edge_ids.get(), device_edge_u.get(), device_edge_v.get(),
        device_dirty_vertices.get(), device_dirty_root_flags.get());
    CheckCuda(cudaGetLastError(), "BuildDirtyRootFlagsKernel launch");
    CheckCuda(cub::DeviceSelect::Flagged(device_active_select_temp.get(), active_select_temp_bytes,
                                         device_active_edge_ids.get(),
                                         device_dirty_root_flags.get(),
                                         device_dirty_root_edge_ids.get(),
                                         device_dirty_root_edge_count.get(), active_edge_count),
              "cub::DeviceSelect::Flagged dirty roots");
    std::int32_t dirty_count = 0;
    device_dirty_root_edge_count.CopyToHost(&dirty_count, 1U);
    if (dirty_count < 0 || dirty_count > active_edge_count) {
      throw std::logic_error("resident dirty root 压缩计数非法");
    }
    return dirty_count;
  };

  const auto compute_exact_box_bound =
      [&](const std::int32_t work_count, const std::int32_t* const work_edge_ids,
          const std::int64_t* const quantized_dual, const std::int32_t cut_count,
          const std::int64_t* const quantized_cut, const std::int64_t* const reduced_cost,
          Signed128* const output) {
        const std::size_t term_count =
            dimension + static_cast<std::size_t>(cut_count) + static_cast<std::size_t>(work_count);
        const std::size_t required =
            (term_count + kBoundReductionThreads - 1U) / kBoundReductionThreads;
        const std::int32_t partial_count = static_cast<std::int32_t>(
            std::max<std::size_t>(1U, std::min(bound_partial_capacity, required)));
        ExactBoxBoundPartialsKernel<static_cast<std::int32_t>(kBoundReductionThreads)>
            <<<partial_count, static_cast<int>(kBoundReductionThreads)>>>(
                graph.dimension, work_count, work_edge_ids, device_edge_active.get(),
                quantized_dual, cut_count, quantized_cut, reduced_cost,
                device_bound_partials.get());
        CheckCuda(cudaGetLastError(), "ExactBoxBoundPartialsKernel launch");
        ExactBoxBoundFinalizeKernel<static_cast<std::int32_t>(kBoundReductionThreads)>
            <<<1, static_cast<int>(kBoundReductionThreads)>>>(partial_count,
                                                              device_bound_partials.get(), output);
        CheckCuda(cudaGetLastError(), "ExactBoxBoundFinalizeKernel launch");
      };

  const auto run_epoch = [&](const EliminationMethod method, const bool main_edge_stage = false,
                             const std::int32_t main_position = 0,
                             const bool extra_edge_stage = false,
                             const std::int32_t* const selected_edge_ids = nullptr,
                             const std::int32_t selected_edge_count = -1,
                             const bool pair_service_stage = true) -> std::size_t {
    rebuild_compact_adjacency();
    if (method == EliminationMethod::kLpBox) {
      if (options.enable_point_nonpair && !pair_service_stage)
        ++result.lp.point_deferred_sweeps;
      // P3：对当前紧凑 CSR 的全部邻边对执行 LP path-system
      // forced-one 授权。这些 bit 会被后续 Main/Quick-HS/HT reply 直接消费。
      BuildPairCountsKernel<<<offset_blocks, kThreads>>>(graph.dimension, device_degree.get(),
                                                         device_pair_counts.get());
      CheckCuda(cudaGetLastError(), "BuildPairCountsKernel launch");
      CheckCuda(cub::DeviceScan::ExclusiveSum(device_pair_scan_temp.get(), pair_scan_temp_bytes,
                                              device_pair_counts.get(), device_pair_offsets.get(),
                                              graph.dimension + 1),
                "cub::DeviceScan::ExclusiveSum pair offsets");
      CheckCuda(cudaMemcpy(&current_pair_count, device_pair_offsets.get() + graph.dimension,
                           sizeof(current_pair_count), cudaMemcpyDeviceToHost),
                "cudaMemcpy resident pair count");
      if (current_pair_count < 0 ||
          static_cast<std::uint64_t>(current_pair_count) >
              static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("resident pair mask 容量溢出");
      }
      const std::size_t pair_count = static_cast<std::size_t>(current_pair_count);
      if (pair_count > pair_capacity) {
        device_nonpair_mask = DeviceBuffer<std::uint8_t>(pair_count, device);
        device_nonpair_mask_next = DeviceBuffer<std::uint8_t>(pair_count, device);
        device_nonpair_proposed = DeviceBuffer<std::uint8_t>(pair_count, device);
        device_nonpair_verified = DeviceBuffer<std::uint8_t>(pair_count, device);
        device_nonpair_fixed_witness = DeviceBuffer<std::int32_t>(pair_count, device);
        device_nonpair_point_witness = DeviceBuffer<std::int32_t>(pair_count, device);
        pair_capacity = pair_count;
        CheckCuda(cudaMemset(device_nonpair_mask.get(), 0, pair_count * sizeof(std::uint8_t)),
                  "cudaMemset resident persistent nonpairs");
        CheckCuda(cudaMemset(device_nonpair_mask_next.get(), 0, pair_count * sizeof(std::uint8_t)),
                  "cudaMemset resident next nonpairs");
        result.resident_bytes += device_nonpair_mask.bytes() + device_nonpair_mask_next.bytes() +
                                 device_nonpair_proposed.bytes() + device_nonpair_verified.bytes() +
                                 device_nonpair_fixed_witness.bytes() +
                                 device_nonpair_point_witness.bytes();
      }
      view.pair_offsets = current_pair_count == 0 ? nullptr : device_pair_offsets.get();
      view.nonpair_mask = current_pair_count == 0 ? nullptr : device_nonpair_mask.get();
    }
    const resident_transaction::EpochSnapshot snapshot{view, snapshot_sequence};
    resident_transaction::CommitGate commit_gate{snapshot.sequence};
    const SteadyClock::time_point kernel_begin = SteadyClock::now();
    const std::int32_t work_count =
        selected_edge_ids == nullptr ? active_edge_count : selected_edge_count;
    const std::int32_t* const work_edge_ids =
        selected_edge_ids == nullptr ? device_active_edge_ids.get() : selected_edge_ids;
    const int work_blocks =
        static_cast<int>((static_cast<std::int64_t>(work_count) + kThreads - 1) / kThreads);
    if (work_count < 0 || work_count > active_edge_count ||
        (work_count != 0 && work_edge_ids == nullptr)) {
      throw std::logic_error("resident epoch 的 active-root 子集非法");
    }
    if (method == EliminationMethod::kLpBox && selected_edge_ids != nullptr) {
      throw std::logic_error("LP 下界必须覆盖全部活动变量，禁止使用 active-root 子集");
    }
    if (work_count == 0) {
      return 0U;
    }
    std::size_t main_pair_words = 0U;
    std::int32_t main_batch_size = work_count;
    if (main_edge_stage && options.main_pair_cache) {
      CheckCuda(cudaMemset(device_main_maximum_span.get(), 0, sizeof(unsigned long long)),
                "clear Main conditional cache span");
      MaximumNeighborSpanKernel<<<vertex_blocks, kThreads>>>(snapshot.graph,
                                                             device_main_maximum_span.get());
      CheckCuda(cudaGetLastError(), "MaximumNeighborSpanKernel launch");
      unsigned long long maximum_span{};
      CheckCuda(cudaMemcpy(&maximum_span, device_main_maximum_span.get(), sizeof(maximum_span),
                           cudaMemcpyDeviceToHost),
                "download Main cache allocation size");
      if (maximum_span > static_cast<unsigned long long>(graph.dimension - 1)) {
        throw std::logic_error("Main conditional cache 的 CSR 跨度非法");
      }
      const auto maximum_pairs = maximum_span < 2U ? 0U : maximum_span * (maximum_span - 1U) / 2U;
      main_pair_words = static_cast<std::size_t>((maximum_pairs + 31U) / 32U);
      const auto potential_count = static_cast<std::size_t>(options.main_edge_potentials);
      if (main_pair_words > SIZE_MAX / potential_count / sizeof(std::uint32_t)) {
        throw std::overflow_error("Main conditional cache 的容量溢出");
      }
      const std::size_t root_words = std::max<std::size_t>(main_pair_words * potential_count, 1U);
      std::size_t free_bytes{}, total_bytes{};
      CheckCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "query Main cache workspace");
      // 仅按可用工作区分批，不截断任何 root/pair/reply；同一个 epoch 完成所有批次才提交。
      const std::size_t available_words =
          std::max(main_pair_cache_capacity, free_bytes / (4U * sizeof(std::uint32_t)));
      const std::size_t roots_fit = available_words / root_words;
      if (roots_fit == 0U) {
        throw std::runtime_error("Main conditional cache 单个 root 工作区不足，未提交本 epoch");
      }
      main_batch_size = static_cast<std::int32_t>(std::min<std::size_t>(work_count, roots_fit));
      const std::size_t required_words = root_words * static_cast<std::size_t>(main_batch_size);
      if (required_words > main_pair_cache_capacity) {
        device_main_pair_cache = DeviceBuffer<std::uint32_t>(required_words, device);
        result.resident_bytes +=
            (required_words - main_pair_cache_capacity) * sizeof(std::uint32_t);
        main_pair_cache_capacity = required_words;
        result.main_pair_cache_bytes = device_main_pair_cache.bytes();
      }
    }
    if (active_edge_count <= 0) {
      throw std::logic_error("resident epoch 收到空活动边集");
    }
    // 稀疏 CSR 的 slot 稳定且按 (cost,node) 排序；删边后谓词直接
    // 跳过 inactive slot，不再为每个 epoch 重建 n×n 邻接表。
    CheckCuda(cudaMemset(device_proposed.get(), 0, edge_count * sizeof(std::uint8_t)),
              "cudaMemset resident proposed");
    CheckCuda(cudaMemset(device_committed.get(), 0, edge_count * sizeof(std::uint8_t)),
              "cudaMemset resident committed");
    CheckCuda(cudaMemset(device_first_witness.get(), 0xff, edge_count * sizeof(std::int32_t)),
              "cudaMemset resident first witness");
    CheckCuda(cudaMemset(device_second_witness.get(), 0xff, edge_count * sizeof(std::int32_t)),
              "cudaMemset resident second witness");
    if (method == EliminationMethod::kLpBox && options.enable_fixing) {
      CheckCuda(cudaMemset(device_fixed_proposed.get(), 0, edge_count * sizeof(std::uint8_t)),
                "cudaMemset resident fixed proposed");
      CheckCuda(cudaMemset(device_fixed_verified.get(), 0, edge_count * sizeof(std::uint8_t)),
                "cudaMemset resident fixed verified");
      CheckCuda(cudaMemset(device_fixed_reason.get(), 0, edge_count * sizeof(std::uint8_t)),
                "cudaMemset resident nonpair fixed reason");
      CheckCuda(cudaMemset(device_nonpair_fix_proposal_count.get(), 0, sizeof(unsigned long long)),
                "cudaMemset resident nonpair fix proposal count");
      CheckCuda(cudaMemset(device_direct_fix_proposal_count.get(), 0, sizeof(unsigned long long)),
                "cudaMemset resident direct fix proposal count");
    }
    std::vector<std::int64_t> host_quantized_dual;
    std::vector<std::int64_t> host_quantized_cut;
    const bool quick_hs_stage = extra_edge_stage || method == EliminationMethod::kGpuQuickHs;
    const std::int64_t* const lp_path_reduced_cost =
        quick_hs_stage && lp_snapshot_ready ? device_reduced_cost.get() : nullptr;
    const Signed128* const lp_path_lower_bound =
        quick_hs_stage && lp_snapshot_ready ? device_lower_bound.get() : nullptr;
    const std::int64_t lp_incumbent_numerator =
        options.enable_pdlp ? options.incumbent_cost * denominator : 0;
    const auto* pair_reduced = options.enable_pdlp ? device_reduced_cost.get() : nullptr;
    const auto* pair_bound = options.enable_pdlp ? device_lower_bound.get() : nullptr;
    // 同一不可变标志同时控制 proposal 和 replay；不能在预热完成后跳过 replay。
    const bool prime_point = !pair_service_stage && !point_prime_completed &&
                             options.point_prime_near && options.point_near_first;
    const bool run_point_service = method == EliminationMethod::kLpBox &&
                                   options.enable_point_nonpair &&
                                   (pair_service_stage || prime_point);
    const bool collect_lp_path_closures =
        (quick_hs_stage && lp_snapshot_ready) || run_point_service;
    const bool collect_point_path_end_closures = run_point_service;
    if (collect_lp_path_closures) {
      CheckCuda(cudaMemset(device_lp_path_closed_replies.get(), 0, sizeof(unsigned long long)),
                "cudaMemset resident LP path closures");
    }
    if (collect_point_path_end_closures) {
      CheckCuda(
          cudaMemset(device_point_path_end_closed_replies.get(), 0, sizeof(unsigned long long)),
          "cudaMemset resident point path-end closures");
    }
    if (extra_edge_stage) {
      if (options.extra_edge_depth == 2U) {
        QuickHsContinuationKernel<2><<<work_count, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), snapshot.graph, device_proposed.get(),
            device_first_witness.get(), device_second_witness.get(),
            static_cast<std::int32_t>(options.quick_hs_candidates),
            static_cast<std::int32_t>(options.quick_hs_pair_trials), options.quick_hs_two_hop,
            lp_path_reduced_cost, lp_path_lower_bound, lp_incumbent_numerator,
            device_lp_path_closed_replies.get());
      } else {
        QuickHsContinuationKernel<1><<<work_count, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), snapshot.graph, device_proposed.get(),
            device_first_witness.get(), device_second_witness.get(),
            static_cast<std::int32_t>(options.quick_hs_candidates),
            static_cast<std::int32_t>(options.quick_hs_pair_trials), options.quick_hs_two_hop,
            lp_path_reduced_cost, lp_path_lower_bound, lp_incumbent_numerator,
            device_lp_path_closed_replies.get());
      }
    } else if (main_edge_stage) {
      for (std::int32_t begin = 0; begin < work_count;) {
        const auto batch_count = std::min(main_batch_size, work_count - begin);
        MainEdgeContinuationKernel<false><<<batch_count, kThreads>>>(
            batch_count, work_edge_ids + begin, device_edge_u.get(), device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), snapshot.graph, device_x.get(),
            device_y.get(), device_geometry_kd.get(), 0,
            static_cast<std::int32_t>(options.main_edge_potentials), main_position,
            static_cast<std::int32_t>(options.main_edge_positions + 1U),
            options.enable_strong_metric, nullptr, device_proposed.get(),
            device_first_witness.get(), device_second_witness.get(), nullptr, nullptr,
            options.main_pair_cache ? device_main_pair_cache.get() : nullptr, main_pair_words,
            options.full_metric);
        CheckCuda(cudaGetLastError(), "Main cached proposal batch launch");
        begin += batch_count;
      }
    } else if (method == EliminationMethod::kJv) {
      JvKernel<<<work_blocks, kThreads>>>(work_count, work_edge_ids, device_edge_u.get(),
                                          device_edge_v.get(), device_edge_active.get(),
                                          device_protected.get(), snapshot.graph,
                                          device_proposed.get(), device_first_witness.get());
    } else if (method == EliminationMethod::kGpuQuickHs) {
      QuickHsContinuationKernel<0><<<work_count, kThreads>>>(
          work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_protected.get(), snapshot.graph, device_proposed.get(),
          device_first_witness.get(), device_second_witness.get(),
          static_cast<std::int32_t>(options.quick_hs_candidates),
          static_cast<std::int32_t>(options.quick_hs_pair_trials), options.quick_hs_two_hop,
          lp_path_reduced_cost, lp_path_lower_bound, lp_incumbent_numerator,
          device_lp_path_closed_replies.get());
    } else if (method == EliminationMethod::kGeometryMain) {
      NearestDistanceKernel<<<vertex_blocks, kThreads>>>(snapshot.graph, device_nearest.get());
      GeometryKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_protected.get(), snapshot.graph, device_x.get(),
          device_y.get(), device_geometry_kd.get(), 0, device_nearest.get(),
          static_cast<std::int32_t>(options.potential_candidates), device_proposed.get(),
          device_first_witness.get(), device_second_witness.get());
    } else if (method == EliminationMethod::kFixedPropagation) {
      CheckCuda(cudaMemset(device_fixed_degree.get(), 0, dimension * sizeof(std::int32_t)),
                "cudaMemset resident fixed degree");
      CountFixedDegreeKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_fixed.get(), device_fixed_degree.get());
      FixedPropagationKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_protected.get(), device_fixed_degree.get(),
          device_proposed.get());
    } else if (method == EliminationMethod::kLpBox) {
      auto lp_phase_begin = SteadyClock::now();
      if (options.enable_pdlp) {
        CheckCuda(cudaMemset(device_summary.get(), 0, 2U * sizeof(unsigned long long)),
                  "cudaMemset resident PDLP summary");
        ActiveCostSummaryKernel<<<work_blocks, kThreads>>>(
            work_count, work_edge_ids, device_edge_weight.get(), device_edge_active.get(),
            device_summary.get(), device_summary.get() + 1);
        CheckCuda(cudaDeviceSynchronize(), "resident PDLP summary synchronize");
        unsigned long long host_summary[2]{};
        device_summary.CopyToHost(host_summary, 2U);
        if (host_summary[0] == 0U) {
          throw std::logic_error("resident PDLP 活动边为空");
        }
        const double average_cost =
            static_cast<double>(host_summary[1]) / static_cast<double>(host_summary[0]);
        const double average_degree =
            2.0 * static_cast<double>(host_summary[0]) / static_cast<double>(graph.dimension);
        const double initial_step = std::max(1.0, average_cost / (average_degree + 2.0));
        const int cut_blocks = (maximum_cut_count + kThreads - 1) / kThreads;
        CheckCuda(cudaMemset(device_dual.get(), 0, dimension * sizeof(double)),
                  "cudaMemset resident degree-box dual");
        CheckCuda(cudaMemset(device_average.get(), 0, dimension * sizeof(double)),
                  "cudaMemset resident degree-box average");
        CheckCuda(cudaMemset(device_quantized_cut.get(), 0,
                             static_cast<std::size_t>(maximum_cut_count) * sizeof(std::int64_t)),
                  "cudaMemset resident degree-box cut numerator");

        // 先求纯 degree-box 基线。它本身是完整合法的 Lagrangian dual，
        // 同时为 strong solve 提供 warm start。
        for (std::uint32_t iteration = 0U; iteration < options.pdlp_iterations; ++iteration) {
          CheckCuda(cudaMemset(device_selected_degree.get(), 0, dimension * sizeof(std::int32_t)),
                    "cudaMemset resident degree-box selected degree");
          SelectDegreeBoxKernel<<<work_blocks, kThreads>>>(
              work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
              device_edge_weight.get(), device_edge_active.get(), device_dual.get(),
              device_selected_degree.get());
          const double step = initial_step / sqrt(1.0 + static_cast<double>(iteration) / 8.0);
          UpdateResidentDualKernel<<<vertex_blocks, kThreads>>>(
              graph.dimension, device_selected_degree.get(), step, dual_limit, device_dual.get(),
              device_average.get());
        }
        CheckCuda(cudaMemcpy(device_degree_dual.get(), device_dual.get(),
                             dimension * sizeof(double), cudaMemcpyDeviceToDevice),
                  "cudaMemcpy resident degree dual snapshot");
        CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
                  "cudaMemset resident degree-box invalid");
        std::int64_t* const degree_quantized = device_degree_quantized_dual.get();
        std::int64_t* const degree_reduced = device_degree_reduced_cost.get();
        Signed128* const degree_bound = device_degree_lower_bound.get();
        QuantizeResidentDualKernel<<<vertex_blocks, kThreads>>>(
            graph.dimension, 1.0 / static_cast<double>(options.pdlp_iterations),
            static_cast<double>(denominator), 1.0e15, device_average.get(), degree_quantized,
            device_invalid.get());
        ReducedCostKernel<<<work_blocks, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_weight.get(), device_edge_active.get(), denominator, degree_quantized,
            device_quantized_cut.get(), device_local_sec_incidence_count.get(),
            device_local_sec_incidence_ids.get(), degree_reduced);
        compute_exact_box_bound(work_count, work_edge_ids, degree_quantized, 0,
                                device_quantized_cut.get(), degree_reduced, degree_bound);
        // 先把 degree 快照发布为当前 best；后续 strong 候选只能由设备端
        // Signed128 严格比较替换它，因此任意有限迭代都不会导致下界倒退。
        CheckCuda(cudaMemcpy(device_quantized_dual.get(), degree_quantized,
                             dimension * sizeof(std::int64_t), cudaMemcpyDeviceToDevice),
                  "cudaMemcpy resident initial best dual");
        CheckCuda(cudaMemset(device_quantized_cut.get(), 0,
                             static_cast<std::size_t>(maximum_cut_count) * sizeof(std::int64_t)),
                  "cudaMemset resident initial best cut dual");
        CheckCuda(cudaMemcpy(device_reduced_cost.get(), degree_reduced,
                             edge_count * sizeof(std::int64_t), cudaMemcpyDeviceToDevice),
                  "cudaMemcpy resident initial best reduced cost");
        CheckCuda(cudaMemcpy(device_lower_bound.get(), degree_bound, sizeof(Signed128),
                             cudaMemcpyDeviceToDevice),
                  "cudaMemcpy resident initial best lower bound");
        CheckCuda(cudaMemset(device_selected_lp_snapshot_kind.get(), 0, sizeof(std::int32_t)),
                  "cudaMemset resident selected LP snapshot kind");

        CheckCuda(cudaDeviceSynchronize(), "resident degree solve timing");
        result.lp.solver_ms += ElapsedMilliseconds(lp_phase_begin);
        lp_phase_begin = SteadyClock::now();
        // P1 多阈值 connectivity separator：阈值只决定候选子集，不参与
        // 最终授权；每个连通分量都重新物化为合法 δ(S)>=2 行。三层 support
        // 覆盖严格负 reduced cost、零邻域和略宽正邻域，形成嵌套候选 cuts。
        CheckCuda(cudaMemset(device_cut_valid.get(), 0,
                             static_cast<std::size_t>(maximum_cut_count) * sizeof(std::uint8_t)),
                  "cudaMemset valid SEC cuts");
        if (local_sec.cut_count != 0) {
          CheckCuda(
              cudaMemset(device_cut_valid.get(), 1,
                         static_cast<std::size_t>(local_sec.cut_count) * sizeof(std::uint8_t)),
              "cudaMemset static SEC cuts valid");
        }
        CheckCuda(cudaMemset(device_connectivity_cut_count.get(), 0, sizeof(unsigned long long)),
                  "cudaMemset connectivity cut count");
        // 先恢复静态 local-window incidence，再按层追加 component cuts。
        BuildLocalSecIncidenceKernel<<<all_edge_blocks, kThreads>>>(
            static_cast<std::int32_t>(edge_count), device_all_edge_ids.get(), device_edge_u.get(),
            device_edge_v.get(), device_geometry_rank.get(), local_sec,
            device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get());
        const double scaled_support_step = average_cost * static_cast<double>(denominator) / 16.0;
        if (!std::isfinite(scaled_support_step) ||
            scaled_support_step > static_cast<double>(INT64_MAX)) {
          throw std::overflow_error("resident connectivity support 阈值溢出");
        }
        const std::int64_t support_step =
            std::max<std::int64_t>(1, static_cast<std::int64_t>(std::llround(scaled_support_step)));
        const std::int64_t support_threshold[kConnectivitySupportLevels] = {-support_step, 0,
                                                                            support_step};
        for (std::int32_t level = 0; level < kConnectivitySupportLevels; ++level) {
          InitializeSupportLabelsKernel<<<vertex_blocks, kThreads>>>(graph.dimension,
                                                                     device_support_label.get());
          std::int32_t component_changed = 1;
          std::int32_t component_round = 0;
          while (component_changed != 0) {
            if (++component_round > graph.dimension + 1) {
              throw std::logic_error("GPU support component 标签传播未收敛");
            }
            CheckCuda(cudaMemset(device_component_changed.get(), 0, sizeof(std::int32_t)),
                      "cudaMemset support component changed");
            RelaxSupportLabelsKernel<<<work_blocks, kThreads>>>(
                work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
                device_edge_active.get(), degree_reduced, support_threshold[level],
                device_support_label.get(), device_component_changed.get());
            CompressSupportLabelsKernel<<<vertex_blocks, kThreads>>>(
                graph.dimension, device_support_label.get(), device_component_changed.get());
            device_component_changed.CopyToHost(&component_changed, 1U);
          }
          CheckCuda(cudaMemset(device_component_size.get(), 0, dimension * sizeof(std::int32_t)),
                    "cudaMemset support component sizes");
          CountSupportComponentsKernel<<<vertex_blocks, kThreads>>>(
              graph.dimension, device_support_label.get(), device_component_size.get());
          CheckCuda(cudaMemcpy(device_sec_membership.get() + level * dimension,
                               device_support_label.get(), dimension * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice),
                    "封存 SEC 子集成员");
          const std::int32_t dynamic_cut_offset = local_sec.cut_count + level * graph.dimension;
          ActivateConnectivityCutsKernel<<<vertex_blocks, kThreads>>>(
              graph.dimension, dynamic_cut_offset, device_support_label.get(),
              device_component_size.get(), device_cut_valid.get(),
              device_connectivity_cut_count.get());
          AppendConnectivityCutIncidenceKernel<<<work_blocks, kThreads>>>(
              work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
              device_edge_active.get(), dynamic_cut_offset, device_support_label.get(),
              device_cut_valid.get(), device_local_sec_incidence_count.get(),
              device_local_sec_incidence_ids.get(), device_invalid.get());
        }
        unsigned long long connectivity_cuts = 0U;
        device_connectivity_cut_count.CopyToHost(&connectivity_cuts, 1U);
        result.lp_connectivity_cuts += static_cast<std::size_t>(connectivity_cuts);
        result.lp.cut_separation_ms += ElapsedMilliseconds(lp_phase_begin);
        lp_phase_begin = SteadyClock::now();

        const int selection_blocks = std::max({vertex_blocks, work_blocks, cut_blocks});
        for (std::int32_t level = 0; level < kConnectivitySupportLevels; ++level) {
          // 每个阈值独立从 degree dual 热启动，只激活同层 component cuts 与
          // 全部静态 local windows。这样短迭代不会因无关 cut 梯度互相稀释。
          CheckCuda(cudaMemcpy(device_dual.get(), device_degree_dual.get(),
                               dimension * sizeof(double), cudaMemcpyDeviceToDevice),
                    "cudaMemcpy resident strong warm start");
          CheckCuda(cudaMemset(device_average.get(), 0, dimension * sizeof(double)),
                    "cudaMemset resident strong average");
          CheckCuda(cudaMemset(device_cut_dual.get(), 0,
                               static_cast<std::size_t>(maximum_cut_count) * sizeof(double)),
                    "cudaMemset resident local SEC dual");
          CheckCuda(cudaMemset(device_cut_average.get(), 0,
                               static_cast<std::size_t>(maximum_cut_count) * sizeof(double)),
                    "cudaMemset resident local SEC average");
          CheckCuda(cudaMemset(device_cut_active.get(), 0,
                               static_cast<std::size_t>(maximum_cut_count) * sizeof(std::uint8_t)),
                    "cudaMemset resident active strong cuts");
          if (local_sec.cut_count != 0) {
            CheckCuda(
                cudaMemset(device_cut_active.get(), 1,
                           static_cast<std::size_t>(local_sec.cut_count) * sizeof(std::uint8_t)),
                "cudaMemset resident static strong cuts");
          }
          const std::int32_t dynamic_cut_offset = local_sec.cut_count + level * graph.dimension;
          CheckCuda(cudaMemcpy(device_cut_active.get() + dynamic_cut_offset,
                               device_cut_valid.get() + dynamic_cut_offset,
                               dimension * sizeof(std::uint8_t), cudaMemcpyDeviceToDevice),
                    "cudaMemcpy resident dynamic strong cuts");
          for (std::uint32_t iteration = 0U; iteration < options.pdlp_iterations; ++iteration) {
            CheckCuda(cudaMemset(device_selected_degree.get(), 0, dimension * sizeof(std::int32_t)),
                      "cudaMemset resident strong selected degree");
            CheckCuda(
                cudaMemset(device_selected_cut.get(), 0,
                           static_cast<std::size_t>(maximum_cut_count) * sizeof(std::int32_t)),
                "cudaMemset resident PDLP selected local SEC");
            SelectStrongBoxKernel<<<work_blocks, kThreads>>>(
                work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
                device_edge_weight.get(), device_edge_active.get(), device_dual.get(),
                device_cut_dual.get(), device_local_sec_incidence_count.get(),
                device_local_sec_incidence_ids.get(), device_selected_degree.get(),
                device_selected_cut.get());
            const double step = initial_step / sqrt(1.0 + static_cast<double>(iteration) / 8.0);
            UpdateResidentDualKernel<<<vertex_blocks, kThreads>>>(
                graph.dimension, device_selected_degree.get(), step, dual_limit, device_dual.get(),
                device_average.get());
            UpdateLocalSecDualKernel<<<cut_blocks, kThreads>>>(
                maximum_cut_count, device_selected_cut.get(), device_cut_active.get(), step,
                dual_limit, device_cut_dual.get(), device_cut_average.get());
          }
          QuantizeResidentDualKernel<<<vertex_blocks, kThreads>>>(
              graph.dimension, 1.0 / static_cast<double>(options.pdlp_iterations),
              static_cast<double>(denominator), 1.0e15, device_average.get(),
              device_candidate_quantized_dual.get(), device_invalid.get());
          QuantizeLocalSecDualKernel<<<cut_blocks, kThreads>>>(
              maximum_cut_count, 1.0 / static_cast<double>(options.pdlp_iterations),
              static_cast<double>(denominator), 1.0e15, device_cut_average.get(),
              device_cut_active.get(), device_candidate_quantized_cut.get(), device_invalid.get());
          ReducedCostKernel<<<work_blocks, kThreads>>>(
              work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
              device_edge_weight.get(), device_edge_active.get(), denominator,
              device_candidate_quantized_dual.get(), device_candidate_quantized_cut.get(),
              device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get(),
              device_candidate_reduced_cost.get());
          compute_exact_box_bound(work_count, work_edge_ids, device_candidate_quantized_dual.get(),
                                  maximum_cut_count, device_candidate_quantized_cut.get(),
                                  device_candidate_reduced_cost.get(),
                                  device_candidate_lower_bound.get());
          if (!options.collect_trace) {
            // decision 与 copy 分成同一 stream 上的两个 kernel，避免跨 block
            // 一边替换下界一边读取选择标志。kind=0 保留给 degree 基线。
            DecideBetterLpSnapshotKernel<<<1, 1>>>(
                device_candidate_lower_bound.get(), device_lower_bound.get(), level + 1, false,
                device_selected_lp_snapshot_kind.get(), device_apply_lp_snapshot.get());
            ApplyBetterLpSnapshotKernel<<<selection_blocks, kThreads>>>(
                graph.dimension, work_count, work_edge_ids, maximum_cut_count,
                device_candidate_quantized_dual.get(), device_candidate_quantized_cut.get(),
                device_candidate_reduced_cost.get(), device_quantized_dual.get(),
                device_quantized_cut.get(), device_reduced_cost.get(),
                device_apply_lp_snapshot.get());
          }
        }

        if (primal_dual_solver != nullptr) {
          CheckCuda(cudaDeviceSynchronize(), "resident SEC ensemble timing");
          result.lp.solver_ms += ElapsedMilliseconds(lp_phase_begin);
          const auto model_begin = SteadyClock::now();
          const SparsePdhgDeviceModel model = primal_dual_model->Build(
              snapshot.graph, device_edge_weight.get(), work_count, work_edge_ids,
              maximum_cut_count, local_sec.cut_count, device_cut_valid.get(),
              device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get(),
              kMaxLocalSecIncidence, snapshot.sequence + 1U);
          CheckCuda(cudaDeviceSynchronize(), "resident sparse LP model synchronize");
          result.lp.pdhg_model_ms += ElapsedMilliseconds(model_begin);
          lp_phase_begin = SteadyClock::now();
          const SparsePdhgDiagnostics diagnostics = primal_dual_solver->Iterate(
              model, std::max(1.0, average_cost), options.pdlp_iterations);
          result.lp.pdhg_iterations += diagnostics.iterations;
          result.lp.pdhg_ms += diagnostics.solve_ms;
          result.lp.pdhg_primal_violation = diagnostics.primal_violation;
          result.lp.pdhg_relative_gap = diagnostics.relative_gap;
          const int dual_blocks = static_cast<int>(
              (static_cast<std::int64_t>(graph.dimension) + maximum_cut_count + kThreads - 1) /
              kThreads);
          QuantizeSparsePdhgDualKernel<<<dual_blocks, kThreads>>>(
              graph.dimension, maximum_cut_count, primal_dual_solver->dual(),
              std::max(1.0, average_cost), denominator, device_cut_valid.get(),
              device_candidate_quantized_dual.get(), device_candidate_quantized_cut.get(),
              device_invalid.get());
          ReducedCostKernel<<<work_blocks, kThreads>>>(
              work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
              device_edge_weight.get(), device_edge_active.get(), denominator,
              device_candidate_quantized_dual.get(), device_candidate_quantized_cut.get(),
              device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get(),
              device_candidate_reduced_cost.get());
          compute_exact_box_bound(work_count, work_edge_ids, device_candidate_quantized_dual.get(),
                                  maximum_cut_count, device_candidate_quantized_cut.get(),
                                  device_candidate_reduced_cost.get(),
                                  device_candidate_lower_bound.get());
          DecideBetterLpSnapshotKernel<<<1, 1>>>(
              device_candidate_lower_bound.get(), device_lower_bound.get(), 4, false,
              device_selected_lp_snapshot_kind.get(), device_apply_lp_snapshot.get());
          ApplyBetterLpSnapshotKernel<<<selection_blocks, kThreads>>>(
              graph.dimension, work_count, work_edge_ids, maximum_cut_count,
              device_candidate_quantized_dual.get(), device_candidate_quantized_cut.get(),
              device_candidate_reduced_cost.get(), device_quantized_dual.get(),
              device_quantized_cut.get(), device_reduced_cost.get(),
              device_apply_lp_snapshot.get());
          const std::uint64_t workspace =
              primal_dual_model->workspace_bytes() + primal_dual_solver->workspace_bytes();
          if (workspace > primal_dual_workspace_bytes) {
            result.resident_bytes += workspace - primal_dual_workspace_bytes;
            primal_dual_workspace_bytes = workspace;
          }
        }
        LpForcedOneKernel<<<work_blocks, kThreads>>>(
            work_count, device_edge_u.get(), work_edge_ids, device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), device_degree.get(),
            device_reduced_cost.get(), device_lower_bound.get(),
            options.incumbent_cost * denominator, device_proposed.get());
        CheckCuda(cudaDeviceSynchronize(), "resident LP solve timing");
        result.lp.solver_ms += ElapsedMilliseconds(lp_phase_begin);
        lp_phase_begin = SteadyClock::now();
      } // LP 关闭时仍执行后面的完整 pair/fixing 服务，不构造伪 LP 下界。
      if (options.enable_fixing) {
        if (options.enable_pdlp) {
          LpForcedZeroKernel<<<work_blocks, kThreads>>>(
              work_count, work_edge_ids, device_edge_active.get(), device_protected.get(),
              device_reduced_cost.get(), device_lower_bound.get(),
              options.incumbent_cost * denominator, device_fixed_proposed.get());
        }
        if (snapshot.graph.pair_offsets != nullptr && snapshot.graph.nonpair_mask != nullptr) {
          NonpairImpliedFixKernel<<<work_blocks, kThreads>>>(
              work_count, work_edge_ids, snapshot.graph, device_protected.get(),
              device_fixed_proposed.get(), device_fixed_reason.get(),
              device_nonpair_fix_proposal_count.get());
          CheckCuda(cudaGetLastError(), "NonpairImpliedFixKernel launch");
          if (options.enable_direct_fix && pair_service_stage) {
            DirectFixKernel<<<work_count, kThreads>>>(
                work_count, work_edge_ids, snapshot.graph, device_protected.get(), pair_reduced,
                pair_bound, lp_incumbent_numerator, device_fixed_proposed.get(),
                device_fixed_reason.get(), device_lp_path_closed_replies.get(),
                device_direct_fix_proposal_count.get());
            CheckCuda(cudaGetLastError(), "DirectFixKernel launch");
          }
        }
      }

      CheckCuda(cudaDeviceSynchronize(), "resident LP fixing timing");
      result.lp.fixing_ms += ElapsedMilliseconds(lp_phase_begin);
      lp_phase_begin = SteadyClock::now();
      const std::size_t pair_count = static_cast<std::size_t>(current_pair_count);
      if (pair_count != 0U) {
        CheckCuda(cudaMemset(device_nonpair_proposed.get(), 0, pair_count * sizeof(std::uint8_t)),
                  "cudaMemset resident proposed nonpairs");
        CheckCuda(cudaMemset(device_nonpair_verified.get(), 0, pair_count * sizeof(std::uint8_t)),
                  "cudaMemset resident verified nonpairs");
        CheckCuda(
            cudaMemset(device_nonpair_fixed_witness.get(), 0xff, pair_count * sizeof(std::int32_t)),
            "cudaMemset resident nonpair fixed witness");
        CheckCuda(
            cudaMemset(device_nonpair_point_witness.get(), 0xff, pair_count * sizeof(std::int32_t)),
            "cudaMemset resident nonpair point witness");
        CheckCuda(
            cudaMemset(device_nonpair_proposal_counts.get(), 0, 3U * sizeof(unsigned long long)),
            "cudaMemset resident nonpair proposal counts");
        if (options.enable_pdlp) {
          LpNonpairKernel<<<graph.dimension, kThreads>>>(
              snapshot.graph, device_reduced_cost.get(), device_lower_bound.get(),
              options.incumbent_cost * denominator, device_pair_offsets.get(),
              device_nonpair_mask.get(), device_nonpair_proposed.get(),
              device_nonpair_proposal_counts.get());
          CheckCuda(cudaGetLastError(), "LpNonpairKernel launch");
        }
        CheckCuda(cub::DeviceSelect::Flagged(
                      device_active_select_temp.get(), active_select_temp_bytes,
                      device_all_edge_ids.get(), device_fixed.get(), device_fixed_edge_ids.get(),
                      device_fixed_edge_count.get(), static_cast<std::int32_t>(edge_count)),
                  "cub::DeviceSelect::Flagged fixed edges");
        FixedAnchorNonpairKernel<<<graph.dimension, kThreads>>>(
            snapshot.graph, device_edge_u.get(), device_edge_v.get(), device_edge_active.get(),
            device_fixed.get(), device_fixed_edge_ids.get(), device_fixed_edge_count.get(),
            device_pair_offsets.get(), device_nonpair_mask.get(), device_nonpair_proposed.get(),
            device_nonpair_fixed_witness.get(), device_nonpair_proposal_counts.get() + 1);
        CheckCuda(cudaGetLastError(), "FixedAnchorNonpairKernel launch");
        CheckCuda(cudaDeviceSynchronize(), "resident LP pair filter timing");
        result.lp.pair_filter_ms += ElapsedMilliseconds(lp_phase_begin);
        lp_phase_begin = SteadyClock::now();
        if (run_point_service) {
          if (prime_point)
            ++result.lp.point_prime_sweeps;
          else
            ++result.lp.point_service_sweeps;
          const std::int64_t point_blocks = std::min<std::int64_t>(current_pair_count, 65535);
          // 两种寄存器预算编译为独立 kernel，搜索域/回复顺序完全相同。
          const auto launch_point = [&]<int MinBlocks>() {
            PointNonpairKernel<MinBlocks><<<static_cast<unsigned int>(point_blocks), kThreads>>>(
                current_pair_count, snapshot.graph, device_pair_offsets.get(),
                device_nonpair_mask.get(), device_nonpair_proposed.get(),
                device_nonpair_point_witness.get(), pair_reduced, pair_bound,
                lp_incumbent_numerator, device_lp_path_closed_replies.get(),
                device_point_path_end_closed_replies.get(),
                device_nonpair_proposal_counts.get() + 2,
                options.point_near_first ? device_point_priority.get() : nullptr,
                options.point_near_first ? device_point_priority_mask.get() : nullptr, prime_point);
          };
          if (options.point_cta_blocks == 4U) {
            launch_point.template operator()<4>();
          } else {
            launch_point.template operator()<2>();
          }
          CheckCuda(cudaGetLastError(), "PointNonpairKernel launch");
          CheckCuda(cudaDeviceSynchronize(), "resident LP point timing");
          const auto point_ms = ElapsedMilliseconds(lp_phase_begin);
          result.lp.point_ms += point_ms;
          if (prime_point) {
            result.lp.point_prime_ms += point_ms;
            point_prime_completed = true;
          }
        }
      }
      CheckCuda(cudaDeviceSynchronize(), "resident PDLP proof synchronize");
      std::int32_t invalid = 0;
      if (options.enable_pdlp) {
        device_invalid.CopyToHost(&invalid, 1U);
        if (invalid != 0) {
          throw std::runtime_error("resident PDLP dual 超出量化安全范围");
        }
        // selected dual、reduced costs 与 gpu-safe replay 副本至此已经完整
        // 验证；后续 local epoch 可把它作为不可变 path-system 下界消费。
        lp_snapshot_ready = true;
      }
      if (pair_count != 0U) {
        unsigned long long proposal_counts[3]{};
        device_nonpair_proposal_counts.CopyToHost(proposal_counts, 3U);
        result.lp_nonpair_committed += static_cast<std::size_t>(proposal_counts[0]);
        result.fixed_anchor_nonpair_committed += static_cast<std::size_t>(proposal_counts[1]);
        result.point_nonpair_committed += static_cast<std::size_t>(proposal_counts[2]);
        if (prime_point)
          result.lp.point_prime_proposals += proposal_counts[2];
      }
    } else {
      throw std::logic_error("resident run_epoch 收到不支持的方法");
    }
    CheckCuda(cudaGetLastError(), "resident elimination kernel launch");
    CheckCuda(cudaDeviceSynchronize(), "resident proposal synchronize");
    if (collect_lp_path_closures) {
      unsigned long long lp_path_closed_replies = 0U;
      device_lp_path_closed_replies.CopyToHost(&lp_path_closed_replies, 1U);
      result.lp_path_closed_replies += static_cast<std::size_t>(lp_path_closed_replies);
    }
    if (collect_point_path_end_closures) {
      unsigned long long point_path_end_closed_replies = 0U;
      device_point_path_end_closed_replies.CopyToHost(&point_path_end_closed_replies, 1U);
      result.point_path_end_closed_replies +=
          static_cast<std::size_t>(point_path_end_closed_replies);
    }
    // proposal、replay、commit 分别计时，避免异步 kernel 的耗时被后续
    // commit 同步重复吸收，导致阶段画像和总耗时无法比较。
    const double device_ms = ElapsedMilliseconds(kernel_begin);
    if (method == EliminationMethod::kLpBox && options.enable_pdlp) {
      Signed128 host_lower_bound{};
      device_lower_bound.CopyToHost(&host_lower_bound, 1U);
      // 这里只生成人类可读 telemetry；证明比较始终使用 device Signed128。
      const double numerator = std::ldexp(static_cast<double>(host_lower_bound.high), 64) +
                               static_cast<double>(host_lower_bound.low);
      result.lp_lower_bound = numerator / static_cast<double>(denominator);
      std::int32_t selected_snapshot_kind = 0;
      device_selected_lp_snapshot_kind.CopyToHost(&selected_snapshot_kind, 1U);
      if (selected_snapshot_kind == 4) {
        ++result.lp.pdhg_selected_snapshots;
      }
      if (selected_snapshot_kind == 0) {
        ++result.lp_degree_snapshots;
      } else {
        ++result.lp_strong_snapshots;
      }
    }

    const Signed128* replay_lower_bound = options.enable_pdlp ? device_lower_bound.get() : nullptr;
    if (method == EliminationMethod::kLpBox && options.gpu_replay && options.enable_pdlp) {
      const SteadyClock::time_point replay_begin = SteadyClock::now();
      CheckCuda(cudaMemset(device_sec_replay_sizes.get(), 0, device_sec_replay_sizes.bytes()),
                "清零 SEC replay 子集计数");
      CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)), "清零 SEC replay 状态");
      const int membership_blocks =
          static_cast<int>((dimension * kConnectivitySupportLevels + kThreads - 1) / kThreads);
      const int sec_blocks = (maximum_cut_count + kThreads - 1) / kThreads;
      resident_sec_replay::CountMembersKernel<<<membership_blocks, kThreads>>>(
          graph.dimension, kConnectivitySupportLevels, device_sec_membership.get(),
          device_sec_replay_sizes.get(), device_invalid.get());
      resident_sec_replay::ValidateCutsKernel<<<sec_blocks, kThreads>>>(
          graph.dimension, local_sec.cut_count, maximum_cut_count, device_sec_membership.get(),
          device_sec_replay_sizes.get(), device_cut_valid.get(), device_quantized_cut.get(),
          device_invalid.get());
      resident_sec_replay::ValidateIncidenceKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
          device_geometry_rank.get(), local_sec, kLocalSecFamilies, kConnectivitySupportLevels,
          kMaxLocalSecIncidence, device_sec_membership.get(), device_cut_valid.get(),
          device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get(),
          device_invalid.get());
      std::int32_t invalid_sec = 0;
      device_invalid.CopyToHost(&invalid_sec, 1U);
      if (invalid_sec != 0) {
        throw std::runtime_error("GPU SEC replay 发现非法 cut、dual 或 incidence");
      }
      // replay 重新生成所有 reduced costs 和全局 Signed128 下界；后续
      // edge/fix/non-pair replay 不复用 proposal 的聚合结果。
      ReducedCostKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
          device_edge_weight.get(), device_edge_active.get(), denominator,
          device_quantized_dual.get(), device_quantized_cut.get(),
          device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get(),
          device_replay_reduced_cost.get());
      compute_exact_box_bound(work_count, work_edge_ids, device_quantized_dual.get(),
                              maximum_cut_count, device_quantized_cut.get(),
                              device_replay_reduced_cost.get(), device_replay_lower_bound.get());
      CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
                "cudaMemset resident LP replay bound validation");
      ValidateLpReplayBoundKernel<<<1, 1>>>(device_lower_bound.get(),
                                            device_replay_lower_bound.get(), device_invalid.get());
      CheckCuda(cudaGetLastError(), "ValidateLpReplayBoundKernel launch");
      CheckCuda(cudaDeviceSynchronize(), "resident LP replay bound synchronize");
      std::int32_t invalid_replay_bound = 0;
      device_invalid.CopyToHost(&invalid_replay_bound, 1U);
      if (invalid_replay_bound != 0) {
        throw std::runtime_error("GPU LP replay 重构的精确下界不一致");
      }
      replay_lower_bound = device_replay_lower_bound.get();
      result.proof_replay_ms += ElapsedMilliseconds(replay_begin);
    }

    const std::uint8_t* authorized_fixed = nullptr;
    if (method == EliminationMethod::kLpBox && options.enable_fixing) {
      authorized_fixed = device_fixed_proposed.get();
      if (options.gpu_replay) {
        const SteadyClock::time_point replay_begin = SteadyClock::now();
        CheckCuda(cudaMemset(device_replay_counters.get(), 0, 2U * sizeof(unsigned long long)),
                  "cudaMemset resident fixed replay counters");
        ReplayLpFixedKernel<<<work_blocks, kThreads>>>(
            work_count, work_edge_ids, snapshot.graph, device_edge_u.get(), device_edge_v.get(),
            device_edge_weight.get(), device_edge_active.get(), device_protected.get(), denominator,
            device_quantized_dual.get(), device_quantized_cut.get(),
            device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get(),
            replay_lower_bound, options.incumbent_cost * denominator, device_fixed_proposed.get(),
            device_fixed_reason.get(), device_fixed_verified.get(), device_replay_counters.get(),
            device_replay_counters.get() + 1);
        CheckCuda(cudaGetLastError(), "resident LP fixed replay launch");
        if (options.enable_direct_fix && pair_service_stage) {
          ReplayDirectFixKernel<<<work_count, kThreads>>>(
              work_count, work_edge_ids, snapshot.graph, device_protected.get(),
              device_fixed_proposed.get(), device_fixed_reason.get(),
              options.enable_pdlp ? device_replay_reduced_cost.get() : nullptr, replay_lower_bound,
              lp_incumbent_numerator, device_fixed_verified.get(), device_replay_counters.get(),
              device_replay_counters.get() + 1);
          CheckCuda(cudaGetLastError(), "resident direct fixed replay launch");
        }
        CheckCuda(cudaDeviceSynchronize(), "resident LP fixed replay synchronize");
        unsigned long long replay_counters[2]{};
        device_replay_counters.CopyToHost(replay_counters, 2U);
        result.proof_replayed += static_cast<std::size_t>(replay_counters[0]);
        result.proof_rejected += static_cast<std::size_t>(replay_counters[1]);
        result.proof_replay_ms += ElapsedMilliseconds(replay_begin);
        if (replay_counters[1] != 0U) {
          throw std::runtime_error("GPU LP fixed replay 拒绝了候选");
        }
        authorized_fixed = device_fixed_verified.get();
      }
    }

    const std::uint8_t* authorized_nonpair = nullptr;
    if (method == EliminationMethod::kLpBox) {
      authorized_nonpair = device_nonpair_proposed.get();
      if (current_pair_count != 0 && options.gpu_replay) {
        const SteadyClock::time_point replay_begin = SteadyClock::now();
        CheckCuda(cudaMemset(device_replay_counters.get(), 0, 2U * sizeof(unsigned long long)),
                  "cudaMemset resident nonpair replay counters");
        ReplayLpNonpairKernel<<<graph.dimension, kThreads>>>(
            snapshot.graph, device_edge_u.get(), device_edge_v.get(), device_edge_weight.get(),
            denominator, device_quantized_dual.get(), device_quantized_cut.get(),
            device_local_sec_incidence_count.get(), device_local_sec_incidence_ids.get(),
            replay_lower_bound, options.incumbent_cost * denominator, device_pair_offsets.get(),
            device_nonpair_proposed.get(), device_nonpair_fixed_witness.get(),
            device_nonpair_point_witness.get(), device_fixed.get(), device_nonpair_verified.get(),
            device_replay_counters.get(), device_replay_counters.get() + 1);
        CheckCuda(cudaGetLastError(), "ReplayLpNonpairKernel launch");
        if (run_point_service) {
          const std::int64_t point_blocks = std::min<std::int64_t>(current_pair_count, 65535);
          ReplayPointNonpairKernel<<<static_cast<unsigned int>(point_blocks), kThreads>>>(
              current_pair_count, snapshot.graph, device_pair_offsets.get(),
              device_nonpair_proposed.get(), device_nonpair_point_witness.get(),
              options.enable_pdlp ? device_replay_reduced_cost.get() : nullptr, replay_lower_bound,
              lp_incumbent_numerator, device_nonpair_verified.get(), device_replay_counters.get(),
              device_replay_counters.get() + 1);
          CheckCuda(cudaGetLastError(), "ReplayPointNonpairKernel launch");
        }
        CheckCuda(cudaDeviceSynchronize(), "resident LP nonpair replay synchronize");
        unsigned long long replay_counters[2]{};
        device_replay_counters.CopyToHost(replay_counters, 2U);
        result.proof_replayed += static_cast<std::size_t>(replay_counters[0]);
        result.proof_rejected += static_cast<std::size_t>(replay_counters[1]);
        result.proof_replay_ms += ElapsedMilliseconds(replay_begin);
        if (replay_counters[1] != 0U) {
          throw std::runtime_error("GPU LP nonpair replay 拒绝了 " +
                                   std::to_string(replay_counters[1]) + " 个候选");
        }
        authorized_nonpair = device_nonpair_verified.get();
      }
    }

    const std::uint8_t* authorized = device_proposed.get();
    const std::int64_t* const lp_replay_reduced_cost =
        lp_snapshot_ready ? device_replay_reduced_cost.get() : nullptr;
    const Signed128* const lp_replay_bound =
        lp_snapshot_ready ? device_replay_lower_bound.get() : nullptr;
    if (options.gpu_replay) {
      const SteadyClock::time_point replay_begin = SteadyClock::now();
      CheckCuda(cudaMemset(device_verified.get(), 0, edge_count * sizeof(std::uint8_t)),
                "cudaMemset resident verified");
      CheckCuda(cudaMemset(device_replay_counters.get(), 0, 2U * sizeof(unsigned long long)),
                "cudaMemset resident replay counters");
      if (extra_edge_stage) {
        if (options.extra_edge_depth == 2U) {
          ReplayQuickHsContinuationKernel<2><<<work_count, kThreads>>>(
              work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
              device_edge_active.get(), device_protected.get(), snapshot.graph,
              device_proposed.get(), device_first_witness.get(), device_second_witness.get(),
              device_verified.get(), device_replay_counters.get(), device_replay_counters.get() + 1,
              lp_replay_reduced_cost, lp_replay_bound, lp_incumbent_numerator);
        } else {
          ReplayQuickHsContinuationKernel<1><<<work_count, kThreads>>>(
              work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
              device_edge_active.get(), device_protected.get(), snapshot.graph,
              device_proposed.get(), device_first_witness.get(), device_second_witness.get(),
              device_verified.get(), device_replay_counters.get(), device_replay_counters.get() + 1,
              lp_replay_reduced_cost, lp_replay_bound, lp_incumbent_numerator);
        }
      } else if (main_edge_stage) {
        for (std::int32_t begin = 0; begin < work_count;) {
          const auto batch_count = std::min(main_batch_size, work_count - begin);
          MainEdgeContinuationKernel<true><<<batch_count, kThreads>>>(
              batch_count, work_edge_ids + begin, device_edge_u.get(), device_edge_v.get(),
              device_edge_active.get(), device_protected.get(), snapshot.graph, device_x.get(),
              device_y.get(), device_geometry_kd.get(), 0,
              static_cast<std::int32_t>(options.main_edge_potentials), main_position,
              static_cast<std::int32_t>(options.main_edge_positions + 1U),
              options.enable_strong_metric, device_proposed.get(), device_verified.get(),
              device_first_witness.get(), device_second_witness.get(), device_replay_counters.get(),
              device_replay_counters.get() + 1,
              options.main_pair_cache ? device_main_pair_cache.get() : nullptr, main_pair_words,
              options.full_metric);
          CheckCuda(cudaGetLastError(), "Main cached replay batch launch");
          begin += batch_count;
        }
      } else if (method == EliminationMethod::kJv) {
        ReplayJvKernel<<<work_blocks, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), snapshot.graph, device_proposed.get(),
            device_first_witness.get(), device_verified.get(), device_replay_counters.get(),
            device_replay_counters.get() + 1);
      } else if (method == EliminationMethod::kGpuQuickHs) {
        ReplayQuickHsContinuationKernel<0><<<work_count, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), snapshot.graph, device_proposed.get(),
            device_first_witness.get(), device_second_witness.get(), device_verified.get(),
            device_replay_counters.get(), device_replay_counters.get() + 1, lp_replay_reduced_cost,
            lp_replay_bound, lp_incumbent_numerator);
      } else if (method == EliminationMethod::kGeometryMain) {
        ReplayGeometryKernel<<<work_blocks, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), snapshot.graph, device_x.get(),
            device_y.get(), device_nearest.get(), device_proposed.get(), device_first_witness.get(),
            device_second_witness.get(), device_verified.get(), device_replay_counters.get(),
            device_replay_counters.get() + 1);
      } else if (method == EliminationMethod::kFixedPropagation) {
        ReplayFixedPropagationKernel<<<work_blocks, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_active.get(), device_protected.get(), device_fixed_degree.get(),
            device_proposed.get(), device_verified.get(), device_replay_counters.get(),
            device_replay_counters.get() + 1);
      } else if (method == EliminationMethod::kLpBox && options.enable_pdlp) {
        ReplayLpKernel<<<work_blocks, kThreads>>>(
            work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
            device_edge_weight.get(), device_edge_active.get(), device_protected.get(),
            device_degree.get(), denominator, device_quantized_dual.get(),
            device_quantized_cut.get(), device_local_sec_incidence_count.get(),
            device_local_sec_incidence_ids.get(), replay_lower_bound,
            options.incumbent_cost * denominator, device_proposed.get(), device_verified.get(),
            device_replay_counters.get(), device_replay_counters.get() + 1);
      }
      CheckCuda(cudaGetLastError(), "resident GPU proof replay launch");
      CheckCuda(cudaDeviceSynchronize(), "resident GPU proof replay synchronize");
      unsigned long long replay_counters[2]{};
      device_replay_counters.CopyToHost(replay_counters, 2U);
      result.proof_replayed += static_cast<std::size_t>(replay_counters[0]);
      result.proof_rejected += static_cast<std::size_t>(replay_counters[1]);
      result.proof_replay_ms += ElapsedMilliseconds(replay_begin);
      if (replay_counters[1] != 0U) {
        throw std::runtime_error("GPU proof replay 拒绝了 " + std::to_string(replay_counters[1]) +
                                 " 个候选");
      }
      authorized = device_verified.get();
    }

    if (!commit_gate.FinishReplay(snapshot.sequence)) {
      throw std::logic_error("resident replay 跨快照或重复完成");
    }
    const SteadyClock::time_point commit_begin = SteadyClock::now();
    CheckCuda(cudaMemset(device_committed_count.get(), 0, sizeof(unsigned long long)),
              "cudaMemset resident committed count");
    if (has_protected_degree_floor) {
      SelectProtectedFloorCommitKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, device_edge_active.get(), device_protected.get(), authorized,
          device_committed.get());
      CheckCuda(cudaGetLastError(), "resident SelectProtectedFloorCommitKernel launch");
    } else {
      SelectDegreeFloorCommitKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, snapshot.graph, device_protected.get(), authorized,
          device_committed.get());
      CheckCuda(cudaGetLastError(), "resident SelectDegreeFloorCommitKernel launch");
    }
    // 所有 replay 已结束。这里仅构造和验证候选终态，live mask 仍未修改。
    const resident_transaction::PendingDelta delta{device_committed.get(), authorized_fixed,
                                                   authorized_nonpair};
    const int pending_blocks = (active_edge_count + kThreads - 1) / kThreads;
    CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
              "cudaMemset pending state validation");
    CheckCuda(cudaMemset(device_fixed_component_size.get(), 0, dimension * sizeof(std::int32_t)),
              "cudaMemset pending fixed component sizes");
    CheckCuda(cudaMemset(device_fixed_component_degree.get(), 0, dimension * sizeof(std::int32_t)),
              "cudaMemset pending fixed component degrees");
    resident_transaction::PendingDegreeKernel<<<vertex_blocks, kThreads>>>(
        snapshot.graph, delta, device_pending_degree.get());
    resident_transaction::PendingFixedKernel<<<pending_blocks, kThreads>>>(
        active_edge_count, device_active_edge_ids.get(), snapshot.graph, delta,
        device_pending_degree.get(), device_pending_fixed.get());
    resident_transaction::ValidateVerticesKernel<<<vertex_blocks, kThreads>>>(
        snapshot.graph, delta, device_pending_fixed.get(), device_invalid.get(),
        device_fixed_parent.get());
    resident_transaction::UnionFixedKernel<<<pending_blocks, kThreads>>>(
        active_edge_count, device_active_edge_ids.get(), snapshot.graph, device_pending_fixed.get(),
        device_fixed_parent.get());
    resident_transaction::CountFixedComponentsKernel<<<vertex_blocks, kThreads>>>(
        snapshot.graph, device_pending_fixed.get(), device_fixed_parent.get(),
        device_fixed_component_size.get(), device_fixed_component_degree.get());
    resident_transaction::ValidateFixedComponentsKernel<<<vertex_blocks, kThreads>>>(
        graph.dimension, device_fixed_component_size.get(), device_fixed_component_degree.get(),
        device_invalid.get());
    CheckCuda(cudaGetLastError(), "resident pending state validation launch");
    std::int32_t invalid_pending = 0;
    device_invalid.CopyToHost(&invalid_pending, 1U);
    if (!commit_gate.Validate(invalid_pending)) {
      throw std::runtime_error("resident 事务验证失败，未修改 live 状态，code=" +
                               std::to_string(invalid_pending));
    }
    if (!commit_gate.Publish(snapshot_sequence)) {
      throw std::logic_error("resident 未验证事务或 snapshot sequence 失配");
    }
    // 同一 stream 上顺序发布；新 epoch 只能在全部发布 kernel 完成后读取。
    if (authorized_fixed != nullptr) {
      CheckCuda(cudaMemset(device_fixed_count.get(), 0, sizeof(unsigned long long)),
                "cudaMemset resident fixed count epoch");
      CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
                "cudaMemset resident fixed conflict");
      ApplyFixedKernel<<<work_blocks, kThreads>>>(work_count, work_edge_ids, device_proposed.get(),
                                                  authorized_fixed, device_protected.get(),
                                                  device_fixed.get(), device_fixed_count.get(),
                                                  device_invalid.get());
      CheckCuda(cudaGetLastError(), "resident ApplyFixedKernel launch");
      CheckCuda(cudaMemset(device_fixed_degree.get(), 0, dimension * sizeof(std::int32_t)),
                "cudaMemset resident post-fix degree");
      CountFixedDegreeKernel<<<work_blocks, kThreads>>>(
          work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_fixed.get(), device_fixed_degree.get());
      ValidateFixedDegreeKernel<<<vertex_blocks, kThreads>>>(
          graph.dimension, device_fixed_degree.get(), device_invalid.get());
      CheckCuda(cudaGetLastError(), "ValidateFixedDegreeKernel launch");
      CheckCuda(cudaDeviceSynchronize(), "resident ApplyFixedKernel synchronize");
      std::int32_t fixed_conflict = 0;
      unsigned long long fixed_count = 0U;
      unsigned long long nonpair_fix_count = 0U;
      unsigned long long direct_fix_count = 0U;
      device_invalid.CopyToHost(&fixed_conflict, 1U);
      device_fixed_count.CopyToHost(&fixed_count, 1U);
      device_nonpair_fix_proposal_count.CopyToHost(&nonpair_fix_count, 1U);
      device_direct_fix_proposal_count.CopyToHost(&direct_fix_count, 1U);
      if (fixed_conflict != 0) {
        throw std::runtime_error("LP fix 与 delete 冲突，或固定度数超过 2");
      }
      result.fixed_count += static_cast<std::size_t>(fixed_count);
      result.nonpair_fix_committed += static_cast<std::size_t>(nonpair_fix_count);
      result.direct_fix_committed += static_cast<std::size_t>(direct_fix_count);
    }
    if (authorized_nonpair != nullptr) {
      if (current_pair_count != 0) {
        const int pair_blocks = static_cast<int>((current_pair_count + kThreads - 1) / kThreads);
        CheckCuda(cudaMemset(device_nonpair_committed_count.get(), 0, sizeof(unsigned long long)),
                  "cudaMemset resident committed nonpair count");
        CommitNonpairMaskKernel<<<pair_blocks, kThreads>>>(current_pair_count, authorized_nonpair,
                                                           device_nonpair_mask.get(),
                                                           device_nonpair_committed_count.get());
        CheckCuda(cudaGetLastError(), "CommitNonpairMaskKernel launch");
        unsigned long long committed_nonpairs = 0U;
        device_nonpair_committed_count.CopyToHost(&committed_nonpairs, 1U);
        result.nonpair_committed += static_cast<std::size_t>(committed_nonpairs);
      }
    }
    ApplyCommitKernel<<<work_blocks, kThreads>>>(
        work_count, work_edge_ids, device_edge_u.get(), device_edge_v.get(),
        device_edge_active.get(), device_committed.get(), device_degree.get(),
        device_committed_count.get(), device_dirty_vertices.get());
    CheckCuda(cudaGetLastError(), "resident ApplyCommitKernel launch");
    const int active_blocks = (active_edge_count + kThreads - 1) / kThreads;
    MarkDegreeTwoFixedKernel<<<active_blocks, kThreads>>>(
        active_edge_count, device_active_edge_ids.get(), device_edge_u.get(), device_edge_v.get(),
        device_edge_active.get(), device_degree.get(), device_protected.get(), device_fixed.get());
    CheckCuda(cudaGetLastError(), "MarkDegreeTwoFixedKernel commit launch");
    CheckCuda(cudaMemset(device_fixed_degree.get(), 0, dimension * sizeof(std::int32_t)),
              "cudaMemset resident committed fixed degree");
    CountFixedDegreeKernel<<<active_blocks, kThreads>>>(
        active_edge_count, device_active_edge_ids.get(), device_edge_u.get(), device_edge_v.get(),
        device_edge_active.get(), device_fixed.get(), device_fixed_degree.get());
    CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
              "cudaMemset resident degree invalid");
    ValidateDegreeFloorKernel<<<vertex_blocks, kThreads>>>(graph.dimension, device_degree.get(),
                                                           device_invalid.get());
    ValidateFixedDegreeKernel<<<vertex_blocks, kThreads>>>(
        graph.dimension, device_fixed_degree.get(), device_invalid.get());
    CheckCuda(cudaGetLastError(), "resident ValidateDegreeFloorKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "resident epoch synchronize");
    std::int32_t invalid_degree = 0;
    device_invalid.CopyToHost(&invalid_degree, 1U);
    if (invalid_degree != 0) {
      std::vector<std::int32_t> debug_degree(dimension);
      std::vector<std::int32_t> debug_fixed_degree(dimension);
      std::vector<std::uint8_t> debug_active(edge_count);
      std::vector<std::uint8_t> debug_fixed(edge_count);
      std::vector<std::uint8_t> debug_reason(edge_count);
      std::vector<std::uint8_t> debug_committed(edge_count);
      device_degree.CopyToHost(debug_degree.data(), dimension);
      device_fixed_degree.CopyToHost(debug_fixed_degree.data(), dimension);
      device_edge_active.CopyToHost(debug_active.data(), edge_count);
      device_fixed.CopyToHost(debug_fixed.data(), edge_count);
      device_fixed_reason.CopyToHost(debug_reason.data(), edge_count);
      device_committed.CopyToHost(debug_committed.data(), edge_count);
      std::string diagnostic = " method=" + std::to_string(static_cast<std::int32_t>(method)) +
                               ":main=" + std::to_string(main_edge_stage ? 1 : 0) +
                               ":extra=" + std::to_string(extra_edge_stage ? 1 : 0) +
                               ":depth=" + std::to_string(options.extra_edge_depth) + ":committed=";
      for (std::size_t edge = 0U; edge < edge_count; ++edge) {
        if (debug_committed[edge] != 0U) {
          diagnostic += std::to_string(edge) + "(" + std::to_string(host_edge_u[edge]) + "," +
                        std::to_string(host_edge_v[edge]) + "),";
        }
      }
      for (std::size_t node = 0U; node < dimension; ++node) {
        if (debug_degree[node] >= 2 && debug_fixed_degree[node] <= 2) {
          continue;
        }
        diagnostic += " node=" + std::to_string(node) +
                      ":degree=" + std::to_string(debug_degree[node]) +
                      ":fixed=" + std::to_string(debug_fixed_degree[node]) + ":edges=";
        for (std::size_t edge = 0U; edge < edge_count; ++edge) {
          if (debug_active[edge] != 0U && debug_fixed[edge] != 0U &&
              (host_edge_u[edge] == static_cast<std::int32_t>(node) ||
               host_edge_v[edge] == static_cast<std::int32_t>(node))) {
            diagnostic += std::to_string(edge) + "(r" + std::to_string(debug_reason[edge]) + "),";
          }
        }
      }
      throw std::logic_error("resident 并行提交破坏了最小度或 fixed-degree 不变式，code=" +
                             std::to_string(invalid_degree) + diagnostic);
    }
    result.commit_ms += ElapsedMilliseconds(commit_begin);
    result.kernel_ms += device_ms;
    if (extra_edge_stage) {
      result.extra_edge_ms += device_ms;
    } else if (main_edge_stage) {
      result.main_edge_ms += device_ms;
    } else if (method == EliminationMethod::kGeometryMain) {
      result.geometry_ms += device_ms;
    } else if (method == EliminationMethod::kLpBox) {
      result.pdlp_ms += device_ms;
    } else if (method == EliminationMethod::kJv) {
      result.jv_ms += device_ms;
    } else if (method == EliminationMethod::kGpuQuickHs) {
      result.quick_hs_ms += device_ms;
    }

    unsigned long long committed_count = 0U;
    CheckCuda(cudaMemcpy(&committed_count, device_committed_count.get(), sizeof(committed_count),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy resident committed count");
    if (options.progress_log) {
      // 只输出已有的阶段计数，不下载候选、不引入 CPU 判定；外部 wall 包含日志时间。
      std::clog << "[hybrid] snapshot=" << snapshot.sequence << " phase="
                << (extra_edge_stage  ? "extra-edge"
                    : main_edge_stage ? "main-edge"
                                      : ToString(method))
                << " position=" << main_position << " edges=" << current_edges
                << " deleted=" << committed_count
                << " nonpairs_added_total=" << result.nonpair_committed
                << " point_sweeps=" << result.lp.point_service_sweeps
                << " point_prime_sweeps=" << result.lp.point_prime_sweeps
                << " proposal_ms=" << device_ms << '\n';
    }
    if (committed_count == 0U) {
      return 0U;
    }
    if (committed_count > current_edges) {
      throw std::logic_error("resident committed count 超过活动边数");
    }
    if (options.collect_trace) {
      const SteadyClock::time_point download_begin = SteadyClock::now();
      device_committed.CopyToHost(host_committed.data(), edge_count);
      device_first_witness.CopyToHost(host_first_witness.data(), edge_count);
      if (method == EliminationMethod::kGpuQuickHs || method == EliminationMethod::kGeometryMain) {
        device_second_witness.CopyToHost(host_second_witness.data(), edge_count);
      }
      if (method == EliminationMethod::kLpBox) {
        host_quantized_dual.resize(dimension);
        device_quantized_dual.CopyToHost(host_quantized_dual.data(), dimension);
        host_quantized_cut.resize(static_cast<std::size_t>(local_sec.cut_count));
        device_quantized_cut.CopyToHost(host_quantized_cut.data(), host_quantized_cut.size());
        if (options.enable_fixing) {
          device_fixed_verified.CopyToHost(host_fixed_epoch.data(), edge_count);
        }
      }
      result.download_ms += ElapsedMilliseconds(download_begin);

      ResidentTraceEpoch trace;
      trace.method = method;
      trace.main_edge_stage = main_edge_stage;
      trace.extra_edge_stage = extra_edge_stage;
      trace.main_position = main_position;
      trace.edges_before = current_edges;
      trace.device_ms = device_ms;
      trace.vertex_dual_numerator = std::move(host_quantized_dual);
      trace.local_sec_dual_numerator = std::move(host_quantized_cut);
      trace.fractional_bits = options.fractional_bits;
      trace.incumbent_cost = options.incumbent_cost;
      trace.edge_ids.reserve(static_cast<std::size_t>(committed_count));
      trace.first_witness.reserve(static_cast<std::size_t>(committed_count));
      trace.second_witness.reserve(static_cast<std::size_t>(committed_count));
      if (method == EliminationMethod::kLpBox && options.enable_fixing) {
        for (std::size_t edge = 0U; edge < edge_count; ++edge) {
          if (host_fixed_epoch[edge] != 0U) {
            trace.fixed_edge_ids.push_back(static_cast<std::int32_t>(edge));
          }
        }
      }
      for (std::size_t edge = 0U; edge < edge_count; ++edge) {
        if (host_committed[edge] == 0U) {
          continue;
        }
        trace.edge_ids.push_back(static_cast<std::int32_t>(edge));
        trace.first_witness.push_back(host_first_witness[edge]);
        trace.second_witness.push_back(
            (method == EliminationMethod::kGpuQuickHs || method == EliminationMethod::kGeometryMain)
                ? host_second_witness[edge]
                : -1);
      }
      if (trace.edge_ids.size() != static_cast<std::size_t>(committed_count)) {
        throw std::logic_error("resident committed bitmap 与计数不一致");
      }
      result.epochs.push_back(std::move(trace));
    }
    ++snapshot_sequence;
    ++result.lp.validated_transactions;
    current_edges -= static_cast<std::size_t>(committed_count);
    compact_active_edges();
    adjacency_dirty = true;
    return static_cast<std::size_t>(committed_count);
  };

  const auto run_jv_fixed_point = [&]() -> bool {
    if (!options.enable_jv) {
      return true;
    }
    std::uint32_t round = 0U;
    while (options.max_jv_rounds == 0U || round < options.max_jv_rounds) {
      ++round;
      ++result.jv_rounds;
      const std::size_t committed = run_epoch(EliminationMethod::kJv);
      result.jv_committed += committed;
      if (committed == 0U) {
        return true;
      }
    }
    return false;
  };

  if (options.enable_geometry) {
    const std::size_t committed = run_epoch(EliminationMethod::kGeometryMain);
    result.geometry_committed += committed;
    // 即使 geometry 没有删除边，小图也需将初始编号顺序转换为成本顺序。
    if (options.gpu_complete_graph)
      adjacency_dirty = true;
  }

  bool orchestration_converged = false;
  bool local_converged = false;
  bool prefer_dirty_quick_start = false;
  // 全点 non-pair 的完整工作量是 sum_v C(deg(v), 2)。当它已经不高于
  // 一轮 Quick-HS 的候选展开量时直接进入联合固定点，避免稀疏图重复执行
  // Main/-e2；只有 pair frontier 更大时才先收敛 edge services。这里不丢弃
  // 任何 point/reply，也不设置规模上限，只用两个实际 frontier 决定顺序。
  std::int64_t initial_pair_frontier = 0;
  if (options.enable_point_nonpair || options.enable_direct_fix) {
    rebuild_compact_adjacency();
    // 此时尚未分配 nonpair mask，current_pair_count 仍为 0；它不是实际 frontier。
    // 必须先在 GPU 上统计当前度数的全部 C(deg,2)，只回传调度所需的单个计数。
    BuildPairCountsKernel<<<offset_blocks, kThreads>>>(graph.dimension, device_degree.get(),
                                                       device_pair_counts.get());
    CheckCuda(cudaGetLastError(), "initial Point frontier count launch");
    CheckCuda(cub::DeviceScan::ExclusiveSum(device_pair_scan_temp.get(), pair_scan_temp_bytes,
                                            device_pair_counts.get(),
                                            device_pair_offsets_next.get(), graph.dimension + 1),
              "initial Point frontier scan");
    CheckCuda(cudaMemcpy(&initial_pair_frontier, device_pair_offsets_next.get() + graph.dimension,
                         sizeof(initial_pair_frontier), cudaMemcpyDeviceToHost),
              "initial Point frontier count download");
    if (initial_pair_frontier < 0)
      throw std::overflow_error("initial Point frontier 计数溢出");
  }
  const std::int64_t edge_frontier =
      static_cast<std::int64_t>(active_edge_count) *
      static_cast<std::int64_t>(std::max<std::uint32_t>(1U, options.quick_hs_candidates));
  bool point_service_ready = !options.point_adaptive_start ||
                             (!options.enable_point_nonpair && !options.enable_direct_fix) ||
                             initial_pair_frontier <= edge_frontier;
  result.lp.point_initial_pairs = static_cast<std::uint64_t>(initial_pair_frontier);
  result.lp.point_initial_edge_frontier = static_cast<std::uint64_t>(edge_frontier);
  result.lp.point_deferred_initially = !point_service_ready;
  std::uint32_t orchestration = 0U;
  const bool pair_services =
      options.enable_point_nonpair || options.enable_direct_fix || options.enable_fixing;
  while (options.enable_main_edge || options.enable_extra_edge || pair_services ||
         (!options.enable_pdlp && orchestration == 0U) ||
         (options.enable_pdlp &&
          (options.max_pdlp_epochs == 0U || orchestration < options.max_pdlp_epochs))) {
    ++orchestration;
    const std::size_t edges_before = current_edges;
    const std::size_t nonpairs_before = result.nonpair_committed;
    const std::size_t fixed_before = result.fixed_count;
    if (options.enable_pdlp || pair_services) {
      if (options.enable_pdlp)
        ++result.pdlp_epochs;
      const std::size_t committed =
          run_epoch(EliminationMethod::kLpBox, false, 0, false, nullptr, -1, point_service_ready);
      result.lp_committed += committed;
      if (options.enable_fixing) {
        const std::size_t propagated = run_epoch(EliminationMethod::kFixedPropagation);
        result.fixed_propagation_committed += propagated;
      }
    }

    bool jv_converged = run_jv_fixed_point();
    bool hs_converged = !options.enable_quick_hs;
    if (jv_converged && options.enable_quick_hs) {
      std::uint32_t epoch = 0U;
      bool full_sweep = !prefer_dirty_quick_start;
      std::int32_t dirty_root_count = 0;
      if (!full_sweep) {
        dirty_root_count = select_dirty_roots(2U);
        full_sweep = dirty_root_count == 0 ||
                     static_cast<std::int64_t>(dirty_root_count) * 4 >= active_edge_count;
      }
      prefer_dirty_quick_start = false;
      while (options.max_hs_epochs == 0U || epoch < options.max_hs_epochs) {
        ++epoch;
        ++result.hs_epochs;
        if (full_sweep) {
          // active sweep 只用于提前消费局部级联；每轮最后仍执行完整 sweep，
          // 因而保守依赖半径漏掉的 root 不会造成伪固定点。
          clear_dirty_vertices();
          ++result.hs_full_sweeps;
          result.hs_full_tasks += static_cast<std::uint64_t>(active_edge_count);
        } else {
          ++result.hs_active_sweeps;
          result.hs_active_tasks += static_cast<std::uint64_t>(dirty_root_count);
        }
        const std::size_t committed =
            run_epoch(EliminationMethod::kGpuQuickHs, false, 0, false,
                      full_sweep ? nullptr : device_dirty_root_edge_ids.get(),
                      full_sweep ? -1 : dirty_root_count);
        result.quick_hs_committed += committed;
        if (committed == 0U) {
          if (full_sweep) {
            hs_converged = true;
            break;
          }
          full_sweep = true;
          continue;
        }
        jv_converged = run_jv_fixed_point();
        if (!jv_converged) {
          break;
        }
        dirty_root_count = select_dirty_roots(2U);
        full_sweep = dirty_root_count == 0 ||
                     static_cast<std::int64_t>(dirty_root_count) * 4 >= active_edge_count;
      }
    }
    local_converged = jv_converged && hs_converged;
    if (!local_converged) {
      break;
    }
    if (options.enable_extra_edge) {
      ++result.extra_edge_epochs;
      const std::size_t committed = run_epoch(EliminationMethod::kGpuQuickHs, false, 0, true);
      result.extra_edge_committed += committed;
      if (committed != 0U) {
        // 深层删除先唤醒局部依赖 root；一次无删除 active sweep 后仍回到
        // 完整 sweep，保证最终 fixed point 不依赖启发式半径。
        prefer_dirty_quick_start = true;
        continue;
      }
    }
    if (options.enable_main_edge) {
      std::size_t sweep_committed = 0U;
      for (std::uint32_t position = 1U; position <= options.main_edge_positions; ++position) {
        ++result.main_edge_epochs;
        const std::size_t committed =
            run_epoch(EliminationMethod::kGeometryMain, true, static_cast<std::int32_t>(position));
        result.main_edge_committed += committed;
        sweep_committed += committed;
      }
      if (sweep_committed != 0U) {
        prefer_dirty_quick_start = true;
        continue;
      }
    }
    if (!point_service_ready) {
      // 到达这里表示当前 snapshot 上所有便宜 edge service 都已无提交。
      // 下一 epoch 在更稀疏图上首次执行完整 point AND；其新 pair/fix
      // 仍会重新唤醒所有 edge service，最终联合固定点不变。
      point_service_ready = true;
      continue;
    }
    if ((!options.enable_pdlp && !pair_services) ||
        (current_edges == edges_before && result.nonpair_committed == nonpairs_before &&
         result.fixed_count == fixed_before)) {
      orchestration_converged = true;
      break;
    }
  }
  // 有限 epoch 的研究命令可能在一次成功提交后立即用尽预算；最终下载前
  // 仍须发布与 active bitmap 一致的 CSR/pair 快照。
  rebuild_compact_adjacency();
  CheckCuda(cudaDeviceSynchronize(), "resident final synchronize");
  const SteadyClock::time_point final_download_begin = SteadyClock::now();
  result.final_active.resize(edge_count);
  result.final_fixed.resize(edge_count);
  device_edge_active.CopyToHost(result.final_active.data(), edge_count);
  device_fixed.CopyToHost(result.final_fixed.data(), edge_count);
  if (current_pair_count > 0 && view.pair_offsets != nullptr && view.nonpair_mask != nullptr) {
    std::vector<std::int64_t> host_pair_offsets(dimension + 1U);
    std::vector<std::int64_t> host_final_rows(dimension + 1U);
    std::vector<std::int32_t> host_final_neighbors(current_edges * 2U);
    std::vector<std::uint8_t> host_nonpair(static_cast<std::size_t>(current_pair_count));
    device_pair_offsets.CopyToHost(host_pair_offsets.data(), host_pair_offsets.size());
    CheckCuda(cudaMemcpy(host_final_rows.data(), view.row_offsets,
                         host_final_rows.size() * sizeof(std::int64_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy final CSR rows");
    CheckCuda(cudaMemcpy(host_final_neighbors.data(), view.neighbors,
                         host_final_neighbors.size() * sizeof(std::int32_t),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy final CSR neighbors");
    CheckCuda(cudaMemcpy(host_nonpair.data(), view.nonpair_mask,
                         host_nonpair.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
              "cudaMemcpy final nonpair mask");
    if (host_final_rows.front() != 0 ||
        host_final_rows.back() != static_cast<std::int64_t>(host_final_neighbors.size())) {
      throw std::logic_error(
          "resident 最终 CSR slot 计数不一致: offsets=" + std::to_string(host_final_rows.back()) +
          " expected=" + std::to_string(host_final_neighbors.size()));
    }
    for (std::int32_t center = 0; center < graph.dimension; ++center) {
      const std::int64_t row_begin = host_final_rows[static_cast<std::size_t>(center)];
      const std::int64_t degree =
          host_final_rows[static_cast<std::size_t>(center) + 1U] - row_begin;
      if (degree < 0 || row_begin < 0 ||
          row_begin + degree > static_cast<std::int64_t>(host_final_neighbors.size())) {
        throw std::logic_error("resident 最终 CSR 行偏移越界: center=" + std::to_string(center));
      }
      const std::int64_t pair_begin = host_pair_offsets[static_cast<std::size_t>(center)];
      for (std::int64_t first = 0; first + 1 < degree; ++first) {
        for (std::int64_t second = first + 1; second < degree; ++second) {
          const std::int64_t local = first * (2 * degree - first - 1) / 2 + (second - first - 1);
          if (host_nonpair[static_cast<std::size_t>(pair_begin + local)] == 0U) {
            continue;
          }
          const std::int32_t first_node =
              host_final_neighbors[static_cast<std::size_t>(row_begin + first)];
          const std::int32_t second_node =
              host_final_neighbors[static_cast<std::size_t>(row_begin + second)];
          result.final_nonpairs.push_back(
              {center, std::min(first_node, second_node), std::max(first_node, second_node)});
        }
      }
    }
    std::sort(result.final_nonpairs.begin(), result.final_nonpairs.end(),
              [](const ResidentNonpair& lhs, const ResidentNonpair& rhs) {
                return std::tie(lhs.center, lhs.first, lhs.second) <
                       std::tie(rhs.center, rhs.first, rhs.second);
              });
  }
  result.download_ms += ElapsedMilliseconds(final_download_begin);
  result.final_edges = static_cast<std::size_t>(
      std::count(result.final_active.begin(), result.final_active.end(), std::uint8_t{1U}));
  if (result.final_edges != current_edges) {
    throw std::logic_error("resident 最终活动边 bitmap 与计数不一致");
  }
  const std::size_t downloaded_fixed = static_cast<std::size_t>(
      std::count(result.final_fixed.begin(), result.final_fixed.end(), std::uint8_t{1U}));
  // LP kernel 的计数只覆盖“由本轮 LP 新固定”的边；度数降到 2 后由 GPU
  // 推导出的隐含固定边不会经过该计数器。最终位图才是跨 epoch 的权威状态。
  result.fixed_count = downloaded_fixed;
  result.converged = local_converged && orchestration_converged;
  result.solve_wall_ms = ElapsedMilliseconds(solve_begin);
  return result;
}

} // namespace cudaee::detail
