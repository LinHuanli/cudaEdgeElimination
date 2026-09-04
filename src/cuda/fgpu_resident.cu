#include "../fgpu/resident_backend.hpp"

#include "../fgpu/quick_hs_predicate.hpp"
#include "cuda_edge_elimination/cuda_device_affinity.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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

__global__ void BuildAdjacencyKernel(const std::int32_t dimension, const std::uint8_t* const active,
                                     const std::int64_t* const distance, std::int32_t* const degree,
                                     std::int32_t* const neighbors) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node >= dimension) {
    return;
  }
  const std::int64_t row = static_cast<std::int64_t>(node) * dimension;
  std::int32_t count = 0;
  for (std::int32_t other = 0; other < dimension; ++other) {
    if (active[row + other] == 0U) {
      continue;
    }
    std::int32_t position = count++;
    const std::int64_t cost = distance[row + other];
    while (position > 0) {
      const std::int32_t previous = neighbors[row + position - 1];
      const std::int64_t previous_cost = distance[row + previous];
      if (previous_cost < cost || (previous_cost == cost && previous < other)) {
        break;
      }
      neighbors[row + position] = previous;
      --position;
    }
    neighbors[row + position] = other;
  }
  degree[node] = count;
}

constexpr std::int32_t kMaxGeometryPotential = 32;

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
                                                     const std::int64_t* const y) {
  const DeviceInterval dx = IntervalSubtract(IntervalInteger(x[lhs]), IntervalInteger(x[rhs]));
  const DeviceInterval dy = IntervalSubtract(IntervalInteger(y[lhs]), IntervalInteger(y[rhs]));
  return IntervalSqrt(IntervalAdd(IntervalSquare(dx), IntervalSquare(dy)));
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
  const DeviceInterval dpq = CoordinateDistanceInterval(p, q, x, y);
  const DeviceInterval dpr = CoordinateDistanceInterval(p, r, x, y);
  const DeviceInterval dqr = CoordinateDistanceInterval(q, r, x, y);
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

__global__ void NearestDistanceKernel(const std::int32_t dimension,
                                      const std::int64_t* const distance,
                                      std::int64_t* const nearest) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node >= dimension) {
    return;
  }
  std::int64_t best = INT64_MAX;
  const std::int64_t row = static_cast<std::int64_t>(node) * dimension;
  for (std::int32_t other = 0; other < dimension; ++other) {
    if (other != node) {
      best = min(best, distance[row + other]);
    }
  }
  nearest[node] = best;
}

__global__ void GeometryKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                               const std::int32_t* const edge_v,
                               const std::uint8_t* const edge_active,
                               const std::uint8_t* const protected_edge,
                               const quick_hs::GraphView graph, const std::int64_t* const x,
                               const std::int64_t* const y, const std::int64_t* const nearest,
                               const std::int32_t potential_count, std::uint8_t* const proposed,
                               std::int32_t* const first_witness,
                               std::int32_t* const second_witness) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge >= edge_count || edge_active[edge] == 0U || protected_edge[edge] != 0U) {
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
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    if (node == p || node == q) {
      continue;
    }
    const double dx =
        2.0 * static_cast<double>(x[node]) - static_cast<double>(x[p]) - static_cast<double>(x[q]);
    const double dy =
        2.0 * static_cast<double>(y[node]) - static_cast<double>(y[p]) - static_cast<double>(y[q]);
    InsertGeometryCandidate(node, dx * dx + dy * dy, potential_count, nodes, scores, &selected);
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
  const std::int32_t row = c * graph.dimension;
  for (std::int32_t index = 0; index < graph.degree[c]; ++index) {
    const std::int32_t d = graph.neighbors[row + index];
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
    const std::int32_t row = from * graph.dimension;
    for (std::int32_t index = 0; index < graph.degree[from]; ++index) {
      const std::int32_t node = graph.neighbors[row + index];
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

__global__ void JvKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                         const std::int32_t* const edge_v, const std::uint8_t* const edge_active,
                         const std::uint8_t* const protected_edge, const quick_hs::GraphView graph,
                         std::uint8_t* const proposed, std::int32_t* const first_witness) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge >= edge_count || edge_active[edge] == 0U || protected_edge[edge] != 0U) {
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

__global__ void QuickHsKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                              const std::int32_t* const edge_v,
                              const std::uint8_t* const edge_active,
                              const std::uint8_t* const protected_edge,
                              const quick_hs::GraphView graph, std::uint8_t* const proposed,
                              std::int32_t* const first_witness,
                              std::int32_t* const second_witness) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge >= edge_count || edge_active[edge] == 0U || protected_edge[edge] != 0U) {
    return;
  }
  const std::int32_t a = edge_u[edge];
  const std::int32_t b = edge_v[edge];
  const quick_hs::Witness witness = quick_hs::FindWitness(graph, a, b);
  if (witness.c >= 0) {
    proposed[edge] = 1U;
    first_witness[edge] = witness.c;
    second_witness[edge] = witness.d;
  }
}

__global__ void ActiveCostSummaryKernel(const std::int32_t edge_count,
                                        const std::int64_t* const edge_weight,
                                        const std::uint8_t* const edge_active,
                                        unsigned long long* const active_count,
                                        unsigned long long* const cost_sum) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge < edge_count && edge_active[edge] != 0U) {
    atomicAdd(active_count, 1ULL);
    atomicAdd(cost_sum, static_cast<unsigned long long>(edge_weight[edge]));
  }
}

__global__ void
SelectDegreeBoxKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                      const std::int32_t* const edge_v, const std::int64_t* const edge_weight,
                      const std::uint8_t* const edge_active, const double* const dual,
                      std::int32_t* const selected_degree) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge >= edge_count || edge_active[edge] == 0U) {
    return;
  }
  const std::int32_t u = edge_u[edge];
  const std::int32_t v = edge_v[edge];
  if (static_cast<double>(edge_weight[edge]) - dual[u] - dual[v] < 0.0) {
    atomicAdd(&selected_degree[u], 1);
    atomicAdd(&selected_degree[v], 1);
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

__global__ void DegreeConstantKernel(const std::int32_t dimension,
                                     const std::int64_t* const quantized,
                                     long long* const lower_bound) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex < dimension) {
    // CUDA 的 64 位 atomicAdd 只暴露 unsigned overload；在已做无溢出门禁后，
    // 二补码模加与有符号精确和的位模式一致。
    atomicAdd(reinterpret_cast<unsigned long long*>(lower_bound),
              static_cast<unsigned long long>(2 * quantized[vertex]));
  }
}

__global__ void ReducedCostKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                                  const std::int32_t* const edge_v,
                                  const std::int64_t* const edge_weight,
                                  const std::uint8_t* const edge_active,
                                  const std::int64_t denominator,
                                  const std::int64_t* const quantized,
                                  std::int64_t* const reduced_cost, long long* const lower_bound) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge >= edge_count) {
    return;
  }
  if (edge_active[edge] == 0U) {
    reduced_cost[edge] = 0;
    return;
  }
  const std::int64_t reduced =
      edge_weight[edge] * denominator - quantized[edge_u[edge]] - quantized[edge_v[edge]];
  reduced_cost[edge] = reduced;
  if (reduced < 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(lower_bound),
              static_cast<unsigned long long>(reduced));
  }
}

__global__ void
LpForcedOneKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                  const std::int32_t* const edge_v, const std::uint8_t* const edge_active,
                  const std::uint8_t* const protected_edge, const std::int32_t* const degree,
                  const std::int64_t* const reduced_cost, const long long* const lower_bound,
                  const std::int64_t incumbent_numerator, std::uint8_t* const proposed) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge >= edge_count || edge_active[edge] == 0U || protected_edge[edge] != 0U ||
      degree[edge_u[edge]] <= 2 || degree[edge_v[edge]] <= 2) {
    return;
  }
  const std::int64_t positive_reduced = reduced_cost[edge] > 0 ? reduced_cost[edge] : 0;
  if (*lower_bound + positive_reduced > incumbent_numerator) {
    proposed[edge] = 1U;
  }
}

// 候选在同一只读快照生成；提交仍按 stable edge id 串行执行最小度门禁。
// 该 kernel 的工作远小于组合搜索，单线程可保证与 CPU replayer 完全同序。
__global__ void CommitKernel(const std::int32_t dimension, const std::int32_t edge_count,
                             const std::int32_t* const edge_u, const std::int32_t* const edge_v,
                             std::uint8_t* const edge_active, std::uint8_t* const active_matrix,
                             const std::uint8_t* const protected_edge,
                             const std::uint8_t* const proposed, std::int32_t* const degree,
                             std::uint8_t* const committed,
                             unsigned long long* const committed_count) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  unsigned long long count = 0U;
  for (std::int32_t edge = 0; edge < edge_count; ++edge) {
    if (edge_active[edge] == 0U || proposed[edge] == 0U || protected_edge[edge] != 0U) {
      continue;
    }
    const std::int32_t u = edge_u[edge];
    const std::int32_t v = edge_v[edge];
    if (degree[u] <= 2 || degree[v] <= 2) {
      continue;
    }
    edge_active[edge] = 0U;
    active_matrix[static_cast<std::int64_t>(u) * dimension + v] = 0U;
    active_matrix[static_cast<std::int64_t>(v) * dimension + u] = 0U;
    --degree[u];
    --degree[v];
    committed[edge] = 1U;
    ++count;
  }
  *committed_count = count;
}

std::vector<std::int64_t> BuildDistanceMatrix(const GraphSnapshot& graph) {
  const std::size_t dimension = static_cast<std::size_t>(graph.dimension);
  std::vector<std::int64_t> result(dimension * dimension);
  for (std::int32_t u = 0; u < graph.dimension; ++u) {
    for (std::int32_t v = 0; v < graph.dimension; ++v) {
      result[static_cast<std::size_t>(u) * dimension + static_cast<std::size_t>(v)] =
          graph.Distance(u, v);
    }
  }
  return result;
}

} // namespace

bool ResidentEliminationCudaAvailable(std::string* const reason) {
  return SelectDevice(-1, reason) >= 0;
}

ResidentGpuResult RunResidentEliminationCuda(const GraphSnapshot& graph,
                                             const std::vector<std::uint8_t>& protected_edges,
                                             const ResidentGpuOptions& options) {
  if (graph.dimension <= 0 || graph.edges.empty() || !graph.integer_coordinates ||
      !graph.integer_distance_safe || protected_edges.size() != graph.edges.size() ||
      options.max_hs_epochs == 0U || options.max_jv_rounds == 0U || options.max_pdlp_epochs == 0U ||
      options.pdlp_iterations == 0U || options.potential_candidates < 2U ||
      options.potential_candidates > 32U || options.fractional_bits > 30U ||
      (options.enable_pdlp && options.incumbent_cost < 0) ||
      (!options.enable_quick_hs && !options.enable_jv && !options.enable_geometry &&
       !options.enable_pdlp)) {
    throw std::invalid_argument("resident GPU 输入图、tour mask、阶段开关或预算非法");
  }
  if (graph.dimension > kMaxResidentDimension ||
      graph.edges.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("resident GPU 当前维度上限为 4096，edge id 为 int32");
  }
  std::string reason;
  const int device = SelectDevice(options.device, &reason);
  if (device < 0) {
    throw std::runtime_error("resident GPU CUDA 不可用: " + reason);
  }

  const SteadyClock::time_point solve_begin = SteadyClock::now();
  const std::size_t dimension = static_cast<std::size_t>(graph.dimension);
  const std::size_t matrix_size = dimension * dimension;
  const std::size_t edge_count = graph.edges.size();
  std::vector<std::int32_t> host_edge_u(edge_count);
  std::vector<std::int32_t> host_edge_v(edge_count);
  std::vector<std::int64_t> host_edge_weight(edge_count);
  std::vector<std::uint8_t> host_edge_active(edge_count);
  std::vector<std::uint8_t> host_active_matrix(matrix_size, 0U);
  for (std::size_t edge = 0; edge < edge_count; ++edge) {
    host_edge_u[edge] = graph.edges[edge].u;
    host_edge_v[edge] = graph.edges[edge].v;
    host_edge_weight[edge] = graph.edges[edge].weight;
    host_edge_active[edge] = graph.edges[edge].active ? 1U : 0U;
    if (graph.edges[edge].active) {
      host_active_matrix[static_cast<std::size_t>(graph.edges[edge].u) * dimension +
                         static_cast<std::size_t>(graph.edges[edge].v)] = 1U;
      host_active_matrix[static_cast<std::size_t>(graph.edges[edge].v) * dimension +
                         static_cast<std::size_t>(graph.edges[edge].u)] = 1U;
    }
  }
  const std::vector<std::int64_t> host_distance = BuildDistanceMatrix(graph);
  std::vector<std::int64_t> host_x(dimension);
  std::vector<std::int64_t> host_y(dimension);
  for (std::size_t node = 0U; node < dimension; ++node) {
    host_x[node] = graph.points[node].integer_x;
    host_y[node] = graph.points[node].integer_y;
  }
  const std::int64_t denominator = std::int64_t{1} << options.fractional_bits;
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
  const long double conservative_accumulator =
      2.0L * dimension * maximum_quantized +
      edge_count *
          (static_cast<long double>(maximum_edge_weight) * denominator + 2.0L * maximum_quantized);
  if (options.enable_pdlp &&
      (maximum_quantized > 1.0e15L ||
       conservative_accumulator > static_cast<long double>(INT64_MAX) / 2.0L ||
       static_cast<long double>(options.incumbent_cost) * denominator >
           static_cast<long double>(INT64_MAX) / 2.0L)) {
    throw std::overflow_error("resident PDLP int64 证明累加器无法覆盖当前实例");
  }

  DeviceBuffer<std::int32_t> device_edge_u(edge_count, device);
  DeviceBuffer<std::int32_t> device_edge_v(edge_count, device);
  DeviceBuffer<std::int64_t> device_edge_weight(edge_count, device);
  DeviceBuffer<std::uint8_t> device_edge_active(edge_count, device);
  DeviceBuffer<std::uint8_t> device_protected(protected_edges.size(), device);
  DeviceBuffer<std::uint8_t> device_active_matrix(matrix_size, device);
  DeviceBuffer<std::int64_t> device_distance(matrix_size, device);
  DeviceBuffer<std::int64_t> device_x(dimension, device);
  DeviceBuffer<std::int64_t> device_y(dimension, device);
  DeviceBuffer<std::int64_t> device_nearest(dimension, device);
  DeviceBuffer<std::int32_t> device_degree(dimension, device);
  DeviceBuffer<std::int32_t> device_neighbors(matrix_size, device);
  DeviceBuffer<std::uint8_t> device_proposed(edge_count, device);
  DeviceBuffer<std::uint8_t> device_committed(edge_count, device);
  DeviceBuffer<std::int32_t> device_first_witness(edge_count, device);
  DeviceBuffer<std::int32_t> device_second_witness(edge_count, device);
  DeviceBuffer<unsigned long long> device_committed_count(1U, device);
  DeviceBuffer<std::int32_t> device_selected_degree(dimension, device);
  DeviceBuffer<double> device_dual(dimension, device);
  DeviceBuffer<double> device_average(dimension, device);
  DeviceBuffer<std::int64_t> device_quantized_dual(dimension, device);
  DeviceBuffer<std::int64_t> device_reduced_cost(edge_count, device);
  DeviceBuffer<unsigned long long> device_summary(2U, device);
  DeviceBuffer<long long> device_lower_bound(1U, device);
  DeviceBuffer<std::int32_t> device_invalid(1U, device);

  ResidentGpuResult result;
  result.selected_device = device;
  result.backend = "cuda-fully-resident-geometry+pdlp+jv+quick-hs";
  result.initial_edges = graph.ActiveEdgeCount();
  result.resident_bytes =
      device_edge_u.bytes() + device_edge_v.bytes() + device_edge_weight.bytes() +
      device_edge_active.bytes() + device_protected.bytes() + device_active_matrix.bytes() +
      device_distance.bytes() + device_x.bytes() + device_y.bytes() + device_nearest.bytes() +
      device_degree.bytes() + device_neighbors.bytes() + device_proposed.bytes() +
      device_committed.bytes() + device_first_witness.bytes() + device_second_witness.bytes() +
      device_committed_count.bytes() + device_selected_degree.bytes() + device_dual.bytes() +
      device_average.bytes() + device_quantized_dual.bytes() + device_reduced_cost.bytes() +
      device_summary.bytes() + device_lower_bound.bytes() + device_invalid.bytes();

  const SteadyClock::time_point upload_begin = SteadyClock::now();
  device_edge_u.CopyFromHost(host_edge_u.data(), edge_count);
  device_edge_v.CopyFromHost(host_edge_v.data(), edge_count);
  device_edge_weight.CopyFromHost(host_edge_weight.data(), edge_count);
  device_edge_active.CopyFromHost(host_edge_active.data(), edge_count);
  device_protected.CopyFromHost(protected_edges.data(), protected_edges.size());
  device_active_matrix.CopyFromHost(host_active_matrix.data(), matrix_size);
  device_distance.CopyFromHost(host_distance.data(), matrix_size);
  device_x.CopyFromHost(host_x.data(), dimension);
  device_y.CopyFromHost(host_y.data(), dimension);
  result.upload_ms = ElapsedMilliseconds(upload_begin);

  constexpr int kThreads = 128;
  const int edge_blocks = (static_cast<std::int32_t>(edge_count) + kThreads - 1) / kThreads;
  const int vertex_blocks = (graph.dimension + kThreads - 1) / kThreads;
  const quick_hs::GraphView view{.dimension = graph.dimension,
                                 .degree = device_degree.get(),
                                 .neighbors = device_neighbors.get(),
                                 .distance = device_distance.get(),
                                 .active = device_active_matrix.get()};
  std::size_t current_edges = result.initial_edges;
  std::vector<std::uint8_t> host_committed(edge_count);
  std::vector<std::int32_t> host_first_witness(edge_count);
  std::vector<std::int32_t> host_second_witness(edge_count);

  const auto rebuild = [&] {
    BuildAdjacencyKernel<<<vertex_blocks, kThreads>>>(graph.dimension, device_active_matrix.get(),
                                                      device_distance.get(), device_degree.get(),
                                                      device_neighbors.get());
    CheckCuda(cudaGetLastError(), "BuildAdjacencyKernel launch");
  };
  const auto run_epoch = [&](const EliminationMethod method) -> std::size_t {
    const SteadyClock::time_point kernel_begin = SteadyClock::now();
    rebuild();
    CheckCuda(cudaMemset(device_proposed.get(), 0, edge_count * sizeof(std::uint8_t)),
              "cudaMemset resident proposed");
    CheckCuda(cudaMemset(device_committed.get(), 0, edge_count * sizeof(std::uint8_t)),
              "cudaMemset resident committed");
    CheckCuda(cudaMemset(device_first_witness.get(), 0xff, edge_count * sizeof(std::int32_t)),
              "cudaMemset resident first witness");
    CheckCuda(cudaMemset(device_second_witness.get(), 0xff, edge_count * sizeof(std::int32_t)),
              "cudaMemset resident second witness");
    std::vector<std::int64_t> host_quantized_dual;
    if (method == EliminationMethod::kJv) {
      JvKernel<<<edge_blocks, kThreads>>>(static_cast<std::int32_t>(edge_count),
                                          device_edge_u.get(), device_edge_v.get(),
                                          device_edge_active.get(), device_protected.get(), view,
                                          device_proposed.get(), device_first_witness.get());
    } else if (method == EliminationMethod::kGpuQuickHs) {
      QuickHsKernel<<<edge_blocks, kThreads>>>(
          static_cast<std::int32_t>(edge_count), device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_protected.get(), view, device_proposed.get(),
          device_first_witness.get(), device_second_witness.get());
    } else if (method == EliminationMethod::kGeometryMain) {
      NearestDistanceKernel<<<vertex_blocks, kThreads>>>(graph.dimension, device_distance.get(),
                                                         device_nearest.get());
      GeometryKernel<<<edge_blocks, kThreads>>>(
          static_cast<std::int32_t>(edge_count), device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_protected.get(), view, device_x.get(), device_y.get(),
          device_nearest.get(), static_cast<std::int32_t>(options.potential_candidates),
          device_proposed.get(), device_first_witness.get(), device_second_witness.get());
    } else if (method == EliminationMethod::kLpBox) {
      CheckCuda(cudaMemset(device_summary.get(), 0, 2U * sizeof(unsigned long long)),
                "cudaMemset resident PDLP summary");
      ActiveCostSummaryKernel<<<edge_blocks, kThreads>>>(
          static_cast<std::int32_t>(edge_count), device_edge_weight.get(), device_edge_active.get(),
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
      CheckCuda(cudaMemset(device_dual.get(), 0, dimension * sizeof(double)),
                "cudaMemset resident PDLP dual");
      CheckCuda(cudaMemset(device_average.get(), 0, dimension * sizeof(double)),
                "cudaMemset resident PDLP average");
      for (std::uint32_t iteration = 0U; iteration < options.pdlp_iterations; ++iteration) {
        CheckCuda(cudaMemset(device_selected_degree.get(), 0, dimension * sizeof(std::int32_t)),
                  "cudaMemset resident PDLP selected degree");
        SelectDegreeBoxKernel<<<edge_blocks, kThreads>>>(
            static_cast<std::int32_t>(edge_count), device_edge_u.get(), device_edge_v.get(),
            device_edge_weight.get(), device_edge_active.get(), device_dual.get(),
            device_selected_degree.get());
        const double step = initial_step / sqrt(1.0 + static_cast<double>(iteration) / 8.0);
        UpdateResidentDualKernel<<<vertex_blocks, kThreads>>>(
            graph.dimension, device_selected_degree.get(), step, dual_limit, device_dual.get(),
            device_average.get());
      }
      CheckCuda(cudaMemset(device_invalid.get(), 0, sizeof(std::int32_t)),
                "cudaMemset resident PDLP invalid");
      QuantizeResidentDualKernel<<<vertex_blocks, kThreads>>>(
          graph.dimension, 1.0 / static_cast<double>(options.pdlp_iterations),
          static_cast<double>(denominator), 1.0e15, device_average.get(),
          device_quantized_dual.get(), device_invalid.get());
      CheckCuda(cudaMemset(device_lower_bound.get(), 0, sizeof(long long)),
                "cudaMemset resident PDLP lower bound");
      DegreeConstantKernel<<<vertex_blocks, kThreads>>>(
          graph.dimension, device_quantized_dual.get(), device_lower_bound.get());
      ReducedCostKernel<<<edge_blocks, kThreads>>>(
          static_cast<std::int32_t>(edge_count), device_edge_u.get(), device_edge_v.get(),
          device_edge_weight.get(), device_edge_active.get(), denominator,
          device_quantized_dual.get(), device_reduced_cost.get(), device_lower_bound.get());
      LpForcedOneKernel<<<edge_blocks, kThreads>>>(
          static_cast<std::int32_t>(edge_count), device_edge_u.get(), device_edge_v.get(),
          device_edge_active.get(), device_protected.get(), device_degree.get(),
          device_reduced_cost.get(), device_lower_bound.get(), options.incumbent_cost * denominator,
          device_proposed.get());
      CheckCuda(cudaDeviceSynchronize(), "resident PDLP proof synchronize");
      std::int32_t invalid = 0;
      device_invalid.CopyToHost(&invalid, 1U);
      if (invalid != 0) {
        throw std::runtime_error("resident PDLP dual 超出量化安全范围");
      }
    } else {
      throw std::logic_error("resident run_epoch 收到不支持的方法");
    }
    CheckCuda(cudaGetLastError(), "resident elimination kernel launch");
    CommitKernel<<<1, 1>>>(graph.dimension, static_cast<std::int32_t>(edge_count),
                           device_edge_u.get(), device_edge_v.get(), device_edge_active.get(),
                           device_active_matrix.get(), device_protected.get(),
                           device_proposed.get(), device_degree.get(), device_committed.get(),
                           device_committed_count.get());
    CheckCuda(cudaGetLastError(), "resident CommitKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "resident epoch synchronize");
    const double device_ms = ElapsedMilliseconds(kernel_begin);
    result.kernel_ms += device_ms;

    unsigned long long committed_count = 0U;
    CheckCuda(cudaMemcpy(&committed_count, device_committed_count.get(), sizeof(committed_count),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy resident committed count");
    if (committed_count == 0U) {
      return 0U;
    }
    if (committed_count > current_edges) {
      throw std::logic_error("resident committed count 超过活动边数");
    }
    const SteadyClock::time_point download_begin = SteadyClock::now();
    device_committed.CopyToHost(host_committed.data(), edge_count);
    device_first_witness.CopyToHost(host_first_witness.data(), edge_count);
    if (method == EliminationMethod::kGpuQuickHs || method == EliminationMethod::kGeometryMain) {
      device_second_witness.CopyToHost(host_second_witness.data(), edge_count);
    }
    if (method == EliminationMethod::kLpBox) {
      host_quantized_dual.resize(dimension);
      device_quantized_dual.CopyToHost(host_quantized_dual.data(), dimension);
    }
    result.download_ms += ElapsedMilliseconds(download_begin);

    ResidentTraceEpoch trace;
    trace.method = method;
    trace.edges_before = current_edges;
    trace.device_ms = device_ms;
    trace.vertex_dual_numerator = std::move(host_quantized_dual);
    trace.fractional_bits = options.fractional_bits;
    trace.incumbent_cost = options.incumbent_cost;
    trace.edge_ids.reserve(static_cast<std::size_t>(committed_count));
    trace.first_witness.reserve(static_cast<std::size_t>(committed_count));
    trace.second_witness.reserve(static_cast<std::size_t>(committed_count));
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
    current_edges -= static_cast<std::size_t>(committed_count);
    result.epochs.push_back(std::move(trace));
    return static_cast<std::size_t>(committed_count);
  };

  const auto run_jv_fixed_point = [&]() -> bool {
    if (!options.enable_jv) {
      return true;
    }
    for (std::uint32_t round = 0U; round < options.max_jv_rounds; ++round) {
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
  }

  const std::uint32_t orchestration_limit = options.enable_pdlp ? options.max_pdlp_epochs : 1U;
  bool orchestration_converged = false;
  bool local_converged = false;
  for (std::uint32_t orchestration = 0U; orchestration < orchestration_limit; ++orchestration) {
    const std::size_t edges_before = current_edges;
    if (options.enable_pdlp) {
      ++result.pdlp_epochs;
      const std::size_t committed = run_epoch(EliminationMethod::kLpBox);
      result.lp_committed += committed;
    }

    bool jv_converged = run_jv_fixed_point();
    bool hs_converged = !options.enable_quick_hs;
    if (jv_converged && options.enable_quick_hs) {
      for (std::uint32_t epoch = 0U; epoch < options.max_hs_epochs; ++epoch) {
        ++result.hs_epochs;
        const std::size_t committed = run_epoch(EliminationMethod::kGpuQuickHs);
        result.quick_hs_committed += committed;
        if (committed == 0U) {
          hs_converged = true;
          break;
        }
        jv_converged = run_jv_fixed_point();
        if (!jv_converged) {
          break;
        }
      }
    }
    local_converged = jv_converged && hs_converged;
    if (!local_converged) {
      break;
    }
    if (!options.enable_pdlp || current_edges == edges_before) {
      orchestration_converged = true;
      break;
    }
  }
  CheckCuda(cudaDeviceSynchronize(), "resident final synchronize");
  const SteadyClock::time_point final_download_begin = SteadyClock::now();
  result.final_active.resize(edge_count);
  device_edge_active.CopyToHost(result.final_active.data(), edge_count);
  result.download_ms += ElapsedMilliseconds(final_download_begin);
  result.final_edges = static_cast<std::size_t>(
      std::count(result.final_active.begin(), result.final_active.end(), std::uint8_t{1U}));
  if (result.final_edges != current_edges) {
    throw std::logic_error("resident 最终活动边 bitmap 与计数不一致");
  }
  result.converged = local_converged && orchestration_converged;
  result.solve_wall_ms = ElapsedMilliseconds(solve_begin);
  return result;
}

} // namespace cudaee::detail
