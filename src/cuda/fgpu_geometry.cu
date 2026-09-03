#include "../fgpu/geometry_backend.hpp"

#include "cuda_edge_elimination/cuda_device_affinity.hpp"

#include <cuda_runtime.h>

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

constexpr int kMaxPotentialCandidates = 32;
void CheckCuda(const cudaError_t status, const char* const operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const std::size_t count, const int device) : count_(count), device_(device) {
    if (count_ != 0U) {
      if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("FGPU geometry CUDA buffer 大小溢出");
      }
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(geometry allocation)");
      CheckCuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc(geometry)");
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

  void CopyFromHost(const T* const source, const std::size_t count) {
    if (count > count_) {
      throw std::logic_error("FGPU geometry H2D buffer 越界");
    }
    if (count != 0U) {
      CheckCuda(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy geometry H2D");
    }
  }

  void CopyToHost(T* const destination, const std::size_t count) const {
    if (count > count_) {
      throw std::logic_error("FGPU geometry D2H buffer 越界");
    }
    if (count != 0U) {
      CheckCuda(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                "cudaMemcpy geometry D2H");
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

__device__ std::uint64_t AbsoluteDifference(const std::int64_t lhs, const std::int64_t rhs) {
  if ((lhs < 0) == (rhs < 0)) {
    return lhs >= rhs ? static_cast<std::uint64_t>(lhs - rhs)
                      : static_cast<std::uint64_t>(rhs - lhs);
  }
  const std::uint64_t positive =
      lhs >= 0 ? static_cast<std::uint64_t>(lhs) : static_cast<std::uint64_t>(rhs);
  const std::int64_t negative = lhs < 0 ? lhs : rhs;
  return positive + static_cast<std::uint64_t>(-(negative + 1)) + 1U;
}

__device__ std::uint64_t IntegerSqrtFloor(const std::uint64_t value) {
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

__device__ std::int64_t RoundedEucDistance(const std::int32_t lhs, const std::int32_t rhs,
                                           const std::int64_t* const x,
                                           const std::int64_t* const y) {
  const std::uint64_t dx = AbsoluteDifference(x[lhs], x[rhs]);
  const std::uint64_t dy = AbsoluteDifference(y[lhs], y[rhs]);
  const std::uint64_t squared = dx * dx + dy * dy;
  const std::uint64_t root = IntegerSqrtFloor(squared);
  return static_cast<std::int64_t>(root + (squared - root * root > root ? 1U : 0U));
}

__device__ double TrueDistance(const std::int32_t lhs, const std::int32_t rhs,
                               const std::int64_t* const x, const std::int64_t* const y) {
  const double dx = static_cast<double>(x[lhs]) - static_cast<double>(x[rhs]);
  const double dy = static_cast<double>(y[lhs]) - static_cast<double>(y[rhs]);
  return hypot(dx, dy);
}

__global__ void NearestRoundedDistanceKernel(const std::int32_t dimension,
                                             const std::int64_t* const x,
                                             const std::int64_t* const y,
                                             std::int64_t* const nearest) {
  const std::int32_t node = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (node >= dimension) {
    return;
  }
  std::int64_t best = LLONG_MAX;
  for (std::int32_t other = 0; other < dimension; ++other) {
    if (other != node) {
      best = min(best, RoundedEucDistance(node, other, x, y));
    }
  }
  nearest[node] = best;
}

struct ApproxPotential {
  std::int32_t node{-1};
  double min_p{};
  double min_q{};
};

__device__ double ClampCosine(const double value) { return fmin(1.0, fmax(-1.0, value)); }

__device__ bool ApproxPotentialBounds(const std::int32_t p, const std::int32_t q,
                                      const std::int32_t r, const std::int64_t lpq_integer,
                                      const std::int64_t nearest, const std::int64_t* const x,
                                      const std::int64_t* const y, const double tolerance,
                                      ApproxPotential* const output) {
  const double delta = static_cast<double>(nearest) - 0.5;
  const double lpq = static_cast<double>(lpq_integer);
  const double lpr = static_cast<double>(RoundedEucDistance(p, r, x, y));
  const double lqr = static_cast<double>(RoundedEucDistance(q, r, x, y));
  const double dpq = TrueDistance(p, q, x, y);
  const double dpr = TrueDistance(p, r, x, y);
  const double dqr = TrueDistance(q, r, x, y);
  if (!(delta > 0.0 && dpq > 0.0 && dpr > 0.0 && dqr > 0.0)) {
    return false;
  }
  const double length_p = delta + lpq - lqr - 1.0;
  const double length_q = delta + lpq - lpr - 1.0;
  if (!(length_p > 0.0 && length_q > 0.0) || length_p + length_q + tolerance < lpq - 0.5) {
    return false;
  }

  const double gamma_numerator = length_p + length_q - lpq + 0.5;
  const double cos_gamma = 1.0 - gamma_numerator * gamma_numerator / (2.0 * delta * delta);
  const double cos_alpha_p_half =
      (length_q * length_q - delta * delta - dqr * dqr) / (2.0 * delta * dqr);
  const double cos_alpha_q_half =
      (length_p * length_p - delta * delta - dpr * dpr) / (2.0 * delta * dpr);
  if (cos_alpha_p_half <= -tolerance || cos_alpha_q_half <= -tolerance ||
      cos_gamma < -1.0 - tolerance || cos_gamma > 1.0 + tolerance ||
      cos_alpha_p_half > 1.0 + tolerance || cos_alpha_q_half > 1.0 + tolerance) {
    return false;
  }
  const double cap = ClampCosine(cos_alpha_p_half);
  const double caq = ClampCosine(cos_alpha_q_half);
  if (2.0 * cap * cap - 1.0 + tolerance <= ClampCosine(cos_gamma) ||
      2.0 * caq * caq - 1.0 + tolerance <= ClampCosine(cos_gamma)) {
    return false;
  }

  const double cos_e_p = ClampCosine((dpq * dpq + dpr * dpr - dqr * dqr) / (2.0 * dpq * dpr));
  const double cos_e_q = ClampCosine((dpq * dpq + dqr * dqr - dpr * dpr) / (2.0 * dpq * dqr));
  const double left_p = ((dqr + delta) * (dqr + delta) + dpq * dpq - length_p * length_p) /
                        (2.0 * (dqr + delta) * dpq);
  const double left_q = ((dpr + delta) * (dpr + delta) + dpq * dpq - length_q * length_q) /
                        (2.0 * (dpr + delta) * dpq);
  if (left_p > cos_e_q + tolerance || left_q > cos_e_p + tolerance) {
    return false;
  }
  const double cos_t_p =
      ClampCosine((length_p * length_p + dpr * dpr - delta * delta) / (2.0 * length_p * dpr));
  const double cos_t_q =
      ClampCosine((length_q * length_q + dqr * dqr - delta * delta) / (2.0 * length_q * dqr));
  if (cos_e_p + cos_t_p < -tolerance || cos_e_q + cos_t_q < -tolerance) {
    return false;
  }
  const double sin_e_p = sqrt(fmax(0.0, 1.0 - cos_e_p * cos_e_p));
  const double sin_e_q = sqrt(fmax(0.0, 1.0 - cos_e_q * cos_e_q));
  const double sin_t_p = sqrt(fmax(0.0, 1.0 - cos_t_p * cos_t_p));
  const double sin_t_q = sqrt(fmax(0.0, 1.0 - cos_t_q * cos_t_q));
  const double cos_sum_p = cos_e_p * cos_t_p - sin_e_p * sin_t_p;
  const double cos_sum_q = cos_e_q * cos_t_q - sin_e_q * sin_t_q;
  const double max_p_squared = dpq * dpq + length_q * length_q - 2.0 * dpq * length_q * cos_sum_q;
  const double max_q_squared = dpq * dpq + length_p * length_p - 2.0 * dpq * length_p * cos_sum_p;
  if (max_p_squared < -tolerance || max_q_squared < -tolerance) {
    return false;
  }
  output->node = r;
  output->min_p = delta - 1.0 - sqrt(fmax(0.0, max_p_squared));
  output->min_q = delta - 1.0 - sqrt(fmax(0.0, max_q_squared));
  return isfinite(output->min_p) && isfinite(output->min_q);
}

__device__ bool CandidateLess(const double lhs_score, const std::int32_t lhs_node,
                              const double rhs_score, const std::int32_t rhs_node) {
  return lhs_score < rhs_score || (lhs_score == rhs_score && lhs_node < rhs_node);
}

__device__ void InsertMidpointCandidate(const std::int32_t node, const double score,
                                        const std::int32_t capacity,
                                        std::int32_t* const candidate_nodes,
                                        double* const candidate_scores,
                                        std::int32_t* const candidate_count) {
  if (*candidate_count == capacity &&
      !CandidateLess(score, node, candidate_scores[capacity - 1], candidate_nodes[capacity - 1])) {
    return;
  }
  std::int32_t position = *candidate_count < capacity ? (*candidate_count)++ : capacity - 1;
  while (position > 0 && CandidateLess(score, node, candidate_scores[position - 1],
                                       candidate_nodes[position - 1])) {
    candidate_scores[position] = candidate_scores[position - 1];
    candidate_nodes[position] = candidate_nodes[position - 1];
    --position;
  }
  candidate_scores[position] = score;
  candidate_nodes[position] = node;
}

__global__ void GeometryCandidatesKernel(
    const std::int32_t dimension, const std::int32_t edge_count, const std::int32_t* const edge_u,
    const std::int32_t* const edge_v, const std::int64_t* const edge_weight,
    const std::uint8_t* const edge_active, const std::int32_t* const degree,
    const std::int64_t* const x, const std::int64_t* const y, const std::int64_t* const nearest,
    const std::int32_t potential_count, const std::int32_t witness_count, const double tolerance,
    std::int32_t* const first_witness, std::int32_t* const second_witness) {
  const std::int32_t edge_id = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge_id >= edge_count || edge_active[edge_id] == 0U) {
    return;
  }
  const std::int32_t p = edge_u[edge_id];
  const std::int32_t q = edge_v[edge_id];
  if (degree[p] <= 2 || degree[q] <= 2) {
    return;
  }

  std::int32_t candidate_nodes[kMaxPotentialCandidates];
  double candidate_scores[kMaxPotentialCandidates];
  std::int32_t selected = 0;
  for (std::int32_t node = 0; node < dimension; ++node) {
    if (node == p || node == q) {
      continue;
    }
    // 乘 2 表达中点，避免先除法；这里只决定候选顺序，不参与证明。
    const double dx =
        2.0 * static_cast<double>(x[node]) - static_cast<double>(x[p]) - static_cast<double>(x[q]);
    const double dy =
        2.0 * static_cast<double>(y[node]) - static_cast<double>(y[p]) - static_cast<double>(y[q]);
    InsertMidpointCandidate(node, dx * dx + dy * dy, potential_count, candidate_nodes,
                            candidate_scores, &selected);
  }

  ApproxPotential potentials[kMaxPotentialCandidates];
  std::int32_t valid_count = 0;
  for (std::int32_t index = 0; index < selected; ++index) {
    ApproxPotential potential;
    if (ApproxPotentialBounds(p, q, candidate_nodes[index], edge_weight[edge_id],
                              nearest[candidate_nodes[index]], x, y, tolerance, &potential)) {
      potentials[valid_count++] = potential;
    }
  }

  std::int32_t emitted = 0;
  for (std::int32_t first = 0; first < valid_count && emitted < witness_count; ++first) {
    for (std::int32_t second = first + 1; second < valid_count && emitted < witness_count;
         ++second) {
      const std::int32_t r = potentials[first].node;
      const std::int32_t s = potentials[second].node;
      const std::int64_t original = edge_weight[edge_id] + RoundedEucDistance(r, s, x, y);
      if (RoundedEucDistance(p, r, x, y) + RoundedEucDistance(q, s, x, y) >= original ||
          RoundedEucDistance(p, s, x, y) + RoundedEucDistance(q, r, x, y) >= original) {
        continue;
      }
      const double lrs = static_cast<double>(RoundedEucDistance(r, s, x, y));
      const double lpq = static_cast<double>(edge_weight[edge_id]);
      const double first_bound = lpq + potentials[second].min_p + potentials[first].min_q - lrs;
      const double second_bound = lpq + potentials[first].min_p + potentials[second].min_q - lrs;
      if (first_bound > -tolerance && second_bound > -tolerance) {
        const std::int64_t slot = static_cast<std::int64_t>(edge_id) * witness_count + emitted++;
        first_witness[slot] = r;
        second_witness[slot] = s;
      }
    }
  }
}

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
  if (selected >= count) {
    if (reason != nullptr) {
      *reason = "FGPU geometry device ordinal 超出可见范围";
    }
    return -1;
  }
  if (selected < 0) {
    std::size_t most_free = 0U;
    for (int device = 0; device < count; ++device) {
      if (cudaSetDevice(device) != cudaSuccess) {
        continue;
      }
      std::size_t free_bytes = 0U;
      std::size_t total_bytes = 0U;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
          (selected < 0 || free_bytes > most_free)) {
        selected = device;
        most_free = free_bytes;
      }
    }
  }
  if (selected < 0 || cudaSetDevice(selected) != cudaSuccess) {
    if (reason != nullptr) {
      *reason = "无法激活 FGPU geometry CUDA device";
    }
    return -1;
  }
  return selected;
}

} // namespace

bool GeometryCudaBackendAvailable(std::string* const reason) {
  return SelectDevice(-1, reason) >= 0;
}

GeometryProposalBatch FindGeometryCandidatesCuda(const GraphSnapshot& graph,
                                                 const GeometryOptions& options) {
  if (graph.edges.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("FGPU geometry 边数超过 int32 kernel 范围");
  }
  if (options.potential_candidates < 2U || options.potential_candidates > 32U ||
      options.witnesses_per_edge == 0U || options.witnesses_per_edge > 8U) {
    throw std::invalid_argument("FGPU geometry kernel 参数越界");
  }
  std::string reason;
  const int device = SelectDevice(options.device, &reason);
  if (device < 0) {
    throw std::runtime_error("CUDA geometry 设备不可用: " + reason);
  }

  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  std::vector<std::int32_t> host_u(graph.edges.size());
  std::vector<std::int32_t> host_v(graph.edges.size());
  std::vector<std::int64_t> host_weight(graph.edges.size());
  std::vector<std::uint8_t> host_active(graph.edges.size());
  std::vector<std::int32_t> host_degree(static_cast<std::size_t>(graph.dimension));
  for (std::size_t index = 0; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }
  for (std::size_t index = 0; index < graph.edges.size(); ++index) {
    host_u[index] = graph.edges[index].u;
    host_v[index] = graph.edges[index].v;
    host_weight[index] = graph.edges[index].weight;
    host_active[index] = graph.edges[index].active ? 1U : 0U;
  }
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    host_degree[static_cast<std::size_t>(node)] = graph.Degree(node);
  }

  const std::size_t output_count = graph.edges.size() * options.witnesses_per_edge;
  GeometryProposalBatch result;
  result.witnesses_per_edge = options.witnesses_per_edge;
  result.first_witness.assign(output_count, -1);
  result.second_witness.assign(output_count, -1);
  result.backend = "cuda-filter+cpu-mpfr";
  result.selected_device = device;

  DeviceBuffer<std::int64_t> device_x(host_x.size(), device);
  DeviceBuffer<std::int64_t> device_y(host_y.size(), device);
  DeviceBuffer<std::int32_t> device_u(host_u.size(), device);
  DeviceBuffer<std::int32_t> device_v(host_v.size(), device);
  DeviceBuffer<std::int64_t> device_weight(host_weight.size(), device);
  DeviceBuffer<std::uint8_t> device_active(host_active.size(), device);
  DeviceBuffer<std::int32_t> device_degree(host_degree.size(), device);
  DeviceBuffer<std::int64_t> device_nearest(graph.points.size(), device);
  DeviceBuffer<std::int32_t> device_first(output_count, device);
  DeviceBuffer<std::int32_t> device_second(output_count, device);

  const auto upload_begin = std::chrono::steady_clock::now();
  device_x.CopyFromHost(host_x.data(), host_x.size());
  device_y.CopyFromHost(host_y.data(), host_y.size());
  device_u.CopyFromHost(host_u.data(), host_u.size());
  device_v.CopyFromHost(host_v.data(), host_v.size());
  device_weight.CopyFromHost(host_weight.data(), host_weight.size());
  device_active.CopyFromHost(host_active.data(), host_active.size());
  device_degree.CopyFromHost(host_degree.data(), host_degree.size());
  device_first.CopyFromHost(result.first_witness.data(), result.first_witness.size());
  device_second.CopyFromHost(result.second_witness.data(), result.second_witness.size());
  result.upload_ms = ElapsedMilliseconds(upload_begin);

  constexpr int kThreads = 128;
  const int nearest_blocks = (graph.dimension + kThreads - 1) / kThreads;
  const auto nearest_begin = std::chrono::steady_clock::now();
  NearestRoundedDistanceKernel<<<nearest_blocks, kThreads>>>(graph.dimension, device_x.get(),
                                                             device_y.get(), device_nearest.get());
  CheckCuda(cudaGetLastError(), "NearestRoundedDistanceKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "NearestRoundedDistanceKernel synchronize");
  result.nearest_ms = ElapsedMilliseconds(nearest_begin);

  if (!graph.edges.empty()) {
    const int blocks = (static_cast<std::int32_t>(graph.edges.size()) + kThreads - 1) / kThreads;
    double tolerance = 1.0e-10;
    if (options.numeric_mode == NumericMode::kFp64) {
      tolerance = 1.0e-12;
    } else if (options.numeric_mode == NumericMode::kAggressiveFp32) {
      tolerance = 1.0e-5;
    }
    const auto kernel_begin = std::chrono::steady_clock::now();
    GeometryCandidatesKernel<<<blocks, kThreads>>>(
        graph.dimension, static_cast<std::int32_t>(graph.edges.size()), device_u.get(),
        device_v.get(), device_weight.get(), device_active.get(), device_degree.get(),
        device_x.get(), device_y.get(), device_nearest.get(),
        static_cast<std::int32_t>(options.potential_candidates),
        static_cast<std::int32_t>(options.witnesses_per_edge), tolerance, device_first.get(),
        device_second.get());
    CheckCuda(cudaGetLastError(), "GeometryCandidatesKernel launch");
    CheckCuda(cudaDeviceSynchronize(), "GeometryCandidatesKernel synchronize");
    result.kernel_ms = ElapsedMilliseconds(kernel_begin);
  }

  const auto download_begin = std::chrono::steady_clock::now();
  device_first.CopyToHost(result.first_witness.data(), result.first_witness.size());
  device_second.CopyToHost(result.second_witness.data(), result.second_witness.size());
  result.download_ms = ElapsedMilliseconds(download_begin);
  return result;
}

} // namespace cudaee::detail
