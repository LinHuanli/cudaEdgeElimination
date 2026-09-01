#include "cuda_edge_elimination/elimination.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (count_ != 0) {
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc");
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
  [[nodiscard]] std::size_t size() const { return count_; }

  void CopyFromHost(const T* source) {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count_, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D");
    }
  }

  void CopyToHost(T* destination) const {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H");
    }
  }

private:
  T* data_{nullptr};
  std::size_t count_{};
};

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
                   const std::int32_t* const neighbors, const std::int64_t* const csr_weights,
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
          static_cast<std::uint64_t>(csr_weights[offset]) +
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
      const std::uint64_t left = cab + static_cast<std::uint64_t>(csr_weights[offset]);
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
  return best_device;
}

} // namespace

bool CudaBackendAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<Candidate> FindJvCandidatesCuda(const GraphSnapshot& graph,
                                            int* const selected_device) {
  if (!graph.integer_coordinates || !graph.integer_distance_safe) {
    throw std::runtime_error("CUDA JV 仅支持平方距离不溢出的整数坐标");
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

  std::vector<std::int32_t> edge_u(graph.edges.size());
  std::vector<std::int32_t> edge_v(graph.edges.size());
  std::vector<std::int64_t> edge_weight(graph.edges.size());
  std::vector<std::int32_t> edge_active(graph.edges.size());
  for (std::size_t i = 0; i < graph.edges.size(); ++i) {
    edge_u[i] = graph.edges[i].u;
    edge_v[i] = graph.edges[i].v;
    edge_weight[i] = graph.edges[i].weight;
    edge_active[i] = graph.edges[i].active ? 1 : 0;
  }
  std::vector<std::int64_t> x(graph.points.size());
  std::vector<std::int64_t> y(graph.points.size());
  for (std::size_t i = 0; i < graph.points.size(); ++i) {
    x[i] = graph.points[i].integer_x;
    y[i] = graph.points[i].integer_y;
  }
  std::vector<std::int32_t> witnesses(graph.edges.size(), -1);

  DeviceBuffer<std::int32_t> d_edge_u(edge_u.size());
  DeviceBuffer<std::int32_t> d_edge_v(edge_v.size());
  DeviceBuffer<std::int64_t> d_edge_weight(edge_weight.size());
  DeviceBuffer<std::int32_t> d_edge_active(edge_active.size());
  DeviceBuffer<std::int32_t> d_row_offsets(graph.row_offsets.size());
  DeviceBuffer<std::int32_t> d_neighbors(graph.neighbors.size());
  DeviceBuffer<std::int64_t> d_csr_weights(graph.csr_weights.size());
  DeviceBuffer<std::int64_t> d_x(x.size());
  DeviceBuffer<std::int64_t> d_y(y.size());
  DeviceBuffer<std::int32_t> d_witnesses(witnesses.size());
  d_edge_u.CopyFromHost(edge_u.data());
  d_edge_v.CopyFromHost(edge_v.data());
  d_edge_weight.CopyFromHost(edge_weight.data());
  d_edge_active.CopyFromHost(edge_active.data());
  d_row_offsets.CopyFromHost(graph.row_offsets.data());
  d_neighbors.CopyFromHost(graph.neighbors.data());
  d_csr_weights.CopyFromHost(graph.csr_weights.data());
  d_x.CopyFromHost(x.data());
  d_y.CopyFromHost(y.data());
  d_witnesses.CopyFromHost(witnesses.data());

  constexpr int kThreads = 128;
  const int blocks = (static_cast<int>(graph.edges.size()) + kThreads - 1) / kThreads;
  JvCandidatesKernel<<<blocks, kThreads>>>(
      static_cast<std::int32_t>(graph.edges.size()), d_edge_u.get(), d_edge_v.get(),
      d_edge_weight.get(), d_edge_active.get(), d_row_offsets.get(), d_neighbors.get(),
      d_csr_weights.get(), d_x.get(), d_y.get(), static_cast<std::uint8_t>(graph.distance_type),
      d_witnesses.get());
  CheckCuda(cudaGetLastError(), "JvCandidatesKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "JvCandidatesKernel synchronize");
  d_witnesses.CopyToHost(witnesses.data());

  std::vector<Candidate> candidates;
  for (std::size_t edge_id = 0; edge_id < witnesses.size(); ++edge_id) {
    if (witnesses[edge_id] >= 0) {
      candidates.push_back(
          {static_cast<std::int32_t>(edge_id), witnesses[edge_id], EliminationMethod::kJv});
    }
  }
  return candidates;
}

} // namespace cudaee
