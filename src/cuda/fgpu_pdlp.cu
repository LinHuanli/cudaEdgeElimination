#include "../fgpu/pdlp_backend.hpp"

#include "cuda_edge_elimination/cuda_device_affinity.hpp"

#include <cooperative_groups.h>
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

namespace cg = cooperative_groups;

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
  DeviceBuffer(const std::size_t count, const int device) : count_(count), device_(device) {
    if (count_ != 0U) {
      CheckCuda(cudaSetDevice(device_), "cudaSetDevice(PDLP allocation)");
      CheckCuda(cudaMalloc(&data_, count_ * sizeof(T)), "cudaMalloc(PDLP)");
    }
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      static_cast<void>(cudaSetDevice(device_));
      static_cast<void>(cudaFree(data_));
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] T* get() { return data_; }
  [[nodiscard]] const T* get() const { return data_; }

private:
  T* data_{nullptr};
  std::size_t count_{};
  int device_{-1};
};

__global__ void
SelectBoxMinimizerKernel(const std::int32_t edge_count, const std::int32_t* const edge_u,
                         const std::int32_t* const edge_v, const std::int64_t* const edge_cost,
                         const double* const dual, std::int32_t* const selected_degree) {
  const std::int32_t edge = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (edge >= edge_count) {
    return;
  }
  const std::int32_t u = edge_u[edge];
  const std::int32_t v = edge_v[edge];
  const double reduced = static_cast<double>(edge_cost[edge]) - dual[u] - dual[v];
  if (reduced < 0.0) {
    atomicAdd(&selected_degree[u], 1);
    atomicAdd(&selected_degree[v], 1);
  }
}

__global__ void UpdateDualKernel(const std::int32_t dimension,
                                 const std::int32_t* const selected_degree, const double step,
                                 double* const dual, double* const average) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex >= dimension) {
    return;
  }
  // L(pi)=2*pi + min_x (c-B^T*pi)x，故 2-degree(x) 是合法超梯度。
  const double gradient = 2.0 - static_cast<double>(selected_degree[vertex]);
  dual[vertex] = fmin(1.0e12, fmax(-1.0e12, dual[vertex] + step * gradient));
  average[vertex] += dual[vertex];
}

__global__ void FinalizeAverageKernel(const std::int32_t dimension, const double inverse_iterations,
                                      double* const average) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex < dimension) {
    average[vertex] *= inverse_iterations;
  }
}

// 每个顶点独立扫描 CSR 邻边，避免逐边 atomicAdd；cooperative grid barrier
// 保证一轮读取的 dual 与下一轮写回严格分离。整个迭代只启动一个 kernel，
// 对中小稀疏图可显著削减数千次 host launch 的固定开销。
__global__ void
PersistentDegreeDualKernel(const std::int32_t dimension, const std::int32_t* const row_offsets,
                           const std::int32_t* const neighbors, const std::int64_t* const arc_cost,
                           const std::uint32_t iterations, const double initial_step,
                           std::int32_t* const selected_degree, double* const dual,
                           double* const average) {
  const cg::grid_group grid = cg::this_grid();
  const std::int32_t thread = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t stride = static_cast<std::int32_t>(gridDim.x * blockDim.x);
  for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
    for (std::int32_t vertex = thread; vertex < dimension; vertex += stride) {
      const double vertex_dual = dual[vertex];
      std::int32_t degree = 0;
      for (std::int32_t arc = row_offsets[vertex]; arc < row_offsets[vertex + 1]; ++arc) {
        const std::int32_t neighbor = neighbors[arc];
        const double reduced = static_cast<double>(arc_cost[arc]) - vertex_dual - dual[neighbor];
        degree += reduced < 0.0 ? 1 : 0;
      }
      selected_degree[vertex] = degree;
    }
    grid.sync();

    const double step = initial_step / sqrt(1.0 + static_cast<double>(iteration) / 8.0);
    for (std::int32_t vertex = thread; vertex < dimension; vertex += stride) {
      const double gradient = 2.0 - static_cast<double>(selected_degree[vertex]);
      dual[vertex] = fmin(1.0e12, fmax(-1.0e12, dual[vertex] + step * gradient));
      average[vertex] += dual[vertex];
    }
    // 防止下一轮邻接扫描读到同一轮尚未完成的写回。
    grid.sync();
  }
  for (std::int32_t vertex = thread; vertex < dimension; vertex += stride) {
    average[vertex] /= static_cast<double>(iterations);
  }
}

int SelectDevice(const int requested, std::string* const reason) {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    if (reason != nullptr) {
      *reason = status == cudaSuccess ? "没有可见 CUDA 设备" : cudaGetErrorString(status);
    }
    return -1;
  }
  int selected = requested;
  if (selected < 0) {
    selected = CudaDevicePreferenceForCurrentThread();
  }
  if (selected >= count) {
    if (reason != nullptr) {
      *reason = "native PDLP device ordinal 超出可见范围";
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
      *reason = "无法激活 native PDLP CUDA device";
    }
    return -1;
  }
  return selected;
}

} // namespace

bool NativePdlpCudaAvailable(std::string* const reason) { return SelectDevice(-1, reason) >= 0; }

NativePdlpDeviceResult SolveDegreeRelaxationCuda(const GraphSnapshot& graph,
                                                 const PdlpOptions& options) {
  if (options.iterations == 0U || graph.dimension <= 0 || graph.ActiveEdgeCount() == 0U) {
    throw std::invalid_argument("native PDLP 输入图或迭代数非法");
  }
  std::string reason;
  const int device = SelectDevice(options.device, &reason);
  if (device < 0) {
    throw std::runtime_error("native PDLP CUDA 设备不可用: " + reason);
  }

  std::vector<std::int32_t> edge_u;
  std::vector<std::int32_t> edge_v;
  std::vector<std::int64_t> edge_cost;
  edge_u.reserve(graph.ActiveEdgeCount());
  edge_v.reserve(graph.ActiveEdgeCount());
  edge_cost.reserve(graph.ActiveEdgeCount());
  long double cost_sum = 0.0L;
  for (const Edge& edge : graph.edges) {
    if (edge.active) {
      edge_u.push_back(edge.u);
      edge_v.push_back(edge.v);
      edge_cost.push_back(edge.weight);
      cost_sum += edge.weight;
    }
  }
  if (edge_u.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error("native PDLP 活动边超过 int32 kernel 范围");
  }

  DeviceBuffer<std::int32_t> device_u(edge_u.size(), device);
  DeviceBuffer<std::int32_t> device_v(edge_v.size(), device);
  DeviceBuffer<std::int64_t> device_cost(edge_cost.size(), device);
  DeviceBuffer<std::int32_t> device_selected(static_cast<std::size_t>(graph.dimension), device);
  DeviceBuffer<double> device_dual(static_cast<std::size_t>(graph.dimension), device);
  DeviceBuffer<double> device_average(static_cast<std::size_t>(graph.dimension), device);
  CheckCuda(cudaMemcpy(device_u.get(), edge_u.data(), edge_u.size() * sizeof(std::int32_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy PDLP u");
  CheckCuda(cudaMemcpy(device_v.get(), edge_v.data(), edge_v.size() * sizeof(std::int32_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy PDLP v");
  CheckCuda(cudaMemcpy(device_cost.get(), edge_cost.data(), edge_cost.size() * sizeof(std::int64_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy PDLP cost");
  CheckCuda(
      cudaMemset(device_dual.get(), 0, static_cast<std::size_t>(graph.dimension) * sizeof(double)),
      "cudaMemset PDLP dual");
  CheckCuda(cudaMemset(device_average.get(), 0,
                       static_cast<std::size_t>(graph.dimension) * sizeof(double)),
            "cudaMemset PDLP average");

  constexpr int kThreads = 256;
  const int edge_blocks = (static_cast<std::int32_t>(edge_u.size()) + kThreads - 1) / kThreads;
  const int vertex_blocks = (graph.dimension + kThreads - 1) / kThreads;
  const long double average_cost = cost_sum / edge_u.size();
  const long double average_degree = 2.0L * edge_u.size() / graph.dimension;
  const double initial_step =
      static_cast<double>(std::max<long double>(1.0L, average_cost / (average_degree + 2.0L)));
  const auto begin = std::chrono::steady_clock::now();
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties(PDLP)");
  bool used_persistent = false;
  // RTX 4000 Ada 实测：极小图由一次 persistent launch 获益；pcb442 上每轮
  // cooperative grid barrier 反而比两次普通 kernel launch 更贵。只在小图
  // 启用，较大图保持吞吐更高的逐边 atomic 路径。
  constexpr std::int32_t kPersistentDimensionLimit = 64;
  if (properties.cooperativeLaunch != 0 && graph.dimension <= kPersistentDimensionLimit) {
    int blocks_per_sm = 0;
    CheckCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks_per_sm, PersistentDegreeDualKernel, kThreads, 0U),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor(PDLP)");
    const int resident_blocks = blocks_per_sm * properties.multiProcessorCount;
    const int requested_blocks = std::max(1, vertex_blocks);
    const int cooperative_blocks = std::min(requested_blocks, resident_blocks);
    if (cooperative_blocks > 0) {
      DeviceBuffer<std::int32_t> device_row_offsets(graph.row_offsets.size(), device);
      DeviceBuffer<std::int32_t> device_neighbors(graph.neighbors.size(), device);
      DeviceBuffer<std::int64_t> device_arc_cost(graph.csr_weights.size(), device);
      CheckCuda(cudaMemcpy(device_row_offsets.get(), graph.row_offsets.data(),
                           graph.row_offsets.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                "cudaMemcpy PDLP row offsets");
      CheckCuda(cudaMemcpy(device_neighbors.get(), graph.neighbors.data(),
                           graph.neighbors.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                "cudaMemcpy PDLP neighbors");
      CheckCuda(cudaMemcpy(device_arc_cost.get(), graph.csr_weights.data(),
                           graph.csr_weights.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice),
                "cudaMemcpy PDLP arc costs");
      std::int32_t dimension = graph.dimension;
      std::uint32_t iterations = options.iterations;
      double step_argument = initial_step;
      std::int32_t* row_offsets_argument = device_row_offsets.get();
      std::int32_t* neighbors_argument = device_neighbors.get();
      std::int64_t* arc_cost_argument = device_arc_cost.get();
      std::int32_t* selected_argument = device_selected.get();
      double* dual_argument = device_dual.get();
      double* average_argument = device_average.get();
      void* arguments[] = {
          &dimension,         &row_offsets_argument, &neighbors_argument,
          &arc_cost_argument, &iterations,           &step_argument,
          &selected_argument, &dual_argument,        &average_argument,
      };
      CheckCuda(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(PersistentDegreeDualKernel),
                                            cooperative_blocks, kThreads, arguments, 0U, nullptr),
                "cudaLaunchCooperativeKernel(PDLP)");
      used_persistent = true;
      CheckCuda(cudaGetLastError(), "native persistent PDLP kernel launch");
      CheckCuda(cudaDeviceSynchronize(), "native persistent PDLP synchronize");
    }
  }
  if (!used_persistent) {
    for (std::uint32_t iteration = 0U; iteration < options.iterations; ++iteration) {
      CheckCuda(cudaMemset(device_selected.get(), 0,
                           static_cast<std::size_t>(graph.dimension) * sizeof(std::int32_t)),
                "cudaMemset PDLP selected degree");
      SelectBoxMinimizerKernel<<<edge_blocks, kThreads>>>(
          static_cast<std::int32_t>(edge_u.size()), device_u.get(), device_v.get(),
          device_cost.get(), device_dual.get(), device_selected.get());
      const double step = initial_step / std::sqrt(1.0 + static_cast<double>(iteration) / 8.0);
      UpdateDualKernel<<<vertex_blocks, kThreads>>>(graph.dimension, device_selected.get(), step,
                                                    device_dual.get(), device_average.get());
    }
    FinalizeAverageKernel<<<vertex_blocks, kThreads>>>(
        graph.dimension, 1.0 / static_cast<double>(options.iterations), device_average.get());
    CheckCuda(cudaGetLastError(), "native atomic PDLP kernel launch");
    CheckCuda(cudaDeviceSynchronize(), "native atomic PDLP synchronize");
  }

  NativePdlpDeviceResult result;
  result.vertex_dual.resize(static_cast<std::size_t>(graph.dimension));
  CheckCuda(cudaMemcpy(result.vertex_dual.data(), device_average.get(),
                       result.vertex_dual.size() * sizeof(double), cudaMemcpyDeviceToHost),
            "cudaMemcpy PDLP dual D2H");
  result.selected_device = device;
  result.iterations = options.iterations;
  result.solve_ms = ElapsedMilliseconds(begin);
  result.implementation = used_persistent ? "persistent-csr" : "edge-atomic";
  return result;
}

} // namespace cudaee::detail
