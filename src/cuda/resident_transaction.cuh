#pragma once

#include "../fgpu/resident_transaction.hpp"

#include <cuda_runtime.h>

namespace cudaee::detail::resident_transaction {

__global__ void PendingDegreeKernel(const quick_hs::GraphView graph, const PendingDelta delta,
                                    std::int32_t* const degree) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex >= graph.dimension) {
    return;
  }
  std::int32_t count = 0;
  for (std::int64_t slot = quick_hs::NeighborBegin(graph, vertex);
       slot < quick_hs::NeighborEnd(graph, vertex); ++slot) {
    count += Survives(graph, delta, graph.neighbor_edge_ids[slot]) ? 1 : 0;
  }
  degree[vertex] = count;
}

__global__ void PendingFixedKernel(const std::int32_t count, const std::int32_t* const ids,
                                   const quick_hs::GraphView graph, const PendingDelta delta,
                                   const std::int32_t* const degree, std::uint8_t* const fixed) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= count) {
    return;
  }
  const std::int32_t edge = ids[work];
  // degree-2 fixing 是已 replay 的删边集合的直接推论，同样纳入提交前
  // 的候选终态检查，不先写 live fixed mask。
  fixed[edge] = Survives(graph, delta, edge) &&
                        ((graph.fixed_edge != nullptr && graph.fixed_edge[edge] != 0U) ||
                         (delta.fixed != nullptr && delta.fixed[edge] != 0U) ||
                         degree[graph.edge_u[edge]] == 2 || degree[graph.edge_v[edge]] == 2)
                    ? 1U
                    : 0U;
}

__global__ void ValidateVerticesKernel(const quick_hs::GraphView graph, const PendingDelta delta,
                                       const std::uint8_t* const fixed, std::int32_t* const invalid,
                                       std::int32_t* const parent) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex < graph.dimension) {
    const std::int32_t status = ValidateVertex(graph, delta, fixed, vertex);
    if (status != kValid) {
      atomicOr(invalid, status);
    }
    parent[vertex] = vertex;
  }
}

// union 的父指针单调减小。所有并发读取也使用原子操作，不能把普通
// load 与 atomicCAS 混合后依赖偶然的可见性。
__device__ inline std::int32_t AtomicRoot(std::int32_t* const parent, std::int32_t vertex) {
  for (;;) {
    const std::int32_t next = atomicAdd(parent + vertex, 0);
    if (next == vertex) {
      return vertex;
    }
    vertex = next;
  }
}

__global__ void UnionFixedKernel(const std::int32_t count, const std::int32_t* const ids,
                                 const quick_hs::GraphView graph, const std::uint8_t* const fixed,
                                 std::int32_t* const parent) {
  const std::int32_t work = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= count || fixed[ids[work]] == 0U) {
    return;
  }
  const std::int32_t edge = ids[work];
  std::int32_t a = graph.edge_u[edge];
  std::int32_t b = graph.edge_v[edge];
  for (;;) {
    a = AtomicRoot(parent, a);
    b = AtomicRoot(parent, b);
    if (a == b) {
      return;
    }
    const std::int32_t low = a < b ? a : b;
    const std::int32_t high = a < b ? b : a;
    if (atomicCAS(parent + high, high, low) == high) {
      return;
    }
  }
}

__global__ void CountFixedComponentsKernel(const quick_hs::GraphView graph,
                                           const std::uint8_t* const fixed,
                                           const std::int32_t* const parent,
                                           std::int32_t* const sizes,
                                           std::int32_t* const degree_sums) {
  const std::int32_t vertex = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (vertex >= graph.dimension) {
    return;
  }
  std::int32_t root = vertex;
  while (parent[root] != root) {
    root = parent[root];
  }
  std::int32_t degree = 0;
  for (std::int64_t slot = quick_hs::NeighborBegin(graph, vertex);
       slot < quick_hs::NeighborEnd(graph, vertex); ++slot) {
    const std::int32_t edge = graph.neighbor_edge_ids[slot];
    degree += graph.edge_active[edge] != 0U && fixed[edge] != 0U ? 1 : 0;
  }
  atomicAdd(sizes + root, 1);
  atomicAdd(degree_sums + root, degree);
}

__global__ void ValidateFixedComponentsKernel(const std::int32_t dimension,
                                              const std::int32_t* const sizes,
                                              const std::int32_t* const degree_sums,
                                              std::int32_t* const invalid) {
  const std::int32_t root = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (root < dimension && sizes[root] > 0 && sizes[root] < dimension &&
      static_cast<std::int64_t>(degree_sums[root]) == 2LL * sizes[root]) {
    atomicOr(invalid, kProperFixedCycle);
  }
}

} // namespace cudaee::detail::resident_transaction
