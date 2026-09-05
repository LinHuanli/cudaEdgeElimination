#include "../../src/cuda/resident_transaction.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tx = cudaee::detail::resident_transaction;

namespace {
void CudaCheck(const cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

class Arena {
public:
  ~Arena() {
    for (void* pointer : pointers_) {
      static_cast<void>(cudaFree(pointer));
    }
  }
  template <typename T> T* Copy(const std::vector<T>& input) {
    T* output = nullptr;
    CudaCheck(cudaMalloc(&output, input.size() * sizeof(T)));
    pointers_.push_back(output);
    CudaCheck(cudaMemcpy(output, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    return output;
  }

private:
  std::vector<void*> pointers_;
};
} // namespace

int main() {
  constexpr std::int32_t n = 6;
  constexpr std::int32_t m = n * (n - 1) / 2;
  int devices = 0;
  CudaCheck(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    std::cout << "SKIP: no CUDA device\n";
    return 77;
  }
  CudaCheck(cudaSetDevice(0));
  std::vector<std::int32_t> u, v, ids, neighbors, arcs;
  std::vector<std::int64_t> rows{0}, pairs{0};
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = a + 1; b < n; ++b) {
      ids.push_back(static_cast<std::int32_t>(u.size()));
      u.push_back(a);
      v.push_back(b);
    }
  }
  const auto edge_id = [&](const std::int32_t a, const std::int32_t b) {
    for (std::int32_t edge = 0; edge < m; ++edge) {
      if (u[edge] == std::min(a, b) && v[edge] == std::max(a, b)) {
        return edge;
      }
    }
    throw std::logic_error("bad test edge");
  };
  for (std::int32_t a = 0; a < n; ++a) {
    for (std::int32_t b = 0; b < n; ++b) {
      if (a != b) {
        neighbors.push_back(b);
        arcs.push_back(edge_id(a, b));
      }
    }
    rows.push_back(static_cast<std::int64_t>(arcs.size()));
    pairs.push_back(pairs.back() + (n - 1) * (n - 2) / 2);
  }
  for (std::int32_t sample = 0; sample < 8; ++sample) {
    Arena arena;
    std::vector<std::uint8_t> active(m, 1U), fixed(m, 0U), deleted(m, 0U), proposed(m, 0U);
    std::vector<std::uint8_t> nonpair(static_cast<std::size_t>(pairs.back()), 0U);
    std::vector<std::uint8_t> proposed_nonpair(nonpair.size(), 0U);
    std::int32_t expected = tx::kValid;
    if (sample == 1) {
      proposed[edge_id(0, 1)] = proposed[edge_id(0, 2)] = proposed[edge_id(1, 2)] = 1U;
      expected = tx::kProperFixedCycle;
    } else if (sample == 2) {
      for (std::int32_t a = 0; a < n; ++a) {
        proposed[edge_id(a, (a + 1) % n)] = 1U;
      }
    } else if (sample == 3) {
      proposed[edge_id(0, 1)] = proposed[edge_id(0, 2)] = proposed[edge_id(0, 3)] = 1U;
      expected = tx::kFixedDegree;
    } else if (sample == 4) {
      proposed[edge_id(0, 1)] = deleted[edge_id(0, 1)] = 1U;
      expected = tx::kDeleteFixedConflict;
    } else if (sample == 5) {
      for (std::int32_t b = 2; b < n; ++b) {
        deleted[edge_id(0, b)] = 1U;
      }
      expected = tx::kDegreeFloor;
    } else if (sample == 6) {
      proposed[edge_id(0, 1)] = proposed[edge_id(0, 2)] = 1U;
      proposed_nonpair[0] = 1U;
      expected = tx::kNoAllowedPair;
    } else if (sample == 7) {
      // degree-2 fixing 推导出的整个 Hamilton 环是合法终态，不能误判
      // 为“不允许路径形成任何环”。
      deleted.assign(m, 1U);
      for (std::int32_t a = 0; a < n; ++a) {
        deleted[edge_id(a, (a + 1) % n)] = 0U;
      }
    }
    cudaee::detail::quick_hs::GraphView graph{};
    graph.dimension = n;
    graph.row_offsets = arena.Copy(rows);
    graph.neighbors = arena.Copy(neighbors);
    graph.neighbor_edge_ids = arena.Copy(arcs);
    graph.edge_u = arena.Copy(u);
    graph.edge_v = arena.Copy(v);
    graph.edge_active = arena.Copy(active);
    graph.fixed_edge = arena.Copy(fixed);
    graph.pair_offsets = arena.Copy(pairs);
    graph.nonpair_mask = arena.Copy(nonpair);
    const tx::PendingDelta delta{arena.Copy(deleted), arena.Copy(proposed),
                                 arena.Copy(proposed_nonpair)};
    auto* pending_fixed = arena.Copy(std::vector<std::uint8_t>(m, 0U));
    auto* pending_degree = arena.Copy(std::vector<std::int32_t>(n, 0));
    auto* parent = arena.Copy(std::vector<std::int32_t>(n, 0));
    auto* sizes = arena.Copy(std::vector<std::int32_t>(n, 0));
    auto* sums = arena.Copy(std::vector<std::int32_t>(n, 0));
    auto* invalid = arena.Copy(std::vector<std::int32_t>(1, 0));
    auto* device_ids = arena.Copy(ids);
    tx::PendingDegreeKernel<<<1, 32>>>(graph, delta, pending_degree);
    tx::PendingFixedKernel<<<1, 32>>>(m, device_ids, graph, delta, pending_degree, pending_fixed);
    tx::ValidateVerticesKernel<<<1, 32>>>(graph, delta, pending_fixed, invalid, parent);
    tx::UnionFixedKernel<<<1, 32>>>(m, device_ids, graph, pending_fixed, parent);
    tx::CountFixedComponentsKernel<<<1, 32>>>(graph, pending_fixed, parent, sizes, sums);
    tx::ValidateFixedComponentsKernel<<<1, 32>>>(n, sizes, sums, invalid);
    CudaCheck(cudaGetLastError());
    std::int32_t actual = -1;
    CudaCheck(cudaMemcpy(&actual, invalid, sizeof(actual), cudaMemcpyDeviceToHost));
    if (actual != expected) {
      throw std::runtime_error("GPU 事务门禁 case=" + std::to_string(sample) + " expected=" +
                               std::to_string(expected) + " actual=" + std::to_string(actual));
    }
    std::vector<std::uint8_t> live_fixed(m, 255U), live_active(m, 255U),
        live_pairs(nonpair.size(), 255U);
    CudaCheck(cudaMemcpy(live_fixed.data(), graph.fixed_edge, m, cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(live_active.data(), graph.edge_active, m, cudaMemcpyDeviceToHost));
    CudaCheck(
        cudaMemcpy(live_pairs.data(), graph.nonpair_mask, nonpair.size(), cudaMemcpyDeviceToHost));
    if (live_fixed != fixed || live_active != active || live_pairs != nonpair) {
      throw std::runtime_error("事务验证写入了 live snapshot");
    }
  }
  std::cout << "GPU resident transaction tests passed (8 cases)\n";
}
