#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cudaee::detail::resident_sec_replay {

// 一个 SEC 的合法性只依赖非空真子集 S，不依赖寻找 S 的浮点算法。
// membership 为按 level 封存的 vertex->subset 标签，不能引用下一轮工作区。
static __global__ void CountMembersKernel(const std::int32_t dimension, const std::int32_t levels,
                                          const std::int32_t* membership, std::int32_t* sizes,
                                          std::int32_t* invalid) {
  const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<std::int64_t>(dimension) * levels) {
    return;
  }
  const std::int32_t label = membership[index];
  if (label < 0 || label >= dimension) {
    atomicExch(invalid, 1);
    return;
  }
  atomicAdd(sizes + (index / dimension) * dimension + label, 1);
}

static __global__ void ValidateCutsKernel(const std::int32_t dimension,
                                          const std::int32_t static_cuts, const std::int32_t cuts,
                                          const std::int32_t* membership, const std::int32_t* sizes,
                                          const std::uint8_t* valid, const std::int64_t* numerator,
                                          std::int32_t* invalid) {
  const std::int64_t cut = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cut >= cuts) {
    return;
  }
  bool expected = true;
  if (cut >= static_cuts) {
    const auto entry = cut - static_cuts;
    expected = sizes[entry] >= 2 && sizes[entry] <= dimension - 2 &&
               membership[entry] == entry % dimension;
  }
  if (valid[cut] != static_cast<std::uint8_t>(expected) || numerator[cut] < 0 ||
      numerator[cut] > 1000000000000000LL || (!expected && numerator[cut] != 0)) {
    atomicExch(invalid, 1);
  }
}

__device__ inline bool ContainsOnce(const std::int32_t* ids, const std::int32_t count,
                                    const std::int32_t cut) {
  int found = 0;
  for (int index = 0; index < count; ++index) {
    found += ids[index] == cut ? 1 : 0;
  }
  return found == 1;
}

// 与 proposal 的端点窗口插入/去重不同：replay 扫描全部静态窗口，
// 按集合异或重新推导非零系数，再逐项检查唯一性及总 nnz，无哈希假设。
template <typename Layout>
static __global__ void ValidateIncidenceKernel(
    const std::int32_t count, const std::int32_t* edge_ids, const std::int32_t* edge_u,
    const std::int32_t* edge_v, const std::int32_t* geometry_rank, const Layout layout,
    const std::int32_t families, const std::int32_t levels, const std::int32_t stride,
    const std::int32_t* membership, const std::uint8_t* valid, const std::uint8_t* incidence_count,
    const std::int32_t* incidence_ids, std::int32_t* invalid) {
  const std::int64_t work = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (work >= count) {
    return;
  }
  const std::int32_t edge = edge_ids[work];
  const std::int32_t u = edge_u[edge], v = edge_v[edge];
  const std::int32_t ru = geometry_rank[u], rv = geometry_rank[v];
  const std::int32_t actual = incidence_count[edge];
  if (actual > stride || ru < 0 || rv < 0 || ru >= layout.dimension || rv >= layout.dimension) {
    atomicExch(invalid, 1);
    return;
  }
  const auto* ids = incidence_ids + static_cast<std::int64_t>(edge) * stride;
  std::int32_t expected = 0;
  bool correct = true;
  for (int family = 0; family < families; ++family) {
    for (int cut = layout.offset[family]; cut < layout.offset[family + 1]; ++cut) {
      const std::int64_t begin =
          static_cast<std::int64_t>(cut - layout.offset[family]) * layout.stride[family];
      const std::int64_t end = begin + layout.size[family];
      if (begin < 0 || end > layout.dimension || end - begin < 2 ||
          end - begin > layout.dimension - 2) {
        correct = false;
        continue;
      }
      if ((ru >= begin && ru < end) != (rv >= begin && rv < end)) {
        ++expected;
        correct = correct && ContainsOnce(ids, actual, cut);
      }
    }
  }
  for (int level = 0; level < levels; ++level) {
    const std::int64_t base = static_cast<std::int64_t>(level) * layout.dimension;
    const auto first = membership[base + u], second = membership[base + v];
    if (first < 0 || second < 0 || first >= layout.dimension || second >= layout.dimension) {
      correct = false;
      continue;
    }
    if (first == second) {
      continue;
    }
    const std::int32_t cuts[2] = {static_cast<std::int32_t>(layout.cut_count + base + first),
                                  static_cast<std::int32_t>(layout.cut_count + base + second)};
    for (const auto cut : cuts) {
      if (valid[cut] != 0U) {
        ++expected;
        correct = correct && ContainsOnce(ids, actual, cut);
      }
    }
  }
  if (!correct || expected != actual) {
    atomicExch(invalid, 1);
  }
}

} // namespace cudaee::detail::resident_sec_replay
