#pragma once

#include "quick_hs_predicate.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define CUDAEE_MAIN_EDGE_HD __host__ __device__
#define CUDAEE_MAIN_EDGE_INLINE __forceinline__
#else
#define CUDAEE_MAIN_EDGE_HD
#define CUDAEE_MAIN_EDGE_INLINE inline
#endif

namespace cudaee::detail::main_edge {

constexpr std::int32_t kMaximumMetricPathNodes = 7;
constexpr std::int32_t kMaximumMetricPathDistances =
    kMaximumMetricPathNodes * (kMaximumMetricPathNodes - 1) / 2;

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE std::int32_t
CachedDistanceIndex(const std::int32_t node_count, std::int32_t first, std::int32_t second) {
  if (first > second) {
    const std::int32_t saved = first;
    first = second;
    second = saved;
  }
  return first * (2 * node_count - first - 1) / 2 + (second - first - 1);
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE std::int64_t
CachedDistance(const std::int64_t* const distances, const std::int32_t node_count,
               const std::int32_t first, const std::int32_t second) {
  return first == second ? 0 : distances[CachedDistanceIndex(node_count, first, second)];
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
CanUseCachedTwoPathOpt(const quick_hs::GraphView& graph, const std::int32_t* const nodes,
                       const std::int32_t first_path_size, const std::int32_t node_count) {
  for (std::int32_t first = 0; first < node_count; ++first) {
    for (std::int32_t second = first + 1; second < node_count; ++second) {
      if (nodes[first] == nodes[second]) {
        return false;
      }
    }
  }
  const std::int32_t endpoints[4] = {nodes[0], nodes[first_path_size - 1], nodes[first_path_size],
                                     nodes[node_count - 1]};
  for (std::int32_t first = 0; first < 4; ++first) {
    for (std::int32_t second = first + 1; second < 4; ++second) {
      if (quick_hs::Fixed(graph, endpoints[first], endpoints[second])) {
        return false;
      }
    }
  }
  return true;
}

// `quick_hs::PathOrderIsOpt` 的两路径专用等价实现。metric-excess 会对同一
// 6/7 点 path system 扫描很多排列；先缓存三角距离表，避免每条转移重复做
// 精确整数平方根。遇到共享节点或 fixed endpoint 时由调用者回退通用实现。
CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
CachedTwoPathOrderIsOpt(const quick_hs::GraphView& graph, const std::int32_t* const source_nodes,
                        const std::int32_t first_path_size, const std::int32_t node_count,
                        const bool reverse_second_path) {
  std::int32_t nodes[kMaximumMetricPathNodes]{};
  for (std::int32_t index = 0; index < first_path_size; ++index) {
    nodes[index] = source_nodes[index];
  }
  const std::int32_t second_size = node_count - first_path_size;
  for (std::int32_t index = 0; index < second_size; ++index) {
    nodes[first_path_size + index] =
        source_nodes[first_path_size + (reverse_second_path ? second_size - 1 - index : index)];
  }

  std::int64_t distances[kMaximumMetricPathDistances]{};
  for (std::int32_t first = 0; first < node_count; ++first) {
    for (std::int32_t second = first + 1; second < node_count; ++second) {
      distances[CachedDistanceIndex(node_count, first, second)] =
          quick_hs::Distance(graph, nodes[first], nodes[second]);
    }
  }
  const std::int64_t forced_cost = static_cast<std::int64_t>(INT_MIN) / (node_count - 1);
  std::int64_t original = CachedDistance(distances, node_count, 0, 1);
  for (std::int32_t position = 1; position < node_count - 1; ++position) {
    original += position == first_path_size - 1
                    ? forced_cost
                    : CachedDistance(distances, node_count, position, position + 1);
  }

  const std::int32_t dynamic_nodes = node_count - 2;
  std::uint8_t order[kMaximumMetricPathNodes - 2]{};
  for (std::int32_t index = 0; index < dynamic_nodes; ++index) {
    order[index] = static_cast<std::uint8_t>(index);
  }
  for (;;) {
    std::int64_t candidate = CachedDistance(distances, node_count, order[0] + 1, 0);
    for (std::int32_t position = 1; position < dynamic_nodes; ++position) {
      const std::int32_t previous = order[position - 1] + 1;
      const std::int32_t current = order[position] + 1;
      const std::int32_t difference = previous - current;
      const std::int32_t lower = previous < current ? previous : current;
      candidate += (difference == 1 || difference == -1) && lower == first_path_size - 1
                       ? forced_cost
                       : CachedDistance(distances, node_count, previous, current);
    }
    const std::int32_t last = order[dynamic_nodes - 1] + 1;
    const std::int32_t difference = last - (node_count - 1);
    const std::int32_t lower = last < node_count - 1 ? last : node_count - 1;
    candidate += (difference == 1 || difference == -1) && lower == first_path_size - 1
                     ? forced_cost
                     : CachedDistance(distances, node_count, last, node_count - 1);
    if (candidate < original) {
      return false;
    }

    std::int32_t pivot = dynamic_nodes - 2;
    while (pivot >= 0 && order[pivot] >= order[pivot + 1]) {
      --pivot;
    }
    if (pivot < 0) {
      break;
    }
    std::int32_t successor = dynamic_nodes - 1;
    while (order[successor] <= order[pivot]) {
      --successor;
    }
    const std::uint8_t saved = order[pivot];
    order[pivot] = order[successor];
    order[successor] = saved;
    for (std::int32_t left = pivot + 1, right = dynamic_nodes - 1; left < right; ++left, --right) {
      const std::uint8_t left_value = order[left];
      order[left] = order[right];
      order[right] = left_value;
    }
  }
  return true;
}

// Hougardy--Schroeder 2014 Step 2 的整数判定。下面的比较直接由
// 论文的 compatible/three-compatible/Main-Edge 不等式推导；保留
// 非严格/严格边界，避免浮点近似改变删边集合。
CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
Compatible(const quick_hs::GraphView& graph, const std::int32_t p, const std::int32_t q,
           const std::int32_t x, const std::int32_t y) {
  const std::int64_t pqxy = quick_hs::Distance(graph, p, q) + quick_hs::Distance(graph, x, y);
  return quick_hs::Distance(graph, p, x) + quick_hs::Distance(graph, q, y) >= pqxy ||
         quick_hs::Distance(graph, p, y) + quick_hs::Distance(graph, q, x) >= pqxy;
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
ThreeCompatible(const quick_hs::GraphView& graph, const std::int32_t p, const std::int32_t q,
                const std::int32_t x, const std::int32_t middle, const std::int32_t y) {
  if (p == middle || q == middle) {
    return true;
  }
  return quick_hs::Distance(graph, p, middle) + quick_hs::Distance(graph, q, middle) +
             quick_hs::Distance(graph, x, y) >=
         quick_hs::Distance(graph, p, q) + quick_hs::Distance(graph, middle, x) +
             quick_hs::Distance(graph, middle, y);
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
Opt24(const quick_hs::GraphView& graph, const std::int32_t a, const std::int32_t b,
      const std::int32_t c1, const std::int32_t c2, const std::int32_t c3, const std::int32_t c4) {
  const std::int32_t nodes[6] = {a, b, c1, c2, c3, c4};
  if (CanUseCachedTwoPathOpt(graph, nodes, 2, 6)) {
    return CachedTwoPathOrderIsOpt(graph, nodes, 2, 6, false) ||
           CachedTwoPathOrderIsOpt(graph, nodes, 2, 6, true);
  }
  const quick_hs::SmallPath paths[quick_hs::kMaxPathCount] = {
      {.size = 2, .node = {a, b}},
      {.size = 4, .node = {c1, c2, c3, c4}},
  };
  return quick_hs::Opt(graph, paths, 2);
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool Opt34(const quick_hs::GraphView& graph,
                                                       const std::int32_t a1, const std::int32_t a2,
                                                       const std::int32_t a3, const std::int32_t b1,
                                                       const std::int32_t b2, const std::int32_t b3,
                                                       const std::int32_t b4) {
  const std::int32_t nodes[7] = {a1, a2, a3, b1, b2, b3, b4};
  if (CanUseCachedTwoPathOpt(graph, nodes, 3, 7)) {
    return CachedTwoPathOrderIsOpt(graph, nodes, 3, 7, false) ||
           CachedTwoPathOrderIsOpt(graph, nodes, 3, 7, true);
  }
  const quick_hs::SmallPath paths[quick_hs::kMaxPathCount] = {
      {.size = 3, .node = {a1, a2, a3}},
      {.size = 4, .node = {b1, b2, b3, b4}},
  };
  return quick_hs::Opt(graph, paths, 2);
}

// KH strong close-point 的 metric-excess 检查。true 表示 z 仍存在一个
// 邻边对无法关闭；false 表示对 z 的全部 Hamilton replies 均有严格更短重连。
CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
MetricExcessAdmitsPair(const quick_hs::GraphView& graph, const std::int32_t z,
                       const std::int32_t s1, const std::int32_t s2, const std::int32_t s3,
                       const std::int32_t s4, const std::int64_t cs1s2, const std::int64_t cs2s3,
                       const std::int64_t cs3s4) {
  static_cast<void>(cs2s3); // 与 KH opt_excess 签名对齐；中段成本已含在两个 z 距离中。
  const std::int64_t cs2z = quick_hs::Distance(graph, s2, z);
  const std::int64_t cs3z = quick_hs::Distance(graph, s3, z);
  const std::int64_t bound = quick_hs::Distance(graph, s1, s4) - cs1s2 - cs3s4;
  for (std::int64_t first_slot = quick_hs::NeighborBegin(graph, z);
       first_slot < quick_hs::NeighborEnd(graph, z); ++first_slot) {
    if (!quick_hs::NeighborActive(graph, first_slot)) {
      continue;
    }
    const std::int32_t z1 = quick_hs::Neighbor(graph, z, first_slot);
    const std::int64_t zz1 = quick_hs::Distance(graph, z, z1);
    if (z1 == s2 || z1 == s3 || zz1 - cs2z - quick_hs::Distance(graph, s3, z1) > bound ||
        zz1 - cs3z - quick_hs::Distance(graph, s2, z1) > bound ||
        !Opt24(graph, z1, z, s1, s2, s3, s4)) {
      continue;
    }
    for (std::int64_t second_slot = first_slot + 1; second_slot < quick_hs::NeighborEnd(graph, z);
         ++second_slot) {
      if (!quick_hs::NeighborActive(graph, second_slot) ||
          quick_hs::PairForbiddenBySlots(graph, z, first_slot, second_slot)) {
        continue;
      }
      const std::int32_t z2 = quick_hs::Neighbor(graph, z, second_slot);
      if (z2 != s2 && z2 != s3 && (z1 != s1 || z2 != s4) && (z1 != s4 || z2 != s1) &&
          Opt34(graph, z1, z, z2, s1, s2, s3, s4)) {
        return true;
      }
    }
  }
  return false;
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
StrongThreeCompatible(const quick_hs::GraphView& graph, std::int32_t p, std::int32_t q,
                      std::int32_t x, const std::int32_t middle, std::int32_t y) {
  // metric-excess 内层会为每个 z 枚举邻边对并执行精确 path-ordering。
  // 将它作为低度数强规则：高度数时退回“仍可能兼容”只会少删边，不会
  // 产生错误证明；低度数后期固定点仍完整执行该规则。
  constexpr std::int32_t kMetricExcessMaximumDegree = 4;
  std::int64_t cpq = quick_hs::Distance(graph, p, q);
  std::int64_t cxm = quick_hs::Distance(graph, x, middle);
  std::int64_t cmy = quick_hs::Distance(graph, middle, y);
  if (!quick_hs::Opt23(graph, p, q, x, middle, y, cpq, cxm, cmy)) {
    return false;
  }
  for (std::int32_t target_orientation = 0; target_orientation < 2; ++target_orientation) {
    for (std::int32_t path_orientation = 0; path_orientation < 2; ++path_orientation) {
      if (q == x) {
        if (quick_hs::Fixed(graph, p, y)) {
          return false;
        }
        if (graph.degree[q] > kMetricExcessMaximumDegree ||
            graph.degree[middle] > kMetricExcessMaximumDegree) {
          return true;
        }
        for (std::int64_t slot = quick_hs::NeighborBegin(graph, q);
             slot < quick_hs::NeighborEnd(graph, q); ++slot) {
          if (!quick_hs::NeighborActive(graph, slot)) {
            continue;
          }
          const std::int32_t z = quick_hs::Neighbor(graph, q, slot);
          if (z != p && z != middle && z != y && graph.degree[z] > kMetricExcessMaximumDegree) {
            return true;
          }
          if (z != p && z != middle && z != y &&
              !MetricExcessAdmitsPair(graph, z, p, x, middle, y, cpq, cxm, cmy)) {
            return false;
          }
        }
        for (std::int64_t slot = quick_hs::NeighborBegin(graph, middle);
             slot < quick_hs::NeighborEnd(graph, middle); ++slot) {
          if (!quick_hs::NeighborActive(graph, slot)) {
            continue;
          }
          const std::int32_t z = quick_hs::Neighbor(graph, middle, slot);
          if (z != p && z != q && z != x && z != y && !quick_hs::Active(graph, q, z) &&
              graph.degree[z] > kMetricExcessMaximumDegree) {
            return true;
          }
          if (z != p && z != q && z != x && z != y && !quick_hs::Active(graph, q, z) &&
              !MetricExcessAdmitsPair(graph, z, p, x, middle, y, cpq, cxm, cmy)) {
            return false;
          }
        }
        return true;
      }
      if (y == p && quick_hs::Fixed(graph, q, x)) {
        return false;
      }
      const std::int32_t saved_x = x;
      x = y;
      y = saved_x;
      const std::int64_t saved_cxm = cxm;
      cxm = cmy;
      cmy = saved_cxm;
    }
    const std::int32_t saved_p = p;
    p = q;
    q = saved_p;
  }
  return true;
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
MainEdgeEliminates(const quick_hs::GraphView& graph, const std::int32_t p, const std::int32_t q,
                   const std::int32_t r1, const std::int32_t r, const std::int32_t r2,
                   const std::int32_t s1, const std::int32_t s, const std::int32_t s2) {
  if (r == s1 || r == s2 || s == r1 || s == r2) {
    return false;
  }

  const std::int64_t beginning =
      quick_hs::Distance(graph, p, q) + quick_hs::Distance(graph, r, r1) +
      quick_hs::Distance(graph, r, r2) + quick_hs::Distance(graph, s, s1) +
      quick_hs::Distance(graph, s, s2);
  const std::int64_t rs = quick_hs::Distance(graph, r, s);

  const std::int32_t r_endpoint[2]{r1, r2};
  const std::int32_t s_endpoint[2]{s1, s2};
  for (std::int32_t swap_r = 0; swap_r < 2; ++swap_r) {
    for (std::int32_t swap_s = 0; swap_s < 2; ++swap_s) {
      const std::int32_t ra = r_endpoint[swap_r];
      const std::int32_t rb = r_endpoint[1 - swap_r];
      const std::int32_t sa = s_endpoint[swap_s];
      const std::int32_t sb = s_endpoint[1 - swap_s];
      const std::int64_t first =
          quick_hs::Distance(graph, p, ra) + quick_hs::Distance(graph, s, sa) + rs +
          quick_hs::Distance(graph, r, rb) + quick_hs::Distance(graph, q, sb);
      const std::int64_t second =
          quick_hs::Distance(graph, p, sa) + quick_hs::Distance(graph, r, ra) + rs +
          quick_hs::Distance(graph, s, sb) + quick_hs::Distance(graph, q, rb);
      if (beginning > first && beginning > second) {
        return true;
      }
    }
  }
  return false;
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
BasicAllowedPair(const quick_hs::GraphView& graph, const std::int32_t p, const std::int32_t q,
                 const std::int32_t middle, const std::int32_t first, const std::int32_t second) {
  return Compatible(graph, p, q, middle, first) && Compatible(graph, p, q, middle, second) &&
         ThreeCompatible(graph, p, q, first, middle, second);
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
AllowedPair(const quick_hs::GraphView& graph, const std::int32_t p, const std::int32_t q,
            const std::int32_t middle, const std::int32_t first, const std::int32_t second) {
  return Compatible(graph, p, q, middle, first) && Compatible(graph, p, q, middle, second) &&
         StrongThreeCompatible(graph, p, q, first, middle, second);
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool HasAllowedPair(const quick_hs::GraphView& graph,
                                                                const std::int32_t p,
                                                                const std::int32_t q,
                                                                const std::int32_t middle) {
  for (std::int64_t first_index = quick_hs::NeighborBegin(graph, middle);
       first_index < quick_hs::NeighborEnd(graph, middle); ++first_index) {
    if (!quick_hs::NeighborActive(graph, first_index)) {
      continue;
    }
    const std::int32_t first = quick_hs::Neighbor(graph, middle, first_index);
    if (!Compatible(graph, p, q, middle, first)) {
      continue;
    }
    for (std::int64_t second_index = first_index + 1;
         second_index < quick_hs::NeighborEnd(graph, middle); ++second_index) {
      if (!quick_hs::NeighborActive(graph, second_index)) {
        continue;
      }
      if (quick_hs::PairForbiddenBySlots(graph, middle, first_index, second_index)) {
        continue;
      }
      const std::int32_t second = quick_hs::Neighbor(graph, middle, second_index);
      if (AllowedPair(graph, p, q, middle, first, second)) {
        return true;
      }
    }
  }
  return false;
}

CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
PotentialPairAdmitsTour(const quick_hs::GraphView& graph, const std::int32_t p,
                        const std::int32_t q, const std::int32_t r, const std::int32_t s) {
  for (std::int64_t r1_index = quick_hs::NeighborBegin(graph, r);
       r1_index < quick_hs::NeighborEnd(graph, r); ++r1_index) {
    if (!quick_hs::NeighborActive(graph, r1_index)) {
      continue;
    }
    const std::int32_t r1 = quick_hs::Neighbor(graph, r, r1_index);
    if (!Compatible(graph, p, q, r, r1)) {
      continue;
    }
    for (std::int64_t r2_index = r1_index + 1; r2_index < quick_hs::NeighborEnd(graph, r);
         ++r2_index) {
      if (!quick_hs::NeighborActive(graph, r2_index)) {
        continue;
      }
      if (quick_hs::PairForbiddenBySlots(graph, r, r1_index, r2_index)) {
        continue;
      }
      const std::int32_t r2 = quick_hs::Neighbor(graph, r, r2_index);
      if (!AllowedPair(graph, p, q, r, r1, r2)) {
        continue;
      }
      for (std::int64_t s1_index = quick_hs::NeighborBegin(graph, s);
           s1_index < quick_hs::NeighborEnd(graph, s); ++s1_index) {
        if (!quick_hs::NeighborActive(graph, s1_index)) {
          continue;
        }
        const std::int32_t s1 = quick_hs::Neighbor(graph, s, s1_index);
        if (!Compatible(graph, p, q, s, s1)) {
          continue;
        }
        for (std::int64_t s2_index = s1_index + 1; s2_index < quick_hs::NeighborEnd(graph, s);
             ++s2_index) {
          if (!quick_hs::NeighborActive(graph, s2_index)) {
            continue;
          }
          if (quick_hs::PairForbiddenBySlots(graph, s, s1_index, s2_index)) {
            continue;
          }
          const std::int32_t s2 = quick_hs::Neighbor(graph, s, s2_index);
          if (AllowedPair(graph, p, q, s, s1, s2) &&
              !MainEdgeEliminates(graph, p, q, r1, r, r2, s1, s, s2)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// potentials 必须是当前边附近的不同节点。任何一个 potential 没有合法邻边对，
// 或任意两个 potentials 的所有邻边对组合都触发 Main Edge Elimination 时，pq
// 不可能属于最优 tour。
CUDAEE_MAIN_EDGE_HD CUDAEE_MAIN_EDGE_INLINE bool
CanEliminate(const quick_hs::GraphView& graph, const std::int32_t p, const std::int32_t q,
             const std::int32_t* const potentials, const std::int32_t potential_count) {
  if (potential_count < 2) {
    return false;
  }
  for (std::int32_t index = 0; index < potential_count; ++index) {
    const std::int32_t node = potentials[index];
    if (node == p || node == q || !HasAllowedPair(graph, p, q, node)) {
      return node != p && node != q;
    }
  }
  for (std::int32_t first = 0; first < potential_count; ++first) {
    for (std::int32_t second = first + 1; second < potential_count; ++second) {
      if (!PotentialPairAdmitsTour(graph, p, q, potentials[first], potentials[second])) {
        return true;
      }
    }
  }
  return false;
}

} // namespace cudaee::detail::main_edge

#undef CUDAEE_MAIN_EDGE_HD
#undef CUDAEE_MAIN_EDGE_INLINE
