#pragma once

#include "../fgpu/main_edge_predicate.hpp"

namespace cudaee::detail::main_metric {

// 一个 warp 合作覆盖 z 的全部 Hamilton replies。z1 的昂贵 Opt24 只算一次，
// 再由 32 个 lane 分摊 z2 的精确 Opt34；分段只改变调度，不设度数门槛。
__device__ inline bool MetricExcessAdmitsPairWarp(const quick_hs::GraphView graph, const int z,
                                                  const int s1, const int s2, const int s3,
                                                  const int s4) {
  constexpr unsigned kMask = 0xffffffffU;
  const int lane = static_cast<int>(threadIdx.x & 31U);
  const auto begin = quick_hs::NeighborBegin(graph, z);
  const auto end = quick_hs::NeighborEnd(graph, z);
  const auto cs2z = quick_hs::Distance(graph, s2, z);
  const auto cs3z = quick_hs::Distance(graph, s3, z);
  const auto bound = quick_hs::Distance(graph, s1, s4) - quick_hs::Distance(graph, s1, s2) -
                     quick_hs::Distance(graph, s3, s4);
  for (auto window = begin; window < end; window += 32) {
    const auto slot = window + lane;
    int z1 = -1;
    bool eligible = false;
    if (slot < end && quick_hs::NeighborActive(graph, slot)) {
      z1 = quick_hs::Neighbor(graph, z, slot);
      const auto zz1 = quick_hs::Distance(graph, z, z1);
      eligible = z1 != s2 && z1 != s3 && zz1 - cs2z - quick_hs::Distance(graph, s3, z1) <= bound &&
                 zz1 - cs3z - quick_hs::Distance(graph, s2, z1) <= bound &&
                 main_edge::Opt24(graph, z1, z, s1, s2, s3, s4);
    }
    unsigned candidates = __ballot_sync(kMask, eligible);
    while (candidates != 0U) {
      const int owner = __ffs(static_cast<int>(candidates)) - 1;
      const int first = __shfl_sync(kMask, z1, owner);
      const auto first_slot = window + owner;
      for (auto second_window = first_slot + 1; second_window < end; second_window += 32) {
        const auto second_slot = second_window + lane;
        bool admits = false;
        if (second_slot < end && quick_hs::NeighborActive(graph, second_slot) &&
            !quick_hs::PairForbiddenBySlots(graph, z, first_slot, second_slot)) {
          const auto second = quick_hs::Neighbor(graph, z, second_slot);
          const bool closes = (first == s1 && second == s4) || (first == s4 && second == s1);
          const quick_hs::SmallPath circuit{.size = 5, .node = {s1, s2, s3, s4, z}};
          admits = second != s2 && second != s3 &&
                   (!closes || quick_hs::ClosedPathsMayCoverWholeGraph(graph, circuit)) &&
                   main_edge::Opt34(graph, first, z, second, s1, s2, s3, s4);
        }
        if (__any_sync(kMask, admits)) {
          return true;
        }
      }
      candidates &= candidates - 1U;
    }
  }
  return false;
}

__device__ inline bool AllowedPairWarp(const quick_hs::GraphView graph, int p, int q, int x,
                                       const int middle, int y) {
  constexpr unsigned kMask = 0xffffffffU;
  const int lane = static_cast<int>(threadIdx.x & 31U);
  int allowed = 0;
  if (lane == 0) {
    allowed =
        main_edge::Compatible(graph, p, q, middle, x) &&
        main_edge::Compatible(graph, p, q, middle, y) &&
        quick_hs::Opt23(graph, p, q, x, middle, y, quick_hs::Distance(graph, p, q),
                        quick_hs::Distance(graph, x, middle), quick_hs::Distance(graph, middle, y));
  }
  if (__shfl_sync(kMask, allowed, 0) == 0) {
    return false;
  }
  for (int target_orientation = 0; target_orientation < 2; ++target_orientation) {
    for (int path_orientation = 0; path_orientation < 2; ++path_orientation) {
      if (q == x) {
        if (quick_hs::Fixed(graph, p, y)) {
          const quick_hs::SmallPath circuit{.size = 4, .node = {p, q, middle, y}};
          return quick_hs::ClosedPathsMayCoverWholeGraph(graph, circuit);
        }
        for (auto slot = quick_hs::NeighborBegin(graph, q); slot < quick_hs::NeighborEnd(graph, q);
             ++slot) {
          const auto z = quick_hs::Neighbor(graph, q, slot);
          if (quick_hs::NeighborActive(graph, slot) && z != p && z != middle && z != y &&
              !MetricExcessAdmitsPairWarp(graph, z, p, x, middle, y)) {
            return false;
          }
        }
        for (auto slot = quick_hs::NeighborBegin(graph, middle);
             slot < quick_hs::NeighborEnd(graph, middle); ++slot) {
          const auto z = quick_hs::Neighbor(graph, middle, slot);
          if (quick_hs::NeighborActive(graph, slot) && z != p && z != q && z != x && z != y &&
              !quick_hs::Active(graph, q, z) &&
              !MetricExcessAdmitsPairWarp(graph, z, p, x, middle, y)) {
            return false;
          }
        }
        return true;
      }
      if (y == p && quick_hs::Fixed(graph, q, x)) {
        const quick_hs::SmallPath circuit{.size = 4, .node = {q, p, middle, x}};
        return quick_hs::ClosedPathsMayCoverWholeGraph(graph, circuit);
      }
      const int saved = x;
      x = y;
      y = saved;
    }
    const int saved = p;
    p = q;
    q = saved;
  }
  return true;
}

} // namespace cudaee::detail::main_metric
