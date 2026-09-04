#pragma once

#include <climits>
#include <cstdint>

#if defined(__CUDACC__)
#define CUDAEE_QUICK_HS_HD __host__ __device__
#define CUDAEE_QUICK_HS_INLINE __forceinline__
#else
#define CUDAEE_QUICK_HS_HD
#define CUDAEE_QUICK_HS_INLINE inline
#endif

namespace cudaee::detail::quick_hs {

// 算法语义依据 Keld Helsgaun 的 MIT-licensed KH-ElimTSP 默认 -q 路径重构；
// 改用固定数组和 host/device 公用实现，便于 CUDA 搜索与独立 CPU 重放一致。
// KH -q 首期只会把 2/3/3 三条短路径交给 opt，合并共享端点后
// 节点数不会超过 8。固定上限避免 device recursion 和动态分配。
constexpr std::int32_t kMaxPathCount = 3;
constexpr std::int32_t kMaxPathNodes = 8;
constexpr std::int32_t kMaxPotentialNodes = 10;

struct GraphView {
  std::int32_t dimension{};
  // 每行占 dimension 个槽，前 degree[row] 个是按 (cost,node) 排序的活动邻点。
  const std::int32_t* degree{};
  const std::int32_t* neighbors{};
  const std::int64_t* distance{};
  const std::uint8_t* active{};
};

struct Witness {
  std::int32_t c{-1};
  std::int32_t d{-1};
};

struct SmallPath {
  std::int32_t size{};
  std::int32_t node[kMaxPathNodes]{};
};

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE std::int64_t
Distance(const GraphView& graph, const std::int32_t a, const std::int32_t b) {
  return graph.distance[static_cast<std::int64_t>(a) * graph.dimension + b];
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Active(const GraphView& graph, const std::int32_t a,
                                                      const std::int32_t b) {
  return graph.active[static_cast<std::int64_t>(a) * graph.dimension + b] != 0U;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Fixed(const GraphView& graph, const std::int32_t a,
                                                     const std::int32_t b) {
  return Active(graph, a, b) && (graph.degree[a] == 2 || graph.degree[b] == 2);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Opt22(const GraphView& graph, const std::int32_t a,
                                                     const std::int32_t b, const std::int32_t c,
                                                     const std::int32_t d, const std::int64_t cab,
                                                     const std::int64_t ccd) {
  return c == a || c == b || d == a || d == b ||
         cab + ccd <= Distance(graph, a, c) + Distance(graph, d, b) ||
         cab + ccd <= Distance(graph, a, d) + Distance(graph, c, b);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Opt23(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t c1,
      const std::int32_t c, const std::int32_t c2, const std::int64_t cab, const std::int64_t cc1c,
      const std::int64_t ccc2) {
  if ((c1 == a && c2 == b) || (c1 == b && c2 == a) ||
      cab + cc1c + ccc2 > Distance(graph, a, c) + Distance(graph, c, b) + Distance(graph, c1, c2)) {
    return false;
  }
  // 对齐 KH 默认 strong_3_opt=0 的两个固定边门禁。
  const std::int32_t row = c * graph.dimension;
  for (std::int32_t index = 0; index < graph.degree[c]; ++index) {
    const std::int32_t z = graph.neighbors[row + index];
    if (z != c1 && z != c2 && graph.degree[z] == 2) {
      return false;
    }
  }
  return !Fixed(graph, c1, c2);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool HasCycle(const std::int32_t* const endpoints,
                                                        const std::int32_t pair_count) {
  std::int32_t unique[2 * kMaxPathCount]{};
  std::int32_t parent[2 * kMaxPathCount]{};
  std::int32_t degree[2 * kMaxPathCount]{};
  std::int32_t unique_count = 0;
  for (std::int32_t pair = 0; pair < pair_count; ++pair) {
    std::int32_t local[2]{};
    for (std::int32_t side = 0; side < 2; ++side) {
      const std::int32_t node = endpoints[2 * pair + side];
      local[side] = -1;
      for (std::int32_t index = 0; index < unique_count; ++index) {
        if (unique[index] == node) {
          local[side] = index;
          break;
        }
      }
      if (local[side] < 0) {
        local[side] = unique_count;
        unique[unique_count] = node;
        parent[unique_count] = unique_count;
        degree[unique_count] = 0;
        ++unique_count;
      }
    }
    if (++degree[local[0]] > 2 || ++degree[local[1]] > 2) {
      return true;
    }
    std::int32_t first = local[0];
    while (parent[first] != first) {
      first = parent[first];
    }
    std::int32_t second = local[1];
    while (parent[second] != second) {
      second = parent[second];
    }
    if (first == second) {
      return true;
    }
    parent[second] = first;
  }
  return false;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE std::int64_t
PathTransitionCost(const GraphView& graph, const std::int32_t* const order,
                   const std::uint8_t* const fixed_after, const std::int32_t first_position,
                   const std::int32_t second_position, const std::int64_t forced_cost) {
  const std::int32_t difference = first_position - second_position;
  if ((difference == 1 || difference == -1)) {
    const std::int32_t lower = first_position < second_position ? first_position : second_position;
    if (fixed_after[lower] != 0U) {
      return forced_cost;
    }
  }
  return Distance(graph, order[first_position], order[second_position]);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
PathOrderIsOpt(const GraphView& graph, const SmallPath* const paths, const std::int32_t path_count,
               const std::int32_t* const permutation, const std::uint32_t reverse_mask) {
  std::int32_t order[kMaxPathNodes]{};
  std::uint8_t fixed_after[kMaxPathNodes]{};
  std::int32_t total = 0;
  if (path_count <= 0 || path_count > kMaxPathCount) {
    return true;
  }
  for (std::int32_t position = 0; position < path_count; ++position) {
    const SmallPath& path = paths[permutation[position]];
    if (path.size <= 0 || path.size > kMaxPathNodes - total) {
      return true;
    }
    const bool reverse = position > 0 && ((reverse_mask >> (position - 1)) & 1U) != 0U;
    for (std::int32_t index = 0; index < path.size; ++index) {
      order[total++] = path.node[reverse ? path.size - 1 - index : index];
    }
    fixed_after[total - 1] = 1U;
  }
  fixed_after[total - 1] = 0U;
  if (total < 3 || total > kMaxPathNodes) {
    return true;
  }

  const std::int32_t dynamic_nodes = total - 2;
  const std::int32_t full_mask = (1 << dynamic_nodes) - 1;
  // KH 用 INT_MIN/(N-1) 作为路径间固定连接的哨兵。这里提升为 int64
  // 只为避免中间和溢出，比较结果与其 32 位安全输入一致。
  const std::int64_t forced_cost = static_cast<std::int64_t>(INT_MIN) / (total - 1);
  std::int64_t original = Distance(graph, order[0], order[1]);
  for (std::int32_t position = 1; position < total - 1; ++position) {
    original += fixed_after[position] != 0U ? forced_cost
                                            : Distance(graph, order[position], order[position + 1]);
  }

  constexpr std::int64_t kInfinity = INT64_MAX / 8;
  std::int64_t dynamic_program[(1 << (kMaxPathNodes - 2)) * (kMaxPathNodes - 2)];
  for (std::int32_t index = 0; index < (1 << (kMaxPathNodes - 2)) * (kMaxPathNodes - 2); ++index) {
    dynamic_program[index] = kInfinity;
  }
  for (std::int32_t node = 0; node < dynamic_nodes; ++node) {
    dynamic_program[(1 << node) * (kMaxPathNodes - 2) + node] =
        Distance(graph, order[node + 1], order[0]);
  }
  for (std::int32_t mask = 1; mask <= full_mask; ++mask) {
    for (std::int32_t last = 0; last < dynamic_nodes; ++last) {
      if ((mask & (1 << last)) == 0) {
        continue;
      }
      const std::int64_t current = dynamic_program[mask * (kMaxPathNodes - 2) + last];
      if (current == kInfinity) {
        continue;
      }
      for (std::int32_t next = 0; next < dynamic_nodes; ++next) {
        if ((mask & (1 << next)) != 0) {
          continue;
        }
        const std::int32_t next_mask = mask | (1 << next);
        const std::int64_t candidate =
            current +
            PathTransitionCost(graph, order, fixed_after, last + 1, next + 1, forced_cost);
        std::int64_t& destination = dynamic_program[next_mask * (kMaxPathNodes - 2) + next];
        if (candidate < destination) {
          destination = candidate;
        }
      }
    }
  }
  for (std::int32_t last = 0; last < dynamic_nodes; ++last) {
    const std::int64_t candidate =
        dynamic_program[full_mask * (kMaxPathNodes - 2) + last] +
        PathTransitionCost(graph, order, fixed_after, last + 1, total - 1, forced_cost);
    if (candidate < original) {
      return false;
    }
  }
  return true;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Opt(const GraphView& graph,
                                                   const SmallPath* const source_paths,
                                                   const std::int32_t source_path_count) {
  SmallPath paths[kMaxPathCount]{};
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
      if ((b1 == b2 && paths[i].size > 1) || (paths[i].size > 2 && Fixed(graph, b1, b2))) {
        return false;
      }
      for (std::int32_t j = 0; j < i && !merged; ++j) {
        const std::int32_t a1 = paths[j].node[0];
        const std::int32_t a2 = paths[j].node[paths[j].size - 1];
        if (j == 0 &&
            ((a1 == a2 && paths[j].size > 1) || (paths[j].size > 2 && Fixed(graph, a1, a2)))) {
          return false;
        }

        std::int32_t first_direction = 1;
        std::int32_t second_direction = 1;
        bool join = false;
        bool shared = false;
        if (a2 == b1 || (a2 != b2 && b1 != a1 && Fixed(graph, a2, b1))) {
          if (a1 == b2) {
            return false;
          }
          join = true;
          shared = a2 == b1;
        } else if (a2 == b2 || (a2 != b1 && b2 != a1 && Fixed(graph, a2, b2))) {
          if (a1 == b1) {
            return false;
          }
          join = true;
          shared = a2 == b2;
          second_direction = -1;
        } else if (a1 == b1 || (a1 != b2 && b1 != a2 && Fixed(graph, a1, b1))) {
          if (a2 == b2) {
            return false;
          }
          join = true;
          shared = a1 == b1;
          first_direction = -1;
        } else if (a1 == b2 || (a1 != b1 && b2 != a2 && Fixed(graph, a1, b2))) {
          if (a2 == b1) {
            return false;
          }
          join = true;
          shared = a1 == b2;
          first_direction = -1;
          second_direction = -1;
        }
        if (!join) {
          continue;
        }

        SmallPath combined;
        for (std::int32_t index = 0; index < paths[j].size; ++index) {
          combined.node[combined.size++] =
              paths[j].node[first_direction > 0 ? index : paths[j].size - 1 - index];
        }
        const std::int32_t start = shared ? 1 : 0;
        for (std::int32_t index = start; index < paths[i].size; ++index) {
          combined.node[combined.size++] =
              paths[i].node[second_direction > 0 ? index : paths[i].size - 1 - index];
        }
        if (combined.size > kMaxPathNodes) {
          return true;
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

  std::int32_t endpoints[2 * kMaxPathCount]{};
  for (std::int32_t path = 0; path < path_count; ++path) {
    endpoints[2 * path] = paths[path].node[0];
    endpoints[2 * path + 1] = paths[path].node[paths[path].size - 1];
  }
  if (HasCycle(endpoints, path_count)) {
    return false;
  }
  if (path_count > 3) {
    return true;
  }

  std::int32_t permutation[kMaxPathCount]{0, 1, 2};
  const std::int32_t permutation_count = path_count == 3 ? 2 : 1;
  const std::uint32_t orientation_count = 1U << (path_count > 0 ? path_count - 1 : 0);
  for (std::int32_t permutation_index = 0; permutation_index < permutation_count;
       ++permutation_index) {
    if (permutation_index == 1) {
      permutation[1] = 2;
      permutation[2] = 1;
    }
    for (std::uint32_t reverse_mask = 0; reverse_mask < orientation_count; ++reverse_mask) {
      if (PathOrderIsOpt(graph, paths, path_count, permutation, reverse_mask)) {
        return true;
      }
    }
  }
  return false;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Opt222(const GraphView& graph, const std::int32_t a,
                                                      const std::int32_t b, const std::int32_t c1,
                                                      const std::int32_t c2, const std::int32_t d1,
                                                      const std::int32_t d2) {
  const SmallPath paths[kMaxPathCount] = {
      {.size = 2, .node = {a, b}},
      {.size = 2, .node = {c1, c2}},
      {.size = 2, .node = {d1, d2}},
  };
  return Opt(graph, paths, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Opt232(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t c1,
       const std::int32_t c, const std::int32_t c2, const std::int32_t d1, const std::int32_t d2) {
  const SmallPath paths[kMaxPathCount] = {
      {.size = 2, .node = {a, b}},
      {.size = 3, .node = {c1, c, c2}},
      {.size = 2, .node = {d1, d2}},
  };
  return Opt(graph, paths, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Opt233(const GraphView& graph, const std::int32_t a,
                                                      const std::int32_t b, const std::int32_t c1,
                                                      const std::int32_t c, const std::int32_t c2,
                                                      const std::int32_t d1, const std::int32_t d,
                                                      const std::int32_t d2) {
  const SmallPath paths[kMaxPathCount] = {
      {.size = 2, .node = {a, b}},
      {.size = 3, .node = {c1, c, c2}},
      {.size = 3, .node = {d1, d, d2}},
  };
  return Opt(graph, paths, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
HasCycle222(const std::int32_t a, const std::int32_t b, const std::int32_t c1,
            const std::int32_t c2, const std::int32_t d1, const std::int32_t d2) {
  const std::int32_t endpoints[6] = {a, b, c1, c2, d1, d2};
  return HasCycle(endpoints, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Compatible(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t c,
           const std::int32_t d, const std::int64_t cab) {
  const std::int64_t ccd = Distance(graph, c, d);
  return c == a || c == b || d == a || d == b || ccd > cab ||
         (Active(graph, c, d) && (cab + ccd <= Distance(graph, a, c) + Distance(graph, d, b) ||
                                  cab + ccd <= Distance(graph, a, d) + Distance(graph, c, b)));
}

// 返回 true 表示固定 c,d 的全部 Hamilton replies 都已被整数局部改进关闭。
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
CanEliminateWithWitness(const GraphView& graph, const std::int32_t a, const std::int32_t b,
                        const std::int32_t c, const std::int32_t d) {
  if (a == b || c == d || !Active(graph, a, b) ||
      Compatible(graph, a, b, c, d, Distance(graph, a, b))) {
    return false;
  }
  const std::int64_t cab = Distance(graph, a, b);
  const std::int32_t c_row = c * graph.dimension;
  const std::int32_t d_row = d * graph.dimension;
  for (std::int32_t c_first = 0; c_first < graph.degree[c]; ++c_first) {
    const std::int32_t c1 = graph.neighbors[c_row + c_first];
    if (c1 == d) {
      continue;
    }
    const std::int64_t cc1c = Distance(graph, c1, c);
    if (!Opt22(graph, c1, c, a, b, cc1c, cab)) {
      continue;
    }
    for (std::int32_t c_second = c_first + 1; c_second < graph.degree[c]; ++c_second) {
      const std::int32_t c2 = graph.neighbors[c_row + c_second];
      if (c2 == d || (c2 == a && c1 == b) || (c2 == b && c1 == a)) {
        continue;
      }
      const std::int64_t ccc2 = Distance(graph, c, c2);
      if (!Opt22(graph, c, c2, a, b, ccc2, cab) ||
          !Opt23(graph, a, b, c1, c, c2, cab, cc1c, ccc2)) {
        continue;
      }
      for (std::int32_t d_first = 0; d_first < graph.degree[d]; ++d_first) {
        const std::int32_t d1 = graph.neighbors[d_row + d_first];
        if (d1 == c || ((d1 == a || d1 == b) && (c1 == d1 || c2 == d1))) {
          continue;
        }
        const std::int64_t cd1d = Distance(graph, d1, d);
        if (!Opt22(graph, d, d1, a, b, cd1d, cab) || !Opt22(graph, d, d1, c1, c, cd1d, cc1c) ||
            !Opt22(graph, d, d1, c, c2, cd1d, ccc2) ||
            !Opt23(graph, d, d1, c1, c, c2, cd1d, cc1c, ccc2) ||
            !Opt222(graph, a, b, c1, c, d, d1) || !Opt222(graph, a, b, c, c2, d, d1) ||
            !Opt232(graph, a, b, c1, c, c2, d1, d)) {
          continue;
        }
        for (std::int32_t d_second = d_first + 1; d_second < graph.degree[d]; ++d_second) {
          const std::int32_t d2 = graph.neighbors[d_row + d_second];
          if (d2 == c || (d2 == a && (d1 == b || c1 == d2 || c2 == d2)) ||
              (d2 == b && (d1 == a || c1 == d2 || c2 == d2)) ||
              (d2 == c1 && (d1 == c2 || a == d2 || b == d2)) ||
              (d2 == c2 && (d1 == c1 || a == d2 || b == d2)) || HasCycle222(a, b, c1, c2, d1, d2)) {
            continue;
          }
          const std::int64_t cdd2 = Distance(graph, d, d2);
          if (Opt22(graph, d, d2, a, b, cdd2, cab) && Opt22(graph, d, d2, c1, c, cdd2, cc1c) &&
              Opt22(graph, d, d2, c, c2, cdd2, ccc2) &&
              Opt23(graph, a, b, d1, d, d2, cab, cd1d, cdd2) &&
              Opt23(graph, d, d2, c1, c, c2, cdd2, cc1c, ccc2) &&
              Opt23(graph, c1, c, d1, d, d2, cc1c, cd1d, cdd2) &&
              Opt23(graph, c, c2, d1, d, d2, ccc2, cd1d, cdd2) &&
              Opt233(graph, a, b, c1, c, c2, d1, d, d2)) {
            // 找到一个不能关闭的 reply，固定 c,d 不能证明目标边。
            return false;
          }
        }
      }
    }
  }
  return true;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE Witness FindWitness(const GraphView& graph,
                                                              const std::int32_t a,
                                                              const std::int32_t b) {
  Witness result;
  if (!Active(graph, a, b) || graph.degree[a] <= 2 || graph.degree[b] <= 2) {
    return result;
  }
  std::int32_t candidate_nodes[kMaxPotentialNodes]{};
  std::int64_t candidate_scores[kMaxPotentialNodes]{};
  std::int32_t candidate_count = 0;
  for (std::int32_t side = 0; side < 2; ++side) {
    const std::int32_t from = side == 0 ? a : b;
    const std::int32_t other = side == 0 ? b : a;
    const std::int32_t row = from * graph.dimension;
    for (std::int32_t edge = 0; edge < graph.degree[from]; ++edge) {
      const std::int32_t node = graph.neighbors[row + edge];
      if (node == a || node == b) {
        continue;
      }
      bool duplicate = false;
      for (std::int32_t existing = 0; existing < candidate_count; ++existing) {
        duplicate = duplicate || candidate_nodes[existing] == node;
      }
      if (duplicate) {
        continue;
      }
      const std::int64_t score = Distance(graph, from, node) + Distance(graph, node, other);
      if (candidate_count == kMaxPotentialNodes &&
          (score > candidate_scores[kMaxPotentialNodes - 1] ||
           (score == candidate_scores[kMaxPotentialNodes - 1] &&
            node >= candidate_nodes[kMaxPotentialNodes - 1]))) {
        continue;
      }
      std::int32_t position =
          candidate_count < kMaxPotentialNodes ? candidate_count++ : kMaxPotentialNodes - 1;
      while (position > 0 &&
             (score < candidate_scores[position - 1] ||
              (score == candidate_scores[position - 1] && node < candidate_nodes[position - 1]))) {
        candidate_scores[position] = candidate_scores[position - 1];
        candidate_nodes[position] = candidate_nodes[position - 1];
        --position;
      }
      candidate_scores[position] = score;
      candidate_nodes[position] = node;
    }
  }

  std::int32_t pair_trials = 0;
  for (std::int32_t first = 1; first < candidate_count; ++first) {
    const std::int32_t c = candidate_nodes[first];
    for (std::int32_t second = 0; second < first; ++second) {
      if (++pair_trials > 10) {
        return result;
      }
      const std::int32_t d = candidate_nodes[second];
      if (!Compatible(graph, a, b, c, d, Distance(graph, a, b)) &&
          CanEliminateWithWitness(graph, a, b, c, d)) {
        result.c = c;
        result.d = d;
        return result;
      }
    }
  }
  return result;
}

} // namespace cudaee::detail::quick_hs

#undef CUDAEE_QUICK_HS_HD
#undef CUDAEE_QUICK_HS_INLINE
