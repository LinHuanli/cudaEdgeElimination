#pragma once

#include "cuda_edge_elimination/fgpu_execution.hpp"
#include <climits>
#include <cmath>
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
// 节点数不会超过 8；-e1 至多 9 点，-e2 三路径 continuation 至多 10 点。
// 固定上限避免 device recursion 和动态分配。
constexpr std::int32_t kMaxPathCount = 3;
constexpr std::int32_t kMaxPathNodes = 10;
// 编译期容量只决定线程私有存储；实际候选数和 pair 扫描宽度由运行配置给出。
// 旧 resident 使用 10/10 复现 KH -q，新 one-shot 可在同一 kernel 内做完整宽度。
constexpr std::int32_t kMaxPotentialNodes = 32;

struct GraphView {
  std::int32_t dimension{};
  // legacy CPU replayer 使用定长稠密行；新 resident 主链使用
  // row_offsets/neighbor_edge_ids 描述的稀疏 CSR。CSR 行按
  // (cost,node) 排序，删边后保留 stable slot，遍历时跳过 inactive edge。
  const std::int32_t* degree{};
  const std::int32_t* neighbors{};
  const std::int64_t* distance{};
  const std::uint8_t* active{};
  const std::int64_t* row_offsets{};
  const std::int32_t* neighbor_edge_ids{};
  // 当前 immutable snapshot 的逐顶点三角 pair mask。pair_offsets[v]
  // 指向 v 的 d(v)*(d(v)-1)/2 个条目，1 表示该 Hamilton 邻边对已获证不可能。
  const std::int64_t* pair_offsets{};
  const std::uint8_t* nonpair_mask{};
  const std::int32_t* edge_u{};
  const std::int32_t* edge_v{};
  const std::uint8_t* edge_active{};
  const std::uint8_t* fixed_edge{};
  const std::int64_t* coordinate_x{};
  const std::int64_t* coordinate_y{};
  std::int64_t edge_count{};
  std::uint8_t distance_type{};
  bool complete_graph{};
  PointLeafKernel point_leaf_kernel{PointLeafKernel::kPermutation};
  // 全实例不可变三角距离缓存，含已删除边；不能从稀疏 CSR 推断距离。
  const std::int64_t* triangular_distance{};
  std::uint32_t coordinate_denominator{1U};
  const std::uint8_t* permutation_orders{};
};

struct Witness {
  std::int32_t c{-1};
  std::int32_t d{-1};
};

struct SmallPath {
  std::int32_t size{};
  std::int32_t node[kMaxPathNodes]{};
};

// 只有真子环才能直接关闭。fixed 连接与共享端点同时出现时，两条
// 路径可能恰好构成覆盖全图的 Hamilton 环；它不是局部不可能状态。
// 大实例首先由节点数上界 O(1) 排除，小图才精确去重计数。
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
ClosedPathsMayCoverWholeGraph(const GraphView& graph, const SmallPath& first,
                              const SmallPath* const second = nullptr,
                              const SmallPath* const third = nullptr) {
  const std::int32_t upper =
      first.size + (second == nullptr ? 0 : second->size) + (third == nullptr ? 0 : third->size);
  if (graph.dimension > upper) {
    return false;
  }
  const SmallPath* const paths[3] = {&first, second, third};
  int distinct = 0;
  for (int p = 0; p < 3; ++p) {
    if (paths[p] == nullptr) {
      continue;
    }
    if (paths[p]->size < 1 || paths[p]->size > kMaxPathNodes) {
      return true;
    }
    for (int index = 0; index < paths[p]->size; ++index) {
      const auto node = paths[p]->node[index];
      if (node < 0 || node >= graph.dimension) {
        return true;
      }
      bool seen = false;
      for (int q = 0; q <= p; ++q) {
        if (paths[q] == nullptr) {
          continue;
        }
        const int end = q == p ? index : paths[q]->size;
        for (int previous = 0; previous < end; ++previous) {
          seen = seen || paths[q]->node[previous] == node;
        }
      }
      distinct += seen ? 0 : 1;
    }
  }
  return distinct == graph.dimension;
}

// endpoint/fixed 合并完成后，各路径必须顶点互异，path-order DP 才能把
// 每个数组位置当成一个 Hamilton 顶点。作者实现只处理共享端点；若内部
// 顶点仍跨路径重复，继续求解会把同一 TSP 顶点当成多个位置并产生假阴性。
// 返回 true 时调用者应保守地保持 reply 开放，而不是据此授权删除。
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
HasAmbiguousPathOverlap(const SmallPath* const paths, const std::int32_t path_count) {
  for (std::int32_t first_path = 0; first_path < path_count; ++first_path) {
    for (std::int32_t first = 0; first < paths[first_path].size; ++first) {
      for (std::int32_t second_path = first_path; second_path < path_count; ++second_path) {
        const std::int32_t begin = second_path == first_path ? first + 1 : 0;
        for (std::int32_t second = begin; second < paths[second_path].size; ++second) {
          if (paths[first_path].node[first] == paths[second_path].node[second]) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool AllNodesDistinct(const std::int32_t* const nodes,
                                                                const std::int32_t count) {
  for (std::int32_t first = 0; first < count; ++first) {
    for (std::int32_t second = first + 1; second < count; ++second) {
      if (nodes[first] == nodes[second]) {
        return false;
      }
    }
  }
  return true;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE std::uint64_t IntegerSqrtFloor(std::uint64_t value) {
#if defined(__CUDA_ARCH__)
  // GPU 热路径先使用硬件 FP64 sqrt 得到候选，再用无溢出的整数除法
  // 精确校正。最终返回值与逐 bit 算法完全相同，但避免每次距离计算
  // 固定执行约 32 轮数据相关分支。
  std::uint64_t root = static_cast<std::uint64_t>(sqrt(static_cast<double>(value)));
  while (root != 0U && root > value / root) {
    --root;
  }
  for (;;) {
    const std::uint64_t next = root + 1U;
    if (next == 0U || next > value / next) {
      break;
    }
    root = next;
  }
  return root;
#else
  std::uint64_t root = 0;
  std::uint64_t bit = std::uint64_t{1} << 62;
  while (bit > value) {
    bit >>= 2;
  }
  while (bit != 0) {
    if (value >= root + bit) {
      value -= root + bit;
      root = (root >> 1) + bit;
    } else {
      root >>= 1;
    }
    bit >>= 2;
  }
  return root;
#endif
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE std::int64_t
Distance(const GraphView& graph, const std::int32_t a, const std::int32_t b) {
  if (a == b) {
    return 0;
  }
  if (graph.triangular_distance != nullptr) {
    const std::int64_t first = a < b ? a : b;
    const std::int64_t second = a < b ? b : a;
    return graph
        .triangular_distance[first * (2LL * graph.dimension - first - 1) / 2 + second - first - 1];
  }
  if (graph.distance != nullptr) {
    return graph.distance[static_cast<std::int64_t>(a) * graph.dimension + b];
  }
  const std::int64_t dx = graph.coordinate_x[a] - graph.coordinate_x[b];
  const std::int64_t dy = graph.coordinate_y[a] - graph.coordinate_y[b];
  const std::uint64_t absolute_x = static_cast<std::uint64_t>(dx < 0 ? -dx : dx);
  const std::uint64_t absolute_y = static_cast<std::uint64_t>(dy < 0 ? -dy : dy);
  const std::uint64_t squared = absolute_x * absolute_x + absolute_y * absolute_y;
  const std::uint64_t root = IntegerSqrtFloor(squared);
  if (graph.coordinate_denominator == 2U) {
    // EUC: sqrt(S)/2 >= k+1/2 等价于 floor(sqrt(S)) >= 2k+1。
    // CEIL: 只有 S=(2k)^2 才不进位；绝不先舍入坐标或除法。
    if (graph.distance_type == 0U)
      return static_cast<std::int64_t>((root + 1U) / 2U);
    const std::uint64_t even = (root / 2U) * 2U;
    return static_cast<std::int64_t>(root / 2U + (squared != even * even));
  }
  std::uint64_t rounded = root;
  if (graph.distance_type == 0U) {
    // EUC_2D: S-r^2 > r 与 sqrt(S) >= r+0.5 等价。
    rounded += static_cast<std::uint64_t>(squared - root * root > root);
  } else {
    rounded += static_cast<std::uint64_t>(root * root != squared);
  }
  return static_cast<std::int64_t>(rounded);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE std::int64_t NeighborBegin(const GraphView& graph,
                                                                     const std::int32_t node) {
  return graph.row_offsets == nullptr ? 0 : graph.row_offsets[node];
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE std::int64_t NeighborEnd(const GraphView& graph,
                                                                   const std::int32_t node) {
  return graph.row_offsets == nullptr ? graph.degree[node] : graph.row_offsets[node + 1];
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE std::int32_t
Neighbor(const GraphView& graph, const std::int32_t node, const std::int64_t slot) {
  return graph.row_offsets == nullptr
             ? graph.neighbors[static_cast<std::int64_t>(node) * graph.dimension + slot]
             : graph.neighbors[slot];
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool NeighborActive(const GraphView& graph,
                                                              const std::int64_t slot) {
  return graph.row_offsets == nullptr || graph.edge_active[graph.neighbor_edge_ids[slot]] != 0U;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
PairForbiddenBySlots(const GraphView& graph, const std::int32_t center,
                     const std::int64_t first_slot, const std::int64_t second_slot) {
  if (graph.pair_offsets == nullptr || graph.nonpair_mask == nullptr || first_slot == second_slot) {
    return false;
  }
  const std::int64_t begin = NeighborBegin(graph, center);
  const std::int64_t degree = NeighborEnd(graph, center) - begin;
  std::int64_t first = first_slot - begin;
  std::int64_t second = second_slot - begin;
  if (first < 0 || second < 0 || first >= degree || second >= degree) {
    return false;
  }
  if (first > second) {
    const std::int64_t temporary = first;
    first = second;
    second = temporary;
  }
  const std::int64_t local = first * (2 * degree - first - 1) / 2 + (second - first - 1);
  return graph.nonpair_mask[graph.pair_offsets[center] + local] != 0U;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool PairForbidden(const GraphView& graph,
                                                             const std::int32_t center,
                                                             const std::int32_t first_neighbor,
                                                             const std::int32_t second_neighbor) {
  if (graph.pair_offsets == nullptr || graph.nonpair_mask == nullptr ||
      first_neighbor == second_neighbor) {
    return false;
  }
  std::int64_t first_slot = -1;
  std::int64_t second_slot = -1;
  for (std::int64_t slot = NeighborBegin(graph, center); slot < NeighborEnd(graph, center);
       ++slot) {
    if (!NeighborActive(graph, slot)) {
      continue;
    }
    const std::int32_t neighbor = Neighbor(graph, center, slot);
    first_slot = neighbor == first_neighbor ? slot : first_slot;
    second_slot = neighbor == second_neighbor ? slot : second_slot;
  }
  return first_slot >= 0 && second_slot >= 0 &&
         PairForbiddenBySlots(graph, center, first_slot, second_slot);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Active(const GraphView& graph, const std::int32_t a,
                                                      const std::int32_t b) {
  if (graph.active != nullptr) {
    return graph.active[static_cast<std::int64_t>(a) * graph.dimension + b] != 0U;
  }
  if (a == b) {
    return false;
  }
  const std::int32_t u = a < b ? a : b;
  const std::int32_t v = a < b ? b : a;
  if (graph.complete_graph) {
    const std::int64_t prefix = static_cast<std::int64_t>(u) * (2LL * graph.dimension - u - 1) / 2;
    const std::int64_t edge = prefix + (v - u - 1);
    return edge >= 0 && edge < graph.edge_count && graph.edge_active[edge] != 0U;
  }
  std::int64_t low = 0;
  std::int64_t high = graph.edge_count;
  while (low < high) {
    const std::int64_t middle = low + (high - low) / 2;
    const std::int32_t middle_u = graph.edge_u[middle];
    const std::int32_t middle_v = graph.edge_v[middle];
    if (middle_u < u || (middle_u == u && middle_v < v)) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < graph.edge_count && graph.edge_u[low] == u && graph.edge_v[low] == v &&
         graph.edge_active[low] != 0U;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Fixed(const GraphView& graph, const std::int32_t a,
                                                     const std::int32_t b) {
  if (!Active(graph, a, b)) {
    return false;
  }
  if (graph.fixed_edge != nullptr) {
    const std::int32_t u = a < b ? a : b;
    const std::int32_t v = a < b ? b : a;
    std::int64_t low = 0;
    std::int64_t high = graph.edge_count;
    if (graph.complete_graph) {
      low = static_cast<std::int64_t>(u) * (2LL * graph.dimension - u - 1) / 2 + (v - u - 1);
    } else {
      while (low < high) {
        const std::int64_t middle = low + (high - low) / 2;
        if (graph.edge_u[middle] < u || (graph.edge_u[middle] == u && graph.edge_v[middle] < v)) {
          low = middle + 1;
        } else {
          high = middle;
        }
      }
    }
    if (low >= 0 && low < graph.edge_count && graph.edge_u[low] == u && graph.edge_v[low] == v &&
        graph.fixed_edge[low] != 0U) {
      return true;
    }
  }
  return graph.degree[a] == 2 || graph.degree[b] == 2;
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
  if ((c1 == a && c2 == b) || (c1 == b && c2 == a)) {
    const SmallPath triangle{.size = 3, .node = {a, c, b}};
    return ClosedPathsMayCoverWholeGraph(graph, triangle);
  }
  if (cab + cc1c + ccc2 > Distance(graph, a, c) + Distance(graph, c, b) + Distance(graph, c1, c2)) {
    return false;
  }
  // 对齐 KH 默认 strong_3_opt=0 的两个固定边门禁。
  for (std::int64_t slot = NeighborBegin(graph, c); slot < NeighborEnd(graph, c); ++slot) {
    if (!NeighborActive(graph, slot)) {
      continue;
    }
    const std::int32_t z = Neighbor(graph, c, slot);
    if (z != c1 && z != c2 && graph.degree[z] == 2) {
      return false;
    }
  }
  if (Fixed(graph, c1, c2)) {
    const SmallPath triangle{.size = 3, .node = {c1, c, c2}};
    return ClosedPathsMayCoverWholeGraph(graph, triangle);
  }
  return true;
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
  std::int64_t original = Distance(graph, order[0], order[1]);
  std::int32_t outside_links = 0;
  for (std::int32_t position = 1; position < total - 1; ++position) {
    if (fixed_after[position] != 0U) {
      ++outside_links;
    } else {
      original += Distance(graph, order[position], order[position + 1]);
    }
  }
  // 真实内部路径成本为 L，所有距离非负。令每条外部强制连接成本为 -(L+1)，
  // 则遗漏任意一条连接的排列都不可能严格优于原路径；不能沿用仅适用于
  // 小成本输入的 32 位 INT_MIN 哨兵。入口的 max(cost)<=INT64_MAX/64
  // 与至多 10 节点/3 路径保证这里的正负中间和均不溢出。
  const std::int64_t forced_cost = -(original + 1);
  original += static_cast<std::int64_t>(outside_links) * forced_cost;

  // 当前三路径 continuation 的内部点最多 8 个。旧实现为每个 CUDA 线程分配
  // subset×last 的 int64 Held--Karp 表，编译后产生约 3 KiB local stack，
  // 并让首轮全图扫描完全受
  // local-memory latency 支配。这里枚举同一个排列空间：起点到首个内部点仍
  // 按 KH 原实现使用真实距离，其余转移继续使用相同的固定边负哨兵。因此
  // “存在严格更短重连”的判定与动态规划完全等价，但线程私有状态只有 8 字节。
  std::uint8_t dynamic_order[kMaxPathNodes - 2]{};
  for (std::int32_t node = 0; node < dynamic_nodes; ++node) {
    dynamic_order[node] = static_cast<std::uint8_t>(node);
  }
  for (;;) {
    std::int64_t candidate = Distance(graph, order[dynamic_order[0] + 1], order[0]);
    for (std::int32_t position = 1; position < dynamic_nodes; ++position) {
      candidate += PathTransitionCost(
          graph, order, fixed_after, static_cast<std::int32_t>(dynamic_order[position - 1]) + 1,
          static_cast<std::int32_t>(dynamic_order[position]) + 1, forced_cost);
    }
    candidate += PathTransitionCost(graph, order, fixed_after,
                                    static_cast<std::int32_t>(dynamic_order[dynamic_nodes - 1]) + 1,
                                    total - 1, forced_cost);
    if (candidate < original) {
      return false;
    }

    // 原地生成下一字典序排列，避免递归及 device 动态分配。
    std::int32_t pivot = dynamic_nodes - 2;
    while (pivot >= 0 && dynamic_order[pivot] >= dynamic_order[pivot + 1]) {
      --pivot;
    }
    if (pivot < 0) {
      break;
    }
    std::int32_t successor = dynamic_nodes - 1;
    while (dynamic_order[successor] <= dynamic_order[pivot]) {
      --successor;
    }
    const std::uint8_t pivot_value = dynamic_order[pivot];
    dynamic_order[pivot] = dynamic_order[successor];
    dynamic_order[successor] = pivot_value;
    for (std::int32_t left = pivot + 1, right = dynamic_nodes - 1; left < right; ++left, --right) {
      const std::uint8_t left_value = dynamic_order[left];
      dynamic_order[left] = dynamic_order[right];
      dynamic_order[right] = left_value;
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
        return ClosedPathsMayCoverWholeGraph(graph, paths[i]);
      }
      for (std::int32_t j = 0; j < i && !merged; ++j) {
        const std::int32_t a1 = paths[j].node[0];
        const std::int32_t a2 = paths[j].node[paths[j].size - 1];
        if (j == 0 &&
            ((a1 == a2 && paths[j].size > 1) || (paths[j].size > 2 && Fixed(graph, a1, a2)))) {
          return ClosedPathsMayCoverWholeGraph(graph, paths[j]);
        }

        std::int32_t first_direction = 1;
        std::int32_t second_direction = 1;
        bool join = false;
        bool shared = false;
        if (a2 == b1 || (a2 != b2 && b1 != a1 && Fixed(graph, a2, b1))) {
          if (a1 == b2) {
            return ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
          }
          join = true;
          shared = a2 == b1;
        } else if (a2 == b2 || (a2 != b1 && b2 != a1 && Fixed(graph, a2, b2))) {
          if (a1 == b1) {
            return ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
          }
          join = true;
          shared = a2 == b2;
          second_direction = -1;
        } else if (a1 == b1 || (a1 != b2 && b1 != a2 && Fixed(graph, a1, b1))) {
          if (a2 == b2) {
            return ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
          }
          join = true;
          shared = a1 == b1;
          first_direction = -1;
        } else if (a1 == b2 || (a1 != b1 && b2 != a2 && Fixed(graph, a1, b2))) {
          if (a2 == b1) {
            return ClosedPathsMayCoverWholeGraph(graph, paths[j], &paths[i]);
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

  if (HasAmbiguousPathOverlap(paths, path_count)) {
    return true;
  }

  std::int32_t endpoints[2 * kMaxPathCount]{};
  for (std::int32_t path = 0; path < path_count; ++path) {
    endpoints[2 * path] = paths[path].node[0];
    endpoints[2 * path + 1] = paths[path].node[paths[path].size - 1];
  }
  if (HasCycle(endpoints, path_count)) {
    return ClosedPathsMayCoverWholeGraph(graph, paths[0], path_count > 1 ? &paths[1] : nullptr,
                                         path_count > 2 ? &paths[2] : nullptr);
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

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Opt33(const GraphView& graph, const std::int32_t a1,
                                                     const std::int32_t a2, const std::int32_t a3,
                                                     const std::int32_t b1, const std::int32_t b2,
                                                     const std::int32_t b3) {
  // 任一 constituent edge 与另一条二边路径已能严格改进时，
  // 整个 3+3 path system 必然关闭。先做四个 KH opt23 必要条件，
  // 只把少数 surviving replies 送入精确 path-ordering 枚举。该蕴含只对
  // 两条顶点不交的路径成立；共享端点时通用 oracle 会先合并路径，不能把
  // constituent path 的阴性结果直接提升成整个 path system 的阴性结果。
  const std::int32_t nodes[6] = {a1, a2, a3, b1, b2, b3};
  bool all_distinct = true;
  for (std::int32_t first = 0; first < 6; ++first) {
    for (std::int32_t second = first + 1; second < 6; ++second) {
      all_distinct = all_distinct && nodes[first] != nodes[second];
    }
  }
  if (all_distinct && (!Opt23(graph, a1, a2, b1, b2, b3, Distance(graph, a1, a2),
                              Distance(graph, b1, b2), Distance(graph, b2, b3)) ||
                       !Opt23(graph, a2, a3, b1, b2, b3, Distance(graph, a2, a3),
                              Distance(graph, b1, b2), Distance(graph, b2, b3)) ||
                       !Opt23(graph, b1, b2, a1, a2, a3, Distance(graph, b1, b2),
                              Distance(graph, a1, a2), Distance(graph, a2, a3)) ||
                       !Opt23(graph, b2, b3, a1, a2, a3, Distance(graph, b2, b3),
                              Distance(graph, a1, a2), Distance(graph, a2, a3)))) {
    return false;
  }
  const SmallPath paths[kMaxPathCount] = {
      {.size = 3, .node = {a1, a2, a3}},
      {.size = 3, .node = {b1, b2, b3}},
  };
  return Opt(graph, paths, 2);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool Opt34(const GraphView& graph, const std::int32_t a1,
                                                     const std::int32_t a2, const std::int32_t a3,
                                                     const std::int32_t b1, const std::int32_t b2,
                                                     const std::int32_t b3, const std::int32_t b4) {
  // 3+4 path system 包含七个 2+3 子系统。任一子系统已存在严格更短
  // 重连时，完整 forced-edge 超集也不可能属于最优 tour。该预筛只在
  // 七个角色互异时使用；共享端点继续交给通用 path normalization。
  const std::int32_t nodes[7] = {a1, a2, a3, b1, b2, b3, b4};
  if (AllNodesDistinct(nodes, 7) && (!Opt23(graph, a1, a2, b1, b2, b3, Distance(graph, a1, a2),
                                            Distance(graph, b1, b2), Distance(graph, b2, b3)) ||
                                     !Opt23(graph, a1, a2, b2, b3, b4, Distance(graph, a1, a2),
                                            Distance(graph, b2, b3), Distance(graph, b3, b4)) ||
                                     !Opt23(graph, a2, a3, b1, b2, b3, Distance(graph, a2, a3),
                                            Distance(graph, b1, b2), Distance(graph, b2, b3)) ||
                                     !Opt23(graph, a2, a3, b2, b3, b4, Distance(graph, a2, a3),
                                            Distance(graph, b2, b3), Distance(graph, b3, b4)) ||
                                     !Opt23(graph, b1, b2, a1, a2, a3, Distance(graph, b1, b2),
                                            Distance(graph, a1, a2), Distance(graph, a2, a3)) ||
                                     !Opt23(graph, b2, b3, a1, a2, a3, Distance(graph, b2, b3),
                                            Distance(graph, a1, a2), Distance(graph, a2, a3)) ||
                                     !Opt23(graph, b3, b4, a1, a2, a3, Distance(graph, b3, b4),
                                            Distance(graph, a1, a2), Distance(graph, a2, a3)))) {
    return false;
  }
  const SmallPath paths[kMaxPathCount] = {
      {.size = 3, .node = {a1, a2, a3}},
      {.size = 4, .node = {b1, b2, b3, b4}},
  };
  return Opt(graph, paths, 2);
}

// 以下两个谓词对应 ElimTSP/KH-elim 的 opt243/opt333。先执行原实现中的
// 低阶必要条件，再进入同一个精确 path-ordering 枚举，避免在大多数 reply 上
// 支付 7! 排列的代价。
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Opt243(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t c1,
       const std::int32_t c2, const std::int32_t c3, const std::int32_t c4, const std::int32_t d1,
       const std::int32_t d2, const std::int32_t d3) {
  if (!Opt23(graph, c1, c2, c2, c3, c4, Distance(graph, c1, c2), Distance(graph, c2, c3),
             Distance(graph, c3, c4)) ||
      !Opt23(graph, c1, c2, d1, d2, d3, Distance(graph, c1, c2), Distance(graph, d1, d2),
             Distance(graph, d2, d3)) ||
      !Opt23(graph, d1, d2, c1, c2, c3, Distance(graph, d1, d2), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, d2, d3, c1, c2, c3, Distance(graph, d2, d3), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, a, b, c1, c2, c3, Distance(graph, a, b), Distance(graph, c1, c2),
             Distance(graph, c2, c3))) {
    return false;
  }
  const SmallPath paths[kMaxPathCount] = {
      {.size = 2, .node = {a, b}},
      {.size = 4, .node = {c1, c2, c3, c4}},
      {.size = 3, .node = {d1, d2, d3}},
  };
  return Opt(graph, paths, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Opt244(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t c1,
       const std::int32_t c2, const std::int32_t c3, const std::int32_t c4, const std::int32_t d1,
       const std::int32_t d2, const std::int32_t d3, const std::int32_t d4) {
  const std::int32_t roles[10] = {a, b, c1, c2, c3, c4, d1, d2, d3, d4};
  if (!AllNodesDistinct(roles, 10)) {
    // q10 的每个 DP 位置必须代表不同 Hamilton 顶点；角色重叠时
    // 保守保持 reply 开放，避免把重复位置当成不同顶点。
    return true;
  }
  // 对齐 KH `opt244`：这些 Opt23 是证明成立的必要条件，不只是
  // path-order DP 的性能预筛。漏掉任意一项都可能把可行 reply 误关掉。
  if (!Opt23(graph, c1, c2, c2, c3, c4, Distance(graph, c1, c2), Distance(graph, c2, c3),
             Distance(graph, c3, c4)) ||
      !Opt23(graph, c1, c2, d1, d2, d3, Distance(graph, c1, c2), Distance(graph, d1, d2),
             Distance(graph, d2, d3)) ||
      !Opt23(graph, c1, c2, d2, d3, d4, Distance(graph, c1, c2), Distance(graph, d2, d3),
             Distance(graph, d3, d4)) ||
      !Opt23(graph, d1, d2, c1, c2, c3, Distance(graph, d1, d2), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, d2, d3, c1, c2, c3, Distance(graph, d2, d3), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, d3, d4, c1, c2, c3, Distance(graph, d3, d4), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, a, b, c1, c2, c3, Distance(graph, a, b), Distance(graph, c1, c2),
             Distance(graph, c2, c3))) {
    return false;
  }
  const SmallPath paths[kMaxPathCount] = {
      {.size = 2, .node = {a, b}},
      {.size = 4, .node = {c1, c2, c3, c4}},
      {.size = 4, .node = {d1, d2, d3, d4}},
  };
  return Opt(graph, paths, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Opt253(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t c1,
       const std::int32_t c2, const std::int32_t c3, const std::int32_t c4, const std::int32_t c5,
       const std::int32_t d1, const std::int32_t d2, const std::int32_t d3) {
  const std::int32_t roles[10] = {a, b, c1, c2, c3, c4, c5, d1, d2, d3};
  if (!AllNodesDistinct(roles, 10)) {
    return true;
  }
  if (!Opt23(graph, c1, c2, c2, c3, c4, Distance(graph, c1, c2), Distance(graph, c2, c3),
             Distance(graph, c3, c4)) ||
      !Opt23(graph, c1, c2, c3, c4, c5, Distance(graph, c1, c2), Distance(graph, c3, c4),
             Distance(graph, c4, c5)) ||
      !Opt23(graph, c1, c2, d1, d2, d3, Distance(graph, c1, c2), Distance(graph, d1, d2),
             Distance(graph, d2, d3)) ||
      !Opt23(graph, d1, d2, c1, c2, c3, Distance(graph, d1, d2), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, d2, d3, c1, c2, c3, Distance(graph, d2, d3), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, a, b, c1, c2, c3, Distance(graph, a, b), Distance(graph, c1, c2),
             Distance(graph, c2, c3))) {
    return false;
  }
  const SmallPath paths[kMaxPathCount] = {
      {.size = 2, .node = {a, b}},
      {.size = 5, .node = {c1, c2, c3, c4, c5}},
      {.size = 3, .node = {d1, d2, d3}},
  };
  return Opt(graph, paths, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Opt333(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t b2,
       const std::int32_t c1, const std::int32_t c2, const std::int32_t c3, const std::int32_t d1,
       const std::int32_t d2, const std::int32_t d3) {
  const SmallPath paths[kMaxPathCount] = {
      {.size = 3, .node = {a, b, b2}},
      {.size = 3, .node = {c1, c2, c3}},
      {.size = 3, .node = {d1, d2, d3}},
  };
  return Opt(graph, paths, 3);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Opt343(const GraphView& graph, const std::int32_t a1, const std::int32_t a2, const std::int32_t a3,
       const std::int32_t b1, const std::int32_t b2, const std::int32_t b3, const std::int32_t b4,
       const std::int32_t c1, const std::int32_t c2, const std::int32_t c3) {
  const std::int32_t roles[10] = {a1, a2, a3, b1, b2, b3, b4, c1, c2, c3};
  if (!AllNodesDistinct(roles, 10)) {
    return true;
  }
  if (!Opt23(graph, b1, b2, b2, b3, b4, Distance(graph, b1, b2), Distance(graph, b2, b3),
             Distance(graph, b3, b4)) ||
      !Opt23(graph, b1, b2, a1, a2, a3, Distance(graph, b1, b2), Distance(graph, a1, a2),
             Distance(graph, a2, a3)) ||
      !Opt23(graph, b1, b2, c1, c2, c3, Distance(graph, b1, b2), Distance(graph, c1, c2),
             Distance(graph, c2, c3)) ||
      !Opt23(graph, a1, a2, b1, b2, b3, Distance(graph, a1, a2), Distance(graph, b1, b2),
             Distance(graph, b2, b3)) ||
      !Opt23(graph, a2, a3, b1, b2, b3, Distance(graph, a2, a3), Distance(graph, b1, b2),
             Distance(graph, b2, b3)) ||
      !Opt23(graph, c1, c2, b1, b2, b3, Distance(graph, c1, c2), Distance(graph, b1, b2),
             Distance(graph, b2, b3)) ||
      !Opt23(graph, c2, c3, b1, b2, b3, Distance(graph, c2, c3), Distance(graph, b1, b2),
             Distance(graph, b2, b3)) ||
      !Opt23(graph, a2, a3, b2, b3, b4, Distance(graph, a2, a3), Distance(graph, b2, b3),
             Distance(graph, b3, b4)) ||
      !Opt23(graph, a2, a3, c1, c2, c3, Distance(graph, a2, a3), Distance(graph, c1, c2),
             Distance(graph, c2, c3))) {
    return false;
  }
  const SmallPath paths[kMaxPathCount] = {
      {.size = 3, .node = {a1, a2, a3}},
      {.size = 4, .node = {b1, b2, b3, b4}},
      {.size = 3, .node = {c1, c2, c3}},
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
HasProperCycle233(const GraphView& graph, const std::int32_t a, const std::int32_t b,
                  const std::int32_t c1, const std::int32_t c, const std::int32_t c2,
                  const std::int32_t d1, const std::int32_t d, const std::int32_t d2) {
  if (!HasCycle222(a, b, c1, c2, d1, d2)) {
    return false;
  }
  // 端点收缩后出现的三角环可能就是完整的五节点 Hamilton 环。
  // fast filter 必须和通用 normalization 一样区分真子环与覆盖全图的环。
  const SmallPath first{.size = 2, .node = {a, b}};
  const SmallPath second{.size = 3, .node = {c1, c, c2}};
  const SmallPath third{.size = 3, .node = {d1, d, d2}};
  return !ClosedPathsMayCoverWholeGraph(graph, first, &second, &third);
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
Compatible(const GraphView& graph, const std::int32_t a, const std::int32_t b, const std::int32_t c,
           const std::int32_t d, const std::int64_t cab) {
  const std::int64_t ccd = Distance(graph, c, d);
  return c == a || c == b || d == a || d == b || ccd > cab ||
         (Active(graph, c, d) && (cab + ccd <= Distance(graph, a, c) + Distance(graph, d, b) ||
                                  cab + ccd <= Distance(graph, a, d) + Distance(graph, c, b)));
}

// 完整移植 KH-ElimTSP `extra_edge_opt(..., e=0, f=0)` 的 -e1 语义。
// 返回 true 表示六个可揭示端点都仍存在至少一个无法由局部重连关闭的 Hamilton
// reply；任一端点的全部 reply 都被关闭时返回 false，当前父 reply 即获证。
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
ExtraEdgeOpt1(const GraphView& graph, const std::int32_t a, const std::int32_t b,
              const std::int32_t c1, const std::int32_t c, const std::int32_t c2,
              const std::int32_t d1, const std::int32_t d, const std::int32_t d2) {
  const std::int32_t path_endpoints[2][2] = {{c1, c2}, {d1, d2}};
  const std::int32_t path_centers[2] = {c, d};
  for (std::int32_t path = 0; path < 2; ++path) {
    const std::int32_t other = 1 - path;
    for (std::int32_t side = 0; side < 2; ++side) {
      const std::int32_t endpoint = path_endpoints[path][side];
      const std::int32_t opposite = path_endpoints[path][1 - side];
      if (endpoint == a || endpoint == b || endpoint == path_endpoints[other][0] ||
          endpoint == path_endpoints[other][1]) {
        continue;
      }
      bool unresolved = false;
      for (std::int64_t slot = NeighborBegin(graph, endpoint); slot < NeighborEnd(graph, endpoint);
           ++slot) {
        if (!NeighborActive(graph, slot)) {
          continue;
        }
        const std::int32_t extension = Neighbor(graph, endpoint, slot);
        if (extension == path_centers[path] || extension == opposite ||
            extension == path_centers[other]) {
          continue;
        }
        if (PairForbidden(graph, endpoint, path_centers[path], extension)) {
          continue;
        }
        if (Opt243(graph, a, b, extension, endpoint, path_centers[path], opposite,
                   path_endpoints[other][0], path_centers[other], path_endpoints[other][1])) {
          unresolved = true;
          break;
        }
      }
      if (!unresolved) {
        return false;
      }
    }
  }

  const std::int32_t target_endpoints[2] = {b, a};
  const std::int32_t target_opposites[2] = {a, b};
  for (std::int32_t side = 0; side < 2; ++side) {
    const std::int32_t endpoint = target_endpoints[side];
    const std::int32_t opposite = target_opposites[side];
    if (endpoint == c1 || endpoint == c2 || endpoint == d1 || endpoint == d2) {
      continue;
    }
    bool unresolved = false;
    for (std::int64_t slot = NeighborBegin(graph, endpoint); slot < NeighborEnd(graph, endpoint);
         ++slot) {
      if (!NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t extension = Neighbor(graph, endpoint, slot);
      if (extension == opposite || extension == c || extension == d) {
        continue;
      }
      if (PairForbidden(graph, endpoint, opposite, extension)) {
        continue;
      }
      if (Opt333(graph, opposite, endpoint, extension, c1, c, c2, d1, d, d2)) {
        unresolved = true;
        break;
      }
    }
    if (!unresolved) {
      return false;
    }
  }
  return true;
}

// 一个固定 (c1-c-c2, d1-d-d2) Hamilton reply 是否仍可能出现在最优巡回中。
// 独立函数供 CTA continuation engine 把不同 reply 分发到多个 lane。
template <bool EnableExtraEdge1 = false>
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
ReplyPassesFastFilters(const GraphView& graph, const std::int32_t a, const std::int32_t b,
                       const std::int32_t c1, const std::int32_t c, const std::int32_t c2,
                       const std::int32_t d1, const std::int32_t d, const std::int32_t d2) {
  if (c1 == d || c2 == d || d1 == c || d2 == c || (c2 == a && c1 == b) || (c2 == b && c1 == a) ||
      ((d1 == a || d1 == b) && (c1 == d1 || c2 == d1)) ||
      (d2 == a && (d1 == b || c1 == d2 || c2 == d2)) ||
      (d2 == b && (d1 == a || c1 == d2 || c2 == d2)) ||
      (d2 == c1 && (d1 == c2 || a == d2 || b == d2)) ||
      (d2 == c2 && (d1 == c1 || a == d2 || b == d2)) ||
      HasProperCycle233(graph, a, b, c1, c, c2, d1, d, d2)) {
    return false;
  }
  const std::int64_t cab = Distance(graph, a, b);
  const std::int64_t cc1c = Distance(graph, c1, c);
  const std::int64_t ccc2 = Distance(graph, c, c2);
  const std::int64_t cd1d = Distance(graph, d1, d);
  const std::int64_t cdd2 = Distance(graph, d, d2);
  return Opt22(graph, c1, c, a, b, cc1c, cab) && Opt22(graph, c, c2, a, b, ccc2, cab) &&
         Opt23(graph, a, b, c1, c, c2, cab, cc1c, ccc2) && Opt22(graph, d, d1, a, b, cd1d, cab) &&
         Opt22(graph, d, d1, c1, c, cd1d, cc1c) && Opt22(graph, d, d1, c, c2, cd1d, ccc2) &&
         Opt23(graph, d, d1, c1, c, c2, cd1d, cc1c, ccc2) && Opt222(graph, a, b, c1, c, d, d1) &&
         Opt222(graph, a, b, c, c2, d, d1) && Opt232(graph, a, b, c1, c, c2, d1, d) &&
         Opt22(graph, d, d2, a, b, cdd2, cab) && Opt22(graph, d, d2, c1, c, cdd2, cc1c) &&
         Opt22(graph, d, d2, c, c2, cdd2, ccc2) && Opt23(graph, a, b, d1, d, d2, cab, cd1d, cdd2) &&
         Opt23(graph, d, d2, c1, c, c2, cdd2, cc1c, ccc2) &&
         Opt23(graph, c1, c, d1, d, d2, cc1c, cd1d, cdd2) &&
         Opt23(graph, c, c2, d1, d, d2, ccc2, cd1d, cdd2);
}

template <bool EnableExtraEdge1 = false>
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
ReplyAdmitsTour(const GraphView& graph, const std::int32_t a, const std::int32_t b,
                const std::int32_t c1, const std::int32_t c, const std::int32_t c2,
                const std::int32_t d1, const std::int32_t d, const std::int32_t d2) {
  return ReplyPassesFastFilters<EnableExtraEdge1>(graph, a, b, c1, c, c2, d1, d, d2) &&
         Opt233(graph, a, b, c1, c, c2, d1, d, d2) &&
         (!EnableExtraEdge1 || ExtraEdgeOpt1(graph, a, b, c1, c, c2, d1, d, d2));
}

// 返回 true 表示固定 c,d 的全部 Hamilton replies 都已被整数局部改进关闭。
template <bool EnableExtraEdge1 = false>
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE bool
CanEliminateWithWitness(const GraphView& graph, const std::int32_t a, const std::int32_t b,
                        const std::int32_t c, const std::int32_t d) {
  if (a == b || c == d || !Active(graph, a, b) ||
      Compatible(graph, a, b, c, d, Distance(graph, a, b))) {
    return false;
  }
  const std::int64_t cab = Distance(graph, a, b);
  for (std::int64_t c_first = NeighborBegin(graph, c); c_first < NeighborEnd(graph, c); ++c_first) {
    if (!NeighborActive(graph, c_first)) {
      continue;
    }
    const std::int32_t c1 = Neighbor(graph, c, c_first);
    if (c1 == d) {
      continue;
    }
    const std::int64_t cc1c = Distance(graph, c1, c);
    if (!Opt22(graph, c1, c, a, b, cc1c, cab)) {
      continue;
    }
    for (std::int64_t c_second = c_first + 1; c_second < NeighborEnd(graph, c); ++c_second) {
      if (!NeighborActive(graph, c_second)) {
        continue;
      }
      if (PairForbiddenBySlots(graph, c, c_first, c_second)) {
        continue;
      }
      const std::int32_t c2 = Neighbor(graph, c, c_second);
      if (c2 == d || (c2 == a && c1 == b) || (c2 == b && c1 == a)) {
        continue;
      }
      const std::int64_t ccc2 = Distance(graph, c, c2);
      if (!Opt22(graph, c, c2, a, b, ccc2, cab) ||
          !Opt23(graph, a, b, c1, c, c2, cab, cc1c, ccc2)) {
        continue;
      }
      for (std::int64_t d_first = NeighborBegin(graph, d); d_first < NeighborEnd(graph, d);
           ++d_first) {
        if (!NeighborActive(graph, d_first)) {
          continue;
        }
        const std::int32_t d1 = Neighbor(graph, d, d_first);
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
        for (std::int64_t d_second = d_first + 1; d_second < NeighborEnd(graph, d); ++d_second) {
          if (!NeighborActive(graph, d_second)) {
            continue;
          }
          if (PairForbiddenBySlots(graph, d, d_first, d_second)) {
            continue;
          }
          const std::int32_t d2 = Neighbor(graph, d, d_second);
          if (d2 == c || (d2 == a && (d1 == b || c1 == d2 || c2 == d2)) ||
              (d2 == b && (d1 == a || c1 == d2 || c2 == d2)) ||
              (d2 == c1 && (d1 == c2 || a == d2 || b == d2)) ||
              (d2 == c2 && (d1 == c1 || a == d2 || b == d2)) ||
              HasProperCycle233(graph, a, b, c1, c, c2, d1, d, d2)) {
            continue;
          }
          const std::int64_t cdd2 = Distance(graph, d, d2);
          if (Opt22(graph, d, d2, a, b, cdd2, cab) && Opt22(graph, d, d2, c1, c, cdd2, cc1c) &&
              Opt22(graph, d, d2, c, c2, cdd2, ccc2) &&
              Opt23(graph, a, b, d1, d, d2, cab, cd1d, cdd2) &&
              Opt23(graph, d, d2, c1, c, c2, cdd2, cc1c, ccc2) &&
              Opt23(graph, c1, c, d1, d, d2, cc1c, cd1d, cdd2) &&
              Opt23(graph, c, c2, d1, d, d2, ccc2, cd1d, cdd2) &&
              Opt233(graph, a, b, c1, c, c2, d1, d, d2) &&
              (!EnableExtraEdge1 || ExtraEdgeOpt1(graph, a, b, c1, c, c2, d1, d, d2))) {
            // 找到一个不能关闭的 reply，固定 c,d 不能证明目标边。
            return false;
          }
        }
      }
    }
  }
  return true;
}

CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE void
InsertWitnessCandidate(const GraphView& graph, const std::int32_t a, const std::int32_t b,
                       const std::int32_t node, const std::int32_t candidate_limit,
                       std::int32_t* const candidate_nodes, std::int64_t* const candidate_scores,
                       std::int32_t* const candidate_count) {
  if (node == a || node == b) {
    return;
  }
  for (std::int32_t existing = 0; existing < *candidate_count; ++existing) {
    if (candidate_nodes[existing] == node) {
      return;
    }
  }
  const std::int64_t score = Distance(graph, a, node) + Distance(graph, node, b);
  if (*candidate_count == candidate_limit && (score > candidate_scores[candidate_limit - 1] ||
                                              (score == candidate_scores[candidate_limit - 1] &&
                                               node >= candidate_nodes[candidate_limit - 1]))) {
    return;
  }
  std::int32_t position =
      *candidate_count < candidate_limit ? (*candidate_count)++ : candidate_limit - 1;
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

template <bool EnableExtraEdge1 = false>
CUDAEE_QUICK_HS_HD CUDAEE_QUICK_HS_INLINE Witness
FindWitness(const GraphView& graph, const std::int32_t a, const std::int32_t b,
            const std::int32_t candidate_limit = 10, const std::int32_t pair_trial_limit = 10,
            const bool include_two_hop = false) {
  Witness result;
  if (!Active(graph, a, b) || graph.degree[a] <= 2 || graph.degree[b] <= 2 || candidate_limit < 2 ||
      candidate_limit > kMaxPotentialNodes || pair_trial_limit < 0) {
    return result;
  }
  std::int32_t candidate_nodes[kMaxPotentialNodes]{};
  std::int64_t candidate_scores[kMaxPotentialNodes]{};
  std::int32_t candidate_count = 0;
  for (std::int32_t side = 0; side < 2; ++side) {
    const std::int32_t from = side == 0 ? a : b;
    for (std::int64_t slot = NeighborBegin(graph, from); slot < NeighborEnd(graph, from); ++slot) {
      if (!NeighborActive(graph, slot)) {
        continue;
      }
      const std::int32_t node = Neighbor(graph, from, slot);
      InsertWitnessCandidate(graph, a, b, node, candidate_limit, candidate_nodes, candidate_scores,
                             &candidate_count);
    }
  }
  if (include_two_hop) {
    // 对齐 KH 非 -q 路径：把两个端点邻点的邻点也纳入同一 (cost,node)
    // top-k 集合。仅扩大 witness 搜索，不改变任何局部证明谓词。
    for (std::int32_t side = 0; side < 2; ++side) {
      const std::int32_t from = side == 0 ? a : b;
      for (std::int64_t first = NeighborBegin(graph, from); first < NeighborEnd(graph, from);
           ++first) {
        if (!NeighborActive(graph, first)) {
          continue;
        }
        const std::int32_t middle = Neighbor(graph, from, first);
        for (std::int64_t second = NeighborBegin(graph, middle);
             second < NeighborEnd(graph, middle); ++second) {
          if (!NeighborActive(graph, second)) {
            continue;
          }
          InsertWitnessCandidate(graph, a, b, Neighbor(graph, middle, second), candidate_limit,
                                 candidate_nodes, candidate_scores, &candidate_count);
        }
      }
    }
  }

  std::int32_t pair_trials = 0;
  for (std::int32_t first = 1; first < candidate_count; ++first) {
    const std::int32_t c = candidate_nodes[first];
    for (std::int32_t second = 0; second < first; ++second) {
      ++pair_trials;
      if (pair_trial_limit != 0 && pair_trials > pair_trial_limit) {
        return result;
      }
      const std::int32_t d = candidate_nodes[second];
      if (!Compatible(graph, a, b, c, d, Distance(graph, a, b)) &&
          CanEliminateWithWitness<EnableExtraEdge1>(graph, a, b, c, d)) {
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
