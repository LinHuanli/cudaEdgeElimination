#include "quick_reply_cache.cuh"

#include <cub/device/device_scan.cuh>

#include <chrono>
#include <limits>

namespace cudaee::detail {
namespace {
constexpr unsigned kThreads = 128;
using Clock = std::chrono::steady_clock;
double Milliseconds(const Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

__global__ void Candidates(const quick_hs::GraphView graph, const int roots, const int* work_edges,
                           const std::uint8_t* protected_edges, const int limit, const bool two_hop,
                           const std::uint64_t snapshot, QuickReplyRow* rows) {
  const int work = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (work >= roots)
    return;
  const int root = work_edges[work], a = graph.edge_u[root], b = graph.edge_v[root];
  int nodes[quick_hs::kMaxPotentialNodes];
  std::int64_t scores[quick_hs::kMaxPotentialNodes];
  int count = 0;
  if (graph.edge_active[root] && !protected_edges[root] && graph.degree[a] > 2 &&
      graph.degree[b] > 2) {
    // 与原 kernel 相同的遍历与 (distance,node) 插入顺序，不按回复数改 OR 域。
    for (int side = 0; side < 2; ++side) {
      const int from = side == 0 ? a : b;
      for (auto slot = quick_hs::NeighborBegin(graph, from);
           slot < quick_hs::NeighborEnd(graph, from); ++slot)
        if (quick_hs::NeighborActive(graph, slot))
          quick_hs::InsertWitnessCandidate(graph, a, b, quick_hs::Neighbor(graph, from, slot),
                                           limit, nodes, scores, &count);
    }
    if (two_hop)
      for (int side = 0; side < 2; ++side) {
        const int from = side == 0 ? a : b;
        for (auto first = quick_hs::NeighborBegin(graph, from);
             first < quick_hs::NeighborEnd(graph, from); ++first) {
          if (!quick_hs::NeighborActive(graph, first))
            continue;
          const int middle = quick_hs::Neighbor(graph, from, first);
          for (auto second = quick_hs::NeighborBegin(graph, middle);
               second < quick_hs::NeighborEnd(graph, middle); ++second)
            if (quick_hs::NeighborActive(graph, second))
              quick_hs::InsertWitnessCandidate(graph, a, b,
                                               quick_hs::Neighbor(graph, middle, second), limit,
                                               nodes, scores, &count);
        }
      }
  }
  for (int i = 0; i < limit; ++i)
    rows[static_cast<std::int64_t>(work) * limit + i] = {root, i < count ? nodes[i] : -1,
                                                         i < count ? scores[i] : 0, snapshot};
}

__device__ bool DecodeSlots(const std::int64_t degree, const std::uint64_t ordinal,
                            std::int64_t* first, std::int64_t* second) {
  if (degree < 2 || ordinal >= static_cast<std::uint64_t>(degree * (degree - 1) / 2))
    return false;
  std::int64_t low = 0, high = degree - 2;
  while (low < high) {
    const auto middle = (low + high + 1) / 2;
    if (static_cast<std::uint64_t>(middle * (2 * degree - middle - 1) / 2) <= ordinal)
      low = middle;
    else
      high = middle - 1;
  }
  *first = low;
  *second = low + 1 + static_cast<std::int64_t>(ordinal) - low * (2 * degree - low - 1) / 2;
  return true;
}

__device__ bool ProposedPair(const quick_hs::GraphView graph, const QuickReplyRow row,
                             const std::int64_t first, const std::int64_t second) {
  const int c = row.center, a = graph.edge_u[row.root], b = graph.edge_v[row.root];
  if (!quick_hs::NeighborActive(graph, first) || !quick_hs::NeighborActive(graph, second) ||
      quick_hs::PairForbiddenBySlots(graph, c, first, second))
    return false;
  const int p = quick_hs::Neighbor(graph, c, first), q = quick_hs::Neighbor(graph, c, second);
  const auto ab = quick_hs::Distance(graph, a, b), pc = quick_hs::Distance(graph, p, c),
             cq = quick_hs::Distance(graph, c, q);
  // 只提前执行 ReplyPassesFastFilters 对单个中心的必要条件。
  // Opt22 的共享端点特例不能用 main_edge::Compatible 的纯距离公式替代。
  return quick_hs::Opt22(graph, p, c, a, b, pc, ab) && quick_hs::Opt22(graph, c, q, a, b, cq, ab) &&
         quick_hs::Opt23(graph, a, b, p, c, q, ab, pc, cq);
}

template <bool Write>
__global__ void Compact(const quick_hs::GraphView graph, const QuickReplyRow* rows,
                        const std::uint64_t* offsets, std::uint64_t* counts,
                        std::uint64_t* ordinals, unsigned long long* counters) {
  const int index = static_cast<int>(blockIdx.x);
  const auto row = rows[index];
  const auto begin = row.center < 0 ? 0 : quick_hs::NeighborBegin(graph, row.center);
  const auto degree = row.center < 0 ? 0 : quick_hs::NeighborEnd(graph, row.center) - begin;
  const auto pairs = static_cast<std::uint64_t>(degree * (degree - 1) / 2);
  const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5U;
  __shared__ unsigned warp_counts[kThreads / 32];
  std::uint64_t written = 0;
  for (std::uint64_t window = 0; window < pairs; window += kThreads) {
    const auto ordinal = window + threadIdx.x;
    std::int64_t first{}, second{};
    const bool allowed = DecodeSlots(degree, ordinal, &first, &second) &&
                         ProposedPair(graph, row, begin + first, begin + second);
    const unsigned ballot = __ballot_sync(0xffffffffU, allowed);
    if (lane == 0)
      warp_counts[warp] = __popc(ballot);
    __syncthreads();
    unsigned prefix = 0, window_count = 0;
    for (unsigned w = 0; w < kThreads / 32; ++w) {
      if (w < warp)
        prefix += warp_counts[w];
      window_count += warp_counts[w];
    }
    if constexpr (Write) {
      const auto position = written + prefix + __popc(ballot & ((1U << lane) - 1U));
      if (allowed) {
        if (position < counts[index])
          ordinals[offsets[index] + position] = ordinal;
        else
          atomicAdd(counters + 1, 1ULL);
      }
    }
    written += window_count;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if constexpr (Write) {
      if (written != counts[index])
        atomicAdd(counters + 1, 1ULL);
    } else {
      counts[index] = written;
      atomicAdd(counters, static_cast<unsigned long long>(pairs));
    }
  }
}

// 独立 verifier：自然 CSR 双循环，不调用 proposer 的 ordinal 解码、
// Opt22/Opt23 或 PairForbiddenBySlots。只共用精确距离和只读图访问。
__device__ bool ReplayPair(const quick_hs::GraphView g, const QuickReplyRow row,
                           const std::int64_t first, const std::int64_t second,
                           const std::uint64_t ordinal) {
  const int c = row.center, a = g.edge_u[row.root], b = g.edge_v[row.root];
  if (!quick_hs::NeighborActive(g, first) || !quick_hs::NeighborActive(g, second))
    return false;
  if (g.pair_offsets && g.nonpair_mask && g.nonpair_mask[g.pair_offsets[c] + ordinal])
    return false;
  const int p = quick_hs::Neighbor(g, c, first), q = quick_hs::Neighbor(g, c, second);
  const auto ab = quick_hs::Distance(g, a, b), pc = quick_hs::Distance(g, p, c),
             cq = quick_hs::Distance(g, c, q);
  if (p != a && p != b && pc + ab > quick_hs::Distance(g, p, a) + quick_hs::Distance(g, b, c) &&
      pc + ab > quick_hs::Distance(g, p, b) + quick_hs::Distance(g, a, c))
    return false;
  if (q != a && q != b && cq + ab > quick_hs::Distance(g, c, a) + quick_hs::Distance(g, b, q) &&
      cq + ab > quick_hs::Distance(g, c, b) + quick_hs::Distance(g, a, q))
    return false;
  // c 排除 root 两端，因此这两个三角形都恰有三个不同顶点。
  if ((p == a && q == b) || (p == b && q == a))
    return g.dimension == 3;
  if (ab + pc + cq >
      quick_hs::Distance(g, a, c) + quick_hs::Distance(g, c, b) + quick_hs::Distance(g, p, q))
    return false;
  for (auto slot = quick_hs::NeighborBegin(g, c); slot < quick_hs::NeighborEnd(g, c); ++slot) {
    const int z = quick_hs::Neighbor(g, c, slot);
    if (quick_hs::NeighborActive(g, slot) && z != p && z != q && g.degree[z] == 2)
      return false;
  }
  // fixed(p,q) 必须先存在活动边；显式 fixed 和 degree-2 隐含 fixed 均覆盖。
  for (auto slot = quick_hs::NeighborBegin(g, p); slot < quick_hs::NeighborEnd(g, p); ++slot) {
    if (quick_hs::Neighbor(g, p, slot) != q || !quick_hs::NeighborActive(g, slot))
      continue;
    const bool fixed = (g.fixed_edge && g.fixed_edge[g.neighbor_edge_ids[slot]]) ||
                       g.degree[p] == 2 || g.degree[q] == 2;
    if (fixed)
      return g.dimension == 3;
  }
  return true;
}

__global__ void Verify(const quick_hs::GraphView graph, const QuickReplyView cache,
                       const int* work_edges, const std::uint64_t snapshot, std::uint8_t* verified,
                       unsigned long long* rejected) {
  const int index = static_cast<int>(blockIdx.x);
  const auto row = cache.rows[index];
  const auto begin = cache.offsets[index], end = cache.offsets[index + 1];
  bool invalid = row.root != work_edges[index / cache.stride] || row.root < 0 ||
                 row.root >= graph.edge_count || row.snapshot != snapshot || begin > end ||
                 end > cache.total || (end != begin && cache.ordinals == nullptr) ||
                 (index == 0 && begin != 0) ||
                 (index + 1 == cache.roots * cache.stride && end != cache.total);
  if (!invalid && row.center >= 0)
    invalid = row.center >= graph.dimension || row.center == graph.edge_u[row.root] ||
              row.center == graph.edge_v[row.root];
  if (row.center < 0)
    invalid = invalid || row.center != -1 || begin != end;
  if (invalid || row.center < 0) {
    if (threadIdx.x == 0) {
      verified[index] = invalid ? 0 : 1;
      if (invalid)
        atomicAdd(rejected, 1ULL);
    }
    return;
  }
  const auto first_slot = quick_hs::NeighborBegin(graph, row.center);
  const auto degree = quick_hs::NeighborEnd(graph, row.center) - first_slot;
  const auto pairs = static_cast<std::uint64_t>(degree * (degree - 1) / 2);
  // 精确验证每个全集元素是否出现，另查严格递增。比仅比较计数更强：
  // 既不能漏掉合法回复，也不能靠重复另一条回复伪造相同计数。
  for (std::uint64_t entry = begin + threadIdx.x; entry < end; entry += blockDim.x)
    invalid = invalid || cache.ordinals[entry] >= pairs ||
              (entry > begin && cache.ordinals[entry - 1] >= cache.ordinals[entry]);
  for (std::int64_t first = threadIdx.x; first + 1 < degree; first += blockDim.x) {
    std::uint64_t ordinal = static_cast<std::uint64_t>(first * (2 * degree - first - 1) / 2);
    std::uint64_t low = begin, high = end;
    while (low < high) {
      const auto middle = low + (high - low) / 2;
      if (cache.ordinals[middle] < ordinal)
        low = middle + 1;
      else
        high = middle;
    }
    for (auto second = first + 1; second < degree; ++second, ++ordinal) {
      const bool expected =
          ReplayPair(graph, row, first_slot + first, first_slot + second, ordinal);
      const bool present = low < end && cache.ordinals[low] == ordinal;
      invalid = invalid || expected != present;
      if (present)
        ++low;
    }
  }
  const bool failed = __syncthreads_or(invalid);
  if (threadIdx.x == 0) {
    verified[index] = failed ? 0 : 1;
    if (failed)
      atomicAdd(rejected, 1ULL);
  }
}
} // namespace

QuickReplyCache::QuickReplyCache(const int device)
    : rows_(device), counts_(device), offsets_(device), ordinals_(device), verified_(device),
      scan_temp_(device), counters_(device) {}

QuickReplyView QuickReplyCache::view() const {
  return {rows_.get(), offsets_.get(), ordinals_.get(), verified_.get(), roots_, stride_, total_};
}

bool QuickReplyCache::Validate(const quick_hs::GraphView graph, const int* work_edges,
                               const std::uint64_t snapshot) {
  CheckWorkspaceCuda(cudaMemset(counters_.get() + 1, 0, sizeof(unsigned long long)));
  Verify<<<row_count_, kThreads>>>(graph, view(), work_edges, snapshot, verified_.get(),
                                   counters_.get() + 1);
  CheckWorkspaceCuda(cudaGetLastError());
  unsigned long long failures{};
  CheckWorkspaceCuda(
      cudaMemcpy(&failures, counters_.get() + 1, sizeof(failures), cudaMemcpyDeviceToHost));
  return failures == 0;
}

QuickReplyCacheMetrics QuickReplyCache::Build(const quick_hs::GraphView graph, const int work_count,
                                              const int* work_edges,
                                              const std::uint8_t* protected_edges,
                                              const int candidates, const bool two_hop,
                                              const std::uint64_t snapshot) {
  const auto begin = Clock::now();
  if (work_count <= 0 || candidates < 2 || candidates > quick_hs::kMaxPotentialNodes ||
      work_count > (std::numeric_limits<int>::max() - 1) / candidates || graph.dimension < 3 ||
      !graph.row_offsets || !graph.neighbor_edge_ids || !graph.edge_active || !work_edges ||
      !protected_edges)
    throw std::invalid_argument("Quick reply cache 需要合法 CSR/root 域");
  roots_ = work_count;
  stride_ = candidates;
  row_count_ = work_count * candidates;
  const auto maximum_pairs = static_cast<std::uint64_t>(graph.dimension - 1) *
                             static_cast<std::uint64_t>(graph.dimension - 2) / 2;
  if (maximum_pairs > std::numeric_limits<std::uint64_t>::max() / row_count_)
    throw std::overflow_error("Quick reply scan 总数溢出");
  rows_.Reserve(row_count_);
  counts_.Reserve(static_cast<std::uint64_t>(row_count_) + 1);
  offsets_.Reserve(static_cast<std::uint64_t>(row_count_) + 1);
  verified_.Reserve(row_count_);
  counters_.Reserve(2);
  CheckWorkspaceCuda(cudaMemset(counters_.get(), 0, counters_.bytes()));
  CheckWorkspaceCuda(cudaMemset(counts_.get() + row_count_, 0, sizeof(std::uint64_t)));
  Candidates<<<(work_count + kThreads - 1) / kThreads, kThreads>>>(
      graph, roots_, work_edges, protected_edges, stride_, two_hop, snapshot, rows_.get());
  CheckWorkspaceCuda(cudaGetLastError());
  Compact<false><<<row_count_, kThreads>>>(graph, rows_.get(), nullptr, counts_.get(), nullptr,
                                           counters_.get());
  CheckWorkspaceCuda(cudaGetLastError());
  std::size_t temporary_bytes{};
  CheckWorkspaceCuda(cub::DeviceScan::ExclusiveSum(nullptr, temporary_bytes, counts_.get(),
                                                   offsets_.get(), row_count_ + 1));
  scan_temp_.Reserve(temporary_bytes);
  CheckWorkspaceCuda(cub::DeviceScan::ExclusiveSum(scan_temp_.get(), temporary_bytes, counts_.get(),
                                                   offsets_.get(), row_count_ + 1));
  CheckWorkspaceCuda(
      cudaMemcpy(&total_, offsets_.get() + row_count_, sizeof(total_), cudaMemcpyDeviceToHost));
  ordinals_.Reserve(total_);
  Compact<true><<<row_count_, kThreads>>>(graph, rows_.get(), offsets_.get(), counts_.get(),
                                          ordinals_.get(), counters_.get());
  CheckWorkspaceCuda(cudaGetLastError());
  unsigned long long counters[2]{};
  CheckWorkspaceCuda(
      cudaMemcpy(counters, counters_.get(), sizeof(counters), cudaMemcpyDeviceToHost));
  if (counters[1])
    throw std::logic_error("Quick reply count/write 不一致，禁止消费不完整流");
  QuickReplyCacheMetrics metrics;
  metrics.raw_pairs = counters[0];
  metrics.compact_pairs = total_;
  metrics.bytes = rows_.bytes() + counts_.bytes() + offsets_.bytes() + ordinals_.bytes() +
                  verified_.bytes() + scan_temp_.bytes() + counters_.bytes();
  metrics.build_ms = Milliseconds(begin);
  const auto validation_begin = Clock::now();
  if (!Validate(graph, work_edges, snapshot))
    throw std::logic_error("Quick reply 独立 GPU 覆盖检查失败，禁止删边");
  metrics.validation_ms = Milliseconds(validation_begin);
  return metrics;
}
} // namespace cudaee::detail
