#pragma once

#include "quick_hs_predicate.hpp"

#include <cstdint>

namespace cudaee::detail::resident_transaction {

// epoch 的图、fixed、non-pair 均只读。dual 在 proposal 前独立封存；
// 所有授权完成前，不能发布任何一种状态修改。
struct EpochSnapshot {
  quick_hs::GraphView graph;
  std::uint64_t sequence{};
};

struct PendingDelta {
  const std::uint8_t* deleted{};
  const std::uint8_t* fixed{};
  const std::uint8_t* nonpair{};
};

enum InvalidState : std::int32_t {
  kValid = 0,
  kDegreeFloor = 1,
  kFixedDegree = 2,
  kDeleteFixedConflict = 4,
  kNoAllowedPair = 8,
  kProperFixedCycle = 16,
};

enum class EpochPhase : std::uint8_t { kProposal, kReplay, kValidated, kPublished };

struct CommitGate {
  std::uint64_t sequence{};
  EpochPhase phase{EpochPhase::kProposal};

  [[nodiscard]] bool FinishReplay(const std::uint64_t replay_sequence) {
    if (phase != EpochPhase::kProposal || sequence != replay_sequence) {
      return false;
    }
    phase = EpochPhase::kReplay;
    return true;
  }
  [[nodiscard]] bool Validate(const std::int32_t invalid) {
    if (phase != EpochPhase::kReplay || invalid != kValid) {
      return false;
    }
    phase = EpochPhase::kValidated;
    return true;
  }
  [[nodiscard]] bool Publish(const std::uint64_t publish_sequence) {
    if (phase != EpochPhase::kValidated || sequence != publish_sequence) {
      return false;
    }
    phase = EpochPhase::kPublished;
    return true;
  }
};

#if defined(__CUDACC__)
#define CUDAEE_TRANSACTION_HD __host__ __device__
#else
#define CUDAEE_TRANSACTION_HD
#endif

CUDAEE_TRANSACTION_HD inline bool Survives(const quick_hs::GraphView& graph,
                                           const PendingDelta delta, const std::int32_t edge) {
  return graph.edge_active[edge] != 0U && (delta.deleted == nullptr || delta.deleted[edge] == 0U);
}

CUDAEE_TRANSACTION_HD inline bool PairForbidden(const quick_hs::GraphView& graph,
                                                const PendingDelta delta, const std::int32_t center,
                                                const std::int64_t first,
                                                const std::int64_t second) {
  if (quick_hs::PairForbiddenBySlots(graph, center, first, second)) {
    return true;
  }
  if (delta.nonpair == nullptr || graph.pair_offsets == nullptr) {
    return false;
  }
  const std::int64_t begin = quick_hs::NeighborBegin(graph, center);
  const std::int64_t degree = quick_hs::NeighborEnd(graph, center) - begin;
  const std::int64_t a = first - begin;
  const std::int64_t b = second - begin;
  const std::int64_t low = a < b ? a : b;
  const std::int64_t high = a < b ? b : a;
  const std::int64_t pair =
      graph.pair_offsets[center] + low * (2 * degree - low - 1) / 2 + high - low - 1;
  return delta.nonpair[pair] != 0U;
}

// 只读检查候选终态：fixed_degree==1 时配对必须包含固定边，==2 时
// 唯一固定边对必须允许；不能只检查 degree>=2 而忽略 non-pair。
CUDAEE_TRANSACTION_HD inline std::int32_t ValidateVertex(const quick_hs::GraphView& graph,
                                                         const PendingDelta delta,
                                                         const std::uint8_t* const pending_fixed,
                                                         const std::int32_t center) {
  const std::int64_t begin = quick_hs::NeighborBegin(graph, center);
  const std::int64_t end = quick_hs::NeighborEnd(graph, center);
  std::int32_t degree = 0;
  std::int32_t fixed_degree = 0;
  std::int32_t invalid = kValid;
  for (std::int64_t slot = begin; slot < end; ++slot) {
    const std::int32_t edge = graph.neighbor_edge_ids[slot];
    if (!Survives(graph, delta, edge)) {
      if ((graph.fixed_edge != nullptr && graph.fixed_edge[edge] != 0U) ||
          (delta.fixed != nullptr && delta.fixed[edge] != 0U)) {
        invalid |= kDeleteFixedConflict;
      }
      continue;
    }
    ++degree;
    fixed_degree += pending_fixed[edge] != 0U ? 1 : 0;
  }
  if (degree < 2) {
    invalid |= kDegreeFloor;
  }
  if (fixed_degree > 2) {
    invalid |= kFixedDegree;
  }
  if (invalid != kValid || (graph.nonpair_mask == nullptr && delta.nonpair == nullptr)) {
    return invalid;
  }
  for (std::int64_t first = begin; first < end; ++first) {
    const std::int32_t first_edge = graph.neighbor_edge_ids[first];
    if (!Survives(graph, delta, first_edge)) {
      continue;
    }
    for (std::int64_t second = first + 1; second < end; ++second) {
      const std::int32_t second_edge = graph.neighbor_edge_ids[second];
      if (!Survives(graph, delta, second_edge)) {
        continue;
      }
      const std::int32_t included_fixed =
          (pending_fixed[first_edge] != 0U ? 1 : 0) + (pending_fixed[second_edge] != 0U ? 1 : 0);
      if (included_fixed == fixed_degree && !PairForbidden(graph, delta, center, first, second)) {
        return kValid;
      }
    }
  }
  return kNoAllowedPair;
}

#undef CUDAEE_TRANSACTION_HD

} // namespace cudaee::detail::resident_transaction
