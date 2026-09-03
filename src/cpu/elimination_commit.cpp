#include "elimination_commit.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee::detail {

std::vector<Candidate> CommitVerifiedCandidates(GraphSnapshot* const graph,
                                                std::vector<Candidate> candidates,
                                                const std::uint64_t expected_snapshot_hash) {
  if (graph == nullptr || graph->ContentHash() != expected_snapshot_hash) {
    throw std::runtime_error("epoch 提交前不可变快照哈希已变化");
  }
  for (const Candidate& candidate : candidates) {
    if (candidate.edge_id < 0 ||
        static_cast<std::size_t>(candidate.edge_id) >= graph->edges.size()) {
      throw std::runtime_error("epoch 候选边编号越界");
    }
    if (candidate.method != EliminationMethod::kJv &&
        candidate.method != EliminationMethod::kHamiltonTutte &&
        candidate.method != EliminationMethod::kGeometryMain &&
        candidate.method != EliminationMethod::kLpBox) {
      throw std::runtime_error("epoch 候选方法不受支持");
    }
    if (!graph->edges[static_cast<std::size_t>(candidate.edge_id)].active) {
      throw std::runtime_error("epoch 候选不是快照中的活动边");
    }
  }

  std::sort(candidates.begin(), candidates.end(), [&](const Candidate& lhs, const Candidate& rhs) {
    const Edge& lhs_edge = graph->edges[static_cast<std::size_t>(lhs.edge_id)];
    const Edge& rhs_edge = graph->edges[static_cast<std::size_t>(rhs.edge_id)];
    return std::tuple{lhs_edge.u,  lhs_edge.v,         static_cast<std::uint8_t>(lhs.method),
                      lhs.witness, lhs.second_witness, lhs.edge_id} <
           std::tuple{rhs_edge.u,  rhs_edge.v,         static_cast<std::uint8_t>(rhs.method),
                      rhs.witness, rhs.second_witness, rhs.edge_id};
  });
  candidates.erase(std::unique(candidates.begin(), candidates.end(),
                               [](const Candidate& lhs, const Candidate& rhs) {
                                 return lhs.edge_id == rhs.edge_id;
                               }),
                   candidates.end());

  std::vector<std::int32_t> degrees(static_cast<std::size_t>(graph->dimension));
  for (std::int32_t vertex = 0; vertex < graph->dimension; ++vertex) {
    degrees[static_cast<std::size_t>(vertex)] = graph->Degree(vertex);
  }

  std::vector<Candidate> committed;
  committed.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    const Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
    if (degrees[static_cast<std::size_t>(edge.u)] <= 2 ||
        degrees[static_cast<std::size_t>(edge.v)] <= 2) {
      continue;
    }
    --degrees[static_cast<std::size_t>(edge.u)];
    --degrees[static_cast<std::size_t>(edge.v)];
    committed.push_back(candidate);
  }

  if (!committed.empty()) {
    // 在副本上完成全部分配与 CSR 重建；异常时原图仍保持完整 epoch 快照。
    GraphSnapshot updated = *graph;
    for (const Candidate& candidate : committed) {
      updated.edges[static_cast<std::size_t>(candidate.edge_id)].active = false;
    }
    updated.RebuildCsr();
    *graph = std::move(updated);
  }
  return committed;
}

} // namespace cudaee::detail
