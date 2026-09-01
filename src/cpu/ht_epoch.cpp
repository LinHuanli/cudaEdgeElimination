#include "cuda_edge_elimination/elimination.hpp"

#include "elimination_commit.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::size_t kMaxHtEpochCandidates = 1000000U;

struct VerifiedHtCandidate {
  std::int32_t edge_id{-1};
  std::string serialized;
  const HtRecursiveProof* proof{nullptr};
};

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

EliminationResult CommitHtProofEpoch(GraphSnapshot* const graph,
                                     const std::vector<HtRecursiveProof>& proofs) {
  if (graph == nullptr) {
    throw std::invalid_argument("HT epoch 的图不能为空");
  }
  if (proofs.size() > kMaxHtEpochCandidates) {
    throw std::invalid_argument("HT epoch 候选数量超过安全上限");
  }

  EliminationResult result;
  result.backend = "ht-sidecar-cpu";
  const auto snapshot_start = std::chrono::steady_clock::now();
  result.initial_hash = graph->ContentHash();
  const std::uint64_t snapshot_hash = result.initial_hash;
  EpochMetrics metrics;
  metrics.epoch = 0U;
  metrics.edges_before = graph->ActiveEdgeCount();
  metrics.proposed = proofs.size();
  metrics.snapshot_ms = ElapsedMilliseconds(snapshot_start);

  // 端点到 edge_id 的映射也属于快照；重复规范边说明 GraphSnapshot 本身不合法。
  std::map<std::pair<std::int32_t, std::int32_t>, std::int32_t> edge_ids;
  for (std::size_t edge_id = 0; edge_id < graph->edges.size(); ++edge_id) {
    const Edge& edge = graph->edges[edge_id];
    if (!edge_ids.emplace(std::pair{edge.u, edge.v}, static_cast<std::int32_t>(edge_id)).second) {
      throw std::runtime_error("HT epoch 图包含重复规范边");
    }
  }

  const auto verify_start = std::chrono::steady_clock::now();
  std::vector<VerifiedHtCandidate> verified;
  verified.reserve(proofs.size());
  for (std::size_t index = 0; index < proofs.size(); ++index) {
    const HtRecursiveProof& proof = proofs[index];
    std::string reason;
    if (!VerifyHtRecursiveProof(*graph, proof, &reason)) {
      // 整批验证发生在任何 active 位修改之前；一个坏 sidecar 会让本 epoch 零提交。
      throw std::runtime_error("HT epoch sidecar[" + std::to_string(index) +
                               "] 复核失败: " + reason);
    }
    const auto edge = edge_ids.find({proof.target_edge.u, proof.target_edge.v});
    if (edge == edge_ids.end()) {
      throw std::runtime_error("HT epoch sidecar 目标边不在不可变快照中");
    }
    verified.push_back(
        {.edge_id = edge->second, .serialized = SerializeHtRecursiveProof(proof), .proof = &proof});
  }
  metrics.verify_ms = ElapsedMilliseconds(verify_start);
  metrics.verified = verified.size();

  // 输入顺序不得影响提交结果；同一目标的多份合法证明保留字节序最小的一份。
  std::sort(verified.begin(), verified.end(),
            [&](const VerifiedHtCandidate& lhs, const VerifiedHtCandidate& rhs) {
              const Edge& lhs_edge = graph->edges[static_cast<std::size_t>(lhs.edge_id)];
              const Edge& rhs_edge = graph->edges[static_cast<std::size_t>(rhs.edge_id)];
              return std::tie(lhs_edge.u, lhs_edge.v, lhs.serialized) <
                     std::tie(rhs_edge.u, rhs_edge.v, rhs.serialized);
            });
  verified.erase(std::unique(verified.begin(), verified.end(),
                             [](const VerifiedHtCandidate& lhs, const VerifiedHtCandidate& rhs) {
                               return lhs.edge_id == rhs.edge_id;
                             }),
                 verified.end());

  std::vector<Candidate> candidates;
  candidates.reserve(verified.size());
  std::map<std::int32_t, const HtRecursiveProof*> selected_proofs;
  for (const VerifiedHtCandidate& candidate : verified) {
    candidates.push_back({candidate.edge_id, -1, EliminationMethod::kHamiltonTutte});
    selected_proofs.emplace(candidate.edge_id, candidate.proof);
  }

  // 先在工作副本上完成 degree gate、CSR 重建和结果证书复制；最后一次 move 才发布 epoch。
  const auto commit_start = std::chrono::steady_clock::now();
  GraphSnapshot updated = *graph;
  const std::vector<Candidate> committed =
      detail::CommitVerifiedCandidates(&updated, std::move(candidates), snapshot_hash);
  metrics.committed = committed.size();
  result.proof.reserve(committed.size());
  result.ht_proofs.reserve(committed.size());
  for (const Candidate& candidate : committed) {
    const auto selected = selected_proofs.find(candidate.edge_id);
    if (selected == selected_proofs.end() || selected->second == nullptr ||
        result.ht_proofs.size() >= std::numeric_limits<std::uint32_t>::max()) {
      throw std::logic_error("HT epoch 已提交候选缺少规范 sidecar");
    }
    const std::uint32_t certificate_index = static_cast<std::uint32_t>(result.ht_proofs.size());
    result.ht_proofs.push_back(*selected->second);
    const Edge& edge = updated.edges[static_cast<std::size_t>(candidate.edge_id)];
    result.proof.push_back({0U, snapshot_hash, candidate.edge_id, edge.u, edge.v, -1,
                            EliminationMethod::kHamiltonTutte, certificate_index});
  }
  result.final_hash = updated.ContentHash();
  *graph = std::move(updated);
  metrics.commit_ms = ElapsedMilliseconds(commit_start);
  result.epochs.push_back(metrics);
  return result;
}

} // namespace cudaee
