#include "../../src/fgpu/resident_transaction.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace tx = cudaee::detail::resident_transaction;

namespace {
void Check(const bool valid, const char* const message) {
  if (!valid) {
    throw std::runtime_error(message);
  }
}
} // namespace

int main() {
  tx::CommitGate gate{7U};
  Check(!gate.Publish(7U), "proposal 不可提前发布");
  Check(!gate.Validate(0), "replay 前不可验证提交");
  Check(!gate.FinishReplay(6U), "拒绝其他快照的 replay");
  Check(gate.FinishReplay(7U), "同快照 replay 完成");
  Check(!gate.FinishReplay(7U), "拒绝重复 replay 完成");
  Check(!gate.Validate(tx::kFixedDegree), "无效候选终态不可发布");
  Check(gate.Validate(tx::kValid), "完整候选状态通过");
  Check(!gate.Publish(8U), "禁止跨版本发布");
  Check(gate.Publish(7U), "已验证事务可发布");
  Check(!gate.Publish(7U), "同一事务不能提交两次");

  // K4 的 CSR；顶点 0 的三个邻边为 stable IDs 0、1、2。
  std::array<std::int64_t, 5> rows{0, 3, 6, 9, 12};
  std::array<std::int32_t, 12> neighbors{1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2};
  std::array<std::int32_t, 12> ids{0, 1, 2, 0, 3, 4, 1, 3, 5, 2, 4, 5};
  std::array<std::int32_t, 6> u{0, 0, 0, 1, 1, 2};
  std::array<std::int32_t, 6> v{1, 2, 3, 2, 3, 3};
  std::array<std::int32_t, 4> degree{3, 3, 3, 3};
  std::array<std::uint8_t, 6> active{1, 1, 1, 1, 1, 1};
  std::array<std::uint8_t, 6> fixed{};
  std::array<std::uint8_t, 6> pending_fixed{};
  std::array<std::uint8_t, 6> deleted{};
  std::array<std::uint8_t, 6> proposed_fixed{};
  std::array<std::int64_t, 5> pairs{0, 3, 6, 9, 12};
  std::array<std::uint8_t, 12> nonpair{};
  std::array<std::uint8_t, 12> proposed_nonpair{};
  cudaee::detail::quick_hs::GraphView graph{};
  graph.dimension = 4;
  graph.degree = degree.data();
  graph.row_offsets = rows.data();
  graph.neighbors = neighbors.data();
  graph.neighbor_edge_ids = ids.data();
  graph.edge_u = u.data();
  graph.edge_v = v.data();
  graph.edge_active = active.data();
  graph.fixed_edge = fixed.data();
  graph.pair_offsets = pairs.data();
  graph.nonpair_mask = nonpair.data();
  const tx::PendingDelta delta{deleted.data(), proposed_fixed.data(), proposed_nonpair.data()};
  Check(tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) == tx::kValid, "初始 K4 合法");

  // 同一 epoch 的 fix 与 non-pair 单独都可能看似合法，组合后必须检查。
  pending_fixed[0] = pending_fixed[1] = 1U;
  proposed_fixed[0] = proposed_fixed[1] = 1U;
  proposed_nonpair[0] = 1U;
  Check(tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) == tx::kNoAllowedPair,
        "检测同轮 fixed pair 与 non-pair 冲突");
  Check(fixed[0] == 0U && nonpair[0] == 0U, "验证不修改 immutable snapshot");
  proposed_nonpair[0] = 0U;
  Check(tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) == tx::kValid,
        "兼容固定边可提交");
  pending_fixed[2] = 1U;
  Check((tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) & tx::kFixedDegree) != 0,
        "三条固定边拒绝发布");
  pending_fixed[2] = 0U;
  deleted[0] = 1U;
  Check((tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) & tx::kDeleteFixedConflict) != 0,
        "候选 delete/fix 冲突被拒绝");
  deleted[1] = 1U;
  Check((tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) & tx::kDegreeFloor) != 0,
        "最小度破坏被拒绝");
  deleted.fill(0U);
  proposed_fixed.fill(0U);
  pending_fixed.fill(0U);
  proposed_nonpair[0] = proposed_nonpair[1] = proposed_nonpair[2] = 1U;
  Check(tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) == tx::kNoAllowedPair,
        "有三条边但无合法 pair 也必须拒绝");
  proposed_nonpair[2] = 0U;
  pending_fixed[0] = 1U;
  Check(tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) == tx::kNoAllowedPair,
        "允许的 pair 必须包含已有固定边");
  pending_fixed[0] = 0U;
  Check(tx::ValidateVertex(graph, delta, pending_fixed.data(), 0) == tx::kValid,
        "剩余一个合法 pair 可保留");
  std::cout << "resident transaction tests passed\n";
}
