#pragma once

#include "cuda_edge_elimination/local_search.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudaee::detail {

struct TransposedLeafBatchResponse {
  PathSystemKOptBatchResult batch;
  std::uint64_t physical_request_count{};
  std::uint64_t physical_state_count{};
  bool owns_physical_metrics{false};
};

// 把多个 target worker 在同一短路位置提交的 leaf window 合成一个异构 k-opt batch。
class TransposedLeafBroker {
public:
  TransposedLeafBroker(const GraphSnapshot& graph, const KOptSnapshotBinding& snapshot_binding,
                       KOptSearchOptions options, std::size_t worker_count, int device_ordinal);
  ~TransposedLeafBroker();

  TransposedLeafBroker(const TransposedLeafBroker&) = delete;
  TransposedLeafBroker& operator=(const TransposedLeafBroker&) = delete;

  [[nodiscard]] TransposedLeafBatchResponse Evaluate(std::vector<NormalizedPathSystem> states,
                                                     NodeEdge required_edge);

  // auto speculation 随仍活跃 target 数降低而增大，补偿尾部跨目标并行度下降。
  [[nodiscard]] std::uint32_t SuggestedSpeculationWidth();

  // 每个 target worker 必须恰好结束一次；count 版本用于线程创建中途失败的收尾。
  void FinishWorkers(std::size_t count = 1U) noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaee::detail
