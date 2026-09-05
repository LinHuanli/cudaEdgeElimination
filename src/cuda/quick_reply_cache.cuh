#pragma once

#include "../fgpu/quick_hs_predicate.hpp"
#include "device_workspace.cuh"

namespace cudaee::detail {

struct QuickReplyRow {
  std::int32_t root{-1};
  std::int32_t center{-1};
  std::int64_t score{};
  std::uint64_t snapshot{};
};

struct QuickReplySpan {
  const std::uint64_t* ordinals{};
  std::uint64_t size{};
  bool valid{false};
};

struct QuickReplyView {
  const QuickReplyRow* rows{};
  const std::uint64_t* offsets{};
  const std::uint64_t* ordinals{};
  const std::uint8_t* verified{};
  std::int32_t roots{};
  std::int32_t stride{};
  std::uint64_t total{};

  __device__ QuickReplySpan Find(const std::int32_t work, const std::int32_t root,
                                 const std::int32_t center, const std::uint64_t snapshot) const {
    if (work < 0 || work >= roots)
      return {};
    for (std::int32_t local = 0; local < stride; ++local) {
      const auto index = static_cast<std::int64_t>(work) * stride + local;
      const auto row = rows[index];
      if (row.center != center)
        continue;
      const auto begin = offsets[index], end = offsets[index + 1];
      if (row.root != root || row.snapshot != snapshot || verified[index] == 0U || begin > end ||
          end > total || (end != begin && ordinals == nullptr))
        return {};
      // 空流允许 data == nullptr；它只在独立完整覆盖检查通过后才表示空 AND。
      return {ordinals == nullptr ? nullptr : ordinals + begin, end - begin, true};
    }
    return {};
  }
};

struct QuickReplyCacheMetrics {
  std::uint64_t raw_pairs{};
  std::uint64_t compact_pairs{};
  std::uint64_t bytes{};
  double build_ms{};
  double validation_ms{};
};

// 缓存只属于当前不可变 epoch，不是可跨 root 发布的全局 nonpair。
// CSR 行按实际 surviving pair 数分配；stride 只表示原方法候选中心数，绝非回复上限。
class QuickReplyCache {
public:
  explicit QuickReplyCache(int device);
  QuickReplyCacheMetrics Build(quick_hs::GraphView graph, std::int32_t work_count,
                               const std::int32_t* work_edges, const std::uint8_t* protected_edges,
                               std::int32_t candidates, bool two_hop, std::uint64_t snapshot);
  // 独立入口也供故障注入测试使用；失败清除授权，不把错误当作空集合。
  bool Validate(quick_hs::GraphView graph, const std::int32_t* work_edges, std::uint64_t snapshot);
  [[nodiscard]] QuickReplyView view() const;

private:
  CudaWorkspace<QuickReplyRow> rows_;
  CudaWorkspace<std::uint64_t> counts_, offsets_, ordinals_;
  CudaWorkspace<std::uint8_t> verified_, scan_temp_;
  CudaWorkspace<unsigned long long> counters_;
  std::int32_t roots_{}, stride_{}, row_count_{};
  std::uint64_t total_{};
};

} // namespace cudaee::detail
