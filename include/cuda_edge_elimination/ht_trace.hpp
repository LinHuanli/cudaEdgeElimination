#pragma once

#include "cuda_edge_elimination/local_search.hpp"

#include <compare>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cudaee {

constexpr std::uint32_t kNoHtTraceNode = UINT32_MAX;

enum class HtTraceNodeKind : std::uint8_t {
  kLeaf,
  kOr,
  kAnd,
};

// Trace 只用于研究调度工作量；它不是 proof，不能授权删除任何边。
struct HtShortCircuitTraceNode {
  HtTraceNodeKind kind{HtTraceNodeKind::kLeaf};
  std::uint32_t parent{kNoHtTraceNode};
  std::uint32_t child_ordinal{};
  std::uint32_t depth{};
  bool value{false};
  std::uint64_t work_units{1U};
  std::vector<std::uint32_t> children;

  auto operator<=>(const HtShortCircuitTraceNode&) const = default;
};

struct HtShortCircuitTrace {
  std::uint64_t snapshot_hash{};
  NodeEdge target_edge{-1, -1};
  std::uint32_t root{kNoHtTraceNode};
  // 资源预算或后端错误产生的部分树不能用于推导完整短路工作量。
  bool complete{false};
  std::vector<HtShortCircuitTraceNode> nodes;

  auto operator<=>(const HtShortCircuitTrace&) const = default;
};

struct HtShortCircuitTraceBundle {
  std::vector<HtShortCircuitTrace> traces;

  auto operator<=>(const HtShortCircuitTraceBundle&) const = default;
};

struct HtTraceReplayResult {
  // 0 表示同一 connective 的全部 child 同时推测执行。
  std::uint32_t speculation_width{1U};
  bool value{false};
  std::uint64_t scheduled_nodes{};
  std::uint64_t scheduled_work_units{};
  std::uint64_t canonical_nodes{};
  std::uint64_t speculative_nodes{};
  std::uint64_t short_circuits{};
  std::uint64_t peak_ready_width{};
};

[[nodiscard]] HtTraceReplayResult ReplayHtShortCircuitTrace(const HtShortCircuitTrace& trace,
                                                            std::uint32_t speculation_width);

void WriteHtShortCircuitTraceBundle(const std::filesystem::path& path,
                                    const HtShortCircuitTraceBundle& bundle);
[[nodiscard]] HtShortCircuitTraceBundle
ReadHtShortCircuitTraceBundle(const std::filesystem::path& path);

[[nodiscard]] std::string ToString(HtTraceNodeKind kind);

} // namespace cudaee
