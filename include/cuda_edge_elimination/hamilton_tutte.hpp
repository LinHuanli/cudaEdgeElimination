#pragma once

#include "cuda_edge_elimination/local_search.hpp"

#include <compare>
#include <cstdint>
#include <string>
#include <vector>

namespace cudaee {

enum class HtSearchStatus : std::uint8_t {
  kProven,
  kUnresolved,
  kInvalid,
};

// Tutte 选择的 c,d 可以是一条已被 2-opt 排除的活动边，也可以本来就不在快照中。
enum class HtCdMode : std::uint8_t {
  kActiveIncompatible,
  kMissingOrIncompatible,
};

struct HtNeighborPair {
  std::int32_t center{-1};
  std::int32_t first{-1};
  std::int32_t second{-1};

  auto operator<=>(const HtNeighborPair&) const = default;
};

struct HtCdCandidate {
  std::int32_t c{-1};
  std::int32_t d{-1};
  std::uint64_t reply_product{};

  auto operator<=>(const HtCdCandidate&) const = default;
};

struct HtCdScreenTask {
  std::int32_t c{-1};
  std::int32_t d{-1};
  std::uint8_t active{0};
};

struct HtCdBatchResult {
  std::vector<HtCdCandidate> candidates;
  std::string backend;
  int selected_device{-1};
  bool cpu_verified{false};
};

struct HtShallowOptions {
  // 0 表示使用除目标端点外的全部节点；只影响候选强度，不影响证明正确性。
  std::uint32_t max_neighborhood{25};
  // 0 表示尝试全部候选 Tutte moves。
  std::uint32_t max_cd_candidates{5};
  // 0 表示不按活动度数筛选候选 c,d。
  std::uint32_t max_candidate_degree{50};
  // 单个 c,d move 的 Hamilton reply 笛卡尔积上限；0 表示不设预算。
  std::uint64_t max_reply_combinations{1000000};
  HtCdMode cd_mode{HtCdMode::kActiveIncompatible};
  PathCompatibilityBackend candidate_backend{PathCompatibilityBackend::kCpu};
  KOptSearchOptions leaf_options{};
};

struct HtReplyProof {
  HtNeighborPair c_reply;
  HtNeighborPair d_reply;
  bool path_infeasible{false};
  PathSystemKOptProof leaf_proof;
};

// 当前证书只表达一个浅层 OR 候选及其全部 AND replies；递归 wavefront 将复用该节点语义。
struct HtShallowProof {
  bool proven{false};
  std::string reason;
  std::uint64_t snapshot_hash{};
  NodeEdge target_edge;
  HtCdMode cd_mode{HtCdMode::kActiveIncompatible};
  std::int32_t c{-1};
  std::int32_t d{-1};
  std::uint64_t cd_candidates_tested{};
  std::uint64_t reply_combinations_tested{};
  std::vector<HtReplyProof> replies;
};

struct HtShallowResult {
  HtSearchStatus status{HtSearchStatus::kInvalid};
  HtShallowProof proof;
};

// 生成确定性排序的 OR 候选；reply_product 越小越优先。
[[nodiscard]] std::vector<HtCdCandidate>
GenerateHtCdCandidates(const GraphSnapshot& graph, NodeEdge target_edge,
                       const HtShallowOptions& options = {});

// CUDA 只筛选 OR 候选；返回前逐项与 CPU flags 比较并由 CPU 计算 reply_product。
[[nodiscard]] HtCdBatchResult EvaluateHtCdCandidates(const GraphSnapshot& graph,
                                                     NodeEdge target_edge,
                                                     const HtShallowOptions& options = {});

// 枚举在目标边存在时，不能被严格 2/3-opt 立即排除的中心点邻边对。
[[nodiscard]] std::vector<HtNeighborPair>
EnumerateHtHamiltonReplies(const GraphSnapshot& graph, NodeEdge target_edge, std::int32_t center);

// 依次尝试 c,d OR move；一个 move 只有在全部 Hamilton replies 有叶证明时才成功。
[[nodiscard]] HtShallowResult ProveEdgeByShallowHt(const GraphSnapshot& graph, NodeEdge target_edge,
                                                   const HtShallowOptions& options = {});

// 不信任搜索器保存的计数或叶结论，重新枚举完整 reply 笛卡尔积并逐叶复核。
[[nodiscard]] bool VerifyHtShallowProof(const GraphSnapshot& graph, const HtShallowProof& proof,
                                        std::string* reason);

namespace detail {

[[nodiscard]] bool HtCdCudaAvailable(std::string* reason);
[[nodiscard]] std::vector<std::uint8_t>
ScreenHtCdCandidatesCuda(const GraphSnapshot& graph, NodeEdge target_edge,
                         const std::vector<HtCdScreenTask>& tasks, HtCdMode mode,
                         int* selected_device);

} // namespace detail

} // namespace cudaee
