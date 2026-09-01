#pragma once

#include "cuda_edge_elimination/local_search.hpp"

#include <compare>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
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

struct HtHamiltonReplyBatchResult {
  // 第 i 个 center 的确定性 reply 区间为 [offsets[i], offsets[i+1])。
  std::vector<std::uint64_t> offsets;
  std::vector<HtNeighborPair> replies;
  std::string backend;
  int selected_device{-1};
  bool cpu_verified{false};
};

struct HtEndReplyTask {
  std::int32_t endpoint{-1};
  std::int32_t internal_neighbor{-1};

  auto operator<=>(const HtEndReplyTask&) const = default;
};

struct HtEndReplyBatchResult {
  // 第 i 个 task 的确定性活动边区间为 [offsets[i], offsets[i+1])。
  std::vector<std::uint64_t> offsets;
  std::vector<NodeEdge> replies;
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

enum class HtMoveType : std::uint8_t {
  kLeaf,
  kCd,
  kPoint,
  kEnd,
};

constexpr std::uint32_t kNoHtChild = UINT32_MAX;

struct HtTreeReply {
  // kCd 使用两个 pair，kPoint 只使用 first_pair，kEnd 只使用 edge。
  HtNeighborPair first_pair;
  HtNeighborPair second_pair;
  NodeEdge edge{-1, -1};
  bool path_infeasible{false};
  std::uint32_t child_index{kNoHtChild};
};

struct HtTreeNode {
  NormalizedPathSystem paths;
  HtMoveType move_type{HtMoveType::kLeaf};
  std::int32_t move_first{-1};
  std::int32_t move_second{-1};
  PathSystemKOptProof leaf_proof;
  std::vector<HtTreeReply> replies;
};

struct HtRecursiveOptions {
  HtShallowOptions root_options{};
  // 根 c,d 之后允许的 point/end 递归层数；0 退化为浅层证明。
  std::uint32_t max_depth{2};
  std::uint64_t max_states{100000};
  std::uint64_t max_total_replies{1000000};
  std::uint64_t max_replies_per_move{10000};
  // 0 表示尝试全部候选。
  std::uint32_t max_point_candidates{3};
  std::uint32_t max_end_candidates{3};
  bool enable_point_moves{true};
  bool enable_end_moves{true};
};

struct HtRecursiveProof {
  bool proven{false};
  std::string reason;
  std::uint64_t snapshot_hash{};
  NodeEdge target_edge;
  HtCdMode cd_mode{HtCdMode::kActiveIncompatible};
  std::uint64_t cd_candidates_tested{};
  std::uint64_t states_expanded{};
  std::uint64_t replies_expanded{};
  std::uint64_t leaf_calls{};
  std::vector<HtTreeNode> nodes;
};

struct HtRecursiveResult {
  HtSearchStatus status{HtSearchStatus::kInvalid};
  HtRecursiveProof proof;
};

struct HtWavefrontOptions {
  HtRecursiveOptions search_options{};
  // 每个 reply 生成 chunk 的父状态上限；0 表示一次覆盖完整 frontier，不裁剪状态。
  std::uint32_t reply_frontier_batch_states{256};
  // 同一复杂度桶内每个 leaf batch 的状态上限；0 表示一次覆盖完整桶。
  std::uint32_t leaf_frontier_batch_states{256};
  // 只控制 continuation 真值传播；leaf/c,d 后端仍由 search_options 分别配置。
  PathCompatibilityBackend propagation_backend{PathCompatibilityBackend::kAuto};
  // 0 自动选择 cooperative residency 内的 block 数；1 强制单 block 正确性基线。
  std::uint32_t propagation_blocks{};
  // 只控制递归 point/end reply 的批量路径冲突标记；CPU 始终规范化并逐项认证。
  PathCompatibilityBackend path_append_backend{PathCompatibilityBackend::kAuto};
  // 控制 c,d/point Hamilton 邻边对与 end 活动边的 count/write；CPU 始终完整比较。
  PathCompatibilityBackend hamilton_reply_backend{PathCompatibilityBackend::kAuto};
};

struct HtWavefrontResult {
  HtSearchStatus status{HtSearchStatus::kInvalid};
  HtRecursiveProof proof;
  std::string propagation_backend{"none"};
  int selected_device{-1};
  bool cpu_verified{false};
  std::uint32_t propagation_blocks{};
  bool propagation_cooperative{false};
  std::string path_append_backend{"none"};
  int path_append_selected_device{-1};
  bool path_append_cpu_verified{false};
  bool path_append_device_children_verified{false};
  std::uint64_t path_append_batches{};
  std::uint64_t path_append_tasks{};
  std::uint64_t path_append_child_edges{};
  std::string leaf_cost_backend{"none"};
  int leaf_cost_selected_device{-1};
  bool leaf_cpu_verified{false};
  std::uint64_t leaf_frontier_batches{};
  std::uint64_t leaf_frontier_states{};
  std::uint64_t leaf_bucket_count{};
  std::uint64_t peak_leaf_frontier_batch{};
  std::uint64_t leaf_cost_batches{};
  std::uint64_t leaf_cost_tasks{};
  std::uint64_t leaf_cost_cells{};
  std::uint64_t leaf_scalar_searches{};
  std::uint64_t leaf_cuda_cost_batches{};
  std::uint64_t leaf_snapshot_cache_hits{};
  std::uint64_t leaf_template_cache_hits{};
  std::uint64_t leaf_workspace_cache_hits{};
  std::uint64_t peak_leaf_device_cache_bytes{};
  std::uint64_t leaf_cpu_long_tail_batches{};
  std::uint64_t leaf_cpu_long_tail_tasks{};
  std::uint64_t leaf_cpu_long_tail_cells{};
  std::string hamilton_reply_backend{"none"};
  int hamilton_reply_selected_device{-1};
  bool hamilton_reply_cpu_verified{false};
  std::uint64_t hamilton_reply_batches{};
  std::uint64_t hamilton_reply_centers{};
  std::uint64_t hamilton_replies_generated{};
  std::string end_reply_backend{"none"};
  int end_reply_selected_device{-1};
  bool end_reply_cpu_verified{false};
  std::uint64_t end_reply_batches{};
  std::uint64_t end_reply_tasks{};
  std::uint64_t end_replies_generated{};
  std::uint64_t reply_frontier_batches{};
  std::uint64_t reply_frontier_states{};
  std::uint64_t peak_reply_frontier_batch{};
  std::uint64_t moves_generated{};
  std::uint64_t peak_frontier{};
};

// GPU continuation 层使用的紧凑只读任务；CPU 在接受结果前复算全部状态。
struct HtWavefrontStateTask {
  std::uint32_t parent_move{kNoHtChild};
  std::uint32_t move_begin{};
  std::uint32_t move_count{};
  std::uint8_t leaf_proven{};
};

struct HtWavefrontMoveTask {
  std::uint32_t parent_state{};
  std::uint32_t reply_begin{};
  std::uint32_t reply_count{};
  std::uint32_t child_count{};
};

struct HtWavefrontReplyTask {
  std::uint32_t child_index{kNoHtChild};
  std::uint8_t path_infeasible{};
};

enum class HtPathAppendKind : std::uint8_t {
  kPoint,
  kEnd,
};

// point 表示加入 first-center-second；end 表示加入 first-second，且 center 必须为 -1。
struct HtPathAppendTask {
  std::uint32_t parent_index{};
  HtPathAppendKind kind{HtPathAppendKind::kPoint};
  std::int32_t first{-1};
  std::int32_t center{-1};
  std::int32_t second{-1};
};

struct HtPathAppendBatchResult {
  std::vector<std::uint8_t> feasible;
  // 与 task 一一对应；不可行项保留 NormalizePathSystem 给出的失败原因。
  std::vector<NormalizedPathSystem> children;
  // 第 i 个 task 的规范 child 边集为 [offsets[i], offsets[i+1])；不可行项为空。
  std::vector<std::uint64_t> child_edge_offsets;
  std::vector<NodeEdge> child_edges;
  std::string backend;
  int selected_device{-1};
  bool cpu_verified{false};
  // 仅当 CUDA 写出的完整 offsets/edges 与 CPU 规范化结果逐项相等时为 true。
  bool device_children_verified{false};
};

// 批量检查递归 HT point/end reply，并用 CPU 规范化结果认证 GPU flags 与 child edge SoA。
[[nodiscard]] HtPathAppendBatchResult
EvaluateHtPathAppends(std::int32_t dimension, const std::vector<NormalizedPathSystem>& parents,
                      const std::vector<HtPathAppendTask>& tasks,
                      PathCompatibilityBackend backend = PathCompatibilityBackend::kAuto);

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

// 批量枚举多个中心的 surviving 邻边对；CUDA count/write 返回前与 CPU 完整列表比较。
[[nodiscard]] HtHamiltonReplyBatchResult
EvaluateHtHamiltonReplies(const GraphSnapshot& graph, NodeEdge target_edge,
                          const std::vector<std::int32_t>& centers,
                          PathCompatibilityBackend backend = PathCompatibilityBackend::kAuto);

// 批量枚举 end move 的活动边 replies；每项排除指向路径内部的唯一邻边。
[[nodiscard]] HtEndReplyBatchResult
EvaluateHtEndReplies(const GraphSnapshot& graph, const std::vector<HtEndReplyTask>& tasks,
                     PathCompatibilityBackend backend = PathCompatibilityBackend::kAuto);

// 依次尝试 c,d OR move；一个 move 只有在全部 Hamilton replies 有叶证明时才成功。
[[nodiscard]] HtShallowResult ProveEdgeByShallowHt(const GraphSnapshot& graph, NodeEdge target_edge,
                                                   const HtShallowOptions& options = {});

// 不信任搜索器保存的计数或叶结论，重新枚举完整 reply 笛卡尔积并逐叶复核。
[[nodiscard]] bool VerifyHtShallowProof(const GraphSnapshot& graph, const HtShallowProof& proof,
                                        std::string* reason);

// CPU DFS 先实现与 wavefront 相同的 AND–OR 真值；成功 proof 使用扁平 continuation arena。
[[nodiscard]] HtRecursiveResult ProveEdgeByRecursiveHt(const GraphSnapshot& graph,
                                                       NodeEdge target_edge,
                                                       const HtRecursiveOptions& options = {});

// 从根路径开始重建每个 move 的完整 replies、子路径系统和嵌套 leaf proof。
[[nodiscard]] bool VerifyHtRecursiveProof(const GraphSnapshot& graph, const HtRecursiveProof& proof,
                                          std::string* reason);

// 主机按层生成完整 AND–OR 工作图，CPU/CUDA continuation 传播后提取可重放成功子树。
[[nodiscard]] HtWavefrontResult ProveEdgeByWavefrontHt(const GraphSnapshot& graph,
                                                       NodeEdge target_edge,
                                                       const HtWavefrontOptions& options = {});

// V1 文本证书嵌入每个 leaf 的 path k-opt V1；读取只做结构校验，授权仍须调用 verifier。
[[nodiscard]] std::string SerializeHtRecursiveProof(const HtRecursiveProof& proof);
[[nodiscard]] HtRecursiveProof ParseHtRecursiveProof(std::string_view serialized);
void WriteHtRecursiveProof(const std::filesystem::path& path, const HtRecursiveProof& proof);
[[nodiscard]] HtRecursiveProof ReadHtRecursiveProof(const std::filesystem::path& path);

namespace detail {

struct HtPathStateSpan {
  std::uint32_t node_begin{};
  std::uint32_t node_count{};
  std::uint32_t edge_begin{};
  std::uint32_t edge_count{};
};

struct HtPathNodeRecord {
  std::int32_t node{-1};
  std::uint32_t component{};
  std::uint8_t degree{};
};

struct HtPathAppendDeviceBatch {
  std::vector<std::uint8_t> feasible;
  std::vector<std::uint64_t> child_edge_offsets;
  std::vector<NodeEdge> child_edges;
};

struct HtHamiltonReplyDeviceBatch {
  std::vector<std::uint64_t> offsets;
  std::vector<HtNeighborPair> replies;
};

struct HtEndReplyDeviceBatch {
  std::vector<std::uint64_t> offsets;
  std::vector<NodeEdge> replies;
};

struct HtWavefrontDeviceResult {
  std::vector<std::uint8_t> status;
  std::uint32_t launched_blocks{};
  bool cooperative{false};
};

[[nodiscard]] bool HtCdCudaAvailable(std::string* reason);
[[nodiscard]] std::vector<std::uint8_t>
ScreenHtCdCandidatesCuda(const GraphSnapshot& graph, NodeEdge target_edge,
                         const std::vector<HtCdScreenTask>& tasks, HtCdMode mode,
                         int* selected_device);

[[nodiscard]] bool HtHamiltonReplyCudaAvailable(std::string* reason);
[[nodiscard]] HtHamiltonReplyDeviceBatch
EvaluateHtHamiltonRepliesCuda(const GraphSnapshot& graph, NodeEdge target_edge,
                              const std::vector<std::int32_t>& centers, int* selected_device);

[[nodiscard]] bool HtEndReplyCudaAvailable(std::string* reason);
[[nodiscard]] HtEndReplyDeviceBatch
EvaluateHtEndRepliesCuda(const GraphSnapshot& graph, const std::vector<HtEndReplyTask>& tasks,
                         int* selected_device);

[[nodiscard]] bool HtWavefrontCudaAvailable(std::string* reason);
[[nodiscard]] HtWavefrontDeviceResult
EvaluateHtWavefrontCuda(const std::vector<HtWavefrontStateTask>& states,
                        const std::vector<HtWavefrontMoveTask>& moves,
                        const std::vector<HtWavefrontReplyTask>& replies,
                        const std::vector<std::uint32_t>& level_offsets,
                        std::uint32_t requested_blocks, int* selected_device);

[[nodiscard]] bool HtPathAppendCudaAvailable(std::string* reason);
[[nodiscard]] HtPathAppendDeviceBatch
EvaluateHtPathAppendsCuda(std::int32_t dimension, const std::vector<HtPathStateSpan>& states,
                          const std::vector<HtPathNodeRecord>& nodes,
                          const std::vector<NodeEdge>& parent_edges,
                          const std::vector<HtPathAppendTask>& tasks, int* selected_device);

} // namespace detail

} // namespace cudaee
