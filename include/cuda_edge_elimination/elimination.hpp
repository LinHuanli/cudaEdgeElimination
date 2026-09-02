#pragma once

#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/hamilton_tutte.hpp"
#include "cuda_edge_elimination/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cudaee {

enum class Backend {
  kAuto,
  kCpu,
  kCuda,
};

struct EpochMetrics {
  std::uint32_t epoch{};
  std::size_t edges_before{};
  std::size_t proposed{};
  std::size_t verified{};
  std::size_t rejected{};
  std::size_t committed{};
  // 细分计时只用于当前运行观测；V1/V2 proof 继续只序列化 propose/verify。
  double snapshot_ms{};
  double propose_ms{};
  double verify_ms{};
  double commit_ms{};
  bool jv_static_cache_hit{false};
  bool jv_workspace_cache_hit{false};
  std::uint64_t jv_resident_bytes{};
  double jv_h2d_ms{};
  double jv_kernel_ms{};
  double jv_d2h_ms{};
};

struct JvCudaCacheUsage {
  bool static_hit{false};
  bool workspace_hit{false};
  std::uint64_t resident_bytes{};
  // 三段同步 wall time 仅用于性能诊断，不进入 proof。
  double h2d_ms{};
  double kernel_ms{};
  double d2h_ms{};
};

struct EliminationResult {
  std::string backend;
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  std::vector<ProofRecord> proof;
  // 只有 HT record 可以引用这里的 V1 continuation arena；JV V1 输出保持不变。
  std::vector<HtRecursiveProof> ht_proofs;
  std::vector<EpochMetrics> epochs;
};

enum class HtTargetOrder : std::uint8_t {
  kCanonical,
  kWeightDescending,
};

struct HtScanOptions {
  HtWavefrontOptions wavefront_options{};
  // offset 在筛掉非活动边和度数门禁后、完成确定性排序的目标序列上解释。
  std::uint64_t target_offset{};
  // 全图搜索必须显式有界；0 非法。
  std::uint64_t max_targets{64U};
  HtTargetOrder target_order{HtTargetOrder::kWeightDescending};
};

struct HtScanAttempt {
  std::int32_t edge_id{-1};
  NodeEdge target_edge;
  HtSearchStatus status{HtSearchStatus::kInvalid};
  std::uint64_t states_expanded{};
  std::uint64_t replies_expanded{};
  std::uint64_t leaf_calls{};
  std::uint64_t moves_generated{};
  std::uint64_t peak_frontier{};
  std::string propagation_backend{"none"};
  int selected_device{-1};
  std::uint32_t propagation_blocks{};
  bool propagation_cooperative{false};
  bool propagation_cpu_verified{false};
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
  std::uint64_t leaf_cursor_searches_started{};
  std::uint64_t leaf_cuda_cost_batches{};
  std::uint64_t leaf_cpu_long_tail_cells{};
  std::uint64_t leaf_cost_rows_consumed{};
  std::uint64_t leaf_candidate_templates_rechecked{};
  std::uint64_t leaf_cpu_completeness_rows{};
  std::uint64_t leaf_cpu_completeness_templates{};
  std::uint64_t leaf_cpu_certified_cost_cells{};
  std::uint64_t leaf_cpu_cost_rows_scored{};
  std::uint64_t leaf_cpu_cost_rows_reused{};
  std::uint64_t leaf_cpu_parallel_cost_batches{};
  std::uint64_t leaf_cpu_parallel_cost_cells{};
  std::uint32_t peak_leaf_cpu_cost_threads{1U};
  std::uint64_t peak_leaf_device_cache_bytes{};
  std::uint64_t path_append_tasks{};
  std::uint64_t root_child_normalizations{};
  std::uint64_t point_candidate_scans{};
  std::uint64_t point_candidate_nodes_checked{};
  std::uint64_t point_candidate_nodes_ranked{};
  std::uint64_t point_candidate_nodes_selected{};
  std::uint64_t hamilton_reply_batches{};
  std::uint64_t hamilton_reply_centers{};
  std::uint64_t hamilton_reply_unique_centers{};
  std::uint64_t hamilton_reply_neighbor_pairs_tested{};
  std::uint64_t hamilton_replies_generated{};
  std::uint64_t reply_cuda_batches{};
  std::uint64_t reply_cuda_graph_cache_hits{};
  std::uint64_t reply_cuda_workspace_cache_hits{};
  std::uint64_t peak_reply_device_cache_bytes{};
  std::uint64_t end_replies_generated{};
  double candidate_ms{};
  double work_graph_ms{};
  double root_child_normalize_ms{};
  double point_candidate_scan_ms{};
  double point_candidate_sort_ms{};
  double leaf_ms{};
  double leaf_setup_ms{};
  double leaf_proof_initialize_ms{};
  double leaf_coverage_scan_ms{};
  double leaf_cursor_construct_ms{};
  double leaf_cursor_prepare_ms{};
  double leaf_cost_evaluate_ms{};
  double leaf_cost_cpu_certify_ms{};
  double leaf_cost_scatter_ms{};
  double leaf_cursor_consume_ms{};
  double leaf_candidate_recheck_ms{};
  double leaf_completeness_fallback_ms{};
  double leaf_scalar_search_ms{};
  double leaf_apply_ms{};
  double leaf_proof_verify_ms{};
  double path_append_ms{};
  double path_append_parent_prepare_ms{};
  double path_append_child_normalize_ms{};
  double path_append_child_edges_ms{};
  double path_append_cuda_evaluate_ms{};
  double path_append_cuda_compare_ms{};
  double hamilton_reply_ms{};
  double hamilton_reply_validation_ms{};
  double hamilton_reply_cpu_enumerate_ms{};
  double hamilton_reply_cuda_evaluate_ms{};
  double hamilton_reply_cuda_compare_ms{};
  double end_reply_ms{};
  double propagation_ms{};
  double proof_extract_ms{};
  double proof_verify_ms{};
  // PROVEN sidecar 离开 wavefront 后的第二次即时 CPU 重放。
  double immediate_verify_ms{};
  double search_ms{};
  std::string reason;
};

struct HtScanResult {
  EliminationResult elimination;
  std::uint64_t eligible_targets{};
  std::uint64_t target_offset{};
  std::uint64_t proven_targets{};
  std::uint64_t unresolved_targets{};
  std::uint64_t states_expanded{};
  std::uint64_t replies_expanded{};
  std::uint64_t leaf_calls{};
  std::uint64_t moves_generated{};
  std::uint64_t leaf_frontier_batches{};
  std::uint64_t leaf_frontier_states{};
  std::uint64_t leaf_bucket_count{};
  std::uint64_t peak_leaf_frontier_batch{};
  std::uint64_t leaf_cost_batches{};
  std::uint64_t leaf_cost_tasks{};
  std::uint64_t leaf_cost_cells{};
  std::uint64_t leaf_cursor_searches_started{};
  std::uint64_t leaf_cuda_cost_batches{};
  std::uint64_t leaf_cpu_long_tail_cells{};
  std::uint64_t leaf_cost_rows_consumed{};
  std::uint64_t leaf_candidate_templates_rechecked{};
  std::uint64_t leaf_cpu_completeness_rows{};
  std::uint64_t leaf_cpu_completeness_templates{};
  std::uint64_t leaf_cpu_certified_cost_cells{};
  std::uint64_t leaf_cpu_cost_rows_scored{};
  std::uint64_t leaf_cpu_cost_rows_reused{};
  std::uint64_t leaf_cpu_parallel_cost_batches{};
  std::uint64_t leaf_cpu_parallel_cost_cells{};
  std::uint32_t peak_leaf_cpu_cost_threads{1U};
  std::uint64_t peak_leaf_device_cache_bytes{};
  std::uint64_t root_child_normalizations{};
  std::uint64_t point_candidate_scans{};
  std::uint64_t point_candidate_nodes_checked{};
  std::uint64_t point_candidate_nodes_ranked{};
  std::uint64_t point_candidate_nodes_selected{};
  std::uint64_t hamilton_reply_batches{};
  std::uint64_t hamilton_reply_centers{};
  std::uint64_t hamilton_reply_unique_centers{};
  std::uint64_t hamilton_reply_neighbor_pairs_tested{};
  std::uint64_t hamilton_replies_generated{};
  std::uint64_t reply_cuda_batches{};
  std::uint64_t reply_cuda_graph_cache_hits{};
  std::uint64_t reply_cuda_workspace_cache_hits{};
  std::uint64_t peak_reply_device_cache_bytes{};
  double target_selection_ms{};
  double candidate_ms{};
  double work_graph_ms{};
  double root_child_normalize_ms{};
  double point_candidate_scan_ms{};
  double point_candidate_sort_ms{};
  double leaf_ms{};
  double leaf_setup_ms{};
  double leaf_proof_initialize_ms{};
  double leaf_coverage_scan_ms{};
  double leaf_cursor_construct_ms{};
  double leaf_cursor_prepare_ms{};
  double leaf_cost_evaluate_ms{};
  double leaf_cost_cpu_certify_ms{};
  double leaf_cost_scatter_ms{};
  double leaf_cursor_consume_ms{};
  double leaf_candidate_recheck_ms{};
  double leaf_completeness_fallback_ms{};
  double leaf_scalar_search_ms{};
  double leaf_apply_ms{};
  double leaf_proof_verify_ms{};
  double path_append_ms{};
  double path_append_parent_prepare_ms{};
  double path_append_child_normalize_ms{};
  double path_append_child_edges_ms{};
  double path_append_cuda_evaluate_ms{};
  double path_append_cuda_compare_ms{};
  double hamilton_reply_ms{};
  double hamilton_reply_validation_ms{};
  double hamilton_reply_cpu_enumerate_ms{};
  double hamilton_reply_cuda_evaluate_ms{};
  double hamilton_reply_cuda_compare_ms{};
  double end_reply_ms{};
  double propagation_ms{};
  double proof_extract_ms{};
  double proof_verify_ms{};
  double immediate_verify_ms{};
  double commit_ms{};
  double search_ms{};
  double total_ms{};
  std::vector<HtScanAttempt> attempts;
};

enum class LocalEliminationStage : std::uint8_t {
  kJv,
  kHamiltonTutte,
};

enum class LocalEliminationTermination : std::uint8_t {
  kConverged,
  kJvRoundLimit,
  kHtEpochLimit,
};

struct LocalEliminationOptions {
  Backend jv_backend{Backend::kAuto};
  // 每次 HT 提交后，JV 必须先重新达到固定点；预算耗尽时不进入下一次 HT scan。
  std::uint32_t max_jv_rounds{100U};
  // 只计算实际执行的 HT 不可变快照 epoch；0 非法。
  std::uint32_t max_ht_epochs{100U};
  // 多 epoch 调度器拥有 offset：调用方必须传 0，提交后重排，无提交时继续当前 sweep。
  HtScanOptions ht_scan_options{};
};

struct LocalEliminationStageMetrics {
  std::uint32_t stage{};
  LocalEliminationStage kind{LocalEliminationStage::kJv};
  std::string backend;
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  std::size_t edges_before{};
  std::size_t edges_after{};
  std::size_t proposed{};
  std::size_t verified{};
  std::size_t rejected{};
  std::size_t committed{};
  std::uint32_t jv_rounds{};
  std::uint64_t eligible_targets{};
  std::uint64_t target_offset{};
  std::uint64_t attempted_targets{};
  std::uint64_t proven_targets{};
  std::uint64_t unresolved_targets{};
  double elapsed_ms{};
};

struct LocalEliminationResult {
  EliminationResult elimination;
  std::vector<LocalEliminationStageMetrics> stages;
  LocalEliminationTermination termination{LocalEliminationTermination::kHtEpochLimit};
};

[[nodiscard]] std::vector<Candidate> FindJvCandidatesCpu(const GraphSnapshot& graph);
[[nodiscard]] bool VerifyJvCandidate(const GraphSnapshot& graph, const Candidate& candidate,
                                     std::string* reason);

[[nodiscard]] bool CudaBackendAvailable(std::string* reason);
[[nodiscard]] std::vector<Candidate> FindJvCandidatesCuda(const GraphSnapshot& graph,
                                                          int* selected_device,
                                                          JvCudaCacheUsage* cache_usage = nullptr);
// 释放当前主机线程在所有设备上的 JV 驻留缓存；测试隔离和显式 teardown 使用。
void ClearJvCudaCache();

EliminationResult RunJvElimination(GraphSnapshot* graph, Backend backend, std::uint32_t max_rounds);

// 返回稳定 edge id；排序不依赖输入边数组的原始排列。
[[nodiscard]] std::vector<std::int32_t> SelectHtTargetEdgeIds(const GraphSnapshot& graph,
                                                              HtTargetOrder order);

// 在一个不可变快照上顺序搜索有界目标，最后通过 CommitHtProofEpoch 原子发布一次。
[[nodiscard]] HtScanResult RunHtScanEpoch(GraphSnapshot* graph, const HtScanOptions& options);

// 先把 JV 跑到固定点，再按稳定切片扫描 HT；任一 HT 提交后从新快照重跑 JV 并重排目标。
// 整个调用在工作副本上完成，异常时调用方图保持不变；返回 proof 可由 ReplayProof 独立重放。
[[nodiscard]] LocalEliminationResult RunLocalElimination(GraphSnapshot* graph,
                                                         const LocalEliminationOptions& options);

// 在同一不可变快照上整批复核 HT sidecars，再按规范边序执行一次原子 epoch 提交。
EliminationResult CommitHtProofEpoch(GraphSnapshot* graph,
                                     const std::vector<HtRecursiveProof>& proofs);

void WriteProof(const std::filesystem::path& path, const EliminationResult& result);
[[nodiscard]] EliminationResult ReadProof(const std::filesystem::path& path);
[[nodiscard]] EliminationResult ReplayProof(GraphSnapshot* graph,
                                            const EliminationResult& expected);

[[nodiscard]] std::string ToString(LocalEliminationStage stage);
[[nodiscard]] std::string ToString(LocalEliminationTermination termination);

} // namespace cudaee
