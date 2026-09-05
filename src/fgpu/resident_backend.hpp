#pragma once

#include "cuda_edge_elimination/fgpu_execution.hpp"
#include "cuda_edge_elimination/fgpu_metrics.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cudaee::detail {

struct ResidentGpuOptions {
  // 由同设备的 GpuBootstrap 持有；消除结束前不得释放。
  const std::int64_t* triangular_distance{};
  const std::uint8_t* permutation_orders{};
  bool gpu_complete_graph{false};
  bool main_pair_cache{false};
  bool full_metric{false};
  bool point_near_first{false};
  bool point_adaptive_start{false};
  bool progress_log{false};
  int device{-1};
  // 0 表示不设人为轮数上限，直到该阶段自然达到固定点。
  std::uint32_t max_hs_epochs{0U};
  std::uint32_t max_jv_rounds{0U};
  bool enable_quick_hs{true};
  bool enable_jv{true};
  bool enable_geometry{false};
  bool enable_pdlp{false};
  bool enable_primal_dual_lp{false};
  PointLeafKernel point_leaf_kernel{PointLeafKernel::kPermutation};
  std::uint32_t point_cta_blocks{2U};
  bool enable_main_edge{false};
  bool enable_strong_metric{false};
  // KH-ElimTSP -e1/-e2：在基础 2/3/3 reply 上继续揭示相邻边。
  bool enable_extra_edge{false};
  std::uint32_t extra_edge_depth{1U};
  std::uint32_t quick_hs_candidates{10U};
  // 0 表示扫描候选集内的全部 pair。
  std::uint32_t quick_hs_pair_trials{10U};
  bool quick_hs_two_hop{false};
  // safe 主链使用与 proposer 分离的 GPU replay kernel 授权提交。
  bool gpu_replay{false};
  bool enable_fixing{false};
  // 对每个未排除邻边对执行完整一层 HT point-move AND replies。
  bool enable_point_nonpair{false};
  bool enable_direct_fix{false};
  // 关闭时不回传逐 epoch bitmap/witness，只保留计数与最终 active mask。
  bool collect_trace{true};
  std::uint32_t potential_candidates{32U};
  // 2014 Step 2 在边中点附近检查的节点数；这是方法深度而非任务预算。
  std::uint32_t main_edge_potentials{11U};
  std::uint32_t main_edge_positions{23U};
  std::uint32_t pdlp_iterations{5000U};
  // 0 表示持续交错 LP/local，直到整个 orchestration 无新增删除。
  std::uint32_t max_pdlp_epochs{0U};
  std::uint32_t fractional_bits{24U};
  std::int64_t incumbent_cost{-1};
};

struct ResidentTraceEpoch {
  EliminationMethod method{EliminationMethod::kJv};
  bool main_edge_stage{false};
  bool extra_edge_stage{false};
  std::int32_t main_position{};
  std::size_t edges_before{};
  std::vector<std::int32_t> edge_ids;
  std::vector<std::int32_t> first_witness;
  std::vector<std::int32_t> second_witness;
  std::vector<std::int32_t> fixed_edge_ids;
  // LP_BOX epoch 的共享量化 dual；其他方法为空。
  std::vector<std::int64_t> vertex_dual_numerator;
  std::vector<std::int64_t> local_sec_dual_numerator;
  std::uint32_t fractional_bits{24U};
  std::int64_t incumbent_cost{-1};
  double device_ms{};
};

struct ResidentNonpair {
  std::int32_t center{-1};
  std::int32_t first{-1};
  std::int32_t second{-1};
};

struct ResidentGpuResult {
  FgpuLpMetrics lp;
  int selected_device{-1};
  std::string backend{"none"};
  std::vector<std::uint8_t> final_active;
  std::vector<std::uint8_t> final_fixed;
  std::vector<ResidentNonpair> final_nonpairs;
  std::vector<ResidentTraceEpoch> epochs;
  std::size_t initial_edges{};
  std::size_t final_edges{};
  std::size_t jv_committed{};
  std::size_t quick_hs_committed{};
  std::size_t extra_edge_committed{};
  std::size_t geometry_committed{};
  std::size_t main_edge_committed{};
  std::size_t lp_committed{};
  std::size_t nonpair_committed{};
  std::size_t lp_nonpair_committed{};
  std::size_t fixed_anchor_nonpair_committed{};
  std::size_t point_nonpair_committed{};
  std::size_t nonpair_fix_committed{};
  std::size_t direct_fix_committed{};
  std::size_t fixed_count{};
  std::size_t fixed_propagation_committed{};
  std::uint32_t hs_epochs{};
  std::uint32_t hs_full_sweeps{};
  std::uint32_t hs_active_sweeps{};
  std::uint64_t hs_full_tasks{};
  std::uint64_t hs_active_tasks{};
  std::uint32_t extra_edge_epochs{};
  std::uint32_t jv_rounds{};
  std::uint32_t pdlp_epochs{};
  std::size_t lp_connectivity_cuts{};
  // 已发布的精确 LP dual 在 Quick/HT reply 层关闭的路径系统数。
  std::size_t lp_path_closed_replies{};
  // point move 的 3+3 reply 经完整 path-end 分支后才关闭的数量。
  std::size_t point_path_end_closed_replies{};
  std::uint32_t lp_degree_snapshots{};
  std::uint32_t lp_strong_snapshots{};
  double lp_lower_bound{};
  std::uint32_t main_edge_epochs{};
  bool converged{false};
  double upload_ms{};
  double kernel_ms{};
  double geometry_ms{};
  double main_edge_ms{};
  double pdlp_ms{};
  double jv_ms{};
  double quick_hs_ms{};
  double extra_edge_ms{};
  double proof_replay_ms{};
  double commit_ms{};
  double compaction_ms{};
  double download_ms{};
  double solve_wall_ms{};
  std::size_t proof_replayed{};
  std::size_t proof_rejected{};
  std::uint64_t resident_bytes{};
  std::uint64_t main_pair_cache_bytes{};
};

// 测试入口：同一批三路径状态分别由 host 精确枚举和 GPU warp-DP
// 判定，用于捕获深层 continuation 的漏状态、严格不等号或 fixed 合并偏差。
struct QuickHsPathDifferentialResult {
  std::size_t cases{};
  std::size_t mismatches{};
};

[[nodiscard]] ResidentGpuResult
RunResidentEliminationCuda(const GraphSnapshot& graph,
                           const std::vector<std::uint8_t>& protected_edges,
                           const ResidentGpuOptions& options);

[[nodiscard]] bool ResidentEliminationCudaAvailable(std::string* reason);

[[nodiscard]] QuickHsPathDifferentialResult RunQuickHsPathDifferentialCuda(int device,
                                                                           std::uint32_t samples);

} // namespace cudaee::detail
