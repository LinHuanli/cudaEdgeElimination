#pragma once
#include "cuda_edge_elimination/fgpu_execution.hpp"
#include "cuda_edge_elimination/fgpu_metrics.hpp"

#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/lp_epoch.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cudaee {

// 对外名称保持英文，关键语义用中文说明，避免把候选精度误当成证明精度。
enum class NumericMode : std::uint8_t {
  kMixedSafe,
  kFp64,
  kAggressiveFp32,
};

enum class VerificationMode : std::uint8_t {
  // 每个不可变 epoch 提交前执行 CPU 精确复核。
  kEpoch,
  // 允许设备影子图先运行，但最终输出前仍须顺序重放完整证明链。
  kDeferred,
};

enum class FgpuSolveMode : std::uint8_t {
  kGpuSafe,
  kGpuFastRaw,
};

enum class FgpuTermination : std::uint8_t {
  kFixedPoint,
  kPartialOom,
  kPartialDeviceError,
};

enum class ProofStatus : std::uint8_t {
  kProved,
  kExhausted,
  kUnresolved,
  kCancelled,
};

enum class PdlpBackend : std::uint8_t {
  kOff,
  kNative,
  kCuoptBaseline,
};

struct GeometryOptions {
  Backend backend{Backend::kAuto};
  NumericMode numeric_mode{NumericMode::kMixedSafe};
  // GPU filter 每条边按中点距离保留的点数；范围 [2,32]。
  std::uint32_t potential_candidates{16U};
  // 为避免首个浮点候选落在区间边界，每条边最多回传多个候选见证。
  std::uint32_t witnesses_per_edge{4U};
  int device{-1};
};

struct GeometryMetrics {
  std::size_t edges_before{};
  std::size_t proposed{};
  std::size_t verified{};
  std::size_t rejected{};
  std::size_t committed{};
  std::string backend;
  int selected_device{-1};
  double nearest_ms{};
  double upload_ms{};
  double kernel_ms{};
  double download_ms{};
  double verify_ms{};
  double commit_ms{};
};

// 最近邻半径属于实例度量，不随删边 epoch 改变；证书重放只构造一次。
struct GeometryVerificationData {
  std::vector<std::int64_t> nearest_rounded_distance;
  std::vector<std::int32_t> nearest_node;
};

struct GeometryEliminationResult {
  EliminationResult elimination;
  GeometryMetrics metrics;
};

struct PdlpOptions {
  PdlpBackend backend{PdlpBackend::kOff};
  std::uint32_t iterations{2000U};
  std::uint32_t fractional_bits{24U};
  int device{-1};
  std::string cuopt_library;
};

struct PdlpResult {
  std::string backend{"off"};
  int selected_device{-1};
  std::uint32_t iterations{};
  double solve_ms{};
  bool cpu_certified{false};
  ExactBound exact_bound;
  std::uint32_t fractional_bits{24U};
  std::vector<std::int64_t> vertex_dual_numerator;
  // 与 GraphSnapshot::edges 的 stable edge id 一一对应；非活动边为负无穷。
  std::vector<double> edge_scores;
};

struct SubtourPdlpOptions {
  std::string cuopt_library;
  // 非空时仅作为研究期 CPU exact-mincut 强度 oracle；正式 GPU 路径必须留空。
  std::filesystem::path mincut_oracle;
  int device{0};
  // 只用于判定数值 support 和严格违反割；不构成人为 epoch/规模上限。
  double support_epsilon{1.0e-8};
  double violation_epsilon{1.0e-7};
  std::uint32_t fractional_bits{24U};
};

struct SubtourPdlpResult {
  std::string backend{"cuopt-subtour-cut-loop"};
  bool converged{false};
  std::uint32_t epochs{};
  std::size_t cuts{};
  std::size_t support_components{};
  std::size_t forced_one_candidates{};
  double objective{};
  double dual_objective{};
  double solve_ms{};
  double total_ms{};
  ExactBound exact_bound;
  // 与 stable edge id 对齐；非活动边为负无穷。
  std::vector<double> edge_scores;
};

[[nodiscard]] PdlpResult RunFgpuPdlp(const GraphSnapshot& graph, const PdlpOptions& options);
[[nodiscard]] SubtourPdlpResult RunFgpuSubtourPdlp(const GraphSnapshot& graph,
                                                   std::int64_t incumbent_cost,
                                                   const SubtourPdlpOptions& options);
[[nodiscard]] EliminationResult RunLpBoxElimination(GraphSnapshot* graph, const PdlpResult& pdlp,
                                                    std::int64_t incumbent_cost);

[[nodiscard]] GeometryVerificationData BuildGeometryVerificationData(const GraphSnapshot& graph);

[[nodiscard]] bool VerifyGeometryCandidate(const GraphSnapshot& graph,
                                           const GeometryVerificationData& verification_data,
                                           const Candidate& candidate, std::string* reason);

[[nodiscard]] GeometryEliminationResult RunGeometryElimination(GraphSnapshot* graph,
                                                               const GeometryOptions& options);

struct FgpuConfig {
  int device{0};
  NumericMode numeric_mode{NumericMode::kMixedSafe};
  VerificationMode verification_mode{VerificationMode::kEpoch};
  PdlpBackend pdlp_backend{PdlpBackend::kOff};
  std::uint32_t potential_candidates{16U};
  std::uint32_t geometry_witnesses_per_edge{4U};
  std::uint32_t max_jv_rounds{100U};
  std::uint32_t max_ht_epochs{100U};
  std::uint64_t ht_targets_per_epoch{64U};
  // 多个 host target worker 共享同一张 GPU；不属于多 GPU 并行。
  std::uint32_t ht_target_workers{4U};
  std::uint32_t max_paths{6U};
  std::uint32_t max_local_nodes{32U};
  std::uint32_t pdlp_iterations{2000U};
  // LP 删除或 local 删除会改变 relaxation；在同一证明哈希链上重新求解到固定点。
  std::uint32_t max_pdlp_epochs{8U};
  std::string cuopt_library;
  bool enable_geometry{true};
  bool enable_jv{true};
  bool enable_hamilton_tutte{true};
};

struct FgpuInput {
  std::filesystem::path instance;
  std::filesystem::path input_edges;
  std::filesystem::path tour;
  bool tour_is_known_optimum{false};
  std::int64_t expected_tour_cost{-1};
};

struct FgpuOutputPaths {
  std::filesystem::path edges;
  std::filesystem::path fixed;
  std::filesystem::path nonpairs;
  std::filesystem::path certificate;
  std::filesystem::path manifest;
};

// 正式单 GPU 主链不暴露阶段开关或搜索预算；所有服务运行到
// 联合不动点。device=-1 时选择当前可见且剩余显存最多的一张卡。
struct FgpuSolveOptions {
  bool hybrid_e2e{false};
  bool enable_lp{true};
  bool distance_cache{true};
  bool main_pair_cache{true};
  bool quick_reply_cache{false};
  bool full_metric{true};
  bool leaf_permutation_cache{false};
  bool point_near_first{false};
  bool point_adaptive_start{false};
  bool point_prime_near{false};
  int device{-1};
  FgpuSolveMode mode{FgpuSolveMode::kGpuSafe};
  bool serialize_certificate{false};
  // 消融只切换 multiplier 求解器，不改变 GPU replay 或完整目标集。
  bool primal_dual_lp{true};
  PointLeafKernel point_leaf_kernel{PointLeafKernel::kPermutation};
  // 编译期驻留策略的运行期选择，不限制搜索任务或回复数量。
  std::uint32_t point_cta_blocks{4U};
};

struct FgpuSolveReport {
  FgpuBootstrapMetrics bootstrap;
  FgpuLpMetrics lp;
  FgpuTermination termination{FgpuTermination::kFixedPoint};
  bool gpu_replayed{false};
  bool unaudited{false};
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  // final_hash 只表示活动图；state hash 还覆盖 fixed 与 non-pair。
  std::uint64_t final_state_hash{};
  std::size_t initial_edges{};
  std::size_t final_edges{};
  std::size_t fixed_edges{};
  std::size_t pairs{};
  std::size_t nonpairs{};
  std::size_t lp_nonpairs{};
  std::size_t fixed_anchor_nonpairs{};
  std::size_t point_nonpairs{};
  std::size_t nonpair_fixed_edges{};
  std::size_t direct_fixed_edges{};
  std::size_t proof_replayed{};
  std::size_t proof_rejected{};
  std::size_t lp_connectivity_cuts{};
  std::size_t lp_path_closed_replies{};
  std::size_t point_path_end_closed_replies{};
  std::uint32_t lp_degree_snapshots{};
  std::uint32_t lp_strong_snapshots{};
  double lp_lower_bound{};
  int selected_device{-1};
  std::uint64_t resident_bytes{};
  double proof_replay_ms{};
  double commit_ms{};
  double gpu_solve_wall_ms{};
  double end_to_end_ms{};
};

struct FgpuRunReport {
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  std::size_t initial_edges{};
  std::size_t final_edges{};
  std::size_t geometry_committed{};
  std::size_t lp_committed{};
  std::size_t fixed_propagation_committed{};
  std::size_t jv_committed{};
  std::size_t ht_committed{};
  std::size_t quick_hs_committed{};
  std::size_t fixed_count{};
  std::size_t nonpair_count{};
  std::uint32_t pdlp_epochs{};
  double pdlp_total_solve_ms{};
  std::uintmax_t certificate_bytes{};
  std::string termination;
  double total_ms{};
  GeometryMetrics geometry;
  PdlpResult pdlp;
  EliminationResult certificate;
};

// 显式的单 GPU 常驻搜索。默认是无证书、无 CPU 逐边审计的全量 raw 路径；
// 如需正式认证可显式打开审计。raw 输出必须标记为 unaudited，不得冒充认证结果。
struct FgpuResidentConfig {
  bool hybrid_e2e{false};
  bool distance_cache{true};
  bool main_pair_cache{false};
  bool quick_reply_cache{false};
  bool full_metric{false};
  bool leaf_permutation_cache{false};
  bool point_near_first{false};
  bool point_adaptive_start{false};
  bool point_prime_near{false};
  int device{0};
  // 三个 max 字段取 0 时表示不设人为上限，运行到自然固定点。
  std::uint32_t max_hs_epochs{0U};
  std::uint32_t max_jv_rounds{0U};
  std::uint32_t potential_candidates{32U};
  std::uint32_t main_edge_potentials{11U};
  std::uint32_t main_edge_positions{23U};
  std::uint32_t quick_hs_candidates{10U};
  std::uint32_t quick_hs_pair_trials{10U};
  std::uint32_t pdlp_iterations{5000U};
  std::uint32_t max_pdlp_epochs{0U};
  bool enable_quick_hs{true};
  bool enable_jv{true};
  bool enable_geometry{true};
  bool enable_pdlp{true};
  // SEC 上的原始-对偶候选与旧 degree-box ensemble 比较精确下界。
  bool enable_primal_dual_lp{false};
  PointLeafKernel point_leaf_kernel{PointLeafKernel::kPermutation};
  std::uint32_t point_cta_blocks{4U};
  // 新 one-shot 路径启用；旧 resident 默认关闭以冻结既有基线语义。
  bool enable_main_edge{false};
  // 低度数 Main-Edge reply 启用 KH strong-3-opt/metric-excess。
  bool enable_strong_metric{false};
  bool enable_extra_edge{false};
  // GPU continuation 深度；当前 1/2 分别对应 KH -e1/-e2。
  std::uint32_t extra_edge_depth{1U};
  bool quick_hs_two_hop{false};
  // tour 仍可提供 LP 上界；关闭此项可做不偏置的论文强度比较。
  bool protect_tour{true};
  bool enable_cpu_audit{false};
  // 仅由正式 solve 路径开启；legacy resident CLI 保持旧基线。
  bool enable_gpu_replay{false};
  bool serialize_gpu_certificate{false};
  bool enable_fixing{false};
  bool enable_point_nonpair{false};
  // 对每条活动边完整枚举两个端点的允许邻边对笛卡尔积。
  bool enable_direct_fix{false};
};

struct FgpuResidentRunReport {
  FgpuBootstrapMetrics bootstrap;
  FgpuLpMetrics lp;
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  std::uint64_t final_state_hash{};
  std::size_t initial_edges{};
  std::size_t final_edges{};
  std::size_t fixed_count{};
  std::size_t pair_count{};
  std::size_t nonpair_count{};
  std::size_t lp_nonpair_committed{};
  std::size_t fixed_anchor_nonpair_committed{};
  std::size_t point_nonpair_committed{};
  std::size_t nonpair_fix_committed{};
  std::size_t direct_fix_committed{};
  std::size_t jv_committed{};
  std::size_t quick_hs_committed{};
  std::size_t extra_edge_committed{};
  std::size_t geometry_committed{};
  std::size_t main_edge_committed{};
  std::size_t lp_committed{};
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
  std::size_t lp_path_closed_replies{};
  std::size_t point_path_end_closed_replies{};
  std::uint32_t lp_degree_snapshots{};
  std::uint32_t lp_strong_snapshots{};
  double lp_lower_bound{};
  std::uint32_t main_edge_epochs{};
  bool converged{false};
  bool cpu_audited{false};
  int selected_device{-1};
  std::uint64_t resident_bytes{};
  std::uint64_t main_pair_cache_bytes{};
  std::uint64_t quick_reply_cache_bytes{};
  std::uint64_t quick_reply_raw_pairs{};
  std::uint64_t quick_reply_compact_pairs{};
  double quick_reply_build_ms{};
  double quick_reply_validation_ms{};
  double upload_ms{};
  double gpu_kernel_ms{};
  double geometry_ms{};
  double main_edge_ms{};
  double pdlp_ms{};
  double jv_ms{};
  double quick_hs_ms{};
  double extra_edge_ms{};
  double proof_replay_ms{};
  double commit_ms{};
  double compaction_ms{};
  double gpu_download_ms{};
  // 从已解析图开始，到设备 fixed point 与最终 mask 回传；不含 CPU audit/文件写出。
  double gpu_solve_wall_ms{};
  double cpu_audit_ms{};
  double output_ms{};
  double end_to_end_ms{};
  double trusted_total_ms{};
  std::uintmax_t certificate_bytes{};
  std::size_t proof_replayed{};
  std::size_t proof_rejected{};
  EliminationResult certificate;
};

[[nodiscard]] FgpuSolveReport RunFgpuElimination(const FgpuInput& input,
                                                 const FgpuOutputPaths& outputs,
                                                 const FgpuSolveOptions& options);

[[nodiscard]] FgpuRunReport RunFgpuElimination(const FgpuInput& input,
                                               const FgpuOutputPaths& outputs,
                                               const FgpuConfig& config);

[[nodiscard]] FgpuRunReport VerifyFgpuCertificate(const FgpuInput& input,
                                                  const FgpuOutputPaths& outputs);

[[nodiscard]] FgpuResidentRunReport RunFgpuResidentLocal(const FgpuInput& input,
                                                         const FgpuOutputPaths& outputs,
                                                         const FgpuResidentConfig& config);

[[nodiscard]] FgpuResidentRunReport RunFgpuResidentElimination(const FgpuInput& input,
                                                               const FgpuOutputPaths& outputs,
                                                               const FgpuResidentConfig& config);

[[nodiscard]] std::string ToString(NumericMode mode);
[[nodiscard]] std::string ToString(VerificationMode mode);
[[nodiscard]] std::string ToString(ProofStatus status);
[[nodiscard]] std::string ToString(PdlpBackend backend);
[[nodiscard]] std::string ToString(FgpuSolveMode mode);
[[nodiscard]] std::string ToString(FgpuTermination termination);

} // namespace cudaee
