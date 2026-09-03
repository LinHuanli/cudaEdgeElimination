#pragma once

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

[[nodiscard]] PdlpResult RunFgpuPdlp(const GraphSnapshot& graph, const PdlpOptions& options);
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

struct FgpuRunReport {
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  std::size_t initial_edges{};
  std::size_t final_edges{};
  std::size_t geometry_committed{};
  std::size_t lp_committed{};
  std::size_t jv_committed{};
  std::size_t ht_committed{};
  std::size_t fixed_count{};
  std::size_t nonpair_count{};
  std::uint32_t pdlp_epochs{};
  double pdlp_total_solve_ms{};
  std::string termination;
  double total_ms{};
  GeometryMetrics geometry;
  PdlpResult pdlp;
  EliminationResult certificate;
};

[[nodiscard]] FgpuRunReport RunFgpuElimination(const FgpuInput& input,
                                               const FgpuOutputPaths& outputs,
                                               const FgpuConfig& config);

[[nodiscard]] FgpuRunReport VerifyFgpuCertificate(const FgpuInput& input,
                                                  const FgpuOutputPaths& outputs);

[[nodiscard]] std::string ToString(NumericMode mode);
[[nodiscard]] std::string ToString(VerificationMode mode);
[[nodiscard]] std::string ToString(ProofStatus status);
[[nodiscard]] std::string ToString(PdlpBackend backend);

} // namespace cudaee
