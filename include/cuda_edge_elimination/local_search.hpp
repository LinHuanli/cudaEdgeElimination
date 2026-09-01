#pragma once

#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/path_system.hpp"

#include <array>
#include <compare>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cudaee {

struct NodeEdge {
  std::int32_t u{};
  std::int32_t v{};

  auto operator<=>(const NodeEdge&) const = default;
};

enum class KOptSearchStatus : std::uint8_t {
  kImproved,
  kNoImprovement,
  kUnresolved,
  kInvalid,
};

struct KOptSearchOptions {
  std::uint32_t max_k{5};
  // 0 表示穷举；非零预算耗尽时返回 unresolved，不能解释为“无改善”。
  std::uint64_t max_deletion_sets{};
  PathCompatibilityBackend cost_backend{PathCompatibilityBackend::kCpu};
  std::uint32_t cost_batch_size{4096};
  // auto 后端中，小于该 cost-cell 数的融合矩阵走 CPU；0 表示始终尝试 CUDA。
  std::uint64_t cuda_min_cost_cells{128};
  // 0 禁用；非零时在 k-opt 未解决后运行收缩 outside matching 的精确 DP（硬上限 18）。
  std::uint32_t exact_fallback_max_blocks{};
};

struct KOptReconnectTable {
  std::uint32_t k{};
  std::vector<EndpointMatching> templates;
  std::uint64_t generator_hash{};
};

// 生成 proper k-opt 模板：单巡回重连，且不重新加入任一被删除的抽象边。
[[nodiscard]] KOptReconnectTable BuildKOptReconnectTable(std::uint32_t k);

constexpr std::int64_t kInvalidKOptTemplateCost = INT64_MAX;

struct KOptCostTask {
  std::array<std::int32_t, 10> port_nodes{};
  std::int64_t deleted_cost{};
};

struct KOptCudaCacheUsage {
  bool snapshot_hit{false};
  bool template_hit{false};
  bool workspace_hit{false};
  std::uint64_t resident_bytes{};
};

struct KOptCostBatchResult {
  std::uint32_t k{};
  std::uint32_t template_count{};
  std::vector<std::int64_t> added_costs;
  std::string backend;
  int selected_device{-1};
  KOptCudaCacheUsage cuda_cache;
};

// 返回 [task][template] 精确成本矩阵。它只是候选 oracle，不能用“无命中”授权证明。
[[nodiscard]] KOptCostBatchResult EvaluateKOptTemplateCosts(const GraphSnapshot& graph,
                                                            std::uint32_t k,
                                                            const std::vector<KOptCostTask>& tasks,
                                                            PathCompatibilityBackend backend);

struct KOptWitness {
  std::uint32_t k{};
  std::int64_t deleted_cost{};
  std::int64_t added_cost{};
  std::vector<NodeEdge> deleted_edges;
  std::vector<NodeEdge> added_edges;
  EndpointMatching inside_matching;
};

struct KOptSearchResult {
  KOptSearchStatus status{KOptSearchStatus::kInvalid};
  std::string reason;
  KOptWitness witness;
  std::uint64_t deletion_sets_tested{};
  std::uint64_t reconnect_matchings_tested{};
  std::uint64_t exact_states_tested{};
};

[[nodiscard]] std::uint64_t ComputePathSystemHash(const NormalizedPathSystem& paths);

// 穷举包含 required_edge（未给定时包含确定性 anchor）的 3/4/5-opt 重连。
[[nodiscard]] KOptSearchResult FindKOptWitness(const GraphSnapshot& graph,
                                               const NormalizedPathSystem& paths,
                                               const EndpointMatching& outside,
                                               const std::optional<NodeEdge>& required_edge,
                                               const KOptSearchOptions& options = {});

// 精确求解所有包含 outside matching 且不含 required_edge 的局部巡回。
[[nodiscard]] KOptSearchResult FindExactTourWitness(const GraphSnapshot& graph,
                                                    const NormalizedPathSystem& paths,
                                                    const EndpointMatching& outside,
                                                    const std::optional<NodeEdge>& required_edge,
                                                    std::uint32_t max_blocks);

// 从原路径巡回独立重建删边、加边、严格成本改善和 inside matching。
[[nodiscard]] bool VerifyKOptWitness(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                                     const EndpointMatching& outside,
                                     const std::optional<NodeEdge>& required_edge,
                                     const KOptWitness& witness, std::string* reason);

struct OutsideKOptWitness {
  std::uint32_t source_outside_index{};
  KOptWitness witness;
};

struct PathSystemKOptProof {
  bool proven{false};
  std::string reason;
  std::uint64_t snapshot_hash{};
  std::uint64_t path_system_hash{};
  std::uint64_t compatibility_table_hash{};
  std::uint32_t path_count{};
  std::uint32_t outside_count{};
  std::uint64_t deletion_sets_tested{};
  std::uint64_t reconnect_matchings_tested{};
  std::uint64_t exact_states_tested{};
  std::vector<OutsideKOptWitness> records;
};

struct PathSystemKOptBatchResult {
  std::vector<PathSystemKOptProof> proofs;
  std::string cost_backend{"none"};
  int selected_device{-1};
  bool cpu_verified{false};
  std::uint64_t cost_batches{};
  std::uint64_t cost_tasks{};
  std::uint64_t cost_cells{};
  std::uint64_t scalar_searches{};
  std::uint64_t cuda_cost_batches{};
  std::uint64_t snapshot_cache_hits{};
  std::uint64_t template_cache_hits{};
  std::uint64_t workspace_cache_hits{};
  std::uint64_t peak_device_cache_bytes{};
  std::uint64_t cpu_long_tail_batches{};
  std::uint64_t cpu_long_tail_tasks{};
  std::uint64_t cpu_long_tail_cells{};
  std::uint64_t cost_rows_consumed{};
  std::uint64_t candidate_templates_rechecked{};
  std::uint64_t cpu_completeness_rows{};
  std::uint64_t cpu_completeness_templates{};
  // 同步 wall time 只用于诊断，不进入 leaf proof。
  double setup_ms{};
  double cursor_prepare_ms{};
  double cost_evaluate_ms{};
  double cost_scatter_ms{};
  double cursor_consume_ms{};
  double candidate_recheck_ms{};
  double completeness_fallback_ms{};
  double scalar_search_ms{};
  double apply_ms{};
  double proof_verify_ms{};
};

// 逐个解决未覆盖 outside matching，并用 inside coverage 合并重复叶证明。
[[nodiscard]] PathSystemKOptProof
ProvePathSystemByKOpt(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                      const std::optional<NodeEdge>& required_edge,
                      const KOptSearchOptions& options = {});

// 以增量组合游标跨 path systems 合并同 k cost blocks；不预展开后续 deletion sets。
[[nodiscard]] PathSystemKOptBatchResult ProvePathSystemsByKOpt(
    const GraphSnapshot& graph, const std::vector<NormalizedPathSystem>& path_systems,
    const std::optional<NodeEdge>& required_edge, const KOptSearchOptions& options = {});

[[nodiscard]] bool VerifyPathSystemKOptProof(const GraphSnapshot& graph,
                                             const NormalizedPathSystem& paths,
                                             const std::optional<NodeEdge>& required_edge,
                                             const PathSystemKOptProof& proof, std::string* reason);

void WritePathSystemKOptProof(const std::filesystem::path& path, const PathSystemKOptProof& proof);
[[nodiscard]] PathSystemKOptProof ReadPathSystemKOptProof(const std::filesystem::path& path);

// 字符串接口供更高层证明格式嵌套 V1 叶证书，仍执行与文件读取器相同的严格校验。
[[nodiscard]] std::string SerializePathSystemKOptProof(const PathSystemKOptProof& proof);
[[nodiscard]] PathSystemKOptProof ParsePathSystemKOptProof(std::string_view serialized);

namespace detail {

[[nodiscard]] bool KOptCostCudaAvailable(std::string* reason);
[[nodiscard]] std::vector<std::int64_t>
EvaluateKOptTemplateCostsCuda(const GraphSnapshot& graph, const KOptReconnectTable& table,
                              const std::vector<KOptCostTask>& tasks, int* selected_device,
                              KOptCudaCacheUsage* cache_usage);

// 释放当前主机线程在所有可见设备上的 k-opt 驻留缓存；主要用于 epoch 切换与测试隔离。
void ClearKOptCostCudaCache();

} // namespace detail

} // namespace cudaee
