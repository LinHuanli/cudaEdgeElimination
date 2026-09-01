#pragma once

#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/path_system.hpp"

#include <compare>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
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
};

struct KOptReconnectTable {
  std::uint32_t k{};
  std::vector<EndpointMatching> templates;
  std::uint64_t generator_hash{};
};

// 生成 proper k-opt 模板：单巡回重连，且不重新加入任一被删除的抽象边。
[[nodiscard]] KOptReconnectTable BuildKOptReconnectTable(std::uint32_t k);

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
};

[[nodiscard]] std::uint64_t ComputePathSystemHash(const NormalizedPathSystem& paths);

// 穷举包含 required_edge（未给定时包含确定性 anchor）的 3/4/5-opt 重连。
[[nodiscard]] KOptSearchResult FindKOptWitness(const GraphSnapshot& graph,
                                               const NormalizedPathSystem& paths,
                                               const EndpointMatching& outside,
                                               const std::optional<NodeEdge>& required_edge,
                                               const KOptSearchOptions& options = {});

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
  std::vector<OutsideKOptWitness> records;
};

// 逐个解决未覆盖 outside matching，并用 inside coverage 合并重复叶证明。
[[nodiscard]] PathSystemKOptProof
ProvePathSystemByKOpt(const GraphSnapshot& graph, const NormalizedPathSystem& paths,
                      const std::optional<NodeEdge>& required_edge,
                      const KOptSearchOptions& options = {});

[[nodiscard]] bool VerifyPathSystemKOptProof(const GraphSnapshot& graph,
                                             const NormalizedPathSystem& paths,
                                             const std::optional<NodeEdge>& required_edge,
                                             const PathSystemKOptProof& proof, std::string* reason);

void WritePathSystemKOptProof(const std::filesystem::path& path, const PathSystemKOptProof& proof);
[[nodiscard]] PathSystemKOptProof ReadPathSystemKOptProof(const std::filesystem::path& path);

} // namespace cudaee
