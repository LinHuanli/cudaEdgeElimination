#pragma once

#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cudaee::detail {

struct ResidentGpuOptions {
  int device{-1};
  // 0 表示不设人为轮数上限，直到该阶段自然达到固定点。
  std::uint32_t max_hs_epochs{0U};
  std::uint32_t max_jv_rounds{0U};
  bool enable_quick_hs{true};
  bool enable_jv{true};
  bool enable_geometry{false};
  bool enable_pdlp{false};
  // 关闭时不回传逐 epoch bitmap/witness，只保留计数与最终 active mask。
  bool collect_trace{true};
  std::uint32_t potential_candidates{32U};
  std::uint32_t pdlp_iterations{5000U};
  // 0 表示持续交错 LP/local，直到整个 orchestration 无新增删除。
  std::uint32_t max_pdlp_epochs{0U};
  std::uint32_t fractional_bits{24U};
  std::int64_t incumbent_cost{-1};
};

struct ResidentTraceEpoch {
  EliminationMethod method{EliminationMethod::kJv};
  std::size_t edges_before{};
  std::vector<std::int32_t> edge_ids;
  std::vector<std::int32_t> first_witness;
  std::vector<std::int32_t> second_witness;
  // LP_BOX epoch 的共享量化 dual；其他方法为空。
  std::vector<std::int64_t> vertex_dual_numerator;
  std::uint32_t fractional_bits{24U};
  std::int64_t incumbent_cost{-1};
  double device_ms{};
};

struct ResidentGpuResult {
  int selected_device{-1};
  std::string backend{"none"};
  std::vector<std::uint8_t> final_active;
  std::vector<ResidentTraceEpoch> epochs;
  std::size_t initial_edges{};
  std::size_t final_edges{};
  std::size_t jv_committed{};
  std::size_t quick_hs_committed{};
  std::size_t geometry_committed{};
  std::size_t lp_committed{};
  std::uint32_t hs_epochs{};
  std::uint32_t jv_rounds{};
  std::uint32_t pdlp_epochs{};
  bool converged{false};
  double upload_ms{};
  double kernel_ms{};
  double geometry_ms{};
  double pdlp_ms{};
  double jv_ms{};
  double quick_hs_ms{};
  double compaction_ms{};
  double download_ms{};
  double solve_wall_ms{};
  std::uint64_t resident_bytes{};
};

[[nodiscard]] ResidentGpuResult
RunResidentEliminationCuda(const GraphSnapshot& graph,
                           const std::vector<std::uint8_t>& protected_edges,
                           const ResidentGpuOptions& options);

[[nodiscard]] bool ResidentEliminationCudaAvailable(std::string* reason);

} // namespace cudaee::detail
