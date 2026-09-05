#pragma once

#include <cstdint>

namespace cudaee {

// LP 服务的互斥阶段计时；pdhg_ms 是 solver_ms 的子集，不可再次求和。
// proposal 计数与实际提交计数分开，避免把重复 reply 当成新增 non-pair。
struct FgpuLpMetrics {
  double solver_ms{};
  double cut_separation_ms{};
  double point_ms{};
  double fixing_ms{};
  double pair_filter_ms{};
  double pdhg_model_ms{};
  double pdhg_ms{};
  double pdhg_primal_violation{};
  double pdhg_relative_gap{};
  std::uint64_t pdhg_iterations{};
  std::uint32_t pdhg_selected_snapshots{};
  std::uint64_t validated_transactions{};
  std::uint32_t point_registers{};
  std::uint32_t point_active_blocks_per_sm{};
  std::uint64_t point_local_bytes_per_thread{};
};

} // namespace cudaee
