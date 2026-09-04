#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace cudaee {

struct LpEpoch {
  std::int32_t rows{};
  std::int32_t columns{};
  std::int32_t objective_sense{1};
  double objective_offset{};
  std::vector<double> objective;
  std::vector<std::int32_t> row_offsets;
  std::vector<std::int32_t> column_indices;
  std::vector<double> values;
  std::vector<char> senses;
  std::vector<double> rhs;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  std::vector<char> variable_types;
  std::vector<std::int32_t> edge_u;
  std::vector<std::int32_t> edge_v;
  std::uint64_t content_hash{};

  void Validate() const;
  [[nodiscard]] std::uint64_t ComputeHash() const;
};

struct ExactBound {
  bool certified{false};
  std::string numerator;
  std::uint64_t denominator{1};
  std::string reason;
};

// 将浮点 dual 统一量化后，同时返回模型下界和每列精确 reduced cost。后者用于
// forced-one 诊断，避免把未量化的求解器 reduced cost 与精确下界混用。
struct ExactModelEvaluation {
  ExactBound bound;
  __int128 lower_bound_numerator{};
  std::vector<__int128> reduced_cost_numerator;
};

struct LpSolution {
  std::string solver;
  std::string solver_version;
  std::string status;
  std::int32_t termination_status{};
  double objective{};
  double dual_objective{};
  double solve_time_seconds{};
  double max_primal_violation{};
  double max_reduced_cost_residual{};
  bool numerically_accepted{false};
  bool warm_start_attempted{false};
  bool warm_start_applied{false};
  double warm_start_column_coverage{};
  double warm_start_row_coverage{};
  std::string warm_start_reason;
  std::uint64_t stable_identity_hash{};
  std::vector<double> primal;
  std::vector<double> dual;
  std::vector<double> reduced_costs;
  ExactBound exact_model_bound;
};

// 由 Concorde 的列—边映射和规范化行内容导出的跨 epoch 稳定身份。
// complete=false 时必须禁用 warm start，不能退化为按位置猜测映射。
struct LpStableIdentity {
  std::vector<std::uint64_t> column_ids;
  std::vector<std::uint64_t> row_ids;
  std::uint64_t identity_hash{};
  bool complete{false};
  std::string reason;
};

struct LpWarmStart {
  std::uint64_t source_model_hash{};
  LpStableIdentity identity;
  std::vector<double> primal;
  std::vector<double> dual;
};

struct LpWarmStartProjection {
  std::vector<double> primal;
  std::vector<double> dual;
  double column_coverage{};
  double row_coverage{};
  bool accepted{false};
  std::string reason;
};

[[nodiscard]] LpEpoch ReadLpEpoch(const std::filesystem::path& path);
void WriteLpEpoch(const std::filesystem::path& path, const LpEpoch& epoch);
void WriteLpSolution(const std::filesystem::path& path, const LpEpoch& epoch,
                     const LpSolution& solution);

[[nodiscard]] LpSolution SolveWithCuOpt(const LpEpoch& epoch, const std::string& library_path);
[[nodiscard]] ExactBound BuildExactModelBound(const LpEpoch& epoch, const std::vector<double>& dual,
                                              std::uint32_t fractional_bits = 24);
[[nodiscard]] ExactModelEvaluation BuildExactModelEvaluation(const LpEpoch& epoch,
                                                             const std::vector<double>& dual,
                                                             std::uint32_t fractional_bits = 24);
[[nodiscard]] LpStableIdentity ComputeLpStableIdentity(const LpEpoch& epoch);
[[nodiscard]] LpWarmStart BuildLpWarmStart(const LpEpoch& epoch, const LpSolution& solution);
[[nodiscard]] LpWarmStartProjection
ProjectLpWarmStart(const LpWarmStart& source, const LpEpoch& target, double minimum_coverage = 0.8);

// 会话复用已加载的 cuOpt C API，并在稳定身份覆盖率达标时投影 PDLP primal/dual。
class CuOptSession {
public:
  explicit CuOptSession(std::string library_path = {});
  ~CuOptSession();
  CuOptSession(CuOptSession&&) noexcept;
  CuOptSession& operator=(CuOptSession&&) noexcept;
  CuOptSession(const CuOptSession&) = delete;
  CuOptSession& operator=(const CuOptSession&) = delete;

  [[nodiscard]] LpSolution Solve(const LpEpoch& epoch, bool enable_warm_start = true,
                                 double minimum_coverage = 0.8);
  void ClearWarmStart();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaee
