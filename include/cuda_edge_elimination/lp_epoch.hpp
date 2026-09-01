#pragma once

#include <cstdint>
#include <filesystem>
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
  std::vector<double> primal;
  std::vector<double> dual;
  std::vector<double> reduced_costs;
  ExactBound exact_model_bound;
};

[[nodiscard]] LpEpoch ReadLpEpoch(const std::filesystem::path& path);
void WriteLpEpoch(const std::filesystem::path& path, const LpEpoch& epoch);
void WriteLpSolution(const std::filesystem::path& path, const LpEpoch& epoch,
                     const LpSolution& solution);

[[nodiscard]] LpSolution SolveWithCuOpt(const LpEpoch& epoch, const std::string& library_path);
[[nodiscard]] ExactBound BuildExactModelBound(const LpEpoch& epoch, const std::vector<double>& dual,
                                              std::uint32_t fractional_bits = 24);

} // namespace cudaee
