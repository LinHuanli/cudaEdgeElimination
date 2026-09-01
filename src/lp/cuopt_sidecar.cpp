#include "cuda_edge_elimination/lp_epoch.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::int32_t kCuOptSuccess = 0;
constexpr std::int32_t kCuOptMinimize = 1;
constexpr std::int32_t kCuOptOptimal = 1;
constexpr std::int32_t kCuOptMethodPdlp = 1;
constexpr std::int32_t kCuOptDoublePrecision = 1;
constexpr std::int32_t kCuOptPresolveOff = 0;
constexpr std::int32_t kCuOptPdlpMethodical = 2;

using Handle = void*;

template <typename Function> Function LoadSymbol(void* library, const char* name) {
  dlerror();
  void* symbol = dlsym(library, name);
  const char* error = dlerror();
  if (error != nullptr || symbol == nullptr) {
    throw std::runtime_error(std::string("cuOpt C API 缺少符号 ") + name + ": " +
                             (error == nullptr ? "unknown" : error));
  }
  return reinterpret_cast<Function>(symbol);
}

class CuOptApi {
public:
  explicit CuOptApi(const std::string& requested_path) {
    const char* path = requested_path.empty() ? "libcuopt.so" : requested_path.c_str();
    library_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      throw std::runtime_error(std::string("无法加载 cuOpt 共享库 ") + path + ": " + dlerror());
    }
    get_float_size = LoadSymbol<GetSize>(library_, "cuOptGetFloatSize");
    get_int_size = LoadSymbol<GetSize>(library_, "cuOptGetIntSize");
    get_version = LoadSymbol<GetVersion>(library_, "cuOptGetVersion");
    create_problem = LoadSymbol<CreateProblem>(library_, "cuOptCreateProblem");
    destroy_problem = LoadSymbol<DestroyHandle>(library_, "cuOptDestroyProblem");
    create_settings = LoadSymbol<CreateHandle>(library_, "cuOptCreateSolverSettings");
    destroy_settings = LoadSymbol<DestroyHandle>(library_, "cuOptDestroySolverSettings");
    set_integer = LoadSymbol<SetInteger>(library_, "cuOptSetIntegerParameter");
    set_float = LoadSymbol<SetFloat>(library_, "cuOptSetFloatParameter");
    solve = LoadSymbol<Solve>(library_, "cuOptSolve");
    destroy_solution = LoadSymbol<DestroyHandle>(library_, "cuOptDestroySolution");
    get_termination = LoadSymbol<GetInteger>(library_, "cuOptGetTerminationStatus");
    get_primal = LoadSymbol<GetArray>(library_, "cuOptGetPrimalSolution");
    get_dual = LoadSymbol<GetArray>(library_, "cuOptGetDualSolution");
    get_reduced_costs = LoadSymbol<GetArray>(library_, "cuOptGetReducedCosts");
    get_objective = LoadSymbol<GetFloat>(library_, "cuOptGetObjectiveValue");
    get_dual_objective = LoadSymbol<GetFloat>(library_, "cuOptGetDualObjectiveValue");
    get_solve_time = LoadSymbol<GetFloat>(library_, "cuOptGetSolveTime");
  }

  ~CuOptApi() {
    if (library_ != nullptr) {
      dlclose(library_);
    }
  }

  CuOptApi(const CuOptApi&) = delete;
  CuOptApi& operator=(const CuOptApi&) = delete;

  using GetSize = std::int8_t (*)();
  using GetVersion = std::int32_t (*)(std::int32_t*, std::int32_t*, std::int32_t*);
  using CreateProblem = std::int32_t (*)(std::int32_t, std::int32_t, std::int32_t, double,
                                         const double*, const std::int32_t*, const std::int32_t*,
                                         const double*, const char*, const double*, const double*,
                                         const double*, const char*, Handle*);
  using CreateHandle = std::int32_t (*)(Handle*);
  using DestroyHandle = void (*)(Handle*);
  using SetInteger = std::int32_t (*)(Handle, const char*, std::int32_t);
  using SetFloat = std::int32_t (*)(Handle, const char*, double);
  using Solve = std::int32_t (*)(Handle, Handle, Handle*);
  using GetInteger = std::int32_t (*)(Handle, std::int32_t*);
  using GetArray = std::int32_t (*)(Handle, double*);
  using GetFloat = std::int32_t (*)(Handle, double*);

  GetSize get_float_size{};
  GetSize get_int_size{};
  GetVersion get_version{};
  CreateProblem create_problem{};
  DestroyHandle destroy_problem{};
  CreateHandle create_settings{};
  DestroyHandle destroy_settings{};
  SetInteger set_integer{};
  SetFloat set_float{};
  Solve solve{};
  DestroyHandle destroy_solution{};
  GetInteger get_termination{};
  GetArray get_primal{};
  GetArray get_dual{};
  GetArray get_reduced_costs{};
  GetFloat get_objective{};
  GetFloat get_dual_objective{};
  GetFloat get_solve_time{};

private:
  void* library_{nullptr};
};

void CheckStatus(const std::int32_t status, const char* operation) {
  if (status != kCuOptSuccess) {
    throw std::runtime_error(std::string(operation) +
                             " 失败，cuOpt status=" + std::to_string(status));
  }
}

std::string TerminationName(const std::int32_t status) {
  switch (status) {
  case 1:
    return "OPTIMAL";
  case 2:
    return "INFEASIBLE";
  case 3:
    return "UNBOUNDED";
  case 4:
    return "ITERATION_LIMIT";
  case 5:
    return "TIME_LIMIT";
  case 6:
    return "NUMERICAL_ERROR";
  case 7:
    return "PRIMAL_FEASIBLE";
  default:
    return "OTHER_" + std::to_string(status);
  }
}

double ComputeMaxPrimalViolation(const LpEpoch& epoch, const std::vector<double>& primal) {
  double maximum = 0.0;
  for (std::int32_t column = 0; column < epoch.columns; ++column) {
    maximum = std::max(maximum, epoch.lower_bounds[static_cast<std::size_t>(column)] -
                                    primal[static_cast<std::size_t>(column)]);
    maximum = std::max(maximum, primal[static_cast<std::size_t>(column)] -
                                    epoch.upper_bounds[static_cast<std::size_t>(column)]);
  }
  for (std::int32_t row = 0; row < epoch.rows; ++row) {
    long double activity = 0.0L;
    for (std::int32_t offset = epoch.row_offsets[static_cast<std::size_t>(row)];
         offset < epoch.row_offsets[static_cast<std::size_t>(row) + 1]; ++offset) {
      activity +=
          epoch.values[static_cast<std::size_t>(offset)] *
          primal[static_cast<std::size_t>(epoch.column_indices[static_cast<std::size_t>(offset)])];
    }
    const double difference =
        static_cast<double>(activity) - epoch.rhs[static_cast<std::size_t>(row)];
    if (epoch.senses[static_cast<std::size_t>(row)] == 'L') {
      maximum = std::max(maximum, difference);
    } else if (epoch.senses[static_cast<std::size_t>(row)] == 'G') {
      maximum = std::max(maximum, -difference);
    } else {
      maximum = std::max(maximum, std::abs(difference));
    }
  }
  return std::max(0.0, maximum);
}

double ComputeMaxReducedCostResidual(const LpEpoch& epoch, const std::vector<double>& dual,
                                     const std::vector<double>& reduced_costs) {
  std::vector<long double> recomputed(epoch.objective.begin(), epoch.objective.end());
  for (std::int32_t row = 0; row < epoch.rows; ++row) {
    for (std::int32_t offset = epoch.row_offsets[static_cast<std::size_t>(row)];
         offset < epoch.row_offsets[static_cast<std::size_t>(row) + 1]; ++offset) {
      recomputed[static_cast<std::size_t>(
          epoch.column_indices[static_cast<std::size_t>(offset)])] -=
          static_cast<long double>(epoch.values[static_cast<std::size_t>(offset)]) *
          dual[static_cast<std::size_t>(row)];
    }
  }
  double maximum = 0.0;
  for (std::int32_t column = 0; column < epoch.columns; ++column) {
    maximum = std::max(maximum,
                       std::abs(static_cast<double>(recomputed[static_cast<std::size_t>(column)]) -
                                reduced_costs[static_cast<std::size_t>(column)]));
  }
  return maximum;
}

} // namespace

LpSolution SolveWithCuOpt(const LpEpoch& epoch, const std::string& library_path) {
  epoch.Validate();
  CuOptApi api(library_path);
  if (api.get_float_size() != static_cast<std::int8_t>(sizeof(double)) ||
      api.get_int_size() != static_cast<std::int8_t>(sizeof(std::int32_t))) {
    throw std::runtime_error("cuOpt C ABI 的 float/int 宽度与 sidecar 不匹配");
  }

  std::int32_t major = 0;
  std::int32_t minor = 0;
  std::int32_t patch = 0;
  CheckStatus(api.get_version(&major, &minor, &patch), "cuOptGetVersion");

  Handle problem = nullptr;
  Handle settings = nullptr;
  Handle solution_handle = nullptr;
  try {
    CheckStatus(api.create_problem(epoch.rows, epoch.columns, kCuOptMinimize,
                                   epoch.objective_offset, epoch.objective.data(),
                                   epoch.row_offsets.data(), epoch.column_indices.data(),
                                   epoch.values.data(), epoch.senses.data(), epoch.rhs.data(),
                                   epoch.lower_bounds.data(), epoch.upper_bounds.data(),
                                   epoch.variable_types.data(), &problem),
                "cuOptCreateProblem");
    CheckStatus(api.create_settings(&settings), "cuOptCreateSolverSettings");
    CheckStatus(api.set_integer(settings, "method", kCuOptMethodPdlp), "设置 method=PDLP");
    CheckStatus(api.set_integer(settings, "pdlp_precision", kCuOptDoublePrecision),
                "设置 pdlp_precision=double");
    // 证书必须保持 Concorde 的原始行列顺序；关闭 presolve，避免对偶回映射含糊。
    CheckStatus(api.set_integer(settings, "presolve", kCuOptPresolveOff), "设置 presolve=off");
    CheckStatus(api.set_integer(settings, "pdlp_solver_mode", kCuOptPdlpMethodical),
                "设置 pdlp_solver_mode=methodical");
    CheckStatus(api.set_integer(settings, "random_seed", 0), "设置 random_seed");
    CheckStatus(api.set_integer(settings, "log_to_console", 0), "关闭 cuOpt 控制台日志");
    CheckStatus(api.set_float(settings, "absolute_primal_tolerance", 1.0e-8),
                "设置 primal tolerance");
    CheckStatus(api.set_float(settings, "absolute_dual_tolerance", 1.0e-8), "设置 dual tolerance");
    CheckStatus(api.set_float(settings, "relative_primal_tolerance", 0.0),
                "设置 relative primal tolerance");
    CheckStatus(api.set_float(settings, "relative_dual_tolerance", 0.0),
                "设置 relative dual tolerance");
    CheckStatus(api.set_float(settings, "absolute_gap_tolerance", 1.0e-8),
                "设置 absolute gap tolerance");
    CheckStatus(api.set_float(settings, "relative_gap_tolerance", 0.0),
                "设置 relative gap tolerance");
    CheckStatus(api.solve(problem, settings, &solution_handle), "cuOptSolve");

    LpSolution solution;
    solution.solver = "cuOpt-C-API";
    solution.solver_version =
        std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    CheckStatus(api.get_termination(solution_handle, &solution.termination_status),
                "cuOptGetTerminationStatus");
    solution.status = TerminationName(solution.termination_status);
    solution.primal.resize(static_cast<std::size_t>(epoch.columns));
    solution.dual.resize(static_cast<std::size_t>(epoch.rows));
    solution.reduced_costs.resize(static_cast<std::size_t>(epoch.columns));
    CheckStatus(api.get_primal(solution_handle, solution.primal.data()), "cuOptGetPrimalSolution");
    CheckStatus(api.get_dual(solution_handle, solution.dual.data()), "cuOptGetDualSolution");
    CheckStatus(api.get_reduced_costs(solution_handle, solution.reduced_costs.data()),
                "cuOptGetReducedCosts");
    CheckStatus(api.get_objective(solution_handle, &solution.objective), "cuOptGetObjectiveValue");
    CheckStatus(api.get_dual_objective(solution_handle, &solution.dual_objective),
                "cuOptGetDualObjectiveValue");
    CheckStatus(api.get_solve_time(solution_handle, &solution.solve_time_seconds),
                "cuOptGetSolveTime");

    solution.max_primal_violation = ComputeMaxPrimalViolation(epoch, solution.primal);
    solution.max_reduced_cost_residual =
        ComputeMaxReducedCostResidual(epoch, solution.dual, solution.reduced_costs);
    solution.exact_model_bound = BuildExactModelBound(epoch, solution.dual);
    solution.numerically_accepted = solution.termination_status == kCuOptOptimal &&
                                    solution.max_primal_violation <= 1.0e-7 &&
                                    solution.max_reduced_cost_residual <= 1.0e-6;
    if (!solution.numerically_accepted) {
      solution.exact_model_bound.certified = false;
      solution.exact_model_bound.reason =
          "cuOpt 状态或独立残差未通过门禁；定点值仅供诊断，不授权任何操作";
    }

    api.destroy_solution(&solution_handle);
    api.destroy_settings(&settings);
    api.destroy_problem(&problem);
    return solution;
  } catch (...) {
    if (solution_handle != nullptr)
      api.destroy_solution(&solution_handle);
    if (settings != nullptr)
      api.destroy_settings(&settings);
    if (problem != nullptr)
      api.destroy_problem(&problem);
    throw;
  }
}

} // namespace cudaee
