#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/hamilton_tutte.hpp"
#include "cuda_edge_elimination/lp_epoch.hpp"
#include "cuda_edge_elimination/path_system.hpp"
#include "cuda_edge_elimination/tour.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Arguments = std::multimap<std::string, std::string>;

void PrintHelp() {
  std::cout
      << "cudaee：可验证 TSP GPU 边消元研究工具\n\n"
      << "命令：\n"
      << "  gpu-eliminate --tsp FILE --edges FILE --output FILE --proof FILE\n"
      << "                [--backend auto|cpu|cuda] [--max-rounds N] [--manifest FILE]\n"
      << "  verify        --tsp FILE --edges FILE --proof FILE\n"
      << "  tour-check    --tsp FILE --edges FILE --tour FILE --expected-cost COST\n"
      << "  ht-commit     --tsp FILE --edges FILE --output FILE --proof FILE\n"
      << "                --ht-proof FILE [--ht-proof FILE ...] [--manifest FILE]\n"
      << "  lp-solve      --input FILE --output FILE [--cuopt-library FILE]\n"
      << "  lp-example    --output FILE\n"
      << "  path-table    --paths 1..5 --output FILE [--backend auto|cpu|cuda]\n"
      << "  ht-prove      --tsp FILE --edges FILE --u NODE --v NODE --proof FILE\n"
      << "                [--scheduler dfs|wavefront] [--backend auto|cpu|cuda]\n"
      << "                [--leaf-backend auto|cpu|cuda]\n"
      << "                [--reply-backend auto|cpu|cuda]\n"
      << "                [--reply-frontier-batch-states N]\n"
      << "                [--leaf-frontier-batch-states N]\n"
      << "                [--cost-batch-size N] [--cuda-min-cost-cells N]\n"
      << "                [--path-append-backend auto|cpu|cuda]\n"
      << "                [--propagation-backend auto|cpu|cuda] [--propagation-blocks N]\n"
      << "                [--max-depth N] [HT budgets]\n"
      << "  ht-verify     --tsp FILE --edges FILE --proof FILE\n"
      << "  ht-scan       --tsp FILE --edges FILE --output FILE --proof FILE --report FILE\n"
      << "                --max-targets N [--target-offset N]\n"
      << "                [--target-order weight-desc|canonical]\n"
      << "                [--protected-tour FILE --expected-cost COST] [HT wavefront options]\n"
      << "  pipeline      与 gpu-eliminate 相同，可附加 --lp-epoch FILE\n"
      << "                --lp-solution FILE [--cuopt-library FILE]\n\n"
      << "所有输出必须位于源码仓库内；不支持或验证失败时不会删除边。\n";
}

Arguments ParseArguments(const int argc, char** argv, const int first) {
  Arguments arguments;
  for (int index = first; index < argc; ++index) {
    std::string key = argv[index];
    if (!key.starts_with("--")) {
      throw std::invalid_argument("无法识别的位置参数: " + key);
    }
    if (index + 1 >= argc || std::string(argv[index + 1]).starts_with("--")) {
      throw std::invalid_argument("参数缺少值: " + key);
    }
    key = key.substr(2);
    if (key != "ht-proof" && arguments.contains(key)) {
      throw std::invalid_argument("参数重复: --" + key);
    }
    arguments.emplace(std::move(key), argv[++index]);
  }
  return arguments;
}

std::vector<std::string> Repeated(const Arguments& arguments, const std::string& name) {
  std::vector<std::string> values;
  const auto [begin, end] = arguments.equal_range(name);
  for (auto iterator = begin; iterator != end; ++iterator) {
    if (iterator->second.empty()) {
      throw std::invalid_argument("参数 --" + name + " 的值不能为空");
    }
    values.push_back(iterator->second);
  }
  return values;
}

const std::string& Required(const Arguments& arguments, const std::string& name) {
  const auto iterator = arguments.find(name);
  if (iterator == arguments.end() || iterator->second.empty()) {
    throw std::invalid_argument("缺少参数 --" + name);
  }
  return iterator->second;
}

std::string Optional(const Arguments& arguments, const std::string& name,
                     const std::string& fallback = {}) {
  const auto iterator = arguments.find(name);
  return iterator == arguments.end() ? fallback : iterator->second;
}

template <typename Integer>
Integer ParseIntegerValue(const std::string& value, const std::string_view description) {
  static_assert(std::is_integral_v<Integer>);
  Integer parsed{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
  if (value.empty() || error != std::errc{} || end != value.data() + value.size()) {
    throw std::invalid_argument(std::string(description) + " 必须是范围内的十进制整数");
  }
  return parsed;
}

template <typename Integer>
Integer RequiredInteger(const Arguments& arguments, const std::string& name) {
  return ParseIntegerValue<Integer>(Required(arguments, name), "--" + name);
}

template <typename Integer>
Integer OptionalInteger(const Arguments& arguments, const std::string& name,
                        const Integer fallback) {
  const auto iterator = arguments.find(name);
  return iterator == arguments.end() ? fallback
                                     : ParseIntegerValue<Integer>(iterator->second, "--" + name);
}

bool IsWithin(const std::filesystem::path& child, const std::filesystem::path& parent) {
  auto child_iterator = child.begin();
  for (auto parent_iterator = parent.begin(); parent_iterator != parent.end();
       ++parent_iterator, ++child_iterator) {
    if (child_iterator == child.end() || *child_iterator != *parent_iterator) {
      return false;
    }
  }
  return true;
}

std::filesystem::path CheckedOutputPath(const std::string& value) {
  const std::filesystem::path repository =
      std::filesystem::weakly_canonical(std::filesystem::path(CUDAEE_SOURCE_DIR));
  std::filesystem::path output = std::filesystem::absolute(value).lexically_normal();
  std::filesystem::path parent = output.parent_path();
  if (parent.empty()) {
    parent = std::filesystem::current_path();
  }
  const std::filesystem::path existing_parent = std::filesystem::weakly_canonical(parent);
  output = existing_parent / output.filename();
  if (!IsWithin(output, repository) || output == repository) {
    throw std::invalid_argument("输出路径必须位于仓库内: " + output.string());
  }
  std::filesystem::create_directories(existing_parent);
  return output;
}

cudaee::Backend ParseBackend(const std::string& value) {
  if (value == "auto")
    return cudaee::Backend::kAuto;
  if (value == "cpu")
    return cudaee::Backend::kCpu;
  if (value == "cuda")
    return cudaee::Backend::kCuda;
  throw std::invalid_argument("--backend 必须是 auto、cpu 或 cuda");
}

cudaee::PathCompatibilityBackend ParsePathCompatibilityBackend(const std::string& value) {
  if (value == "auto")
    return cudaee::PathCompatibilityBackend::kAuto;
  if (value == "cpu")
    return cudaee::PathCompatibilityBackend::kCpu;
  if (value == "cuda")
    return cudaee::PathCompatibilityBackend::kCuda;
  throw std::invalid_argument("--backend 必须是 auto、cpu 或 cuda");
}

cudaee::HtCdMode ParseHtCdMode(const std::string& value) {
  if (value == "active-incompatible")
    return cudaee::HtCdMode::kActiveIncompatible;
  if (value == "missing-or-incompatible")
    return cudaee::HtCdMode::kMissingOrIncompatible;
  throw std::invalid_argument("--cd-mode 必须是 active-incompatible 或 missing-or-incompatible");
}

bool ParseBooleanOption(const Arguments& arguments, const std::string& name, const bool fallback) {
  const std::uint32_t value = OptionalInteger<std::uint32_t>(arguments, name, fallback ? 1U : 0U);
  if (value > 1U) {
    throw std::invalid_argument("--" + name + " 必须是 0 或 1");
  }
  return value == 1U;
}

void WriteManifest(const std::filesystem::path& path, const cudaee::GraphSnapshot& graph,
                   const cudaee::EliminationResult& result, const Arguments& arguments) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建运行清单: " + path.string());
  }
  const auto now = std::chrono::system_clock::now();
  const auto timestamp =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  output << "CUDAEE_RUN_MANIFEST_V1\n";
  output << "unix_timestamp " << timestamp << '\n';
  output << "backend " << result.backend << '\n';
  output << "distance_type " << cudaee::ToString(graph.distance_type) << '\n';
  output << "dimension " << graph.dimension << '\n';
  output << "initial_hash " << cudaee::HexHash(result.initial_hash) << '\n';
  output << "final_hash " << cudaee::HexHash(result.final_hash) << '\n';
  output << "tsp " << Required(arguments, "tsp") << '\n';
  output << "edges " << Required(arguments, "edges") << '\n';
  output << "epochs " << result.epochs.size() << '\n';
  output << "END\n";
}

void PrintEliminationSummary(const cudaee::GraphSnapshot& graph,
                             const cudaee::EliminationResult& result) {
  std::size_t committed = 0;
  for (const cudaee::EpochMetrics& epoch : result.epochs) {
    committed += epoch.committed;
    std::cout << "epoch=" << epoch.epoch << " edges_before=" << epoch.edges_before
              << " proposed=" << epoch.proposed << " verified=" << epoch.verified
              << " rejected=" << epoch.rejected << " committed=" << epoch.committed
              << " snapshot_ms=" << std::fixed << std::setprecision(3) << epoch.snapshot_ms
              << " propose_ms=" << epoch.propose_ms << " verify_ms=" << epoch.verify_ms
              << " commit_ms=" << epoch.commit_ms
              << " jv_static_cache_hit=" << (epoch.jv_static_cache_hit ? 1 : 0)
              << " jv_workspace_cache_hit=" << (epoch.jv_workspace_cache_hit ? 1 : 0)
              << " jv_resident_bytes=" << epoch.jv_resident_bytes
              << " jv_h2d_ms=" << epoch.jv_h2d_ms << " jv_kernel_ms=" << epoch.jv_kernel_ms
              << " jv_d2h_ms=" << epoch.jv_d2h_ms << '\n';
  }
  std::cout << "status=OK backend=" << result.backend << " committed=" << committed
            << " active_edges=" << graph.ActiveEdgeCount()
            << " final_hash=" << cudaee::HexHash(result.final_hash) << '\n';
}

cudaee::EliminationResult RunEliminationCommand(const Arguments& arguments) {
  const std::filesystem::path output_path = CheckedOutputPath(Required(arguments, "output"));
  const std::filesystem::path proof_path = CheckedOutputPath(Required(arguments, "proof"));
  cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));
  const auto max_rounds = OptionalInteger<std::uint32_t>(arguments, "max-rounds", 100U);
  cudaee::EliminationResult result = cudaee::RunJvElimination(
      &graph, ParseBackend(Optional(arguments, "backend", "auto")), max_rounds);
  graph.WriteActiveEdges(output_path);
  cudaee::WriteProof(proof_path, result);
  const std::string manifest = Optional(arguments, "manifest");
  if (!manifest.empty()) {
    WriteManifest(CheckedOutputPath(manifest), graph, result, arguments);
  }
  PrintEliminationSummary(graph, result);
  return result;
}

void VerifyCommand(const Arguments& arguments) {
  cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));
  const cudaee::EliminationResult proof = cudaee::ReadProof(Required(arguments, "proof"));
  const cudaee::EliminationResult replayed = cudaee::ReplayProof(&graph, proof);
  std::cout << "status=VERIFIED records=" << replayed.proof.size()
            << " active_edges=" << graph.ActiveEdgeCount()
            << " final_hash=" << cudaee::HexHash(replayed.final_hash) << '\n';
}

void TourCheckCommand(const Arguments& arguments) {
  const cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));
  const std::vector<std::int32_t> tour =
      cudaee::ReadTsplibTour(Required(arguments, "tour"), graph.dimension);
  const cudaee::ProtectedTourCheck check = cudaee::CheckProtectedTour(graph, tour);
  const std::int64_t expected_cost = RequiredInteger<std::int64_t>(arguments, "expected-cost");
  if (expected_cost < 0 || check.cost != expected_cost) {
    throw std::runtime_error("受保护 tour 成本与 expected-cost 不一致: actual=" +
                             std::to_string(check.cost));
  }
  if (check.missing_edges != 0U) {
    throw std::runtime_error("活动边集缺少受保护 tour 边: " + std::to_string(check.missing_edges));
  }
  std::cout << "status=VERIFIED dimension=" << graph.dimension << " tour_cost=" << check.cost
            << " missing_edges=" << check.missing_edges
            << " tour_hash=" << cudaee::HexHash(check.tour_hash) << '\n';
}

void HtCommitCommand(const Arguments& arguments) {
  const std::filesystem::path output_path = CheckedOutputPath(Required(arguments, "output"));
  const std::filesystem::path proof_path = CheckedOutputPath(Required(arguments, "proof"));
  const std::vector<std::string> sidecar_paths = Repeated(arguments, "ht-proof");
  if (sidecar_paths.empty()) {
    throw std::invalid_argument("ht-commit 至少需要一个 --ht-proof sidecar");
  }

  cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));
  std::vector<cudaee::HtRecursiveProof> sidecars;
  sidecars.reserve(sidecar_paths.size());
  for (const std::string& sidecar_path : sidecar_paths) {
    sidecars.push_back(cudaee::ReadHtRecursiveProof(sidecar_path));
  }

  cudaee::EliminationResult result = cudaee::CommitHtProofEpoch(&graph, sidecars);
  graph.WriteActiveEdges(output_path);
  cudaee::WriteProof(proof_path, result);
  const std::string manifest = Optional(arguments, "manifest");
  if (!manifest.empty()) {
    WriteManifest(CheckedOutputPath(manifest), graph, result, arguments);
  }
  PrintEliminationSummary(graph, result);
}

cudaee::LpSolution LpSolveCommand(const Arguments& arguments) {
  cudaee::LpEpoch epoch = cudaee::ReadLpEpoch(Required(arguments, "input"));
  const std::string library = Optional(arguments, "cuopt-library");
  cudaee::LpSolution solution = cudaee::SolveWithCuOpt(epoch, library);
  cudaee::WriteLpSolution(CheckedOutputPath(Required(arguments, "output")), epoch, solution);
  std::cout << "status=" << solution.status << " solver=" << solution.solver << '-'
            << solution.solver_version << " objective=" << std::setprecision(17)
            << solution.objective << " primal_violation=" << solution.max_primal_violation
            << " acceptance=" << (solution.numerically_accepted ? "ACCEPTED" : "REJECTED")
            << " exact_model_bound=";
  if (solution.exact_model_bound.certified) {
    std::cout << solution.exact_model_bound.numerator << '/'
              << solution.exact_model_bound.denominator;
  } else {
    std::cout << "UNAVAILABLE";
  }
  std::cout << '\n';
  return solution;
}

void LpExampleCommand(const Arguments& arguments) {
  cudaee::LpEpoch epoch;
  epoch.rows = 1;
  epoch.columns = 2;
  epoch.objective_sense = 1;
  epoch.objective_offset = 0.0;
  epoch.objective = {1.0, 2.0};
  epoch.row_offsets = {0, 2};
  epoch.column_indices = {0, 1};
  epoch.values = {1.0, 1.0};
  epoch.senses = {'G'};
  epoch.rhs = {1.0};
  epoch.lower_bounds = {0.0, 0.0};
  epoch.upper_bounds = {1.0, 1.0};
  epoch.variable_types = {'C', 'C'};
  epoch.edge_u = {-1, -1};
  epoch.edge_v = {-1, -1};
  cudaee::WriteLpEpoch(CheckedOutputPath(Required(arguments, "output")), epoch);
  std::cout << "status=OK objective=min(x+2y) constraint=x+y>=1 expected_objective=1\n";
}

void PathTableCommand(const Arguments& arguments) {
  const std::uint32_t parsed_path_count = RequiredInteger<std::uint32_t>(arguments, "paths");
  if (parsed_path_count == 0 || parsed_path_count > cudaee::kMaxGpuPathCount) {
    throw std::invalid_argument("path-table 的 --paths 必须位于 [1,5]");
  }
  const auto path_count = parsed_path_count;
  const cudaee::PathCompatibilityTable table = cudaee::BuildPathCompatibilityTable(path_count);

  std::vector<cudaee::PathCompatibilityQuery> queries;
  queries.reserve(static_cast<std::size_t>(table.outside_count) * table.inside_count);
  for (std::uint32_t inside_index = 0; inside_index < table.inside_count; ++inside_index) {
    for (std::uint32_t outside_index = 0; outside_index < table.outside_count; ++outside_index) {
      queries.push_back({outside_index, inside_index});
    }
  }
  const cudaee::PathCompatibilityBatchResult result = cudaee::EvaluatePathCompatibility(
      path_count, queries, ParsePathCompatibilityBackend(Optional(arguments, "backend", "auto")));
  if (!result.cpu_verified || result.generator_hash != table.generator_hash) {
    throw std::runtime_error("路径兼容表未通过生成器哈希与 CPU 复核门禁");
  }
  const std::size_t compatible_pairs =
      static_cast<std::size_t>(std::count_if(result.compatible.begin(), result.compatible.end(),
                                             [](const std::uint8_t value) { return value != 0; }));

  const std::filesystem::path output_path = CheckedOutputPath(Required(arguments, "output"));
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("无法创建路径兼容表清单: " + output_path.string());
  }
  output << "CUDAEE_PATH_COMPATIBILITY_MANIFEST_V1\n";
  output << "path_count " << table.path_count << '\n';
  output << "outside_count " << table.outside_count << '\n';
  output << "inside_count " << table.inside_count << '\n';
  output << "words_per_inside " << table.words_per_inside << '\n';
  output << "packed_bytes " << table.coverage.size() * sizeof(std::uint64_t) << '\n';
  output << "generator_hash " << cudaee::HexHash(table.generator_hash) << '\n';
  output << "backend " << result.backend << '\n';
  output << "selected_device " << result.selected_device << '\n';
  output << "queries " << queries.size() << '\n';
  output << "compatible_pairs " << compatible_pairs << '\n';
  output << "cpu_verified 1\nEND\n";
  if (!output) {
    throw std::runtime_error("写入路径兼容表清单失败: " + output_path.string());
  }
  std::cout << "status=OK paths=" << path_count << " backend=" << result.backend
            << " table_hash=" << cudaee::HexHash(table.generator_hash)
            << " packed_bytes=" << table.coverage.size() * sizeof(std::uint64_t)
            << " queries=" << queries.size() << " compatible_pairs=" << compatible_pairs
            << " cpu_verified=1\n";
}

cudaee::HtRecursiveOptions ParseHtRecursiveOptions(const Arguments& arguments) {
  cudaee::HtRecursiveOptions options;
  cudaee::HtShallowOptions& root = options.root_options;
  root.max_neighborhood =
      OptionalInteger<std::uint32_t>(arguments, "max-neighborhood", root.max_neighborhood);
  root.max_cd_candidates =
      OptionalInteger<std::uint32_t>(arguments, "max-cd-candidates", root.max_cd_candidates);
  root.max_candidate_degree =
      OptionalInteger<std::uint32_t>(arguments, "max-candidate-degree", root.max_candidate_degree);
  root.max_reply_combinations =
      OptionalInteger<std::uint64_t>(arguments, "max-root-replies", root.max_reply_combinations);
  root.cd_mode = ParseHtCdMode(Optional(arguments, "cd-mode", "active-incompatible"));
  const std::string candidate_backend = Optional(arguments, "backend", "auto");
  root.candidate_backend = ParsePathCompatibilityBackend(candidate_backend);
  root.leaf_options.max_k =
      OptionalInteger<std::uint32_t>(arguments, "max-k", root.leaf_options.max_k);
  if (root.leaf_options.max_k < 3U || root.leaf_options.max_k > 5U) {
    throw std::invalid_argument("--max-k 必须位于 [3,5]");
  }
  root.leaf_options.max_deletion_sets =
      OptionalInteger<std::uint64_t>(arguments, "max-deletion-sets", 100000U);
  // 默认继承旧的 --backend 语义；显式 leaf 后端允许只把高算术强度 cost matrix 放到 GPU。
  root.leaf_options.cost_backend =
      ParsePathCompatibilityBackend(Optional(arguments, "leaf-backend", candidate_backend));
  root.leaf_options.cost_batch_size = OptionalInteger<std::uint32_t>(
      arguments, "cost-batch-size", root.leaf_options.cost_batch_size);
  if (root.leaf_options.cost_batch_size == 0U) {
    throw std::invalid_argument("--cost-batch-size 必须大于 0");
  }
  root.leaf_options.cuda_min_cost_cells = OptionalInteger<std::uint64_t>(
      arguments, "cuda-min-cost-cells", root.leaf_options.cuda_min_cost_cells);
  root.leaf_options.exact_fallback_max_blocks =
      OptionalInteger<std::uint32_t>(arguments, "exact-blocks", 0U);
  if (root.leaf_options.exact_fallback_max_blocks > 18U) {
    throw std::invalid_argument("--exact-blocks 不得超过 18");
  }

  options.max_depth = OptionalInteger<std::uint32_t>(arguments, "max-depth", options.max_depth);
  options.max_states = OptionalInteger<std::uint64_t>(arguments, "max-states", options.max_states);
  options.max_total_replies =
      OptionalInteger<std::uint64_t>(arguments, "max-total-replies", options.max_total_replies);
  options.max_replies_per_move = OptionalInteger<std::uint64_t>(arguments, "max-replies-per-move",
                                                                options.max_replies_per_move);
  options.max_point_candidates = OptionalInteger<std::uint32_t>(arguments, "max-point-candidates",
                                                                options.max_point_candidates);
  options.max_end_candidates =
      OptionalInteger<std::uint32_t>(arguments, "max-end-candidates", options.max_end_candidates);
  options.enable_point_moves = ParseBooleanOption(arguments, "enable-point", true);
  options.enable_end_moves = ParseBooleanOption(arguments, "enable-end", true);
  return options;
}

cudaee::HtWavefrontOptions
ParseHtWavefrontOptions(const Arguments& arguments,
                        const cudaee::HtRecursiveOptions& search_options) {
  return {.search_options = search_options,
          .reply_frontier_batch_states =
              OptionalInteger<std::uint32_t>(arguments, "reply-frontier-batch-states", 256U),
          .leaf_frontier_batch_states =
              OptionalInteger<std::uint32_t>(arguments, "leaf-frontier-batch-states", 256U),
          .propagation_backend =
              ParsePathCompatibilityBackend(Optional(arguments, "propagation-backend", "auto")),
          .propagation_blocks = OptionalInteger<std::uint32_t>(arguments, "propagation-blocks", 0U),
          .path_append_backend =
              ParsePathCompatibilityBackend(Optional(arguments, "path-append-backend", "auto")),
          .hamilton_reply_backend =
              ParsePathCompatibilityBackend(Optional(arguments, "reply-backend", "auto"))};
}

cudaee::HtTargetOrder ParseHtTargetOrder(const std::string& value) {
  if (value == "canonical") {
    return cudaee::HtTargetOrder::kCanonical;
  }
  if (value == "weight-desc") {
    return cudaee::HtTargetOrder::kWeightDescending;
  }
  throw std::invalid_argument("--target-order 必须是 weight-desc 或 canonical");
}

const char* HtTargetOrderName(const cudaee::HtTargetOrder order) {
  return order == cudaee::HtTargetOrder::kCanonical ? "canonical" : "weight-desc";
}

const char* HtSearchStatusName(const cudaee::HtSearchStatus status) {
  switch (status) {
  case cudaee::HtSearchStatus::kProven:
    return "PROVEN";
  case cudaee::HtSearchStatus::kUnresolved:
    return "UNRESOLVED";
  case cudaee::HtSearchStatus::kInvalid:
    return "INVALID";
  }
  return "INVALID";
}

bool HtProveCommand(const Arguments& arguments) {
  const std::filesystem::path proof_path = CheckedOutputPath(Required(arguments, "proof"));
  const cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));
  const cudaee::NodeEdge target{RequiredInteger<std::int32_t>(arguments, "u"),
                                RequiredInteger<std::int32_t>(arguments, "v")};
  const cudaee::HtRecursiveOptions options = ParseHtRecursiveOptions(arguments);

  const std::string scheduler = Optional(arguments, "scheduler", "dfs");
  cudaee::HtSearchStatus search_status = cudaee::HtSearchStatus::kInvalid;
  cudaee::HtRecursiveProof proof;
  std::string propagation_backend = "none";
  std::string path_append_backend = "none";
  std::string hamilton_reply_backend = "none";
  int selected_device = -1;
  std::uint32_t propagation_blocks = 0;
  bool propagation_cooperative = false;
  int path_append_selected_device = -1;
  int hamilton_reply_selected_device = -1;
  bool path_append_cpu_verified = false;
  bool path_append_device_children_verified = false;
  bool hamilton_reply_cpu_verified = false;
  std::uint64_t path_append_batches = 0;
  std::uint64_t path_append_tasks = 0;
  std::uint64_t path_append_child_edges = 0;
  std::string leaf_cost_backend = "none";
  int leaf_cost_selected_device = -1;
  bool leaf_cpu_verified = false;
  std::uint64_t leaf_frontier_batches = 0;
  std::uint64_t leaf_frontier_states = 0;
  std::uint64_t leaf_bucket_count = 0;
  std::uint64_t peak_leaf_frontier_batch = 0;
  std::uint64_t leaf_cost_batches = 0;
  std::uint64_t leaf_cost_tasks = 0;
  std::uint64_t leaf_cost_cells = 0;
  std::uint64_t leaf_scalar_searches = 0;
  std::uint64_t leaf_cuda_cost_batches = 0;
  std::uint64_t leaf_snapshot_cache_hits = 0;
  std::uint64_t leaf_template_cache_hits = 0;
  std::uint64_t leaf_workspace_cache_hits = 0;
  std::uint64_t peak_leaf_device_cache_bytes = 0;
  std::uint64_t leaf_cpu_long_tail_batches = 0;
  std::uint64_t leaf_cpu_long_tail_tasks = 0;
  std::uint64_t leaf_cpu_long_tail_cells = 0;
  std::uint64_t hamilton_reply_batches = 0;
  std::uint64_t hamilton_reply_centers = 0;
  std::uint64_t hamilton_replies_generated = 0;
  std::string end_reply_backend = "none";
  int end_reply_selected_device = -1;
  bool end_reply_cpu_verified = false;
  std::uint64_t end_reply_batches = 0;
  std::uint64_t end_reply_tasks = 0;
  std::uint64_t end_replies_generated = 0;
  std::uint64_t reply_frontier_batches = 0;
  std::uint64_t reply_frontier_states = 0;
  std::uint64_t peak_reply_frontier_batch = 0;
  std::uint64_t moves_generated = 0;
  std::uint64_t peak_frontier = 0;
  double candidate_ms = 0.0;
  double work_graph_ms = 0.0;
  double leaf_ms = 0.0;
  double path_append_ms = 0.0;
  double hamilton_reply_ms = 0.0;
  double end_reply_ms = 0.0;
  double propagation_ms = 0.0;
  double proof_extract_ms = 0.0;
  double proof_verify_ms = 0.0;
  if (scheduler == "dfs") {
    cudaee::HtRecursiveResult result = cudaee::ProveEdgeByRecursiveHt(graph, target, options);
    search_status = result.status;
    proof = std::move(result.proof);
  } else if (scheduler == "wavefront") {
    cudaee::HtWavefrontResult result =
        cudaee::ProveEdgeByWavefrontHt(graph, target, ParseHtWavefrontOptions(arguments, options));
    search_status = result.status;
    proof = std::move(result.proof);
    propagation_backend = std::move(result.propagation_backend);
    path_append_backend = std::move(result.path_append_backend);
    hamilton_reply_backend = std::move(result.hamilton_reply_backend);
    selected_device = result.selected_device;
    propagation_blocks = result.propagation_blocks;
    propagation_cooperative = result.propagation_cooperative;
    path_append_selected_device = result.path_append_selected_device;
    hamilton_reply_selected_device = result.hamilton_reply_selected_device;
    path_append_cpu_verified = result.path_append_cpu_verified;
    path_append_device_children_verified = result.path_append_device_children_verified;
    hamilton_reply_cpu_verified = result.hamilton_reply_cpu_verified;
    path_append_batches = result.path_append_batches;
    path_append_tasks = result.path_append_tasks;
    path_append_child_edges = result.path_append_child_edges;
    leaf_cost_backend = std::move(result.leaf_cost_backend);
    leaf_cost_selected_device = result.leaf_cost_selected_device;
    leaf_cpu_verified = result.leaf_cpu_verified;
    leaf_frontier_batches = result.leaf_frontier_batches;
    leaf_frontier_states = result.leaf_frontier_states;
    leaf_bucket_count = result.leaf_bucket_count;
    peak_leaf_frontier_batch = result.peak_leaf_frontier_batch;
    leaf_cost_batches = result.leaf_cost_batches;
    leaf_cost_tasks = result.leaf_cost_tasks;
    leaf_cost_cells = result.leaf_cost_cells;
    leaf_scalar_searches = result.leaf_scalar_searches;
    leaf_cuda_cost_batches = result.leaf_cuda_cost_batches;
    leaf_snapshot_cache_hits = result.leaf_snapshot_cache_hits;
    leaf_template_cache_hits = result.leaf_template_cache_hits;
    leaf_workspace_cache_hits = result.leaf_workspace_cache_hits;
    peak_leaf_device_cache_bytes = result.peak_leaf_device_cache_bytes;
    leaf_cpu_long_tail_batches = result.leaf_cpu_long_tail_batches;
    leaf_cpu_long_tail_tasks = result.leaf_cpu_long_tail_tasks;
    leaf_cpu_long_tail_cells = result.leaf_cpu_long_tail_cells;
    hamilton_reply_batches = result.hamilton_reply_batches;
    hamilton_reply_centers = result.hamilton_reply_centers;
    hamilton_replies_generated = result.hamilton_replies_generated;
    end_reply_backend = std::move(result.end_reply_backend);
    end_reply_selected_device = result.end_reply_selected_device;
    end_reply_cpu_verified = result.end_reply_cpu_verified;
    end_reply_batches = result.end_reply_batches;
    end_reply_tasks = result.end_reply_tasks;
    end_replies_generated = result.end_replies_generated;
    reply_frontier_batches = result.reply_frontier_batches;
    reply_frontier_states = result.reply_frontier_states;
    peak_reply_frontier_batch = result.peak_reply_frontier_batch;
    moves_generated = result.moves_generated;
    peak_frontier = result.peak_frontier;
    candidate_ms = result.candidate_ms;
    work_graph_ms = result.work_graph_ms;
    leaf_ms = result.leaf_ms;
    path_append_ms = result.path_append_ms;
    hamilton_reply_ms = result.hamilton_reply_ms;
    end_reply_ms = result.end_reply_ms;
    propagation_ms = result.propagation_ms;
    proof_extract_ms = result.proof_extract_ms;
    proof_verify_ms = result.proof_verify_ms;
  } else {
    throw std::invalid_argument("--scheduler 必须是 dfs 或 wavefront");
  }

  cudaee::WriteHtRecursiveProof(proof_path, proof);
  std::cout << "status=" << HtSearchStatusName(search_status) << " scheduler=" << scheduler
            << " target=" << proof.target_edge.u << '-' << proof.target_edge.v
            << " nodes=" << proof.nodes.size() << " states=" << proof.states_expanded
            << " replies=" << proof.replies_expanded << " leaf_calls=" << proof.leaf_calls;
  if (scheduler == "wavefront") {
    std::cout << " propagation_backend=" << propagation_backend
              << " selected_device=" << selected_device << " moves=" << moves_generated
              << " propagation_blocks=" << propagation_blocks
              << " propagation_cooperative=" << (propagation_cooperative ? 1 : 0)
              << " peak_frontier=" << peak_frontier
              << " path_append_backend=" << path_append_backend
              << " path_append_device=" << path_append_selected_device
              << " path_append_batches=" << path_append_batches
              << " path_append_tasks=" << path_append_tasks
              << " path_append_child_edges=" << path_append_child_edges
              << " path_append_cpu_verified=" << (path_append_cpu_verified ? 1 : 0)
              << " path_append_device_children_verified="
              << (path_append_device_children_verified ? 1 : 0)
              << " leaf_cost_backend=" << leaf_cost_backend
              << " leaf_cost_device=" << leaf_cost_selected_device
              << " leaf_cpu_verified=" << (leaf_cpu_verified ? 1 : 0)
              << " leaf_frontier_batches=" << leaf_frontier_batches
              << " leaf_frontier_states=" << leaf_frontier_states
              << " leaf_bucket_count=" << leaf_bucket_count
              << " peak_leaf_frontier_batch=" << peak_leaf_frontier_batch
              << " leaf_cost_batches=" << leaf_cost_batches
              << " leaf_cost_tasks=" << leaf_cost_tasks << " leaf_cost_cells=" << leaf_cost_cells
              << " leaf_scalar_searches=" << leaf_scalar_searches
              << " leaf_cuda_cost_batches=" << leaf_cuda_cost_batches
              << " leaf_snapshot_cache_hits=" << leaf_snapshot_cache_hits
              << " leaf_template_cache_hits=" << leaf_template_cache_hits
              << " leaf_workspace_cache_hits=" << leaf_workspace_cache_hits
              << " peak_leaf_device_cache_bytes=" << peak_leaf_device_cache_bytes
              << " leaf_cpu_long_tail_batches=" << leaf_cpu_long_tail_batches
              << " leaf_cpu_long_tail_tasks=" << leaf_cpu_long_tail_tasks
              << " leaf_cpu_long_tail_cells=" << leaf_cpu_long_tail_cells
              << " reply_backend=" << hamilton_reply_backend
              << " reply_device=" << hamilton_reply_selected_device
              << " reply_batches=" << hamilton_reply_batches
              << " reply_centers=" << hamilton_reply_centers
              << " replies_generated=" << hamilton_replies_generated
              << " reply_cpu_verified=" << (hamilton_reply_cpu_verified ? 1 : 0)
              << " end_reply_backend=" << end_reply_backend
              << " end_reply_device=" << end_reply_selected_device
              << " end_reply_batches=" << end_reply_batches
              << " end_reply_tasks=" << end_reply_tasks
              << " end_replies_generated=" << end_replies_generated
              << " end_reply_cpu_verified=" << (end_reply_cpu_verified ? 1 : 0)
              << " reply_frontier_batches=" << reply_frontier_batches
              << " reply_frontier_states=" << reply_frontier_states
              << " peak_reply_frontier_batch=" << peak_reply_frontier_batch
              << " candidate_ms=" << candidate_ms << " work_graph_ms=" << work_graph_ms
              << " leaf_ms=" << leaf_ms << " path_append_ms=" << path_append_ms
              << " hamilton_reply_ms=" << hamilton_reply_ms << " end_reply_ms=" << end_reply_ms
              << " propagation_ms=" << propagation_ms << " proof_extract_ms=" << proof_extract_ms
              << " proof_verify_ms=" << proof_verify_ms;
  }
  std::cout << " reason=" << std::quoted(proof.reason) << '\n';
  if (search_status == cudaee::HtSearchStatus::kInvalid) {
    throw std::runtime_error("递归 HT 输入或内部复核失败: " + proof.reason);
  }
  return search_status == cudaee::HtSearchStatus::kProven;
}

std::string SingleLine(std::string value) {
  std::replace_if(
      value.begin(), value.end(),
      [](const char character) {
        return character == '\n' || character == '\r' || character == '\t';
      },
      ' ');
  return value;
}

void WriteHtScanReport(const std::filesystem::path& path, const cudaee::HtScanResult& scan,
                       const cudaee::HtScanOptions& options,
                       const cudaee::ProtectedTourCheck* const protected_tour) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 HT scan 报告: " + path.string());
  }
  output << "CUDAEE_HT_SCAN_REPORT_V2\n";
  output << "initial_hash " << cudaee::HexHash(scan.elimination.initial_hash) << '\n';
  output << "final_hash " << cudaee::HexHash(scan.elimination.final_hash) << '\n';
  output << "target_order " << HtTargetOrderName(options.target_order) << '\n';
  output << "eligible_targets " << scan.eligible_targets << '\n';
  output << "target_offset " << scan.target_offset << '\n';
  output << "target_end_offset " << scan.target_offset + scan.attempts.size() << '\n';
  output << "max_targets " << options.max_targets << '\n';
  output << "attempted_targets " << scan.attempts.size() << '\n';
  output << "proven_targets " << scan.proven_targets << '\n';
  output << "unresolved_targets " << scan.unresolved_targets << '\n';
  output << "committed_targets " << scan.elimination.proof.size() << '\n';
  output << "states_expanded " << scan.states_expanded << '\n';
  output << "replies_expanded " << scan.replies_expanded << '\n';
  output << "leaf_calls " << scan.leaf_calls << '\n';
  output << "moves_generated " << scan.moves_generated << '\n';
  output << "leaf_cost_cells " << scan.leaf_cost_cells << '\n';
  output << "leaf_cuda_cost_batches " << scan.leaf_cuda_cost_batches << '\n';
  output << "leaf_cpu_long_tail_cells " << scan.leaf_cpu_long_tail_cells << '\n';
  output << "peak_leaf_device_cache_bytes " << scan.peak_leaf_device_cache_bytes << '\n';
  output << "protected_tour_checked " << (protected_tour != nullptr ? 1 : 0) << '\n';
  if (protected_tour != nullptr) {
    output << "protected_tour_cost " << protected_tour->cost << '\n';
    output << "protected_tour_hash " << cudaee::HexHash(protected_tour->tour_hash) << '\n';
  }
  output << std::fixed << std::setprecision(6);
  output << "target_selection_ms " << scan.target_selection_ms << '\n';
  output << "candidate_ms " << scan.candidate_ms << '\n';
  output << "work_graph_ms " << scan.work_graph_ms << '\n';
  output << "leaf_ms " << scan.leaf_ms << '\n';
  output << "path_append_ms " << scan.path_append_ms << '\n';
  output << "hamilton_reply_ms " << scan.hamilton_reply_ms << '\n';
  output << "end_reply_ms " << scan.end_reply_ms << '\n';
  output << "propagation_ms " << scan.propagation_ms << '\n';
  output << "proof_extract_ms " << scan.proof_extract_ms << '\n';
  output << "proof_verify_ms " << scan.proof_verify_ms << '\n';
  output << "immediate_verify_ms " << scan.immediate_verify_ms << '\n';
  output << "commit_ms " << scan.commit_ms << '\n';
  output << "search_ms " << scan.search_ms << '\n';
  output << "total_ms " << scan.total_ms << '\n';
  output << "attempt_fields index edge_id u v status states replies leaf_calls moves "
            "peak_frontier propagation_backend device blocks cooperative propagation_verified "
            "leaf_backend leaf_device leaf_verified leaf_cells leaf_cuda_batches "
            "leaf_cpu_long_tail_cells peak_leaf_cache_bytes path_append_tasks hamilton_replies "
            "end_replies candidate_ms work_graph_ms leaf_ms path_append_ms hamilton_reply_ms "
            "end_reply_ms propagation_ms proof_extract_ms proof_verify_ms immediate_verify_ms "
            "search_ms reason\n";
  for (std::size_t index = 0U; index < scan.attempts.size(); ++index) {
    const cudaee::HtScanAttempt& attempt = scan.attempts[index];
    output << "attempt " << index << ' ' << attempt.edge_id << ' ' << attempt.target_edge.u << ' '
           << attempt.target_edge.v << ' ' << HtSearchStatusName(attempt.status) << ' '
           << attempt.states_expanded << ' ' << attempt.replies_expanded << ' '
           << attempt.leaf_calls << ' ' << attempt.moves_generated << ' ' << attempt.peak_frontier
           << ' ' << attempt.propagation_backend << ' ' << attempt.selected_device << ' '
           << attempt.propagation_blocks << ' ' << (attempt.propagation_cooperative ? 1 : 0) << ' '
           << (attempt.propagation_cpu_verified ? 1 : 0) << ' ' << attempt.leaf_cost_backend << ' '
           << attempt.leaf_cost_selected_device << ' ' << (attempt.leaf_cpu_verified ? 1 : 0) << ' '
           << attempt.leaf_cost_cells << ' ' << attempt.leaf_cuda_cost_batches << ' '
           << attempt.leaf_cpu_long_tail_cells << ' ' << attempt.peak_leaf_device_cache_bytes << ' '
           << attempt.path_append_tasks << ' ' << attempt.hamilton_replies_generated << ' '
           << attempt.end_replies_generated << ' ' << attempt.candidate_ms << ' '
           << attempt.work_graph_ms << ' ' << attempt.leaf_ms << ' ' << attempt.path_append_ms
           << ' ' << attempt.hamilton_reply_ms << ' ' << attempt.end_reply_ms << ' '
           << attempt.propagation_ms << ' ' << attempt.proof_extract_ms << ' '
           << attempt.proof_verify_ms << ' ' << attempt.immediate_verify_ms << ' '
           << attempt.search_ms << ' ' << std::quoted(SingleLine(attempt.reason)) << '\n';
  }
  output << "END\n";
  if (!output) {
    throw std::runtime_error("写入 HT scan 报告失败: " + path.string());
  }
}

void HtScanCommand(const Arguments& arguments) {
  const std::filesystem::path output_path = CheckedOutputPath(Required(arguments, "output"));
  const std::filesystem::path proof_path = CheckedOutputPath(Required(arguments, "proof"));
  const std::filesystem::path report_path = CheckedOutputPath(Required(arguments, "report"));
  cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));

  cudaee::HtScanOptions options;
  options.wavefront_options =
      ParseHtWavefrontOptions(arguments, ParseHtRecursiveOptions(arguments));
  options.target_offset = OptionalInteger<std::uint64_t>(arguments, "target-offset", 0U);
  options.max_targets = RequiredInteger<std::uint64_t>(arguments, "max-targets");
  options.target_order = ParseHtTargetOrder(Optional(arguments, "target-order", "weight-desc"));
  if (arguments.contains("scheduler") && Optional(arguments, "scheduler") != "wavefront") {
    throw std::invalid_argument("ht-scan 只支持 wavefront scheduler");
  }

  const std::string protected_tour_path = Optional(arguments, "protected-tour");
  const bool has_protected_tour = !protected_tour_path.empty();
  const bool has_expected_cost = arguments.contains("expected-cost");
  if (has_protected_tour != has_expected_cost) {
    throw std::invalid_argument("ht-scan 的 --protected-tour 与 --expected-cost 必须同时提供");
  }
  std::vector<std::int32_t> protected_tour_nodes;
  cudaee::ProtectedTourCheck protected_tour_check;
  const cudaee::ProtectedTourCheck* protected_tour_report = nullptr;
  std::int64_t protected_tour_cost = -1;
  if (has_protected_tour) {
    protected_tour_cost = RequiredInteger<std::int64_t>(arguments, "expected-cost");
    if (protected_tour_cost < 0) {
      throw std::invalid_argument("ht-scan 的 --expected-cost 不得为负数");
    }
    protected_tour_nodes = cudaee::ReadTsplibTour(protected_tour_path, graph.dimension);
    protected_tour_check = cudaee::CheckProtectedTour(graph, protected_tour_nodes);
    if (protected_tour_check.cost != protected_tour_cost ||
        protected_tour_check.missing_edges != 0U) {
      throw std::runtime_error("HT scan 初始图未通过受保护 tour 门禁");
    }
    protected_tour_report = &protected_tour_check;
  }

  const cudaee::HtScanResult scan = cudaee::RunHtScanEpoch(&graph, options);
  if (!protected_tour_nodes.empty()) {
    const cudaee::ProtectedTourCheck final_check =
        cudaee::CheckProtectedTour(graph, protected_tour_nodes);
    if (final_check.cost != protected_tour_cost || final_check.missing_edges != 0U ||
        final_check.tour_hash != protected_tour_check.tour_hash) {
      throw std::runtime_error("HT scan 最终图未通过受保护 tour 门禁；未写出结果");
    }
  }
  graph.WriteActiveEdges(output_path);
  cudaee::WriteProof(proof_path, scan.elimination);
  WriteHtScanReport(report_path, scan, options, protected_tour_report);
  const std::string manifest = Optional(arguments, "manifest");
  if (!manifest.empty()) {
    WriteManifest(CheckedOutputPath(manifest), graph, scan.elimination, arguments);
  }

  for (std::size_t index = 0U; index < scan.attempts.size(); ++index) {
    const cudaee::HtScanAttempt& attempt = scan.attempts[index];
    std::cout << "target_index=" << scan.target_offset + index << " edge_id=" << attempt.edge_id
              << " target=" << attempt.target_edge.u << '-' << attempt.target_edge.v
              << " status=" << HtSearchStatusName(attempt.status)
              << " states=" << attempt.states_expanded << " replies=" << attempt.replies_expanded
              << " leaf_calls=" << attempt.leaf_calls
              << " propagation_backend=" << attempt.propagation_backend
              << " selected_device=" << attempt.selected_device
              << " leaf_cost_backend=" << attempt.leaf_cost_backend
              << " leaf_cost_cells=" << attempt.leaf_cost_cells << " work_graph_ms=" << std::fixed
              << std::setprecision(3) << attempt.work_graph_ms << " leaf_ms=" << attempt.leaf_ms
              << " propagation_ms=" << attempt.propagation_ms
              << " proof_verify_ms=" << attempt.proof_verify_ms
              << " immediate_verify_ms=" << attempt.immediate_verify_ms
              << " search_ms=" << attempt.search_ms
              << " reason=" << std::quoted(SingleLine(attempt.reason)) << '\n';
  }
  std::cout << "scan_status=OK target_order=" << HtTargetOrderName(options.target_order)
            << " eligible=" << scan.eligible_targets << " attempted=" << scan.attempts.size()
            << " proven=" << scan.proven_targets << " unresolved=" << scan.unresolved_targets
            << " committed=" << scan.elimination.proof.size() << " search_ms=" << scan.search_ms
            << " work_graph_ms=" << scan.work_graph_ms << " leaf_ms=" << scan.leaf_ms
            << " propagation_ms=" << scan.propagation_ms << " commit_ms=" << scan.commit_ms
            << " total_ms=" << scan.total_ms
            << " protected_tour_checked=" << (protected_tour_report != nullptr ? 1 : 0) << '\n';
  PrintEliminationSummary(graph, scan.elimination);
}

void HtVerifyCommand(const Arguments& arguments) {
  const cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));
  const cudaee::HtRecursiveProof proof = cudaee::ReadHtRecursiveProof(Required(arguments, "proof"));
  std::string reason;
  if (!cudaee::VerifyHtRecursiveProof(graph, proof, &reason)) {
    throw std::runtime_error("递归 HT proof 复核失败: " + reason);
  }
  std::cout << "status=VERIFIED target=" << proof.target_edge.u << '-' << proof.target_edge.v
            << " nodes=" << proof.nodes.size() << " replies=" << proof.replies_expanded << '\n';
}

} // namespace

int main(const int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
      PrintHelp();
      return argc < 2 ? 1 : 0;
    }
    const std::string command = argv[1];
    const Arguments arguments = ParseArguments(argc, argv, 2);
    if (command == "gpu-eliminate") {
      RunEliminationCommand(arguments);
    } else if (command == "verify") {
      VerifyCommand(arguments);
    } else if (command == "tour-check") {
      TourCheckCommand(arguments);
    } else if (command == "ht-commit") {
      HtCommitCommand(arguments);
    } else if (command == "lp-solve") {
      LpSolveCommand(arguments);
    } else if (command == "lp-example") {
      LpExampleCommand(arguments);
    } else if (command == "path-table") {
      PathTableCommand(arguments);
    } else if (command == "ht-prove") {
      if (!HtProveCommand(arguments)) {
        return 3;
      }
    } else if (command == "ht-scan") {
      HtScanCommand(arguments);
    } else if (command == "ht-verify") {
      HtVerifyCommand(arguments);
    } else if (command == "pipeline") {
      const std::string lp_epoch = Optional(arguments, "lp-epoch");
      const std::string lp_solution = Optional(arguments, "lp-solution");
      if (!lp_epoch.empty() || !lp_solution.empty()) {
        if (lp_epoch.empty() || lp_solution.empty()) {
          throw std::invalid_argument("pipeline 的 --lp-epoch 与 --lp-solution 必须同时提供");
        }
        Arguments lp_arguments{{"input", lp_epoch}, {"output", lp_solution}};
        const std::string library = Optional(arguments, "cuopt-library");
        if (!library.empty())
          lp_arguments.emplace("cuopt-library", library);
        LpSolveCommand(lp_arguments);
      }
      RunEliminationCommand(arguments);
    } else {
      throw std::invalid_argument("未知命令: " + command);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "status=ERROR message=" << error.what() << '\n';
    return 2;
  }
}
