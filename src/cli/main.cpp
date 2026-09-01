#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/hamilton_tutte.hpp"
#include "cuda_edge_elimination/lp_epoch.hpp"
#include "cuda_edge_elimination/path_system.hpp"

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

using Arguments = std::map<std::string, std::string>;

void PrintHelp() {
  std::cout
      << "cudaee：可验证 TSP GPU 边消元研究工具\n\n"
      << "命令：\n"
      << "  gpu-eliminate --tsp FILE --edges FILE --output FILE --proof FILE\n"
      << "                [--backend auto|cpu|cuda] [--max-rounds N] [--manifest FILE]\n"
      << "  verify        --tsp FILE --edges FILE --proof FILE\n"
      << "  lp-solve      --input FILE --output FILE [--cuopt-library FILE]\n"
      << "  lp-example    --output FILE\n"
      << "  path-table    --paths 1..5 --output FILE [--backend auto|cpu|cuda]\n"
      << "  ht-prove      --tsp FILE --edges FILE --u NODE --v NODE --proof FILE\n"
      << "                [--scheduler dfs|wavefront] [--backend auto|cpu|cuda]\n"
      << "                [--reply-backend auto|cpu|cuda]\n"
      << "                [--reply-frontier-batch-states N]\n"
      << "                [--leaf-frontier-batch-states N]\n"
      << "                [--path-append-backend auto|cpu|cuda]\n"
      << "                [--propagation-backend auto|cpu|cuda] [--max-depth N] [HT budgets]\n"
      << "  ht-verify     --tsp FILE --edges FILE --proof FILE\n"
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
    if (!arguments.emplace(key.substr(2), argv[++index]).second) {
      throw std::invalid_argument("参数重复: " + key);
    }
  }
  return arguments;
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
              << " propose_ms=" << std::fixed << std::setprecision(3) << epoch.propose_ms
              << " verify_ms=" << epoch.verify_ms << '\n';
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

bool HtProveCommand(const Arguments& arguments) {
  const std::filesystem::path proof_path = CheckedOutputPath(Required(arguments, "proof"));
  const cudaee::GraphSnapshot graph =
      cudaee::GraphSnapshot::Load(Required(arguments, "tsp"), Required(arguments, "edges"));
  const cudaee::NodeEdge target{RequiredInteger<std::int32_t>(arguments, "u"),
                                RequiredInteger<std::int32_t>(arguments, "v")};

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
  root.candidate_backend = ParsePathCompatibilityBackend(Optional(arguments, "backend", "auto"));
  root.leaf_options.max_k =
      OptionalInteger<std::uint32_t>(arguments, "max-k", root.leaf_options.max_k);
  if (root.leaf_options.max_k < 3U || root.leaf_options.max_k > 5U) {
    throw std::invalid_argument("--max-k 必须位于 [3,5]");
  }
  root.leaf_options.max_deletion_sets =
      OptionalInteger<std::uint64_t>(arguments, "max-deletion-sets", 100000U);
  root.leaf_options.cost_backend = root.candidate_backend;
  root.leaf_options.cost_batch_size = OptionalInteger<std::uint32_t>(
      arguments, "cost-batch-size", root.leaf_options.cost_batch_size);
  if (root.leaf_options.cost_batch_size == 0U) {
    throw std::invalid_argument("--cost-batch-size 必须大于 0");
  }
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

  const std::string scheduler = Optional(arguments, "scheduler", "dfs");
  cudaee::HtSearchStatus search_status = cudaee::HtSearchStatus::kInvalid;
  cudaee::HtRecursiveProof proof;
  std::string propagation_backend = "none";
  std::string path_append_backend = "none";
  std::string hamilton_reply_backend = "none";
  int selected_device = -1;
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
  if (scheduler == "dfs") {
    cudaee::HtRecursiveResult result = cudaee::ProveEdgeByRecursiveHt(graph, target, options);
    search_status = result.status;
    proof = std::move(result.proof);
  } else if (scheduler == "wavefront") {
    cudaee::HtWavefrontResult result = cudaee::ProveEdgeByWavefrontHt(
        graph, target,
        {.search_options = options,
         .reply_frontier_batch_states =
             OptionalInteger<std::uint32_t>(arguments, "reply-frontier-batch-states", 256U),
         .leaf_frontier_batch_states =
             OptionalInteger<std::uint32_t>(arguments, "leaf-frontier-batch-states", 256U),
         .propagation_backend =
             ParsePathCompatibilityBackend(Optional(arguments, "propagation-backend", "auto")),
         .path_append_backend =
             ParsePathCompatibilityBackend(Optional(arguments, "path-append-backend", "auto")),
         .hamilton_reply_backend =
             ParsePathCompatibilityBackend(Optional(arguments, "reply-backend", "auto"))});
    search_status = result.status;
    proof = std::move(result.proof);
    propagation_backend = std::move(result.propagation_backend);
    path_append_backend = std::move(result.path_append_backend);
    hamilton_reply_backend = std::move(result.hamilton_reply_backend);
    selected_device = result.selected_device;
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
  } else {
    throw std::invalid_argument("--scheduler 必须是 dfs 或 wavefront");
  }

  cudaee::WriteHtRecursiveProof(proof_path, proof);
  const char* const status = search_status == cudaee::HtSearchStatus::kProven       ? "PROVEN"
                             : search_status == cudaee::HtSearchStatus::kUnresolved ? "UNRESOLVED"
                                                                                    : "INVALID";
  std::cout << "status=" << status << " scheduler=" << scheduler
            << " target=" << proof.target_edge.u << '-' << proof.target_edge.v
            << " nodes=" << proof.nodes.size() << " states=" << proof.states_expanded
            << " replies=" << proof.replies_expanded << " leaf_calls=" << proof.leaf_calls;
  if (scheduler == "wavefront") {
    std::cout << " propagation_backend=" << propagation_backend
              << " selected_device=" << selected_device << " moves=" << moves_generated
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
              << " peak_reply_frontier_batch=" << peak_reply_frontier_batch;
  }
  std::cout << " reason=" << std::quoted(proof.reason) << '\n';
  if (search_status == cudaee::HtSearchStatus::kInvalid) {
    throw std::runtime_error("递归 HT 输入或内部复核失败: " + proof.reason);
  }
  return search_status == cudaee::HtSearchStatus::kProven;
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
