#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/lp_epoch.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Arguments = std::map<std::string, std::string>;

void PrintHelp() {
  std::cout << "cudaee：可验证 TSP GPU 边消元研究工具\n\n"
            << "命令：\n"
            << "  gpu-eliminate --tsp FILE --edges FILE --output FILE --proof FILE\n"
            << "                [--backend auto|cpu|cuda] [--max-rounds N] [--manifest FILE]\n"
            << "  verify        --tsp FILE --edges FILE --proof FILE\n"
            << "  lp-solve      --input FILE --output FILE [--cuopt-library FILE]\n"
            << "  lp-example    --output FILE\n"
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
  const auto max_rounds =
      static_cast<std::uint32_t>(std::stoul(Optional(arguments, "max-rounds", "100")));
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
