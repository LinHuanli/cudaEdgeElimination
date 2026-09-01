#include "cuda_edge_elimination/elimination.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace {

struct RunMetrics {
  double algorithm_ms{};
  double snapshot_ms{};
  double propose_ms{};
  double verify_ms{};
  double commit_ms{};
  double replay_ms{};
  std::size_t edges_scanned{};
  std::size_t committed{};
  std::size_t active_edges{};
  std::uint64_t final_hash{};
  std::uint64_t static_cache_hits{};
  std::uint64_t workspace_cache_hits{};
  std::uint64_t peak_resident_bytes{};
};

std::size_t ParseRuns(const std::string_view text) {
  std::size_t value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size() || value == 0U ||
      value > 1000U) {
    throw std::invalid_argument("RUNS 必须位于 [1,1000]");
  }
  return value;
}

void CheckSameGraph(const cudaee::GraphSnapshot& lhs, const cudaee::GraphSnapshot& rhs,
                    const std::string_view description) {
  if (lhs.ContentHash() != rhs.ContentHash() || lhs.edges.size() != rhs.edges.size()) {
    throw std::runtime_error(std::string(description) + "的图哈希或边数不一致");
  }
  for (std::size_t edge_id = 0; edge_id < lhs.edges.size(); ++edge_id) {
    if (lhs.edges[edge_id].active != rhs.edges[edge_id].active) {
      throw std::runtime_error(std::string(description) + "的 active 位不一致");
    }
  }
}

RunMetrics RunOnce(const cudaee::GraphSnapshot& initial, const cudaee::Backend backend,
                   cudaee::GraphSnapshot* const final_graph) {
  cudaee::GraphSnapshot graph = initial;
  const auto begin = std::chrono::steady_clock::now();
  const cudaee::EliminationResult result = cudaee::RunJvElimination(&graph, backend, 100U);
  const auto end = std::chrono::steady_clock::now();

  cudaee::GraphSnapshot replay = initial;
  const auto replay_begin = std::chrono::steady_clock::now();
  const cudaee::EliminationResult replayed = cudaee::ReplayProof(&replay, result);
  const auto replay_end = std::chrono::steady_clock::now();
  if (replayed.final_hash != result.final_hash) {
    throw std::runtime_error("JV benchmark proof 重放哈希不一致");
  }
  CheckSameGraph(graph, replay, "JV benchmark proof 重放");

  RunMetrics metrics;
  metrics.algorithm_ms = std::chrono::duration<double, std::milli>(end - begin).count();
  metrics.replay_ms = std::chrono::duration<double, std::milli>(replay_end - replay_begin).count();
  metrics.committed = result.proof.size();
  metrics.active_edges = graph.ActiveEdgeCount();
  metrics.final_hash = result.final_hash;
  for (const cudaee::EpochMetrics& epoch : result.epochs) {
    metrics.edges_scanned += epoch.edges_before;
    metrics.snapshot_ms += epoch.snapshot_ms;
    metrics.propose_ms += epoch.propose_ms;
    metrics.verify_ms += epoch.verify_ms;
    metrics.commit_ms += epoch.commit_ms;
    metrics.static_cache_hits += epoch.jv_static_cache_hit ? 1U : 0U;
    metrics.workspace_cache_hits += epoch.jv_workspace_cache_hit ? 1U : 0U;
    metrics.peak_resident_bytes = std::max(metrics.peak_resident_bytes, epoch.jv_resident_bytes);
  }
  *final_graph = std::move(graph);
  return metrics;
}

void PrintMetrics(const std::string_view backend, const std::size_t run,
                  const RunMetrics& metrics) {
  std::cout << backend << ',' << run << ',' << std::fixed << std::setprecision(6)
            << metrics.algorithm_ms << ',' << metrics.propose_ms << ',' << metrics.verify_ms << ','
            << metrics.replay_ms << ',' << metrics.edges_scanned << ',' << metrics.committed << ','
            << metrics.active_edges << ',' << cudaee::HexHash(metrics.final_hash) << ','
            << metrics.snapshot_ms << ',' << metrics.commit_ms << ',' << metrics.static_cache_hits
            << ',' << metrics.workspace_cache_hits << ',' << metrics.peak_resident_bytes << '\n';
}

} // namespace

int main(const int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) {
      throw std::invalid_argument("用法：cudaee_jv_benchmark TSP EDGES [RUNS]");
    }
    const std::size_t runs = argc == 4 ? ParseRuns(argv[3]) : 5U;
    const cudaee::GraphSnapshot initial =
        cudaee::GraphSnapshot::Load(std::filesystem::path(argv[1]), std::filesystem::path(argv[2]));
    std::string reason;
    if (!cudaee::CudaBackendAvailable(&reason)) {
      throw std::runtime_error("CUDA JV benchmark 不可用: " + reason);
    }

    // 预热不计时：完成 CUDA context 初始化，并先建立 CPU/GPU 结果等价基线。
    cudaee::GraphSnapshot cpu_reference;
    cudaee::GraphSnapshot cuda_reference;
    static_cast<void>(RunOnce(initial, cudaee::Backend::kCpu, &cpu_reference));
    static_cast<void>(RunOnce(initial, cudaee::Backend::kCuda, &cuda_reference));
    CheckSameGraph(cpu_reference, cuda_reference, "JV benchmark CPU/CUDA 预热");

    std::cout << "backend,run,algorithm_ms,propose_ms,verify_ms,replay_ms,edges_scanned,committed,"
                 "active_edges,final_hash,snapshot_ms,commit_ms,static_cache_hits,"
                 "workspace_cache_hits,peak_resident_bytes\n";
    for (std::size_t run = 1U; run <= runs; ++run) {
      cudaee::GraphSnapshot cpu_graph;
      const RunMetrics cpu = RunOnce(initial, cudaee::Backend::kCpu, &cpu_graph);
      CheckSameGraph(cpu_reference, cpu_graph, "JV benchmark CPU 重复运行");
      PrintMetrics("cpu", run, cpu);

      cudaee::GraphSnapshot cuda_graph;
      const RunMetrics cuda = RunOnce(initial, cudaee::Backend::kCuda, &cuda_graph);
      CheckSameGraph(cpu_reference, cuda_graph, "JV benchmark CPU/CUDA 计时运行");
      PrintMetrics("cuda", run, cuda);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
