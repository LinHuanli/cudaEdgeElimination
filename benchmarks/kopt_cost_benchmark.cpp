#include "cuda_edge_elimination/local_search.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

cudaee::GraphSnapshot MakeGraph(const std::int32_t dimension) {
  cudaee::GraphSnapshot graph;
  graph.dimension = dimension;
  graph.distance_type = cudaee::DistanceType::kEuc2D;
  graph.integer_coordinates = true;
  graph.integer_distance_safe = true;
  graph.points.reserve(static_cast<std::size_t>(dimension));
  for (std::int32_t node = 0; node < dimension; ++node) {
    const std::int64_t x = (static_cast<std::int64_t>(node) * 7919) % 100003;
    const std::int64_t y = (static_cast<std::int64_t>(node) * node * 17 + 23) % 100019;
    graph.points.push_back({static_cast<double>(x), static_cast<double>(y), x, y});
  }
  return graph;
}

std::vector<cudaee::KOptCostTask> MakeTasks(const cudaee::GraphSnapshot& graph,
                                            const std::uint32_t k, const std::size_t task_count) {
  std::vector<cudaee::KOptCostTask> tasks(task_count);
  const auto dimension = static_cast<std::uint32_t>(graph.dimension);
  for (std::size_t task_index = 0U; task_index < task_count; ++task_index) {
    cudaee::KOptCostTask& task = tasks[task_index];
    const std::uint32_t offset = static_cast<std::uint32_t>((task_index * 37U) % dimension);
    for (std::uint32_t port = 0U; port < 2U * k; ++port) {
      task.port_nodes[port] = static_cast<std::int32_t>((offset + port) % dimension);
    }
    for (std::uint32_t edge = 0U; edge < k; ++edge) {
      const std::size_t port = std::size_t{2} * edge;
      task.deleted_cost += graph.Distance(task.port_nodes[port], task.port_nodes[port + 1U]);
    }
  }
  return tasks;
}

double MedianMicroseconds(const cudaee::GraphSnapshot& graph, const std::uint32_t k,
                          const std::vector<cudaee::KOptCostTask>& tasks,
                          const cudaee::PathCompatibilityBackend backend,
                          const std::size_t repetitions,
                          std::vector<std::int64_t>* const last_costs) {
  std::vector<double> samples;
  samples.reserve(repetitions);
  // 预热同时把 CUDA snapshot/template/workspace 扩到本行规模；报告稳态同步调用耗时。
  *last_costs = cudaee::EvaluateKOptTemplateCosts(graph, k, tasks, backend).added_costs;
  for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
    const auto begin = std::chrono::steady_clock::now();
    *last_costs = cudaee::EvaluateKOptTemplateCosts(graph, k, tasks, backend).added_costs;
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2U];
}

} // namespace

int main() {
  try {
    std::string reason;
    if (!cudaee::detail::KOptCostCudaAvailable(&reason)) {
      throw std::runtime_error("CUDA k-opt benchmark 不可用: " + reason);
    }
    constexpr std::int32_t kDimension = 4096;
    const cudaee::GraphSnapshot graph = MakeGraph(kDimension);
    const std::vector<std::size_t> task_counts = {1U, 4U, 16U, 64U, 256U, 1024U, 4096U, 16384U};
    cudaee::detail::ClearKOptCostCudaCache();
    std::cout << "k,tasks,cells,cpu_us,cuda_us,cuda_speedup\n";
    for (std::uint32_t k = 3U; k <= 5U; ++k) {
      const std::size_t template_count = cudaee::BuildKOptReconnectTable(k).templates.size();
      for (const std::size_t task_count : task_counts) {
        const std::vector<cudaee::KOptCostTask> tasks = MakeTasks(graph, k, task_count);
        const std::size_t cells = task_count * template_count;
        const std::size_t repetitions = cells <= 100000U ? 7U : 3U;
        std::vector<std::int64_t> cpu_costs;
        std::vector<std::int64_t> cuda_costs;
        const double cpu_us = MedianMicroseconds(
            graph, k, tasks, cudaee::PathCompatibilityBackend::kCpu, repetitions, &cpu_costs);
        const double cuda_us = MedianMicroseconds(
            graph, k, tasks, cudaee::PathCompatibilityBackend::kCuda, repetitions, &cuda_costs);
        if (cpu_costs != cuda_costs) {
          throw std::runtime_error("benchmark CPU/CUDA cost matrix 不一致");
        }
        std::cout << k << ',' << task_count << ',' << cells << ',' << std::fixed
                  << std::setprecision(3) << cpu_us << ',' << cuda_us << ',' << cpu_us / cuda_us
                  << '\n';
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
