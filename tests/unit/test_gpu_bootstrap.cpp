#include "../../src/fgpu/gpu_bootstrap.hpp"
#include "../../src/fgpu/resident_backend.hpp"
#include "cuda_edge_elimination/distance.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
std::int64_t Cost(const cudaee::GraphSnapshot& graph, const std::vector<int>& tour) {
  std::int64_t cost = 0;
  for (std::size_t i = 0; i < tour.size(); ++i)
    cost += graph.Distance(tour[i], tour[(i + 1) % tour.size()]);
  return cost;
}
// 测试 oracle 用显式重建 tour/重新计算成本，不复用 CUDA 增量差值公式。
void CheckLocalFixedPoint(const cudaee::GraphSnapshot& graph, const std::vector<int>& tour) {
  const auto cost = Cost(graph, tour);
  const int n = graph.dimension;
  for (int i = 0; i < n; ++i)
    for (int j = i + 2; j < n; ++j) {
      auto next = tour;
      std::reverse(next.begin() + i + 1, next.begin() + j + 1);
      Require(Cost(graph, next) >= cost, "GPU tour 不是 2-opt 局部不动点");
    }
  for (int length = 1; length <= 3 && length < n - 1; ++length) {
    for (int i = 0; i < n; ++i) {
      std::vector<int> rotated(n);
      for (int j = 0; j < n; ++j)
        rotated[j] = tour[(i + j) % n];
      std::vector<int> segment(rotated.begin(), rotated.begin() + length);
      std::vector<int> rest(rotated.begin() + length, rotated.end());
      for (std::size_t position = 1; position < rest.size(); ++position) {
        auto next = rest;
        next.insert(next.begin() + static_cast<std::ptrdiff_t>(position), segment.begin(),
                    segment.end());
        Require(Cost(graph, next) >= cost, "GPU tour 不是循环 Or-opt 局部不动点");
      }
    }
  }
}
} // namespace

int main() try {
  std::string reason;
  if (!cudaee::detail::ResidentEliminationCudaAvailable(&reason)) {
    std::cout << reason << '\n';
    return 77;
  }
  std::mt19937 random(20142023U);
  for (const auto denominator : {1U, 2U})
    for (const auto metric : {cudaee::DistanceType::kEuc2D, cudaee::DistanceType::kCeil2D}) {
      for (int n : {3, 4, 5, 8, 16, 31}) {
        cudaee::GraphSnapshot graph;
        graph.dimension = n;
        graph.integer_coordinates = true;
        graph.integer_distance_safe = true;
        graph.distance_type = metric;
        graph.integer_coordinate_denominator = denominator;
        graph.integer_coordinates = denominator == 1U;
        for (int i = 0; i < n; ++i) {
          const auto x = static_cast<std::int64_t>(random() % 101) - 50;
          const auto y = static_cast<std::int64_t>(random() % 101) - 50;
          graph.points.push_back(
              {static_cast<double>(x) / denominator, static_cast<double>(y) / denominator, x, y});
        }
        if (n == 5)
          graph.points.assign(n, graph.points.front()); // 零距离、多最优解边界。
        cudaee::detail::GpuBootstrap bootstrap(graph, 0);
        bootstrap.BuildCompleteGraph(&graph);
        Require(graph.edges.size() == static_cast<std::size_t>(n * (n - 1) / 2),
                "GPU 完整图大小错误");
        std::size_t edge = 0;
        for (int u = 0; u < n; ++u)
          for (int v = u + 1; v < n; ++v, ++edge) {
            const auto& actual = graph.edges[edge];
            Require(actual.u == u && actual.v == v && actual.active &&
                        actual.weight == graph.Distance(u, v),
                    "GPU 图/距离与独立 CPU 距离不同");
          }
        bootstrap.GenerateIncumbent();
        const auto tour = bootstrap.tour();
        auto permutation = tour;
        std::sort(permutation.begin(), permutation.end());
        for (int i = 0; i < n; ++i)
          Require(permutation[i] == i, "GPU tour 不是排列");
        Require(Cost(graph, tour) == bootstrap.metrics().incumbent_cost,
                "GPU incumbent 成本不一致");
        CheckLocalFixedPoint(graph, tour);
        bootstrap.GenerateIncumbent();
        Require(tour == bootstrap.tour(), "确定性 GPU 多起点产生不同 tour");
      }
    }
  std::cout << "GPU bootstrap: 24 coordinate/distance/tour/fixed-point cases passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
