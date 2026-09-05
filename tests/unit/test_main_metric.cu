#include <cuda_runtime.h>

#include "../../src/cuda/main_edge_metric.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kCases = 240;
constexpr int kStride = 24;
struct Case {
  int n;
  int node[5];
  int degree[kStride];
  int neighbors[kStride * kStride];
  std::int64_t distance[kStride * kStride];
  std::uint8_t active[kStride * kStride];
  int expected;
  int actual;
};

void Check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

__host__ __device__ cudaee::detail::quick_hs::GraphView View(const Case& input) {
  return {.dimension = input.n,
          .degree = input.degree,
          .neighbors = input.neighbors,
          .distance = input.distance,
          .active = input.active};
}

__global__ void Evaluate(Case* const cases) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (index >= kCases)
    return;
  auto& input = cases[index];
  const auto* v = input.node;
  const bool result = cudaee::detail::main_metric::MetricExcessAdmitsPairWarp(
      View(input), v[0], v[1], v[2], v[3], v[4]);
  if ((threadIdx.x & 31U) == 0)
    input.actual = result ? 1 : 0;
}
} // namespace

int main() {
  try {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
      return 77;
    Case* cases{};
    Check(cudaMallocManaged(&cases, sizeof(Case) * kCases));
    std::mt19937 random(20142023);
    int closed = 0, open = 0;
    for (int sample = 0; sample < kCases; ++sample) {
      auto& input = cases[sample];
      input = {};
      const int sizes[]{5, 6, 8, 12, 24};
      const int n = input.n = sizes[sample % 5];
      std::vector<std::pair<int, int>> points(n);
      for (auto& [x, y] : points) {
        x = sample % 7 == 0 ? 0 : static_cast<int>(random() % 100);
        y = sample % 7 == 0 ? 0 : static_cast<int>(random() % 100);
      }
      for (int first = 0; first < n; ++first) {
        for (int second = first + 1; second < n; ++second) {
          const auto dx = points[first].first - points[second].first;
          const auto dy = points[first].second - points[second].second;
          const auto cost =
              static_cast<std::int64_t>(std::floor(std::sqrt(dx * dx + dy * dy) + 0.5));
          input.distance[first * n + second] = input.distance[second * n + first] = cost;
          // 包含完整高度数图、稀疏图和 degree-2 固定环，不只测低度数退化情形。
          const bool active = second == first + 1 || (first == 0 && second == n - 1) ||
                              sample % 3 == 0 || (sample % 3 == 1 && random() % 3 == 0);
          input.active[first * n + second] = input.active[second * n + first] = active;
        }
      }
      for (int first = 0; first < n; ++first) {
        auto* row = input.neighbors + first * n;
        for (int second = 0; second < n; ++second) {
          if (input.active[first * n + second])
            row[input.degree[first]++] = second;
        }
        std::sort(row, row + input.degree[first], [&](int a, int b) {
          return std::pair(input.distance[first * n + a], a) <
                 std::pair(input.distance[first * n + b], b);
        });
      }
      std::vector<int> order(n);
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), random);
      std::copy_n(order.begin(), 5, input.node);
      const auto* v = input.node;
      const auto view = View(input);
      using namespace cudaee::detail;
      input.expected = main_edge::MetricExcessAdmitsPair(
          view, v[0], v[1], v[2], v[3], v[4], quick_hs::Distance(view, v[1], v[2]),
          quick_hs::Distance(view, v[2], v[3]), quick_hs::Distance(view, v[3], v[4]));
      input.actual = -1;
      input.expected ? ++open : ++closed;
    }
    Evaluate<<<(kCases + 3) / 4, 128>>>(cases);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    for (int sample = 0; sample < kCases; ++sample) {
      if (cases[sample].actual != cases[sample].expected) {
        throw std::runtime_error("full-degree metric mismatch: " + std::to_string(sample));
      }
    }
    Check(cudaFree(cases));
    if (!open || !closed)
      throw std::runtime_error("metric 测试缺少开放/关闭分支覆盖");
    std::cout << "full-degree metric differential: " << kCases << " cases, open=" << open
              << " closed=" << closed << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
