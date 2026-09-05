#include "../../src/fgpu/main_edge_predicate.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

int main() {
  namespace hs = cudaee::detail::quick_hs;
  namespace main_edge = cudaee::detail::main_edge;
  std::mt19937 rng(0xF6A2026U);
  for (int sample = 0; sample < 36; ++sample) {
    const int n = 3 + sample % 6;
    std::vector<int> x(n), y(n), u, v;
    std::vector<std::int64_t> distances(n * n);
    for (int i = 0; i < n; ++i) {
      x[i] = static_cast<int>(rng() % 12U);
      y[i] = static_cast<int>(rng() % 12U);
    }
    if (sample >= 24) {
      x[n - 1] = x[0];
      y[n - 1] = y[0];
    }
    for (int a = 0; a < n; ++a)
      for (int b = 0; b < n; ++b) {
        const int square = (x[a] - x[b]) * (x[a] - x[b]) + (y[a] - y[b]) * (y[a] - y[b]);
        int root = 0;
        while ((root + 1) * (root + 1) <= square)
          ++root;
        distances[a * n + b] =
            root + (sample / 6 % 2 ? square > root * root : square - root * root > root);
      }
    auto edge_id = [n](int a, int b) {
      if (a > b)
        std::swap(a, b);
      return a * (2 * n - a - 1) / 2 + b - a - 1;
    };
    for (int a = 0; a < n; ++a)
      for (int b = a + 1; b < n; ++b) {
        u.push_back(a);
        v.push_back(b);
      }
    std::vector<int> tour(n);
    std::iota(tour.begin(), tour.end(), 0);
    std::vector<std::vector<int>> optima;
    std::int64_t optimum = std::numeric_limits<std::int64_t>::max();
    std::uint64_t edge_union = 0, mandatory = 0;
    do {
      if (tour[1] > tour[n - 1])
        continue;
      std::int64_t cost = 0;
      std::uint64_t edges = 0;
      for (int i = 0; i < n; ++i) {
        cost += distances[tour[i] * n + tour[(i + 1) % n]];
        edges |= std::uint64_t{1} << edge_id(tour[i], tour[(i + 1) % n]);
      }
      if (cost < optimum) {
        optimum = cost;
        optima.clear();
        edge_union = edges;
        mandatory = edges;
      }
      if (cost == optimum) {
        optima.push_back(tour);
        edge_union |= edges;
        mandatory &= edges;
      }
    } while (std::next_permutation(tour.begin() + 1, tour.end()));
    for (int mode = 0; mode < 3; ++mode) {
      // 完全图、最优边并集、含若干额外边的中间快照，都包含所有最优解。
      std::vector<std::uint8_t> active(n * n), edge_active(u.size()), fixed(u.size());
      std::vector<int> degree(n), neighbors(n * n);
      for (int e = 0; e < static_cast<int>(u.size()); ++e) {
        fixed[e] = (mandatory >> e) & 1U;
        edge_active[e] = mode == 0 || ((edge_union >> e) & 1U) || (mode == 2 && rng() % 2);
        if (!edge_active[e])
          continue;
        const int a = u[e], b = v[e];
        active[a * n + b] = active[b * n + a] = 1;
        neighbors[a * n + degree[a]++] = b;
        neighbors[b * n + degree[b]++] = a;
      }
      const hs::GraphView graph{.dimension = n,
                                .degree = degree.data(),
                                .neighbors = neighbors.data(),
                                .distance = distances.data(),
                                .active = active.data(),
                                .edge_u = u.data(),
                                .edge_v = v.data(),
                                .edge_active = edge_active.data(),
                                .fixed_edge = fixed.data(),
                                .edge_count = static_cast<std::int64_t>(u.size()),
                                .complete_graph = true};
      for (const auto& optimal : optima) {
        for (int root = 0; root < n; ++root) {
          const int a = optimal[root], b = optimal[(root + 1) % n];
          for (int center = 0; center < n; ++center) {
            const int c = optimal[center];
            if (c == a || c == b)
              continue;
            const int first = optimal[(center + n - 1) % n], second = optimal[(center + 1) % n];
            if (!main_edge::AllowedPair(graph, a, b, c, first, second)) {
              std::cerr << "false AllowedPair sample=" << sample << " mode=" << mode << " n=" << n
                        << " root=" << a << ',' << b << " path=" << first << ',' << c << ','
                        << second << "\n";
              return 1;
            }
          }
        }
      }
      for (int edge = 0; edge < static_cast<int>(u.size()); ++edge) {
        if (((edge_union >> edge) & 1U) == 0)
          continue;
        const auto witness = hs::FindWitness(graph, u[edge], v[edge], 10, 0, true);
        std::vector<int> potentials;
        for (int c = 0; c < n; ++c)
          if (c != u[edge] && c != v[edge])
            potentials.push_back(c);
        if (witness.c >= 0 || main_edge::CanEliminate(graph, u[edge], v[edge], potentials.data(),
                                                      static_cast<int>(potentials.size()))) {
          std::cerr << "false edge sample=" << sample << " mode=" << mode << " n=" << n
                    << " root=" << u[edge] << ',' << v[edge] << "\n";
          return 1;
        }
      }
    }
    std::cout << "CPU property sample=" << sample << " n=" << n << " optima=" << optima.size()
              << " all optimum path filters verified\n"
              << std::flush;
  }
}
