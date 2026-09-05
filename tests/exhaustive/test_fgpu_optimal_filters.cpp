#include "../../src/fgpu/gpu_bootstrap.hpp"
#include "../../src/fgpu/main_edge_predicate.hpp"
#include "../../src/fgpu/resident_backend.hpp"

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
  std::string gpu_reason;
  const bool has_gpu = cudaee::detail::ResidentEliminationCudaAvailable(&gpu_reason);
#ifdef CUDAEE_TEST_REQUIRE_GPU
  if (!has_gpu) {
    std::cout << "GPU property checks unavailable: " << gpu_reason << '\n';
    return 77;
  }
#endif
  std::uint64_t prime_proposals = 0;
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
    if (has_gpu) {
      // 完全图与包含全部最优 tour 的中间快照，均关闭 geometry。
      // 小候选域用于触发延期；这是正确性夹具，不用于性能或无标签入口验收。
      cudaee::GraphSnapshot input;
      input.dimension = n;
      input.distance_type =
          sample / 6 % 2 ? cudaee::DistanceType::kCeil2D : cudaee::DistanceType::kEuc2D;
      input.integer_coordinates = input.integer_distance_safe = true;
      for (int node = 0; node < n; ++node)
        input.points.push_back(
            {static_cast<double>(x[node]), static_cast<double>(y[node]), x[node], y[node]});
      for (std::size_t edge = 0; edge < u.size(); ++edge)
        input.edges.push_back({u[edge], v[edge], distances[u[edge] * n + v[edge]], true});
      input.RebuildCsr();
      cudaee::detail::GpuBootstrap bootstrap(input, 0);
      bootstrap.BuildPermutationCatalog();
      for (const bool sparse_snapshot : {false, true}) {
        for (std::size_t edge = 0; edge < u.size(); ++edge)
          input.edges[edge].active = !sparse_snapshot || ((edge_union >> edge) & 1U) ||
                                     (edge + static_cast<std::size_t>(sample)) % 3 != 0;
        input.RebuildCsr();
        std::uint64_t pairs = 0, degree_sum = 0;
        for (int node = 0; node < n; ++node) {
          unsigned degree = 0;
          for (const auto& edge : input.edges)
            degree += edge.active && (edge.u == node || edge.v == node);
          pairs += degree * (degree - 1U) / 2U;
          degree_sum += degree;
        }
        // 无向边计数 = degree sum / 2，quick_hs_candidates = 2。
        const bool deferred = pairs > degree_sum;
        cudaee::detail::ResidentGpuResult raw_result;
        for (int variant = 0; variant < 4; ++variant) {
          const bool prime = variant / 2 != 0, compact = variant % 2 != 0;
          cudaee::detail::ResidentGpuOptions options;
          options.device = 0;
          options.collect_trace = false;
          options.gpu_replay = true;
          options.enable_point_nonpair = true;
          options.enable_fixing = true;
          options.point_near_first = true;
          options.point_adaptive_start = true;
          options.point_prime_near = prime;
          options.quick_reply_cache = compact;
          // 2/4 CTA 驻留策略与目录有/无均覆盖；不改变叶子的完整枚举域。
          options.point_cta_blocks = sample % 2 ? 4U : 2U;
          options.permutation_orders = sample % 3 ? bootstrap.permutations() : nullptr;
          options.quick_hs_candidates = 2;
          options.quick_hs_pair_trials = 0;
          const auto result = cudaee::detail::RunResidentEliminationCuda(
              input, std::vector<std::uint8_t>(u.size(), 0U), options);
          if (!compact)
            raw_result = result;
          else if (result.final_active != raw_result.final_active ||
                   result.final_fixed != raw_result.final_fixed ||
                   result.final_nonpairs.size() != raw_result.final_nonpairs.size() ||
                   !std::equal(result.final_nonpairs.begin(), result.final_nonpairs.end(),
                               raw_result.final_nonpairs.begin(), [](const auto& a, const auto& b) {
                                 return a.center == b.center && a.first == b.first &&
                                        a.second == b.second;
                               })) {
            std::cerr << "Quick reply compaction changed fixed point sample=" << sample << '\n';
            return 1;
          }
          if (!result.converged || result.proof_rejected != 0 ||
              result.lp.point_initial_pairs != pairs ||
              result.lp.point_initial_edge_frontier != degree_sum ||
              result.lp.point_service_sweeps == 0 ||
              result.lp.point_prime_sweeps != static_cast<unsigned>(prime && deferred)) {
            std::cerr << "Point prime replay/convergence/sweep failure sample=" << sample << '\n';
            return 1;
          }
          prime_proposals += result.lp.point_prime_proposals;
          std::cout << "GPU Point sample=" << sample << " sparse=" << sparse_snapshot
                    << " prime=" << prime << " compact=" << compact
                    << " prime_proposals=" << result.lp.point_prime_proposals
                    << " point_proposals=" << result.point_nonpair_committed << '\n';
          for (std::size_t edge = 0; edge < u.size(); ++edge) {
            if ((((edge_union >> edge) & 1U) && !result.final_active[edge]) ||
                (result.final_fixed[edge] && !((mandatory >> edge) & 1U))) {
              std::cerr << "Point prime changed an optimum edge/fix sample=" << sample << '\n';
              return 1;
            }
          }
          for (const auto& pair : result.final_nonpairs)
            for (const auto& optimal : optima)
              for (int position = 0; position < n; ++position) {
                const int first = optimal[(position + n - 1) % n];
                const int second = optimal[(position + 1) % n];
                if (pair.center == optimal[position] && pair.first == std::min(first, second) &&
                    pair.second == std::max(first, second)) {
                  std::cerr << "Point prime removed an optimum pair sample=" << sample << '\n';
                  return 1;
                }
              }
        }
      }
    }
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
  if (has_gpu && prime_proposals == 0) {
    std::cerr << "Point prime tests never exercised positive proposals/replay\n";
    return 1;
  }
  if (has_gpu)
    std::cout << "Point prime positive proposals with independent all-opt oracle: "
              << prime_proposals << '\n';
}
