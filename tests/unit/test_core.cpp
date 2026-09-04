#include "cuda_edge_elimination/distance.hpp"
#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/lp_epoch.hpp"
#include "cuda_edge_elimination/tour.hpp"

#include "../../src/fgpu/main_edge_predicate.hpp"
#include "../../src/fgpu/resident_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("test failure: " + message);
  }
}

cudaee::LpEpoch TinyLp() {
  cudaee::LpEpoch epoch;
  epoch.rows = 1;
  epoch.columns = 2;
  epoch.objective_sense = 1;
  epoch.objective = {1.0, 2.0};
  epoch.row_offsets = {0, 2};
  epoch.column_indices = {0, 1};
  epoch.values = {1.0, 1.0};
  epoch.senses = {'G'};
  epoch.rhs = {1.0};
  epoch.lower_bounds = {0.0, 0.0};
  epoch.upper_bounds = {1.0, 1.0};
  epoch.variable_types = {'C', 'C'};
  epoch.edge_u = {0, 0};
  epoch.edge_v = {1, 2};
  return epoch;
}

void TestDistances() {
  cudaee::Point origin{0.0, 0.0, 0, 0};
  cudaee::Point three_four{3.0, 4.0, 3, 4};
  std::int64_t distance = 0;
  std::string error;
  Check(cudaee::ExactIntegerDistance(origin, three_four, cudaee::DistanceType::kEuc2D, &distance,
                                     &error),
        error);
  Check(distance == 5, "3-4-5 EUC_2D");

  cudaee::Point one_one{1.0, 1.0, 1, 1};
  Check(cudaee::ExactIntegerDistance(origin, one_one, cudaee::DistanceType::kEuc2D, &distance,
                                     &error),
        error);
  Check(distance == 1, "sqrt(2) EUC_2D");
  Check(cudaee::ExactIntegerDistance(origin, one_one, cudaee::DistanceType::kCeil2D, &distance,
                                     &error),
        error);
  Check(distance == 2, "sqrt(2) CEIL_2D");
  Check(cudaee::IntegerSqrtFloor(0) == 0, "isqrt(0)");
  Check(cudaee::IntegerSqrtFloor(15) == 3, "isqrt(15)");
  Check(cudaee::IntegerSqrtFloor(16) == 4, "isqrt(16)");
}

void TestMetricPathDistanceCache() {
  constexpr std::int32_t dimension = 9;
  std::vector<std::int32_t> degree(static_cast<std::size_t>(dimension), dimension - 1);
  std::vector<std::int32_t> neighbors(
      static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension));
  std::vector<std::int64_t> distance(neighbors.size());
  std::vector<std::uint8_t> active(neighbors.size(), 1U);
  for (std::int32_t node = 0; node < dimension; ++node) {
    for (std::int32_t slot = 0; slot < dimension; ++slot) {
      neighbors[static_cast<std::size_t>(node) * dimension + slot] = slot;
    }
    active[static_cast<std::size_t>(node) * dimension + node] = 0U;
  }
  cudaee::detail::quick_hs::GraphView graph{
      .dimension = dimension,
      .degree = degree.data(),
      .neighbors = neighbors.data(),
      .distance = distance.data(),
      .active = active.data(),
  };

  // 用非度量的确定性权重也能覆盖所有严格比较边界；缓存实现必须逐 bit
  // 等价于通用 path-ordering oracle，而不能依赖欧氏三角不等式。
  for (std::int32_t sample = 0; sample < 128; ++sample) {
    for (std::int32_t first = 0; first < dimension; ++first) {
      distance[static_cast<std::size_t>(first) * dimension + first] = 0;
      for (std::int32_t second = first + 1; second < dimension; ++second) {
        const std::int64_t weight =
            1 + (sample * 131 + first * 47 + second * 89 + first * second * 17) % 997;
        distance[static_cast<std::size_t>(first) * dimension + second] = weight;
        distance[static_cast<std::size_t>(second) * dimension + first] = weight;
      }
    }
    std::int32_t node[7]{};
    for (std::int32_t index = 0; index < 7; ++index) {
      node[index] = (index + sample) % dimension;
    }
    const cudaee::detail::quick_hs::SmallPath paths24[3] = {
        {.size = 2, .node = {node[0], node[1]}},
        {.size = 4, .node = {node[2], node[3], node[4], node[5]}},
    };
    Check(cudaee::detail::main_edge::Opt24(graph, node[0], node[1], node[2], node[3],
                                           node[4], node[5]) ==
              cudaee::detail::quick_hs::Opt(graph, paths24, 2),
          "cached opt24 equals generic path oracle");

    const cudaee::detail::quick_hs::SmallPath paths34[3] = {
        {.size = 3, .node = {node[0], node[1], node[2]}},
        {.size = 4, .node = {node[3], node[4], node[5], node[6]}},
    };
    Check(cudaee::detail::main_edge::Opt34(graph, node[0], node[1], node[2], node[3],
                                           node[4], node[5], node[6]) ==
              cudaee::detail::quick_hs::Opt(graph, paths34, 2),
          "cached opt34 equals generic path oracle");

    const cudaee::detail::quick_hs::SmallPath paths33[3] = {
        {.size = 3, .node = {node[0], node[1], node[2]}},
        {.size = 3, .node = {node[3], node[4], node[5]}},
    };
    Check(cudaee::detail::quick_hs::Opt33(graph, node[0], node[1], node[2],
                                          node[3], node[4], node[5]) ==
              cudaee::detail::quick_hs::Opt(graph, paths33, 2),
          "fast opt33 equals generic path oracle");

    // Point move 的 reply 允许通过端点与根路径相接。覆盖所有跨路径单点
    // 重合形状，防止只对顶点不交路径成立的快速必要条件误删合法 reply。
    for (std::int32_t root_position = 0; root_position < 3; ++root_position) {
      for (std::int32_t point_position = 0; point_position < 3; ++point_position) {
        std::int32_t overlapping[3] = {node[3], node[4], node[5]};
        overlapping[point_position] = node[root_position];
        const cudaee::detail::quick_hs::SmallPath overlap_paths[3] = {
            {.size = 3, .node = {node[0], node[1], node[2]}},
            {.size = 3,
             .node = {overlapping[0], overlapping[1], overlapping[2]}},
        };
        Check(cudaee::detail::quick_hs::Opt33(
                  graph, node[0], node[1], node[2], overlapping[0],
                  overlapping[1], overlapping[2]) ==
                  cudaee::detail::quick_hs::Opt(graph, overlap_paths, 2),
              "fast opt33 overlap equals generic path oracle");
      }
    }
  }
}

std::int32_t CompleteEdgeId(const std::int32_t dimension, std::int32_t first,
                            std::int32_t second) {
  if (first > second) {
    std::swap(first, second);
  }
  return first * (2 * dimension - first - 1) / 2 + (second - first - 1);
}

void TestFixedAnchorNonpairTheorem() {
  constexpr std::int32_t dimension = 7;
  constexpr std::int32_t edge_count = dimension * (dimension - 1) / 2;
  std::vector<std::int32_t> degree(static_cast<std::size_t>(dimension), dimension - 1);
  std::vector<std::int32_t> neighbors(
      static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension), -1);
  std::vector<std::uint8_t> active(neighbors.size(), 0U);
  std::vector<std::int32_t> edge_u(static_cast<std::size_t>(edge_count));
  std::vector<std::int32_t> edge_v(static_cast<std::size_t>(edge_count));
  std::vector<std::uint8_t> fixed(static_cast<std::size_t>(edge_count));
  std::vector<std::int64_t> x(static_cast<std::size_t>(dimension));
  std::vector<std::int64_t> y(static_cast<std::size_t>(dimension));
  for (std::int32_t center = 0; center < dimension; ++center) {
    std::int32_t slot = 0;
    for (std::int32_t node = 0; node < dimension; ++node) {
      active[static_cast<std::size_t>(center) * dimension + node] =
          center == node ? 0U : 1U;
      if (center != node) {
        neighbors[static_cast<std::size_t>(center) * dimension + slot++] = node;
      }
      if (center < node) {
        const std::int32_t edge = CompleteEdgeId(dimension, center, node);
        edge_u[static_cast<std::size_t>(edge)] = center;
        edge_v[static_cast<std::size_t>(edge)] = node;
      }
    }
  }
  cudaee::detail::quick_hs::GraphView graph{
      .dimension = dimension,
      .degree = degree.data(),
      .neighbors = neighbors.data(),
      .distance = nullptr,
      .active = active.data(),
      .edge_u = edge_u.data(),
      .edge_v = edge_v.data(),
      .edge_active = nullptr,
      .fixed_edge = fixed.data(),
      .coordinate_x = x.data(),
      .coordinate_y = y.data(),
      .edge_count = edge_count,
      .distance_type = 0U,
      .complete_graph = true,
  };

  for (std::int32_t sample = 0; sample < 32; ++sample) {
    for (std::int32_t node = 0; node < dimension; ++node) {
      x[static_cast<std::size_t>(node)] =
          (sample * 97 + node * 211 + node * node * 31) % 1009;
      y[static_cast<std::size_t>(node)] =
          (sample * 193 + node * 127 + node * node * 53) % 1013;
    }
    std::vector<std::int32_t> permutation(static_cast<std::size_t>(dimension - 1));
    std::iota(permutation.begin(), permutation.end(), 1);
    std::int64_t best_cost = std::numeric_limits<std::int64_t>::max();
    std::vector<std::vector<std::int32_t>> best_tours;
    do {
      std::vector<std::int32_t> tour{0};
      tour.insert(tour.end(), permutation.begin(), permutation.end());
      std::int64_t cost = 0;
      for (std::int32_t index = 0; index < dimension; ++index) {
        cost += cudaee::detail::quick_hs::Distance(
            graph, tour[static_cast<std::size_t>(index)],
            tour[static_cast<std::size_t>((index + 1) % dimension)]);
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_tours.clear();
      }
      if (cost == best_cost) {
        best_tours.push_back(std::move(tour));
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    std::fill(fixed.begin(), fixed.end(), 1U);
    for (const std::vector<std::int32_t>& tour : best_tours) {
      std::vector<std::uint8_t> in_tour(static_cast<std::size_t>(edge_count), 0U);
      for (std::int32_t index = 0; index < dimension; ++index) {
        in_tour[static_cast<std::size_t>(CompleteEdgeId(
            dimension, tour[static_cast<std::size_t>(index)],
            tour[static_cast<std::size_t>((index + 1) % dimension)]))] = 1U;
      }
      for (std::int32_t edge = 0; edge < edge_count; ++edge) {
        fixed[static_cast<std::size_t>(edge)] &= in_tour[static_cast<std::size_t>(edge)];
      }
    }

    for (const std::vector<std::int32_t>& tour : best_tours) {
      for (std::int32_t index = 0; index < dimension; ++index) {
        const std::int32_t center = tour[static_cast<std::size_t>(index)];
        const std::int32_t first =
            tour[static_cast<std::size_t>((index + dimension - 1) % dimension)];
        const std::int32_t second =
            tour[static_cast<std::size_t>((index + 1) % dimension)];
        for (std::int32_t edge = 0; edge < edge_count; ++edge) {
          if (fixed[static_cast<std::size_t>(edge)] == 0U) {
            continue;
          }
          const std::int32_t p = edge_u[static_cast<std::size_t>(edge)];
          const std::int32_t q = edge_v[static_cast<std::size_t>(edge)];
          if (p == center || p == first || p == second || q == center ||
              q == first || q == second) {
            continue;
          }
          Check(cudaee::detail::quick_hs::Opt23(
                    graph, p, q, first, center, second,
                    cudaee::detail::quick_hs::Distance(graph, p, q),
                    cudaee::detail::quick_hs::Distance(graph, first, center),
                    cudaee::detail::quick_hs::Distance(graph, center, second)),
                "disjoint fixed edge cannot exclude an optimal-tour pair");
        }
      }
    }
  }
}

void TestLpEpochAndExactBound() {
  cudaee::LpEpoch epoch = TinyLp();
  epoch.Validate();
  const cudaee::ExactBound bound = cudaee::BuildExactModelBound(epoch, {1.0});
  Check(bound.certified, bound.reason);
  Check(bound.numerator == "16777216", "exact lower-bound numerator");
  Check(bound.denominator == 16777216, "exact lower-bound denominator");
  const cudaee::ExactModelEvaluation evaluation = cudaee::BuildExactModelEvaluation(epoch, {1.0});
  Check(evaluation.bound.certified && evaluation.bound.numerator == bound.numerator,
        "exact model evaluation preserves bound");
  Check(evaluation.lower_bound_numerator == static_cast<__int128>(16777216) &&
            evaluation.reduced_cost_numerator.size() == 2U &&
            evaluation.reduced_cost_numerator[0] == 0 &&
            evaluation.reduced_cost_numerator[1] == static_cast<__int128>(16777216),
        "exact model evaluation exposes quantized reduced costs");

  const std::filesystem::path directory = CUDAEE_TEST_TMP_DIR;
  std::filesystem::create_directories(directory);
  const std::filesystem::path path = directory / "tiny.lp-epoch";
  cudaee::WriteLpEpoch(path, epoch);
  const cudaee::LpEpoch loaded = cudaee::ReadLpEpoch(path);
  Check(loaded.rows == 1, "LP rows round trip");
  Check(loaded.columns == 2, "LP columns round trip");
  Check(loaded.ComputeHash() == loaded.content_hash, "LP hash round trip");
}

void TestLpStableIdentityAndWarmProjection() {
  const cudaee::LpEpoch source = TinyLp();
  const cudaee::LpStableIdentity source_identity = cudaee::ComputeLpStableIdentity(source);
  Check(source_identity.complete, source_identity.reason);

  cudaee::LpSolution accepted;
  accepted.numerically_accepted = true;
  accepted.primal = {0.25, 0.75};
  accepted.dual = {1.0};
  const cudaee::LpWarmStart warm = cudaee::BuildLpWarmStart(source, accepted);

  cudaee::LpEpoch reordered = source;
  reordered.objective = {2.0, 1.0};
  reordered.edge_u = {0, 0};
  reordered.edge_v = {2, 1};
  const cudaee::LpStableIdentity reordered_identity = cudaee::ComputeLpStableIdentity(reordered);
  Check(reordered_identity.complete, reordered_identity.reason);
  Check(reordered_identity.identity_hash == source_identity.identity_hash,
        "LP stable identity hash ignores column order");
  const cudaee::LpWarmStartProjection reordered_projection =
      cudaee::ProjectLpWarmStart(warm, reordered);
  Check(reordered_projection.accepted, reordered_projection.reason);
  Check(reordered_projection.primal == std::vector<double>({0.75, 0.25}),
        "LP primal maps by edge identity rather than position");
  Check(reordered_projection.dual == std::vector<double>({1.0}),
        "LP dual maps by canonical row identity");
  Check(reordered_projection.column_coverage == 1.0 && reordered_projection.row_coverage == 1.0,
        "LP reorder keeps full warm-start coverage");

  cudaee::LpEpoch expanded = reordered;
  expanded.rows = 2;
  expanded.columns = 3;
  expanded.objective = {2.0, 1.0, 3.0};
  expanded.row_offsets = {0, 2, 3};
  expanded.column_indices = {0, 1, 2};
  expanded.values = {1.0, 1.0, 1.0};
  expanded.senses = {'G', 'G'};
  expanded.rhs = {1.0, 0.0};
  expanded.lower_bounds = {0.0, 0.0, 0.0};
  expanded.upper_bounds = {1.0, 1.0, 1.0};
  expanded.variable_types = {'C', 'C', 'C'};
  expanded.edge_u = {0, 0, 1};
  expanded.edge_v = {2, 1, 2};
  const cudaee::LpWarmStartProjection strict_projection =
      cudaee::ProjectLpWarmStart(warm, expanded, 0.8);
  Check(!strict_projection.accepted, "LP warm start rejects insufficient identity coverage");
  Check(std::abs(strict_projection.column_coverage - (2.0 / 3.0)) < 1.0e-12 &&
            strict_projection.row_coverage == 0.5,
        "LP warm-start coverage is measured on target identities");
  const cudaee::LpWarmStartProjection permissive_projection =
      cudaee::ProjectLpWarmStart(warm, expanded, 0.5);
  Check(permissive_projection.accepted, permissive_projection.reason);
  Check(permissive_projection.primal[2] == 0.0 && permissive_projection.dual[1] == 0.0,
        "new LP identities receive bounded zero defaults");

  cudaee::LpWarmStart tampered = warm;
  tampered.identity.identity_hash ^= 1U;
  Check(!cudaee::ProjectLpWarmStart(tampered, reordered).accepted,
        "tampered warm-start identity hash fails closed");

  cudaee::LpEpoch duplicate = source;
  duplicate.edge_u[1] = duplicate.edge_u[0];
  duplicate.edge_v[1] = duplicate.edge_v[0];
  Check(!cudaee::ComputeLpStableIdentity(duplicate).complete,
        "duplicate edge identities fail closed");

  bool rejected_solution = false;
  try {
    cudaee::LpSolution rejected = accepted;
    rejected.numerically_accepted = false;
    static_cast<void>(cudaee::BuildLpWarmStart(source, rejected));
  } catch (const std::invalid_argument&) {
    rejected_solution = true;
  }
  Check(rejected_solution, "numerically rejected solution cannot seed warm start");
}

void TestGraphCsrAndVerifierSafety() {
  cudaee::GraphSnapshot graph;
  graph.dimension = 4;
  graph.distance_type = cudaee::DistanceType::kEuc2D;
  graph.integer_coordinates = true;
  graph.integer_distance_safe = true;
  graph.points = {{0.0, 0.0, 0, 0}, {10.0, 0.0, 10, 0}, {0.0, 10.0, 0, 10}, {10.0, 10.0, 10, 10}};
  for (std::int32_t u = 0; u < graph.dimension; ++u) {
    for (std::int32_t v = u + 1; v < graph.dimension; ++v) {
      graph.edges.push_back({u, v, graph.Distance(u, v), true});
    }
  }
  graph.RebuildCsr();
  Check(graph.ActiveEdgeCount() == 6, "K4 edge count");
  for (std::int32_t vertex = 0; vertex < graph.dimension; ++vertex) {
    Check(graph.Degree(vertex) == 3, "K4 degree");
  }
  const auto candidates = cudaee::FindJvCandidatesCpu(graph);
  for (const cudaee::Candidate& candidate : candidates) {
    std::string reason;
    Check(cudaee::VerifyJvCandidate(graph, candidate, &reason), reason);
  }
  const std::uint64_t first_hash = graph.ContentHash();
  Check(first_hash == graph.ContentHash(), "graph hash determinism");

  // 非规范存储顺序必须走排序回退，不能破坏 CSR 二分查找契约。
  std::swap(graph.edges.front(), graph.edges.back());
  graph.RebuildCsr();
  for (const cudaee::Edge& edge : graph.edges) {
    Check(graph.HasActiveEdge(edge.u, edge.v), "unsorted graph CSR fallback");
  }
}

void TestProtectedTour() {
  const std::filesystem::path source = CUDAEE_SOURCE_DIR;
  cudaee::GraphSnapshot graph = cudaee::GraphSnapshot::Load(
      source / "tests/data/recursive-point.tsp", source / "tests/data/recursive-point.edg");
  std::vector<std::int32_t> tour =
      cudaee::ReadTsplibTour(source / "tests/data/recursive-point.tour", graph.dimension);
  const cudaee::ProtectedTourCheck checked = cudaee::CheckProtectedTour(graph, tour);
  Check(checked.cost == 395, "protected tour cost");
  Check(checked.missing_edges == 0U, "protected tour present in complete fixture");

  graph.edges.front().active = false;
  graph.RebuildCsr();
  const cudaee::ProtectedTourCheck damaged = cudaee::CheckProtectedTour(graph, tour);
  Check(damaged.cost == checked.cost, "protected tour metric cost is independent of sparse graph");
  Check(damaged.missing_edges == 1U, "protected tour detects removed edge");

  std::reverse(tour.begin() + 1, tour.end());
  const cudaee::ProtectedTourCheck reversed = cudaee::CheckProtectedTour(graph, tour);
  Check(reversed.tour_hash == checked.tour_hash, "tour hash ignores direction");

  const std::filesystem::path temporary_directory = CUDAEE_TEST_TMP_DIR;
  std::filesystem::create_directories(temporary_directory);
  const std::filesystem::path damaged_tour = temporary_directory / "damaged.tour";
  {
    std::ofstream output(damaged_tour);
    output << "TYPE : TOUR\nDIMENSION : 8\nTOUR_SECTION\n1 2 3 4 5 6 7 7\n-1\nEOF\n";
  }
  bool rejected = false;
  try {
    static_cast<void>(cudaee::ReadTsplibTour(damaged_tour, graph.dimension));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  Check(rejected, "tour parser rejects duplicate node");
}

void TestCompleteGraphLoader() {
  const std::filesystem::path source = CUDAEE_SOURCE_DIR;
  const std::filesystem::path tsp = source / "tests/data/recursive-point.tsp";
  const cudaee::GraphSnapshot complete = cudaee::GraphSnapshot::LoadComplete(tsp);
  Check(complete.dimension == 8, "complete graph dimension");
  Check(complete.ActiveEdgeCount() == 28U, "complete graph edge count");
  for (std::int32_t vertex = 0; vertex < complete.dimension; ++vertex) {
    Check(complete.Degree(vertex) == 7, "complete graph degree");
  }

  const std::filesystem::path directory = CUDAEE_TEST_TMP_DIR;
  std::filesystem::create_directories(directory);
  const std::filesystem::path edges = directory / "recursive-point.complete.edg";
  complete.WriteActiveEdges(edges);
  const cudaee::GraphSnapshot reloaded = cudaee::GraphSnapshot::Load(tsp, edges);
  Check(reloaded.ContentHash() == complete.ContentHash(), "complete graph write/load hash");
}

void CheckSameCandidates(const std::vector<cudaee::Candidate>& expected,
                         const std::vector<cudaee::Candidate>& actual,
                         const std::string& description) {
  Check(expected.size() == actual.size(), description + " candidate count");
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    Check(expected[index].edge_id == actual[index].edge_id &&
              expected[index].witness == actual[index].witness &&
              expected[index].method == actual[index].method,
          description + " candidate content");
  }
}

void TestJvCudaResidentCache() {
  std::string reason;
  if (!cudaee::CudaBackendAvailable(&reason)) {
    return;
  }
  const std::filesystem::path source = CUDAEE_SOURCE_DIR;
  cudaee::GraphSnapshot graph = cudaee::GraphSnapshot::Load(
      source / "tests/data/recursive-point.tsp", source / "tests/data/recursive-point.edg");
  cudaee::ClearJvCudaCache();

  int selected_device = -1;
  cudaee::JvCudaCacheUsage first_usage;
  const std::vector<cudaee::Candidate> first =
      cudaee::FindJvCandidatesCuda(graph, &selected_device, &first_usage);
  CheckSameCandidates(cudaee::FindJvCandidatesCpu(graph), first, "first CUDA JV cache run");
  Check(selected_device >= 0 && !first_usage.static_hit && !first_usage.workspace_hit &&
            first_usage.resident_bytes > 0U,
        "first CUDA JV cache run uploads static graph and workspace");
  const std::uint64_t edge_count = graph.edges.size();
  const std::uint64_t node_count = graph.points.size();
  const std::uint64_t adjacency_count = graph.neighbors.size();
  // CSR 动态区应为两个 int32 数组，不允许悄悄退回重复的 int64 权重副本。
  const std::uint64_t expected_resident_bytes =
      24U * edge_count + 20U * node_count + 4U + 8U * adjacency_count;
  Check(first_usage.resident_bytes == expected_resident_bytes,
        "CUDA JV resident bytes use int32 CSR edge ids");
  Check(std::isfinite(first_usage.h2d_ms) && first_usage.h2d_ms >= 0.0 &&
            std::isfinite(first_usage.kernel_ms) && first_usage.kernel_ms >= 0.0 &&
            std::isfinite(first_usage.d2h_ms) && first_usage.d2h_ms >= 0.0,
        "CUDA JV phase timings are finite and non-negative");

  cudaee::JvCudaCacheUsage second_usage;
  const std::vector<cudaee::Candidate> second =
      cudaee::FindJvCandidatesCuda(graph, &selected_device, &second_usage);
  CheckSameCandidates(first, second, "second CUDA JV cache run");
  Check(second_usage.static_hit && second_usage.workspace_hit &&
            second_usage.resident_bytes == first_usage.resident_bytes,
        "second CUDA JV cache run reuses exact resident data");

  graph.edges.front().active = false;
  graph.RebuildCsr();
  cudaee::JvCudaCacheUsage active_usage;
  const std::vector<cudaee::Candidate> active =
      cudaee::FindJvCandidatesCuda(graph, &selected_device, &active_usage);
  CheckSameCandidates(cudaee::FindJvCandidatesCpu(graph), active, "active-only CUDA JV cache run");
  Check(active_usage.static_hit && active_usage.workspace_hit,
        "active-only change reuses immutable buffers but refreshes dynamic CSR");

  graph.points.front().integer_x += 1000;
  graph.points.front().x += 1000.0;
  for (cudaee::Edge& edge : graph.edges) {
    edge.weight = graph.Distance(edge.u, edge.v);
  }
  graph.RebuildCsr();
  cudaee::JvCudaCacheUsage changed_usage;
  const std::vector<cudaee::Candidate> changed =
      cudaee::FindJvCandidatesCuda(graph, &selected_device, &changed_usage);
  CheckSameCandidates(cudaee::FindJvCandidatesCpu(graph), changed,
                      "changed-static CUDA JV cache run");
  Check(!changed_usage.static_hit && changed_usage.workspace_hit,
        "coordinate change invalidates exact static cache key");
  cudaee::ClearJvCudaCache();
}

void TestQuickHsWarpPathDifferential() {
  std::string reason;
  if (!cudaee::detail::ResidentEliminationCudaAvailable(&reason)) {
    return;
  }
  const cudaee::detail::QuickHsPathDifferentialResult result =
      cudaee::detail::RunQuickHsPathDifferentialCuda(-1, 64U);
  Check(result.cases == 64U, "Quick-HS warp path differential case count");
  Check(result.mismatches == 0U,
        "Quick-HS warp-DP exactly matches host permutation oracle");
}

} // namespace

int main() {
  TestDistances();
  TestMetricPathDistanceCache();
  TestFixedAnchorNonpairTheorem();
  TestLpEpochAndExactBound();
  TestLpStableIdentityAndWarmProjection();
  TestGraphCsrAndVerifierSafety();
  TestProtectedTour();
  TestCompleteGraphLoader();
  TestJvCudaResidentCache();
  TestQuickHsWarpPathDifferential();
  std::cout << "unit tests passed\n";
  return 0;
}
