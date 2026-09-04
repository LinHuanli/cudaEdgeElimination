#include "cuda_edge_elimination/distance.hpp"
#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/lp_epoch.hpp"
#include "cuda_edge_elimination/tour.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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

} // namespace

int main() {
  TestDistances();
  TestLpEpochAndExactBound();
  TestLpStableIdentityAndWarmProjection();
  TestGraphCsrAndVerifierSafety();
  TestProtectedTour();
  TestCompleteGraphLoader();
  TestJvCudaResidentCache();
  std::cout << "unit tests passed\n";
  return 0;
}
