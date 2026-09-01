#include "cuda_edge_elimination/distance.hpp"
#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/lp_epoch.hpp"

#include <filesystem>
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
  epoch.edge_u = {-1, -1};
  epoch.edge_v = {-1, -1};
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

  const std::filesystem::path directory = CUDAEE_TEST_TMP_DIR;
  std::filesystem::create_directories(directory);
  const std::filesystem::path path = directory / "tiny.lp-epoch";
  cudaee::WriteLpEpoch(path, epoch);
  const cudaee::LpEpoch loaded = cudaee::ReadLpEpoch(path);
  Check(loaded.rows == 1, "LP rows round trip");
  Check(loaded.columns == 2, "LP columns round trip");
  Check(loaded.ComputeHash() == loaded.content_hash, "LP hash round trip");
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
}

} // namespace

int main() {
  TestDistances();
  TestLpEpochAndExactBound();
  TestGraphCsrAndVerifierSafety();
  std::cout << "unit tests passed\n";
  return 0;
}
