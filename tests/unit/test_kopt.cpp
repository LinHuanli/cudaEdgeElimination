#include "cuda_edge_elimination/local_search.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void CCelim_compare_three_swap(int a, int b, int c, int d, int e, int f, int** matrix, int* good,
                               int* added);
void CCelim_compare_four_swap(int a, int b, int c, int d, int e, int f, int g, int h, int** matrix,
                              int* good, int* added);
void CCelim_compare_five_swap(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j,
                              int** matrix, int* good, int* added);
}

namespace {

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("test failure: " + message);
  }
}

cudaee::GraphSnapshot MakeGraph(const std::vector<cudaee::Point>& points) {
  cudaee::GraphSnapshot graph;
  graph.dimension = static_cast<std::int32_t>(points.size());
  graph.distance_type = cudaee::DistanceType::kEuc2D;
  graph.integer_coordinates = true;
  graph.integer_distance_safe = true;
  graph.points = points;
  return graph;
}

void TestReconnectTemplateGeneration() {
  constexpr std::array<std::size_t, 3> kExpectedCounts = {4, 25, 208};
  constexpr std::array<std::uint64_t, 3> kExpectedHashes = {
      0xe58af5e08d290d04ULL, 0x03179e3ca191ce82ULL, 0x6696dde548591bceULL};
  for (std::uint32_t k = 3; k <= 5; ++k) {
    const cudaee::KOptReconnectTable table = cudaee::BuildKOptReconnectTable(k);
    const std::size_t expected_index = static_cast<std::size_t>(k - 3U);
    Check(table.templates.size() == kExpectedCounts[expected_index],
          "proper reconnect template count");
    Check(table.generator_hash == kExpectedHashes[expected_index],
          "pinned reconnect-template generator hash");
    for (const cudaee::EndpointMatching& matching : table.templates) {
      Check(cudaee::IsPerfectEndpointMatching(matching, k), "reconnect template is perfect");
      for (std::uint32_t edge = 0; edge < k; ++edge) {
        const std::size_t first_endpoint = std::size_t{2} * edge;
        Check(matching.mate[first_endpoint] != first_endpoint + 1,
              "reconnect template does not reuse a deleted edge");
      }
    }
    std::cout << "k=" << k << " reconnect_hash=" << table.generator_hash
              << " templates=" << table.templates.size() << '\n';
  }
}

void TestReconnectTemplatesAgainstElimTspOracle() {
  std::mt19937 random(20260902U); // NOLINT(bugprone-random-generator-seed): 回归必须可复现。
  std::uniform_int_distribution<int> cost_distribution(1, 1000);
  for (std::uint32_t k = 3; k <= 5; ++k) {
    const cudaee::KOptReconnectTable table = cudaee::BuildKOptReconnectTable(k);
    const std::size_t endpoint_count = std::size_t{2} * k;
    for (std::uint32_t trial = 0; trial < 2000; ++trial) {
      std::vector<std::vector<int>> matrix(endpoint_count, std::vector<int>(endpoint_count, 0));
      for (std::size_t first = 0; first < endpoint_count; ++first) {
        for (std::size_t second = first + 1; second < endpoint_count; ++second) {
          matrix[first][second] = matrix[second][first] = cost_distribution(random);
        }
      }

      int minimum_added_cost = std::numeric_limits<int>::max();
      for (const cudaee::EndpointMatching& matching : table.templates) {
        int added_cost = 0;
        for (std::uint32_t endpoint = 0; endpoint < matching.endpoint_count; ++endpoint) {
          const std::uint32_t partner = matching.mate[endpoint];
          if (endpoint < partner) {
            added_cost += matrix[endpoint][partner];
          }
        }
        minimum_added_cost = std::min(minimum_added_cost, added_cost);
      }
      const int delta = static_cast<int>(trial % 5U) - 2;
      const int target_deleted_cost = std::max(static_cast<int>(k), minimum_added_cost + delta);
      const int quotient = target_deleted_cost / static_cast<int>(k);
      const int remainder = target_deleted_cost % static_cast<int>(k);
      for (std::uint32_t edge = 0; edge < k; ++edge) {
        const std::size_t first_endpoint = std::size_t{2} * edge;
        const int value = quotient + (static_cast<int>(edge) < remainder ? 1 : 0);
        matrix[first_endpoint][first_endpoint + 1] = matrix[first_endpoint + 1][first_endpoint] =
            value;
      }

      std::vector<int*> rows;
      rows.reserve(endpoint_count);
      for (std::vector<int>& row : matrix) {
        rows.push_back(row.data());
      }

      int oracle_good = 0;
      std::array<int, 10> oracle_added{};
      if (k == 3) {
        CCelim_compare_three_swap(0, 1, 2, 3, 4, 5, rows.data(), &oracle_good, oracle_added.data());
      } else if (k == 4) {
        CCelim_compare_four_swap(0, 1, 2, 3, 4, 5, 6, 7, rows.data(), &oracle_good,
                                 oracle_added.data());
      } else {
        CCelim_compare_five_swap(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, rows.data(), &oracle_good,
                                 oracle_added.data());
      }

      int deleted_cost = 0;
      for (std::uint32_t edge = 0; edge < k; ++edge) {
        const std::size_t first_endpoint = std::size_t{2} * edge;
        deleted_cost += matrix[first_endpoint][first_endpoint + 1];
      }
      const bool generated_good = minimum_added_cost < deleted_cost;
      Check(generated_good == (oracle_good != 0),
            "generated reconnect templates equal ElimTSP swap.c oracle");
      if (oracle_good != 0) {
        cudaee::EndpointMatching oracle_matching;
        oracle_matching.endpoint_count = static_cast<std::uint8_t>(endpoint_count);
        oracle_matching.mate.fill(cudaee::kUnmatchedEndpoint);
        for (std::uint32_t edge = 0; edge < k; ++edge) {
          const std::size_t first_endpoint = std::size_t{2} * edge;
          const auto first = static_cast<std::uint8_t>(oracle_added[first_endpoint]);
          const auto second = static_cast<std::uint8_t>(oracle_added[first_endpoint + 1]);
          oracle_matching.mate[first] = second;
          oracle_matching.mate[second] = first;
        }
        Check(std::find(table.templates.begin(), table.templates.end(), oracle_matching) !=
                  table.templates.end(),
              "ElimTSP oracle witness belongs to generated template set");
      }
    }
  }
}

void TestImprovingWitnessAndProof() {
  // 原巡回 0-1-2-3-4-0 长于 proper 3-opt 巡回 0-4-1-3-2-0。
  const cudaee::GraphSnapshot graph = MakeGraph(
      {{0.0, 0.0, 0, 0}, {2.0, 1.0, 2, 1}, {0.0, 1.0, 0, 1}, {1.0, 2.0, 1, 2}, {1.0, 0.0, 1, 0}});
  const cudaee::NormalizedPathSystem paths =
      cudaee::NormalizePathSystem({{0, 1, 2, 3, 4}}, graph.dimension);
  Check(paths.valid, paths.reason);
  const cudaee::EndpointMatching outside = cudaee::EnumerateOutsideMatchings(1).front();
  const std::optional<cudaee::NodeEdge> required = cudaee::NodeEdge{0, 1};

  const cudaee::KOptSearchResult search =
      cudaee::FindKOptWitness(graph, paths, outside, required, {.max_k = 3});
  Check(search.status == cudaee::KOptSearchStatus::kImproved, search.reason);
  Check(search.witness.k == 3, "expected 3-opt witness");
  Check(search.witness.added_cost < search.witness.deleted_cost, "strict improvement");
  Check(std::find(search.witness.deleted_edges.begin(), search.witness.deleted_edges.end(),
                  cudaee::NodeEdge{0, 1}) != search.witness.deleted_edges.end(),
        "required edge deleted");
  std::string reason;
  Check(cudaee::VerifyKOptWitness(graph, paths, outside, required, search.witness, &reason),
        reason);

  cudaee::KOptWitness tampered = search.witness;
  ++tampered.added_cost;
  Check(!cudaee::VerifyKOptWitness(graph, paths, outside, required, tampered, &reason),
        "tampered cost rejected");

  const cudaee::PathSystemKOptProof proof =
      cudaee::ProvePathSystemByKOpt(graph, paths, required, {.max_k = 3});
  Check(proof.proven, proof.reason);
  Check(proof.records.size() == 1, "m=1 proof record count");
  Check(cudaee::VerifyPathSystemKOptProof(graph, paths, required, proof, &reason), reason);

  const std::filesystem::path proof_path =
      std::filesystem::path(CUDAEE_KOPT_TEST_TMP_DIR) / "tiny.path-kopt-proof";
  std::filesystem::create_directories(proof_path.parent_path());
  cudaee::WritePathSystemKOptProof(proof_path, proof);
  const cudaee::PathSystemKOptProof loaded = cudaee::ReadPathSystemKOptProof(proof_path);
  Check(cudaee::VerifyPathSystemKOptProof(graph, paths, required, loaded, &reason), reason);
  Check(loaded.records.size() == proof.records.size(), "path k-opt proof round trip");

  cudaee::PathSystemKOptProof wrong_hash = proof;
  ++wrong_hash.snapshot_hash;
  Check(!cudaee::VerifyPathSystemKOptProof(graph, paths, required, wrong_hash, &reason),
        "snapshot hash tamper rejected");
}

void TestNoImprovementAndBudget() {
  const cudaee::GraphSnapshot graph = MakeGraph(
      {{0.0, 0.0, 0, 0}, {1.0, 0.0, 1, 0}, {2.0, 0.0, 2, 0}, {3.0, 0.0, 3, 0}, {4.0, 0.0, 4, 0}});
  const cudaee::NormalizedPathSystem paths =
      cudaee::NormalizePathSystem({{0, 1, 2, 3, 4}}, graph.dimension);
  const cudaee::EndpointMatching outside = cudaee::EnumerateOutsideMatchings(1).front();
  const cudaee::KOptSearchResult no_improvement =
      cudaee::FindKOptWitness(graph, paths, outside, cudaee::NodeEdge{0, 1}, {.max_k = 3});
  Check(no_improvement.status == cudaee::KOptSearchStatus::kNoImprovement,
        "optimal collinear tour has no strict 3-opt improvement");

  const cudaee::KOptSearchResult unresolved = cudaee::FindKOptWitness(
      graph, paths, outside, cudaee::NodeEdge{0, 1}, {.max_k = 3, .max_deletion_sets = 1});
  Check(unresolved.status == cudaee::KOptSearchStatus::kUnresolved,
        "budget exhaustion remains unresolved");

  const cudaee::KOptSearchResult missing =
      cudaee::FindKOptWitness(graph, paths, outside, cudaee::NodeEdge{0, 4}, {.max_k = 3});
  Check(missing.status == cudaee::KOptSearchStatus::kInvalid,
        "required outside edge cannot be deleted as path edge");

  const cudaee::GraphSnapshot seven_node_graph = MakeGraph({{0.0, 0.0, 0, 0},
                                                            {1.0, 0.0, 1, 0},
                                                            {2.0, 0.0, 2, 0},
                                                            {3.0, 0.0, 3, 0},
                                                            {4.0, 0.0, 4, 0},
                                                            {5.0, 0.0, 5, 0},
                                                            {6.0, 0.0, 6, 0}});
  const cudaee::NormalizedPathSystem seven_node_paths =
      cudaee::NormalizePathSystem({{0, 1, 2, 3, 4, 5, 6}}, seven_node_graph.dimension);
  const cudaee::KOptSearchResult exhaustive = cudaee::FindKOptWitness(
      seven_node_graph, seven_node_paths, outside, cudaee::NodeEdge{0, 1}, {.max_k = 5});
  Check(exhaustive.status == cudaee::KOptSearchStatus::kNoImprovement,
        "collinear tour has no strict 3/4/5-opt improvement");
  Check(exhaustive.deletion_sets_tested == 25, "all anchored 3/4/5 deletion sets tested");
  Check(exhaustive.reconnect_matchings_tested == 1330,
        "all proper 3/4/5 reconnect templates tested");
}

void TestTwoPathCoverageProof() {
  const cudaee::GraphSnapshot graph = MakeGraph({{0.0, 0.0, 0, 0},
                                                 {100.0, 0.0, 100, 0},
                                                 {0.0, 1.0, 0, 1},
                                                 {1.0, 0.0, 1, 0},
                                                 {1.0, 1.0, 1, 1},
                                                 {2.0, 0.0, 2, 0}});
  const cudaee::NormalizedPathSystem paths =
      cudaee::NormalizePathSystem({{0, 1, 2}, {3, 4, 5}}, graph.dimension);
  Check(paths.valid, paths.reason);
  const std::optional<cudaee::NodeEdge> required = cudaee::NodeEdge{0, 1};
  const cudaee::PathSystemKOptProof proof =
      cudaee::ProvePathSystemByKOpt(graph, paths, required, {.max_k = 3});
  Check(proof.proven, proof.reason);
  Check(proof.outside_count == 2, "m=2 outside count");
  std::string reason;
  Check(cudaee::VerifyPathSystemKOptProof(graph, paths, required, proof, &reason), reason);
}

} // namespace

int main() {
  try {
    TestReconnectTemplateGeneration();
    TestReconnectTemplatesAgainstElimTspOracle();
    TestImprovingWitnessAndProof();
    TestNoImprovementAndBudget();
    TestTwoPathCoverageProof();
    std::cout << "k-opt tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
