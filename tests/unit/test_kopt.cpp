#include "cuda_edge_elimination/local_search.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
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

cudaee::NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? cudaee::NodeEdge{first, second} : cudaee::NodeEdge{second, first};
}

std::vector<cudaee::NodeEdge> OutsideEdges(const cudaee::NormalizedPathSystem& paths,
                                           const cudaee::EndpointMatching& outside) {
  std::vector<std::int32_t> endpoints;
  endpoints.reserve(2U * paths.paths.size());
  for (const cudaee::Path& path : paths.paths) {
    endpoints.push_back(path.front());
    endpoints.push_back(path.back());
  }
  std::vector<cudaee::NodeEdge> edges;
  for (std::uint32_t endpoint = 0; endpoint < outside.endpoint_count; ++endpoint) {
    const std::uint32_t partner = outside.mate[endpoint];
    if (endpoint < partner) {
      edges.push_back(CanonicalEdge(endpoints[endpoint], endpoints[partner]));
    }
  }
  return edges;
}

std::int64_t OriginalTourCost(const cudaee::GraphSnapshot& graph,
                              const cudaee::NormalizedPathSystem& paths,
                              const cudaee::EndpointMatching& outside) {
  std::int64_t cost = 0;
  for (const cudaee::Path& path : paths.paths) {
    for (std::size_t index = 1; index < path.size(); ++index) {
      cost += graph.Distance(path[index - 1], path[index]);
    }
  }
  for (const cudaee::NodeEdge& edge : OutsideEdges(paths, outside)) {
    cost += graph.Distance(edge.u, edge.v);
  }
  return cost;
}

// 测试 oracle 直接枚举巡回，不复用生产 DP 的 block 状态或 predecessor。
std::int64_t BruteForceConstrainedTourCost(const cudaee::GraphSnapshot& graph,
                                           const cudaee::NormalizedPathSystem& paths,
                                           const cudaee::EndpointMatching& outside,
                                           const cudaee::NodeEdge& forbidden) {
  std::vector<std::int32_t> nodes;
  for (const cudaee::Path& path : paths.paths) {
    nodes.insert(nodes.end(), path.begin(), path.end());
  }
  std::sort(nodes.begin(), nodes.end());
  const std::int32_t start = nodes.front();
  std::vector<std::int32_t> tail(nodes.begin() + 1, nodes.end());
  const std::vector<cudaee::NodeEdge> forced = OutsideEdges(paths, outside);
  const cudaee::NodeEdge canonical_forbidden = CanonicalEdge(forbidden.u, forbidden.v);
  std::int64_t best = std::numeric_limits<std::int64_t>::max();
  do {
    std::vector<std::int32_t> order = {start};
    order.insert(order.end(), tail.begin(), tail.end());
    std::vector<cudaee::NodeEdge> tour_edges;
    tour_edges.reserve(order.size());
    std::int64_t cost = 0;
    bool valid = true;
    for (std::size_t index = 0; index < order.size(); ++index) {
      const std::int32_t first = order[index];
      const std::int32_t second = order[(index + 1) % order.size()];
      const cudaee::NodeEdge edge = CanonicalEdge(first, second);
      if (edge == canonical_forbidden) {
        valid = false;
        break;
      }
      tour_edges.push_back(edge);
      cost += graph.Distance(first, second);
    }
    if (!valid) {
      continue;
    }
    for (const cudaee::NodeEdge& edge : forced) {
      if (std::find(tour_edges.begin(), tour_edges.end(), edge) == tour_edges.end()) {
        valid = false;
        break;
      }
    }
    if (valid) {
      best = std::min(best, cost);
    }
  } while (std::next_permutation(tail.begin(), tail.end()));
  return best;
}

void TestKOptCostMatrixCpuCuda() {
  std::vector<cudaee::Point> points;
  for (std::int64_t node = 0; node < 17; ++node) {
    const std::int64_t x = (7 * node) % 23;
    const std::int64_t y = (node * node + 3 * node) % 19;
    points.push_back({static_cast<double>(x), static_cast<double>(y), static_cast<std::int64_t>(x),
                      static_cast<std::int64_t>(y)});
  }
  const cudaee::GraphSnapshot graph = MakeGraph(points);
  constexpr std::array<std::size_t, 3> kExpectedTemplateCounts = {4, 25, 208};
  for (std::uint32_t k = 3; k <= 5; ++k) {
    std::vector<cudaee::KOptCostTask> tasks(3);
    for (std::uint32_t port = 0; port < 2U * k; ++port) {
      tasks[0].port_nodes[port] = static_cast<std::int32_t>(port);
      tasks[2].port_nodes[port] = static_cast<std::int32_t>((3U * port + 2U) % 17U);
    }
    const std::array<std::int32_t, 10> adjacent_ports = {0, 1, 1, 2, 3, 4, 5, 6, 7, 8};
    tasks[1].port_nodes = adjacent_ports;
    for (cudaee::KOptCostTask& task : tasks) {
      for (std::uint32_t edge = 0; edge < k; ++edge) {
        const std::size_t first_port = std::size_t{2} * edge;
        task.deleted_cost +=
            graph.Distance(task.port_nodes[first_port], task.port_nodes[first_port + 1]);
      }
    }

    const cudaee::KOptCostBatchResult cpu =
        cudaee::EvaluateKOptTemplateCosts(graph, k, tasks, cudaee::PathCompatibilityBackend::kCpu);
    Check(cpu.backend == "cpu", "CPU k-opt cost backend");
    Check(cpu.template_count == kExpectedTemplateCounts[static_cast<std::size_t>(k - 3U)],
          "k-opt cost template count");
    Check(cpu.added_costs.size() == tasks.size() * cpu.template_count,
          "CPU k-opt cost matrix shape");

#ifdef CUDAEE_HAS_CUDA
    std::string unavailable_reason;
    if (cudaee::detail::KOptCostCudaAvailable(&unavailable_reason)) {
      const cudaee::KOptCostBatchResult gpu = cudaee::EvaluateKOptTemplateCosts(
          graph, k, tasks, cudaee::PathCompatibilityBackend::kCuda);
      Check(gpu.backend == "cuda", "CUDA k-opt cost backend");
      Check(gpu.selected_device >= 0, "CUDA k-opt cost selected device");
      Check(gpu.added_costs == cpu.added_costs, "CPU/CUDA k-opt cost matrices are exact");
    } else {
      std::cout << "CUDA k-opt cost skipped: " << unavailable_reason << '\n';
    }
#endif
  }

  cudaee::GraphSnapshot ceil_graph = graph;
  ceil_graph.distance_type = cudaee::DistanceType::kCeil2D;
  cudaee::KOptCostTask ceil_task;
  for (std::uint32_t port = 0; port < 10; ++port) {
    ceil_task.port_nodes[port] = static_cast<std::int32_t>(port);
  }
  const std::vector<cudaee::KOptCostTask> ceil_tasks = {ceil_task};
  const cudaee::KOptCostBatchResult ceil_cpu = cudaee::EvaluateKOptTemplateCosts(
      ceil_graph, 5, ceil_tasks, cudaee::PathCompatibilityBackend::kCpu);
#ifdef CUDAEE_HAS_CUDA
  std::string ceil_unavailable_reason;
  if (cudaee::detail::KOptCostCudaAvailable(&ceil_unavailable_reason)) {
    const cudaee::KOptCostBatchResult ceil_gpu = cudaee::EvaluateKOptTemplateCosts(
        ceil_graph, 5, ceil_tasks, cudaee::PathCompatibilityBackend::kCuda);
    Check(ceil_gpu.added_costs == ceil_cpu.added_costs,
          "CPU/CUDA CEIL_2D k-opt cost matrices are exact");
  }
#endif
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

  const cudaee::KOptSearchResult exact =
      cudaee::FindExactTourWitness(graph, paths, outside, required, 10);
  Check(exact.status == cudaee::KOptSearchStatus::kImproved, exact.reason);
  Check(exact.exact_states_tested > 0, "exact fallback visits DP states");
  Check(cudaee::VerifyKOptWitness(graph, paths, outside, required, exact.witness, &reason), reason);

  const cudaee::KOptSearchResult batched = cudaee::FindKOptWitness(
      graph, paths, outside, required,
      {.max_k = 3, .cost_backend = cudaee::PathCompatibilityBackend::kAuto, .cost_batch_size = 2});
  Check(batched.status == cudaee::KOptSearchStatus::kImproved,
        "batched cost oracle finds a CPU-verifiable witness");
  Check(cudaee::VerifyKOptWitness(graph, paths, outside, required, batched.witness, &reason),
        reason);

#ifdef CUDAEE_HAS_CUDA
  std::string cuda_reason;
  if (cudaee::detail::KOptCostCudaAvailable(&cuda_reason)) {
    const cudaee::KOptSearchResult cuda_search =
        cudaee::FindKOptWitness(graph, paths, outside, required,
                                {.max_k = 3,
                                 .cost_backend = cudaee::PathCompatibilityBackend::kCuda,
                                 .cost_batch_size = 2});
    Check(cuda_search.status == cudaee::KOptSearchStatus::kImproved,
          "CUDA cost oracle finds a CPU-verifiable witness");
    Check(cudaee::VerifyKOptWitness(graph, paths, outside, required, cuda_search.witness, &reason),
          reason);
  }
#endif

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

  const cudaee::PathSystemKOptProof exact_proof = cudaee::ProvePathSystemByKOpt(
      graph, paths, required,
      {.max_k = 3, .max_deletion_sets = 1, .exact_fallback_max_blocks = 10});
  Check(exact_proof.proven, exact_proof.reason);
  Check(exact_proof.deletion_sets_tested == 1, "k-opt budget is exhausted before exact fallback");
  Check(exact_proof.exact_states_tested > 0, "proof records exact fallback work");
  Check(cudaee::VerifyPathSystemKOptProof(graph, paths, required, exact_proof, &reason), reason);
  const std::filesystem::path exact_proof_path =
      std::filesystem::path(CUDAEE_KOPT_TEST_TMP_DIR) / "tiny-exact.path-kopt-proof";
  cudaee::WritePathSystemKOptProof(exact_proof_path, exact_proof);
  const cudaee::PathSystemKOptProof loaded_exact =
      cudaee::ReadPathSystemKOptProof(exact_proof_path);
  Check(cudaee::VerifyPathSystemKOptProof(graph, paths, required, loaded_exact, &reason), reason);

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
  const cudaee::KOptSearchResult batched_no_improvement = cudaee::FindKOptWitness(
      graph, paths, outside, cudaee::NodeEdge{0, 1},
      {.max_k = 3, .cost_backend = cudaee::PathCompatibilityBackend::kAuto, .cost_batch_size = 2});
  Check(batched_no_improvement.status == cudaee::KOptSearchStatus::kNoImprovement,
        "batched candidate oracle falls back before concluding no improvement");

  const cudaee::KOptSearchResult exact_no_improvement =
      cudaee::FindExactTourWitness(graph, paths, outside, cudaee::NodeEdge{0, 1}, 10);
  Check(exact_no_improvement.status == cudaee::KOptSearchStatus::kNoImprovement,
        "exact fallback proves no strictly shorter constrained tour");
  Check(exact_no_improvement.exact_states_tested > 0, "exact no-improvement visits DP states");

  const cudaee::KOptSearchResult exact_too_large =
      cudaee::FindExactTourWitness(graph, paths, outside, cudaee::NodeEdge{0, 1}, 2);
  Check(exact_too_large.status == cudaee::KOptSearchStatus::kUnresolved,
        "exact fallback block cap remains unresolved");

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

  const cudaee::KOptSearchOptions cursor_options = {
      .max_k = 5, .cost_backend = cudaee::PathCompatibilityBackend::kAuto, .cost_batch_size = 2};
  const cudaee::PathSystemKOptProof scalar_cursor = cudaee::ProvePathSystemByKOpt(
      seven_node_graph, seven_node_paths, cudaee::NodeEdge{0, 1}, cursor_options);
  const cudaee::PathSystemKOptBatchResult cursor_batch =
      cudaee::ProvePathSystemsByKOpt(seven_node_graph, {seven_node_paths, seven_node_paths},
                                     cudaee::NodeEdge{0, 1}, cursor_options);
  Check(cursor_batch.cpu_verified && cursor_batch.scalar_searches == 0U &&
            cursor_batch.cost_tasks == 2U * scalar_cursor.deletion_sets_tested &&
            cursor_batch.cost_batches == 13U && cursor_batch.cost_tasks == 50U &&
            cursor_batch.cost_cells == 2660U &&
            cudaee::SerializePathSystemKOptProof(cursor_batch.proofs[0]) ==
                cudaee::SerializePathSystemKOptProof(scalar_cursor) &&
            cudaee::SerializePathSystemKOptProof(cursor_batch.proofs[1]) ==
                cudaee::SerializePathSystemKOptProof(scalar_cursor),
        "incremental leaf cursors fuse all 3/4/5 deletion blocks without changing proof bytes");

  cudaee::KOptSearchOptions budget_cursor_options = cursor_options;
  budget_cursor_options.max_deletion_sets = 3;
  const cudaee::PathSystemKOptProof scalar_budget_cursor = cudaee::ProvePathSystemByKOpt(
      seven_node_graph, seven_node_paths, cudaee::NodeEdge{0, 1}, budget_cursor_options);
  const cudaee::PathSystemKOptBatchResult budget_cursor_batch =
      cudaee::ProvePathSystemsByKOpt(seven_node_graph, {seven_node_paths, seven_node_paths},
                                     cudaee::NodeEdge{0, 1}, budget_cursor_options);
  Check(budget_cursor_batch.cost_batches == 2U && budget_cursor_batch.cost_tasks == 6U &&
            budget_cursor_batch.cost_cells == 24U &&
            cudaee::SerializePathSystemKOptProof(budget_cursor_batch.proofs[0]) ==
                cudaee::SerializePathSystemKOptProof(scalar_budget_cursor) &&
            cudaee::SerializePathSystemKOptProof(budget_cursor_batch.proofs[1]) ==
                cudaee::SerializePathSystemKOptProof(scalar_budget_cursor),
        "incremental leaf cursor preserves a budget boundary inside the second cost block");
}

void TestPathSystemLeafCostBatch() {
  const cudaee::GraphSnapshot graph = MakeGraph(
      {{0.0, 0.0, 0, 0}, {2.0, 1.0, 2, 1}, {0.0, 1.0, 0, 1}, {1.0, 2.0, 1, 2}, {1.0, 0.0, 1, 0}});
  const cudaee::NormalizedPathSystem paths =
      cudaee::NormalizePathSystem({{0, 1, 2, 3, 4}}, graph.dimension);
  Check(paths.valid, paths.reason);
  const std::optional<cudaee::NodeEdge> required = cudaee::NodeEdge{0, 1};
  const cudaee::KOptSearchOptions options = {.max_k = 3,
                                             .max_deletion_sets = 1,
                                             .cost_backend =
                                                 cudaee::PathCompatibilityBackend::kAuto,
                                             .cost_batch_size = 8,
                                             .exact_fallback_max_blocks = 10};
  const cudaee::PathSystemKOptProof scalar =
      cudaee::ProvePathSystemByKOpt(graph, paths, required, options);
  Check(scalar.proven, scalar.reason);

  const cudaee::PathSystemKOptBatchResult batch =
      cudaee::ProvePathSystemsByKOpt(graph, {paths, paths, paths}, required, options);
  Check(batch.cpu_verified && batch.proofs.size() == 3U && batch.cost_batches == 1U &&
            batch.cost_tasks == 3U && batch.cost_cells == 12U && batch.scalar_searches == 0U,
        "leaf cost batch fuses three first deletion-set rows");
  const std::string expected = cudaee::SerializePathSystemKOptProof(scalar);
  for (const cudaee::PathSystemKOptProof& proof : batch.proofs) {
    Check(cudaee::SerializePathSystemKOptProof(proof) == expected,
          "batched leaf proof is byte-identical to scalar search");
  }

  cudaee::KOptSearchOptions cpu_options = options;
  cpu_options.cost_backend = cudaee::PathCompatibilityBackend::kCpu;
  const cudaee::PathSystemKOptProof cpu_scalar =
      cudaee::ProvePathSystemByKOpt(graph, paths, required, cpu_options);
  const cudaee::PathSystemKOptBatchResult cpu_batch =
      cudaee::ProvePathSystemsByKOpt(graph, {paths, paths}, required, cpu_options);
  Check(cpu_batch.cost_backend == "cpu-scalar" && cpu_batch.cost_batches == 0U &&
            cpu_batch.scalar_searches > 0U &&
            cudaee::SerializePathSystemKOptProof(cpu_batch.proofs[0]) ==
                cudaee::SerializePathSystemKOptProof(cpu_scalar) &&
            cudaee::SerializePathSystemKOptProof(cpu_batch.proofs[1]) ==
                cudaee::SerializePathSystemKOptProof(cpu_scalar),
        "CPU leaf bucket keeps the scalar proof semantics");
}

void TestLeafCursorDifferential() {
  std::mt19937 random(24092026U); // NOLINT(bugprone-random-generator-seed): 固定差分种子。
  std::uniform_int_distribution<std::int64_t> coordinate(-40, 40);
  for (std::uint32_t trial = 0U; trial < 12U; ++trial) {
    std::vector<cudaee::Point> points;
    points.reserve(7U);
    for (std::uint32_t node = 0U; node < 7U; ++node) {
      const std::int64_t x = coordinate(random);
      const std::int64_t y = coordinate(random);
      points.push_back({static_cast<double>(x), static_cast<double>(y), x, y});
    }
    const cudaee::GraphSnapshot graph = MakeGraph(points);
    const std::vector<cudaee::NormalizedPathSystem> path_systems = {
        cudaee::NormalizePathSystem({{0, 1, 2, 3, 4, 5, 6}}, graph.dimension),
        cudaee::NormalizePathSystem({{0, 1, 3, 5, 2, 6, 4}}, graph.dimension),
        cudaee::NormalizePathSystem({{0, 1, 4}, {2, 5, 3, 6}}, graph.dimension)};
    Check(std::all_of(path_systems.begin(), path_systems.end(),
                      [](const cudaee::NormalizedPathSystem& paths) { return paths.valid; }),
          "random leaf cursor inputs are canonical");
    const std::uint64_t deletion_budget = trial % 4U == 0U ? 0U : 1U + trial % 7U;
    const cudaee::KOptSearchOptions options = {.max_k = 3U + trial % 3U,
                                               .max_deletion_sets = deletion_budget,
                                               .cost_backend =
                                                   cudaee::PathCompatibilityBackend::kAuto,
                                               .cost_batch_size = 1U + trial % 4U};
    const cudaee::PathSystemKOptBatchResult batch =
        cudaee::ProvePathSystemsByKOpt(graph, path_systems, cudaee::NodeEdge{0, 1}, options);
    Check(batch.cpu_verified && batch.proofs.size() == path_systems.size(),
          "random leaf cursor batch returns aligned CPU-verified proofs");
    for (std::size_t index = 0U; index < path_systems.size(); ++index) {
      const cudaee::PathSystemKOptProof scalar = cudaee::ProvePathSystemByKOpt(
          graph, path_systems[index], cudaee::NodeEdge{0, 1}, options);
      Check(cudaee::SerializePathSystemKOptProof(batch.proofs[index]) ==
                cudaee::SerializePathSystemKOptProof(scalar),
            "incremental cursor matches scalar proof on randomized paths and budgets");
    }
  }
}

void TestExactFallbackAgainstBruteForce() {
  std::mt19937 random(19870217U); // NOLINT(bugprone-random-generator-seed): 固定回归种子。
  std::uniform_int_distribution<std::int64_t> coordinate(-30, 30);
  for (std::uint32_t trial = 0; trial < 60; ++trial) {
    std::vector<cudaee::Point> points;
    points.reserve(7);
    for (std::uint32_t node = 0; node < 7; ++node) {
      const std::int64_t x = coordinate(random);
      const std::int64_t y = coordinate(random);
      points.push_back({static_cast<double>(x), static_cast<double>(y), x, y});
    }
    const cudaee::GraphSnapshot graph = MakeGraph(points);
    const cudaee::NormalizedPathSystem paths =
        cudaee::NormalizePathSystem({{0, 1, 2}, {3, 4, 5, 6}}, graph.dimension);
    Check(paths.valid, paths.reason);
    const cudaee::NodeEdge required{0, 1};
    for (const cudaee::EndpointMatching& outside : cudaee::EnumerateOutsideMatchings(2)) {
      const std::int64_t original = OriginalTourCost(graph, paths, outside);
      const std::int64_t oracle = BruteForceConstrainedTourCost(graph, paths, outside, required);
      Check(oracle != std::numeric_limits<std::int64_t>::max(),
            "brute-force constrained tour exists");
      const cudaee::KOptSearchResult exact =
          cudaee::FindExactTourWitness(graph, paths, outside, required, 10);
      if (oracle < original) {
        Check(exact.status == cudaee::KOptSearchStatus::kImproved,
              "exact fallback agrees with brute-force improvement");
        Check(original - exact.witness.deleted_cost + exact.witness.added_cost == oracle,
              "exact fallback equals brute-force optimum");
        std::string reason;
        Check(cudaee::VerifyKOptWitness(graph, paths, outside, required, exact.witness, &reason),
              reason);
      } else {
        Check(exact.status == cudaee::KOptSearchStatus::kNoImprovement,
              "exact fallback agrees with brute-force no-improvement");
      }
    }
  }
}

void TestExactSevenOptProofRoundTrip() {
  const cudaee::GraphSnapshot graph = MakeGraph({{10.0, 0.0, 10, 0},
                                                 {7.0, 7.0, 7, 7},
                                                 {0.0, 10.0, 0, 10},
                                                 {-7.0, 7.0, -7, 7},
                                                 {-10.0, 0.0, -10, 0},
                                                 {-7.0, -7.0, -7, -7},
                                                 {0.0, -10.0, 0, -10},
                                                 {7.0, -7.0, 7, -7}});
  const cudaee::NormalizedPathSystem paths =
      cudaee::NormalizePathSystem({{0, 2, 4, 6, 1, 3, 5, 7}}, graph.dimension);
  Check(paths.valid, paths.reason);
  const cudaee::EndpointMatching outside = cudaee::EnumerateOutsideMatchings(1).front();
  const cudaee::NodeEdge required{0, 2};
  const cudaee::KOptSearchResult exact =
      cudaee::FindExactTourWitness(graph, paths, outside, required, 10);
  Check(exact.status == cudaee::KOptSearchStatus::kImproved, exact.reason);
  Check(exact.witness.k == 7, "exact fallback emits a generic 7-opt witness");
  std::string reason;
  Check(cudaee::VerifyKOptWitness(graph, paths, outside, required, exact.witness, &reason), reason);

  cudaee::PathSystemKOptProof proof;
  proof.proven = true;
  proof.reason = "精确 7-opt 测试证明";
  proof.snapshot_hash = graph.ContentHash();
  proof.path_system_hash = cudaee::ComputePathSystemHash(paths);
  proof.compatibility_table_hash = cudaee::BuildPathCompatibilityTable(1).generator_hash;
  proof.path_count = 1;
  proof.outside_count = 1;
  proof.exact_states_tested = exact.exact_states_tested;
  proof.records.push_back({0, exact.witness});
  Check(cudaee::VerifyPathSystemKOptProof(graph, paths, required, proof, &reason), reason);

  const std::filesystem::path proof_path =
      std::filesystem::path(CUDAEE_KOPT_TEST_TMP_DIR) / "seven-opt.path-kopt-proof";
  cudaee::WritePathSystemKOptProof(proof_path, proof);
  const cudaee::PathSystemKOptProof loaded = cudaee::ReadPathSystemKOptProof(proof_path);
  Check(loaded.records.front().witness.k == 7, "V1 parser accepts independently verified k>5");
  Check(cudaee::VerifyPathSystemKOptProof(graph, paths, required, loaded, &reason), reason);
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

  const cudaee::KOptSearchOptions batch_options = {.max_k = 3,
                                                   .max_deletion_sets = 1,
                                                   .cost_backend =
                                                       cudaee::PathCompatibilityBackend::kAuto,
                                                   .cost_batch_size = 8,
                                                   .exact_fallback_max_blocks = 10};
  const cudaee::PathSystemKOptProof scalar_budgeted =
      cudaee::ProvePathSystemByKOpt(graph, paths, required, batch_options);
  const cudaee::PathSystemKOptBatchResult batch =
      cudaee::ProvePathSystemsByKOpt(graph, {paths, paths}, required, batch_options);
  Check(batch.cpu_verified && batch.proofs.size() == 2U && batch.cost_batches > 0U &&
            batch.cost_tasks >= 2U &&
            cudaee::SerializePathSystemKOptProof(batch.proofs[0]) ==
                cudaee::SerializePathSystemKOptProof(scalar_budgeted) &&
            cudaee::SerializePathSystemKOptProof(batch.proofs[1]) ==
                cudaee::SerializePathSystemKOptProof(scalar_budgeted),
        "two-path leaf batching preserves outside coverage and proof bytes");
}

} // namespace

int main() {
  try {
    TestKOptCostMatrixCpuCuda();
    TestReconnectTemplateGeneration();
    TestReconnectTemplatesAgainstElimTspOracle();
    TestImprovingWitnessAndProof();
    TestNoImprovementAndBudget();
    TestPathSystemLeafCostBatch();
    TestLeafCursorDifferential();
    TestExactFallbackAgainstBruteForce();
    TestExactSevenOptProofRoundTrip();
    TestTwoPathCoverageProof();
    std::cout << "k-opt tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
