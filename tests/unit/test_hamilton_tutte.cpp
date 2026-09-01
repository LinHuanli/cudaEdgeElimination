#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("test failure: " + message);
  }
}

cudaee::GraphSnapshot MakeGraph(const std::vector<cudaee::Point>& points,
                                const std::vector<std::pair<std::int32_t, std::int32_t>>& edges) {
  cudaee::GraphSnapshot graph;
  graph.dimension = static_cast<std::int32_t>(points.size());
  graph.distance_type = cudaee::DistanceType::kEuc2D;
  graph.integer_coordinates = true;
  graph.integer_distance_safe = true;
  graph.points = points;
  for (auto [first, second] : edges) {
    if (first > second) {
      std::swap(first, second);
    }
    graph.edges.push_back({first, second, graph.Distance(first, second), true});
  }
  std::sort(graph.edges.begin(), graph.edges.end(),
            [](const cudaee::Edge& lhs, const cudaee::Edge& rhs) {
              return std::pair{lhs.u, lhs.v} < std::pair{rhs.u, rhs.v};
            });
  graph.RebuildCsr();
  return graph;
}

cudaee::GraphSnapshot MakeCompleteGraph(const std::vector<cudaee::Point>& points) {
  std::vector<std::pair<std::int32_t, std::int32_t>> edges;
  for (std::int32_t first = 0; first < static_cast<std::int32_t>(points.size()); ++first) {
    for (std::int32_t second = first + 1; second < static_cast<std::int32_t>(points.size());
         ++second) {
      edges.emplace_back(first, second);
    }
  }
  return MakeGraph(points, edges);
}

bool ReferenceCompatible(const cudaee::GraphSnapshot& graph, const cudaee::NodeEdge target,
                         const std::int32_t center, const std::int32_t neighbor) {
  const std::int64_t original =
      graph.Distance(target.u, target.v) + graph.Distance(center, neighbor);
  return graph.Distance(target.u, center) + graph.Distance(target.v, neighbor) >= original ||
         graph.Distance(target.v, center) + graph.Distance(target.u, neighbor) >= original;
}

std::vector<cudaee::HtNeighborPair> ReferenceReplies(const cudaee::GraphSnapshot& graph,
                                                     const cudaee::NodeEdge target,
                                                     const std::int32_t center) {
  std::vector<cudaee::HtNeighborPair> replies;
  const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(center)];
  const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(center) + 1U];
  for (std::int32_t first_offset = begin; first_offset < end; ++first_offset) {
    const std::int32_t first = graph.neighbors[static_cast<std::size_t>(first_offset)];
    if (!ReferenceCompatible(graph, target, center, first)) {
      continue;
    }
    for (std::int32_t second_offset = first_offset + 1; second_offset < end; ++second_offset) {
      const std::int32_t second = graph.neighbors[static_cast<std::size_t>(second_offset)];
      if (!ReferenceCompatible(graph, target, center, second) ||
          ((first == target.u && second == target.v) ||
           (first == target.v && second == target.u))) {
        continue;
      }
      const std::int64_t original = graph.Distance(target.u, target.v) +
                                    graph.Distance(center, first) + graph.Distance(center, second);
      const std::int64_t replacement = graph.Distance(first, second) +
                                       graph.Distance(target.u, center) +
                                       graph.Distance(target.v, center);
      if (original <= replacement) {
        replies.push_back({center, first, second});
      }
    }
  }
  return replies;
}

void TestHamiltonRepliesAgainstReferenceFormula() {
  std::mt19937 random(20260903U); // NOLINT(bugprone-random-generator-seed): 固定差分种子。
  std::uniform_int_distribution<std::int64_t> coordinate(-100, 100);
  for (std::uint32_t trial = 0; trial < 30; ++trial) {
    std::vector<cudaee::Point> points;
    for (std::uint32_t node = 0; node < 8; ++node) {
      const std::int64_t x = coordinate(random);
      const std::int64_t y = coordinate(random);
      points.push_back({static_cast<double>(x), static_cast<double>(y), x, y});
    }
    const cudaee::GraphSnapshot graph = MakeCompleteGraph(points);
    for (std::int32_t first = 0; first < graph.dimension; ++first) {
      for (std::int32_t second = first + 1; second < graph.dimension; ++second) {
        const cudaee::NodeEdge target{first, second};
        for (std::int32_t center = 0; center < graph.dimension; ++center) {
          if (center == first || center == second) {
            continue;
          }
          const std::vector<cudaee::HtNeighborPair> expected =
              ReferenceReplies(graph, target, center);
          const std::vector<cudaee::HtNeighborPair> actual =
              cudaee::EnumerateHtHamiltonReplies(graph, target, center);
          Check(actual == expected, "Hamilton replies equal LocalElimination formulas");
        }
      }
    }
  }
}

void TestCdCandidatesCpuCuda() {
#ifdef CUDAEE_HAS_CUDA
  std::string unavailable_reason;
  if (!cudaee::detail::HtCdCudaAvailable(&unavailable_reason)) {
    std::cout << "CUDA HT c,d skipped: " << unavailable_reason << '\n';
    return;
  }
  std::vector<cudaee::Point> points;
  for (std::int32_t node = 0; node < 32; ++node) {
    const std::int64_t x = (37 * node + 11) % 101;
    const std::int64_t y = (node * node * 13 + 7 * node) % 97;
    points.push_back({static_cast<double>(x), static_cast<double>(y), x, y});
  }
  std::vector<std::pair<std::int32_t, std::int32_t>> sparse_edges;
  for (std::int32_t first = 0; first < 32; ++first) {
    for (std::int32_t second = first + 1; second < 32; ++second) {
      if ((first == 0 && second == 1) || (17 * first + 13 * second) % 5 == 0) {
        sparse_edges.emplace_back(first, second);
      }
    }
  }
  cudaee::GraphSnapshot graph = MakeGraph(points, sparse_edges);
  for (const cudaee::DistanceType distance_type :
       {cudaee::DistanceType::kEuc2D, cudaee::DistanceType::kCeil2D}) {
    graph.distance_type = distance_type;
    const cudaee::HtShallowOptions cpu_options = {
        .max_neighborhood = 0,
        .max_cd_candidates = 0,
        .max_candidate_degree = 0,
        .cd_mode = cudaee::HtCdMode::kMissingOrIncompatible,
        .candidate_backend = cudaee::PathCompatibilityBackend::kCpu};
    cudaee::HtShallowOptions gpu_options = cpu_options;
    gpu_options.candidate_backend = cudaee::PathCompatibilityBackend::kCuda;
    const cudaee::HtCdBatchResult cpu = cudaee::EvaluateHtCdCandidates(graph, {0, 1}, cpu_options);
    const cudaee::HtCdBatchResult gpu = cudaee::EvaluateHtCdCandidates(graph, {0, 1}, gpu_options);
    Check(cpu.backend == "cpu" && cpu.cpu_verified, "CPU c,d batch metadata");
    Check(gpu.backend == "cuda" && gpu.selected_device >= 0 && gpu.cpu_verified,
          "CUDA c,d batch metadata");
    Check(gpu.candidates == cpu.candidates, "CPU/CUDA c,d candidates are exact");
  }
#endif
}

void TestVacuousAndProofAndTamperRejection() {
  const std::vector<cudaee::Point> points = {
      {0.0, 0.0, 0, 0},       {100.0, 0.0, 100, 0},   {50.0, -20.0, 50, -20}, {50.0, 20.0, 50, 20},
      {50.0, -30.0, 50, -30}, {50.0, -40.0, 50, -40}, {50.0, 30.0, 50, 30},   {50.0, 40.0, 50, 40}};
  const cudaee::GraphSnapshot graph =
      MakeGraph(points, {{0, 1}, {2, 3}, {2, 4}, {2, 5}, {3, 6}, {3, 7}});
  const cudaee::HtShallowResult result =
      cudaee::ProveEdgeByShallowHt(graph, {0, 1},
                                   {.max_neighborhood = 0,
                                    .max_cd_candidates = 0,
                                    .max_candidate_degree = 0,
                                    .cd_mode = cudaee::HtCdMode::kActiveIncompatible});
  Check(result.status == cudaee::HtSearchStatus::kProven, result.proof.reason);
  Check(result.proof.replies.empty(), "zero surviving Hamilton reply closes the AND node");
  std::string reason;
  Check(cudaee::VerifyHtShallowProof(graph, result.proof, &reason), reason);

  cudaee::HtShallowProof wrong_hash = result.proof;
  ++wrong_hash.snapshot_hash;
  Check(!cudaee::VerifyHtShallowProof(graph, wrong_hash, &reason),
        "snapshot hash tamper is rejected");
  cudaee::HtShallowProof wrong_move = result.proof;
  wrong_move.c = 0;
  Check(!cudaee::VerifyHtShallowProof(graph, wrong_move, &reason), "invalid c,d move is rejected");
}

void TestNonemptyAndProof() {
  const cudaee::GraphSnapshot graph = MakeCompleteGraph({{-11.0, -13.0, -11, -13},
                                                         {25.0, 30.0, 25, 30},
                                                         {-36.0, 18.0, -36, 18},
                                                         {-50.0, -53.0, -50, -53},
                                                         {-51.0, 33.0, -51, 33},
                                                         {-8.0, 20.0, -8, 20},
                                                         {-11.0, 24.0, -11, 24},
                                                         {0.0, -46.0, 0, -46}});
  const cudaee::HtShallowResult result = cudaee::ProveEdgeByShallowHt(
      graph, {0, 5},
      {.max_neighborhood = 0,
       .max_cd_candidates = 20,
       .max_candidate_degree = 0,
       .max_reply_combinations = 10000,
       .cd_mode = cudaee::HtCdMode::kActiveIncompatible,
       .leaf_options = {.max_k = 3, .max_deletion_sets = 1, .exact_fallback_max_blocks = 10}});
  Check(result.status == cudaee::HtSearchStatus::kProven, result.proof.reason);
  Check(result.proof.c == 1 && result.proof.d == 2, "pinned c,d move is deterministic");
  Check(result.proof.replies.size() == 30, "pinned c,d move covers all 30 replies");
  Check(std::any_of(result.proof.replies.begin(), result.proof.replies.end(),
                    [](const cudaee::HtReplyProof& reply) { return !reply.path_infeasible; }),
        "nonempty proof contains a leaf certificate");
  std::string reason;
  Check(cudaee::VerifyHtShallowProof(graph, result.proof, &reason), reason);
  Check(result.proof.reply_combinations_tested >= result.proof.replies.size(),
        "search records reply work");

  cudaee::HtShallowProof missing_reply = result.proof;
  missing_reply.replies.pop_back();
  Check(!cudaee::VerifyHtShallowProof(graph, missing_reply, &reason),
        "missing AND reply is rejected");
  cudaee::HtShallowProof wrong_leaf = result.proof;
  const auto leaf =
      std::find_if(wrong_leaf.replies.begin(), wrong_leaf.replies.end(),
                   [](const cudaee::HtReplyProof& reply) { return !reply.path_infeasible; });
  Check(leaf != wrong_leaf.replies.end(), "nonempty proof contains a leaf certificate");
  ++leaf->leaf_proof.snapshot_hash;
  Check(!cudaee::VerifyHtShallowProof(graph, wrong_leaf, &reason),
        "tampered leaf binding is rejected");

  const cudaee::HtShallowResult budget =
      cudaee::ProveEdgeByShallowHt(graph, {0, 5},
                                   {.max_neighborhood = 0,
                                    .max_cd_candidates = 1,
                                    .max_candidate_degree = 0,
                                    .max_reply_combinations = 1,
                                    .cd_mode = cudaee::HtCdMode::kActiveIncompatible,
                                    .leaf_options = {.max_k = 3, .exact_fallback_max_blocks = 10}});
  Check(budget.status == cudaee::HtSearchStatus::kUnresolved,
        "reply budget exhaustion remains unresolved");

#ifdef CUDAEE_HAS_CUDA
  std::string cuda_reason;
  if (cudaee::detail::HtCdCudaAvailable(&cuda_reason)) {
    const cudaee::HtShallowResult gpu_result = cudaee::ProveEdgeByShallowHt(
        graph, {0, 5},
        {.max_neighborhood = 0,
         .max_cd_candidates = 20,
         .max_candidate_degree = 0,
         .max_reply_combinations = 10000,
         .cd_mode = cudaee::HtCdMode::kActiveIncompatible,
         .candidate_backend = cudaee::PathCompatibilityBackend::kCuda,
         .leaf_options = {.max_k = 3, .max_deletion_sets = 1, .exact_fallback_max_blocks = 10}});
    Check(gpu_result.status == cudaee::HtSearchStatus::kProven, gpu_result.proof.reason);
    Check(cudaee::VerifyHtShallowProof(graph, gpu_result.proof, &reason), reason);
  }
#endif
}

} // namespace

int main() {
  try {
    TestHamiltonRepliesAgainstReferenceFormula();
    TestCdCandidatesCpuCuda();
    TestVacuousAndProofAndTamperRejection();
    TestNonemptyAndProof();
    std::cout << "Hamilton-Tutte tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
