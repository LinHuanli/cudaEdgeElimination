#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
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

template <typename Callback> void CheckThrows(Callback&& callback, const std::string& message) {
  try {
    callback();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("test failure: " + message);
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

cudaee::NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? cudaee::NodeEdge{first, second} : cudaee::NodeEdge{second, first};
}

std::pair<std::int64_t, std::int64_t> BruteForceTourCosts(const cudaee::GraphSnapshot& graph,
                                                          const cudaee::NodeEdge raw_target) {
  const cudaee::NodeEdge target = CanonicalEdge(raw_target.u, raw_target.v);
  std::vector<std::int32_t> tail(static_cast<std::size_t>(graph.dimension - 1));
  for (std::int32_t node = 1; node < graph.dimension; ++node) {
    tail[static_cast<std::size_t>(node - 1)] = node;
  }
  std::int64_t best = std::numeric_limits<std::int64_t>::max();
  std::int64_t best_with_target = std::numeric_limits<std::int64_t>::max();
  do {
    std::vector<std::int32_t> order = {0};
    order.insert(order.end(), tail.begin(), tail.end());
    std::int64_t cost = 0;
    bool contains_target = false;
    for (std::size_t index = 0; index < order.size(); ++index) {
      const std::int32_t first = order[index];
      const std::int32_t second = order[(index + 1U) % order.size()];
      cost += graph.Distance(first, second);
      contains_target = contains_target || CanonicalEdge(first, second) == target;
    }
    best = std::min(best, cost);
    if (contains_target) {
      best_with_target = std::min(best_with_target, cost);
    }
  } while (std::next_permutation(tail.begin(), tail.end()));
  return {best, best_with_target};
}

void CheckTargetIsNotOptimal(const cudaee::GraphSnapshot& graph, const cudaee::NodeEdge target) {
  const auto [best, best_with_target] = BruteForceTourCosts(graph, target);
  Check(best_with_target > best, "HT-proven target belongs to no optimal tour");
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

void TestHamiltonReplyBatch() {
  std::vector<cudaee::Point> points;
  for (std::int32_t node = 0; node < 12; ++node) {
    const std::int64_t x = (41 * node + 17) % 109;
    const std::int64_t y = (19 * node * node + 5 * node + 3) % 103;
    points.push_back({static_cast<double>(x), static_cast<double>(y), x, y});
  }
  cudaee::GraphSnapshot graph = MakeCompleteGraph(points);
  const std::vector<std::int32_t> centers = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2};
  for (const cudaee::DistanceType distance_type :
       {cudaee::DistanceType::kEuc2D, cudaee::DistanceType::kCeil2D}) {
    graph.distance_type = distance_type;
    const cudaee::HtHamiltonReplyBatchResult cpu = cudaee::EvaluateHtHamiltonReplies(
        graph, {0, 1}, centers, cudaee::PathCompatibilityBackend::kCpu);
    Check(cpu.backend == "cpu" && cpu.cpu_verified && cpu.offsets.size() == centers.size() + 1U &&
              cpu.offsets.front() == 0U && cpu.offsets.back() == cpu.replies.size(),
          "CPU Hamilton reply batch metadata and offsets are complete");
    for (std::size_t index = 0; index < centers.size(); ++index) {
      const std::vector<cudaee::HtNeighborPair> expected =
          cudaee::EnumerateHtHamiltonReplies(graph, {0, 1}, centers[index]);
      const auto begin = cpu.replies.begin() + static_cast<std::ptrdiff_t>(cpu.offsets[index]);
      const auto end = cpu.replies.begin() + static_cast<std::ptrdiff_t>(cpu.offsets[index + 1U]);
      Check(std::vector<cudaee::HtNeighborPair>(begin, end) == expected,
            "batched Hamilton replies preserve each center's canonical order");
    }

#ifdef CUDAEE_HAS_CUDA
    std::string reason;
    if (cudaee::detail::HtHamiltonReplyCudaAvailable(&reason)) {
      const cudaee::HtHamiltonReplyBatchResult gpu = cudaee::EvaluateHtHamiltonReplies(
          graph, {0, 1}, centers, cudaee::PathCompatibilityBackend::kCuda);
      Check(gpu.backend == "cuda" && gpu.selected_device >= 0 && gpu.cpu_verified &&
                gpu.offsets == cpu.offsets && gpu.replies == cpu.replies,
            "CUDA Hamilton reply count/write exactly matches the CPU batch");
    }
#else
    const cudaee::HtHamiltonReplyBatchResult fallback = cudaee::EvaluateHtHamiltonReplies(
        graph, {0, 1}, centers, cudaee::PathCompatibilityBackend::kAuto);
    Check(fallback.backend == "cpu-fallback" && fallback.offsets == cpu.offsets &&
              fallback.replies == cpu.replies,
          "Hamilton reply auto backend safely falls back in a CPU-only build");
#endif
  }
  CheckThrows(
      [&] {
        const auto ignored = cudaee::EvaluateHtHamiltonReplies(
            graph, {0, 1}, {0}, cudaee::PathCompatibilityBackend::kCpu);
        static_cast<void>(ignored);
      },
      "Hamilton reply batch rejects a target endpoint as center");
#ifndef CUDAEE_HAS_CUDA
  CheckThrows(
      [&] {
        const auto ignored = cudaee::EvaluateHtHamiltonReplies(
            graph, {0, 1}, centers, cudaee::PathCompatibilityBackend::kCuda);
        static_cast<void>(ignored);
      },
      "explicit CUDA Hamilton reply batch remains unavailable in a CPU-only build");
#endif
}

void TestEndReplyBatch() {
  std::vector<cudaee::Point> points;
  for (std::int32_t node = 0; node < 9; ++node) {
    points.push_back({static_cast<double>(node * 7), static_cast<double>((node * node + 3) % 17),
                      node * 7, (node * node + 3) % 17});
  }
  const cudaee::GraphSnapshot graph = MakeCompleteGraph(points);
  const std::vector<cudaee::HtEndReplyTask> tasks = {{0, 1}, {1, 0}, {8, 3}, {0, 1}};
  const cudaee::HtEndReplyBatchResult cpu =
      cudaee::EvaluateHtEndReplies(graph, tasks, cudaee::PathCompatibilityBackend::kCpu);
  Check(cpu.backend == "cpu" && cpu.cpu_verified && cpu.offsets.size() == tasks.size() + 1U &&
            cpu.offsets.front() == 0U && cpu.offsets.back() == cpu.replies.size(),
        "CPU end reply batch metadata and offsets are complete");
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    std::vector<cudaee::NodeEdge> expected;
    for (std::int32_t neighbor = 0; neighbor < graph.dimension; ++neighbor) {
      if (neighbor != tasks[index].endpoint && neighbor != tasks[index].internal_neighbor) {
        expected.push_back(CanonicalEdge(tasks[index].endpoint, neighbor));
      }
    }
    const auto begin = cpu.replies.begin() + static_cast<std::ptrdiff_t>(cpu.offsets[index]);
    const auto end = cpu.replies.begin() + static_cast<std::ptrdiff_t>(cpu.offsets[index + 1U]);
    Check(std::vector<cudaee::NodeEdge>(begin, end) == expected,
          "batched end replies preserve CSR neighbor order");
  }
  const cudaee::GraphSnapshot leaf_graph = MakeGraph(points, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
  const cudaee::HtEndReplyBatchResult empty_cpu =
      cudaee::EvaluateHtEndReplies(leaf_graph, {{0, 1}}, cudaee::PathCompatibilityBackend::kCpu);
  Check(empty_cpu.offsets == std::vector<std::uint64_t>({0U, 0U}) && empty_cpu.replies.empty(),
        "degree-one endpoint produces an empty but complete end reply slice");

#ifdef CUDAEE_HAS_CUDA
  std::string reason;
  if (cudaee::detail::HtEndReplyCudaAvailable(&reason)) {
    const cudaee::HtEndReplyBatchResult gpu =
        cudaee::EvaluateHtEndReplies(graph, tasks, cudaee::PathCompatibilityBackend::kCuda);
    Check(gpu.backend == "cuda" && gpu.selected_device >= 0 && gpu.cpu_verified &&
              gpu.offsets == cpu.offsets && gpu.replies == cpu.replies,
          "CUDA end reply count/write exactly matches the CPU batch");
    const cudaee::HtEndReplyBatchResult empty_gpu =
        cudaee::EvaluateHtEndReplies(leaf_graph, {{0, 1}}, cudaee::PathCompatibilityBackend::kCuda);
    Check(empty_gpu.offsets == empty_cpu.offsets && empty_gpu.replies.empty(),
          "CUDA end reply count/write preserves an empty reply slice");
  }
#else
  const cudaee::HtEndReplyBatchResult fallback =
      cudaee::EvaluateHtEndReplies(graph, tasks, cudaee::PathCompatibilityBackend::kAuto);
  Check(fallback.backend == "cpu-fallback" && fallback.offsets == cpu.offsets &&
            fallback.replies == cpu.replies,
        "end reply auto backend safely falls back in a CPU-only build");
  CheckThrows(
      [&] {
        const auto ignored =
            cudaee::EvaluateHtEndReplies(graph, tasks, cudaee::PathCompatibilityBackend::kCuda);
        static_cast<void>(ignored);
      },
      "explicit CUDA end reply batch remains unavailable in a CPU-only build");
#endif
  CheckThrows(
      [&] {
        const auto ignored =
            cudaee::EvaluateHtEndReplies(graph, {{2, 2}}, cudaee::PathCompatibilityBackend::kCpu);
        static_cast<void>(ignored);
      },
      "end reply batch rejects the endpoint as its own internal neighbor");
  CheckThrows(
      [&] {
        const auto ignored = cudaee::EvaluateHtEndReplies(leaf_graph, {{0, 2}},
                                                          cudaee::PathCompatibilityBackend::kCpu);
        static_cast<void>(ignored);
      },
      "end reply batch rejects an inactive internal edge");
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

void TestDfsWavefrontRandomDifferential() {
  std::mt19937 random(0x48545746U); // NOLINT: 固定 HT wavefront 差分种子。
  std::uniform_int_distribution<std::int64_t> coordinate(-100, 100);
  for (std::uint32_t trial = 0; trial < 8U; ++trial) {
    std::vector<cudaee::Point> points;
    for (std::uint32_t node = 0; node < 7U; ++node) {
      const std::int64_t x = coordinate(random);
      const std::int64_t y = coordinate(random);
      points.push_back({static_cast<double>(x), static_cast<double>(y), x, y});
    }
    const cudaee::GraphSnapshot graph = MakeCompleteGraph(points);
    const std::int32_t first = static_cast<std::int32_t>(trial % 7U);
    std::int32_t second = static_cast<std::int32_t>((3U * trial + 1U) % 7U);
    if (second == first) {
      second = (second + 1) % 7;
    }
    const cudaee::NodeEdge target = CanonicalEdge(first, second);
    const cudaee::HtRecursiveOptions options = {
        .root_options = {.max_neighborhood = 0,
                         .max_cd_candidates = 0,
                         .max_candidate_degree = 0,
                         .max_reply_combinations = 0,
                         .cd_mode = cudaee::HtCdMode::kActiveIncompatible,
                         .candidate_backend = cudaee::PathCompatibilityBackend::kCpu,
                         .leaf_options = {.max_k = 3, .max_deletion_sets = 2}},
        .max_depth = 1,
        .max_states = 0,
        .max_total_replies = 0,
        .max_replies_per_move = 0,
        .max_point_candidates = 2,
        .max_end_candidates = 2};
    const cudaee::HtRecursiveResult dfs = cudaee::ProveEdgeByRecursiveHt(graph, target, options);
    const cudaee::HtWavefrontResult wavefront = cudaee::ProveEdgeByWavefrontHt(
        graph, target,
        {.search_options = options,
         .propagation_backend = cudaee::PathCompatibilityBackend::kCpu,
         .path_append_backend = cudaee::PathCompatibilityBackend::kCpu,
         .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCpu});
    Check(dfs.status == wavefront.status, "random DFS/wavefront truth mismatch");
    if (dfs.status == cudaee::HtSearchStatus::kProven) {
      std::string reason;
      Check(cudaee::VerifyHtRecursiveProof(graph, dfs.proof, &reason), reason);
      Check(cudaee::VerifyHtRecursiveProof(graph, wavefront.proof, &reason), reason);
      CheckTargetIsNotOptimal(graph, target);
    }
  }
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
  CheckTargetIsNotOptimal(graph, {0, 5});
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

  const cudaee::HtRecursiveOptions recursive_options = {
      .root_options = {.max_neighborhood = 0,
                       .max_cd_candidates = 20,
                       .max_candidate_degree = 0,
                       .max_reply_combinations = 10000,
                       .cd_mode = cudaee::HtCdMode::kActiveIncompatible,
                       .leaf_options = {.max_k = 3,
                                        .max_deletion_sets = 1,
                                        .exact_fallback_max_blocks = 10}},
      .max_depth = 0};
  const cudaee::HtRecursiveResult recursive =
      cudaee::ProveEdgeByRecursiveHt(graph, {0, 5}, recursive_options);
  Check(recursive.status == cudaee::HtSearchStatus::kProven, recursive.proof.reason);
  Check(recursive.proof.nodes.front().move_type == cudaee::HtMoveType::kCd,
        "recursive arena starts with c,d root");
  Check(cudaee::VerifyHtRecursiveProof(graph, recursive.proof, &reason), reason);

  const cudaee::HtWavefrontResult wavefront = cudaee::ProveEdgeByWavefrontHt(
      graph, {0, 5},
      {.search_options = recursive_options,
       .propagation_backend = cudaee::PathCompatibilityBackend::kCpu,
       .path_append_backend = cudaee::PathCompatibilityBackend::kCpu,
       .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCpu});
  Check(wavefront.status == cudaee::HtSearchStatus::kProven, wavefront.proof.reason);
  Check(wavefront.propagation_backend == "cpu" && wavefront.cpu_verified,
        "CPU wavefront propagation is recorded");
  Check(cudaee::VerifyHtRecursiveProof(graph, wavefront.proof, &reason), reason);

  cudaee::HtWavefrontOptions invalid_wavefront_options = {
      .search_options = recursive_options,
      .propagation_backend = static_cast<cudaee::PathCompatibilityBackend>(255)};
  Check(cudaee::ProveEdgeByWavefrontHt(graph, {0, 5}, invalid_wavefront_options).status ==
            cudaee::HtSearchStatus::kInvalid,
        "unknown wavefront propagation backend is rejected");
  invalid_wavefront_options.propagation_backend = cudaee::PathCompatibilityBackend::kCpu;
  invalid_wavefront_options.path_append_backend =
      static_cast<cudaee::PathCompatibilityBackend>(255);
  Check(cudaee::ProveEdgeByWavefrontHt(graph, {0, 5}, invalid_wavefront_options).status ==
            cudaee::HtSearchStatus::kInvalid,
        "unknown wavefront path-append backend is rejected");
  invalid_wavefront_options.path_append_backend = cudaee::PathCompatibilityBackend::kCpu;
  invalid_wavefront_options.hamilton_reply_backend =
      static_cast<cudaee::PathCompatibilityBackend>(255);
  Check(cudaee::ProveEdgeByWavefrontHt(graph, {0, 5}, invalid_wavefront_options).status ==
            cudaee::HtSearchStatus::kInvalid,
        "unknown wavefront Hamilton reply backend is rejected");
  Check(wavefront.proof.nodes.size() == recursive.proof.nodes.size(),
        "DFS and wavefront shallow arenas have the same size");

  const cudaee::HtRecursiveProof parsed_recursive =
      cudaee::ParseHtRecursiveProof(cudaee::SerializeHtRecursiveProof(recursive.proof));
  Check(cudaee::VerifyHtRecursiveProof(graph, parsed_recursive, &reason), reason);
  cudaee::HtRecursiveProof wrong_nested_leaf = parsed_recursive;
  const auto nested_leaf = std::find_if(
      wrong_nested_leaf.nodes.begin(), wrong_nested_leaf.nodes.end(),
      [](const cudaee::HtTreeNode& node) { return node.move_type == cudaee::HtMoveType::kLeaf; });
  Check(nested_leaf != wrong_nested_leaf.nodes.end(), "recursive proof contains a nested leaf");
  nested_leaf->leaf_proof.path_system_hash ^= 1U;
  Check(!cudaee::VerifyHtRecursiveProof(graph, wrong_nested_leaf, &reason),
        "tampered serialized nested leaf is rejected");

  cudaee::HtRecursiveProof bad_child = recursive.proof;
  const auto child_reply =
      std::find_if(bad_child.nodes.front().replies.begin(), bad_child.nodes.front().replies.end(),
                   [](const cudaee::HtTreeReply& reply) { return !reply.path_infeasible; });
  Check(child_reply != bad_child.nodes.front().replies.end(), "recursive root has a child state");
  child_reply->child_index = cudaee::kNoHtChild;
  Check(!cudaee::VerifyHtRecursiveProof(graph, bad_child, &reason),
        "missing recursive child is rejected");

  cudaee::HtRecursiveOptions state_budget_options = recursive_options;
  state_budget_options.max_states = 1;
  const cudaee::HtRecursiveResult state_budget =
      cudaee::ProveEdgeByRecursiveHt(graph, {0, 5}, state_budget_options);
  Check(state_budget.status == cudaee::HtSearchStatus::kUnresolved,
        "recursive state budget remains unresolved");
  const cudaee::HtWavefrontResult wavefront_state_budget = cudaee::ProveEdgeByWavefrontHt(
      graph, {0, 5},
      {.search_options = state_budget_options,
       .propagation_backend = cudaee::PathCompatibilityBackend::kCpu,
       .path_append_backend = cudaee::PathCompatibilityBackend::kCpu,
       .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCpu});
  Check(wavefront_state_budget.status == cudaee::HtSearchStatus::kUnresolved,
        "wavefront state budget remains unresolved");

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

void TestRecursivePointProof() {
  const cudaee::GraphSnapshot graph = MakeCompleteGraph({{17.0, 28.0, 17, 28},
                                                         {3.0, -18.0, 3, -18},
                                                         {34.0, -5.0, 34, -5},
                                                         {-13.0, -26.0, -13, -26},
                                                         {-45.0, -15.0, -45, -15},
                                                         {11.0, 30.0, 11, 30},
                                                         {31.0, -47.0, 31, -47},
                                                         {27.0, -19.0, 27, -19}});
  const cudaee::HtShallowOptions root_options = {
      .max_neighborhood = 0,
      .max_cd_candidates = 8,
      .max_candidate_degree = 0,
      .max_reply_combinations = 1000,
      .cd_mode = cudaee::HtCdMode::kActiveIncompatible,
      .leaf_options = {.max_k = 3, .max_deletion_sets = 1}};
  const cudaee::HtShallowResult shallow = cudaee::ProveEdgeByShallowHt(graph, {2, 4}, root_options);
  Check(shallow.status == cudaee::HtSearchStatus::kUnresolved,
        "pinned target is not solved by shallow leaves");

  const cudaee::HtRecursiveOptions options = {.root_options = root_options,
                                              .max_depth = 1,
                                              .max_states = 20000,
                                              .max_total_replies = 200000,
                                              .max_replies_per_move = 1000,
                                              .max_point_candidates = 5,
                                              .max_end_candidates = 5};
  const cudaee::HtRecursiveResult result = cudaee::ProveEdgeByRecursiveHt(graph, {2, 4}, options);
  Check(result.status == cudaee::HtSearchStatus::kProven, result.proof.reason);
  CheckTargetIsNotOptimal(graph, {2, 4});
  Check(result.proof.nodes.size() == 4, "pinned recursive arena node count");
  Check(result.proof.nodes.front().move_type == cudaee::HtMoveType::kCd &&
            result.proof.nodes.front().move_first == 0 &&
            result.proof.nodes.front().move_second == 3,
        "pinned recursive root move");
  Check(std::all_of(result.proof.nodes.begin() + 1, result.proof.nodes.end(),
                    [](const cudaee::HtTreeNode& node) {
                      return node.move_type == cudaee::HtMoveType::kPoint;
                    }),
        "recursive proof genuinely uses point moves");
  std::string reason;
  Check(cudaee::VerifyHtRecursiveProof(graph, result.proof, &reason), reason);

  const cudaee::HtWavefrontResult wavefront = cudaee::ProveEdgeByWavefrontHt(
      graph, {2, 4},
      {.search_options = options,
       .propagation_backend = cudaee::PathCompatibilityBackend::kCpu,
       .path_append_backend = cudaee::PathCompatibilityBackend::kCpu,
       .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCpu});
  Check(wavefront.status == cudaee::HtSearchStatus::kProven, wavefront.proof.reason);
  Check(wavefront.moves_generated > 0U && wavefront.peak_frontier > 0U,
        "wavefront records generated moves and frontier width");
  Check(wavefront.path_append_backend == "cpu" && wavefront.path_append_cpu_verified &&
            wavefront.path_append_batches > 0U && wavefront.path_append_tasks > 0U,
        "CPU wavefront records fully verified path-append batches");
  Check(wavefront.hamilton_reply_backend == "cpu" && wavefront.hamilton_reply_cpu_verified &&
            wavefront.hamilton_reply_batches > 0U && wavefront.hamilton_reply_centers > 0U &&
            wavefront.hamilton_replies_generated > 0U,
        "CPU wavefront records fully verified Hamilton reply batches");
  Check(wavefront.reply_frontier_batches > 0U && wavefront.reply_frontier_states > 0U &&
            wavefront.peak_reply_frontier_batch > 1U,
        "wavefront records multi-parent reply generation chunks");
  Check(wavefront.proof.nodes.size() == result.proof.nodes.size(),
        "DFS and wavefront point arenas have the same size");
  Check(std::all_of(wavefront.proof.nodes.begin() + 1, wavefront.proof.nodes.end(),
                    [](const cudaee::HtTreeNode& node) {
                      return node.move_type == cudaee::HtMoveType::kPoint;
                    }),
        "wavefront proof genuinely uses point moves");
  Check(cudaee::VerifyHtRecursiveProof(graph, wavefront.proof, &reason), reason);

  const cudaee::HtWavefrontResult single_state_batches = cudaee::ProveEdgeByWavefrontHt(
      graph, {2, 4},
      {.search_options = options,
       .reply_frontier_batch_states = 1,
       .propagation_backend = cudaee::PathCompatibilityBackend::kCpu,
       .path_append_backend = cudaee::PathCompatibilityBackend::kCpu,
       .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCpu});
  Check(single_state_batches.status == wavefront.status &&
            cudaee::SerializeHtRecursiveProof(single_state_batches.proof) ==
                cudaee::SerializeHtRecursiveProof(wavefront.proof),
        "frontier reply chunk size preserves the canonical recursive proof");
  Check(single_state_batches.reply_frontier_batches > wavefront.reply_frontier_batches &&
            single_state_batches.hamilton_reply_batches > wavefront.hamilton_reply_batches &&
            single_state_batches.path_append_batches > wavefront.path_append_batches &&
            single_state_batches.end_reply_tasks == wavefront.end_reply_tasks &&
            single_state_batches.end_replies_generated == wavefront.end_replies_generated,
        "multi-parent chunks reduce launch count without speculative end work");

#ifndef CUDAEE_HAS_CUDA
  const cudaee::HtWavefrontResult auto_fallback = cudaee::ProveEdgeByWavefrontHt(
      graph, {2, 4},
      {.search_options = options,
       .propagation_backend = cudaee::PathCompatibilityBackend::kAuto,
       .path_append_backend = cudaee::PathCompatibilityBackend::kAuto,
       .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kAuto});
  Check(auto_fallback.status == cudaee::HtSearchStatus::kProven &&
            auto_fallback.propagation_backend == "cpu" && auto_fallback.cpu_verified &&
            auto_fallback.hamilton_reply_backend == "cpu-fallback" &&
            auto_fallback.hamilton_reply_cpu_verified,
        "auto wavefront safely falls back to CPU without a CUDA build");
  const cudaee::HtWavefrontResult unavailable_cuda = cudaee::ProveEdgeByWavefrontHt(
      graph, {2, 4},
      {.search_options = options,
       .propagation_backend = cudaee::PathCompatibilityBackend::kCuda,
       .path_append_backend = cudaee::PathCompatibilityBackend::kCpu,
       .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCpu});
  Check(unavailable_cuda.status == cudaee::HtSearchStatus::kUnresolved,
        "explicit unavailable CUDA wavefront remains unresolved");
#endif

  const std::string serialized = cudaee::SerializeHtRecursiveProof(result.proof);
  const cudaee::HtRecursiveProof parsed = cudaee::ParseHtRecursiveProof(serialized);
  Check(cudaee::VerifyHtRecursiveProof(graph, parsed, &reason), reason);
  Check(cudaee::SerializeHtRecursiveProof(parsed) == serialized,
        "recursive HT V1 serialization is canonical");
  CheckThrows(
      [&serialized] {
        const auto ignored = cudaee::ParseHtRecursiveProof(serialized + "trailing\n");
        static_cast<void>(ignored);
      },
      "recursive HT parser rejects trailing fields");

  const std::filesystem::path proof_path =
      std::filesystem::path(CUDAEE_HT_TEST_TMP_DIR) / "recursive-point.ht-proof";
  std::filesystem::create_directories(proof_path.parent_path());
  cudaee::WriteHtRecursiveProof(proof_path, result.proof);
  const cudaee::HtRecursiveProof loaded = cudaee::ReadHtRecursiveProof(proof_path);
  Check(cudaee::VerifyHtRecursiveProof(graph, loaded, &reason), reason);

  cudaee::HtRecursiveOptions shallow_depth = options;
  shallow_depth.max_depth = 0;
  const cudaee::HtRecursiveResult no_recursion =
      cudaee::ProveEdgeByRecursiveHt(graph, {2, 4}, shallow_depth);
  Check(no_recursion.status == cudaee::HtSearchStatus::kUnresolved,
        "depth zero cannot replace the required point moves");

  cudaee::HtRecursiveProof wrong_point = result.proof;
  wrong_point.nodes[1].move_first = wrong_point.nodes[1].paths.paths.front().front();
  Check(!cudaee::VerifyHtRecursiveProof(graph, wrong_point, &reason),
        "point already present in the state is rejected");
  cudaee::HtRecursiveProof missing_arena_node = result.proof;
  missing_arena_node.nodes.pop_back();
  Check(!cudaee::VerifyHtRecursiveProof(graph, missing_arena_node, &reason),
        "truncated continuation arena is rejected");

#ifdef CUDAEE_HAS_CUDA
  std::string cuda_reason;
  if (cudaee::detail::HtCdCudaAvailable(&cuda_reason)) {
    cudaee::HtRecursiveOptions cuda_options = options;
    cuda_options.root_options.candidate_backend = cudaee::PathCompatibilityBackend::kCuda;
    const cudaee::HtRecursiveResult cuda_result =
        cudaee::ProveEdgeByRecursiveHt(graph, {2, 4}, cuda_options);
    Check(cuda_result.status == cudaee::HtSearchStatus::kProven, cuda_result.proof.reason);
    Check(cudaee::VerifyHtRecursiveProof(graph, cuda_result.proof, &reason), reason);

    const cudaee::HtWavefrontResult cuda_wavefront = cudaee::ProveEdgeByWavefrontHt(
        graph, {2, 4},
        {.search_options = options,
         .propagation_backend = cudaee::PathCompatibilityBackend::kCuda,
         .path_append_backend = cudaee::PathCompatibilityBackend::kCuda,
         .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCuda});
    Check(cuda_wavefront.status == cudaee::HtSearchStatus::kProven, cuda_wavefront.proof.reason);
    Check(cuda_wavefront.propagation_backend == "cuda" && cuda_wavefront.selected_device >= 0 &&
              cuda_wavefront.cpu_verified,
          "CUDA wavefront propagation is fully CPU verified");
    Check(cuda_wavefront.path_append_backend == "cuda" &&
              cuda_wavefront.path_append_selected_device >= 0 &&
              cuda_wavefront.path_append_cpu_verified && cuda_wavefront.path_append_tasks > 0U,
          "CUDA wavefront path-append batches are fully CPU verified");
    Check(cuda_wavefront.hamilton_reply_backend == "cuda" &&
              cuda_wavefront.hamilton_reply_selected_device >= 0 &&
              cuda_wavefront.hamilton_reply_cpu_verified &&
              cuda_wavefront.hamilton_reply_centers > 0U,
          "CUDA wavefront Hamilton reply count/write is fully CPU verified");
    Check(cudaee::VerifyHtRecursiveProof(graph, cuda_wavefront.proof, &reason), reason);
  }
#endif
}

void TestRecursiveEndProof() {
  const cudaee::GraphSnapshot graph = MakeCompleteGraph({{53.0, 71.0, 53, 71},
                                                         {-43.0, -76.0, -43, -76},
                                                         {33.0, -58.0, 33, -58},
                                                         {-29.0, -46.0, -29, -46},
                                                         {49.0, -71.0, 49, -71},
                                                         {-13.0, -28.0, -13, -28},
                                                         {16.0, -37.0, 16, -37},
                                                         {-42.0, -7.0, -42, -7}});
  const cudaee::HtShallowOptions root_options = {
      .max_neighborhood = 0,
      .max_cd_candidates = 8,
      .max_candidate_degree = 0,
      .max_reply_combinations = 1000,
      .cd_mode = cudaee::HtCdMode::kActiveIncompatible,
      .leaf_options = {.max_k = 3, .max_deletion_sets = 1}};
  const cudaee::HtShallowResult shallow = cudaee::ProveEdgeByShallowHt(graph, {1, 2}, root_options);
  Check(shallow.status == cudaee::HtSearchStatus::kUnresolved,
        "pinned end target is not solved by shallow leaves");

  const cudaee::HtRecursiveOptions options = {.root_options = root_options,
                                              .max_depth = 1,
                                              .max_states = 50000,
                                              .max_total_replies = 500000,
                                              .max_replies_per_move = 1000,
                                              .max_point_candidates = 0,
                                              .max_end_candidates = 8,
                                              .enable_point_moves = false,
                                              .enable_end_moves = true};
  const cudaee::HtRecursiveResult result = cudaee::ProveEdgeByRecursiveHt(graph, {1, 2}, options);
  Check(result.status == cudaee::HtSearchStatus::kProven, result.proof.reason);
  CheckTargetIsNotOptimal(graph, {1, 2});
  Check(result.proof.nodes.size() == 4, "pinned end arena node count");
  Check(result.proof.nodes[0].move_type == cudaee::HtMoveType::kCd &&
            result.proof.nodes[0].move_first == 4 && result.proof.nodes[0].move_second == 6,
        "pinned end root move");
  Check(result.proof.nodes[1].move_type == cudaee::HtMoveType::kEnd &&
            result.proof.nodes[1].move_first == 3 && result.proof.nodes[1].move_second == 6 &&
            result.proof.nodes[1].replies.size() == 6,
        "recursive proof genuinely uses a complete end move");
  std::string reason;
  Check(cudaee::VerifyHtRecursiveProof(graph, result.proof, &reason), reason);

  const cudaee::HtWavefrontResult wavefront = cudaee::ProveEdgeByWavefrontHt(
      graph, {1, 2},
      {.search_options = options,
       .propagation_backend = cudaee::PathCompatibilityBackend::kCpu,
       .path_append_backend = cudaee::PathCompatibilityBackend::kCpu,
       .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCpu});
  Check(wavefront.status == cudaee::HtSearchStatus::kProven, wavefront.proof.reason);
  Check(wavefront.proof.nodes.size() == result.proof.nodes.size(),
        "DFS and wavefront end arenas have the same size");
  Check(wavefront.proof.nodes[1].move_type == cudaee::HtMoveType::kEnd,
        "wavefront proof genuinely uses an end move");
  Check(wavefront.end_reply_backend == "cpu" && wavefront.end_reply_cpu_verified &&
            wavefront.end_reply_batches > 0U && wavefront.end_reply_tasks > 0U &&
            wavefront.end_replies_generated > 0U,
        "CPU wavefront records fully verified end reply batches");
  Check(cudaee::VerifyHtRecursiveProof(graph, wavefront.proof, &reason), reason);

#ifdef CUDAEE_HAS_CUDA
  std::string unavailable_reason;
  if (cudaee::detail::HtWavefrontCudaAvailable(&unavailable_reason) &&
      cudaee::detail::HtPathAppendCudaAvailable(&unavailable_reason) &&
      cudaee::detail::HtHamiltonReplyCudaAvailable(&unavailable_reason) &&
      cudaee::detail::HtEndReplyCudaAvailable(&unavailable_reason)) {
    const cudaee::HtWavefrontResult cuda_wavefront = cudaee::ProveEdgeByWavefrontHt(
        graph, {1, 2},
        {.search_options = options,
         .propagation_backend = cudaee::PathCompatibilityBackend::kCuda,
         .path_append_backend = cudaee::PathCompatibilityBackend::kCuda,
         .hamilton_reply_backend = cudaee::PathCompatibilityBackend::kCuda});
    Check(cuda_wavefront.status == cudaee::HtSearchStatus::kProven, cuda_wavefront.proof.reason);
    Check(cuda_wavefront.end_reply_backend == "cuda" &&
              cuda_wavefront.end_reply_selected_device >= 0 &&
              cuda_wavefront.end_reply_cpu_verified && cuda_wavefront.end_reply_batches > 0U &&
              cuda_wavefront.end_reply_tasks > 0U,
          "CUDA wavefront end reply count/write is fully CPU verified");
    Check(cudaee::VerifyHtRecursiveProof(graph, cuda_wavefront.proof, &reason), reason);
  }
#endif

  cudaee::HtRecursiveOptions no_depth = options;
  no_depth.max_depth = 0;
  Check(cudaee::ProveEdgeByRecursiveHt(graph, {1, 2}, no_depth).status ==
            cudaee::HtSearchStatus::kUnresolved,
        "depth zero cannot replace the required end move");

  cudaee::HtRecursiveProof missing_end_reply = result.proof;
  missing_end_reply.nodes[1].replies.pop_back();
  Check(!cudaee::VerifyHtRecursiveProof(graph, missing_end_reply, &reason),
        "missing end reply is rejected");
  cudaee::HtRecursiveProof wrong_internal = result.proof;
  wrong_internal.nodes[1].move_second = 0;
  Check(!cudaee::VerifyHtRecursiveProof(graph, wrong_internal, &reason),
        "wrong end internal neighbor is rejected");
}

void TestPathAppendBatch() {
  const cudaee::NormalizedPathSystem parent = cudaee::NormalizePathSystem({{0, 1, 2}, {3, 4}}, 8);
  Check(parent.valid, parent.reason);
  const cudaee::NormalizedPathSystem second_parent =
      cudaee::NormalizePathSystem({{5, 6}, {0, 7, 3}}, 8);
  Check(second_parent.valid, second_parent.reason);
  const std::vector<cudaee::NormalizedPathSystem> parents = {parent, second_parent};
  const std::vector<cudaee::HtPathAppendTask> tasks = {
      {0, cudaee::HtPathAppendKind::kPoint, 0, 5, 3},
      {0, cudaee::HtPathAppendKind::kPoint, 0, 6, 2},
      {0, cudaee::HtPathAppendKind::kPoint, 1, 6, 7},
      {0, cudaee::HtPathAppendKind::kPoint, 0, 6, 7},
      {0, cudaee::HtPathAppendKind::kEnd, 0, -1, 3},
      {0, cudaee::HtPathAppendKind::kEnd, 0, -1, 2},
      {0, cudaee::HtPathAppendKind::kEnd, 0, -1, 1},
      {0, cudaee::HtPathAppendKind::kEnd, 0, -1, 7},
      {1, cudaee::HtPathAppendKind::kPoint, 5, 1, 0},
      {1, cudaee::HtPathAppendKind::kEnd, 6, -1, 7},
      {1, cudaee::HtPathAppendKind::kEnd, 6, -1, 2}};
  const std::vector<std::uint8_t> expected = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1};
  const cudaee::HtPathAppendBatchResult cpu =
      cudaee::EvaluateHtPathAppends(8, parents, tasks, cudaee::PathCompatibilityBackend::kCpu);
  Check(cpu.backend == "cpu" && cpu.cpu_verified && cpu.feasible == expected,
        "CPU path-append batch covers merge, cycle, degree, and new-node cases");
  Check(cpu.children.size() == tasks.size() && cpu.children.front().valid &&
            cpu.children.front().edge_count == parent.edge_count + 2U && !cpu.children[1].valid,
        "path-append batch keeps canonical children and infeasibility records aligned");

  std::vector<cudaee::HtPathAppendTask> bad_point = tasks;
  bad_point.front().center = 1;
  CheckThrows(
      [&] {
        const auto ignored = cudaee::EvaluateHtPathAppends(8, parents, bad_point,
                                                           cudaee::PathCompatibilityBackend::kCpu);
        static_cast<void>(ignored);
      },
      "path-append rejects a point center already present in the parent");
  std::vector<cudaee::HtPathAppendTask> bad_end = tasks;
  bad_end.back().first = 7;
  CheckThrows(
      [&] {
        const auto ignored = cudaee::EvaluateHtPathAppends(8, parents, bad_end,
                                                           cudaee::PathCompatibilityBackend::kCpu);
        static_cast<void>(ignored);
      },
      "path-append rejects an end move from an internal node");

#ifdef CUDAEE_HAS_CUDA
  std::string reason;
  if (cudaee::detail::HtPathAppendCudaAvailable(&reason)) {
    const cudaee::HtPathAppendBatchResult gpu =
        cudaee::EvaluateHtPathAppends(8, parents, tasks, cudaee::PathCompatibilityBackend::kCuda);
    Check(gpu.backend == "cuda" && gpu.selected_device >= 0 && gpu.cpu_verified &&
              gpu.feasible == expected,
          "CUDA path-append flags exactly match CPU normalization");
    for (std::size_t index = 0; index < tasks.size(); ++index) {
      Check(gpu.children[index].valid == cpu.children[index].valid &&
                gpu.children[index].paths == cpu.children[index].paths,
            "CUDA path-append keeps CPU-certified canonical children");
    }
  }
#else
  const cudaee::HtPathAppendBatchResult fallback =
      cudaee::EvaluateHtPathAppends(8, parents, tasks, cudaee::PathCompatibilityBackend::kAuto);
  Check(fallback.backend == "cpu-fallback" && fallback.feasible == expected,
        "path-append auto backend safely falls back in a CPU-only build");
  CheckThrows(
      [&] {
        const auto ignored = cudaee::EvaluateHtPathAppends(8, parents, tasks,
                                                           cudaee::PathCompatibilityBackend::kCuda);
        static_cast<void>(ignored);
      },
      "explicit CUDA path-append remains unavailable in a CPU-only build");
#endif
}

#ifdef CUDAEE_HAS_CUDA
void TestCudaWavefrontTruthTable() {
  std::string reason;
  if (!cudaee::detail::HtWavefrontCudaAvailable(&reason)) {
    return;
  }
  // 根的第一个 move 是 success AND failure，第二个 move 单独成功，覆盖 AND/OR 两级真值。
  const std::vector<cudaee::HtWavefrontStateTask> states = {
      {cudaee::kNoHtChild, 0, 2, 0}, {0, 2, 0, 1}, {0, 2, 0, 0}, {1, 2, 0, 1}};
  const std::vector<cudaee::HtWavefrontMoveTask> moves = {{0, 0, 2, 2}, {0, 2, 1, 1}};
  const std::vector<cudaee::HtWavefrontReplyTask> replies = {{1, 0}, {2, 0}, {3, 0}};
  int selected_device = -1;
  const std::vector<std::uint8_t> status =
      cudaee::detail::EvaluateHtWavefrontCuda(states, moves, replies, {0, 1, 4}, &selected_device);
  Check(status == std::vector<std::uint8_t>({1, 1, 0, 1}) && selected_device >= 0,
        "CUDA wavefront evaluates the pinned AND/OR truth table");

  const std::vector<cudaee::HtWavefrontStateTask> failed_states = {
      {cudaee::kNoHtChild, 0, 1, 0}, {0, 1, 0, 1}, {0, 1, 0, 0}};
  const std::vector<cudaee::HtWavefrontMoveTask> failed_moves = {{0, 0, 2, 2}};
  const std::vector<cudaee::HtWavefrontReplyTask> failed_replies = {{1, 0}, {2, 0}};
  const std::vector<std::uint8_t> failed_status = cudaee::detail::EvaluateHtWavefrontCuda(
      failed_states, failed_moves, failed_replies, {0, 1, 3}, &selected_device);
  Check(failed_status == std::vector<std::uint8_t>({0, 1, 0}),
        "CUDA continuation marks a state failed only after all moves fail");

  std::vector<cudaee::HtWavefrontReplyTask> invalid = replies;
  invalid.front().child_index = 0;
  CheckThrows(
      [&] {
        const auto ignored = cudaee::detail::EvaluateHtWavefrontCuda(states, moves, invalid,
                                                                     {0, 1, 4}, &selected_device);
        static_cast<void>(ignored);
      },
      "CUDA wavefront rejects a non-forward continuation");
}
#endif

} // namespace

int main() {
  try {
    TestHamiltonRepliesAgainstReferenceFormula();
    TestHamiltonReplyBatch();
    TestEndReplyBatch();
    TestCdCandidatesCpuCuda();
    TestDfsWavefrontRandomDifferential();
    TestVacuousAndProofAndTamperRejection();
    TestNonemptyAndProof();
    TestRecursivePointProof();
    TestRecursiveEndProof();
    TestPathAppendBatch();
#ifdef CUDAEE_HAS_CUDA
    TestCudaWavefrontTruthTable();
#endif
    std::cout << "Hamilton-Tutte tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
