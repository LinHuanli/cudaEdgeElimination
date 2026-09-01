#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

using SteadyClock = std::chrono::steady_clock;

double ElapsedMilliseconds(const SteadyClock::time_point begin) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - begin).count();
}

NodeEdge CanonicalEdge(const std::int32_t first, const std::int32_t second) {
  return first < second ? NodeEdge{first, second} : NodeEdge{second, first};
}

void SetReason(std::string* const reason, std::string value) {
  if (reason != nullptr) {
    *reason = std::move(value);
  }
}

bool IsKnownCdMode(const HtCdMode mode) {
  return mode == HtCdMode::kActiveIncompatible || mode == HtCdMode::kMissingOrIncompatible;
}

bool ValidateHtGraph(const GraphSnapshot& graph, std::string* const reason) {
  if (!graph.integer_coordinates || !graph.integer_distance_safe || graph.dimension < 4 ||
      graph.points.size() != static_cast<std::size_t>(graph.dimension) ||
      graph.row_offsets.size() != static_cast<std::size_t>(graph.dimension) + 1U ||
      graph.row_offsets.front() != 0 || graph.row_offsets.back() < 0 ||
      static_cast<std::size_t>(graph.row_offsets.back()) != graph.neighbors.size() ||
      graph.neighbors.size() != 2U * graph.ActiveEdgeCount()) {
    SetReason(reason, "HT 需要有效的整数距离 CSR 快照");
    return false;
  }
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(node)];
    const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(node) + 1U];
    if (begin < 0 || end < begin || static_cast<std::size_t>(end) > graph.neighbors.size()) {
      SetReason(reason, "HT CSR row offset 非法");
      return false;
    }
    std::int32_t previous = -1;
    for (std::int32_t offset = begin; offset < end; ++offset) {
      const std::int32_t neighbor = graph.neighbors[static_cast<std::size_t>(offset)];
      if (neighbor <= previous || neighbor < 0 || neighbor >= graph.dimension || neighbor == node ||
          !graph.HasActiveEdge(neighbor, node)) {
        SetReason(reason, "HT CSR 邻接表未排序、越界或不对称");
        return false;
      }
      previous = neighbor;
    }
  }
  return true;
}

bool ValidateTarget(const GraphSnapshot& graph, const NodeEdge target, std::string* const reason) {
  if (target.u < 0 || target.v >= graph.dimension || target.u >= target.v ||
      !graph.HasActiveEdge(target.u, target.v)) {
    SetReason(reason, "HT 目标必须是规范的活动边");
    return false;
  }
  return true;
}

// true 表示两条边至少有一个 2-opt 方向不是严格改善，与参考 compatible_test 同义。
bool EdgesSurviveTwoOpt(const GraphSnapshot& graph, const NodeEdge target, const NodeEdge other) {
  const __int128 original =
      static_cast<__int128>(graph.Distance(target.u, target.v)) + graph.Distance(other.u, other.v);
  const __int128 first =
      static_cast<__int128>(graph.Distance(target.u, other.v)) + graph.Distance(target.v, other.u);
  const __int128 second =
      static_cast<__int128>(graph.Distance(target.u, other.u)) + graph.Distance(target.v, other.v);
  return first >= original || second >= original;
}

bool CandidateIsAdmissible(const GraphSnapshot& graph, const NodeEdge target, const std::int32_t c,
                           const std::int32_t d, const HtCdMode mode) {
  if (!IsKnownCdMode(mode) || c < 0 || d < 0 || c >= graph.dimension || d >= graph.dimension ||
      c == d || c == target.u || c == target.v || d == target.u || d == target.v) {
    return false;
  }
  const bool active = graph.HasActiveEdge(c, d);
  const bool incompatible = !EdgesSurviveTwoOpt(graph, target, CanonicalEdge(c, d));
  if (mode == HtCdMode::kActiveIncompatible) {
    return active && incompatible;
  }
  return !active || incompatible;
}

bool MultiplyWithoutOverflow(const std::size_t first, const std::size_t second,
                             std::uint64_t* const product) {
  if (first > std::numeric_limits<std::uint64_t>::max() ||
      second > std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  const auto first64 = static_cast<std::uint64_t>(first);
  const auto second64 = static_cast<std::uint64_t>(second);
  if (second64 != 0 && first64 > std::numeric_limits<std::uint64_t>::max() / second64) {
    return false;
  }
  *product = first64 * second64;
  return true;
}

std::vector<Path> BuildReplyPaths(const NodeEdge target, const HtNeighborPair& c_reply,
                                  const HtNeighborPair& d_reply) {
  return {{target.u, target.v},
          {c_reply.first, c_reply.center, c_reply.second},
          {d_reply.first, d_reply.center, d_reply.second}};
}

// verifier 单独遍历 CSR 的全部邻边对，不信任 proof 中保存的 replies。
std::vector<HtNeighborPair> EnumerateRepliesForVerifier(const GraphSnapshot& graph,
                                                        const NodeEdge target,
                                                        const std::int32_t center) {
  std::vector<HtNeighborPair> replies;
  const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(center)];
  const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(center) + 1U];
  for (std::int32_t first_offset = begin; first_offset < end; ++first_offset) {
    const std::int32_t first = graph.neighbors[static_cast<std::size_t>(first_offset)];
    for (std::int32_t second_offset = first_offset + 1; second_offset < end; ++second_offset) {
      const std::int32_t second = graph.neighbors[static_cast<std::size_t>(second_offset)];

      // 故意展开 production predicate，避免 verifier 仅重放候选器布尔值。
      if (CanonicalEdge(first, second) == target) {
        continue;
      }
      const NodeEdge first_edge = CanonicalEdge(center, first);
      const NodeEdge second_edge = CanonicalEdge(center, second);
      const __int128 target_cost = graph.Distance(target.u, target.v);
      const auto survives_two_opt = [&](const NodeEdge edge) {
        const __int128 original = target_cost + graph.Distance(edge.u, edge.v);
        const __int128 orientation0 = static_cast<__int128>(graph.Distance(target.u, edge.v)) +
                                      graph.Distance(target.v, edge.u);
        const __int128 orientation1 = static_cast<__int128>(graph.Distance(target.u, edge.u)) +
                                      graph.Distance(target.v, edge.v);
        return orientation0 >= original || orientation1 >= original;
      };
      if (!survives_two_opt(first_edge) || !survives_two_opt(second_edge)) {
        continue;
      }
      const __int128 original =
          target_cost + graph.Distance(center, first) + graph.Distance(center, second);
      const __int128 replacement = static_cast<__int128>(graph.Distance(first, second)) +
                                   graph.Distance(target.u, center) +
                                   graph.Distance(target.v, center);
      if (original <= replacement) {
        replies.push_back({center, first, second});
      }
    }
  }
  return replies;
}

std::vector<HtCdScreenTask> BuildCdScreenTasks(const GraphSnapshot& graph, const NodeEdge target,
                                               const HtShallowOptions& options) {
  struct RankedNode {
    std::int32_t node{};
    __int128 midpoint_score{};
  };
  std::vector<RankedNode> neighborhood;
  neighborhood.reserve(static_cast<std::size_t>(graph.dimension - 2));
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    if (node == target.u || node == target.v ||
        (options.max_candidate_degree != 0 &&
         static_cast<std::uint32_t>(graph.Degree(node)) > options.max_candidate_degree)) {
      continue;
    }
    // 用二倍中点坐标排序，避免引入浮点数或除法。
    const __int128 dx =
        static_cast<__int128>(2) * graph.points[static_cast<std::size_t>(node)].integer_x -
        graph.points[static_cast<std::size_t>(target.u)].integer_x -
        graph.points[static_cast<std::size_t>(target.v)].integer_x;
    const __int128 dy =
        static_cast<__int128>(2) * graph.points[static_cast<std::size_t>(node)].integer_y -
        graph.points[static_cast<std::size_t>(target.u)].integer_y -
        graph.points[static_cast<std::size_t>(target.v)].integer_y;
    neighborhood.push_back({node, dx * dx + dy * dy});
  }
  std::sort(
      neighborhood.begin(), neighborhood.end(), [](const RankedNode& lhs, const RankedNode& rhs) {
        return std::tie(lhs.midpoint_score, lhs.node) < std::tie(rhs.midpoint_score, rhs.node);
      });
  if (options.max_neighborhood != 0 && neighborhood.size() > options.max_neighborhood) {
    neighborhood.resize(options.max_neighborhood);
  }

  std::vector<HtCdScreenTask> tasks;
  if (neighborhood.size() > 1) {
    tasks.reserve(neighborhood.size() * (neighborhood.size() - 1U) / 2U);
  }
  for (std::size_t first_index = 0; first_index < neighborhood.size(); ++first_index) {
    for (std::size_t second_index = first_index + 1U; second_index < neighborhood.size();
         ++second_index) {
      const std::int32_t c =
          std::min(neighborhood[first_index].node, neighborhood[second_index].node);
      const std::int32_t d =
          std::max(neighborhood[first_index].node, neighborhood[second_index].node);
      tasks.push_back({c, d, static_cast<std::uint8_t>(graph.HasActiveEdge(c, d))});
    }
  }
  return tasks;
}

std::vector<std::uint8_t> ScreenCdTasksCpu(const GraphSnapshot& graph, const NodeEdge target,
                                           const std::vector<HtCdScreenTask>& tasks,
                                           const HtCdMode mode) {
  std::vector<std::uint8_t> flags;
  flags.reserve(tasks.size());
  for (const HtCdScreenTask& task : tasks) {
    flags.push_back(
        static_cast<std::uint8_t>(CandidateIsAdmissible(graph, target, task.c, task.d, mode)));
  }
  return flags;
}

void AppendHtHamiltonRepliesUnchecked(const GraphSnapshot& graph, const NodeEdge target,
                                      const std::int32_t center,
                                      std::vector<HtNeighborPair>* const replies) {
  struct NeighborFilter {
    std::int32_t node{-1};
    std::int64_t center_cost{};
    bool survives_two_opt{false};
  };

  const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(center)];
  const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(center) + 1U];
  std::vector<NeighborFilter> neighbors;
  neighbors.reserve(static_cast<std::size_t>(end - begin));
  for (std::int32_t offset = begin; offset < end; ++offset) {
    const std::int32_t node = graph.neighbors[static_cast<std::size_t>(offset)];
    neighbors.push_back({node, graph.Distance(center, node),
                         EdgesSurviveTwoOpt(graph, target, CanonicalEdge(center, node))});
  }

  const __int128 target_cost = graph.Distance(target.u, target.v);
  const __int128 center_target_cost =
      static_cast<__int128>(graph.Distance(target.u, center)) + graph.Distance(target.v, center);
  for (std::size_t first_index = 0U; first_index < neighbors.size(); ++first_index) {
    const NeighborFilter& first = neighbors[first_index];
    for (std::size_t second_index = first_index + 1U; second_index < neighbors.size();
         ++second_index) {
      const NeighborFilter& second = neighbors[second_index];
      if (CanonicalEdge(first.node, second.node) == target || !first.survives_two_opt ||
          !second.survives_two_opt) {
        continue;
      }
      const __int128 original = target_cost + first.center_cost + second.center_cost;
      const __int128 replacement =
          static_cast<__int128>(graph.Distance(first.node, second.node)) + center_target_cost;
      if (original <= replacement) {
        replies->push_back({center, first.node, second.node});
      }
    }
  }
}

std::vector<HtCdCandidate> FinalizeCdCandidates(const GraphSnapshot& graph, const NodeEdge target,
                                                const HtShallowOptions& options,
                                                const std::vector<HtCdScreenTask>& tasks,
                                                const std::vector<std::uint8_t>& flags) {
  if (flags.size() != tasks.size()) {
    throw std::logic_error("HT c,d 筛选结果数量错误");
  }
  std::vector<std::vector<HtNeighborPair>> reply_cache(static_cast<std::size_t>(graph.dimension));
  std::vector<bool> reply_cached(static_cast<std::size_t>(graph.dimension), false);
  const auto replies_for = [&](const std::int32_t node) -> const std::vector<HtNeighborPair>& {
    const auto index = static_cast<std::size_t>(node);
    if (!reply_cached[index]) {
      AppendHtHamiltonRepliesUnchecked(graph, target, node, &reply_cache[index]);
      reply_cached[index] = true;
    }
    return reply_cache[index];
  };

  std::vector<HtCdCandidate> candidates;
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    if (flags[index] == 0) {
      continue;
    }
    const HtCdScreenTask& task = tasks[index];
    if (!CandidateIsAdmissible(graph, target, task.c, task.d, options.cd_mode)) {
      throw std::runtime_error("HT c,d 候选未通过 CPU 独立复核");
    }
    std::uint64_t reply_product = 0;
    if (!MultiplyWithoutOverflow(replies_for(task.c).size(), replies_for(task.d).size(),
                                 &reply_product)) {
      reply_product = std::numeric_limits<std::uint64_t>::max();
    }
    candidates.push_back({task.c, task.d, reply_product});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const HtCdCandidate& lhs, const HtCdCandidate& rhs) {
              return std::tie(lhs.reply_product, lhs.c, lhs.d) <
                     std::tie(rhs.reply_product, rhs.c, rhs.d);
            });
  if (options.max_cd_candidates != 0 && candidates.size() > options.max_cd_candidates) {
    candidates.resize(options.max_cd_candidates);
  }
  return candidates;
}

} // namespace

std::vector<HtNeighborPair> EnumerateHtHamiltonReplies(const GraphSnapshot& graph,
                                                       const NodeEdge raw_target,
                                                       const std::int32_t center) {
  std::string reason;
  if (!ValidateHtGraph(graph, &reason)) {
    throw std::invalid_argument(reason);
  }
  const NodeEdge target = CanonicalEdge(raw_target.u, raw_target.v);
  if (!ValidateTarget(graph, target, &reason) || center < 0 || center >= graph.dimension ||
      center == target.u || center == target.v) {
    throw std::invalid_argument(reason.empty() ? "HT reply 中心点非法" : reason);
  }
  std::vector<HtNeighborPair> replies;
  AppendHtHamiltonRepliesUnchecked(graph, target, center, &replies);
  return replies;
}

HtHamiltonReplyBatchResult EvaluateHtHamiltonReplies(const GraphSnapshot& graph,
                                                     const NodeEdge raw_target,
                                                     const std::vector<std::int32_t>& centers,
                                                     const PathCompatibilityBackend backend) {
  HtHamiltonReplyBatchResult result;
  const SteadyClock::time_point validation_begin = SteadyClock::now();
  std::string reason;
  if (!ValidateHtGraph(graph, &reason)) {
    throw std::invalid_argument(reason);
  }
  const NodeEdge target = CanonicalEdge(raw_target.u, raw_target.v);
  if (!ValidateTarget(graph, target, &reason)) {
    throw std::invalid_argument(reason);
  }
  if (backend != PathCompatibilityBackend::kAuto && backend != PathCompatibilityBackend::kCpu &&
      backend != PathCompatibilityBackend::kCuda) {
    throw std::invalid_argument("未知 HT Hamilton reply 后端");
  }
  for (const std::int32_t center : centers) {
    if (center < 0 || center >= graph.dimension || center == target.u || center == target.v) {
      throw std::invalid_argument("HT reply 中心点非法");
    }
  }
  result.validation_ms = ElapsedMilliseconds(validation_begin);

  result.offsets.reserve(centers.size() + 1U);
  result.offsets.push_back(0U);
  std::vector<std::int32_t> cache_index(static_cast<std::size_t>(graph.dimension), -1);
  std::vector<std::vector<HtNeighborPair>> reply_cache;
  reply_cache.reserve(std::min(centers.size(), static_cast<std::size_t>(graph.dimension)));
  const SteadyClock::time_point cpu_begin = SteadyClock::now();
  for (const std::int32_t center : centers) {
    std::int32_t& cached = cache_index[static_cast<std::size_t>(center)];
    if (cached < 0) {
      if (reply_cache.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("HT Hamilton reply 中心缓存索引溢出");
      }
      cached = static_cast<std::int32_t>(reply_cache.size());
      reply_cache.emplace_back();
      AppendHtHamiltonRepliesUnchecked(graph, target, center, &reply_cache.back());
      ++result.unique_centers;
      const std::uint64_t degree = static_cast<std::uint64_t>(graph.Degree(center));
      const std::uint64_t pair_count = degree < 2U ? 0U : degree * (degree - 1U) / 2U;
      if (pair_count > std::numeric_limits<std::uint64_t>::max() - result.neighbor_pairs_tested) {
        throw std::overflow_error("HT Hamilton reply 邻边对计数溢出");
      }
      result.neighbor_pairs_tested += pair_count;
    }
    const std::vector<HtNeighborPair>& center_replies =
        reply_cache[static_cast<std::size_t>(cached)];
    if (center_replies.size() > std::numeric_limits<std::uint64_t>::max() - result.offsets.back()) {
      throw std::overflow_error("HT Hamilton reply 总数溢出");
    }
    result.replies.insert(result.replies.end(), center_replies.begin(), center_replies.end());
    result.offsets.push_back(result.offsets.back() + center_replies.size());
  }
  result.cpu_enumerate_ms = ElapsedMilliseconds(cpu_begin);
  result.cpu_verified = true;
  if (backend == PathCompatibilityBackend::kCpu) {
    result.backend = "cpu";
    return result;
  }

  if (!detail::HtHamiltonReplyCudaAvailable(&reason)) {
    if (backend == PathCompatibilityBackend::kCuda) {
      throw std::runtime_error("CUDA HT Hamilton reply 后端不可用: " + reason);
    }
    result.backend = "cpu-fallback";
    return result;
  }

  detail::HtHamiltonReplyDeviceBatch cuda_batch;
  const SteadyClock::time_point cuda_begin = SteadyClock::now();
  try {
    cuda_batch =
        detail::EvaluateHtHamiltonRepliesCuda(graph, target, centers, &result.selected_device);
  } catch (const std::exception&) {
    result.cuda_evaluate_ms = ElapsedMilliseconds(cuda_begin);
    if (backend == PathCompatibilityBackend::kCuda) {
      throw;
    }
    result.selected_device = -1;
    result.backend = "cpu-fallback";
    return result;
  }
  result.cuda_evaluate_ms = ElapsedMilliseconds(cuda_begin);
  const SteadyClock::time_point compare_begin = SteadyClock::now();
  const bool matches = cuda_batch.offsets == result.offsets && cuda_batch.replies == result.replies;
  result.cuda_compare_ms = ElapsedMilliseconds(compare_begin);
  if (!matches) {
    throw std::logic_error("CUDA HT Hamilton replies 与 CPU 完整枚举不一致");
  }
  result.backend = "cuda";
  return result;
}

HtEndReplyBatchResult EvaluateHtEndReplies(const GraphSnapshot& graph,
                                           const std::vector<HtEndReplyTask>& tasks,
                                           const PathCompatibilityBackend backend) {
  std::string reason;
  if (!ValidateHtGraph(graph, &reason)) {
    throw std::invalid_argument(reason);
  }
  if (backend != PathCompatibilityBackend::kAuto && backend != PathCompatibilityBackend::kCpu &&
      backend != PathCompatibilityBackend::kCuda) {
    throw std::invalid_argument("未知 HT end reply 后端");
  }

  HtEndReplyBatchResult result;
  result.offsets.reserve(tasks.size() + 1U);
  result.offsets.push_back(0U);
  for (const HtEndReplyTask& task : tasks) {
    if (task.endpoint < 0 || task.endpoint >= graph.dimension || task.internal_neighbor < 0 ||
        task.internal_neighbor >= graph.dimension || task.endpoint == task.internal_neighbor ||
        !graph.HasActiveEdge(task.endpoint, task.internal_neighbor)) {
      throw std::invalid_argument("HT end reply task 必须指定一条活动的路径内部边");
    }
    const std::int32_t begin = graph.row_offsets[static_cast<std::size_t>(task.endpoint)];
    const std::int32_t end = graph.row_offsets[static_cast<std::size_t>(task.endpoint) + 1U];
    const std::uint64_t reply_count = static_cast<std::uint64_t>(end - begin - 1);
    if (reply_count > std::numeric_limits<std::uint64_t>::max() - result.offsets.back()) {
      throw std::overflow_error("HT end reply 总数溢出");
    }
    for (std::int32_t offset = begin; offset < end; ++offset) {
      const std::int32_t neighbor = graph.neighbors[static_cast<std::size_t>(offset)];
      if (neighbor != task.internal_neighbor) {
        result.replies.push_back(CanonicalEdge(task.endpoint, neighbor));
      }
    }
    result.offsets.push_back(result.offsets.back() + reply_count);
  }
  result.cpu_verified = true;
  if (backend == PathCompatibilityBackend::kCpu) {
    result.backend = "cpu";
    return result;
  }

  if (!detail::HtEndReplyCudaAvailable(&reason)) {
    if (backend == PathCompatibilityBackend::kCuda) {
      throw std::runtime_error("CUDA HT end reply 后端不可用: " + reason);
    }
    result.backend = "cpu-fallback";
    return result;
  }

  detail::HtEndReplyDeviceBatch cuda_batch;
  try {
    cuda_batch = detail::EvaluateHtEndRepliesCuda(graph, tasks, &result.selected_device);
  } catch (const std::exception&) {
    if (backend == PathCompatibilityBackend::kCuda) {
      throw;
    }
    result.selected_device = -1;
    result.backend = "cpu-fallback";
    return result;
  }
  if (cuda_batch.offsets != result.offsets || cuda_batch.replies != result.replies) {
    throw std::logic_error("CUDA HT end replies 与 CPU 完整枚举不一致");
  }
  result.backend = "cuda";
  return result;
}

std::vector<HtCdCandidate> GenerateHtCdCandidates(const GraphSnapshot& graph,
                                                  const NodeEdge raw_target,
                                                  const HtShallowOptions& options) {
  std::string reason;
  if (!ValidateHtGraph(graph, &reason)) {
    throw std::invalid_argument(reason);
  }
  const NodeEdge target = CanonicalEdge(raw_target.u, raw_target.v);
  if (!ValidateTarget(graph, target, &reason) || !IsKnownCdMode(options.cd_mode)) {
    throw std::invalid_argument(reason.empty() ? "HT c,d 模式非法" : reason);
  }
  const std::vector<HtCdScreenTask> tasks = BuildCdScreenTasks(graph, target, options);
  return FinalizeCdCandidates(graph, target, options, tasks,
                              ScreenCdTasksCpu(graph, target, tasks, options.cd_mode));
}

HtCdBatchResult EvaluateHtCdCandidates(const GraphSnapshot& graph, const NodeEdge raw_target,
                                       const HtShallowOptions& options) {
  std::string reason;
  if (!ValidateHtGraph(graph, &reason)) {
    throw std::invalid_argument(reason);
  }
  const NodeEdge target = CanonicalEdge(raw_target.u, raw_target.v);
  if (!ValidateTarget(graph, target, &reason) || !IsKnownCdMode(options.cd_mode)) {
    throw std::invalid_argument(reason.empty() ? "HT c,d 模式非法" : reason);
  }
  if (options.candidate_backend != PathCompatibilityBackend::kAuto &&
      options.candidate_backend != PathCompatibilityBackend::kCpu &&
      options.candidate_backend != PathCompatibilityBackend::kCuda) {
    throw std::invalid_argument("未知 HT c,d 候选后端");
  }

  const std::vector<HtCdScreenTask> tasks = BuildCdScreenTasks(graph, target, options);
  const std::vector<std::uint8_t> cpu_flags =
      ScreenCdTasksCpu(graph, target, tasks, options.cd_mode);
  HtCdBatchResult result;
  bool use_cuda = options.candidate_backend == PathCompatibilityBackend::kCuda;
  if (options.candidate_backend == PathCompatibilityBackend::kAuto) {
    use_cuda = detail::HtCdCudaAvailable(&reason);
  }
  if (use_cuda) {
    if (!detail::HtCdCudaAvailable(&reason)) {
      throw std::runtime_error("CUDA HT c,d 后端不可用: " + reason);
    }
    const std::vector<std::uint8_t> gpu_flags = detail::ScreenHtCdCandidatesCuda(
        graph, target, tasks, options.cd_mode, &result.selected_device);
    if (gpu_flags != cpu_flags) {
      throw std::runtime_error("CUDA HT c,d flags 与 CPU 规范结果不一致");
    }
    result.backend = "cuda";
    result.cpu_verified = true;
  } else {
    result.backend = "cpu";
    result.cpu_verified = true;
  }
  result.candidates = FinalizeCdCandidates(graph, target, options, tasks, cpu_flags);
  return result;
}

HtShallowResult ProveEdgeByShallowHt(const GraphSnapshot& graph, const NodeEdge raw_target,
                                     const HtShallowOptions& options) {
  HtShallowResult result;
  HtShallowProof& proof = result.proof;
  proof.snapshot_hash = graph.ContentHash();
  proof.target_edge = CanonicalEdge(raw_target.u, raw_target.v);
  proof.cd_mode = options.cd_mode;

  HtCdBatchResult candidate_batch;
  try {
    candidate_batch = EvaluateHtCdCandidates(graph, proof.target_edge, options);
  } catch (const std::exception& error) {
    if (options.candidate_backend == PathCompatibilityBackend::kAuto) {
      try {
        HtShallowOptions cpu_options = options;
        cpu_options.candidate_backend = PathCompatibilityBackend::kCpu;
        candidate_batch = EvaluateHtCdCandidates(graph, proof.target_edge, cpu_options);
      } catch (const std::exception& cpu_error) {
        proof.reason = cpu_error.what();
        return result;
      }
    } else if (options.candidate_backend == PathCompatibilityBackend::kCuda) {
      result.status = HtSearchStatus::kUnresolved;
      proof.reason = std::string("CUDA HT c,d 筛选失败: ") + error.what();
      return result;
    } else {
      proof.reason = error.what();
      return result;
    }
  }
  const std::vector<HtCdCandidate>& candidates = candidate_batch.candidates;
  if (candidates.empty()) {
    result.status = HtSearchStatus::kUnresolved;
    proof.reason = "没有可用的浅层 c,d Tutte move";
    return result;
  }

  bool budget_blocked = false;
  for (const HtCdCandidate& candidate : candidates) {
    ++proof.cd_candidates_tested;
    const std::vector<HtNeighborPair> c_replies =
        EnumerateHtHamiltonReplies(graph, proof.target_edge, candidate.c);
    const std::vector<HtNeighborPair> d_replies =
        EnumerateHtHamiltonReplies(graph, proof.target_edge, candidate.d);
    std::uint64_t reply_product = 0;
    if (!MultiplyWithoutOverflow(c_replies.size(), d_replies.size(), &reply_product) ||
        (options.max_reply_combinations != 0 && reply_product > options.max_reply_combinations)) {
      budget_blocked = true;
      continue;
    }

    std::vector<HtReplyProof> candidate_records;
    candidate_records.reserve(static_cast<std::size_t>(reply_product));
    bool candidate_proven = true;
    for (const HtNeighborPair& c_reply : c_replies) {
      for (const HtNeighborPair& d_reply : d_replies) {
        ++proof.reply_combinations_tested;
        HtReplyProof record;
        record.c_reply = c_reply;
        record.d_reply = d_reply;
        const NormalizedPathSystem paths = NormalizePathSystem(
            BuildReplyPaths(proof.target_edge, c_reply, d_reply), graph.dimension);
        if (!paths.valid) {
          record.path_infeasible = true;
          candidate_records.push_back(std::move(record));
          continue;
        }
        record.leaf_proof =
            ProvePathSystemByKOpt(graph, paths, proof.target_edge, options.leaf_options);
        if (!record.leaf_proof.proven) {
          candidate_proven = false;
          break;
        }
        candidate_records.push_back(std::move(record));
      }
      if (!candidate_proven) {
        break;
      }
    }
    if (!candidate_proven) {
      continue;
    }

    proof.proven = true;
    proof.reason = "一个 c,d Tutte move 的全部 Hamilton replies 均已证明";
    proof.c = candidate.c;
    proof.d = candidate.d;
    proof.replies = std::move(candidate_records);
    std::string verify_reason;
    if (!VerifyHtShallowProof(graph, proof, &verify_reason)) {
      proof.proven = false;
      proof.reason = "内部浅层 HT 复核失败: " + verify_reason;
      result.status = HtSearchStatus::kInvalid;
      return result;
    }
    result.status = HtSearchStatus::kProven;
    return result;
  }

  result.status = HtSearchStatus::kUnresolved;
  proof.reason = budget_blocked ? "浅层 HT reply 预算不足或叶节点未解决"
                                : "所有浅层 c,d moves 都存在未解决 reply";
  return result;
}

bool VerifyHtShallowProof(const GraphSnapshot& graph, const HtShallowProof& proof,
                          std::string* const reason) {
  std::string graph_reason;
  if (!proof.proven || proof.snapshot_hash != graph.ContentHash() ||
      !ValidateHtGraph(graph, &graph_reason) ||
      !ValidateTarget(graph, proof.target_edge, &graph_reason) ||
      !CandidateIsAdmissible(graph, proof.target_edge, proof.c, proof.d, proof.cd_mode)) {
    SetReason(reason,
              graph_reason.empty() ? "浅层 HT proof 的状态、快照或 c,d move 非法" : graph_reason);
    return false;
  }
  const std::vector<HtNeighborPair> c_replies =
      EnumerateRepliesForVerifier(graph, proof.target_edge, proof.c);
  const std::vector<HtNeighborPair> d_replies =
      EnumerateRepliesForVerifier(graph, proof.target_edge, proof.d);
  std::uint64_t expected_count = 0;
  if (!MultiplyWithoutOverflow(c_replies.size(), d_replies.size(), &expected_count) ||
      expected_count != proof.replies.size()) {
    SetReason(reason, "浅层 HT proof 没有覆盖完整 Hamilton reply 笛卡尔积");
    return false;
  }

  std::size_t record_index = 0;
  for (const HtNeighborPair& c_reply : c_replies) {
    for (const HtNeighborPair& d_reply : d_replies) {
      const HtReplyProof& record = proof.replies[record_index++];
      if (record.c_reply != c_reply || record.d_reply != d_reply) {
        SetReason(reason, "浅层 HT reply 顺序或节点不一致");
        return false;
      }
      const NormalizedPathSystem paths = NormalizePathSystem(
          BuildReplyPaths(proof.target_edge, c_reply, d_reply), graph.dimension);
      if (!paths.valid) {
        if (!record.path_infeasible || record.leaf_proof.proven) {
          SetReason(reason, "无效路径 reply 没有使用纯 path-infeasibility 证明");
          return false;
        }
        continue;
      }
      if (record.path_infeasible) {
        SetReason(reason, "合法路径 reply 被错误标为 infeasible");
        return false;
      }
      std::string leaf_reason;
      if (!VerifyPathSystemKOptProof(graph, paths, proof.target_edge, record.leaf_proof,
                                     &leaf_reason)) {
        SetReason(reason, "浅层 HT leaf proof 失败: " + leaf_reason);
        return false;
      }
    }
  }
  SetReason(reason, "OK");
  return true;
}

} // namespace cudaee
