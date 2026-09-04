#include "quick_hs_verifier.hpp"

#include "quick_hs_predicate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace cudaee::detail {

QuickHsVerificationData BuildQuickHsVerificationData(const GraphSnapshot& graph) {
  if (graph.dimension <= 0 || graph.edges.empty()) {
    throw std::invalid_argument("Quick-HS verifier 需要非空图");
  }
  const std::size_t dimension = static_cast<std::size_t>(graph.dimension);
  if (dimension > std::numeric_limits<std::size_t>::max() / dimension) {
    throw std::overflow_error("Quick-HS verifier 稠密视图大小溢出");
  }
  const std::size_t matrix_size = dimension * dimension;
  QuickHsVerificationData result;
  result.dimension = graph.dimension;
  result.degree.resize(dimension);
  result.neighbors.assign(matrix_size, -1);
  result.distance.resize(matrix_size);
  result.active.assign(matrix_size, 0U);

  for (std::int32_t from = 0; from < graph.dimension; ++from) {
    for (std::int32_t to = 0; to < graph.dimension; ++to) {
      result.distance[static_cast<std::size_t>(from) * dimension + static_cast<std::size_t>(to)] =
          graph.Distance(from, to);
    }
  }
  for (const Edge& edge : graph.edges) {
    if (!edge.active) {
      continue;
    }
    result.active[static_cast<std::size_t>(edge.u) * dimension + static_cast<std::size_t>(edge.v)] =
        1U;
    result.active[static_cast<std::size_t>(edge.v) * dimension + static_cast<std::size_t>(edge.u)] =
        1U;
  }
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    std::vector<std::int32_t> row;
    row.reserve(dimension);
    for (std::int32_t other = 0; other < graph.dimension; ++other) {
      if (result.active[static_cast<std::size_t>(node) * dimension +
                        static_cast<std::size_t>(other)] != 0U) {
        row.push_back(other);
      }
    }
    std::sort(row.begin(), row.end(), [&](const std::int32_t lhs, const std::int32_t rhs) {
      return std::tie(result.distance[static_cast<std::size_t>(node) * dimension +
                                      static_cast<std::size_t>(lhs)],
                      lhs) < std::tie(result.distance[static_cast<std::size_t>(node) * dimension +
                                                      static_cast<std::size_t>(rhs)],
                                      rhs);
    });
    result.degree[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(row.size());
    std::copy(row.begin(), row.end(),
              result.neighbors.begin() + static_cast<std::ptrdiff_t>(node) * graph.dimension);
  }
  return result;
}

bool VerifyQuickHsCandidate(const GraphSnapshot& graph,
                            const QuickHsVerificationData& verification_data,
                            const Candidate& candidate, std::string* const reason) {
  const auto fail = [&](const std::string& message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  if (candidate.method != EliminationMethod::kGpuQuickHs || candidate.edge_id < 0 ||
      static_cast<std::size_t>(candidate.edge_id) >= graph.edges.size()) {
    return fail("Quick-HS 方法或 edge id 非法");
  }
  if (verification_data.dimension != graph.dimension ||
      verification_data.degree.size() != static_cast<std::size_t>(graph.dimension)) {
    return fail("Quick-HS 共享快照维度不匹配");
  }
  const Edge& edge = graph.edges[static_cast<std::size_t>(candidate.edge_id)];
  if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
    return fail("Quick-HS 目标边不活动或端点度数不满足前提");
  }
  if (candidate.witness < 0 || candidate.witness >= graph.dimension ||
      candidate.second_witness < 0 || candidate.second_witness >= graph.dimension ||
      candidate.witness == candidate.second_witness) {
    return fail("Quick-HS c,d 见证越界或重复");
  }

  const quick_hs::GraphView view{.dimension = verification_data.dimension,
                                 .degree = verification_data.degree.data(),
                                 .neighbors = verification_data.neighbors.data(),
                                 .distance = verification_data.distance.data(),
                                 .active = verification_data.active.data()};
  if (!quick_hs::CanEliminateWithWitness(view, edge.u, edge.v, candidate.witness,
                                         candidate.second_witness)) {
    return fail("Quick-HS c,d 的某个 Hamilton reply 未被局部整数证明关闭");
  }
  return true;
}

} // namespace cudaee::detail
