#pragma once

#include "cuda_edge_elimination/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace cudaee {

struct ProtectedTourCheck {
  std::int64_t cost{};
  std::size_t missing_edges{};
  std::uint64_t tour_hash{};
};

// 严格读取 TSPLIB TOUR_SECTION；节点转换为内部 0-based 编号。
[[nodiscard]] std::vector<std::int32_t> ReadTsplibTour(const std::filesystem::path& path,
                                                       std::int32_t expected_dimension);

// 成本在完整度量图上计算，missing_edges 只针对当前活动稀疏图。
[[nodiscard]] ProtectedTourCheck CheckProtectedTour(const GraphSnapshot& graph,
                                                    const std::vector<std::int32_t>& tour);

} // namespace cudaee
