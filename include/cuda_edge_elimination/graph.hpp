#pragma once

#include "cuda_edge_elimination/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cudaee {

class GraphSnapshot {
public:
  static GraphSnapshot Load(const std::filesystem::path& tsp_path,
                            const std::filesystem::path& edge_path);
  // 从 TSPLIB 坐标直接构造规范排序的完全图，供端到端删边实验使用。
  static GraphSnapshot LoadComplete(const std::filesystem::path& tsp_path);

  void RebuildCsr();
  void WriteActiveEdges(const std::filesystem::path& path) const;

  [[nodiscard]] std::int64_t Distance(std::int32_t u, std::int32_t v) const;
  [[nodiscard]] std::size_t ActiveEdgeCount() const;
  [[nodiscard]] std::uint64_t ContentHash() const;
  [[nodiscard]] std::int32_t Degree(std::int32_t vertex) const;
  [[nodiscard]] bool HasActiveEdge(std::int32_t u, std::int32_t v) const;

  std::int32_t dimension{};
  DistanceType distance_type{DistanceType::kEuc2D};
  bool integer_coordinates{false};
  bool integer_distance_safe{false};
  std::vector<Point> points;
  std::vector<Edge> edges;
  std::vector<std::int32_t> row_offsets;
  std::vector<std::int32_t> neighbors;
  std::vector<std::int32_t> csr_edge_ids;
  std::vector<std::int64_t> csr_weights;
};

[[nodiscard]] std::string HexHash(std::uint64_t value);

} // namespace cudaee
