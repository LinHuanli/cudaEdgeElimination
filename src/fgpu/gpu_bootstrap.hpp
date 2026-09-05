#pragma once

#include "cuda_edge_elimination/fgpu_metrics.hpp"
#include "cuda_edge_elimination/graph.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace cudaee::detail {

using GpuBootstrapMetrics = FgpuBootstrapMetrics;

// RAII 所有权跨 bootstrap/消除阶段存活，缓存不会下载后再上传。
// 公共头不依赖 CUDA，CPU 构建提供显式失败的 stub。
class GpuBootstrap {
public:
  GpuBootstrap(const GraphSnapshot& coordinates, int device);
  ~GpuBootstrap();
  GpuBootstrap(const GpuBootstrap&) = delete;
  GpuBootstrap& operator=(const GpuBootstrap&) = delete;
  void BuildCompleteGraph(GraphSnapshot* graph);
  void GenerateIncumbent();
  void BuildPermutationCatalog();
  [[nodiscard]] const std::int64_t* distances() const;
  [[nodiscard]] const std::uint8_t* permutations() const;
  [[nodiscard]] int device() const;
  [[nodiscard]] const GpuBootstrapMetrics& metrics() const;
  [[nodiscard]] const std::vector<std::int32_t>& tour() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cudaee::detail
