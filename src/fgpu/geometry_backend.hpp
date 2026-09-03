#pragma once

#include "cuda_edge_elimination/fgpu.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cudaee::detail {

struct GeometryProposalBatch {
  std::uint32_t witnesses_per_edge{};
  std::vector<std::int32_t> first_witness;
  std::vector<std::int32_t> second_witness;
  std::string backend;
  int selected_device{-1};
  double nearest_ms{};
  double upload_ms{};
  double kernel_ms{};
  double download_ms{};
};

[[nodiscard]] GeometryProposalBatch FindGeometryCandidatesCuda(const GraphSnapshot& graph,
                                                               const GeometryOptions& options);

[[nodiscard]] bool GeometryCudaBackendAvailable(std::string* reason);

} // namespace cudaee::detail
