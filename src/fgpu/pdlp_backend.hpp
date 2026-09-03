#pragma once

#include "cuda_edge_elimination/fgpu.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cudaee::detail {

struct NativePdlpDeviceResult {
  std::vector<double> vertex_dual;
  int selected_device{-1};
  std::uint32_t iterations{};
  double solve_ms{};
  std::string implementation{"edge-atomic"};
};

[[nodiscard]] NativePdlpDeviceResult SolveDegreeRelaxationCuda(const GraphSnapshot& graph,
                                                               const PdlpOptions& options);

[[nodiscard]] bool NativePdlpCudaAvailable(std::string* reason);

} // namespace cudaee::detail
