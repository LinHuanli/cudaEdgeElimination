#include "../fgpu/resident_backend.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool ResidentEliminationCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前构建未启用 CUDA";
  }
  return false;
}

ResidentGpuResult RunResidentEliminationCuda(const GraphSnapshot&, const std::vector<std::uint8_t>&,
                                             const ResidentGpuOptions&) {
  throw std::runtime_error("当前构建未启用 resident GPU CUDA 后端");
}

QuickHsPathDifferentialResult RunQuickHsPathDifferentialCuda(const int, const std::uint32_t) {
  throw std::runtime_error("当前构建未启用 resident GPU CUDA 后端");
}

} // namespace cudaee::detail
