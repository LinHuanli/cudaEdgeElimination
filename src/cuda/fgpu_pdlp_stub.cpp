#include "../fgpu/pdlp_backend.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool NativePdlpCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "构建时未启用 CUDA";
  }
  return false;
}

NativePdlpDeviceResult SolveDegreeRelaxationCuda(const GraphSnapshot&, const PdlpOptions&) {
  throw std::runtime_error("构建时未启用 native PDLP CUDA 后端");
}

} // namespace cudaee::detail
