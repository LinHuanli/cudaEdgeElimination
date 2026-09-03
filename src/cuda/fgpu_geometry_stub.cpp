#include "../fgpu/geometry_backend.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool GeometryCudaBackendAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "构建时未启用 CUDA";
  }
  return false;
}

GeometryProposalBatch FindGeometryCandidatesCuda(const GraphSnapshot&, const GeometryOptions&) {
  throw std::runtime_error("构建时未启用 CUDA geometry 后端");
}

} // namespace cudaee::detail
