#include "cuda_edge_elimination/elimination.hpp"

#include <stdexcept>

namespace cudaee {

bool CudaBackendAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA 后端";
  }
  return false;
}

std::vector<Candidate> FindJvCandidatesCuda(const GraphSnapshot&, int*, JvCudaCacheUsage*) {
  throw std::runtime_error("当前二进制未编译 CUDA 后端");
}

void ClearJvCudaCache() {}

} // namespace cudaee
