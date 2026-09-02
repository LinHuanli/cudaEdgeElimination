#include "cuda_edge_elimination/local_search.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool ExactTourCostCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前构建未启用 CUDA";
  }
  return false;
}

std::vector<std::int64_t> EvaluateExactTourCostsCuda(const GraphSnapshot&,
                                                     const std::vector<ExactTourCostTask>&, int*,
                                                     std::uint64_t*) {
  throw std::runtime_error("当前构建未启用 CUDA exact DP");
}

void ClearExactTourCostCudaCache() {}

} // namespace cudaee::detail
