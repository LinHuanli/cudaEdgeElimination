#include "cuda_edge_elimination/path_system.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool PathCompatibilityCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA 后端";
  }
  return false;
}

std::vector<std::uint8_t> LookupPathCompatibilityCuda(const PathCompatibilityTable&,
                                                      const std::vector<PathCompatibilityQuery>&,
                                                      int*) {
  throw std::runtime_error("当前二进制未编译 CUDA 路径兼容后端");
}

} // namespace cudaee::detail
