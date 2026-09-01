#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool HtPathAppendCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA 后端";
  }
  return false;
}

HtPathAppendDeviceBatch EvaluateHtPathAppendsCuda(const std::int32_t,
                                                  const std::vector<HtPathStateSpan>&,
                                                  const std::vector<HtPathNodeRecord>&,
                                                  const std::vector<NodeEdge>&,
                                                  const std::vector<HtPathAppendTask>&, int*) {
  throw std::runtime_error("当前二进制未编译 CUDA HT path-append 后端");
}

} // namespace cudaee::detail
