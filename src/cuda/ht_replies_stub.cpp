#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool HtHamiltonReplyCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA 后端";
  }
  return false;
}

HtHamiltonReplyDeviceBatch EvaluateHtHamiltonRepliesCuda(const GraphSnapshot&, const NodeEdge,
                                                         const std::vector<std::int32_t>&, int*) {
  throw std::runtime_error("当前二进制未编译 CUDA HT reply 后端");
}

bool HtEndReplyCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA 后端";
  }
  return false;
}

HtEndReplyDeviceBatch EvaluateHtEndRepliesCuda(const GraphSnapshot&,
                                               const std::vector<HtEndReplyTask>&, int*) {
  throw std::runtime_error("当前二进制未编译 CUDA HT end reply 后端");
}

} // namespace cudaee::detail
