#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <stdexcept>

namespace cudaee::detail {

bool HtCdCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA 后端";
  }
  return false;
}

std::vector<std::uint8_t> ScreenHtCdCandidatesCuda(const GraphSnapshot&, NodeEdge,
                                                   const std::vector<HtCdScreenTask>&, HtCdMode,
                                                   int*) {
  throw std::runtime_error("当前二进制未编译 CUDA HT c,d 后端");
}

bool HtWavefrontCudaAvailable(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "当前二进制未编译 CUDA HT wavefront 后端";
  }
  return false;
}

HtWavefrontDeviceResult EvaluateHtWavefrontCuda(const std::vector<HtWavefrontStateTask>&,
                                                const std::vector<HtWavefrontMoveTask>&,
                                                const std::vector<HtWavefrontReplyTask>&,
                                                const std::vector<std::uint32_t>&, std::uint32_t,
                                                int*) {
  throw std::runtime_error("当前二进制未编译 CUDA HT wavefront 后端");
}

} // namespace cudaee::detail
