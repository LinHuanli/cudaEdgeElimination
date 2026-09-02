#include "cuda_edge_elimination/cuda_device_affinity.hpp"

namespace cudaee::detail {

int CudaDevicePreferenceForCurrentThread() noexcept { return -1; }

int VisibleCudaDeviceCount(std::string* const reason) {
  if (reason != nullptr) {
    *reason = "构建时未启用 CUDA";
  }
  return 0;
}

bool SetCudaDevicePreferenceForCurrentThread(const int device, std::string* const reason) {
  if (reason != nullptr) {
    reason->clear();
  }
  if (device == -1) {
    return true;
  }
  if (reason != nullptr) {
    *reason = "构建时未启用 CUDA，不能绑定 target worker";
  }
  return false;
}

} // namespace cudaee::detail
