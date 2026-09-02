#include "cuda_edge_elimination/cuda_device_affinity.hpp"

#include <cuda_runtime_api.h>

#include <string>

namespace cudaee::detail {
namespace {

thread_local int g_cuda_device_preference = -1;

std::string CudaFailure(const char* const operation, const cudaError_t status) {
  return std::string(operation) + ": " + cudaGetErrorString(status);
}

} // namespace

int CudaDevicePreferenceForCurrentThread() noexcept { return g_cuda_device_preference; }

int VisibleCudaDeviceCount(std::string* const reason) {
  if (reason != nullptr) {
    reason->clear();
  }
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    if (reason != nullptr) {
      *reason = CudaFailure("cudaGetDeviceCount", status);
    }
    return 0;
  }
  if (device_count == 0 && reason != nullptr) {
    *reason = "没有可见 CUDA 设备";
  }
  return device_count;
}

bool SetCudaDevicePreferenceForCurrentThread(const int device, std::string* const reason) {
  if (reason != nullptr) {
    reason->clear();
  }
  if (device == -1) {
    g_cuda_device_preference = -1;
    return true;
  }
  if (device < -1) {
    if (reason != nullptr) {
      *reason = "CUDA device ordinal 不能小于 -1";
    }
    return false;
  }
  const int device_count = VisibleCudaDeviceCount(reason);
  if (device >= device_count) {
    if (reason != nullptr && device_count > 0) {
      *reason = "CUDA device ordinal 超出当前可见范围";
    }
    return false;
  }
  const cudaError_t status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    if (reason != nullptr) {
      *reason = CudaFailure("cudaSetDevice(target worker)", status);
    }
    return false;
  }
  g_cuda_device_preference = device;
  return true;
}

} // namespace cudaee::detail
