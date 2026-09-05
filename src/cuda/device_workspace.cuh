#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaee::detail {

inline void CheckWorkspaceCuda(const cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA workspace: ") + cudaGetErrorString(status));
  }
}

// 不收缩的设备 workspace。owner 的析构和扩容均恢复正确 device，避免
// 切换可见 GPU 后错误地释放其他上下文的缓冲区。
template <typename T> class CudaWorkspace {
public:
  explicit CudaWorkspace(const int device) : device_(device) {}
  ~CudaWorkspace() {
    if (data_ != nullptr) {
      static_cast<void>(cudaSetDevice(device_));
      static_cast<void>(cudaFree(data_));
    }
  }
  CudaWorkspace(const CudaWorkspace&) = delete;
  CudaWorkspace& operator=(const CudaWorkspace&) = delete;
  void Reserve(const std::uint64_t count) {
    if (count <= count_) {
      return;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("CUDA workspace 容量溢出");
    }
    CheckWorkspaceCuda(cudaSetDevice(device_));
    T* next = nullptr;
    CheckWorkspaceCuda(cudaMalloc(&next, static_cast<std::size_t>(count) * sizeof(T)));
    // scratch 内容不跨扩容复用，只有分配成功才替换旧缓冲区。
    if (data_ != nullptr) {
      const cudaError_t status = cudaFree(data_);
      if (status != cudaSuccess) {
        static_cast<void>(cudaFree(next));
        CheckWorkspaceCuda(status);
      }
    }
    data_ = next;
    count_ = count;
  }
  [[nodiscard]] T* get() const { return data_; }
  [[nodiscard]] std::uint64_t bytes() const { return count_ * sizeof(T); }

private:
  int device_;
  T* data_{};
  std::uint64_t count_{};
};

} // namespace cudaee::detail
