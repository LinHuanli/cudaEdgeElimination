#pragma once

#include <string>

namespace cudaee::detail {

// 返回当前线程的强制 CUDA device ordinal；-1 表示沿用后端自动选择。
[[nodiscard]] int CudaDevicePreferenceForCurrentThread() noexcept;

// device=-1 清除强制选择；非负值会立即验证可见范围并激活对应设备。
[[nodiscard]] bool SetCudaDevicePreferenceForCurrentThread(int device, std::string* reason);

// 返回当前进程可见的 CUDA 设备数；失败时返回 0 并填写原因。
[[nodiscard]] int VisibleCudaDeviceCount(std::string* reason);

} // namespace cudaee::detail
