#include "cuda_edge_elimination/cuda_device_affinity.hpp"
#include "cuda_edge_elimination/path_system.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaee::detail {
namespace {

void CheckCuda(const cudaError_t status, const char* const operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(const std::size_t count) : count_(count) {
    if (count_ != 0) {
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(path compatibility)");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] T* get() { return data_; }
  [[nodiscard]] const T* get() const { return data_; }

  void CopyFromHost(const T* const source) {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(data_, source, sizeof(T) * count_, cudaMemcpyHostToDevice),
                "cudaMemcpy H2D(path compatibility)");
    }
  }

  void CopyToHost(T* const destination) const {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(path compatibility)");
    }
  }

private:
  T* data_{nullptr};
  std::size_t count_{};
};

int SelectDevice(std::string* const reason) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    if (reason != nullptr) {
      *reason =
          count_status == cudaSuccess ? "没有可见 CUDA 设备" : cudaGetErrorString(count_status);
    }
    return -1;
  }

  const int forced_device = CudaDevicePreferenceForCurrentThread();
  if (forced_device >= 0) {
    if (forced_device >= device_count) {
      if (reason != nullptr) {
        *reason = "path compatibility 强制 CUDA device ordinal 超出当前可见范围";
      }
      return -1;
    }
    const cudaError_t select_status = cudaSetDevice(forced_device);
    if (select_status != cudaSuccess) {
      if (reason != nullptr) {
        *reason = std::string("cudaSetDevice(path compatibility forced): ") +
                  cudaGetErrorString(select_status);
      }
      return -1;
    }
    return forced_device;
  }

  int best_device = -1;
  std::size_t best_free_bytes = 0;
  for (int device = 0; device < device_count; ++device) {
    if (cudaSetDevice(device) != cudaSuccess) {
      continue;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
        (best_device < 0 || free_bytes > best_free_bytes)) {
      best_device = device;
      best_free_bytes = free_bytes;
    }
  }
  if (best_device < 0) {
    if (reason != nullptr) {
      *reason = "所有可见 GPU 均无法查询显存";
    }
    return -1;
  }
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(path compatibility)");
  return best_device;
}

__global__ void LookupCompatibilityKernel(const std::uint64_t* const coverage,
                                          const std::uint32_t words_per_inside,
                                          const PathCompatibilityQuery* const queries,
                                          const std::size_t query_count,
                                          std::uint8_t* const result) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= query_count) {
    return;
  }
  const PathCompatibilityQuery query = queries[index];
  const std::size_t word_index =
      static_cast<std::size_t>(query.inside_index) * words_per_inside + query.outside_index / 64U;
  result[index] = static_cast<std::uint8_t>(
      (coverage[word_index] & (std::uint64_t{1} << (query.outside_index % 64U))) != 0);
}

} // namespace

bool PathCompatibilityCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::uint8_t>
LookupPathCompatibilityCuda(const PathCompatibilityTable& table,
                            const std::vector<PathCompatibilityQuery>& queries,
                            int* const selected_device) {
  if (table.path_count == 0 || table.path_count > kMaxGpuPathCount ||
      table.coverage.size() !=
          static_cast<std::size_t>(table.inside_count) * table.words_per_inside) {
    throw std::invalid_argument("CUDA 路径兼容表元数据非法");
  }
  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA 路径兼容后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  if (queries.empty()) {
    return {};
  }

  DeviceBuffer<std::uint64_t> device_coverage(table.coverage.size());
  DeviceBuffer<PathCompatibilityQuery> device_queries(queries.size());
  DeviceBuffer<std::uint8_t> device_result(queries.size());
  device_coverage.CopyFromHost(table.coverage.data());
  device_queries.CopyFromHost(queries.data());

  constexpr std::size_t kThreads = 256;
  const std::size_t blocks = (queries.size() + kThreads - 1) / kThreads;
  LookupCompatibilityKernel<<<static_cast<unsigned int>(blocks),
                              static_cast<unsigned int>(kThreads)>>>(
      device_coverage.get(), table.words_per_inside, device_queries.get(), queries.size(),
      device_result.get());
  CheckCuda(cudaGetLastError(), "LookupCompatibilityKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "LookupCompatibilityKernel synchronize");

  std::vector<std::uint8_t> result(queries.size());
  device_result.CopyToHost(result.data());
  return result;
}

} // namespace cudaee::detail
