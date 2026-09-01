#include "cuda_edge_elimination/hamilton_tutte.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
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
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(HT c,d)");
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
                "cudaMemcpy H2D(HT c,d)");
    }
  }

  void CopyToHost(T* const destination) const {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(HT c,d)");
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
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(HT c,d)");
  return best_device;
}

__device__ std::uint64_t IntegerSqrtFloorDevice(const std::uint64_t value) {
  std::uint64_t remainder = value;
  std::uint64_t root = 0;
  std::uint64_t bit = std::uint64_t{1} << 62;
  while (bit > remainder) {
    bit >>= 2;
  }
  while (bit != 0) {
    if (remainder >= root + bit) {
      remainder -= root + bit;
      root = (root >> 1) + bit;
    } else {
      root >>= 1;
    }
    bit >>= 2;
  }
  return root;
}

__device__ std::uint64_t AbsoluteDifference(const std::int64_t first, const std::int64_t second) {
  return first >= second ? static_cast<std::uint64_t>(first - second)
                         : static_cast<std::uint64_t>(second - first);
}

__device__ std::uint64_t ExactDistanceDevice(const std::int32_t first, const std::int32_t second,
                                             const std::int64_t* const x,
                                             const std::int64_t* const y,
                                             const std::uint8_t distance_type) {
  const std::uint64_t dx = AbsoluteDifference(x[first], x[second]);
  const std::uint64_t dy = AbsoluteDifference(y[first], y[second]);
  const std::uint64_t squared = dx * dx + dy * dy;
  const std::uint64_t root = IntegerSqrtFloorDevice(squared);
  std::uint64_t rounded = root;
  if (distance_type == static_cast<std::uint8_t>(DistanceType::kEuc2D)) {
    rounded += squared - root * root > root ? 1U : 0U;
  } else {
    rounded += root * root != squared ? 1U : 0U;
  }
  return rounded;
}

__global__ void ScreenHtCdKernel(const NodeEdge target, const HtCdScreenTask* const tasks,
                                 const std::size_t task_count, const std::int64_t* const x,
                                 const std::int64_t* const y, const std::uint8_t distance_type,
                                 const std::uint8_t mode, std::uint8_t* const flags) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= task_count) {
    return;
  }
  const HtCdScreenTask task = tasks[index];
  const std::uint64_t original = ExactDistanceDevice(target.u, target.v, x, y, distance_type) +
                                 ExactDistanceDevice(task.c, task.d, x, y, distance_type);
  const std::uint64_t orientation0 = ExactDistanceDevice(target.u, task.d, x, y, distance_type) +
                                     ExactDistanceDevice(target.v, task.c, x, y, distance_type);
  const std::uint64_t orientation1 = ExactDistanceDevice(target.u, task.c, x, y, distance_type) +
                                     ExactDistanceDevice(target.v, task.d, x, y, distance_type);
  const bool incompatible = orientation0 < original && orientation1 < original;
  if (mode == static_cast<std::uint8_t>(HtCdMode::kActiveIncompatible)) {
    flags[index] = static_cast<std::uint8_t>(task.active != 0 && incompatible);
  } else {
    flags[index] = static_cast<std::uint8_t>(task.active == 0 || incompatible);
  }
}

} // namespace

bool HtCdCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::uint8_t> ScreenHtCdCandidatesCuda(const GraphSnapshot& graph,
                                                   const NodeEdge target_edge,
                                                   const std::vector<HtCdScreenTask>& tasks,
                                                   const HtCdMode mode,
                                                   int* const selected_device) {
  if (target_edge.u < 0 || target_edge.v >= graph.dimension || target_edge.u >= target_edge.v ||
      (mode != HtCdMode::kActiveIncompatible && mode != HtCdMode::kMissingOrIncompatible)) {
    throw std::invalid_argument("CUDA HT c,d 输入非法");
  }
  for (const HtCdScreenTask& task : tasks) {
    if (task.c < 0 || task.d >= graph.dimension || task.c >= task.d || task.c == target_edge.u ||
        task.c == target_edge.v || task.d == target_edge.u || task.d == target_edge.v ||
        task.active > 1 || (task.active != 0) != graph.HasActiveEdge(task.c, task.d)) {
      throw std::invalid_argument("CUDA HT c,d task 非法或活动位不一致");
    }
  }

  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA HT c,d 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  if (tasks.empty()) {
    return {};
  }
  constexpr std::size_t kThreads = 256;
  const std::size_t blocks = (tasks.size() + kThreads - 1U) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA HT c,d 网格过大");
  }

  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  for (std::size_t index = 0; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }
  DeviceBuffer<HtCdScreenTask> device_tasks(tasks.size());
  DeviceBuffer<std::int64_t> device_x(host_x.size());
  DeviceBuffer<std::int64_t> device_y(host_y.size());
  DeviceBuffer<std::uint8_t> device_flags(tasks.size());
  device_tasks.CopyFromHost(tasks.data());
  device_x.CopyFromHost(host_x.data());
  device_y.CopyFromHost(host_y.data());

  ScreenHtCdKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(kThreads)>>>(
      target_edge, device_tasks.get(), tasks.size(), device_x.get(), device_y.get(),
      static_cast<std::uint8_t>(graph.distance_type), static_cast<std::uint8_t>(mode),
      device_flags.get());
  CheckCuda(cudaGetLastError(), "ScreenHtCdKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "ScreenHtCdKernel synchronize");

  std::vector<std::uint8_t> flags(tasks.size());
  device_flags.CopyToHost(flags.data());
  return flags;
}

} // namespace cudaee::detail
