#include "cuda_edge_elimination/local_search.hpp"

#include <cuda_runtime.h>

#include <climits>
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
      CheckCuda(cudaMalloc(&data_, sizeof(T) * count_), "cudaMalloc(k-opt cost)");
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
                "cudaMemcpy H2D(k-opt cost)");
    }
  }

  void CopyToHost(T* const destination) const {
    if (count_ != 0) {
      CheckCuda(cudaMemcpy(destination, data_, sizeof(T) * count_, cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H(k-opt cost)");
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
  CheckCuda(cudaSetDevice(best_device), "cudaSetDevice(k-opt cost)");
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

__device__ std::int64_t ExactDistanceDevice(const std::int32_t first, const std::int32_t second,
                                            const std::int64_t* const x,
                                            const std::int64_t* const y,
                                            const std::uint8_t distance_type) {
  const std::uint64_t dx = AbsoluteDifference(x[first], x[second]);
  const std::uint64_t dy = AbsoluteDifference(y[first], y[second]);
  const std::uint64_t squared = dx * dx + dy * dy;
  const std::uint64_t root = IntegerSqrtFloorDevice(squared);
  std::uint64_t rounded = root;
  if (distance_type == static_cast<std::uint8_t>(DistanceType::kEuc2D)) {
    rounded += squared - root * root > root ? 1 : 0;
  } else {
    rounded += root * root != squared ? 1 : 0;
  }
  return static_cast<std::int64_t>(rounded);
}

__device__ void CanonicalEndpoints(const std::int32_t first, const std::int32_t second,
                                   std::int32_t* const u, std::int32_t* const v) {
  if (first < second) {
    *u = first;
    *v = second;
  } else {
    *u = second;
    *v = first;
  }
}

__global__ void KOptTemplateCostsKernel(const std::uint32_t k, const KOptCostTask* const tasks,
                                        const std::size_t task_count,
                                        const EndpointMatching* const templates,
                                        const std::uint32_t template_count,
                                        const std::int64_t* const x, const std::int64_t* const y,
                                        const std::uint8_t distance_type,
                                        std::int64_t* const costs) {
  const std::size_t flat_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t cell_count = task_count * template_count;
  if (flat_index >= cell_count) {
    return;
  }
  const std::size_t task_index = flat_index / template_count;
  const std::uint32_t template_index = static_cast<std::uint32_t>(flat_index % template_count);
  const KOptCostTask task = tasks[task_index];
  const EndpointMatching matching = templates[template_index];

  std::int32_t added_u[5];
  std::int32_t added_v[5];
  std::uint32_t added_count = 0;
  std::int64_t total = 0;
  for (std::uint32_t port = 0; port < 2U * k; ++port) {
    const std::uint32_t partner = matching.mate[port];
    if (port >= partner) {
      continue;
    }
    std::int32_t u = -1;
    std::int32_t v = -1;
    CanonicalEndpoints(task.port_nodes[port], task.port_nodes[partner], &u, &v);
    if (u == v) {
      costs[flat_index] = kInvalidKOptTemplateCost;
      return;
    }
    for (std::uint32_t edge = 0; edge < k; ++edge) {
      std::int32_t deleted_u = -1;
      std::int32_t deleted_v = -1;
      CanonicalEndpoints(task.port_nodes[2U * edge], task.port_nodes[2U * edge + 1U], &deleted_u,
                         &deleted_v);
      if (u == deleted_u && v == deleted_v) {
        costs[flat_index] = kInvalidKOptTemplateCost;
        return;
      }
    }
    for (std::uint32_t edge = 0; edge < added_count; ++edge) {
      if (u == added_u[edge] && v == added_v[edge]) {
        costs[flat_index] = kInvalidKOptTemplateCost;
        return;
      }
    }
    const std::int64_t distance = ExactDistanceDevice(u, v, x, y, distance_type);
    if (distance < 0 || total > LLONG_MAX - distance) {
      costs[flat_index] = kInvalidKOptTemplateCost;
      return;
    }
    added_u[added_count] = u;
    added_v[added_count] = v;
    ++added_count;
    total += distance;
  }
  costs[flat_index] = added_count == k ? total : kInvalidKOptTemplateCost;
}

} // namespace

bool KOptCostCudaAvailable(std::string* const reason) { return SelectDevice(reason) >= 0; }

std::vector<std::int64_t> EvaluateKOptTemplateCostsCuda(const GraphSnapshot& graph,
                                                        const KOptReconnectTable& table,
                                                        const std::vector<KOptCostTask>& tasks,
                                                        int* const selected_device) {
  std::string reason;
  const int device = SelectDevice(&reason);
  if (device < 0) {
    throw std::runtime_error("CUDA k-opt cost 后端不可用: " + reason);
  }
  if (selected_device != nullptr) {
    *selected_device = device;
  }
  if (tasks.empty()) {
    return {};
  }

  const std::size_t cell_count = tasks.size() * table.templates.size();
  constexpr std::size_t kThreads = 256;
  const std::size_t blocks = (cell_count + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    throw std::overflow_error("CUDA k-opt cost 网格过大");
  }
  std::vector<std::int64_t> host_x(graph.points.size());
  std::vector<std::int64_t> host_y(graph.points.size());
  for (std::size_t index = 0; index < graph.points.size(); ++index) {
    host_x[index] = graph.points[index].integer_x;
    host_y[index] = graph.points[index].integer_y;
  }

  DeviceBuffer<KOptCostTask> device_tasks(tasks.size());
  DeviceBuffer<EndpointMatching> device_templates(table.templates.size());
  DeviceBuffer<std::int64_t> device_x(host_x.size());
  DeviceBuffer<std::int64_t> device_y(host_y.size());
  DeviceBuffer<std::int64_t> device_costs(cell_count);
  device_tasks.CopyFromHost(tasks.data());
  device_templates.CopyFromHost(table.templates.data());
  device_x.CopyFromHost(host_x.data());
  device_y.CopyFromHost(host_y.data());

  KOptTemplateCostsKernel<<<static_cast<unsigned int>(blocks),
                            static_cast<unsigned int>(kThreads)>>>(
      table.k, device_tasks.get(), tasks.size(), device_templates.get(),
      static_cast<std::uint32_t>(table.templates.size()), device_x.get(), device_y.get(),
      static_cast<std::uint8_t>(graph.distance_type), device_costs.get());
  CheckCuda(cudaGetLastError(), "KOptTemplateCostsKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "KOptTemplateCostsKernel synchronize");

  std::vector<std::int64_t> costs(cell_count);
  device_costs.CopyToHost(costs.data());
  return costs;
}

} // namespace cudaee::detail
