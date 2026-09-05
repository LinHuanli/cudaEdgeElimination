#include "cuda_edge_elimination/build_identity.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>

namespace cudaee {

void WriteGpuIdentityJson(std::ostream& output, const int device) {
  cudaDeviceProp properties{};
  char bus_id[64]{};
  int driver = 0;
  int runtime = 0;
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device) != cudaSuccess ||
      cudaDriverGetVersion(&driver) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime) != cudaSuccess) {
    throw std::runtime_error("无法记录实际使用 GPU 的身份");
  }
  constexpr char hex[] = "0123456789abcdef";
  std::string uuid{"GPU-"};
  for (std::size_t index = 0; index < sizeof(properties.uuid.bytes); ++index) {
    if (index == 4U || index == 6U || index == 8U || index == 10U) {
      uuid.push_back('-');
    }
    const unsigned char byte = static_cast<unsigned char>(properties.uuid.bytes[index]);
    uuid.push_back(hex[byte >> 4U]);
    uuid.push_back(hex[byte & 15U]);
  }
  output << "{\"uuid\":" << JsonString(uuid) << ",\"name\":" << JsonString(properties.name)
         << ",\"pci_bus_id\":" << JsonString(bus_id) << ",\"visible_ordinal\":" << device
         << ",\"compute_major\":" << properties.major << ",\"compute_minor\":" << properties.minor
         << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime
         << ",\"multiprocessors\":" << properties.multiProcessorCount
         << ",\"max_threads_per_sm\":" << properties.maxThreadsPerMultiProcessor
         << ",\"registers_per_sm\":" << properties.regsPerMultiprocessor
         << ",\"global_memory_bytes\":" << properties.totalGlobalMem << '}';
}

} // namespace cudaee
