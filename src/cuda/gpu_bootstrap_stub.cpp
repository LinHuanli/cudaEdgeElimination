#include "../fgpu/gpu_bootstrap.hpp"
#include <stdexcept>

namespace cudaee::detail {
struct GpuBootstrap::Impl {};
GpuBootstrap::GpuBootstrap(const GraphSnapshot&, int) {
  throw std::runtime_error("GPU bootstrap 需要 CUDA 构建");
}
GpuBootstrap::~GpuBootstrap() = default;
void GpuBootstrap::BuildCompleteGraph(GraphSnapshot*) {
  throw std::runtime_error("GPU bootstrap 需要 CUDA 构建");
}
void GpuBootstrap::GenerateIncumbent() { throw std::runtime_error("GPU bootstrap 需要 CUDA 构建"); }
void GpuBootstrap::BuildPermutationCatalog() {
  throw std::runtime_error("GPU 组合缓存需要 CUDA 构建");
}
const std::int64_t* GpuBootstrap::distances() const { return nullptr; }
const std::uint8_t* GpuBootstrap::permutations() const { return nullptr; }
int GpuBootstrap::device() const { return -1; }
const GpuBootstrapMetrics& GpuBootstrap::metrics() const {
  throw std::runtime_error("GPU bootstrap 需要 CUDA 构建");
}
const std::vector<std::int32_t>& GpuBootstrap::tour() const {
  throw std::runtime_error("GPU bootstrap 需要 CUDA 构建");
}
} // namespace cudaee::detail
