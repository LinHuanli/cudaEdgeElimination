#include "../fgpu/sparse_pdhg.hpp"

#include <stdexcept>

namespace cudaee::detail {
class SparsePdhgCuda::Impl {};
SparsePdhgCuda::SparsePdhgCuda(const int) {
  throw std::runtime_error("当前构建未启用 sparse PDHG CUDA 后端");
}
SparsePdhgCuda::~SparsePdhgCuda() = default;
SparsePdhgDiagnostics SparsePdhgCuda::Iterate(const SparsePdhgDeviceModel&, const double,
                                              const std::uint32_t) {
  throw std::runtime_error("当前构建未启用 sparse PDHG CUDA 后端");
}
const double* SparsePdhgCuda::primal() const { return nullptr; }
const double* SparsePdhgCuda::dual() const { return nullptr; }
std::uint64_t SparsePdhgCuda::workspace_bytes() const { return 0U; }
} // namespace cudaee::detail
