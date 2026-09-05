#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cudaee::detail {

// 新 PDHG 与旧 SEC ensemble 共用整数授权格式：这里只转换 multiplier，
// 必须继续重算 reduced cost、Signed128 bound，并比较更好的完整快照。
__global__ void QuantizeSparsePdhgDualKernel(const std::int32_t dimension, const std::int32_t cuts,
                                             const double* dual, const double scale,
                                             const std::int64_t denominator,
                                             const std::uint8_t* valid_cuts,
                                             std::int64_t* vertex_numerator,
                                             std::int64_t* cut_numerator, std::int32_t* invalid) {
  const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= static_cast<std::int64_t>(dimension) + cuts) {
    return;
  }
  const bool vertex = row < dimension;
  const std::int64_t cut = row - dimension;
  const double value = !vertex && valid_cuts[cut] == 0U ? 0.0 : dual[row] * scale;
  const double scaled = value * static_cast<double>(denominator);
  std::int64_t quantized = 0;
  if (!isfinite(scaled) || fabs(scaled) > 1.0e15 || (!vertex && scaled < 0.0)) {
    atomicExch(invalid, 1);
  } else {
    quantized = static_cast<std::int64_t>(scaled >= 0.0 ? floor(scaled + 0.5) : ceil(scaled - 0.5));
  }
  if (vertex) {
    vertex_numerator[row] = quantized;
  } else {
    cut_numerator[cut] = quantized;
  }
}

} // namespace cudaee::detail
