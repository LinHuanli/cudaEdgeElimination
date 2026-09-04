#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define CUDAEE_SIGNED128_HD __host__ __device__
#define CUDAEE_SIGNED128_INLINE __forceinline__
#else
#define CUDAEE_SIGNED128_HD
#define CUDAEE_SIGNED128_INLINE inline
#endif

namespace cudaee::detail {

// CUDA device 没有可移植的 __int128。这个双 limb 二补数只用于
// 精确授权 kernel；guidance kernel 仍可使用 FP32/FP64。
struct Signed128 {
  std::uint64_t low{};
  std::int64_t high{};
};

CUDAEE_SIGNED128_HD CUDAEE_SIGNED128_INLINE Signed128 Signed128FromInt64(
    const std::int64_t value) {
  return {static_cast<std::uint64_t>(value), value < 0 ? -1 : 0};
}

CUDAEE_SIGNED128_HD CUDAEE_SIGNED128_INLINE Signed128 Signed128Add(
    const Signed128 lhs, const Signed128 rhs) {
  Signed128 result;
  result.low = lhs.low + rhs.low;
  const std::uint64_t carry = static_cast<std::uint64_t>(result.low < lhs.low);
  const std::uint64_t high_bits = static_cast<std::uint64_t>(lhs.high) +
                                  static_cast<std::uint64_t>(rhs.high) + carry;
  result.high = static_cast<std::int64_t>(high_bits);
  return result;
}

CUDAEE_SIGNED128_HD CUDAEE_SIGNED128_INLINE Signed128 Signed128AddInt64(
    const Signed128 lhs, const std::int64_t rhs) {
  return Signed128Add(lhs, Signed128FromInt64(rhs));
}

CUDAEE_SIGNED128_HD CUDAEE_SIGNED128_INLINE bool Signed128Greater(
    const Signed128 lhs, const Signed128 rhs) {
  return lhs.high > rhs.high || (lhs.high == rhs.high && lhs.low > rhs.low);
}

CUDAEE_SIGNED128_HD CUDAEE_SIGNED128_INLINE bool Signed128GreaterThanInt64(
    const Signed128 lhs, const std::int64_t rhs) {
  return Signed128Greater(lhs, Signed128FromInt64(rhs));
}

} // namespace cudaee::detail

#undef CUDAEE_SIGNED128_HD
#undef CUDAEE_SIGNED128_INLINE
