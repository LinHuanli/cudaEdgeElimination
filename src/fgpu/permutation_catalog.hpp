#pragma once

#include <cstddef>

#if defined(__CUDACC__)
#define CUDAEE_CATALOG_HD __host__ __device__
#else
#define CUDAEE_CATALOG_HD
#endif

namespace cudaee::detail::permutation_catalog {
constexpr int kMaximumNodes = 8;
CUDAEE_CATALOG_HD constexpr int Factorial(int count) {
  int result = 1;
  for (int i = 2; i <= count; ++i)
    result *= i;
  return result;
}
CUDAEE_CATALOG_HD constexpr std::size_t Offset(int count) {
  std::size_t result = 0;
  for (int i = 1; i < count; ++i)
    result += static_cast<std::size_t>(i) * Factorial(i);
  return result;
}
constexpr std::size_t kBytes = Offset(kMaximumNodes + 1);
} // namespace cudaee::detail::permutation_catalog

#undef CUDAEE_CATALOG_HD
