#pragma once

#include <cstdint>
#include <string_view>

namespace cudaee {

// 三种后端仅改变 Opt34 的执行方式，不改变枚举域或 GPU replay。
enum class PointLeafKernel : std::uint8_t {
  kPermutation,
  kPrescreenPermutation,
  kPrescreenSubsetDp,
};

constexpr std::string_view ToString(const PointLeafKernel kernel) {
  switch (kernel) {
  case PointLeafKernel::kPermutation:
    return "permutation";
  case PointLeafKernel::kPrescreenPermutation:
    return "prescreen-permutation";
  case PointLeafKernel::kPrescreenSubsetDp:
    return "prescreen-subset-dp";
  }
  return "invalid";
}

} // namespace cudaee
