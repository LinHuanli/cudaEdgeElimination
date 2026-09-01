#pragma once

#include "cuda_edge_elimination/types.hpp"

#include <cstdint>

namespace cudaee {

// 对整数坐标使用纯整数运算，保证 CPU 与 GPU 的 TSPLIB 舍入完全一致。
[[nodiscard]] std::uint64_t IntegerSqrtFloor(std::uint64_t value);

[[nodiscard]] bool ExactIntegerDistance(const Point& a, const Point& b, DistanceType type,
                                        std::int64_t* distance, std::string* error);

[[nodiscard]] std::int64_t FloatingDistance(const Point& a, const Point& b, DistanceType type);

} // namespace cudaee
