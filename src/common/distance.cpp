#include "cuda_edge_elimination/distance.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cudaee {

std::uint64_t IntegerSqrtFloor(const std::uint64_t value) {
  // restoring square-root：全程整数运算，不依赖平台浮点舍入模式。
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

bool ExactIntegerDistance(const Point& a, const Point& b, const DistanceType type,
                          std::int64_t* const distance, std::string* const error) {
  const __int128 dx = static_cast<__int128>(a.integer_x) - b.integer_x;
  const __int128 dy = static_cast<__int128>(a.integer_y) - b.integer_y;
  const __int128 squared = dx * dx + dy * dy;
  if (squared < 0 || squared > std::numeric_limits<std::uint64_t>::max()) {
    if (error != nullptr) {
      *error = "坐标平方距离超过 uint64 范围";
    }
    return false;
  }

  const auto squared64 = static_cast<std::uint64_t>(squared);
  const std::uint64_t root = IntegerSqrtFloor(squared64);
  std::uint64_t rounded = root;
  if (type == DistanceType::kEuc2D) {
    // S 是整数，因此 S > r^2+r 等价于 sqrt(S) >= r+0.5。
    if (squared64 - root * root > root) {
      ++rounded;
    }
  } else if (type == DistanceType::kCeil2D) {
    if (root * root != squared64) {
      ++rounded;
    }
  } else {
    if (error != nullptr) {
      *error = "整数距离不支持该 EDGE_WEIGHT_TYPE";
    }
    return false;
  }

  if (rounded > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    if (error != nullptr) {
      *error = "距离超过 int64 范围";
    }
    return false;
  }
  *distance = static_cast<std::int64_t>(rounded);
  return true;
}

std::int64_t FloatingDistance(const Point& a, const Point& b, const DistanceType type) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double raw = std::hypot(dx, dy);
  if (!std::isfinite(raw) || raw > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("浮点距离无效或溢出");
  }
  if (type == DistanceType::kEuc2D) {
    return static_cast<std::int64_t>(std::floor(raw + 0.5));
  }
  if (type == DistanceType::kCeil2D) {
    return static_cast<std::int64_t>(std::ceil(raw));
  }
  throw std::invalid_argument("不支持的距离类型");
}

} // namespace cudaee
