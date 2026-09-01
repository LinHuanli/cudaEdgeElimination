#pragma once

#include <cstdint>
#include <string>

namespace cudaee {

enum class DistanceType : std::uint8_t {
  kEuc2D = 0,
  kCeil2D = 1,
};

struct Point {
  double x{};
  double y{};
  std::int64_t integer_x{};
  std::int64_t integer_y{};
};

struct Edge {
  std::int32_t u{};
  std::int32_t v{};
  std::int64_t weight{};
  bool active{true};
};

enum class EliminationMethod : std::uint8_t {
  kJv = 1,
};

struct Candidate {
  std::int32_t edge_id{-1};
  std::int32_t witness{-1};
  EliminationMethod method{EliminationMethod::kJv};
};

struct ProofRecord {
  std::uint32_t epoch{};
  std::uint64_t snapshot_hash{};
  std::int32_t edge_id{-1};
  std::int32_t u{-1};
  std::int32_t v{-1};
  std::int32_t witness{-1};
  EliminationMethod method{EliminationMethod::kJv};
};

[[nodiscard]] std::string ToString(DistanceType type);
[[nodiscard]] std::string ToString(EliminationMethod method);

} // namespace cudaee
