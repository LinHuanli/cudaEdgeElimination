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
  kHamiltonTutte = 2,
  // Hougardy--Schroeder Main Edge Elimination 的几何见证。
  kGeometryMain = 3,
  // 量化 Lagrangian box bound 证明强制目标边后下界超过 incumbent。
  kLpBox = 4,
  // GPU-resident KH -q 浅层 Hamilton--Tutte；record 直接保存 c,d 紧凑见证。
  kGpuQuickHs = 5,
};

constexpr std::uint32_t kNoEliminationCertificate = UINT32_MAX;

struct Candidate {
  std::int32_t edge_id{-1};
  std::int32_t witness{-1};
  EliminationMethod method{EliminationMethod::kJv};
  // 几何 Main Edge 与 GPU Quick-HS 分别需要第二个 potential/d 点。
  std::int32_t second_witness{-1};
};

struct ProofRecord {
  std::uint32_t epoch{};
  std::uint64_t snapshot_hash{};
  std::int32_t edge_id{-1};
  std::int32_t u{-1};
  std::int32_t v{-1};
  std::int32_t witness{-1};
  EliminationMethod method{EliminationMethod::kJv};
  // HT 记录指向 EliminationResult::ht_proofs；JV 不携带嵌套证书。
  std::uint32_t certificate_index{kNoEliminationCertificate};
  // V3+ 几何记录的第二个 potential 点，或 Quick-HS 的 d 点。
  std::int32_t second_witness{-1};
};

[[nodiscard]] std::string ToString(DistanceType type);
[[nodiscard]] std::string ToString(EliminationMethod method);

} // namespace cudaee
