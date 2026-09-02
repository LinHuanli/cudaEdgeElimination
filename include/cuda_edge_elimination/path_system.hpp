#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cudaee {

constexpr std::uint32_t kMaxTestablePathCount = 7;
constexpr std::uint32_t kMaxGpuPathCount = 5;
constexpr std::uint32_t kMaxPathEndpoints = 2 * kMaxTestablePathCount;
constexpr std::uint8_t kUnmatchedEndpoint = 0xffU;

using Path = std::vector<std::int32_t>;

struct NormalizedPathSystem {
  bool valid{false};
  std::string reason;
  std::vector<Path> paths;
  std::size_t edge_count{};
};

// 把输入路径的边并集规范化为确定顺序的节点不交链；任何环或歧义均闭门失败。
[[nodiscard]] NormalizedPathSystem NormalizePathSystem(const std::vector<Path>& paths,
                                                       std::int32_t node_count);

struct EndpointMatching {
  std::uint8_t endpoint_count{};
  std::array<std::uint8_t, kMaxPathEndpoints> mate{};

  bool operator==(const EndpointMatching&) const = default;
};

[[nodiscard]] std::size_t ExpectedOutsideMatchingCount(std::uint32_t path_count);
[[nodiscard]] std::size_t ExpectedInsideMatchingCount(std::uint32_t path_count);
[[nodiscard]] std::vector<EndpointMatching> EnumerateOutsideMatchings(std::uint32_t path_count);
[[nodiscard]] std::vector<EndpointMatching> EnumerateInsideMatchings(std::uint32_t path_count);
[[nodiscard]] bool IsPerfectEndpointMatching(const EndpointMatching& matching,
                                             std::uint32_t path_count);
[[nodiscard]] bool IsAlternatingHamiltonianCycle(const EndpointMatching& outside,
                                                 const EndpointMatching& inside,
                                                 std::uint32_t path_count);

// coverage 按 [inside][outside_word] 排列；每一位表示 inside 是否覆盖对应 outside。
struct PathCompatibilityTable {
  std::uint32_t path_count{};
  std::uint32_t outside_count{};
  std::uint32_t inside_count{};
  std::uint32_t words_per_inside{};
  std::vector<std::uint64_t> coverage;
  std::uint64_t generator_hash{};

  [[nodiscard]] bool Covers(std::uint32_t outside_index, std::uint32_t inside_index) const;
};

[[nodiscard]] PathCompatibilityTable BuildPathCompatibilityTable(std::uint32_t path_count);

enum class PathCompatibilityBackend : std::uint8_t {
  kAuto,
  kCpu,
  kCuda,
};

struct PathCompatibilityQuery {
  std::uint32_t outside_index{};
  std::uint32_t inside_index{};
};

struct PathCompatibilityBatchResult {
  std::vector<std::uint8_t> compatible;
  std::string backend;
  int selected_device{-1};
  std::uint64_t generator_hash{};
  bool cpu_verified{false};
};

// m<=5 可查 CUDA bitset 表；m=6,7 无条件转为 CPU 直接判定，绝不截断搜索空间。
[[nodiscard]] PathCompatibilityBatchResult
EvaluatePathCompatibility(std::uint32_t path_count,
                          const std::vector<PathCompatibilityQuery>& queries,
                          PathCompatibilityBackend backend);

namespace detail {

// 生成器内部的稀疏规范化快路径；公开 verifier 继续使用上面的 dense 实现独立重放。
[[nodiscard]] NormalizedPathSystem NormalizeSparsePathSystem(const std::vector<Path>& paths,
                                                             std::int32_t node_count);

[[nodiscard]] bool PathCompatibilityCudaAvailable(std::string* reason);
[[nodiscard]] std::vector<std::uint8_t>
LookupPathCompatibilityCuda(const PathCompatibilityTable& table,
                            const std::vector<PathCompatibilityQuery>& queries,
                            int* selected_device);

} // namespace detail
} // namespace cudaee
