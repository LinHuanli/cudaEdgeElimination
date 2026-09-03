#include "cuda_edge_elimination/path_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("test failure: " + message);
  }
}

class DisjointSet {
public:
  explicit DisjointSet(const std::uint32_t count) {
    for (std::uint32_t value = 0; value < count; ++value) {
      parent_[value] = static_cast<std::uint8_t>(value);
    }
  }

  std::uint8_t Find(const std::uint8_t value) {
    if (parent_[value] != value) {
      parent_[value] = Find(parent_[value]);
    }
    return parent_[value];
  }

  void Unite(const std::uint8_t first, const std::uint8_t second) {
    const std::uint8_t first_root = Find(first);
    const std::uint8_t second_root = Find(second);
    if (first_root != second_root) {
      parent_[second_root] = first_root;
    }
  }

private:
  std::array<std::uint8_t, cudaee::kMaxPathEndpoints> parent_{};
};

bool IndependentConnectedUnion(const cudaee::EndpointMatching& outside,
                               const cudaee::EndpointMatching& inside,
                               const std::uint32_t path_count) {
  DisjointSet sets(2U * path_count);
  for (std::uint32_t endpoint = 0; endpoint < 2U * path_count; ++endpoint) {
    sets.Unite(static_cast<std::uint8_t>(endpoint), outside.mate[endpoint]);
    sets.Unite(static_cast<std::uint8_t>(endpoint), inside.mate[endpoint]);
  }
  const std::uint8_t root = sets.Find(0);
  for (std::uint32_t endpoint = 1; endpoint < 2U * path_count; ++endpoint) {
    if (sets.Find(static_cast<std::uint8_t>(endpoint)) != root) {
      return false;
    }
  }
  return true;
}

void TestPathNormalization() {
  const cudaee::NormalizedPathSystem merged =
      cudaee::NormalizePathSystem({{2, 1, 0}, {2, 3}, {10, 9}}, 11);
  Check(merged.valid, merged.reason);
  Check(merged.edge_count == 4, "merged edge count");
  Check(merged.paths == std::vector<cudaee::Path>({{0, 1, 2, 3}, {9, 10}}),
        "deterministic merged paths");

  Check(!cudaee::NormalizePathSystem({}, 4).valid, "empty system rejected");
  Check(!cudaee::NormalizePathSystem({{0}}, 4).valid, "singleton rejected");
  Check(!cudaee::NormalizePathSystem({{0, 1, 0}}, 4).valid, "repeated path node rejected");
  Check(!cudaee::NormalizePathSystem({{0, 1}, {1, 0}}, 4).valid, "duplicate edge rejected");
  Check(!cudaee::NormalizePathSystem({{0, 1}, {1, 2}, {2, 0}}, 4).valid, "circuit rejected");
  Check(!cudaee::NormalizePathSystem({{0, 1}, {1, 2}, {1, 3}}, 4).valid, "degree three rejected");
  Check(!cudaee::NormalizePathSystem({{0, 4}}, 4).valid, "out-of-range node rejected");

  std::mt19937 generator(0x5a17c0deU);
  for (std::uint32_t trial = 0U; trial < 2000U; ++trial) {
    const std::int32_t node_count = 2 + static_cast<std::int32_t>(generator() % 31U);
    const std::size_t path_count = 1U + generator() % 8U;
    std::vector<cudaee::Path> paths;
    paths.reserve(path_count);
    for (std::size_t path_index = 0U; path_index < path_count; ++path_index) {
      const std::size_t path_size = 1U + generator() % 6U;
      cudaee::Path path;
      path.reserve(path_size);
      for (std::size_t node_index = 0U; node_index < path_size; ++node_index) {
        // 约 1/16 的样例包含边界外节点，用于锁定失败原因顺序。
        const std::uint32_t draw = static_cast<std::uint32_t>(generator());
        const std::int32_t node =
            draw % 16U == 0U ? node_count : static_cast<std::int32_t>(draw % node_count);
        path.push_back(node);
      }
      paths.push_back(std::move(path));
    }
    const cudaee::NormalizedPathSystem dense = cudaee::NormalizePathSystem(paths, node_count);
    const cudaee::NormalizedPathSystem sparse =
        cudaee::detail::NormalizeSparsePathSystem(paths, node_count);
    Check(dense.valid == sparse.valid && dense.reason == sparse.reason &&
              dense.paths == sparse.paths && dense.edge_count == sparse.edge_count,
          "sparse normalization equals dense oracle");
  }
}

void TestMatchingEnumerationAndTables() {
  constexpr std::array<std::size_t, 7> kOutsideCounts = {1, 2, 8, 48, 384, 3840, 46080};
  constexpr std::array<std::size_t, 7> kInsideCounts = {1, 3, 15, 105, 945, 10395, 135135};
  constexpr std::array<std::uint64_t, 6> kExpectedTableHashes = {
      0x4104b5c5658e8f3aULL, 0x5fcd7fdac93b4fe9ULL, 0x9642d8a1cb6bf1eeULL,
      0x1853eb4cc99dd217ULL, 0xf6bccacc5c1fa84fULL, 0x750842211d2a93e7ULL};

  for (std::uint32_t path_count = 1; path_count <= cudaee::kMaxTestablePathCount; ++path_count) {
    const auto outside = cudaee::EnumerateOutsideMatchings(path_count);
    const auto inside = cudaee::EnumerateInsideMatchings(path_count);
    const std::size_t count_index = static_cast<std::size_t>(path_count - 1U);
    Check(outside.size() == kOutsideCounts[count_index], "outside matching count");
    Check(inside.size() == kInsideCounts[count_index], "inside matching count");
    Check(outside.size() == cudaee::ExpectedOutsideMatchingCount(path_count), "outside formula");
    Check(inside.size() == cudaee::ExpectedInsideMatchingCount(path_count), "inside formula");

    std::set<std::array<std::uint8_t, cudaee::kMaxPathEndpoints>> unique_outside;
    for (const cudaee::EndpointMatching& matching : outside) {
      Check(cudaee::IsPerfectEndpointMatching(matching, path_count), "outside matching is perfect");
      Check(unique_outside.insert(matching.mate).second, "outside matching unique");
    }
    std::set<std::array<std::uint8_t, cudaee::kMaxPathEndpoints>> unique_inside;
    for (const cudaee::EndpointMatching& matching : inside) {
      Check(cudaee::IsPerfectEndpointMatching(matching, path_count), "inside matching is perfect");
      Check(unique_inside.insert(matching.mate).second, "inside matching unique");
    }

    if (path_count <= cudaee::kMaxGpuPathCount) {
      const cudaee::PathCompatibilityTable table = cudaee::BuildPathCompatibilityTable(path_count);
      Check(table.coverage.size() ==
                static_cast<std::size_t>(table.inside_count) * table.words_per_inside,
            "packed table size");
      Check(table.generator_hash == kExpectedTableHashes[static_cast<std::size_t>(path_count - 1U)],
            "pinned compatibility-table generator hash");
      const auto check_cell = [&](const std::uint32_t outside_index,
                                  const std::uint32_t inside_index) {
        const bool independent =
            IndependentConnectedUnion(outside[outside_index], inside[inside_index], path_count);
        Check(table.Covers(outside_index, inside_index) == independent,
              "table equals independent connectivity oracle");
      };
      if (path_count == 6U) {
        check_cell(0U, 0U);
        check_cell(table.outside_count - 1U, table.inside_count - 1U);
        for (std::uint32_t index = 1U; index <= 16384U; ++index) {
          check_cell((index * 2654435761U) % table.outside_count,
                     (index * 2246822519U) % table.inside_count);
        }
      } else {
        for (std::uint32_t inside_index = 0; inside_index < table.inside_count; ++inside_index) {
          for (std::uint32_t outside_index = 0; outside_index < table.outside_count;
               ++outside_index) {
            check_cell(outside_index, inside_index);
          }
        }
      }
      std::cout << "path_count=" << path_count << " table_hash=" << table.generator_hash
                << " bytes=" << table.coverage.size() * sizeof(std::uint64_t) << '\n';
    }
  }
}

void TestCpuFallbackAndCudaDifferential() {
  const std::vector<cudaee::PathCompatibilityQuery> fallback_queries_m7 = {{0, 0}, {46079, 135134}};
  const cudaee::PathCompatibilityBatchResult fallback_m7 = cudaee::EvaluatePathCompatibility(
      7, fallback_queries_m7, cudaee::PathCompatibilityBackend::kAuto);
  Check(fallback_m7.backend == "cpu-fallback-m>6", "m=7 uses automatic CPU fallback");
  Check(fallback_m7.cpu_verified, "m=7 fallback verified");

#ifdef CUDAEE_HAS_CUDA
  std::string unavailable_reason;
  if (!cudaee::detail::PathCompatibilityCudaAvailable(&unavailable_reason)) {
    std::cout << "CUDA path compatibility skipped: " << unavailable_reason << '\n';
    return;
  }
  for (std::uint32_t path_count = 1; path_count <= cudaee::kMaxGpuPathCount; ++path_count) {
    const std::uint32_t outside_count =
        static_cast<std::uint32_t>(cudaee::ExpectedOutsideMatchingCount(path_count));
    const std::uint32_t inside_count =
        static_cast<std::uint32_t>(cudaee::ExpectedInsideMatchingCount(path_count));
    std::vector<cudaee::PathCompatibilityQuery> queries;
    if (path_count == 6U) {
      // m=6 全表约 5 MiB；差分取边界和确定性步长样本，表本身已在上方逐位检查。
      queries = {{0U, 0U}, {outside_count - 1U, inside_count - 1U}};
      for (std::uint32_t index = 1U; index <= 4096U; ++index) {
        queries.push_back(
            {(index * 2654435761U) % outside_count, (index * 2246822519U) % inside_count});
      }
    } else {
      queries.reserve(static_cast<std::size_t>(outside_count) * inside_count);
      for (std::uint32_t inside_index = 0; inside_index < inside_count; ++inside_index) {
        for (std::uint32_t outside_index = 0; outside_index < outside_count; ++outside_index) {
          queries.push_back({outside_index, inside_index});
        }
      }
    }
    const cudaee::PathCompatibilityBatchResult result = cudaee::EvaluatePathCompatibility(
        path_count, queries, cudaee::PathCompatibilityBackend::kCuda);
    Check(result.backend == "cuda", "m<=6 selected CUDA");
    Check(result.cpu_verified, "CUDA results independently verified");
    Check(result.compatible.size() == queries.size(), "CUDA query count");
  }
#endif
}

} // namespace

int main() {
  try {
    TestPathNormalization();
    TestMatchingEnumerationAndTables();
    TestCpuFallbackAndCudaDifferential();
    std::cout << "path-system tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
