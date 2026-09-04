#pragma once

#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cudaee::detail {

// GPU Quick-HS 紧凑证书的不可变快照视图。距离矩阵和定长邻接行只在
// 每个 replay epoch 构造一次，逐 record 只执行整数谓词。
struct QuickHsVerificationData {
  std::int32_t dimension{};
  std::vector<std::int32_t> degree;
  std::vector<std::int32_t> neighbors;
  std::vector<std::int64_t> distance;
  std::vector<std::uint8_t> active;
};

[[nodiscard]] QuickHsVerificationData BuildQuickHsVerificationData(const GraphSnapshot& graph);

[[nodiscard]] bool VerifyQuickHsCandidate(const GraphSnapshot& graph,
                                          const QuickHsVerificationData& verification_data,
                                          const Candidate& candidate, std::string* reason);

} // namespace cudaee::detail
