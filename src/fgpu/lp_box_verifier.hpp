#pragma once

#include "cuda_edge_elimination/elimination.hpp"

#include <cstdint>
#include <string>

namespace cudaee::detail {

// 调用方已在不可变 epoch 边界计算 actual_snapshot_hash 时使用，避免对同一
// sidecar 的每条候选边反复线性扫描整张 stable-edge 数组。
[[nodiscard]] bool VerifyLpBoxCandidateForSnapshot(const GraphSnapshot& graph,
                                                   const LpBoxProof& proof,
                                                   const LpBoxVerificationData& verification_data,
                                                   const Candidate& candidate,
                                                   std::uint64_t actual_snapshot_hash,
                                                   std::string* reason);

} // namespace cudaee::detail
