#pragma once

#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/types.hpp"

#include <cstdint>
#include <vector>

namespace cudaee::detail {

// 调用方必须先完成方法专属的 CPU 证明复核；本函数只负责快照绑定和确定性度数门禁。
[[nodiscard]] std::vector<Candidate> CommitVerifiedCandidates(GraphSnapshot* graph,
                                                              std::vector<Candidate> candidates,
                                                              std::uint64_t expected_snapshot_hash);

} // namespace cudaee::detail
