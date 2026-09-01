#pragma once

#include "cuda_edge_elimination/graph.hpp"
#include "cuda_edge_elimination/hamilton_tutte.hpp"
#include "cuda_edge_elimination/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cudaee {

enum class Backend {
  kAuto,
  kCpu,
  kCuda,
};

struct EpochMetrics {
  std::uint32_t epoch{};
  std::size_t edges_before{};
  std::size_t proposed{};
  std::size_t verified{};
  std::size_t rejected{};
  std::size_t committed{};
  // 细分计时只用于当前运行观测；V1/V2 proof 继续只序列化 propose/verify。
  double snapshot_ms{};
  double propose_ms{};
  double verify_ms{};
  double commit_ms{};
  bool jv_static_cache_hit{false};
  bool jv_workspace_cache_hit{false};
  std::uint64_t jv_resident_bytes{};
  double jv_h2d_ms{};
  double jv_kernel_ms{};
  double jv_d2h_ms{};
};

struct JvCudaCacheUsage {
  bool static_hit{false};
  bool workspace_hit{false};
  std::uint64_t resident_bytes{};
  // 三段同步 wall time 仅用于性能诊断，不进入 proof。
  double h2d_ms{};
  double kernel_ms{};
  double d2h_ms{};
};

struct EliminationResult {
  std::string backend;
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  std::vector<ProofRecord> proof;
  // 只有 HT record 可以引用这里的 V1 continuation arena；JV V1 输出保持不变。
  std::vector<HtRecursiveProof> ht_proofs;
  std::vector<EpochMetrics> epochs;
};

[[nodiscard]] std::vector<Candidate> FindJvCandidatesCpu(const GraphSnapshot& graph);
[[nodiscard]] bool VerifyJvCandidate(const GraphSnapshot& graph, const Candidate& candidate,
                                     std::string* reason);

[[nodiscard]] bool CudaBackendAvailable(std::string* reason);
[[nodiscard]] std::vector<Candidate> FindJvCandidatesCuda(const GraphSnapshot& graph,
                                                          int* selected_device,
                                                          JvCudaCacheUsage* cache_usage = nullptr);
// 释放当前主机线程在所有设备上的 JV 驻留缓存；测试隔离和显式 teardown 使用。
void ClearJvCudaCache();

EliminationResult RunJvElimination(GraphSnapshot* graph, Backend backend, std::uint32_t max_rounds);

// 在同一不可变快照上整批复核 HT sidecars，再按规范边序执行一次原子 epoch 提交。
EliminationResult CommitHtProofEpoch(GraphSnapshot* graph,
                                     const std::vector<HtRecursiveProof>& proofs);

void WriteProof(const std::filesystem::path& path, const EliminationResult& result);
[[nodiscard]] EliminationResult ReadProof(const std::filesystem::path& path);
[[nodiscard]] EliminationResult ReplayProof(GraphSnapshot* graph,
                                            const EliminationResult& expected);

} // namespace cudaee
