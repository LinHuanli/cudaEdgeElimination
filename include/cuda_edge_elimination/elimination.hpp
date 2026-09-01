#pragma once

#include "cuda_edge_elimination/graph.hpp"
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
  double propose_ms{};
  double verify_ms{};
};

struct EliminationResult {
  std::string backend;
  std::uint64_t initial_hash{};
  std::uint64_t final_hash{};
  std::vector<ProofRecord> proof;
  std::vector<EpochMetrics> epochs;
};

[[nodiscard]] std::vector<Candidate> FindJvCandidatesCpu(const GraphSnapshot& graph);
[[nodiscard]] bool VerifyJvCandidate(const GraphSnapshot& graph, const Candidate& candidate,
                                     std::string* reason);

[[nodiscard]] bool CudaBackendAvailable(std::string* reason);
[[nodiscard]] std::vector<Candidate> FindJvCandidatesCuda(const GraphSnapshot& graph,
                                                          int* selected_device);

EliminationResult RunJvElimination(GraphSnapshot* graph, Backend backend, std::uint32_t max_rounds);

void WriteProof(const std::filesystem::path& path, const EliminationResult& result);
[[nodiscard]] EliminationResult ReadProof(const std::filesystem::path& path);
[[nodiscard]] EliminationResult ReplayProof(GraphSnapshot* graph,
                                            const EliminationResult& expected);

} // namespace cudaee
