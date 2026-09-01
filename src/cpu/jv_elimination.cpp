#include "cuda_edge_elimination/elimination.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::size_t kMaxCandidateNodes = 10;

struct ScoredNode {
  std::int64_t score{};
  std::int32_t node{-1};
};

bool LessScoredNode(const ScoredNode& lhs, const ScoredNode& rhs) {
  return std::tie(lhs.score, lhs.node) < std::tie(rhs.score, rhs.node);
}

std::vector<std::int32_t> PotentialCandidateNodes(const GraphSnapshot& graph, const std::int32_t a,
                                                  const std::int32_t b) {
  std::vector<ScoredNode> scored;
  scored.reserve(kMaxCandidateNodes + 1);

  const auto visit_row = [&](const std::int32_t from, const std::int32_t other, const bool second) {
    const auto row_begin = graph.row_offsets[static_cast<std::size_t>(from)];
    const auto row_end = graph.row_offsets[static_cast<std::size_t>(from) + 1];
    for (std::int32_t offset = row_begin; offset < row_end; ++offset) {
      const std::int32_t candidate = graph.neighbors[static_cast<std::size_t>(offset)];
      if (candidate == a || candidate == b) {
        continue;
      }
      // ElimTSP quick search 首先访问 a 的邻点；第二行遇到重复点时不重新计分。
      if (second && graph.HasActiveEdge(a, candidate)) {
        continue;
      }
      const __int128 score =
          static_cast<__int128>(graph.csr_weights[static_cast<std::size_t>(offset)]) +
          graph.Distance(candidate, other);
      if (score > std::numeric_limits<std::int64_t>::max()) {
        continue;
      }
      scored.push_back({static_cast<std::int64_t>(score), candidate});
      std::sort(scored.begin(), scored.end(), LessScoredNode);
      if (scored.size() > kMaxCandidateNodes) {
        scored.pop_back();
      }
    }
  };

  visit_row(a, b, false);
  visit_row(b, a, true);
  std::vector<std::int32_t> result;
  result.reserve(scored.size());
  for (const ScoredNode& item : scored) {
    result.push_back(item.node);
  }
  return result;
}

bool IsJvWitness(const GraphSnapshot& graph, const Edge& edge, const std::int32_t witness,
                 std::string* const reason) {
  if (witness < 0 || witness >= graph.dimension || witness == edge.u || witness == edge.v) {
    if (reason != nullptr) {
      *reason = "见证点越界或与端点相同";
    }
    return false;
  }

  const std::int64_t cab = edge.weight;
  const std::int64_t cac = graph.Distance(edge.u, witness);
  const std::int64_t cbc = graph.Distance(edge.v, witness);
  const auto begin = graph.row_offsets[static_cast<std::size_t>(witness)];
  const auto end = graph.row_offsets[static_cast<std::size_t>(witness) + 1];
  for (std::int32_t offset = begin; offset < end; ++offset) {
    const std::int32_t d = graph.neighbors[static_cast<std::size_t>(offset)];
    if (d == edge.u || d == edge.v) {
      continue;
    }
    const std::int64_t ccd = graph.csr_weights[static_cast<std::size_t>(offset)];
    const __int128 left = static_cast<__int128>(cab) + ccd;
    const __int128 first = static_cast<__int128>(cac) + graph.Distance(d, edge.v);
    const __int128 second = static_cast<__int128>(graph.Distance(edge.u, d)) + cbc;
    if (left <= first || left <= second) {
      if (reason != nullptr) {
        *reason = "存在与端点边兼容的邻点 d=" + std::to_string(d);
      }
      return false;
    }
  }
  return true;
}

std::vector<Candidate> VerifyCandidates(const GraphSnapshot& graph,
                                        const std::vector<Candidate>& candidates,
                                        std::size_t* const rejected) {
  std::vector<Candidate> accepted;
  accepted.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    std::string reason;
    if (VerifyJvCandidate(graph, candidate, &reason)) {
      accepted.push_back(candidate);
    } else {
      ++*rejected;
    }
  }
  return accepted;
}

std::vector<Candidate> CommitCandidates(GraphSnapshot* const graph,
                                        std::vector<Candidate> candidates) {
  std::sort(candidates.begin(), candidates.end(), [&](const Candidate& lhs, const Candidate& rhs) {
    const Edge& lhs_edge = graph->edges[static_cast<std::size_t>(lhs.edge_id)];
    const Edge& rhs_edge = graph->edges[static_cast<std::size_t>(rhs.edge_id)];
    return std::tie(lhs_edge.u, lhs_edge.v, lhs.witness, lhs.edge_id) <
           std::tie(rhs_edge.u, rhs_edge.v, rhs.witness, rhs.edge_id);
  });
  candidates.erase(std::unique(candidates.begin(), candidates.end(),
                               [](const Candidate& lhs, const Candidate& rhs) {
                                 return lhs.edge_id == rhs.edge_id;
                               }),
                   candidates.end());

  std::vector<std::int32_t> degrees(static_cast<std::size_t>(graph->dimension));
  for (std::int32_t vertex = 0; vertex < graph->dimension; ++vertex) {
    degrees[static_cast<std::size_t>(vertex)] = graph->Degree(vertex);
  }

  std::vector<Candidate> committed;
  committed.reserve(candidates.size());
  for (const Candidate& candidate : candidates) {
    Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
    if (!edge.active || degrees[static_cast<std::size_t>(edge.u)] <= 2 ||
        degrees[static_cast<std::size_t>(edge.v)] <= 2) {
      continue;
    }
    edge.active = false;
    --degrees[static_cast<std::size_t>(edge.u)];
    --degrees[static_cast<std::size_t>(edge.v)];
    committed.push_back(candidate);
  }
  if (!committed.empty()) {
    graph->RebuildCsr();
  }
  return committed;
}

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

std::vector<Candidate> FindJvCandidatesCpu(const GraphSnapshot& graph) {
  std::vector<Candidate> candidates;
  for (std::size_t edge_id = 0; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
      continue;
    }
    for (const std::int32_t witness : PotentialCandidateNodes(graph, edge.u, edge.v)) {
      if (IsJvWitness(graph, edge, witness, nullptr)) {
        candidates.push_back({static_cast<std::int32_t>(edge_id), witness, EliminationMethod::kJv});
        break;
      }
    }
  }
  return candidates;
}

bool VerifyJvCandidate(const GraphSnapshot& graph, const Candidate& candidate,
                       std::string* const reason) {
  if (candidate.method != EliminationMethod::kJv) {
    if (reason != nullptr) {
      *reason = "验证器不支持该方法";
    }
    return false;
  }
  if (candidate.edge_id < 0 || static_cast<std::size_t>(candidate.edge_id) >= graph.edges.size()) {
    if (reason != nullptr) {
      *reason = "边编号越界";
    }
    return false;
  }
  const Edge& edge = graph.edges[static_cast<std::size_t>(candidate.edge_id)];
  if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
    if (reason != nullptr) {
      *reason = "边不活动或端点度数不满足前提";
    }
    return false;
  }
  return IsJvWitness(graph, edge, candidate.witness, reason);
}

EliminationResult RunJvElimination(GraphSnapshot* const graph, const Backend backend,
                                   const std::uint32_t max_rounds) {
  if (graph == nullptr || max_rounds == 0) {
    throw std::invalid_argument("图不能为空且 max_rounds 必须大于 0");
  }

  bool use_cuda = false;
  std::string unavailable_reason;
  if (backend != Backend::kCpu) {
    if (!graph->integer_coordinates) {
      unavailable_reason = "CUDA JV 首期仅支持整数坐标";
    } else if (!graph->integer_distance_safe) {
      unavailable_reason = "整数平方距离超过 CUDA 首期安全范围";
    } else {
      use_cuda = CudaBackendAvailable(&unavailable_reason);
    }
    if (backend == Backend::kCuda && !use_cuda) {
      throw std::runtime_error("CUDA 后端不可用: " + unavailable_reason);
    }
  }

  EliminationResult result;
  result.backend = use_cuda ? "cuda" : "cpu";
  result.initial_hash = graph->ContentHash();
  for (std::uint32_t epoch = 0; epoch < max_rounds; ++epoch) {
    EpochMetrics metrics;
    metrics.epoch = epoch;
    metrics.edges_before = graph->ActiveEdgeCount();
    const std::uint64_t snapshot_hash = graph->ContentHash();

    const auto propose_start = std::chrono::steady_clock::now();
    int selected_device = -1;
    std::vector<Candidate> proposed =
        use_cuda ? FindJvCandidatesCuda(*graph, &selected_device) : FindJvCandidatesCpu(*graph);
    metrics.propose_ms = ElapsedMilliseconds(propose_start);
    metrics.proposed = proposed.size();

    const auto verify_start = std::chrono::steady_clock::now();
    std::vector<Candidate> verified = VerifyCandidates(*graph, proposed, &metrics.rejected);
    metrics.verify_ms = ElapsedMilliseconds(verify_start);
    metrics.verified = verified.size();
    if (use_cuda && metrics.rejected != 0) {
      throw std::runtime_error("CUDA 候选未通过 CPU 复核；已停止 epoch，未提交删除");
    }

    std::vector<Candidate> committed = CommitCandidates(graph, std::move(verified));
    metrics.committed = committed.size();
    for (const Candidate& candidate : committed) {
      const Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
      result.proof.push_back({epoch, snapshot_hash, candidate.edge_id, edge.u, edge.v,
                              candidate.witness, candidate.method});
    }
    result.epochs.push_back(metrics);
    if (committed.empty()) {
      break;
    }
  }
  result.final_hash = graph->ContentHash();
  return result;
}

void WriteProof(const std::filesystem::path& path, const EliminationResult& result) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建证明文件: " + path.string());
  }
  output << "CUDAEE_PROOF_V1\n";
  output << "backend " << result.backend << '\n';
  output << "initial_hash " << HexHash(result.initial_hash) << '\n';
  for (const ProofRecord& record : result.proof) {
    output << "record " << record.epoch << ' ' << HexHash(record.snapshot_hash) << ' '
           << record.edge_id << ' ' << record.u << ' ' << record.v << ' ' << ToString(record.method)
           << ' ' << record.witness << '\n';
  }
  for (const EpochMetrics& metrics : result.epochs) {
    output << "metrics " << metrics.epoch << ' ' << metrics.edges_before << ' ' << metrics.proposed
           << ' ' << metrics.verified << ' ' << metrics.rejected << ' ' << metrics.committed << ' '
           << metrics.propose_ms << ' ' << metrics.verify_ms << '\n';
  }
  output << "final_hash " << HexHash(result.final_hash) << '\n';
  output << "END\n";
  if (!output) {
    throw std::runtime_error("写证明文件失败: " + path.string());
  }
}

EliminationResult ReadProof(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("无法打开证明文件: " + path.string());
  }
  std::string magic;
  std::getline(input, magic);
  if (magic != "CUDAEE_PROOF_V1") {
    throw std::runtime_error("证明版本不受支持");
  }

  EliminationResult result;
  std::string line;
  bool saw_initial = false;
  bool saw_final = false;
  bool saw_end = false;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string kind;
    fields >> kind;
    if (kind == "backend") {
      fields >> result.backend;
    } else if (kind == "initial_hash") {
      std::string hash;
      fields >> hash;
      result.initial_hash = std::stoull(hash, nullptr, 16);
      saw_initial = true;
    } else if (kind == "record") {
      ProofRecord record;
      std::string hash;
      std::string method;
      fields >> record.epoch >> hash >> record.edge_id >> record.u >> record.v >> method >>
          record.witness;
      if (!fields || method != "JV") {
        throw std::runtime_error("证明 record 行无效");
      }
      record.snapshot_hash = std::stoull(hash, nullptr, 16);
      record.method = EliminationMethod::kJv;
      result.proof.push_back(record);
    } else if (kind == "metrics") {
      EpochMetrics metrics;
      fields >> metrics.epoch >> metrics.edges_before >> metrics.proposed >> metrics.verified >>
          metrics.rejected >> metrics.committed >> metrics.propose_ms >> metrics.verify_ms;
      if (!fields) {
        throw std::runtime_error("证明 metrics 行无效");
      }
      result.epochs.push_back(metrics);
    } else if (kind == "final_hash") {
      std::string hash;
      fields >> hash;
      result.final_hash = std::stoull(hash, nullptr, 16);
      saw_final = true;
    } else if (kind == "END") {
      saw_end = true;
      break;
    } else if (!kind.empty()) {
      throw std::runtime_error("证明包含未知行类型: " + kind);
    }
  }
  if (!saw_initial || !saw_final || !saw_end) {
    throw std::runtime_error("证明文件不完整");
  }
  return result;
}

EliminationResult ReplayProof(GraphSnapshot* const graph, const EliminationResult& expected) {
  if (graph == nullptr || graph->ContentHash() != expected.initial_hash) {
    throw std::runtime_error("证明的初始图哈希不匹配");
  }
  EliminationResult replayed;
  replayed.backend = "replay-cpu";
  replayed.initial_hash = graph->ContentHash();

  std::size_t cursor = 0;
  std::uint32_t expected_epoch = 0;
  while (cursor < expected.proof.size()) {
    const std::uint32_t epoch = expected.proof[cursor].epoch;
    if (epoch != expected_epoch) {
      throw std::runtime_error("证明 epoch 不连续");
    }
    const std::uint64_t snapshot_hash = graph->ContentHash();
    std::vector<Candidate> candidates;
    while (cursor < expected.proof.size() && expected.proof[cursor].epoch == epoch) {
      const ProofRecord& record = expected.proof[cursor++];
      if (record.snapshot_hash != snapshot_hash || record.edge_id < 0 ||
          static_cast<std::size_t>(record.edge_id) >= graph->edges.size()) {
        throw std::runtime_error("证明快照哈希或边编号不匹配");
      }
      const Edge& edge = graph->edges[static_cast<std::size_t>(record.edge_id)];
      if (edge.u != record.u || edge.v != record.v) {
        throw std::runtime_error("证明的边端点不匹配");
      }
      Candidate candidate{record.edge_id, record.witness, record.method};
      std::string reason;
      if (!VerifyJvCandidate(*graph, candidate, &reason)) {
        throw std::runtime_error("证明复核失败: " + reason);
      }
      candidates.push_back(candidate);
    }
    std::vector<Candidate> committed = CommitCandidates(graph, candidates);
    if (committed.size() != candidates.size()) {
      throw std::runtime_error("证明违反确定性最小度提交门禁");
    }
    for (const Candidate& candidate : committed) {
      const Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
      replayed.proof.push_back({epoch, snapshot_hash, candidate.edge_id, edge.u, edge.v,
                                candidate.witness, candidate.method});
    }
    ++expected_epoch;
  }
  replayed.final_hash = graph->ContentHash();
  if (replayed.final_hash != expected.final_hash) {
    throw std::runtime_error("证明重放后的最终图哈希不匹配");
  }
  return replayed;
}

} // namespace cudaee
