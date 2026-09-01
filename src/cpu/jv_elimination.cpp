#include "cuda_edge_elimination/elimination.hpp"

#include "elimination_commit.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

constexpr std::size_t kMaxCandidateNodes = 10;
constexpr std::uintmax_t kMaxEliminationProofBytes = 512U * 1024U * 1024U;
constexpr std::size_t kMaxEmbeddedHtProofBytes = 256U * 1024U * 1024U;
constexpr std::size_t kMaxEmbeddedHtAggregateBytes = 256U * 1024U * 1024U;
constexpr std::size_t kMaxEmbeddedHtProofs = 1000000U;
constexpr std::size_t kMaxEliminationRecords = 1000000U;
constexpr std::size_t kMaxEpochMetrics = 1000000U;

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

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

template <typename Integer>
Integer ParseIntegerToken(const std::string& token, const int base,
                          const std::string_view description) {
  static_assert(std::is_integral_v<Integer>);
  Integer value{};
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value, base);
  if (token.empty() || error != std::errc{} || end != token.data() + token.size()) {
    throw std::runtime_error("证明的" + std::string(description) + "非法");
  }
  return value;
}

std::uint64_t ParseHashToken(const std::string& token, const std::string_view description) {
  if (token.size() != 16U) {
    throw std::runtime_error("证明的" + std::string(description) + "非法");
  }
  return ParseIntegerToken<std::uint64_t>(token, 16, description);
}

void RequireLineEnd(std::istringstream* const fields, const std::string_view description) {
  std::string extra;
  if (*fields >> extra) {
    throw std::runtime_error("证明的" + std::string(description) + "含多余字段");
  }
}

EliminationMethod ParseEliminationMethod(const std::string_view method) {
  if (method == "JV") {
    return EliminationMethod::kJv;
  }
  if (method == "HT") {
    return EliminationMethod::kHamiltonTutte;
  }
  throw std::runtime_error("证明 record 的方法不受支持");
}

void ValidateProofContainer(const EliminationResult& result) {
  if (result.proof.size() > kMaxEliminationRecords || result.epochs.size() > kMaxEpochMetrics ||
      result.ht_proofs.size() > kMaxEmbeddedHtProofs ||
      result.ht_proofs.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("消元证明的 record、metrics 或 HT sidecar 数量超过上限");
  }
  std::vector<bool> referenced(result.ht_proofs.size(), false);
  for (const ProofRecord& record : result.proof) {
    if (record.method == EliminationMethod::kJv) {
      if (record.certificate_index != kNoEliminationCertificate) {
        throw std::runtime_error("JV record 不得引用 HT sidecar");
      }
      continue;
    }
    if (record.method != EliminationMethod::kHamiltonTutte || record.witness != -1 ||
        record.certificate_index >= result.ht_proofs.size() ||
        referenced[record.certificate_index]) {
      throw std::runtime_error("HT record 的 sidecar 引用非法或重复");
    }
    const HtRecursiveProof& proof = result.ht_proofs[record.certificate_index];
    if (!proof.proven || proof.snapshot_hash != record.snapshot_hash ||
        proof.target_edge.u != record.u || proof.target_edge.v != record.v) {
      throw std::runtime_error("HT record 与内嵌 sidecar 绑定不一致");
    }
    referenced[record.certificate_index] = true;
  }
  if (std::find(referenced.begin(), referenced.end(), false) != referenced.end()) {
    throw std::runtime_error("消元证明含未引用的 HT sidecar");
  }
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
    const auto snapshot_start = std::chrono::steady_clock::now();
    metrics.edges_before = graph->ActiveEdgeCount();
    const std::uint64_t snapshot_hash = graph->ContentHash();
    metrics.snapshot_ms = ElapsedMilliseconds(snapshot_start);

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

    const auto commit_start = std::chrono::steady_clock::now();
    std::vector<Candidate> committed =
        detail::CommitVerifiedCandidates(graph, std::move(verified), snapshot_hash);
    metrics.committed = committed.size();
    for (const Candidate& candidate : committed) {
      const Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
      result.proof.push_back({epoch, snapshot_hash, candidate.edge_id, edge.u, edge.v,
                              candidate.witness, candidate.method});
    }
    metrics.commit_ms = ElapsedMilliseconds(commit_start);
    result.epochs.push_back(metrics);
    if (committed.empty()) {
      break;
    }
  }
  result.final_hash = graph->ContentHash();
  return result;
}

void WriteProof(const std::filesystem::path& path, const EliminationResult& result) {
  ValidateProofContainer(result);
  const bool version2 = !result.ht_proofs.empty();
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("无法创建证明文件: " + path.string());
  }
  output << (version2 ? "CUDAEE_PROOF_V2\n" : "CUDAEE_PROOF_V1\n");
  output << "backend " << result.backend << '\n';
  output << "initial_hash " << HexHash(result.initial_hash) << '\n';
  for (const ProofRecord& record : result.proof) {
    output << "record " << record.epoch << ' ' << HexHash(record.snapshot_hash) << ' '
           << record.edge_id << ' ' << record.u << ' ' << record.v << ' ' << ToString(record.method)
           << ' ' << record.witness;
    if (version2) {
      output << ' ' << record.certificate_index;
    }
    output << '\n';
  }
  for (const EpochMetrics& metrics : result.epochs) {
    output << "metrics " << metrics.epoch << ' ' << metrics.edges_before << ' ' << metrics.proposed
           << ' ' << metrics.verified << ' ' << metrics.rejected << ' ' << metrics.committed << ' '
           << metrics.propose_ms << ' ' << metrics.verify_ms << '\n';
  }
  if (version2) {
    output << "ht_proof_count " << result.ht_proofs.size() << '\n';
    std::size_t total_bytes = 0U;
    for (std::size_t index = 0; index < result.ht_proofs.size(); ++index) {
      const std::string serialized = SerializeHtRecursiveProof(result.ht_proofs[index]);
      if (serialized.empty() || serialized.size() > kMaxEmbeddedHtProofBytes ||
          total_bytes > kMaxEmbeddedHtAggregateBytes - serialized.size()) {
        throw std::runtime_error("消元证明的内嵌 HT sidecar 超出大小上限");
      }
      total_bytes += serialized.size();
      output << "ht_proof " << index << ' ' << serialized.size() << '\n';
      output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
      output << "end_ht_proof\n";
    }
  }
  output << "final_hash " << HexHash(result.final_hash) << '\n';
  output << "END\n";
  if (!output) {
    throw std::runtime_error("写证明文件失败: " + path.string());
  }
}

EliminationResult ReadProof(const std::filesystem::path& path) {
  std::error_code size_error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, size_error);
  if (size_error || file_size > kMaxEliminationProofBytes) {
    throw std::runtime_error("消元证明无法读取或超过文件大小上限: " + path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("无法打开证明文件: " + path.string());
  }
  std::string magic;
  std::getline(input, magic);
  const bool version2 = magic == "CUDAEE_PROOF_V2";
  if (!version2 && magic != "CUDAEE_PROOF_V1") {
    throw std::runtime_error("证明版本不受支持");
  }

  EliminationResult result;
  std::string line;
  bool saw_backend = false;
  bool saw_initial = false;
  bool saw_final = false;
  bool saw_end = false;
  bool saw_ht_proof_count = false;
  std::size_t declared_ht_proofs = 0U;
  std::size_t embedded_bytes = 0U;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string kind;
    fields >> kind;
    if (kind == "backend") {
      if (saw_backend || !(fields >> result.backend) || result.backend.empty()) {
        throw std::runtime_error("证明 backend 行无效或重复");
      }
      RequireLineEnd(&fields, "backend 行");
      saw_backend = true;
    } else if (kind == "initial_hash") {
      if (saw_initial) {
        throw std::runtime_error("证明 initial_hash 重复");
      }
      std::string hash;
      if (!(fields >> hash)) {
        throw std::runtime_error("证明 initial_hash 行无效");
      }
      RequireLineEnd(&fields, "initial_hash 行");
      result.initial_hash = ParseHashToken(hash, " initial_hash");
      saw_initial = true;
    } else if (kind == "record") {
      ProofRecord record;
      std::string hash;
      std::string method;
      if (!(fields >> record.epoch >> hash >> record.edge_id >> record.u >> record.v >> method >>
            record.witness)) {
        throw std::runtime_error("证明 record 行无效");
      }
      record.snapshot_hash = ParseHashToken(hash, " record snapshot_hash");
      record.method = ParseEliminationMethod(method);
      if (version2) {
        std::string certificate_index;
        if (!(fields >> certificate_index)) {
          throw std::runtime_error("V2 证明 record 缺少 certificate_index");
        }
        record.certificate_index =
            ParseIntegerToken<std::uint32_t>(certificate_index, 10, " record certificate_index");
      } else if (record.method != EliminationMethod::kJv) {
        throw std::runtime_error("V1 证明只支持 JV record");
      }
      RequireLineEnd(&fields, "record 行");
      if (result.proof.size() >= kMaxEliminationRecords) {
        throw std::runtime_error("证明 record 数量超过上限");
      }
      result.proof.push_back(record);
    } else if (kind == "metrics") {
      EpochMetrics metrics;
      if (!(fields >> metrics.epoch >> metrics.edges_before >> metrics.proposed >>
            metrics.verified >> metrics.rejected >> metrics.committed >> metrics.propose_ms >>
            metrics.verify_ms) ||
          !std::isfinite(metrics.propose_ms) || !std::isfinite(metrics.verify_ms) ||
          metrics.propose_ms < 0.0 || metrics.verify_ms < 0.0) {
        throw std::runtime_error("证明 metrics 行无效");
      }
      RequireLineEnd(&fields, "metrics 行");
      if (result.epochs.size() >= kMaxEpochMetrics) {
        throw std::runtime_error("证明 metrics 数量超过上限");
      }
      result.epochs.push_back(metrics);
    } else if (kind == "ht_proof_count") {
      std::string count;
      if (!version2 || saw_ht_proof_count || !(fields >> count)) {
        throw std::runtime_error("证明 ht_proof_count 行无效或重复");
      }
      RequireLineEnd(&fields, "ht_proof_count 行");
      declared_ht_proofs = ParseIntegerToken<std::size_t>(count, 10, " ht_proof_count");
      if (declared_ht_proofs > kMaxEmbeddedHtProofs || declared_ht_proofs > result.proof.size()) {
        throw std::runtime_error("证明 ht_proof_count 超过上限");
      }
      result.ht_proofs.reserve(declared_ht_proofs);
      saw_ht_proof_count = true;
    } else if (kind == "ht_proof") {
      std::string index_token;
      std::string size_token;
      if (!version2 || !saw_ht_proof_count || !(fields >> index_token >> size_token)) {
        throw std::runtime_error("证明 ht_proof 头无效");
      }
      RequireLineEnd(&fields, "ht_proof 头");
      const std::size_t index = ParseIntegerToken<std::size_t>(index_token, 10, " ht_proof index");
      const std::size_t byte_count =
          ParseIntegerToken<std::size_t>(size_token, 10, " ht_proof byte_count");
      if (index != result.ht_proofs.size() || index >= declared_ht_proofs || byte_count == 0U ||
          byte_count > kMaxEmbeddedHtProofBytes ||
          embedded_bytes > kMaxEmbeddedHtAggregateBytes - byte_count) {
        throw std::runtime_error("证明 ht_proof 索引或大小非法");
      }
      embedded_bytes += byte_count;
      std::string serialized(byte_count, '\0');
      input.read(serialized.data(), static_cast<std::streamsize>(byte_count));
      if (static_cast<std::size_t>(input.gcount()) != byte_count) {
        throw std::runtime_error("证明内嵌 HT sidecar 被截断");
      }
      std::string end_marker;
      if (!std::getline(input, end_marker) || end_marker != "end_ht_proof") {
        throw std::runtime_error("证明内嵌 HT sidecar 缺少结束标记");
      }
      result.ht_proofs.push_back(ParseHtRecursiveProof(serialized));
    } else if (kind == "final_hash") {
      if (saw_final) {
        throw std::runtime_error("证明 final_hash 重复");
      }
      std::string hash;
      if (!(fields >> hash)) {
        throw std::runtime_error("证明 final_hash 行无效");
      }
      RequireLineEnd(&fields, "final_hash 行");
      result.final_hash = ParseHashToken(hash, " final_hash");
      saw_final = true;
    } else if (kind == "END") {
      RequireLineEnd(&fields, "END 行");
      saw_end = true;
      break;
    } else if (!kind.empty()) {
      throw std::runtime_error("证明包含未知行类型: " + kind);
    }
  }
  std::string trailing;
  if (input >> trailing) {
    throw std::runtime_error("证明 END 后存在多余字段");
  }
  if (!saw_backend || !saw_initial || !saw_final || !saw_end ||
      (version2 && (!saw_ht_proof_count || result.ht_proofs.size() != declared_ht_proofs))) {
    throw std::runtime_error("证明文件不完整");
  }
  ValidateProofContainer(result);
  return result;
}

EliminationResult ReplayProof(GraphSnapshot* const graph, const EliminationResult& expected) {
  if (graph == nullptr || graph->ContentHash() != expected.initial_hash) {
    throw std::runtime_error("证明的初始图哈希不匹配");
  }
  ValidateProofContainer(expected);
  // 完整重放在副本上进行；任意晚到的坏 sidecar 或最终哈希错误都不会部分修改调用方图。
  GraphSnapshot working = *graph;
  EliminationResult replayed;
  replayed.backend = "replay-cpu";
  replayed.initial_hash = working.ContentHash();

  std::size_t cursor = 0;
  std::uint32_t expected_epoch = 0;
  while (cursor < expected.proof.size()) {
    const std::uint32_t epoch = expected.proof[cursor].epoch;
    if (epoch != expected_epoch) {
      throw std::runtime_error("证明 epoch 不连续");
    }
    const std::uint64_t snapshot_hash = working.ContentHash();
    std::vector<Candidate> candidates;
    std::vector<const ProofRecord*> records;
    while (cursor < expected.proof.size() && expected.proof[cursor].epoch == epoch) {
      const ProofRecord& record = expected.proof[cursor++];
      if (record.snapshot_hash != snapshot_hash || record.edge_id < 0 ||
          static_cast<std::size_t>(record.edge_id) >= working.edges.size()) {
        throw std::runtime_error("证明快照哈希或边编号不匹配");
      }
      const Edge& edge = working.edges[static_cast<std::size_t>(record.edge_id)];
      if (edge.u != record.u || edge.v != record.v) {
        throw std::runtime_error("证明的边端点不匹配");
      }
      Candidate candidate{record.edge_id, record.witness, record.method};
      std::string reason;
      if (record.method == EliminationMethod::kJv) {
        if (!VerifyJvCandidate(working, candidate, &reason)) {
          throw std::runtime_error("JV 证明复核失败: " + reason);
        }
      } else if (record.method == EliminationMethod::kHamiltonTutte) {
        const HtRecursiveProof& ht_proof = expected.ht_proofs[record.certificate_index];
        if (!VerifyHtRecursiveProof(working, ht_proof, &reason)) {
          throw std::runtime_error("HT 证明复核失败: " + reason);
        }
      } else {
        throw std::runtime_error("证明方法不受支持");
      }
      candidates.push_back(candidate);
      records.push_back(&record);
    }
    std::vector<Candidate> committed =
        detail::CommitVerifiedCandidates(&working, std::move(candidates), snapshot_hash);
    if (committed.size() != records.size()) {
      throw std::runtime_error("证明违反确定性最小度提交门禁");
    }
    for (std::size_t index = 0; index < committed.size(); ++index) {
      const Candidate& candidate = committed[index];
      const ProofRecord& record = *records[index];
      if (candidate.edge_id != record.edge_id || candidate.witness != record.witness ||
          candidate.method != record.method) {
        throw std::runtime_error("证明 record 未按确定性提交顺序排列");
      }
      const Edge& edge = working.edges[static_cast<std::size_t>(candidate.edge_id)];
      std::uint32_t certificate_index = kNoEliminationCertificate;
      if (candidate.method == EliminationMethod::kHamiltonTutte) {
        if (record.certificate_index != replayed.ht_proofs.size()) {
          throw std::runtime_error("HT sidecar 未按确定性 record 顺序排列");
        }
        certificate_index = static_cast<std::uint32_t>(replayed.ht_proofs.size());
        replayed.ht_proofs.push_back(expected.ht_proofs[record.certificate_index]);
      }
      replayed.proof.push_back({epoch, snapshot_hash, candidate.edge_id, edge.u, edge.v,
                                candidate.witness, candidate.method, certificate_index});
    }
    ++expected_epoch;
  }
  replayed.final_hash = working.ContentHash();
  if (replayed.final_hash != expected.final_hash) {
    throw std::runtime_error("证明重放后的最终图哈希不匹配");
  }
  *graph = std::move(working);
  return replayed;
}

} // namespace cudaee
