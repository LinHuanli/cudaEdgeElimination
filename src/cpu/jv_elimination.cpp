#include "../fgpu/lp_box_verifier.hpp"
#include "../fgpu/quick_hs_verifier.hpp"
#include "cuda_edge_elimination/elimination.hpp"
#include "cuda_edge_elimination/fgpu.hpp"

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
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef CUDAEE_HAVE_ZLIB
#include <zlib.h>
#endif

namespace cudaee {
namespace {

constexpr std::size_t kMaxCandidateNodes = 10;
constexpr std::uintmax_t kMaxEliminationProofBytes = std::uintmax_t{512} * 1024U * 1024U;
constexpr std::size_t kMaxEmbeddedHtProofBytes = 256U * 1024U * 1024U;
constexpr std::size_t kMaxEmbeddedHtAggregateBytes = 256U * 1024U * 1024U;
// pcb442 的完整深扫已实测超过 2 GiB 原文；从完全图开始的单份
// one-shot 证书又实测超过 384 MiB 压缩 payload。V5 仍对原文、
// 压缩数据、单份 sidecar 和最终文件分别设置硬上限，不接受无界输入。
constexpr std::size_t kMaxEmbeddedHtRawAggregateBytes = std::size_t{8} * 1024U * 1024U * 1024U;
constexpr std::size_t kMaxEmbeddedHtCompressedAggregateBytes = 448U * 1024U * 1024U;
constexpr std::size_t kMaxEmbeddedHtProofs = 1000000U;
constexpr std::size_t kMaxEmbeddedLpProofs = 1000000U;
constexpr std::size_t kMaxEliminationRecords = 1000000U;
constexpr std::size_t kMaxEpochMetrics = 1000000U;

struct ScoredNode {
  std::int64_t score{};
  std::int32_t node{-1};
};

struct EncodedHtProof {
  std::string payload;
  std::size_t raw_size{};
  std::uint32_t checksum{};
};

#ifdef CUDAEE_HAVE_ZLIB
std::uint32_t Crc32(const std::string_view content) {
  uLong checksum = crc32(0L, Z_NULL, 0U);
  std::size_t offset = 0U;
  while (offset < content.size()) {
    const std::size_t remaining = content.size() - offset;
    const uInt chunk =
        static_cast<uInt>(std::min<std::size_t>(remaining, std::numeric_limits<uInt>::max()));
    checksum = crc32(checksum, reinterpret_cast<const Bytef*>(content.data() + offset), chunk);
    offset += chunk;
  }
  return static_cast<std::uint32_t>(checksum);
}

std::string CompressHtProof(const std::string_view raw) {
  if (raw.empty() || raw.size() > static_cast<std::size_t>(std::numeric_limits<uLong>::max())) {
    throw std::runtime_error("HT sidecar 无法交给 zlib 压缩");
  }
  uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
  std::string compressed(static_cast<std::size_t>(compressed_size), '\0');
  const int status = compress2(reinterpret_cast<Bytef*>(compressed.data()), &compressed_size,
                               reinterpret_cast<const Bytef*>(raw.data()),
                               static_cast<uLong>(raw.size()), Z_BEST_SPEED);
  if (status != Z_OK) {
    throw std::runtime_error("zlib 压缩 HT sidecar 失败，状态=" + std::to_string(status));
  }
  compressed.resize(static_cast<std::size_t>(compressed_size));
  return compressed;
}

std::string DecompressHtProof(const std::string_view compressed, const std::size_t raw_size) {
  if (compressed.empty() || raw_size == 0U ||
      compressed.size() > static_cast<std::size_t>(std::numeric_limits<uLong>::max()) ||
      raw_size > static_cast<std::size_t>(std::numeric_limits<uLongf>::max())) {
    throw std::runtime_error("压缩 HT sidecar 的长度无法交给 zlib");
  }
  uLongf actual_size = static_cast<uLongf>(raw_size);
  std::string raw(raw_size, '\0');
  const int status = uncompress(reinterpret_cast<Bytef*>(raw.data()), &actual_size,
                                reinterpret_cast<const Bytef*>(compressed.data()),
                                static_cast<uLong>(compressed.size()));
  if (status != Z_OK || actual_size != raw_size) {
    throw std::runtime_error("zlib 解压 HT sidecar 失败或长度不匹配，状态=" +
                             std::to_string(status));
  }
  return raw;
}
#endif

std::vector<EncodedHtProof> PrepareHtProofs(const std::vector<HtRecursiveProof>& proofs,
                                            const bool compressed) {
  std::vector<EncodedHtProof> result;
  result.reserve(proofs.size());
  std::size_t total_raw = 0U;
  std::size_t total_payload = 0U;
  for (const HtRecursiveProof& proof : proofs) {
    const std::string raw = SerializeHtRecursiveProof(proof);
    if (raw.empty() || raw.size() > kMaxEmbeddedHtProofBytes ||
        total_raw > kMaxEmbeddedHtRawAggregateBytes - raw.size()) {
      throw std::runtime_error("消元证明的 HT sidecar 原文超出有界解压上限");
    }
    EncodedHtProof encoded;
    encoded.raw_size = raw.size();
#ifdef CUDAEE_HAVE_ZLIB
    if (compressed) {
      encoded.checksum = Crc32(raw);
      encoded.payload = CompressHtProof(raw);
    } else
#else
    if (compressed) {
      throw std::runtime_error("当前构建未启用 zlib，不能生成 V5 HT 证书");
    } else
#endif
    {
      encoded.payload = raw;
    }
    const std::size_t payload_limit =
        compressed ? kMaxEmbeddedHtCompressedAggregateBytes : kMaxEmbeddedHtAggregateBytes;
    if (encoded.payload.empty() || encoded.payload.size() > kMaxEmbeddedHtProofBytes) {
      throw std::runtime_error("消元证明的单份内嵌 HT sidecar 超出大小上限: payload=" +
                               std::to_string(encoded.payload.size()) +
                               ", limit=" + std::to_string(kMaxEmbeddedHtProofBytes));
    }
    if (total_payload > payload_limit - encoded.payload.size()) {
      throw std::runtime_error("消元证明的内嵌 HT sidecar 累计 payload 超出大小上限: " +
                               std::to_string(total_payload + encoded.payload.size()) + " > " +
                               std::to_string(payload_limit));
    }
    total_raw += raw.size();
    total_payload += encoded.payload.size();
    result.push_back(std::move(encoded));
  }
  return result;
}

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
  if (method == "GEOM_MAIN") {
    return EliminationMethod::kGeometryMain;
  }
  if (method == "LP_BOX") {
    return EliminationMethod::kLpBox;
  }
  if (method == "GPU_QUICK_HS") {
    return EliminationMethod::kGpuQuickHs;
  }
  throw std::runtime_error("证明 record 的方法不受支持");
}

void ValidateProofContainer(const EliminationResult& result) {
  if (result.proof.size() > kMaxEliminationRecords || result.epochs.size() > kMaxEpochMetrics ||
      result.ht_proofs.size() > kMaxEmbeddedHtProofs ||
      result.ht_proofs.size() > std::numeric_limits<std::uint32_t>::max() ||
      result.lp_box_proofs.size() > kMaxEmbeddedLpProofs ||
      result.lp_box_proofs.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("消元证明的 record、metrics 或 HT sidecar 数量超过上限");
  }
  std::vector<bool> referenced(result.ht_proofs.size(), false);
  std::vector<bool> lp_referenced(result.lp_box_proofs.size(), false);
  for (const ProofRecord& record : result.proof) {
    if (record.method == EliminationMethod::kJv) {
      if (record.certificate_index != kNoEliminationCertificate || record.second_witness != -1) {
        throw std::runtime_error("JV record 不得引用 HT sidecar");
      }
      continue;
    }
    if (record.method == EliminationMethod::kGeometryMain) {
      if (record.certificate_index != kNoEliminationCertificate || record.second_witness < 0) {
        throw std::runtime_error("GEOM_MAIN record 的第二见证或 sidecar 非法");
      }
      continue;
    }
    if (record.method == EliminationMethod::kGpuQuickHs) {
      if (record.certificate_index != kNoEliminationCertificate || record.witness < 0 ||
          record.second_witness < 0) {
        throw std::runtime_error("GPU_QUICK_HS record 的 c,d 见证或 sidecar 非法");
      }
      continue;
    }
    if (record.method == EliminationMethod::kLpBox) {
      if (record.witness != -1 || record.second_witness != -1 ||
          record.certificate_index >= result.lp_box_proofs.size()) {
        throw std::runtime_error("LP_BOX record 的见证或 sidecar 引用非法");
      }
      const LpBoxProof& proof = result.lp_box_proofs[record.certificate_index];
      if (proof.snapshot_hash != record.snapshot_hash || proof.incumbent_cost < 0 ||
          proof.fractional_bits > 40U) {
        throw std::runtime_error("LP_BOX record 与量化 sidecar 绑定不一致");
      }
      lp_referenced[record.certificate_index] = true;
      continue;
    }
    if (record.method != EliminationMethod::kHamiltonTutte || record.witness != -1 ||
        record.second_witness != -1 || record.certificate_index >= result.ht_proofs.size() ||
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
  if (std::find(lp_referenced.begin(), lp_referenced.end(), false) != lp_referenced.end()) {
    throw std::runtime_error("消元证明含未引用的 LP box sidecar");
  }
}

} // namespace

std::vector<Candidate> FindJvCandidatesCpu(const GraphSnapshot& graph, const JvCandidateMode mode) {
  std::vector<Candidate> candidates;
  for (std::size_t edge_id = 0; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
      continue;
    }
    std::vector<std::int32_t> witness_nodes;
    if (mode == JvCandidateMode::kExhaustive) {
      witness_nodes.resize(static_cast<std::size_t>(graph.dimension));
      std::iota(witness_nodes.begin(), witness_nodes.end(), 0);
    } else {
      witness_nodes = PotentialCandidateNodes(graph, edge.u, edge.v);
    }
    for (const std::int32_t witness : witness_nodes) {
      if (witness == edge.u || witness == edge.v) {
        continue;
      }
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
                                   const std::uint32_t max_rounds, const JvCandidateMode mode) {
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
  result.backend = (use_cuda ? "cuda" : "cpu") +
                   std::string(mode == JvCandidateMode::kExhaustive ? "-exhaustive" : "-quick");
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
    JvCudaCacheUsage cuda_cache;
    std::vector<Candidate> proposed =
        use_cuda ? FindJvCandidatesCuda(*graph, &selected_device, &cuda_cache, mode)
                 : FindJvCandidatesCpu(*graph, mode);
    metrics.propose_ms = ElapsedMilliseconds(propose_start);
    metrics.proposed = proposed.size();
    metrics.jv_static_cache_hit = cuda_cache.static_hit;
    metrics.jv_workspace_cache_hit = cuda_cache.workspace_hit;
    metrics.jv_resident_bytes = cuda_cache.resident_bytes;
    metrics.jv_h2d_ms = cuda_cache.h2d_ms;
    metrics.jv_kernel_ms = cuda_cache.kernel_ms;
    metrics.jv_d2h_ms = cuda_cache.d2h_ms;

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
  // V5 把大量重复的 HT 文本 sidecar 逐个 zlib 压缩，同时保留原始
  // 长度和 CRC32。先完成全部编码与上限检查，避免失败时留下半个证书。
#ifdef CUDAEE_HAVE_ZLIB
  const bool version5 = !result.ht_proofs.empty();
#else
  const bool version5 = false;
#endif
  const std::vector<EncodedHtProof> encoded_ht = PrepareHtProofs(result.ht_proofs, version5);
  const bool version4 = version5 || !result.lp_box_proofs.empty();
  const bool version3 =
      version4 ||
      std::any_of(result.proof.begin(), result.proof.end(), [](const ProofRecord& record) {
        return record.method == EliminationMethod::kGeometryMain ||
               record.method == EliminationMethod::kGpuQuickHs;
      });
  const bool version2 = version3 || !result.ht_proofs.empty();
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("无法创建证明文件: " + path.string());
  }
  output << (version5   ? "CUDAEE_PROOF_V5\n"
             : version4 ? "CUDAEE_PROOF_V4\n"
             : version3 ? "CUDAEE_PROOF_V3\n"
                        : (version2 ? "CUDAEE_PROOF_V2\n" : "CUDAEE_PROOF_V1\n"));
  output << "backend " << result.backend << '\n';
  output << "initial_hash " << HexHash(result.initial_hash) << '\n';
  for (const ProofRecord& record : result.proof) {
    output << "record " << record.epoch << ' ' << HexHash(record.snapshot_hash) << ' '
           << record.edge_id << ' ' << record.u << ' ' << record.v << ' ' << ToString(record.method)
           << ' ' << record.witness;
    if (version2) {
      output << ' ' << record.certificate_index;
    }
    if (version3) {
      output << ' ' << record.second_witness;
    }
    output << '\n';
  }
  for (const EpochMetrics& metrics : result.epochs) {
    // wall-clock 只属于 report/manifest，不能进入可移植数学证书；否则同一
    // proof 在不同调度和硬件上只因计时抖动就产生不同字节。
    output << "metrics " << metrics.epoch << ' ' << metrics.edges_before << ' ' << metrics.proposed
           << ' ' << metrics.verified << ' ' << metrics.rejected << ' ' << metrics.committed << ' '
           << 0 << ' ' << 0 << '\n';
  }
  if (version2) {
    output << "ht_proof_count " << encoded_ht.size() << '\n';
    for (std::size_t index = 0; index < encoded_ht.size(); ++index) {
      const EncodedHtProof& proof = encoded_ht[index];
      if (version5) {
        output << "ht_proof_zlib " << index << ' ' << proof.raw_size << ' ' << proof.payload.size()
               << ' ' << std::hex << std::setw(8) << std::setfill('0') << proof.checksum << std::dec
               << std::setfill(' ') << '\n';
        output.write(proof.payload.data(), static_cast<std::streamsize>(proof.payload.size()));
        output << "end_ht_proof_zlib\n";
      } else {
        output << "ht_proof " << index << ' ' << proof.payload.size() << '\n';
        output.write(proof.payload.data(), static_cast<std::streamsize>(proof.payload.size()));
        output << "end_ht_proof\n";
      }
    }
  }
  if (version4) {
    output << "lp_box_proof_count " << result.lp_box_proofs.size() << '\n';
    for (std::size_t index = 0U; index < result.lp_box_proofs.size(); ++index) {
      const LpBoxProof& proof = result.lp_box_proofs[index];
      output << "lp_box_proof " << index << ' ' << HexHash(proof.snapshot_hash) << ' '
             << proof.fractional_bits << ' ' << proof.incumbent_cost << '\n';
      output << "lp_box_dual " << proof.vertex_dual_numerator.size();
      for (const std::int64_t value : proof.vertex_dual_numerator) {
        output << ' ' << value;
      }
      output << "\nend_lp_box_proof\n";
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
  const bool version5 = magic == "CUDAEE_PROOF_V5";
#ifndef CUDAEE_HAVE_ZLIB
  if (version5) {
    throw std::runtime_error("当前构建未启用 zlib，无法读取 V5 HT 证书");
  }
#endif
  const bool version4 = version5 || magic == "CUDAEE_PROOF_V4";
  const bool version3 = version4 || magic == "CUDAEE_PROOF_V3";
  const bool version2 = version3 || magic == "CUDAEE_PROOF_V2";
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
  bool saw_lp_proof_count = false;
  std::size_t declared_ht_proofs = 0U;
  std::size_t declared_lp_proofs = 0U;
  std::size_t embedded_bytes = 0U;
  std::size_t embedded_raw_bytes = 0U;
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
      if (version3) {
        std::string second_witness;
        if (!(fields >> second_witness)) {
          throw std::runtime_error("V3 证明 record 缺少 second_witness");
        }
        record.second_witness =
            ParseIntegerToken<std::int32_t>(second_witness, 10, " record second_witness");
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
      if (!version2 || version5 || !saw_ht_proof_count || !(fields >> index_token >> size_token)) {
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
    } else if (kind == "ht_proof_zlib") {
      std::string index_token;
      std::string raw_size_token;
      std::string compressed_size_token;
      std::string checksum_token;
      if (!version5 || !saw_ht_proof_count ||
          !(fields >> index_token >> raw_size_token >> compressed_size_token >> checksum_token)) {
        throw std::runtime_error("证明 ht_proof_zlib 头无效");
      }
      RequireLineEnd(&fields, "ht_proof_zlib 头");
      const std::size_t index =
          ParseIntegerToken<std::size_t>(index_token, 10, " ht_proof_zlib index");
      const std::size_t raw_size =
          ParseIntegerToken<std::size_t>(raw_size_token, 10, " ht_proof_zlib raw_size");
      const std::size_t compressed_size = ParseIntegerToken<std::size_t>(
          compressed_size_token, 10, " ht_proof_zlib compressed_size");
      if (checksum_token.size() != 8U) {
        throw std::runtime_error("证明 ht_proof_zlib checksum 必须是 8 位十六进制数");
      }
      const std::uint32_t expected_checksum =
          ParseIntegerToken<std::uint32_t>(checksum_token, 16, " ht_proof_zlib checksum");
      if (index != result.ht_proofs.size() || index >= declared_ht_proofs || raw_size == 0U ||
          raw_size > kMaxEmbeddedHtProofBytes || compressed_size == 0U ||
          compressed_size > kMaxEmbeddedHtProofBytes ||
          embedded_raw_bytes > kMaxEmbeddedHtRawAggregateBytes - raw_size ||
          embedded_bytes > kMaxEmbeddedHtCompressedAggregateBytes - compressed_size) {
        throw std::runtime_error("证明 ht_proof_zlib 索引或大小非法");
      }
      embedded_raw_bytes += raw_size;
      embedded_bytes += compressed_size;
      std::string compressed(compressed_size, '\0');
      input.read(compressed.data(), static_cast<std::streamsize>(compressed_size));
      if (static_cast<std::size_t>(input.gcount()) != compressed_size) {
        throw std::runtime_error("证明内嵌压缩 HT sidecar 被截断");
      }
      std::string end_marker;
      if (!std::getline(input, end_marker) || end_marker != "end_ht_proof_zlib") {
        throw std::runtime_error("证明内嵌压缩 HT sidecar 缺少结束标记");
      }
#ifdef CUDAEE_HAVE_ZLIB
      const std::string serialized = DecompressHtProof(compressed, raw_size);
      if (Crc32(serialized) != expected_checksum) {
        throw std::runtime_error("证明内嵌压缩 HT sidecar 的 CRC32 校验失败");
      }
      result.ht_proofs.push_back(ParseHtRecursiveProof(serialized));
#else
      static_cast<void>(expected_checksum);
      throw std::runtime_error("当前构建未启用 zlib，无法解压 HT sidecar");
#endif
    } else if (kind == "lp_box_proof_count") {
      std::string count;
      if (!version4 || saw_lp_proof_count || !(fields >> count)) {
        throw std::runtime_error("证明 lp_box_proof_count 行无效或重复");
      }
      RequireLineEnd(&fields, "lp_box_proof_count 行");
      declared_lp_proofs = ParseIntegerToken<std::size_t>(count, 10, " lp_box_proof_count");
      if (declared_lp_proofs > kMaxEmbeddedLpProofs || declared_lp_proofs > result.proof.size()) {
        throw std::runtime_error("证明 lp_box_proof_count 超过上限");
      }
      result.lp_box_proofs.reserve(declared_lp_proofs);
      saw_lp_proof_count = true;
    } else if (kind == "lp_box_proof") {
      std::string index_token;
      std::string hash_token;
      LpBoxProof proof;
      if (!version4 || !saw_lp_proof_count ||
          !(fields >> index_token >> hash_token >> proof.fractional_bits >> proof.incumbent_cost)) {
        throw std::runtime_error("证明 lp_box_proof 头无效");
      }
      RequireLineEnd(&fields, "lp_box_proof 头");
      const std::size_t index =
          ParseIntegerToken<std::size_t>(index_token, 10, " lp_box_proof index");
      if (index != result.lp_box_proofs.size() || index >= declared_lp_proofs) {
        throw std::runtime_error("证明 lp_box_proof 索引非法");
      }
      proof.snapshot_hash = ParseHashToken(hash_token, " lp_box_proof snapshot_hash");
      std::string dual_line;
      if (!std::getline(input, dual_line)) {
        throw std::runtime_error("证明 lp_box_dual 被截断");
      }
      std::istringstream dual_fields(dual_line);
      std::string dual_kind;
      std::size_t dual_count = 0U;
      if (!(dual_fields >> dual_kind >> dual_count) || dual_kind != "lp_box_dual" ||
          dual_count > 100000000U) {
        throw std::runtime_error("证明 lp_box_dual 头非法");
      }
      proof.vertex_dual_numerator.resize(dual_count);
      for (std::int64_t& value : proof.vertex_dual_numerator) {
        if (!(dual_fields >> value)) {
          throw std::runtime_error("证明 lp_box_dual 数组被截断");
        }
      }
      RequireLineEnd(&dual_fields, "lp_box_dual 行");
      std::string end_marker;
      if (!std::getline(input, end_marker) || end_marker != "end_lp_box_proof") {
        throw std::runtime_error("证明 lp_box_proof 缺少结束标记");
      }
      result.lp_box_proofs.push_back(std::move(proof));
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
      (version2 && (!saw_ht_proof_count || result.ht_proofs.size() != declared_ht_proofs)) ||
      (version4 && (!saw_lp_proof_count || result.lp_box_proofs.size() != declared_lp_proofs))) {
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
  replayed.lp_box_proofs = expected.lp_box_proofs;
  GeometryVerificationData geometry_data;
  bool geometry_data_ready = false;
  std::vector<std::optional<LpBoxVerificationData>> lp_verification_cache(
      expected.lp_box_proofs.size());

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
      candidates.push_back({record.edge_id, record.witness, record.method, record.second_witness});
      records.push_back(&record);
    }

    // sidecar/最近邻的共享部分每个不可变 epoch 只构造一次；逐 record 的纯只读
    // 数学复核可并行，最后仍按规范 record 顺序报告首个错误并原子提交。
    const bool needs_geometry =
        std::any_of(records.begin(), records.end(), [](const ProofRecord* const record) {
          return record->method == EliminationMethod::kGeometryMain;
        });
    if (needs_geometry && !geometry_data_ready) {
      geometry_data = BuildGeometryVerificationData(working);
      geometry_data_ready = true;
    }
    const bool needs_quick_hs =
        std::any_of(records.begin(), records.end(), [](const ProofRecord* const record) {
          return record->method == EliminationMethod::kGpuQuickHs;
        });
    std::optional<detail::QuickHsVerificationData> quick_hs_data;
    if (needs_quick_hs) {
      quick_hs_data.emplace(detail::BuildQuickHsVerificationData(working));
    }
    for (const ProofRecord* const record : records) {
      if (record->method != EliminationMethod::kLpBox) {
        continue;
      }
      if (record->certificate_index >= expected.lp_box_proofs.size()) {
        throw std::runtime_error("LP_BOX record 的 sidecar 索引越界");
      }
      std::optional<LpBoxVerificationData>& cached =
          lp_verification_cache[record->certificate_index];
      if (!cached.has_value()) {
        cached =
            BuildLpBoxVerificationData(working, expected.lp_box_proofs[record->certificate_index]);
        if (!cached->certified) {
          throw std::runtime_error("LP_BOX 共享证明复核失败: " + cached->reason);
        }
      }
    }

    std::vector<std::uint8_t> valid(records.size(), 0U);
    std::vector<std::string> reasons(records.size());
#ifdef CUDAEE_HAS_OPENMP
    // HT sidecar 与 Quick-HS 的邻域组合量可相差数个数量级；64 个 record
    // 的粗粒度 chunk 会留下单线程长尾。二者逐份动态领取，其他廉价
    // record 仍保留 64 条批量以减少调度开销。
    const bool needs_ht =
        std::any_of(records.begin(), records.end(), [](const ProofRecord* const record) {
          return record->method == EliminationMethod::kHamiltonTutte;
        });
    const int verification_chunk_size = (needs_ht || needs_quick_hs) ? 1 : 64;
#pragma omp parallel for schedule(dynamic, verification_chunk_size)
#endif
    for (std::int64_t record_index = 0; record_index < static_cast<std::int64_t>(records.size());
         ++record_index) {
      const std::size_t index = static_cast<std::size_t>(record_index);
      const ProofRecord& record = *records[index];
      const Candidate& candidate = candidates[index];
      bool accepted = false;
      try {
        if (record.method == EliminationMethod::kJv) {
          accepted = VerifyJvCandidate(working, candidate, &reasons[index]);
        } else if (record.method == EliminationMethod::kHamiltonTutte) {
          accepted = VerifyHtRecursiveProof(working, expected.ht_proofs[record.certificate_index],
                                            &reasons[index]);
        } else if (record.method == EliminationMethod::kGeometryMain) {
          accepted = VerifyGeometryCandidate(working, geometry_data, candidate, &reasons[index]);
        } else if (record.method == EliminationMethod::kLpBox) {
          accepted = detail::VerifyLpBoxCandidateForSnapshot(
              working, expected.lp_box_proofs[record.certificate_index],
              *lp_verification_cache[record.certificate_index], candidate, snapshot_hash,
              &reasons[index]);
        } else if (record.method == EliminationMethod::kGpuQuickHs) {
          accepted =
              detail::VerifyQuickHsCandidate(working, *quick_hs_data, candidate, &reasons[index]);
        } else {
          reasons[index] = "证明方法不受支持";
        }
      } catch (const std::exception& error) {
        reasons[index] = std::string("复核器异常: ") + error.what();
      }
      valid[index] = static_cast<std::uint8_t>(accepted);
    }
    for (std::size_t index = 0U; index < records.size(); ++index) {
      if (valid[index] != 0U) {
        continue;
      }
      const EliminationMethod method = records[index]->method;
      const std::string prefix = method == EliminationMethod::kJv              ? "JV"
                                 : method == EliminationMethod::kHamiltonTutte ? "HT"
                                 : method == EliminationMethod::kGeometryMain  ? "GEOM_MAIN"
                                 : method == EliminationMethod::kLpBox         ? "LP_BOX"
                                                                               : "GPU_QUICK_HS";
      throw std::runtime_error(prefix + " 证明复核失败: " + reasons[index]);
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
          candidate.method != record.method || candidate.second_witness != record.second_witness) {
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
      } else if (candidate.method == EliminationMethod::kLpBox) {
        certificate_index = record.certificate_index;
      }
      replayed.proof.push_back({epoch, snapshot_hash, candidate.edge_id, edge.u, edge.v,
                                candidate.witness, candidate.method, certificate_index,
                                candidate.second_witness});
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
