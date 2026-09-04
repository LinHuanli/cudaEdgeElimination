#include "cuda_edge_elimination/fgpu.hpp"

#include "../cpu/elimination_commit.hpp"
#include "lp_box_verifier.hpp"
#include "quick_hs_verifier.hpp"
#include "resident_backend.hpp"

#include "cuda_edge_elimination/tour.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaee {
namespace {

using SteadyClock = std::chrono::steady_clock;

double ElapsedMilliseconds(const SteadyClock::time_point begin) {
  return std::chrono::duration<double, std::milli>(SteadyClock::now() - begin).count();
}

std::vector<std::uint8_t> BuildProtectedEdgeMask(const GraphSnapshot& graph,
                                                 const std::vector<std::int32_t>& tour) {
  std::vector<std::uint8_t> result(graph.edges.size(), 0U);
  if (tour.empty()) {
    return result;
  }
  std::vector<std::int32_t> edge_id_by_pair(
      static_cast<std::size_t>(graph.dimension) * static_cast<std::size_t>(graph.dimension), -1);
  for (std::size_t edge_id = 0U; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    edge_id_by_pair[static_cast<std::size_t>(edge.u) * static_cast<std::size_t>(graph.dimension) +
                    static_cast<std::size_t>(edge.v)] = static_cast<std::int32_t>(edge_id);
    edge_id_by_pair[static_cast<std::size_t>(edge.v) * static_cast<std::size_t>(graph.dimension) +
                    static_cast<std::size_t>(edge.u)] = static_cast<std::int32_t>(edge_id);
  }
  for (std::size_t index = 0U; index < tour.size(); ++index) {
    const std::int32_t u = tour[index];
    const std::int32_t v = tour[(index + 1U) % tour.size()];
    const std::int32_t edge_id =
        edge_id_by_pair[static_cast<std::size_t>(u) * static_cast<std::size_t>(graph.dimension) +
                        static_cast<std::size_t>(v)];
    if (edge_id < 0 || !graph.edges[static_cast<std::size_t>(edge_id)].active) {
      throw std::runtime_error("resident 输入图缺少受保护 tour 边");
    }
    result[static_cast<std::size_t>(edge_id)] = 1U;
  }
  return result;
}

std::vector<Edge> DeriveDegreeTwoFixedEdges(const GraphSnapshot& graph) {
  std::vector<Edge> fixed;
  for (const Edge& edge : graph.edges) {
    if (edge.active && (graph.Degree(edge.u) == 2 || graph.Degree(edge.v) == 2)) {
      fixed.push_back(edge);
    }
  }
  return fixed;
}

void WriteFixedEdges(const std::filesystem::path& path, const std::int32_t dimension,
                     const std::vector<Edge>& fixed) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 resident fixed 输出: " + path.string());
  }
  output << dimension << ' ' << fixed.size() << '\n';
  for (const Edge& edge : fixed) {
    output << edge.u << ' ' << edge.v << ' ' << edge.weight << '\n';
  }
}

void WriteEmptyNonpairs(const std::filesystem::path& path, const std::int32_t dimension) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 resident non-pair 输出: " + path.string());
  }
  output << dimension << " 0\n";
  for (std::int32_t node = 0; node < dimension; ++node) {
    output << node << " 0\n";
  }
}

void WriteResidentManifest(const std::filesystem::path& path, const FgpuInput& input,
                           const FgpuResidentConfig& config, const FgpuResidentRunReport& report,
                           const ProtectedTourCheck* const tour_check) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 resident manifest: " + path.string());
  }
  output << "FGPU_RESIDENT_MANIFEST_V2\n";
  output << "instance " << input.instance.string() << '\n';
  output << "input_edges " << input.input_edges.string() << '\n';
  output << "device " << config.device << '\n';
  output << "selected_device " << report.selected_device << '\n';
  output << "enable_quick_hs " << (config.enable_quick_hs ? 1 : 0) << '\n';
  output << "enable_jv " << (config.enable_jv ? 1 : 0) << '\n';
  output << "enable_geometry " << (config.enable_geometry ? 1 : 0) << '\n';
  output << "enable_pdlp " << (config.enable_pdlp ? 1 : 0) << '\n';
  output << "cpu_audit_enabled " << (config.enable_cpu_audit ? 1 : 0) << '\n';
  output << "result_trust " << (report.cpu_audited ? "cpu-audited" : "gpu-raw") << '\n';
  output << "potential_candidates " << config.potential_candidates << '\n';
  output << "pdlp_iterations_budget " << config.pdlp_iterations << '\n';
  output << "max_pdlp_epochs " << config.max_pdlp_epochs << '\n';
  output << "max_hs_epochs " << config.max_hs_epochs << '\n';
  output << "max_jv_rounds " << config.max_jv_rounds << '\n';
  output << "initial_hash " << HexHash(report.initial_hash) << '\n';
  output << "final_hash " << HexHash(report.final_hash) << '\n';
  output << "initial_edges " << report.initial_edges << '\n';
  output << "final_edges " << report.final_edges << '\n';
  output << "jv_committed " << report.jv_committed << '\n';
  output << "quick_hs_committed " << report.quick_hs_committed << '\n';
  output << "geometry_committed " << report.geometry_committed << '\n';
  output << "lp_committed " << report.lp_committed << '\n';
  output << "hs_epochs " << report.hs_epochs << '\n';
  output << "jv_rounds " << report.jv_rounds << '\n';
  output << "pdlp_epochs " << report.pdlp_epochs << '\n';
  output << "converged " << (report.converged ? 1 : 0) << '\n';
  output << "resident_bytes " << report.resident_bytes << '\n';
  output << "upload_ms " << report.upload_ms << '\n';
  output << "gpu_kernel_ms " << report.gpu_kernel_ms << '\n';
  output << "geometry_ms " << report.geometry_ms << '\n';
  output << "pdlp_ms " << report.pdlp_ms << '\n';
  output << "jv_ms " << report.jv_ms << '\n';
  output << "quick_hs_ms " << report.quick_hs_ms << '\n';
  output << "compaction_ms " << report.compaction_ms << '\n';
  output << "gpu_download_ms " << report.gpu_download_ms << '\n';
  output << "gpu_solve_wall_ms " << report.gpu_solve_wall_ms << '\n';
  output << "cpu_audit_ms " << report.cpu_audit_ms << '\n';
  output << "output_ms " << report.output_ms << '\n';
  output << "end_to_end_ms " << report.end_to_end_ms << '\n';
  output << "trusted_total_ms " << report.trusted_total_ms << '\n';
  output << "certificate_records " << report.certificate.proof.size() << '\n';
  output << "certificate_bytes " << report.certificate_bytes << '\n';
  output << "known_tour_checked " << (tour_check != nullptr ? 1 : 0) << '\n';
  if (tour_check != nullptr) {
    output << "known_tour_cost " << tour_check->cost << '\n';
    output << "known_tour_missing_edges " << tour_check->missing_edges << '\n';
    output << "known_tour_hash " << HexHash(tour_check->tour_hash) << '\n';
  }
  output << "END\n";
  if (!output) {
    throw std::runtime_error("写 resident manifest 失败");
  }
}

} // namespace

FgpuResidentRunReport RunFgpuResidentElimination(const FgpuInput& input,
                                                 const FgpuOutputPaths& outputs,
                                                 const FgpuResidentConfig& config) {
  const SteadyClock::time_point total_begin = SteadyClock::now();
  if (input.instance.empty() || outputs.edges.empty() || outputs.fixed.empty() ||
      outputs.nonpairs.empty() || outputs.manifest.empty() ||
      (config.enable_cpu_audit && outputs.certificate.empty())) {
    throw std::invalid_argument(
        "resident 需要 instance、边/fixed/nonpairs/manifest，启用审计时还需要 certificate");
  }
  if (config.device < 0 || config.pdlp_iterations == 0U || config.potential_candidates < 2U ||
      config.potential_candidates > 32U ||
      (!config.enable_quick_hs && !config.enable_jv && !config.enable_geometry &&
       !config.enable_pdlp)) {
    throw std::invalid_argument("resident device、预算或阶段开关非法");
  }
  if (config.enable_pdlp && input.tour.empty()) {
    throw std::invalid_argument("resident PDLP 需要 tour 提供合法 incumbent 上界");
  }
  GraphSnapshot initial = input.input_edges.empty()
                              ? GraphSnapshot::LoadComplete(input.instance)
                              : GraphSnapshot::Load(input.instance, input.input_edges);
  std::vector<std::int32_t> tour;
  ProtectedTourCheck initial_tour_check;
  if (!input.tour.empty()) {
    tour = ReadTsplibTour(input.tour, initial.dimension);
    initial_tour_check = CheckProtectedTour(initial, tour);
    if ((input.tour_is_known_optimum && initial_tour_check.missing_edges != 0U) ||
        (input.expected_tour_cost >= 0 && initial_tour_check.cost != input.expected_tour_cost)) {
      throw std::runtime_error("resident 输入 tour 门禁失败");
    }
  }
  const std::vector<std::uint8_t> protected_edges = BuildProtectedEdgeMask(initial, tour);

  detail::ResidentGpuOptions device_options;
  device_options.device = config.device;
  device_options.max_hs_epochs = config.max_hs_epochs;
  device_options.max_jv_rounds = config.max_jv_rounds;
  device_options.enable_quick_hs = config.enable_quick_hs;
  device_options.enable_jv = config.enable_jv;
  device_options.enable_geometry = config.enable_geometry;
  device_options.enable_pdlp = config.enable_pdlp;
  device_options.collect_trace = config.enable_cpu_audit;
  device_options.potential_candidates = config.potential_candidates;
  device_options.pdlp_iterations = config.pdlp_iterations;
  device_options.max_pdlp_epochs = config.max_pdlp_epochs;
  device_options.incumbent_cost = tour.empty() ? -1 : initial_tour_check.cost;
  const detail::ResidentGpuResult device =
      detail::RunResidentEliminationCuda(initial, protected_edges, device_options);

  FgpuResidentRunReport report;
  report.initial_hash = initial.ContentHash();
  report.initial_edges = initial.ActiveEdgeCount();
  report.final_edges = device.final_edges;
  report.jv_committed = device.jv_committed;
  report.quick_hs_committed = device.quick_hs_committed;
  report.geometry_committed = device.geometry_committed;
  report.lp_committed = device.lp_committed;
  report.hs_epochs = device.hs_epochs;
  report.jv_rounds = device.jv_rounds;
  report.pdlp_epochs = device.pdlp_epochs;
  report.converged = device.converged;
  report.cpu_audited = config.enable_cpu_audit;
  report.selected_device = device.selected_device;
  report.resident_bytes = device.resident_bytes;
  report.upload_ms = device.upload_ms;
  report.gpu_kernel_ms = device.kernel_ms;
  report.geometry_ms = device.geometry_ms;
  report.pdlp_ms = device.pdlp_ms;
  report.jv_ms = device.jv_ms;
  report.quick_hs_ms = device.quick_hs_ms;
  report.compaction_ms = device.compaction_ms;
  report.gpu_download_ms = device.download_ms;
  report.gpu_solve_wall_ms = device.solve_wall_ms;
  report.certificate.backend =
      config.enable_cpu_audit ? "cuda-fully-resident-cpu-audited" : "cuda-fully-resident-gpu-raw";
  report.certificate.initial_hash = report.initial_hash;
  report.certificate.final_hash = report.initial_hash;

  GraphSnapshot audited = initial;
  if (config.enable_cpu_audit) {
    const SteadyClock::time_point audit_begin = SteadyClock::now();
    std::uint32_t proof_epoch = 0U;
    for (const detail::ResidentTraceEpoch& trace : device.epochs) {
      if (trace.edge_ids.size() != trace.first_witness.size() ||
          trace.edge_ids.size() != trace.second_witness.size() ||
          trace.edges_before != audited.ActiveEdgeCount()) {
        throw std::logic_error("resident trace 的数组或活动边计数不一致");
      }
      const std::uint64_t snapshot_hash = audited.ContentHash();
      std::vector<Candidate> candidates;
      candidates.reserve(trace.edge_ids.size());
      for (std::size_t index = 0U; index < trace.edge_ids.size(); ++index) {
        candidates.push_back({trace.edge_ids[index], trace.first_witness[index], trace.method,
                              trace.second_witness[index]});
      }
      std::vector<std::uint8_t> valid(candidates.size(), 0U);
      std::vector<std::string> reasons(candidates.size());
      std::optional<detail::QuickHsVerificationData> quick_hs_data;
      std::optional<GeometryVerificationData> geometry_data;
      std::optional<LpBoxProof> lp_proof;
      std::optional<LpBoxVerificationData> lp_data;
      std::uint32_t lp_certificate_index = kNoEliminationCertificate;
      if (trace.method == EliminationMethod::kGpuQuickHs) {
        quick_hs_data.emplace(detail::BuildQuickHsVerificationData(audited));
      } else if (trace.method == EliminationMethod::kGeometryMain) {
        geometry_data.emplace(BuildGeometryVerificationData(audited));
      } else if (trace.method == EliminationMethod::kLpBox) {
        if (trace.vertex_dual_numerator.size() != static_cast<std::size_t>(audited.dimension)) {
          throw std::runtime_error("resident LP trace 的 dual 维度非法");
        }
        lp_proof.emplace(LpBoxProof{snapshot_hash, trace.fractional_bits, trace.incumbent_cost,
                                    trace.vertex_dual_numerator});
        lp_data.emplace(BuildLpBoxVerificationData(audited, *lp_proof));
        if (!lp_data->certified) {
          throw std::runtime_error("resident LP 共享证明未通过 CPU audit: " + lp_data->reason);
        }
        if (report.certificate.lp_box_proofs.size() >=
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
          throw std::overflow_error("resident LP sidecar 索引溢出");
        }
        lp_certificate_index = static_cast<std::uint32_t>(report.certificate.lp_box_proofs.size());
      }
#ifdef CUDAEE_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
      for (std::int64_t index = 0; index < static_cast<std::int64_t>(candidates.size()); ++index) {
        const std::size_t offset = static_cast<std::size_t>(index);
        bool accepted = false;
        if (trace.method == EliminationMethod::kJv) {
          accepted = VerifyJvCandidate(audited, candidates[offset], &reasons[offset]);
        } else if (trace.method == EliminationMethod::kGpuQuickHs) {
          accepted = detail::VerifyQuickHsCandidate(audited, *quick_hs_data, candidates[offset],
                                                    &reasons[offset]);
        } else if (trace.method == EliminationMethod::kGeometryMain) {
          accepted = VerifyGeometryCandidate(audited, *geometry_data, candidates[offset],
                                             &reasons[offset]);
        } else if (trace.method == EliminationMethod::kLpBox) {
          accepted = detail::VerifyLpBoxCandidateForSnapshot(
              audited, *lp_proof, *lp_data, candidates[offset], snapshot_hash, &reasons[offset]);
        } else {
          reasons[offset] = "resident trace 方法不受支持";
        }
        valid[offset] = static_cast<std::uint8_t>(accepted);
      }
      for (std::size_t index = 0U; index < valid.size(); ++index) {
        if (valid[index] == 0U) {
          throw std::runtime_error(ToString(trace.method) +
                                   " GPU 命中未通过 CPU audit: " + reasons[index]);
        }
      }
      const std::vector<Candidate> committed =
          detail::CommitVerifiedCandidates(&audited, candidates, snapshot_hash);
      if (committed.size() != candidates.size()) {
        throw std::runtime_error("resident GPU 与 CPU 最小度提交结果不一致");
      }
      if (lp_proof.has_value()) {
        report.certificate.lp_box_proofs.push_back(std::move(*lp_proof));
      }
      for (const Candidate& candidate : committed) {
        const Edge& edge = audited.edges[static_cast<std::size_t>(candidate.edge_id)];
        report.certificate.proof.push_back({proof_epoch, snapshot_hash, candidate.edge_id, edge.u,
                                            edge.v, candidate.witness, candidate.method,
                                            candidate.method == EliminationMethod::kLpBox
                                                ? lp_certificate_index
                                                : kNoEliminationCertificate,
                                            candidate.second_witness});
      }
      report.certificate.epochs.push_back({.epoch = proof_epoch,
                                           .edges_before = trace.edges_before,
                                           .proposed = candidates.size(),
                                           .verified = candidates.size(),
                                           .rejected = 0U,
                                           .committed = committed.size()});
      ++proof_epoch;
    }
    if (device.final_active.size() != audited.edges.size()) {
      throw std::runtime_error("resident GPU 最终 mask 长度与 stable edge 数不一致");
    }
    for (std::size_t edge = 0U; edge < audited.edges.size(); ++edge) {
      if (static_cast<std::uint8_t>(audited.edges[edge].active) != device.final_active[edge]) {
        throw std::runtime_error("resident GPU 最终 mask 与 CPU audit 边集不一致");
      }
    }
    report.certificate.final_hash = audited.ContentHash();
    report.final_hash = report.certificate.final_hash;
    report.cpu_audit_ms = ElapsedMilliseconds(audit_begin);
  } else {
    if (!device.epochs.empty()) {
      throw std::logic_error("resident raw 模式不应回传逐边 trace");
    }
    if (device.final_active.size() != audited.edges.size()) {
      throw std::runtime_error("resident GPU 最终 mask 长度与 stable edge 数不一致");
    }
    for (std::size_t edge = 0U; edge < audited.edges.size(); ++edge) {
      audited.edges[edge].active = device.final_active[edge] != 0U;
    }
    // raw 模式只将设备最终位图物化为输出图，不逐边构造或验证证明。
    audited.RebuildCsr();
    if (audited.ActiveEdgeCount() != device.final_edges) {
      throw std::logic_error("resident GPU 最终 mask 与活动边计数不一致");
    }
    report.final_hash = audited.ContentHash();
    report.certificate.final_hash = report.final_hash;
  }

  ProtectedTourCheck final_tour_check;
  ProtectedTourCheck* final_tour_ptr = nullptr;
  if (!tour.empty()) {
    final_tour_check = CheckProtectedTour(audited, tour);
    if ((input.tour_is_known_optimum && final_tour_check.missing_edges != 0U) ||
        (input.expected_tour_cost >= 0 && final_tour_check.cost != input.expected_tour_cost)) {
      throw std::runtime_error("resident 最终 tour 门禁失败");
    }
    final_tour_ptr = &final_tour_check;
  }

  const SteadyClock::time_point output_begin = SteadyClock::now();
  if (config.enable_cpu_audit) {
    WriteProof(outputs.certificate, report.certificate);
    std::error_code size_error;
    report.certificate_bytes = std::filesystem::file_size(outputs.certificate, size_error);
    if (size_error) {
      throw std::runtime_error("无法读取 resident certificate 大小");
    }
  }
  audited.WriteActiveEdges(outputs.edges);
  WriteFixedEdges(outputs.fixed, audited.dimension, DeriveDegreeTwoFixedEdges(audited));
  WriteEmptyNonpairs(outputs.nonpairs, audited.dimension);
  report.output_ms = ElapsedMilliseconds(output_begin);
  report.end_to_end_ms = ElapsedMilliseconds(total_begin);
  report.trusted_total_ms = config.enable_cpu_audit ? report.end_to_end_ms : 0.0;
  WriteResidentManifest(outputs.manifest, input, config, report, final_tour_ptr);
  return report;
}

FgpuResidentRunReport RunFgpuResidentLocal(const FgpuInput& input, const FgpuOutputPaths& outputs,
                                           const FgpuResidentConfig& config) {
  if (input.input_edges.empty()) {
    throw std::invalid_argument("resident-local 必须显式提供 --input-edges");
  }
  FgpuResidentConfig local_config = config;
  local_config.enable_geometry = false;
  local_config.enable_pdlp = false;
  return RunFgpuResidentElimination(input, outputs, local_config);
}

} // namespace cudaee
