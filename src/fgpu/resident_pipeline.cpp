#include "cuda_edge_elimination/build_identity.hpp"
#include "cuda_edge_elimination/fgpu.hpp"

#include "../cpu/elimination_commit.hpp"
#include "gpu_bootstrap.hpp"
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
#include <iomanip>
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

void HashUint64(std::uint64_t* const hash, const std::uint64_t value) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
    *hash ^= (value >> shift) & 0xffU;
    *hash *= kFnvPrime;
  }
}

std::uint64_t ComputeResidentStateHash(const GraphSnapshot& graph, const std::vector<Edge>& fixed,
                                       const std::vector<detail::ResidentNonpair>& nonpairs) {
  constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr std::uint64_t kStateDomain = 0x4657505553544154ULL;
  std::uint64_t hash = kFnvOffset;
  HashUint64(&hash, kStateDomain);
  HashUint64(&hash, graph.ContentHash());
  HashUint64(&hash, static_cast<std::uint64_t>(fixed.size()));
  for (const Edge& edge : fixed) {
    HashUint64(&hash, static_cast<std::uint32_t>(edge.u));
    HashUint64(&hash, static_cast<std::uint32_t>(edge.v));
    HashUint64(&hash, static_cast<std::uint64_t>(edge.weight));
  }
  HashUint64(&hash, static_cast<std::uint64_t>(nonpairs.size()));
  for (const detail::ResidentNonpair& pair : nonpairs) {
    HashUint64(&hash, static_cast<std::uint32_t>(pair.center));
    HashUint64(&hash, static_cast<std::uint32_t>(pair.first));
    HashUint64(&hash, static_cast<std::uint32_t>(pair.second));
  }
  return hash;
}

std::vector<std::uint8_t> BuildProtectedEdgeMask(const GraphSnapshot& graph,
                                                 const std::vector<std::int32_t>& tour) {
  std::vector<std::uint8_t> result(graph.edges.size(), 0U);
  if (tour.empty()) {
    return result;
  }
  for (std::size_t index = 0U; index < tour.size(); ++index) {
    std::int32_t u = tour[index];
    std::int32_t v = tour[(index + 1U) % tour.size()];
    if (u > v) {
      std::swap(u, v);
    }
    const auto found =
        std::lower_bound(graph.edges.begin(), graph.edges.end(), std::pair{u, v},
                         [](const Edge& edge, const std::pair<std::int32_t, std::int32_t>& key) {
                           return std::tie(edge.u, edge.v) < std::tie(key.first, key.second);
                         });
    if (found == graph.edges.end() || found->u != u || found->v != v || !found->active) {
      throw std::runtime_error("resident 输入图缺少受保护 tour 边");
    }
    result[static_cast<std::size_t>(found - graph.edges.begin())] = 1U;
  }
  return result;
}

std::vector<Edge> DeriveOutputFixedEdges(const GraphSnapshot& graph,
                                         const std::vector<std::uint8_t>& explicit_fixed) {
  if (!explicit_fixed.empty() && explicit_fixed.size() != graph.edges.size()) {
    throw std::logic_error("GPU explicit fixed bitmap 长度与 stable edge 数不一致");
  }
  std::vector<Edge> fixed;
  fixed.reserve(graph.edges.size());
  for (std::size_t edge_id = 0U; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    if (edge.active && ((!explicit_fixed.empty() && explicit_fixed[edge_id] != 0U) ||
                        graph.Degree(edge.u) == 2 || graph.Degree(edge.v) == 2)) {
      fixed.push_back(edge);
    }
  }
  return fixed;
}

std::size_t CountNeighborPairs(const GraphSnapshot& graph) {
  std::size_t result = 0U;
  for (std::int32_t node = 0; node < graph.dimension; ++node) {
    const std::size_t degree = static_cast<std::size_t>(graph.Degree(node));
    const std::size_t pairs = degree * (degree - (degree != 0U ? 1U : 0U)) / 2U;
    if (pairs > std::numeric_limits<std::size_t>::max() - result) {
      throw std::overflow_error("最终邻边 pair 计数溢出");
    }
    result += pairs;
  }
  return result;
}

void CheckTourNonpairs(const std::vector<std::int32_t>& tour,
                       const std::vector<detail::ResidentNonpair>& nonpairs) {
  if (tour.empty()) {
    return;
  }
  std::vector<std::int32_t> previous(tour.size(), -1);
  std::vector<std::int32_t> next(tour.size(), -1);
  for (std::size_t index = 0U; index < tour.size(); ++index) {
    const std::int32_t center = tour[index];
    previous[static_cast<std::size_t>(center)] = tour[(index + tour.size() - 1U) % tour.size()];
    next[static_cast<std::size_t>(center)] = tour[(index + 1U) % tour.size()];
  }
  for (const detail::ResidentNonpair& nonpair : nonpairs) {
    const std::int32_t first = previous[static_cast<std::size_t>(nonpair.center)];
    const std::int32_t second = next[static_cast<std::size_t>(nonpair.center)];
    if ((nonpair.first == first && nonpair.second == second) ||
        (nonpair.first == second && nonpair.second == first)) {
      throw std::runtime_error("resident non-pair 与提供的 incumbent tour 冲突: center=" +
                               std::to_string(nonpair.center));
    }
  }
}

void CheckKnownOptimalTourFixedEdges(const std::vector<std::int32_t>& tour,
                                     const std::vector<Edge>& fixed) {
  if (tour.empty()) {
    return;
  }
  std::vector<std::int32_t> previous(tour.size(), -1);
  std::vector<std::int32_t> next(tour.size(), -1);
  for (std::size_t index = 0U; index < tour.size(); ++index) {
    const std::int32_t center = tour[index];
    previous[static_cast<std::size_t>(center)] = tour[(index + tour.size() - 1U) % tour.size()];
    next[static_cast<std::size_t>(center)] = tour[(index + 1U) % tour.size()];
  }
  for (const Edge& edge : fixed) {
    if (previous[static_cast<std::size_t>(edge.u)] != edge.v &&
        next[static_cast<std::size_t>(edge.u)] != edge.v) {
      throw std::runtime_error("resident fixed edge 不属于已知最优 tour: " +
                               std::to_string(edge.u) + "-" + std::to_string(edge.v));
    }
  }
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

void WriteNonpairs(const std::filesystem::path& path, const std::int32_t dimension,
                   const std::vector<detail::ResidentNonpair>& nonpairs) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 resident non-pair 输出: " + path.string());
  }
  output << dimension << ' ' << nonpairs.size() << '\n';
  std::size_t cursor = 0U;
  for (std::int32_t node = 0; node < dimension; ++node) {
    const std::size_t begin = cursor;
    while (cursor < nonpairs.size() && nonpairs[cursor].center == node) {
      if (nonpairs[cursor].first < 0 || nonpairs[cursor].second >= dimension ||
          nonpairs[cursor].first >= nonpairs[cursor].second || nonpairs[cursor].first == node ||
          nonpairs[cursor].second == node) {
        throw std::logic_error("GPU non-pair 三元组非法: center=" + std::to_string(node) +
                               " first=" + std::to_string(nonpairs[cursor].first) +
                               " second=" + std::to_string(nonpairs[cursor].second));
      }
      ++cursor;
    }
    output << node << ' ' << (cursor - begin) << '\n';
    for (std::size_t index = begin; index < cursor; ++index) {
      output << nonpairs[index].first << ' ' << nonpairs[index].second << '\n';
    }
  }
  if (cursor != nonpairs.size() || !output) {
    throw std::runtime_error("写 resident non-pair 输出失败");
  }
}

void WriteGpuReplayLog(const std::filesystem::path& path, const detail::ResidentGpuResult& device,
                       const std::uint64_t initial_hash, const std::uint64_t final_hash) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 GPU replay log: " + path.string());
  }
  output << "FGPU_GPU_REPLAY_LOG_V1\n";
  output << "initial_hash " << HexHash(initial_hash) << '\n';
  output << "final_hash " << HexHash(final_hash) << '\n';
  output << "epochs " << device.epochs.size() << '\n';
  for (std::size_t epoch_id = 0U; epoch_id < device.epochs.size(); ++epoch_id) {
    const detail::ResidentTraceEpoch& epoch = device.epochs[epoch_id];
    output << "epoch " << epoch_id << ' ' << ToString(epoch.method) << ' '
           << (epoch.main_edge_stage ? 1 : 0) << ' ' << (epoch.extra_edge_stage ? 1 : 0) << ' '
           << epoch.main_position << ' ' << epoch.edges_before << ' ' << epoch.edge_ids.size()
           << ' ' << epoch.fixed_edge_ids.size() << ' ' << epoch.fractional_bits << ' '
           << epoch.incumbent_cost << '\n';
    output << "dual " << epoch.vertex_dual_numerator.size();
    for (const std::int64_t value : epoch.vertex_dual_numerator) {
      output << ' ' << value;
    }
    output << '\n';
    output << "local_sec_dual " << epoch.local_sec_dual_numerator.size();
    for (const std::int64_t value : epoch.local_sec_dual_numerator) {
      output << ' ' << value;
    }
    output << '\n';
    for (std::size_t index = 0U; index < epoch.edge_ids.size(); ++index) {
      output << "proof " << epoch.edge_ids[index] << ' ' << epoch.first_witness[index] << ' '
             << epoch.second_witness[index] << '\n';
    }
    for (const std::int32_t edge_id : epoch.fixed_edge_ids) {
      output << "fixed_proof " << edge_id << '\n';
    }
    output << "end_epoch\n";
  }
  output << "nonpairs " << device.final_nonpairs.size() << '\n';
  for (const detail::ResidentNonpair& pair : device.final_nonpairs) {
    output << "nonpair " << pair.center << ' ' << pair.first << ' ' << pair.second << '\n';
  }
  output << "END\n";
  if (!output) {
    throw std::runtime_error("写 GPU replay log 失败");
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
  output << "enable_main_edge " << (config.enable_main_edge ? 1 : 0) << '\n';
  output << "enable_strong_metric " << (config.enable_strong_metric ? 1 : 0) << '\n';
  output << "enable_point_nonpair " << (config.enable_point_nonpair ? 1 : 0) << '\n';
  output << "enable_direct_fix " << (config.enable_direct_fix ? 1 : 0) << '\n';
  output << "enable_fixing " << (config.enable_fixing ? 1 : 0) << '\n';
  output << "enable_extra_edge " << (config.enable_extra_edge ? 1 : 0) << '\n';
  output << "extra_edge_depth " << config.extra_edge_depth << '\n';
  output << "protect_tour " << (config.protect_tour ? 1 : 0) << '\n';
  output << "cpu_audit_enabled " << (config.enable_cpu_audit ? 1 : 0) << '\n';
  output << "result_trust "
         << (report.cpu_audited ? "cpu-audited"
                                : (config.enable_gpu_replay ? "gpu-replayed" : "gpu-raw"))
         << '\n';
  output << "potential_candidates " << config.potential_candidates << '\n';
  output << "main_edge_potentials " << config.main_edge_potentials << '\n';
  output << "main_edge_positions " << config.main_edge_positions << '\n';
  output << "quick_hs_candidates " << config.quick_hs_candidates << '\n';
  output << "quick_hs_pair_trials " << config.quick_hs_pair_trials << '\n';
  output << "quick_hs_two_hop " << (config.quick_hs_two_hop ? 1 : 0) << '\n';
  output << "pdlp_iterations_budget " << config.pdlp_iterations << '\n';
  output << "max_pdlp_epochs " << config.max_pdlp_epochs << '\n';
  output << "max_hs_epochs " << config.max_hs_epochs << '\n';
  output << "max_jv_rounds " << config.max_jv_rounds << '\n';
  output << "initial_hash " << HexHash(report.initial_hash) << '\n';
  output << "final_hash " << HexHash(report.final_hash) << '\n';
  output << "final_state_hash " << HexHash(report.final_state_hash) << '\n';
  output << "initial_edges " << report.initial_edges << '\n';
  output << "final_edges " << report.final_edges << '\n';
  output << "main_edge_committed " << report.main_edge_committed << '\n';
  output << "main_edge_epochs " << report.main_edge_epochs << '\n';
  output << "main_edge_ms " << report.main_edge_ms << '\n';
  output << "jv_committed " << report.jv_committed << '\n';
  output << "quick_hs_committed " << report.quick_hs_committed << '\n';
  output << "extra_edge_committed " << report.extra_edge_committed << '\n';
  output << "extra_edge_epochs " << report.extra_edge_epochs << '\n';
  output << "geometry_committed " << report.geometry_committed << '\n';
  output << "lp_committed " << report.lp_committed << '\n';
  output << "fixed_count " << report.fixed_count << '\n';
  output << "pair_count " << report.pair_count << '\n';
  output << "nonpair_count " << report.nonpair_count << '\n';
  output << "lp_nonpair_committed " << report.lp_nonpair_committed << '\n';
  output << "fixed_anchor_nonpair_committed " << report.fixed_anchor_nonpair_committed << '\n';
  output << "point_nonpair_committed " << report.point_nonpair_committed << '\n';
  output << "nonpair_fix_committed " << report.nonpair_fix_committed << '\n';
  output << "direct_fix_committed " << report.direct_fix_committed << '\n';
  output << "nonpair_ratio "
         << (report.pair_count == 0U ? 0.0
                                     : static_cast<double>(report.nonpair_count) /
                                           static_cast<double>(report.pair_count))
         << '\n';
  output << "fixed_propagation_committed " << report.fixed_propagation_committed << '\n';
  output << "hs_epochs " << report.hs_epochs << '\n';
  output << "hs_full_sweeps " << report.hs_full_sweeps << '\n';
  output << "hs_active_sweeps " << report.hs_active_sweeps << '\n';
  output << "hs_full_tasks " << report.hs_full_tasks << '\n';
  output << "hs_active_tasks " << report.hs_active_tasks << '\n';
  output << "jv_rounds " << report.jv_rounds << '\n';
  output << "pdlp_epochs " << report.pdlp_epochs << '\n';
  output << "lp_connectivity_cuts " << report.lp_connectivity_cuts << '\n';
  output << "lp_path_closed_replies " << report.lp_path_closed_replies << '\n';
  output << "point_path_end_closed_replies " << report.point_path_end_closed_replies << '\n';
  output << "lp_degree_snapshots " << report.lp_degree_snapshots << '\n';
  output << "lp_strong_snapshots " << report.lp_strong_snapshots << '\n';
  output << "lp_lower_bound " << report.lp_lower_bound << '\n';
  output << "converged " << (report.converged ? 1 : 0) << '\n';
  output << "resident_bytes " << report.resident_bytes << '\n';
  output << "upload_ms " << report.upload_ms << '\n';
  output << "gpu_kernel_ms " << report.gpu_kernel_ms << '\n';
  output << "geometry_ms " << report.geometry_ms << '\n';
  output << "pdlp_ms " << report.pdlp_ms << '\n';
  output << "jv_ms " << report.jv_ms << '\n';
  output << "quick_hs_ms " << report.quick_hs_ms << '\n';
  output << "extra_edge_ms " << report.extra_edge_ms << '\n';
  output << "proof_replayed " << report.proof_replayed << '\n';
  output << "proof_rejected " << report.proof_rejected << '\n';
  output << "proof_replay_ms " << report.proof_replay_ms << '\n';
  output << "commit_ms " << report.commit_ms << '\n';
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

void WriteSolveManifest(const std::filesystem::path& path, const FgpuInput& input,
                        const FgpuSolveOptions& options, const FgpuSolveReport& report,
                        const FgpuResidentRunReport& resident) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 solve manifest: " + path.string());
  }
  output << "{\n  \"build_identity\": ";
  WriteBuildIdentityJson(output);
  output << ",\n  \"gpu_identity\": ";
  WriteGpuIdentityJson(output, report.selected_device);
  output << ",\n  \"instance_sha256\": " << JsonString(Sha256File(input.instance))
         << ",\n  \"input_edges_sha256\": "
         << (input.input_edges.empty() ? "null" : JsonString(Sha256File(input.input_edges)))
         << ",\n  \"tour_sha256\": "
         << (input.tour.empty() ? "null" : JsonString(Sha256File(input.tour))) << ",\n"
         << "  \"format\": \"FGPU_SOLVE_MANIFEST_V1\",\n"
         << "  \"instance\": " << JsonString(input.instance.string()) << ",\n"
         << "  \"input_edges\": " << JsonString(input.input_edges.string()) << ",\n"
         << "  \"mode\": " << JsonString(ToString(options.mode)) << ",\n"
         << "  \"profile\": " << JsonString(options.hybrid_e2e ? "hybrid-e2e" : "legacy") << ",\n"
         << "  \"distance_cache\": "
         << (options.hybrid_e2e && options.distance_cache ? "true" : "false") << ",\n"
         << "  \"distance_cache_bytes\": " << report.bootstrap.distance_bytes << ",\n"
         << "  \"main_pair_cache\": "
         << (options.hybrid_e2e && options.main_pair_cache ? "true" : "false") << ",\n"
         << "  \"main_pair_cache_bytes\": " << resident.main_pair_cache_bytes << ",\n"
         << "  \"leaf_permutation_cache\": "
         << (options.hybrid_e2e && options.leaf_permutation_cache ? "true" : "false") << ",\n"
         << "  \"point_near_first\": "
         << (options.hybrid_e2e && options.point_near_first ? "true" : "false") << ",\n"
         << "  \"point_adaptive_start\": "
         << (options.hybrid_e2e && options.point_adaptive_start ? "true" : "false") << ",\n"
         << "  \"point_prime_near\": "
         << (options.hybrid_e2e && options.point_prime_near ? "true" : "false") << ",\n"
         << "  \"permutation_cache_bytes\": " << report.bootstrap.permutation_bytes << ",\n"
         << "  \"permutation_build_replay_ms\": " << report.bootstrap.permutation_ms << ",\n"
         << "  \"full_degree_metric\": "
         << (options.hybrid_e2e && options.full_metric ? "true" : "false") << ",\n"
         << "  \"bootstrap_ms\": " << report.bootstrap.total_ms << ",\n"
         << "  \"distance_build_replay_ms\": " << report.bootstrap.distance_ms << ",\n"
         << "  \"incumbent_origin\": "
         << JsonString(options.hybrid_e2e
                           ? (options.enable_lp ? "gpu-nn-2opt-oropt" : "not-required-lp-off")
                           : "provided-tour")
         << ",\n"
         << "  \"incumbent_cost\": " << report.bootstrap.incumbent_cost << ",\n"
         << "  \"incumbent_starts\": " << report.bootstrap.starts << ",\n"
         << "  \"incumbent_improvements\": " << report.bootstrap.improvements << ",\n"
         << "  \"incumbent_ms\": " << report.bootstrap.incumbent_ms << ",\n"
         << "  \"input_optimum_labels\": "
         << (input.tour_is_known_optimum && !input.tour.empty() ? "true" : "false") << ",\n"
         << "  \"implemented_max_paths\": 3,\n"
         << "  \"implemented_extra_edge_depth\": 2,\n"
         << "  \"strong_metric\": true,\n"
         << "  \"point_nonpair\": true,\n"
         << "  \"point_path_end_branches\": 4,\n"
         << "  \"point_leaf_kernel\": " << JsonString(ToString(options.point_leaf_kernel)) << ",\n"
         << "  \"point_cta_blocks\": " << options.point_cta_blocks << ",\n"
         << "  \"point_registers\": " << report.lp.point_registers << ",\n"
         << "  \"point_active_blocks_per_sm\": " << report.lp.point_active_blocks_per_sm << ",\n"
         << "  \"point_local_bytes_per_thread\": " << report.lp.point_local_bytes_per_thread
         << ",\n"
         << "  \"termination\": " << JsonString(ToString(report.termination)) << ",\n"
         << "  \"gpu_replayed\": " << (report.gpu_replayed ? "true" : "false") << ",\n"
         << "  \"unaudited\": " << (report.unaudited ? "true" : "false") << ",\n"
         << "  \"selected_device\": " << report.selected_device << ",\n"
         << "  \"initial_hash\": " << JsonString(HexHash(report.initial_hash)) << ",\n"
         << "  \"final_hash\": " << JsonString(HexHash(report.final_hash)) << ",\n"
         << "  \"final_state_hash\": " << JsonString(HexHash(report.final_state_hash)) << ",\n"
         << "  \"initial_edges\": " << report.initial_edges << ",\n"
         << "  \"final_edges\": " << report.final_edges << ",\n"
         << "  \"fixed_edges\": " << report.fixed_edges << ",\n"
         << "  \"pairs\": " << report.pairs << ",\n"
         << "  \"nonpairs\": " << report.nonpairs << ",\n"
         << "  \"lp_nonpairs\": " << report.lp_nonpairs << ",\n"
         << "  \"fixed_anchor_nonpairs\": " << report.fixed_anchor_nonpairs << ",\n"
         << "  \"point_nonpairs\": " << report.point_nonpairs << ",\n"
         << "  \"nonpair_fixed_edges\": " << report.nonpair_fixed_edges << ",\n"
         << "  \"direct_fixed_edges\": " << report.direct_fixed_edges << ",\n"
         << "  \"nonpair_ratio\": "
         << (report.pairs == 0U
                 ? 0.0
                 : static_cast<double>(report.nonpairs) / static_cast<double>(report.pairs))
         << ",\n"
         << "  \"proof_replayed\": " << report.proof_replayed << ",\n"
         << "  \"proof_rejected\": " << report.proof_rejected << ",\n"
         << "  \"resident_bytes\": " << report.resident_bytes << ",\n"
         << "  \"geometry_deleted\": " << resident.geometry_committed << ",\n"
         << "  \"lp_deleted\": " << resident.lp_committed << ",\n"
         << "  \"lp_connectivity_cuts\": " << resident.lp_connectivity_cuts << ",\n"
         << "  \"lp_path_closed_replies\": " << resident.lp_path_closed_replies << ",\n"
         << "  \"point_path_end_closed_replies\": " << resident.point_path_end_closed_replies
         << ",\n"
         << "  \"lp_degree_snapshots\": " << resident.lp_degree_snapshots << ",\n"
         << "  \"lp_strong_snapshots\": " << resident.lp_strong_snapshots << ",\n"
         << "  \"lp_lower_bound\": " << resident.lp_lower_bound << ",\n"
         << "  \"fixed_propagation_deleted\": " << resident.fixed_propagation_committed << ",\n"
         << "  \"jv_deleted\": " << resident.jv_committed << ",\n"
         << "  \"quick_hs_deleted\": " << resident.quick_hs_committed << ",\n"
         << "  \"quick_hs_full_sweeps\": " << resident.hs_full_sweeps << ",\n"
         << "  \"quick_hs_active_sweeps\": " << resident.hs_active_sweeps << ",\n"
         << "  \"quick_hs_full_tasks\": " << resident.hs_full_tasks << ",\n"
         << "  \"quick_hs_active_tasks\": " << resident.hs_active_tasks << ",\n"
         << "  \"main_edge_deleted\": " << resident.main_edge_committed << ",\n"
         << "  \"ht_extra_edge_deleted\": " << resident.extra_edge_committed << ",\n"
         << "  \"upload_ms\": " << resident.upload_ms << ",\n"
         << "  \"geometry_ms\": " << resident.geometry_ms << ",\n"
         << "  \"lp_ms\": " << resident.pdlp_ms << ",\n"
         << "  \"lp_backend\": "
         << JsonString(!options.enable_lp
                           ? "off"
                           : (resident.lp.pdhg_iterations != 0U ? "primal-dual-sec" : "sec-dual"))
         << ",\n"
         << "  \"lp_solver_ms\": " << resident.lp.solver_ms << ",\n"
         << "  \"lp_cut_separation_ms\": " << resident.lp.cut_separation_ms << ",\n"
         << "  \"lp_point_ms\": " << resident.lp.point_ms << ",\n"
         << "  \"point_initial_pairs\": " << resident.lp.point_initial_pairs << ",\n"
         << "  \"point_initial_edge_frontier\": " << resident.lp.point_initial_edge_frontier
         << ",\n"
         << "  \"point_deferred_initially\": "
         << (resident.lp.point_deferred_initially ? "true" : "false") << ",\n"
         << "  \"point_service_sweeps\": " << resident.lp.point_service_sweeps << ",\n"
         << "  \"point_deferred_sweeps\": " << resident.lp.point_deferred_sweeps << ",\n"
         << "  \"point_prime_sweeps\": " << resident.lp.point_prime_sweeps << ",\n"
         << "  \"point_prime_proposals\": " << resident.lp.point_prime_proposals << ",\n"
         << "  \"point_prime_ms\": " << resident.lp.point_prime_ms << ",\n"
         << "  \"lp_fixing_ms\": " << resident.lp.fixing_ms << ",\n"
         << "  \"lp_pair_filter_ms\": " << resident.lp.pair_filter_ms << ",\n"
         << "  \"pdhg_model_ms\": " << resident.lp.pdhg_model_ms << ",\n"
         << "  \"pdhg_ms\": " << resident.lp.pdhg_ms << ",\n"
         << "  \"pdhg_primal_violation\": " << resident.lp.pdhg_primal_violation << ",\n"
         << "  \"pdhg_relative_gap\": " << resident.lp.pdhg_relative_gap << ",\n"
         << "  \"pdhg_iterations\": " << resident.lp.pdhg_iterations << ",\n"
         << "  \"pdhg_selected_snapshots\": " << resident.lp.pdhg_selected_snapshots << ",\n"
         << "  \"validated_transactions\": " << resident.lp.validated_transactions << ",\n"
         << "  \"jv_ms\": " << resident.jv_ms << ",\n"
         << "  \"quick_hs_ms\": " << resident.quick_hs_ms << ",\n"
         << "  \"main_edge_ms\": " << resident.main_edge_ms << ",\n"
         << "  \"ht_extra_edge_ms\": " << resident.extra_edge_ms << ",\n"
         << "  \"proof_replay_ms\": " << report.proof_replay_ms << ",\n"
         << "  \"commit_ms\": " << report.commit_ms << ",\n"
         << "  \"compaction_ms\": " << resident.compaction_ms << ",\n"
         << "  \"gpu_solve_wall_ms\": " << report.gpu_solve_wall_ms << ",\n"
         << "  \"end_to_end_ms\": " << report.end_to_end_ms << "\n"
         << "}\n";
  if (!output) {
    throw std::runtime_error("写 solve manifest 失败");
  }
}

} // namespace

FgpuResidentRunReport RunFgpuResidentElimination(const FgpuInput& input,
                                                 const FgpuOutputPaths& outputs,
                                                 const FgpuResidentConfig& config) {
  const SteadyClock::time_point total_begin = SteadyClock::now();
  if (input.instance.empty() || outputs.edges.empty() || outputs.fixed.empty() ||
      outputs.nonpairs.empty() || outputs.manifest.empty() ||
      ((config.enable_cpu_audit || config.serialize_gpu_certificate) &&
       outputs.certificate.empty())) {
    throw std::invalid_argument(
        "resident 需要 instance、边/fixed/nonpairs/manifest，序列化审计时还需要 certificate");
  }
  if (config.device < -1 || config.pdlp_iterations == 0U || config.potential_candidates < 2U ||
      config.potential_candidates > 32U || config.main_edge_potentials < 2U ||
      config.main_edge_potentials > 32U || config.main_edge_positions == 0U ||
      config.quick_hs_candidates < 2U || config.quick_hs_candidates > 32U ||
      config.extra_edge_depth < 1U || config.extra_edge_depth > 2U ||
      (config.serialize_gpu_certificate && !config.enable_gpu_replay) ||
      (config.enable_cpu_audit &&
       (config.enable_main_edge || config.enable_extra_edge || config.enable_point_nonpair ||
        config.enable_direct_fix || config.quick_hs_candidates != 10U ||
        config.quick_hs_pair_trials != 10U || config.quick_hs_two_hop)) ||
      (!config.enable_quick_hs && !config.enable_jv && !config.enable_geometry &&
       !config.enable_pdlp && !config.enable_main_edge && !config.enable_extra_edge)) {
    throw std::invalid_argument("resident device、预算或阶段开关非法");
  }
  if (config.enable_pdlp && input.tour.empty() && !config.hybrid_e2e) {
    throw std::invalid_argument("resident PDLP 需要 tour 提供合法 incumbent 上界");
  }
  if (config.hybrid_e2e &&
      (!input.input_edges.empty() || !input.tour.empty() || input.expected_tour_cost >= 0 ||
       config.protect_tour || config.enable_cpu_audit || !config.enable_gpu_replay)) {
    throw std::invalid_argument(
        "hybrid-e2e 只接收原始坐标，不接收 tour/最优成本/预处理边集/CPU audit");
  }
  GraphSnapshot initial =
      config.hybrid_e2e
          ? GraphSnapshot::LoadCoordinates(input.instance)
          : (input.input_edges.empty() ? GraphSnapshot::LoadComplete(input.instance)
                                       : GraphSnapshot::Load(input.instance, input.input_edges));
  std::unique_ptr<detail::GpuBootstrap> bootstrap;
  if (config.hybrid_e2e) {
    bootstrap = std::make_unique<detail::GpuBootstrap>(initial, config.device);
    bootstrap->BuildCompleteGraph(&initial);
    if (config.leaf_permutation_cache)
      bootstrap->BuildPermutationCatalog();
    if (config.enable_pdlp)
      bootstrap->GenerateIncumbent();
  }
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
  const std::vector<std::uint8_t> protected_edges =
      BuildProtectedEdgeMask(initial, config.protect_tour ? tour : std::vector<std::int32_t>{});

  detail::ResidentGpuOptions device_options;
  device_options.main_pair_cache = config.main_pair_cache;
  device_options.full_metric = config.full_metric;
  device_options.point_near_first = config.point_near_first;
  device_options.point_adaptive_start = config.point_adaptive_start;
  device_options.point_prime_near = config.point_prime_near;
  device_options.progress_log = config.hybrid_e2e;
  device_options.device = config.device;
  if (bootstrap != nullptr) {
    device_options.device = bootstrap->device();
    device_options.triangular_distance = config.distance_cache ? bootstrap->distances() : nullptr;
    device_options.permutation_orders = bootstrap->permutations();
    device_options.gpu_complete_graph = true;
  }
  device_options.max_hs_epochs = config.max_hs_epochs;
  device_options.max_jv_rounds = config.max_jv_rounds;
  device_options.enable_quick_hs = config.enable_quick_hs;
  device_options.enable_jv = config.enable_jv;
  device_options.enable_geometry = config.enable_geometry;
  device_options.enable_pdlp = config.enable_pdlp;
  device_options.enable_primal_dual_lp = config.enable_primal_dual_lp;
  device_options.point_leaf_kernel = config.point_leaf_kernel;
  device_options.point_cta_blocks = config.point_cta_blocks;
  device_options.enable_main_edge = config.enable_main_edge;
  device_options.enable_strong_metric = config.enable_strong_metric;
  device_options.enable_extra_edge = config.enable_extra_edge;
  device_options.extra_edge_depth = config.extra_edge_depth;
  device_options.quick_hs_candidates = config.quick_hs_candidates;
  device_options.quick_hs_pair_trials = config.quick_hs_pair_trials;
  device_options.quick_hs_two_hop = config.quick_hs_two_hop;
  device_options.gpu_replay = config.enable_gpu_replay;
  device_options.enable_fixing = config.enable_fixing;
  device_options.enable_point_nonpair = config.enable_point_nonpair;
  device_options.enable_direct_fix = config.enable_direct_fix;
  device_options.collect_trace = config.enable_cpu_audit || config.serialize_gpu_certificate;
  device_options.potential_candidates = config.potential_candidates;
  device_options.main_edge_potentials = config.main_edge_potentials;
  device_options.main_edge_positions = config.main_edge_positions;
  device_options.pdlp_iterations = config.pdlp_iterations;
  device_options.max_pdlp_epochs = config.max_pdlp_epochs;
  device_options.incumbent_cost = tour.empty() ? -1 : initial_tour_check.cost;
  if (bootstrap != nullptr)
    device_options.incumbent_cost = bootstrap->metrics().incumbent_cost;
  const detail::ResidentGpuResult device =
      detail::RunResidentEliminationCuda(initial, protected_edges, device_options);

  FgpuResidentRunReport report;
  if (bootstrap != nullptr)
    report.bootstrap = bootstrap->metrics();
  else
    report.bootstrap.incumbent_cost = device_options.incumbent_cost;
  report.initial_hash = initial.ContentHash();
  report.initial_edges = initial.ActiveEdgeCount();
  report.final_edges = device.final_edges;
  report.jv_committed = device.jv_committed;
  report.quick_hs_committed = device.quick_hs_committed;
  report.extra_edge_committed = device.extra_edge_committed;
  report.geometry_committed = device.geometry_committed;
  report.main_edge_committed = device.main_edge_committed;
  report.main_pair_cache_bytes = device.main_pair_cache_bytes;
  report.lp_committed = device.lp_committed;
  report.nonpair_count = device.final_nonpairs.size();
  report.lp_nonpair_committed = device.lp_nonpair_committed;
  report.fixed_anchor_nonpair_committed = device.fixed_anchor_nonpair_committed;
  report.point_nonpair_committed = device.point_nonpair_committed;
  report.nonpair_fix_committed = device.nonpair_fix_committed;
  report.direct_fix_committed = device.direct_fix_committed;
  report.fixed_propagation_committed = device.fixed_propagation_committed;
  report.hs_epochs = device.hs_epochs;
  report.hs_full_sweeps = device.hs_full_sweeps;
  report.hs_active_sweeps = device.hs_active_sweeps;
  report.hs_full_tasks = device.hs_full_tasks;
  report.hs_active_tasks = device.hs_active_tasks;
  report.extra_edge_epochs = device.extra_edge_epochs;
  report.jv_rounds = device.jv_rounds;
  report.pdlp_epochs = device.pdlp_epochs;
  report.lp_connectivity_cuts = device.lp_connectivity_cuts;
  report.lp_path_closed_replies = device.lp_path_closed_replies;
  report.point_path_end_closed_replies = device.point_path_end_closed_replies;
  report.lp_degree_snapshots = device.lp_degree_snapshots;
  report.lp = device.lp;
  report.lp_strong_snapshots = device.lp_strong_snapshots;
  report.lp_lower_bound = device.lp_lower_bound;
  report.main_edge_epochs = device.main_edge_epochs;
  report.converged = device.converged;
  report.cpu_audited = config.enable_cpu_audit;
  report.selected_device = device.selected_device;
  report.resident_bytes = device.resident_bytes;
  report.upload_ms = device.upload_ms;
  report.gpu_kernel_ms = device.kernel_ms;
  report.geometry_ms = device.geometry_ms;
  report.main_edge_ms = device.main_edge_ms;
  report.pdlp_ms = device.pdlp_ms;
  report.jv_ms = device.jv_ms;
  report.quick_hs_ms = device.quick_hs_ms;
  report.extra_edge_ms = device.extra_edge_ms;
  report.proof_replay_ms = device.proof_replay_ms;
  report.commit_ms = device.commit_ms;
  report.proof_replayed = device.proof_replayed;
  report.proof_rejected = device.proof_rejected;
  report.compaction_ms = device.compaction_ms;
  report.gpu_download_ms = device.download_ms;
  report.gpu_solve_wall_ms = device.solve_wall_ms;
  report.certificate.backend = config.enable_cpu_audit
                                   ? "cuda-fully-resident-cpu-audited"
                                   : (config.enable_gpu_replay ? "cuda-fully-resident-gpu-replayed"
                                                               : "cuda-fully-resident-gpu-raw");
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
    if (!config.serialize_gpu_certificate && !device.epochs.empty()) {
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
    // 普通启发式 incumbent 只是上界；其中边/邻边对可以被安全排除。
    if (input.tour_is_known_optimum)
      CheckTourNonpairs(tour, device.final_nonpairs);
    final_tour_ptr = &final_tour_check;
  }
  report.pair_count = CountNeighborPairs(audited);

  const SteadyClock::time_point output_begin = SteadyClock::now();
  const std::vector<Edge> fixed_edges = DeriveOutputFixedEdges(
      audited, config.enable_gpu_replay ? device.final_fixed : std::vector<std::uint8_t>{});
  if (input.tour_is_known_optimum) {
    // 这是已知标签上的最终否定门禁，不参与 GPU proof 授权。命中只会让
    // 整次运行失败，不能把未经证明的 fixed edge 改写成合法输出。
    CheckKnownOptimalTourFixedEdges(tour, fixed_edges);
  }
  report.fixed_count = fixed_edges.size();
  report.final_state_hash = ComputeResidentStateHash(audited, fixed_edges, device.final_nonpairs);
  if (config.enable_cpu_audit) {
    WriteProof(outputs.certificate, report.certificate);
    std::error_code size_error;
    report.certificate_bytes = std::filesystem::file_size(outputs.certificate, size_error);
    if (size_error) {
      throw std::runtime_error("无法读取 resident certificate 大小");
    }
  } else if (config.serialize_gpu_certificate) {
    WriteGpuReplayLog(outputs.certificate, device, report.initial_hash, report.final_hash);
    std::error_code size_error;
    report.certificate_bytes = std::filesystem::file_size(outputs.certificate, size_error);
    if (size_error) {
      throw std::runtime_error("无法读取 GPU replay log 大小");
    }
  }
  audited.WriteActiveEdges(outputs.edges);
  WriteFixedEdges(outputs.fixed, audited.dimension, fixed_edges);
  WriteNonpairs(outputs.nonpairs, audited.dimension, device.final_nonpairs);
  report.output_ms = ElapsedMilliseconds(output_begin);
  report.end_to_end_ms = ElapsedMilliseconds(total_begin);
  report.trusted_total_ms =
      (config.enable_cpu_audit || config.enable_gpu_replay) ? report.end_to_end_ms : 0.0;
  WriteResidentManifest(outputs.manifest, input, config, report, final_tour_ptr);
  return report;
}

FgpuSolveReport RunFgpuElimination(const FgpuInput& input, const FgpuOutputPaths& outputs,
                                   const FgpuSolveOptions& options) {
  if (input.tour.empty() && !options.hybrid_e2e && options.enable_lp) {
    throw std::invalid_argument("fgpu-elim solve 需要 --tour 提供合法 incumbent");
  }
  if (options.serialize_certificate && outputs.certificate.empty()) {
    throw std::invalid_argument("启用 GPU replay log 时必须提供 --certificate");
  }
  if (options.serialize_certificate && !options.enable_lp) {
    throw std::invalid_argument(
        "旧 GPU replay log 格式不支持独立 pair/fixing epoch；LP off 请不指定 certificate");
  }
  if (options.mode == FgpuSolveMode::kGpuFastRaw && options.serialize_certificate) {
    throw std::invalid_argument("gpu-fast-raw 没有 replay 授权，不能生成 replay log");
  }

  FgpuResidentConfig config;
  config.hybrid_e2e = options.hybrid_e2e;
  config.distance_cache = options.distance_cache;
  config.main_pair_cache = options.hybrid_e2e && options.main_pair_cache;
  config.full_metric = options.hybrid_e2e && options.full_metric;
  config.leaf_permutation_cache = options.hybrid_e2e && options.leaf_permutation_cache;
  config.point_near_first = options.hybrid_e2e && options.point_near_first;
  config.point_adaptive_start = options.hybrid_e2e && options.point_adaptive_start;
  config.point_prime_near = options.hybrid_e2e && options.point_prime_near;
  config.device = options.device;
  config.max_hs_epochs = 0U;
  config.max_jv_rounds = 0U;
  config.max_pdlp_epochs = 0U;
  config.enable_geometry = true;
  config.enable_pdlp = options.enable_lp;
  config.enable_primal_dual_lp = options.primal_dual_lp && !options.serialize_certificate;
  if (ToString(options.point_leaf_kernel) == "invalid") {
    throw std::invalid_argument("未知 Point leaf kernel");
  }
  config.point_leaf_kernel = options.point_leaf_kernel;
  if (options.point_cta_blocks != 2U && options.point_cta_blocks != 4U) {
    throw std::invalid_argument("Point CTA 驻留策略只能为 2 或 4");
  }
  config.point_cta_blocks = options.point_cta_blocks;
  config.enable_jv = true;
  config.enable_quick_hs = true;
  config.enable_main_edge = true;
  config.enable_strong_metric = true;
  // -e2 continuation 作为正式深层服务；其候选和 replay kernel 分离，
  // 所有第二层 endpoint reveal 仍基于同一不可变快照。
  config.enable_extra_edge = true;
  config.extra_edge_depth = 2U;
  config.potential_candidates = 32U;
  config.main_edge_potentials = 11U;
  config.main_edge_positions = 23U;
  config.quick_hs_candidates = 16U;
  config.quick_hs_pair_trials = 0U;
  config.quick_hs_two_hop = true;
  config.protect_tour = false;
  config.enable_cpu_audit = false;
  config.enable_gpu_replay = options.mode == FgpuSolveMode::kGpuSafe;
  config.serialize_gpu_certificate = options.serialize_certificate;
  config.enable_fixing = true;
  config.enable_point_nonpair = true;
  config.enable_direct_fix = true;

  const FgpuResidentRunReport resident = RunFgpuResidentElimination(input, outputs, config);
  if (!resident.converged) {
    throw std::runtime_error("solve 在无预算上限模式下未到达自然不动点");
  }

  FgpuSolveReport report;
  report.bootstrap = resident.bootstrap;
  report.termination = FgpuTermination::kFixedPoint;
  report.gpu_replayed = config.enable_gpu_replay;
  report.unaudited = !config.enable_gpu_replay;
  report.initial_hash = resident.initial_hash;
  report.final_hash = resident.final_hash;
  report.final_state_hash = resident.final_state_hash;
  report.initial_edges = resident.initial_edges;
  report.final_edges = resident.final_edges;
  report.fixed_edges = resident.fixed_count;
  report.pairs = resident.pair_count;
  report.nonpairs = resident.nonpair_count;
  report.lp_nonpairs = resident.lp_nonpair_committed;
  report.fixed_anchor_nonpairs = resident.fixed_anchor_nonpair_committed;
  report.point_nonpairs = resident.point_nonpair_committed;
  report.nonpair_fixed_edges = resident.nonpair_fix_committed;
  report.direct_fixed_edges = resident.direct_fix_committed;
  report.proof_replayed = resident.proof_replayed;
  report.proof_rejected = resident.proof_rejected;
  report.lp_connectivity_cuts = resident.lp_connectivity_cuts;
  report.lp_path_closed_replies = resident.lp_path_closed_replies;
  report.point_path_end_closed_replies = resident.point_path_end_closed_replies;
  report.lp_degree_snapshots = resident.lp_degree_snapshots;
  report.lp = resident.lp;
  report.lp_strong_snapshots = resident.lp_strong_snapshots;
  report.lp_lower_bound = resident.lp_lower_bound;
  report.selected_device = resident.selected_device;
  report.resident_bytes = resident.resident_bytes;
  report.proof_replay_ms = resident.proof_replay_ms;
  report.commit_ms = resident.commit_ms;
  report.gpu_solve_wall_ms = resident.gpu_solve_wall_ms;
  report.end_to_end_ms = resident.end_to_end_ms;
  WriteSolveManifest(outputs.manifest, input, options, report, resident);
  return report;
}

std::string ToString(const FgpuSolveMode mode) {
  switch (mode) {
  case FgpuSolveMode::kGpuSafe:
    return "gpu-safe";
  case FgpuSolveMode::kGpuFastRaw:
    return "gpu-fast-raw";
  }
  return "unknown";
}

std::string ToString(const FgpuTermination termination) {
  switch (termination) {
  case FgpuTermination::kFixedPoint:
    return "fixed-point";
  case FgpuTermination::kPartialOom:
    return "partial-oom";
  case FgpuTermination::kPartialDeviceError:
    return "partial-device-error";
  }
  return "unknown";
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
