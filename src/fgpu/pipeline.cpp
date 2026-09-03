#include "cuda_edge_elimination/fgpu.hpp"

#include "cuda_edge_elimination/cuda_device_affinity.hpp"
#include "cuda_edge_elimination/tour.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaee {
namespace {

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

void AppendProofStage(EliminationResult* const combined, const EliminationResult& stage) {
  if (combined == nullptr || combined->final_hash != stage.initial_hash) {
    throw std::logic_error("FGPU 子阶段 proof 哈希链断裂");
  }
  if (stage.proof.empty()) {
    if (stage.initial_hash != stage.final_hash || !stage.ht_proofs.empty() ||
        !stage.lp_box_proofs.empty()) {
      throw std::logic_error("FGPU 空 proof 子阶段修改了图或含孤立 sidecar");
    }
    return;
  }
  if (combined->proof.size() > std::numeric_limits<std::size_t>::max() - stage.proof.size() ||
      combined->ht_proofs.size() >
          std::numeric_limits<std::size_t>::max() - stage.ht_proofs.size() ||
      combined->lp_box_proofs.size() >
          std::numeric_limits<std::size_t>::max() - stage.lp_box_proofs.size() ||
      combined->ht_proofs.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("FGPU 聚合 proof 数量溢出");
  }
  const std::uint32_t epoch_base = combined->proof.empty() ? 0U : combined->proof.back().epoch + 1U;
  const std::uint32_t certificate_base = static_cast<std::uint32_t>(combined->ht_proofs.size());
  if (combined->lp_box_proofs.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("FGPU LP sidecar 索引溢出");
  }
  const std::uint32_t lp_certificate_base =
      static_cast<std::uint32_t>(combined->lp_box_proofs.size());
  for (const ProofRecord& source : stage.proof) {
    if (source.epoch > std::numeric_limits<std::uint32_t>::max() - epoch_base) {
      throw std::overflow_error("FGPU proof epoch 溢出");
    }
    ProofRecord record = source;
    record.epoch += epoch_base;
    if (record.method == EliminationMethod::kHamiltonTutte) {
      if (record.certificate_index >= stage.ht_proofs.size() ||
          record.certificate_index > std::numeric_limits<std::uint32_t>::max() - certificate_base) {
        throw std::logic_error("FGPU HT sidecar 索引非法");
      }
      record.certificate_index += certificate_base;
    } else if (record.method == EliminationMethod::kLpBox) {
      if (record.certificate_index >= stage.lp_box_proofs.size() ||
          record.certificate_index >
              std::numeric_limits<std::uint32_t>::max() - lp_certificate_base) {
        throw std::logic_error("FGPU LP sidecar 索引非法");
      }
      record.certificate_index += lp_certificate_base;
    } else if (record.certificate_index != kNoEliminationCertificate) {
      throw std::logic_error("FGPU 非 HT record 引用了 sidecar");
    }
    combined->proof.push_back(record);
  }
  combined->ht_proofs.insert(combined->ht_proofs.end(), stage.ht_proofs.begin(),
                             stage.ht_proofs.end());
  combined->lp_box_proofs.insert(combined->lp_box_proofs.end(), stage.lp_box_proofs.begin(),
                                 stage.lp_box_proofs.end());
  for (const EpochMetrics& source : stage.epochs) {
    if (source.committed == 0U) {
      continue;
    }
    EpochMetrics metrics = source;
    metrics.epoch += epoch_base;
    combined->epochs.push_back(metrics);
  }
  combined->final_hash = stage.final_hash;
}

HtWavefrontOptions ArticleStrengthWavefrontOptions(const FgpuConfig& config) {
  HtWavefrontOptions options;
  HtRecursiveOptions& search = options.search_options;
  HtShallowOptions& root = search.root_options;
  // 与 KH -Jq 的候选顺序和穷举门禁一致；算术密集部分交给现有 CUDA 后端。
  root.max_neighborhood = 10U;
  root.max_cd_candidates = 10U;
  root.max_candidate_degree = 0U;
  root.max_reply_combinations = 0U;
  root.cd_mode = HtCdMode::kActiveIncompatible;
  root.candidate_backend = PathCompatibilityBackend::kCuda;
  root.neighborhood_mode = HtNeighborhoodMode::kQuickEndpoint;
  root.cd_order = HtCdOrder::kInput;
  root.max_cd_pair_trials = 10U;
  root.leaf_options.max_k = std::min<std::uint32_t>(5U, config.max_paths);
  root.leaf_options.max_deletion_sets = 0U;
  root.leaf_options.cost_backend = PathCompatibilityBackend::kCuda;
  root.leaf_options.exact_fallback_max_blocks = 0U;
  root.leaf_options.exact_backend = PathCompatibilityBackend::kCuda;
  search.max_depth = 0U;
  search.max_point_candidates = 0U;
  search.max_end_candidates = 0U;
  search.enable_point_moves = false;
  search.enable_end_moves = false;
  options.reply_frontier_batch_states = 256U;
  options.leaf_frontier_batch_states = 256U;
  options.fuse_leaf_buckets = false;
  options.propagation_backend = PathCompatibilityBackend::kCuda;
  options.path_append_backend = PathCompatibilityBackend::kCuda;
  options.hamilton_reply_backend = PathCompatibilityBackend::kCuda;
  options.scheduler = HtScheduler::kWavefront;
  options.speculation_width = 4U;
  return options;
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
    throw std::runtime_error("无法创建 fixed 输出: " + path.string());
  }
  output << dimension << ' ' << fixed.size() << '\n';
  for (const Edge& edge : fixed) {
    output << edge.u << ' ' << edge.v << ' ' << edge.weight << '\n';
  }
}

void WriteEmptyNonpairs(const std::filesystem::path& path, const std::int32_t dimension) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 non-pair 输出: " + path.string());
  }
  output << dimension << " 0\n";
  for (std::int32_t node = 0; node < dimension; ++node) {
    output << node << " 0\n";
  }
}

void WriteManifest(const std::filesystem::path& path, const FgpuInput& input,
                   const FgpuConfig& config, const FgpuRunReport& report,
                   const ProtectedTourCheck* const tour_check) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("无法创建 FGPU manifest: " + path.string());
  }
  output << "FGPU_ELIM_MANIFEST_V1\n";
  output << "instance " << input.instance.string() << '\n';
  output << "device " << config.device << '\n';
  output << "numeric " << ToString(config.numeric_mode) << '\n';
  output << "verification " << ToString(config.verification_mode) << '\n';
  output << "pdlp " << ToString(config.pdlp_backend) << '\n';
  output << "pdlp_backend " << report.pdlp.backend << '\n';
  output << "pdlp_device " << report.pdlp.selected_device << '\n';
  output << "pdlp_iterations " << report.pdlp.iterations << '\n';
  output << "pdlp_solve_ms " << report.pdlp.solve_ms << '\n';
  output << "pdlp_epochs " << report.pdlp_epochs << '\n';
  output << "pdlp_total_solve_ms " << report.pdlp_total_solve_ms << '\n';
  output << "pdlp_cpu_certified " << (report.pdlp.cpu_certified ? 1 : 0) << '\n';
  output << "ht_target_workers " << config.ht_target_workers << '\n';
  if (report.pdlp.exact_bound.certified) {
    output << "pdlp_bound_numerator " << report.pdlp.exact_bound.numerator << '\n';
    output << "pdlp_bound_denominator " << report.pdlp.exact_bound.denominator << '\n';
  }
  output << "initial_hash " << HexHash(report.initial_hash) << '\n';
  output << "final_hash " << HexHash(report.final_hash) << '\n';
  output << "initial_edges " << report.initial_edges << '\n';
  output << "final_edges " << report.final_edges << '\n';
  output << "geometry_committed " << report.geometry_committed << '\n';
  output << "lp_committed " << report.lp_committed << '\n';
  output << "jv_committed " << report.jv_committed << '\n';
  output << "ht_committed " << report.ht_committed << '\n';
  output << "fixed_count " << report.fixed_count << '\n';
  output << "nonpair_count " << report.nonpair_count << '\n';
  output << "termination " << report.termination << '\n';
  output << "certificate_records " << report.certificate.proof.size() << '\n';
  output << "certificate_bytes " << report.certificate_bytes << '\n';
  output << "geometry_backend " << report.geometry.backend << '\n';
  output << "geometry_device " << report.geometry.selected_device << '\n';
  output << "geometry_kernel_ms " << report.geometry.kernel_ms << '\n';
  output << "geometry_verify_ms " << report.geometry.verify_ms << '\n';
  output << "total_ms " << report.total_ms << '\n';
  output << "known_tour_checked " << (tour_check != nullptr ? 1 : 0) << '\n';
  if (tour_check != nullptr) {
    output << "known_tour_cost " << tour_check->cost << '\n';
    output << "known_tour_missing_edges " << tour_check->missing_edges << '\n';
    output << "known_tour_hash " << HexHash(tour_check->tour_hash) << '\n';
  }
  output << "END\n";
  if (!output) {
    throw std::runtime_error("写 FGPU manifest 失败: " + path.string());
  }
}

void CompareOutputGraph(const GraphSnapshot& replayed, const GraphSnapshot& output) {
  if (replayed.dimension != output.dimension || replayed.ActiveEdgeCount() != output.edges.size()) {
    throw std::runtime_error("证书重放图与 .edg 输出规模不一致");
  }
  for (const Edge& edge : output.edges) {
    if (!replayed.HasActiveEdge(edge.u, edge.v) ||
        replayed.Distance(edge.u, edge.v) != edge.weight) {
      throw std::runtime_error("证书重放图与 .edg 输出边集不一致");
    }
  }
}

void CompareFixedOutput(const GraphSnapshot& graph, const std::filesystem::path& instance,
                        const std::filesystem::path& path) {
  const std::vector<Edge> expected = DeriveDegreeTwoFixedEdges(graph);
  const GraphSnapshot output = GraphSnapshot::Load(instance, path);
  if (output.edges.size() != expected.size()) {
    throw std::runtime_error("证书推导的 fixed 边数与 .fix 输出不一致");
  }
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    const Edge& lhs = expected[index];
    const Edge& rhs = output.edges[index];
    if (lhs.u != rhs.u || lhs.v != rhs.v || lhs.weight != rhs.weight) {
      throw std::runtime_error("证书推导的 fixed 边集与 .fix 输出不一致");
    }
  }
}

void VerifyEmptyNonpairOutput(const GraphSnapshot& graph, const std::filesystem::path& path) {
  std::ifstream input(path);
  std::int32_t dimension = 0;
  std::uint64_t total_pairs = 0U;
  if (!input || !(input >> dimension >> total_pairs) || dimension != graph.dimension ||
      total_pairs != 0U) {
    throw std::runtime_error("当前证书未产生 non-pair，但 .nonpairs 头部并非规范空集合");
  }
  for (std::int32_t expected_node = 0; expected_node < graph.dimension; ++expected_node) {
    std::int32_t node = -1;
    std::uint64_t pair_count = 0U;
    if (!(input >> node >> pair_count) || node != expected_node || pair_count != 0U) {
      throw std::runtime_error(".nonpairs 不是规范的逐节点空集合");
    }
  }
  std::string trailing;
  if (input >> trailing) {
    throw std::runtime_error(".nonpairs 在规范空集合后含多余字段");
  }
}

std::filesystem::path ComparablePath(const std::filesystem::path& path) {
  return path.empty() ? std::filesystem::path{}
                      : std::filesystem::weakly_canonical(std::filesystem::absolute(path));
}

void ValidateDistinctOutputs(const FgpuInput& input, const FgpuOutputPaths& outputs) {
  const std::vector<std::filesystem::path> output_paths = {
      ComparablePath(outputs.edges),    ComparablePath(outputs.fixed),
      ComparablePath(outputs.nonpairs), ComparablePath(outputs.certificate),
      ComparablePath(outputs.manifest),
  };
  for (std::size_t first = 0U; first < output_paths.size(); ++first) {
    for (std::size_t second = first + 1U; second < output_paths.size(); ++second) {
      if (!output_paths[first].empty() && output_paths[first] == output_paths[second]) {
        throw std::invalid_argument("FGPU 五个输出路径必须互不相同");
      }
    }
  }

  const std::vector<std::filesystem::path> input_paths = {ComparablePath(input.instance),
                                                          ComparablePath(input.input_edges),
                                                          ComparablePath(input.tour)};
  for (const std::filesystem::path& output : output_paths) {
    for (const std::filesystem::path& source : input_paths) {
      if (!output.empty() && !source.empty() && output == source) {
        throw std::invalid_argument("FGPU 输出路径不得覆盖 instance、input-edges 或 tour");
      }
    }
  }
}

} // namespace

FgpuRunReport RunFgpuElimination(const FgpuInput& input, const FgpuOutputPaths& outputs,
                                 const FgpuConfig& config) {
  const auto total_begin = std::chrono::steady_clock::now();
  if (input.instance.empty() || outputs.edges.empty() || outputs.fixed.empty() ||
      outputs.nonpairs.empty() || outputs.certificate.empty() || outputs.manifest.empty()) {
    throw std::invalid_argument("FGPU 输入实例和五个输出路径均不能为空");
  }
  ValidateDistinctOutputs(input, outputs);
  if (config.device < 0 || config.max_jv_rounds == 0U || config.max_ht_epochs == 0U ||
      config.max_pdlp_epochs == 0U || config.ht_targets_per_epoch == 0U ||
      config.ht_target_workers == 0U || config.ht_target_workers > 32U || config.max_paths < 3U ||
      config.max_paths > 6U || config.max_local_nodes < 2U || config.max_local_nodes > 32U) {
    throw std::invalid_argument("FGPU device 或搜索预算参数越界");
  }
  if (config.enable_hamilton_tutte && !config.enable_jv) {
    throw std::invalid_argument("当前统一 local fixed-point 要求启用 HT 时同时启用 JV");
  }
  std::string affinity_reason;
  if (!detail::SetCudaDevicePreferenceForCurrentThread(config.device, &affinity_reason)) {
    throw std::runtime_error("无法绑定单 GPU: " + affinity_reason);
  }

  GraphSnapshot graph = input.input_edges.empty()
                            ? GraphSnapshot::LoadComplete(input.instance)
                            : GraphSnapshot::Load(input.instance, input.input_edges);
  const GraphSnapshot initial_graph = graph;
  FgpuRunReport report;
  report.initial_hash = graph.ContentHash();
  report.initial_edges = graph.ActiveEdgeCount();
  report.certificate.backend = "fgpu-single-gpu-cpu-certified";
  report.certificate.initial_hash = report.initial_hash;
  report.certificate.final_hash = report.initial_hash;

  std::vector<std::int32_t> protected_tour;
  std::int64_t incumbent_cost = -1;
  if (!input.tour.empty()) {
    protected_tour = ReadTsplibTour(input.tour, graph.dimension);
    const ProtectedTourCheck initial_check = CheckProtectedTour(graph, protected_tour);
    if ((input.tour_is_known_optimum && initial_check.missing_edges != 0U) ||
        (input.expected_tour_cost >= 0 && initial_check.cost != input.expected_tour_cost)) {
      throw std::runtime_error("输入边集未包含已知最优 tour，或 tour 成本与标签不符");
    }
    incumbent_cost = initial_check.cost;
  }

  if (config.enable_geometry && graph.distance_type == DistanceType::kEuc2D &&
      graph.integer_coordinates && graph.integer_distance_safe) {
    GeometryOptions geometry_options;
    geometry_options.backend = Backend::kCuda;
    geometry_options.numeric_mode = config.numeric_mode;
    geometry_options.potential_candidates = config.potential_candidates;
    geometry_options.witnesses_per_edge = config.geometry_witnesses_per_edge;
    geometry_options.device = config.device;
    GeometryEliminationResult geometry = RunGeometryElimination(&graph, geometry_options);
    report.geometry = geometry.metrics;
    report.geometry_committed = geometry.metrics.committed;
    AppendProofStage(&report.certificate, geometry.elimination);
  } else if (config.enable_geometry) {
    // CEIL_2D 等实例仍可进入 LP/JV/HT；不把尚未证明适用的几何谓词强行使用。
    report.geometry.backend = "disabled-unsupported-metric";
  }

  const bool pdlp_enabled = config.pdlp_backend != PdlpBackend::kOff;
  const bool local_enabled = config.enable_jv || config.enable_hamilton_tutte;
  const std::uint32_t orchestration_limit = pdlp_enabled ? config.max_pdlp_epochs : 1U;
  bool orchestration_converged = false;
  bool pdlp_epoch_limit = false;
  bool ht_epoch_limit = false;
  for (std::uint32_t orchestration_epoch = 0U; orchestration_epoch < orchestration_limit;
       ++orchestration_epoch) {
    const std::size_t edges_before = graph.ActiveEdgeCount();

    if (pdlp_enabled) {
      PdlpOptions pdlp_options;
      pdlp_options.backend = config.pdlp_backend;
      pdlp_options.iterations = config.pdlp_iterations;
      pdlp_options.device = config.device;
      pdlp_options.cuopt_library = config.cuopt_library;
      report.pdlp = RunFgpuPdlp(graph, pdlp_options);
      ++report.pdlp_epochs;
      report.pdlp_total_solve_ms += report.pdlp.solve_ms;
      if (incumbent_cost >= 0) {
        EliminationResult lp_elimination = RunLpBoxElimination(&graph, report.pdlp, incumbent_cost);
        report.lp_committed += lp_elimination.proof.size();
        AppendProofStage(&report.certificate, lp_elimination);
      }
    }

    if (local_enabled) {
      if (config.enable_hamilton_tutte) {
        LocalEliminationOptions options;
        options.jv_backend = config.enable_jv ? Backend::kCuda : Backend::kCpu;
        options.jv_candidate_mode = JvCandidateMode::kExhaustive;
        options.max_jv_rounds = config.max_jv_rounds;
        options.max_ht_epochs = config.max_ht_epochs;
        options.ht_scan_options.max_targets = config.ht_targets_per_epoch;
        options.ht_scan_options.target_order = pdlp_enabled
                                                   ? HtTargetOrder::kExternalScoreDescending
                                                   : HtTargetOrder::kWeightDescending;
        if (pdlp_enabled) {
          options.ht_scan_options.target_scores = report.pdlp.edge_scores;
        }
        options.ht_scan_options.target_workers = config.ht_target_workers;
        options.ht_scan_options.target_devices = {config.device};
        options.ht_scan_options.wavefront_options = ArticleStrengthWavefrontOptions(config);
        LocalEliminationResult local = RunLocalElimination(&graph, options);
        report.termination = ToString(local.termination);
        ht_epoch_limit = local.termination == LocalEliminationTermination::kHtEpochLimit;
        for (const ProofRecord& record : local.elimination.proof) {
          if (record.method == EliminationMethod::kJv) {
            ++report.jv_committed;
          } else if (record.method == EliminationMethod::kHamiltonTutte) {
            ++report.ht_committed;
          }
        }
        AppendProofStage(&report.certificate, local.elimination);
      } else {
        EliminationResult jv = RunJvElimination(&graph, Backend::kCuda, config.max_jv_rounds,
                                                JvCandidateMode::kExhaustive);
        report.jv_committed += jv.proof.size();
        report.termination = jv.epochs.empty() || jv.epochs.back().committed != 0U
                                 ? "jv-round-limit"
                                 : "jv-fixed-point";
        AppendProofStage(&report.certificate, jv);
      }
    }

    const bool changed = graph.ActiveEdgeCount() != edges_before;
    // local 自身已经达到当前图上的固定点；只有 LP 真正可删边且有合法 U 时，
    // 才需要用新图重新求 multiplier 并再次交错 local rules。
    if (!changed || !pdlp_enabled || incumbent_cost < 0 || ht_epoch_limit) {
      orchestration_converged = !changed || !pdlp_enabled || incumbent_cost < 0;
      break;
    }
    if (orchestration_epoch + 1U == orchestration_limit) {
      pdlp_epoch_limit = true;
    }
  }

  if (pdlp_epoch_limit) {
    report.termination = "pdlp-epoch-limit";
  } else if (!local_enabled) {
    if (pdlp_enabled) {
      report.termination = orchestration_converged ? "pdlp-fixed-point" : "pdlp-partial";
    } else if (config.enable_geometry) {
      report.termination = "geometry-complete";
    } else {
      report.termination = "no-elimination-stage";
    }
  } else if (pdlp_enabled && orchestration_converged && !ht_epoch_limit) {
    report.termination =
        config.enable_hamilton_tutte ? "local-pdlp-fixed-point" : "jv-pdlp-fixed-point";
  }

  report.certificate.final_hash = graph.ContentHash();
  report.final_hash = report.certificate.final_hash;
  report.final_edges = graph.ActiveEdgeCount();
  const std::vector<Edge> fixed_edges = DeriveDegreeTwoFixedEdges(graph);
  report.fixed_count = fixed_edges.size();

  ProtectedTourCheck final_tour_check;
  ProtectedTourCheck* final_tour_ptr = nullptr;
  if (!protected_tour.empty()) {
    final_tour_check = CheckProtectedTour(graph, protected_tour);
    if ((input.tour_is_known_optimum && final_tour_check.missing_edges != 0U) ||
        (input.expected_tour_cost >= 0 && final_tour_check.cost != input.expected_tour_cost)) {
      throw std::runtime_error("FGPU 回归门禁发现受保护 tour 边被删除");
    }
    final_tour_ptr = &final_tour_check;
  }

  // 无论 online gate 模式如何，正式输出前从初始快照顺序重放整条证书。
  GraphSnapshot replay_graph = initial_graph;
  const EliminationResult replayed = ReplayProof(&replay_graph, report.certificate);
  if (replayed.final_hash != graph.ContentHash() ||
      replay_graph.ActiveEdgeCount() != graph.ActiveEdgeCount()) {
    throw std::runtime_error("FGPU 最终 CPU 证书重放与设备主链结果不一致");
  }

  graph.WriteActiveEdges(outputs.edges);
  WriteFixedEdges(outputs.fixed, graph.dimension, fixed_edges);
  WriteEmptyNonpairs(outputs.nonpairs, graph.dimension);
  WriteProof(outputs.certificate, report.certificate);
  std::error_code certificate_size_error;
  report.certificate_bytes =
      std::filesystem::file_size(outputs.certificate, certificate_size_error);
  if (certificate_size_error) {
    throw std::runtime_error("无法读取刚写出的 FGPU 证书大小");
  }
  report.total_ms = ElapsedMilliseconds(total_begin);
  WriteManifest(outputs.manifest, input, config, report, final_tour_ptr);
  return report;
}

FgpuRunReport VerifyFgpuCertificate(const FgpuInput& input, const FgpuOutputPaths& outputs) {
  if (input.instance.empty() || outputs.edges.empty() || outputs.certificate.empty()) {
    throw std::invalid_argument("FGPU verify 需要 instance、edges 和 certificate");
  }
  GraphSnapshot initial = input.input_edges.empty()
                              ? GraphSnapshot::LoadComplete(input.instance)
                              : GraphSnapshot::Load(input.instance, input.input_edges);
  FgpuRunReport report;
  report.initial_hash = initial.ContentHash();
  report.initial_edges = initial.ActiveEdgeCount();
  report.certificate = ReadProof(outputs.certificate);
  std::error_code certificate_size_error;
  report.certificate_bytes =
      std::filesystem::file_size(outputs.certificate, certificate_size_error);
  if (certificate_size_error) {
    throw std::runtime_error("无法读取 FGPU 证书大小");
  }
  const EliminationResult replayed = ReplayProof(&initial, report.certificate);
  if (replayed.final_hash != report.certificate.final_hash) {
    throw std::logic_error("FGPU verify 重放结果哈希不一致");
  }
  const GraphSnapshot output = GraphSnapshot::Load(input.instance, outputs.edges);
  CompareOutputGraph(initial, output);
  if (!outputs.fixed.empty()) {
    CompareFixedOutput(initial, input.instance, outputs.fixed);
  }
  if (!outputs.nonpairs.empty()) {
    VerifyEmptyNonpairOutput(initial, outputs.nonpairs);
  }
  if (!input.tour.empty()) {
    const std::vector<std::int32_t> tour = ReadTsplibTour(input.tour, initial.dimension);
    const ProtectedTourCheck check = CheckProtectedTour(initial, tour);
    if ((input.tour_is_known_optimum && check.missing_edges != 0U) ||
        (input.expected_tour_cost >= 0 && check.cost != input.expected_tour_cost)) {
      throw std::runtime_error("FGPU verify 的受保护 tour 门禁失败");
    }
  }
  report.final_hash = initial.ContentHash();
  report.final_edges = initial.ActiveEdgeCount();
  report.fixed_count = DeriveDegreeTwoFixedEdges(initial).size();
  report.geometry_committed = static_cast<std::size_t>(std::count_if(
      report.certificate.proof.begin(), report.certificate.proof.end(),
      [](const ProofRecord& record) { return record.method == EliminationMethod::kGeometryMain; }));
  report.jv_committed = static_cast<std::size_t>(std::count_if(
      report.certificate.proof.begin(), report.certificate.proof.end(),
      [](const ProofRecord& record) { return record.method == EliminationMethod::kJv; }));
  report.lp_committed = static_cast<std::size_t>(std::count_if(
      report.certificate.proof.begin(), report.certificate.proof.end(),
      [](const ProofRecord& record) { return record.method == EliminationMethod::kLpBox; }));
  report.ht_committed = static_cast<std::size_t>(
      std::count_if(report.certificate.proof.begin(), report.certificate.proof.end(),
                    [](const ProofRecord& record) {
                      return record.method == EliminationMethod::kHamiltonTutte;
                    }));
  report.termination = "verified";
  return report;
}

} // namespace cudaee
