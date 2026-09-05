#include "cuda_edge_elimination/fgpu.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace {

using Arguments = std::map<std::string, std::string>;

void PrintHelp() {
  std::cout << "fgpu-elim：单 GPU、可重放证明的 TSP 边消除闭环\n\n"
            << "用法：\n"
            << "  fgpu-elim solve --instance FILE [--input-edges FILE] --tour FILE\n"
            << "    [--mode gpu-safe|gpu-fast-raw] [--device auto|N]\n"
            << "    --output-edges FILE --fixed FILE --nonpairs FILE --manifest FILE\n"
            << "    [--certificate FILE] [--tour-role incumbent|known-optimum]\n"
            << "    [--expected-cost N]\n"
            << "    [--profile legacy|hybrid-e2e]（hybrid 仅接收坐标，GPU 自建上界）\n"
            << "    [--distance-cache 0|1]（hybrid 距离缓存消融）\n"
            << "    [--lp-backend off|primal-dual-sec|sec-dual]（LP 关闭不关闭 pair/fix）\n"
            << "    [--point-leaf-kernel permutation|prescreen-permutation|prescreen-subset-dp]\n"
            << "    [--point-cta-blocks 2|4]（寄存器/驻留策略，不限制任务数量）\n"
            << "  fgpu-elim run --instance FILE [--input-edges FILE] [--tour FILE]\n"
            << "    --device N --numeric mixed-safe|fp64|aggressive-fp32\n"
            << "    --verification epoch|deferred --pdlp off|native|cuopt-baseline\n"
            << "    --certificate FILE --output-edges FILE --fixed FILE --nonpairs FILE\n"
            << "    --manifest FILE [--expected-cost N] [--potential-candidates 2..32]\n"
            << "    [--geometry-witnesses 1..8] [--max-jv-rounds N]\n"
            << "    [--pdlp-iterations N] [--max-pdlp-epochs N] [--cuopt-library FILE]\n"
            << "    [--max-ht-epochs N] [--ht-targets-per-epoch N]\n"
            << "    [--ht-target-workers 1..32]（共享同一张 GPU）\n"
            << "    [--max-paths 3..6] [--max-local-nodes 2..32]\n"
            << "    [--enable-geometry 0|1] [--enable-jv 0|1] [--enable-ht 0|1]\n"
            << "  fgpu-elim resident-local --instance FILE --input-edges FILE [--tour FILE]\n"
            << "    --device N [--max-hs-epochs N] [--max-jv-rounds N]\n"
            << "    [--enable-quick-hs 0|1] [--enable-jv 0|1]\n"
            << "    [--cpu-audit 0|1] --output-edges FILE --fixed FILE --nonpairs FILE\n"
            << "    [--certificate FILE（cpu-audit=1 时必需）]\n"
            << "    --manifest FILE\n"
            << "  fgpu-elim resident --instance FILE [--input-edges FILE] --tour FILE\n"
            << "    --device N [--potential-candidates 2..32] [--pdlp-iterations N]\n"
            << "    [--max-pdlp-epochs N] [--max-hs-epochs N] [--max-jv-rounds N]\n"
            << "    [--enable-geometry 0|1] [--enable-pdlp 0|1]\n"
            << "    [--enable-quick-hs 0|1] [--enable-jv 0|1]，输出参数同上\n"
            << "  fgpu-elim resident-oneshot --instance FILE [--input-edges FILE] --tour FILE\n"
            << "    --device N [--potential-candidates 2..32]\n"
            << "    [--main-edge-potentials 2..32] [--main-edge-positions N]\n"
            << "    [--enable-strong-metric 0|1]（低度数 metric-excess）\n"
            << "    [--enable-point-nonpair 0|1]（完整一层 HT point replies）\n"
            << "    [--enable-direct-fix 0|1]（完整端点邻边对笛卡尔积 fixing）\n"
            << "    [--quick-hs-candidates 2..32] [--quick-hs-pair-trials N]\n"
            << "    [--quick-hs-two-hop 0|1]\n"
            << "    [--enable-extra-edge 0|1] [--extra-edge-depth 1|2]（KH -e1/-e2）\n"
            << "    [--protect-tour 0|1]，其余输出参数同 resident\n"
            << "    （固定运行到自然固定点，不接受 CPU audit 或非零 epoch 上限）\n"
            << "  fgpu-elim pdlp-inspect --instance FILE [--input-edges FILE]\n"
            << "    --pdlp native|cuopt-baseline|cuopt-subtour --expected-cost N [--device N]\n"
            << "    [--pdlp-iterations N] [--cuopt-library FILE]\n"
            << "    [--mincut-oracle FILE（仅 CPU 强度诊断）]\n"
            << "  fgpu-elim verify --instance FILE [--input-edges FILE] --certificate FILE\n"
            << "    --output-edges FILE [--fixed FILE --nonpairs FILE]\n"
            << "    [--tour FILE --tour-role incumbent|known-optimum --expected-cost N]\n\n"
            << "resident 的三个 max epoch/round 参数取 0 表示运行到自然固定点。\n"
            << "--gpus 0 可作为 --device 0 的兼容写法；多 GPU 列表会显式拒绝。\n";
}

Arguments ParseArguments(const int argc, char** argv, const int first) {
  Arguments result;
  for (int index = first; index < argc; ++index) {
    std::string key = argv[index];
    if (!key.starts_with("--") || index + 1 >= argc ||
        std::string_view(argv[index + 1]).starts_with("--")) {
      throw std::invalid_argument("参数必须采用 --name value: " + key);
    }
    key.erase(0U, 2U);
    if (!result.emplace(key, argv[++index]).second) {
      throw std::invalid_argument("参数重复: --" + key);
    }
  }
  return result;
}

const std::string& Required(const Arguments& arguments, const std::string& name) {
  const auto iterator = arguments.find(name);
  if (iterator == arguments.end() || iterator->second.empty()) {
    throw std::invalid_argument("缺少参数 --" + name);
  }
  return iterator->second;
}

std::string Optional(const Arguments& arguments, const std::string& name,
                     const std::string& fallback = {}) {
  const auto iterator = arguments.find(name);
  return iterator == arguments.end() ? fallback : iterator->second;
}

template <typename Integer>
Integer ParseInteger(const std::string& text, const std::string_view description) {
  static_assert(std::is_integral_v<Integer>);
  Integer value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument(std::string(description) + " 必须是范围内十进制整数");
  }
  return value;
}

template <typename Integer>
Integer OptionalInteger(const Arguments& arguments, const std::string& name,
                        const Integer fallback) {
  const auto iterator = arguments.find(name);
  return iterator == arguments.end() ? fallback
                                     : ParseInteger<Integer>(iterator->second, "--" + name);
}

bool IsWithin(const std::filesystem::path& child, const std::filesystem::path& parent) {
  auto child_iterator = child.begin();
  for (auto parent_iterator = parent.begin(); parent_iterator != parent.end();
       ++parent_iterator, ++child_iterator) {
    if (child_iterator == child.end() || *child_iterator != *parent_iterator) {
      return false;
    }
  }
  return true;
}

std::filesystem::path CheckedOutput(const std::string& raw) {
  const std::filesystem::path repository =
      std::filesystem::weakly_canonical(std::filesystem::path(CUDAEE_SOURCE_DIR));
  // weakly_canonical 会解析最长已存在前缀，也会解析“目标文件本身”的
  // 符号链接。只解析 parent 会允许 repo/out -> /tmp/out 这类文件级逃逸。
  const std::filesystem::path output =
      std::filesystem::weakly_canonical(std::filesystem::absolute(raw));
  if (!IsWithin(output, repository) || output == repository) {
    throw std::invalid_argument("输出路径必须位于仓库中: " + output.string());
  }
  std::filesystem::create_directories(output.parent_path());
  return output;
}

void ValidateKeys(const Arguments& arguments, const std::set<std::string>& allowed) {
  for (const auto& [name, unused_value] : arguments) {
    static_cast<void>(unused_value);
    if (!allowed.contains(name)) {
      throw std::invalid_argument("未知参数 --" + name);
    }
  }
}

int ParseDevice(const Arguments& arguments) {
  const bool has_device = arguments.contains("device");
  const bool has_gpus = arguments.contains("gpus");
  if (has_device && has_gpus) {
    throw std::invalid_argument("--device 与 --gpus 不能同时给出");
  }
  const std::string value =
      has_gpus ? Required(arguments, "gpus") : Optional(arguments, "device", "0");
  if (value.find(',') != std::string::npos) {
    throw std::invalid_argument("当前正式范围为单 GPU；--gpus 不接受多个 ordinal");
  }
  return ParseInteger<int>(value, has_gpus ? "--gpus" : "--device");
}

int ParseSolveDevice(const Arguments& arguments) {
  const bool has_device = arguments.contains("device");
  const bool has_gpus = arguments.contains("gpus");
  if (has_device && has_gpus) {
    throw std::invalid_argument("--device 与 --gpus 不能同时给出");
  }
  const std::string value =
      has_gpus ? Required(arguments, "gpus") : Optional(arguments, "device", "auto");
  if (value == "auto") {
    return -1;
  }
  if (value.find(',') != std::string::npos) {
    throw std::invalid_argument("当前正式范围为单 GPU；--gpus 不接受多个 ordinal");
  }
  const int device = ParseInteger<int>(value, has_gpus ? "--gpus" : "--device");
  if (device < 0) {
    throw std::invalid_argument("--device 必须是 auto 或非负 ordinal");
  }
  return device;
}

cudaee::FgpuSolveMode ParseSolveMode(const std::string& value) {
  if (value == "gpu-safe") {
    return cudaee::FgpuSolveMode::kGpuSafe;
  }
  if (value == "gpu-fast-raw") {
    return cudaee::FgpuSolveMode::kGpuFastRaw;
  }
  throw std::invalid_argument("--mode 必须是 gpu-safe 或 gpu-fast-raw");
}

cudaee::NumericMode ParseNumeric(const std::string& value) {
  if (value == "mixed-safe") {
    return cudaee::NumericMode::kMixedSafe;
  }
  if (value == "fp64") {
    return cudaee::NumericMode::kFp64;
  }
  if (value == "aggressive-fp32") {
    return cudaee::NumericMode::kAggressiveFp32;
  }
  throw std::invalid_argument("--numeric 必须是 mixed-safe、fp64 或 aggressive-fp32");
}

cudaee::VerificationMode ParseVerification(const std::string& value) {
  if (value == "epoch") {
    return cudaee::VerificationMode::kEpoch;
  }
  if (value == "deferred") {
    return cudaee::VerificationMode::kDeferred;
  }
  throw std::invalid_argument("--verification 必须是 epoch 或 deferred");
}

cudaee::PdlpBackend ParsePdlp(const std::string& value) {
  if (value == "off") {
    return cudaee::PdlpBackend::kOff;
  }
  if (value == "native") {
    return cudaee::PdlpBackend::kNative;
  }
  if (value == "cuopt-baseline") {
    return cudaee::PdlpBackend::kCuoptBaseline;
  }
  throw std::invalid_argument("--pdlp 必须是 off、native 或 cuopt-baseline");
}

bool OptionalBoolean(const Arguments& arguments, const std::string& name, const bool fallback) {
  const std::uint32_t value = OptionalInteger<std::uint32_t>(arguments, name, fallback ? 1U : 0U);
  if (value > 1U) {
    throw std::invalid_argument("--" + name + " 必须是 0 或 1");
  }
  return value != 0U;
}

cudaee::FgpuInput ParseInput(const Arguments& arguments) {
  cudaee::FgpuInput input;
  input.instance = Required(arguments, "instance");
  input.input_edges = Optional(arguments, "input-edges");
  input.tour = Optional(arguments, "tour");
  const std::string role = Optional(arguments, "tour-role", "known-optimum");
  if (role != "incumbent" && role != "known-optimum") {
    throw std::invalid_argument("--tour-role 必须是 incumbent 或 known-optimum");
  }
  input.tour_is_known_optimum = role == "known-optimum";
  input.expected_tour_cost = OptionalInteger<std::int64_t>(arguments, "expected-cost", -1);
  return input;
}

cudaee::FgpuOutputPaths ParseRunOutputs(const Arguments& arguments,
                                        const bool require_certificate = true) {
  cudaee::FgpuOutputPaths outputs;
  outputs.edges = CheckedOutput(Required(arguments, "output-edges"));
  outputs.fixed = CheckedOutput(Required(arguments, "fixed"));
  outputs.nonpairs = CheckedOutput(Required(arguments, "nonpairs"));
  if (require_certificate || arguments.contains("certificate")) {
    outputs.certificate = CheckedOutput(Required(arguments, "certificate"));
  }
  outputs.manifest = CheckedOutput(Required(arguments, "manifest"));
  return outputs;
}

void RunCommand(const Arguments& arguments) {
  ValidateKeys(arguments, {"instance",
                           "input-edges",
                           "tour",
                           "tour-role",
                           "expected-cost",
                           "device",
                           "gpus",
                           "numeric",
                           "verification",
                           "pdlp",
                           "potential-candidates",
                           "geometry-witnesses",
                           "max-jv-rounds",
                           "max-ht-epochs",
                           "ht-targets-per-epoch",
                           "ht-target-workers",
                           "max-paths",
                           "max-local-nodes",
                           "pdlp-iterations",
                           "max-pdlp-epochs",
                           "cuopt-library",
                           "enable-geometry",
                           "enable-jv",
                           "enable-ht",
                           "output-edges",
                           "fixed",
                           "nonpairs",
                           "certificate",
                           "manifest"});
  cudaee::FgpuConfig config;
  config.device = ParseDevice(arguments);
  config.numeric_mode = ParseNumeric(Optional(arguments, "numeric", "mixed-safe"));
  config.verification_mode = ParseVerification(Optional(arguments, "verification", "epoch"));
  config.pdlp_backend = ParsePdlp(Optional(arguments, "pdlp", "off"));
  config.potential_candidates =
      OptionalInteger<std::uint32_t>(arguments, "potential-candidates", 16U);
  config.geometry_witnesses_per_edge =
      OptionalInteger<std::uint32_t>(arguments, "geometry-witnesses", 4U);
  config.max_jv_rounds = OptionalInteger<std::uint32_t>(arguments, "max-jv-rounds", 100U);
  config.max_ht_epochs = OptionalInteger<std::uint32_t>(arguments, "max-ht-epochs", 100U);
  config.ht_targets_per_epoch =
      OptionalInteger<std::uint64_t>(arguments, "ht-targets-per-epoch", 64U);
  config.ht_target_workers = OptionalInteger<std::uint32_t>(arguments, "ht-target-workers", 4U);
  config.max_paths = OptionalInteger<std::uint32_t>(arguments, "max-paths", 6U);
  config.max_local_nodes = OptionalInteger<std::uint32_t>(arguments, "max-local-nodes", 32U);
  config.pdlp_iterations = OptionalInteger<std::uint32_t>(arguments, "pdlp-iterations", 2000U);
  config.max_pdlp_epochs = OptionalInteger<std::uint32_t>(arguments, "max-pdlp-epochs", 8U);
  config.cuopt_library = Optional(arguments, "cuopt-library");
  config.enable_geometry = OptionalBoolean(arguments, "enable-geometry", true);
  config.enable_jv = OptionalBoolean(arguments, "enable-jv", true);
  config.enable_hamilton_tutte = OptionalBoolean(arguments, "enable-ht", true);
  const cudaee::FgpuRunReport report =
      cudaee::RunFgpuElimination(ParseInput(arguments), ParseRunOutputs(arguments), config);
  std::cout << "status=OK initial_edges=" << report.initial_edges
            << " geometry_deleted=" << report.geometry_committed
            << " lp_deleted=" << report.lp_committed << " pdlp_epochs=" << report.pdlp_epochs
            << " jv_deleted=" << report.jv_committed << " ht_deleted=" << report.ht_committed
            << " final_edges=" << report.final_edges << " termination=" << report.termination
            << " final_hash=" << cudaee::HexHash(report.final_hash)
            << " geometry_backend=" << report.geometry.backend
            << " geometry_device=" << report.geometry.selected_device
            << " geometry_proposed=" << report.geometry.proposed
            << " geometry_rejected=" << report.geometry.rejected
            << " geometry_kernel_ms=" << std::fixed << std::setprecision(3)
            << report.geometry.kernel_ms << " geometry_verify_ms=" << report.geometry.verify_ms
            << " certificate_bytes=" << report.certificate_bytes << " total_ms=" << report.total_ms
            << '\n';
}

void ResidentLocalCommand(const Arguments& arguments) {
  ValidateKeys(arguments,
               {"instance", "input-edges", "tour", "tour-role", "expected-cost", "device", "gpus",
                "max-hs-epochs", "max-jv-rounds", "enable-quick-hs", "enable-jv", "output-edges",
                "fixed", "nonpairs", "certificate", "manifest", "cpu-audit"});
  cudaee::FgpuResidentConfig config;
  config.device = ParseDevice(arguments);
  config.max_hs_epochs = OptionalInteger<std::uint32_t>(arguments, "max-hs-epochs", 0U);
  config.max_jv_rounds = OptionalInteger<std::uint32_t>(arguments, "max-jv-rounds", 0U);
  config.enable_quick_hs = OptionalBoolean(arguments, "enable-quick-hs", true);
  config.enable_jv = OptionalBoolean(arguments, "enable-jv", true);
  config.enable_cpu_audit = OptionalBoolean(arguments, "cpu-audit", false);
  const cudaee::FgpuResidentRunReport report = cudaee::RunFgpuResidentLocal(
      ParseInput(arguments), ParseRunOutputs(arguments, config.enable_cpu_audit), config);
  std::cout << "status=OK mode=resident-local initial_edges=" << report.initial_edges
            << " jv_deleted=" << report.jv_committed
            << " quick_hs_deleted=" << report.quick_hs_committed
            << " final_edges=" << report.final_edges << " converged=" << (report.converged ? 1 : 0)
            << " cpu_audited=" << (report.cpu_audited ? 1 : 0)
            << " final_hash=" << cudaee::HexHash(report.final_hash)
            << " final_state_hash=" << cudaee::HexHash(report.final_state_hash)
            << " selected_device=" << report.selected_device
            << " resident_bytes=" << report.resident_bytes << " upload_ms=" << std::fixed
            << std::setprecision(3) << report.upload_ms << " gpu_kernel_ms=" << report.gpu_kernel_ms
            << " geometry_ms=" << report.geometry_ms << " pdlp_ms=" << report.pdlp_ms
            << " jv_ms=" << report.jv_ms << " quick_hs_ms=" << report.quick_hs_ms
            << " compaction_ms=" << report.compaction_ms
            << " gpu_download_ms=" << report.gpu_download_ms
            << " gpu_solve_wall_ms=" << report.gpu_solve_wall_ms
            << " cpu_audit_ms=" << report.cpu_audit_ms << " output_ms=" << report.output_ms
            << " end_to_end_ms=" << report.end_to_end_ms
            << " trusted_total_ms=" << report.trusted_total_ms
            << " certificate_bytes=" << report.certificate_bytes << '\n';
}

void SolveCommand(const Arguments& arguments) {
  ValidateKeys(
      arguments,
      {"instance",         "input-edges", "tour",           "tour-role",       "expected-cost",
       "device",           "gpus",        "mode",           "output-edges",    "fixed",
       "nonpairs",         "certificate", "manifest",       "lp-backend",      "point-leaf-kernel",
       "point-cta-blocks", "profile",     "distance-cache", "main-pair-cache", "full-metric"});
  cudaee::FgpuSolveOptions options;
  const auto profile = Optional(arguments, "profile", "legacy");
  if (profile != "legacy" && profile != "hybrid-e2e")
    throw std::invalid_argument("未知 solve profile");
  options.hybrid_e2e = profile == "hybrid-e2e";
  options.distance_cache = OptionalBoolean(arguments, "distance-cache", true);
  options.main_pair_cache = OptionalBoolean(arguments, "main-pair-cache", true);
  options.full_metric = OptionalBoolean(arguments, "full-metric", true);
  if (!options.hybrid_e2e &&
      (arguments.contains("distance-cache") || arguments.contains("main-pair-cache") ||
       arguments.contains("full-metric"))) {
    throw std::invalid_argument("hybrid cache/metric 消融参数仅适用于 hybrid-e2e");
  }
  if (options.hybrid_e2e && options.full_metric && !options.main_pair_cache)
    throw std::invalid_argument(
        "全度数 metric 需要条件 pair cache；缓存消融请同时 --full-metric 0");
  if (options.hybrid_e2e &&
      (arguments.contains("tour") || arguments.contains("input-edges") ||
       arguments.contains("expected-cost") || arguments.contains("tour-role"))) {
    throw std::invalid_argument(
        "hybrid-e2e 禁止输入最优标签、tour 和预处理边集；标签仅用于事后检查");
  }
  options.device = ParseSolveDevice(arguments);
  options.mode = ParseSolveMode(Optional(arguments, "mode", "gpu-safe"));
  options.serialize_certificate = arguments.contains("certificate");
  const std::string lp_backend = Optional(arguments, "lp-backend", "primal-dual-sec");
  if (lp_backend != "off" && lp_backend != "primal-dual-sec" && lp_backend != "sec-dual") {
    throw std::invalid_argument(
        "--lp-backend 必须为 off、primal-dual-sec 或 sec-dual（尚未实现的 cut 模式不接受）");
  }
  options.enable_lp = lp_backend != "off";
  options.primal_dual_lp = lp_backend == "primal-dual-sec";
  const std::string leaf = Optional(arguments, "point-leaf-kernel", "permutation");
  if (leaf == "permutation") {
    options.point_leaf_kernel = cudaee::PointLeafKernel::kPermutation;
  } else if (leaf == "prescreen-permutation") {
    options.point_leaf_kernel = cudaee::PointLeafKernel::kPrescreenPermutation;
  } else if (leaf == "prescreen-subset-dp") {
    options.point_leaf_kernel = cudaee::PointLeafKernel::kPrescreenSubsetDp;
  } else {
    throw std::invalid_argument("--point-leaf-kernel 的后端名称无效");
  }
  const std::string cta = Optional(arguments, "point-cta-blocks", "4");
  if (cta != "2" && cta != "4") {
    throw std::invalid_argument("--point-cta-blocks 只能为 2 或 4");
  }
  options.point_cta_blocks = cta == "4" ? 4U : 2U;
  const cudaee::FgpuSolveReport report =
      cudaee::RunFgpuElimination(ParseInput(arguments), ParseRunOutputs(arguments, false), options);
  std::cout << "status=OK mode=" << cudaee::ToString(options.mode)
            << " termination=" << cudaee::ToString(report.termination)
            << " initial_edges=" << report.initial_edges << " final_edges=" << report.final_edges
            << " fixed_edges=" << report.fixed_edges << " pairs=" << report.pairs
            << " nonpairs=" << report.nonpairs << " lp_nonpairs=" << report.lp_nonpairs
            << " fixed_anchor_nonpairs=" << report.fixed_anchor_nonpairs
            << " point_nonpairs=" << report.point_nonpairs
            << " nonpair_fixed_edges=" << report.nonpair_fixed_edges
            << " direct_fixed_edges=" << report.direct_fixed_edges
            << " gpu_replayed=" << (report.gpu_replayed ? 1 : 0)
            << " unaudited=" << (report.unaudited ? 1 : 0)
            << " proof_replayed=" << report.proof_replayed
            << " proof_rejected=" << report.proof_rejected
            << " lp_connectivity_cuts=" << report.lp_connectivity_cuts
            << " lp_path_closed_replies=" << report.lp_path_closed_replies
            << " point_path_end_closed_replies=" << report.point_path_end_closed_replies
            << " lp_degree_snapshots=" << report.lp_degree_snapshots
            << " lp_strong_snapshots=" << report.lp_strong_snapshots
            << " lp_lower_bound=" << report.lp_lower_bound
            << " final_hash=" << cudaee::HexHash(report.final_hash)
            << " final_state_hash=" << cudaee::HexHash(report.final_state_hash)
            << " selected_device=" << report.selected_device
            << " resident_bytes=" << report.resident_bytes << " proof_replay_ms=" << std::fixed
            << std::setprecision(3) << report.proof_replay_ms << " commit_ms=" << report.commit_ms
            << " gpu_solve_wall_ms=" << report.gpu_solve_wall_ms
            << " end_to_end_ms=" << report.end_to_end_ms << '\n';
}

void ResidentCommand(const Arguments& arguments, const bool one_shot = false) {
  ValidateKeys(arguments, {"instance",
                           "input-edges",
                           "tour",
                           "tour-role",
                           "expected-cost",
                           "device",
                           "gpus",
                           "potential-candidates",
                           "main-edge-potentials",
                           "main-edge-positions",
                           "quick-hs-candidates",
                           "quick-hs-pair-trials",
                           "quick-hs-two-hop",
                           "enable-extra-edge",
                           "extra-edge-depth",
                           "pdlp-iterations",
                           "max-pdlp-epochs",
                           "max-hs-epochs",
                           "max-jv-rounds",
                           "enable-geometry",
                           "enable-pdlp",
                           "enable-quick-hs",
                           "enable-jv",
                           "enable-main-edge",
                           "enable-strong-metric",
                           "enable-point-nonpair",
                           "enable-direct-fix",
                           "protect-tour",
                           "cpu-audit",
                           "output-edges",
                           "fixed",
                           "nonpairs",
                           "certificate",
                           "manifest"});
  cudaee::FgpuResidentConfig config;
  config.device = ParseDevice(arguments);
  config.potential_candidates =
      OptionalInteger<std::uint32_t>(arguments, "potential-candidates", 32U);
  config.main_edge_potentials =
      OptionalInteger<std::uint32_t>(arguments, "main-edge-potentials", 11U);
  config.main_edge_positions =
      OptionalInteger<std::uint32_t>(arguments, "main-edge-positions", 23U);
  config.quick_hs_candidates =
      OptionalInteger<std::uint32_t>(arguments, "quick-hs-candidates", one_shot ? 16U : 10U);
  config.quick_hs_pair_trials =
      OptionalInteger<std::uint32_t>(arguments, "quick-hs-pair-trials", one_shot ? 0U : 10U);
  config.pdlp_iterations = OptionalInteger<std::uint32_t>(arguments, "pdlp-iterations", 5000U);
  config.max_pdlp_epochs = OptionalInteger<std::uint32_t>(arguments, "max-pdlp-epochs", 0U);
  config.max_hs_epochs = OptionalInteger<std::uint32_t>(arguments, "max-hs-epochs", 0U);
  config.max_jv_rounds = OptionalInteger<std::uint32_t>(arguments, "max-jv-rounds", 0U);
  config.enable_geometry = OptionalBoolean(arguments, "enable-geometry", true);
  config.enable_pdlp = OptionalBoolean(arguments, "enable-pdlp", true);
  config.enable_quick_hs = OptionalBoolean(arguments, "enable-quick-hs", true);
  config.enable_jv = OptionalBoolean(arguments, "enable-jv", true);
  config.enable_main_edge = OptionalBoolean(arguments, "enable-main-edge", one_shot);
  config.enable_strong_metric = OptionalBoolean(arguments, "enable-strong-metric", false);
  config.enable_point_nonpair = OptionalBoolean(arguments, "enable-point-nonpair", one_shot);
  config.enable_direct_fix = OptionalBoolean(arguments, "enable-direct-fix", one_shot);
  config.enable_extra_edge = OptionalBoolean(arguments, "enable-extra-edge", one_shot);
  config.extra_edge_depth =
      OptionalInteger<std::uint32_t>(arguments, "extra-edge-depth", one_shot ? 2U : 1U);
  config.quick_hs_two_hop = OptionalBoolean(arguments, "quick-hs-two-hop", one_shot);
  config.protect_tour = OptionalBoolean(arguments, "protect-tour", !one_shot);
  config.enable_cpu_audit = OptionalBoolean(arguments, "cpu-audit", false);
  config.enable_fixing = one_shot;
  if (one_shot &&
      (!config.enable_main_edge || config.enable_cpu_audit || config.max_pdlp_epochs != 0U ||
       config.max_hs_epochs != 0U || config.max_jv_rounds != 0U)) {
    throw std::invalid_argument(
        "resident-oneshot 必须启用 Main Edge、关闭 CPU audit，并把所有 epoch/round 上限设为 0");
  }
  const cudaee::FgpuResidentRunReport report = cudaee::RunFgpuResidentElimination(
      ParseInput(arguments), ParseRunOutputs(arguments, config.enable_cpu_audit), config);
  std::cout << "status=OK mode=" << (one_shot ? "resident-oneshot" : "resident")
            << " initial_edges=" << report.initial_edges
            << " geometry_deleted=" << report.geometry_committed
            << " main_edge_deleted=" << report.main_edge_committed
            << " strong_metric=" << (config.enable_strong_metric ? 1 : 0)
            << " main_edge_epochs=" << report.main_edge_epochs
            << " lp_deleted=" << report.lp_committed << " pdlp_epochs=" << report.pdlp_epochs
            << " lp_connectivity_cuts=" << report.lp_connectivity_cuts
            << " lp_path_closed_replies=" << report.lp_path_closed_replies
            << " point_path_end_closed_replies=" << report.point_path_end_closed_replies
            << " lp_degree_snapshots=" << report.lp_degree_snapshots
            << " lp_strong_snapshots=" << report.lp_strong_snapshots
            << " lp_lower_bound=" << report.lp_lower_bound << " jv_deleted=" << report.jv_committed
            << " quick_hs_deleted=" << report.quick_hs_committed
            << " hs_full_sweeps=" << report.hs_full_sweeps
            << " hs_active_sweeps=" << report.hs_active_sweeps
            << " extra_edge_deleted=" << report.extra_edge_committed
            << " extra_edge_depth=" << config.extra_edge_depth
            << " extra_edge_epochs=" << report.extra_edge_epochs
            << " final_edges=" << report.final_edges << " converged=" << (report.converged ? 1 : 0)
            << " pairs=" << report.pair_count << " nonpairs=" << report.nonpair_count
            << " lp_nonpairs=" << report.lp_nonpair_committed
            << " fixed_anchor_nonpairs=" << report.fixed_anchor_nonpair_committed
            << " point_nonpairs=" << report.point_nonpair_committed
            << " nonpair_fixed_edges=" << report.nonpair_fix_committed
            << " direct_fixed_edges=" << report.direct_fix_committed
            << " cpu_audited=" << (report.cpu_audited ? 1 : 0)
            << " final_hash=" << cudaee::HexHash(report.final_hash)
            << " final_state_hash=" << cudaee::HexHash(report.final_state_hash)
            << " selected_device=" << report.selected_device
            << " resident_bytes=" << report.resident_bytes << " upload_ms=" << std::fixed
            << std::setprecision(3) << report.upload_ms << " gpu_kernel_ms=" << report.gpu_kernel_ms
            << " geometry_ms=" << report.geometry_ms << " pdlp_ms=" << report.pdlp_ms
            << " main_edge_ms=" << report.main_edge_ms << " jv_ms=" << report.jv_ms
            << " quick_hs_ms=" << report.quick_hs_ms << " extra_edge_ms=" << report.extra_edge_ms
            << " compaction_ms=" << report.compaction_ms
            << " gpu_download_ms=" << report.gpu_download_ms
            << " gpu_solve_wall_ms=" << report.gpu_solve_wall_ms
            << " cpu_audit_ms=" << report.cpu_audit_ms << " output_ms=" << report.output_ms
            << " end_to_end_ms=" << report.end_to_end_ms
            << " trusted_total_ms=" << report.trusted_total_ms
            << " certificate_bytes=" << report.certificate_bytes << '\n';
}

void PdlpInspectCommand(const Arguments& arguments) {
  ValidateKeys(arguments, {"instance", "input-edges", "expected-cost", "device", "gpus", "pdlp",
                           "pdlp-iterations", "cuopt-library", "mincut-oracle"});
  const std::filesystem::path instance = Required(arguments, "instance");
  const std::filesystem::path input_edges = Optional(arguments, "input-edges");
  cudaee::GraphSnapshot graph = input_edges.empty()
                                    ? cudaee::GraphSnapshot::LoadComplete(instance)
                                    : cudaee::GraphSnapshot::Load(instance, input_edges);
  const std::int64_t incumbent =
      ParseInteger<std::int64_t>(Required(arguments, "expected-cost"), "--expected-cost");
  if (incumbent < 0) {
    throw std::invalid_argument("--expected-cost 必须是非负整数");
  }

  const std::string backend = Optional(arguments, "pdlp", "cuopt-baseline");
  if (backend == "cuopt-subtour") {
    cudaee::SubtourPdlpOptions options;
    options.device = ParseDevice(arguments);
    options.cuopt_library = Optional(arguments, "cuopt-library");
    options.mincut_oracle = Optional(arguments, "mincut-oracle");
    const cudaee::SubtourPdlpResult subtour = cudaee::RunFgpuSubtourPdlp(graph, incumbent, options);
    std::cout << "status=OK mode=pdlp-inspect backend=" << subtour.backend
              << " initial_edges=" << graph.ActiveEdgeCount() << " epochs=" << subtour.epochs
              << " cuts=" << subtour.cuts << " support_components=" << subtour.support_components
              << " forced_one_candidates=" << subtour.forced_one_candidates
              << " objective=" << std::setprecision(15) << subtour.objective
              << " dual_objective=" << subtour.dual_objective
              << " exact_bound_numerator=" << subtour.exact_bound.numerator
              << " exact_bound_denominator=" << subtour.exact_bound.denominator
              << " cuopt_solve_ms=" << std::fixed << std::setprecision(3) << subtour.solve_ms
              << " total_ms=" << subtour.total_ms << " separation="
              << (options.mincut_oracle.empty() ? "host-heuristic" : "concorde-cpu-oracle")
              << " output_files=0\n";
    return;
  }

  cudaee::PdlpOptions options;
  options.backend = ParsePdlp(backend);
  if (options.backend == cudaee::PdlpBackend::kOff) {
    throw std::invalid_argument("pdlp-inspect 不接受 --pdlp off");
  }
  options.device = ParseDevice(arguments);
  options.iterations = OptionalInteger<std::uint32_t>(arguments, "pdlp-iterations", 5000U);
  options.cuopt_library = Optional(arguments, "cuopt-library");

  const cudaee::PdlpResult pdlp = cudaee::RunFgpuPdlp(graph, options);
  cudaee::GraphSnapshot reduced = graph;
  const cudaee::EliminationResult diagnostic =
      cudaee::RunLpBoxElimination(&reduced, pdlp, incumbent);
  std::cout << "status=OK mode=pdlp-inspect backend=" << pdlp.backend
            << " initial_edges=" << graph.ActiveEdgeCount()
            << " eliminable_edges=" << diagnostic.proof.size()
            << " remaining_edges=" << reduced.ActiveEdgeCount()
            << " exact_bound_numerator=" << pdlp.exact_bound.numerator
            << " exact_bound_denominator=" << pdlp.exact_bound.denominator
            << " pdlp_solve_ms=" << std::fixed << std::setprecision(3) << pdlp.solve_ms
            << " diagnostic_cpu_int128=1 output_files=0\n";
}

void VerifyCommand(const Arguments& arguments) {
  ValidateKeys(arguments, {"instance", "input-edges", "tour", "tour-role", "expected-cost",
                           "output-edges", "fixed", "nonpairs", "certificate"});
  cudaee::FgpuOutputPaths outputs;
  outputs.edges = Required(arguments, "output-edges");
  outputs.fixed = Optional(arguments, "fixed");
  outputs.nonpairs = Optional(arguments, "nonpairs");
  outputs.certificate = Required(arguments, "certificate");
  const cudaee::FgpuRunReport report =
      cudaee::VerifyFgpuCertificate(ParseInput(arguments), outputs);
  std::cout << "status=VERIFIED records=" << report.certificate.proof.size()
            << " geometry=" << report.geometry_committed << " jv=" << report.jv_committed
            << " lp=" << report.lp_committed << " ht=" << report.ht_committed
            << " quick_hs=" << report.quick_hs_committed << " active_edges=" << report.final_edges
            << " certificate_bytes=" << report.certificate_bytes
            << " final_hash=" << cudaee::HexHash(report.final_hash) << '\n';
}

} // namespace

int main(const int argc, char** argv) {
  try {
    if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "help") {
      PrintHelp();
      return argc < 2 ? 2 : 0;
    }
    const Arguments arguments = ParseArguments(argc, argv, 2);
    const std::string command = argv[1];
    if (command == "solve") {
      SolveCommand(arguments);
    } else if (command == "run") {
      RunCommand(arguments);
    } else if (command == "resident-local") {
      ResidentLocalCommand(arguments);
    } else if (command == "resident") {
      ResidentCommand(arguments);
    } else if (command == "resident-oneshot") {
      ResidentCommand(arguments, true);
    } else if (command == "pdlp-inspect") {
      PdlpInspectCommand(arguments);
    } else if (command == "verify") {
      VerifyCommand(arguments);
    } else {
      throw std::invalid_argument("未知命令: " + command);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "status=ERROR message=" << error.what() << '\n';
    return 1;
  }
}
