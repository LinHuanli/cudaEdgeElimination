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
            << "  fgpu-elim verify --instance FILE [--input-edges FILE] --certificate FILE\n"
            << "    --output-edges FILE [--fixed FILE --nonpairs FILE]\n"
            << "    [--tour FILE --tour-role incumbent|known-optimum --expected-cost N]\n\n"
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

cudaee::FgpuOutputPaths ParseRunOutputs(const Arguments& arguments) {
  cudaee::FgpuOutputPaths outputs;
  outputs.edges = CheckedOutput(Required(arguments, "output-edges"));
  outputs.fixed = CheckedOutput(Required(arguments, "fixed"));
  outputs.nonpairs = CheckedOutput(Required(arguments, "nonpairs"));
  outputs.certificate = CheckedOutput(Required(arguments, "certificate"));
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
            << " geometry_kernel_ms=" << std::fixed << std::setprecision(3)
            << report.geometry.kernel_ms << " geometry_verify_ms=" << report.geometry.verify_ms
            << " certificate_bytes=" << report.certificate_bytes << " total_ms=" << report.total_ms
            << '\n';
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
            << " active_edges=" << report.final_edges
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
    if (command == "run") {
      RunCommand(arguments);
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
