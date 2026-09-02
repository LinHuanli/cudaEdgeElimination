#include "cuda_edge_elimination/elimination.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaee {
namespace {

constexpr std::uint32_t kMaxLocalEpochs = 1000000U;
constexpr std::size_t kMaxCombinedProofRecords = 1000000U;
constexpr std::size_t kMaxCombinedHtProofs = 1000000U;

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

void AddChecked(std::size_t* const destination, const std::size_t value,
                const char* const description) {
  if (*destination > std::numeric_limits<std::size_t>::max() - value) {
    throw std::overflow_error(std::string("Local Elimination ") + description + " 计数溢出");
  }
  *destination += value;
}

// 每个子阶段本身已通过方法专属 CPU 验证；这里只做 proof 容器的哈希链、epoch 和 sidecar 重编号。
void AppendEliminationStage(EliminationResult* const combined, const EliminationResult& stage) {
  if (combined == nullptr || combined->final_hash != stage.initial_hash) {
    throw std::logic_error("Local Elimination 子阶段未绑定前一阶段最终快照");
  }
  if (stage.proof.empty()) {
    if (!stage.ht_proofs.empty() || stage.initial_hash != stage.final_hash) {
      throw std::logic_error("Local Elimination 空子阶段含孤立 sidecar 或修改了图");
    }
    return;
  }

  if (stage.proof.size() > kMaxCombinedProofRecords ||
      combined->proof.size() > kMaxCombinedProofRecords - stage.proof.size() ||
      stage.ht_proofs.size() > kMaxCombinedHtProofs ||
      combined->ht_proofs.size() > kMaxCombinedHtProofs - stage.ht_proofs.size() ||
      combined->ht_proofs.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::overflow_error("Local Elimination 聚合 proof 超过安全数量上限");
  }

  if (!combined->proof.empty() &&
      combined->proof.back().epoch == std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("Local Elimination 聚合 proof epoch 溢出");
  }
  const std::uint32_t first_epoch =
      combined->proof.empty() ? 0U : combined->proof.back().epoch + 1U;
  const std::uint32_t certificate_base = static_cast<std::uint32_t>(combined->ht_proofs.size());
  bool first_record = true;
  std::uint32_t current_source_epoch = 0U;
  for (const ProofRecord& source : stage.proof) {
    if (first_record) {
      if (source.epoch != 0U) {
        throw std::logic_error("Local Elimination 子阶段 proof epoch 不从 0 开始");
      }
      first_record = false;
    } else if (source.epoch != current_source_epoch) {
      if (current_source_epoch == std::numeric_limits<std::uint32_t>::max() ||
          source.epoch != current_source_epoch + 1U) {
        throw std::logic_error("Local Elimination 子阶段 proof epoch 不连续或溢出");
      }
      current_source_epoch = source.epoch;
    }
    if (first_epoch > std::numeric_limits<std::uint32_t>::max() - source.epoch) {
      throw std::logic_error("Local Elimination 聚合 proof epoch 溢出");
    }

    ProofRecord rebased = source;
    rebased.epoch = first_epoch + source.epoch;
    if (rebased.method == EliminationMethod::kHamiltonTutte) {
      if (rebased.certificate_index >= stage.ht_proofs.size() ||
          certificate_base >
              std::numeric_limits<std::uint32_t>::max() - rebased.certificate_index) {
        throw std::logic_error("Local Elimination HT sidecar 索引非法或溢出");
      }
      rebased.certificate_index += certificate_base;
    } else if (rebased.certificate_index != kNoEliminationCertificate) {
      throw std::logic_error("Local Elimination JV record 非法引用 HT sidecar");
    }
    combined->proof.push_back(rebased);
  }

  combined->ht_proofs.insert(combined->ht_proofs.end(), stage.ht_proofs.begin(),
                             stage.ht_proofs.end());
  // proof metrics 不参与授权；只保留实际提交 epoch，避免固定点空轮在 record 序列中制造间隙。
  for (const EpochMetrics& source : stage.epochs) {
    if (source.committed == 0U) {
      continue;
    }
    if (source.epoch > current_source_epoch) {
      throw std::logic_error("Local Elimination 已提交 metrics 缺少对应 proof epoch");
    }
    EpochMetrics rebased = source;
    rebased.epoch = first_epoch + source.epoch;
    combined->epochs.push_back(rebased);
  }
  combined->final_hash = stage.final_hash;
}

LocalEliminationStageMetrics SummarizeJvStage(const std::uint32_t stage_index,
                                              const std::size_t edges_before,
                                              const std::size_t edges_after,
                                              const EliminationResult& result,
                                              const double elapsed_ms) {
  LocalEliminationStageMetrics metrics;
  metrics.stage = stage_index;
  metrics.kind = LocalEliminationStage::kJv;
  metrics.backend = result.backend;
  metrics.initial_hash = result.initial_hash;
  metrics.final_hash = result.final_hash;
  metrics.edges_before = edges_before;
  metrics.edges_after = edges_after;
  metrics.committed = result.proof.size();
  metrics.jv_rounds = static_cast<std::uint32_t>(result.epochs.size());
  metrics.elapsed_ms = elapsed_ms;
  for (const EpochMetrics& epoch : result.epochs) {
    AddChecked(&metrics.proposed, epoch.proposed, "JV proposed");
    AddChecked(&metrics.verified, epoch.verified, "JV verified");
    AddChecked(&metrics.rejected, epoch.rejected, "JV rejected");
  }
  return metrics;
}

LocalEliminationStageMetrics SummarizeHtStage(const std::uint32_t stage_index,
                                              const std::size_t edges_before,
                                              const std::size_t edges_after,
                                              const HtScanResult& scan) {
  const EpochMetrics& epoch = scan.elimination.epochs.front();
  return {.stage = stage_index,
          .kind = LocalEliminationStage::kHamiltonTutte,
          .backend = scan.elimination.backend,
          .initial_hash = scan.elimination.initial_hash,
          .final_hash = scan.elimination.final_hash,
          .edges_before = edges_before,
          .edges_after = edges_after,
          .proposed = epoch.proposed,
          .verified = epoch.verified,
          .rejected = epoch.rejected,
          .committed = epoch.committed,
          .jv_rounds = 0U,
          .eligible_targets = scan.eligible_targets,
          .target_offset = scan.target_offset,
          .attempted_targets = scan.attempts.size(),
          .proven_targets = scan.proven_targets,
          .unresolved_targets = scan.unresolved_targets,
          .elapsed_ms = scan.total_ms};
}

} // namespace

std::string ToString(const LocalEliminationStage stage) {
  switch (stage) {
  case LocalEliminationStage::kJv:
    return "JV";
  case LocalEliminationStage::kHamiltonTutte:
    return "HT";
  }
  throw std::invalid_argument("未知 Local Elimination 阶段");
}

std::string ToString(const LocalEliminationTermination termination) {
  switch (termination) {
  case LocalEliminationTermination::kConverged:
    return "converged";
  case LocalEliminationTermination::kJvRoundLimit:
    return "jv-round-limit";
  case LocalEliminationTermination::kHtEpochLimit:
    return "ht-epoch-limit";
  }
  throw std::invalid_argument("未知 Local Elimination 终止原因");
}

LocalEliminationResult RunLocalElimination(GraphSnapshot* const graph,
                                           const LocalEliminationOptions& options) {
  if (graph == nullptr) {
    throw std::invalid_argument("Local Elimination 的图不能为空");
  }
  if (options.max_jv_rounds == 0U || options.max_jv_rounds > kMaxLocalEpochs ||
      options.max_ht_epochs == 0U || options.max_ht_epochs > kMaxLocalEpochs) {
    throw std::invalid_argument("Local Elimination 的 JV/HT epoch 预算必须位于 [1,1000000]");
  }
  if (options.ht_scan_options.target_offset != 0U) {
    throw std::invalid_argument("Local Elimination 自动管理 HT target_offset，入口值必须为 0");
  }

  GraphSnapshot working = *graph;
  LocalEliminationResult local;
  local.elimination.backend = "local-jv-ht-cpu-verified";
  local.elimination.initial_hash = working.ContentHash();
  local.elimination.final_hash = local.elimination.initial_hash;

  const auto run_jv_fixed_point = [&]() -> bool {
    const std::size_t edges_before = working.ActiveEdgeCount();
    const auto begin = std::chrono::steady_clock::now();
    const EliminationResult jv =
        RunJvElimination(&working, options.jv_backend, options.max_jv_rounds);
    const double elapsed_ms = ElapsedMilliseconds(begin);
    if (jv.epochs.empty()) {
      throw std::logic_error("Local Elimination 的 JV 子阶段未产生终止 metrics");
    }
    local.stages.push_back(SummarizeJvStage(static_cast<std::uint32_t>(local.stages.size()),
                                            edges_before, working.ActiveEdgeCount(), jv,
                                            elapsed_ms));
    AppendEliminationStage(&local.elimination, jv);
    return jv.epochs.back().committed == 0U;
  };

  if (!run_jv_fixed_point()) {
    local.termination = LocalEliminationTermination::kJvRoundLimit;
    *graph = std::move(working);
    return local;
  }

  std::uint64_t target_offset = 0U;
  for (std::uint32_t ht_epoch = 0U; ht_epoch < options.max_ht_epochs; ++ht_epoch) {
    HtScanOptions scan_options = options.ht_scan_options;
    scan_options.target_offset = target_offset;
    const std::size_t edges_before = working.ActiveEdgeCount();
    HtScanResult scan = RunHtScanEpoch(&working, scan_options);
    if (scan.elimination.epochs.size() != 1U) {
      throw std::logic_error("Local Elimination 的 HT scan 未产生唯一提交 metrics");
    }
    const std::size_t committed = scan.elimination.proof.size();
    const std::uint64_t attempted = scan.attempts.size();
    local.stages.push_back(SummarizeHtStage(static_cast<std::uint32_t>(local.stages.size()),
                                            edges_before, working.ActiveEdgeCount(), scan));
    AppendEliminationStage(&local.elimination, scan.elimination);

    if (committed != 0U) {
      // 新快照可能产生新的 JV 见证；先恢复 JV 固定点，再从权重/端点序列开头重排。
      target_offset = 0U;
      if (!run_jv_fixed_point()) {
        local.termination = LocalEliminationTermination::kJvRoundLimit;
        *graph = std::move(working);
        return local;
      }
      continue;
    }

    if (attempted > scan.eligible_targets - scan.target_offset) {
      throw std::logic_error("Local Elimination 的 HT scan 返回非法目标切片");
    }
    target_offset = scan.target_offset + attempted;
    if (target_offset == scan.eligible_targets) {
      local.termination = LocalEliminationTermination::kConverged;
      *graph = std::move(working);
      return local;
    }
    if (attempted == 0U) {
      throw std::logic_error("Local Elimination 的 HT sweep 未前进");
    }
  }

  local.termination = LocalEliminationTermination::kHtEpochLimit;
  *graph = std::move(working);
  return local;
}

} // namespace cudaee
