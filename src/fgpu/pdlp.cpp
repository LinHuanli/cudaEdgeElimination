#include "cuda_edge_elimination/fgpu.hpp"

#include "../cpu/elimination_commit.hpp"
#include "lp_box_verifier.hpp"
#include "pdlp_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaee {
namespace {

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

struct DegreeModel {
  LpEpoch epoch;
  std::vector<std::int32_t> column_edge_id;
};

DegreeModel BuildDegreeModel(const GraphSnapshot& graph) {
  DegreeModel model;
  std::vector<std::int32_t> edge_column(graph.edges.size(), -1);
  model.column_edge_id.reserve(graph.ActiveEdgeCount());
  for (std::size_t edge_id = 0U; edge_id < graph.edges.size(); ++edge_id) {
    if (graph.edges[edge_id].active) {
      if (model.column_edge_id.size() >=
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("PDLP degree relaxation 列数超过 int32");
      }
      edge_column[edge_id] = static_cast<std::int32_t>(model.column_edge_id.size());
      model.column_edge_id.push_back(static_cast<std::int32_t>(edge_id));
    }
  }

  LpEpoch& epoch = model.epoch;
  epoch.rows = graph.dimension;
  epoch.columns = static_cast<std::int32_t>(model.column_edge_id.size());
  epoch.objective_sense = 1;
  epoch.objective_offset = 0.0;
  epoch.objective.reserve(model.column_edge_id.size());
  epoch.lower_bounds.assign(model.column_edge_id.size(), 0.0);
  epoch.upper_bounds.assign(model.column_edge_id.size(), 1.0);
  epoch.variable_types.assign(model.column_edge_id.size(), 'C');
  epoch.edge_u.reserve(model.column_edge_id.size());
  epoch.edge_v.reserve(model.column_edge_id.size());
  for (const std::int32_t edge_id : model.column_edge_id) {
    const Edge& edge = graph.edges[static_cast<std::size_t>(edge_id)];
    epoch.objective.push_back(static_cast<double>(edge.weight));
    epoch.edge_u.push_back(edge.u);
    epoch.edge_v.push_back(edge.v);
  }

  epoch.row_offsets.reserve(static_cast<std::size_t>(graph.dimension) + 1U);
  epoch.row_offsets.push_back(0);
  epoch.senses.assign(static_cast<std::size_t>(graph.dimension), 'E');
  epoch.rhs.assign(static_cast<std::size_t>(graph.dimension), 2.0);
  epoch.column_indices.reserve(2U * model.column_edge_id.size());
  epoch.values.reserve(2U * model.column_edge_id.size());
  for (std::int32_t vertex = 0; vertex < graph.dimension; ++vertex) {
    for (std::int32_t offset = graph.row_offsets[static_cast<std::size_t>(vertex)];
         offset < graph.row_offsets[static_cast<std::size_t>(vertex) + 1U]; ++offset) {
      const std::int32_t edge_id = graph.csr_edge_ids[static_cast<std::size_t>(offset)];
      const std::int32_t column = edge_column[static_cast<std::size_t>(edge_id)];
      if (column < 0) {
        throw std::logic_error("PDLP degree relaxation CSR 引用了非活动边");
      }
      epoch.column_indices.push_back(column);
      epoch.values.push_back(1.0);
    }
    if (epoch.column_indices.size() >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw std::overflow_error("PDLP degree relaxation 非零元超过 int32");
    }
    epoch.row_offsets.push_back(static_cast<std::int32_t>(epoch.column_indices.size()));
  }
  epoch.Validate();
  epoch.content_hash = epoch.ComputeHash();
  return model;
}

std::vector<std::int64_t> QuantizeDual(const std::vector<double>& dual,
                                       const std::uint32_t fractional_bits) {
  if (fractional_bits > 40U) {
    throw std::invalid_argument("PDLP dual 定点位数非法");
  }
  const std::int64_t denominator = std::int64_t{1} << fractional_bits;
  std::vector<std::int64_t> quantized(dual.size());
  for (std::size_t index = 0; index < dual.size(); ++index) {
    const long double scaled = static_cast<long double>(dual[index]) * denominator;
    if (!std::isfinite(dual[index]) || std::abs(scaled) > 1.0e15L) {
      throw std::runtime_error("PDLP dual 无法安全量化");
    }
    quantized[index] = static_cast<std::int64_t>(std::round(scaled));
  }
  return quantized;
}

void FillScores(const GraphSnapshot& graph, const DegreeModel& model,
                const std::vector<std::int64_t>& quantized, const std::uint32_t fractional_bits,
                std::vector<double>* const scores) {
  if (quantized.size() != static_cast<std::size_t>(graph.dimension) || fractional_bits > 40U) {
    throw std::invalid_argument("PDLP score 的 dual 维度或定点位数非法");
  }
  const std::int64_t denominator = std::int64_t{1} << fractional_bits;
  scores->assign(graph.edges.size(), -std::numeric_limits<double>::infinity());
  for (std::size_t column = 0; column < model.column_edge_id.size(); ++column) {
    const std::int32_t edge_id = model.column_edge_id[column];
    const Edge& edge = graph.edges[static_cast<std::size_t>(edge_id)];
    const __int128 reduced = static_cast<__int128>(edge.weight) * denominator -
                             quantized[static_cast<std::size_t>(edge.u)] -
                             quantized[static_cast<std::size_t>(edge.v)];
    (*scores)[static_cast<std::size_t>(edge_id)] =
        static_cast<double>(static_cast<long double>(reduced) / denominator);
  }
}

} // namespace

PdlpResult RunFgpuPdlp(const GraphSnapshot& graph, const PdlpOptions& options) {
  PdlpResult result;
  result.backend = ToString(options.backend);
  result.edge_scores.assign(graph.edges.size(), -std::numeric_limits<double>::infinity());
  if (options.backend == PdlpBackend::kOff) {
    return result;
  }
  if (options.iterations == 0U || options.fractional_bits > 40U) {
    throw std::invalid_argument("PDLP iterations 必须大于 0，fractional_bits 不得超过 40");
  }
  const DegreeModel model = BuildDegreeModel(graph);
  const auto begin = std::chrono::steady_clock::now();
  std::vector<double> dual;
  if (options.backend == PdlpBackend::kNative) {
    std::string reason;
    if (!detail::NativePdlpCudaAvailable(&reason)) {
      throw std::runtime_error("native PDLP CUDA 后端不可用: " + reason);
    }
    const detail::NativePdlpDeviceResult device = detail::SolveDegreeRelaxationCuda(graph, options);
    dual = device.vertex_dual;
    result.selected_device = device.selected_device;
    result.iterations = device.iterations;
    result.solve_ms = device.solve_ms;
    result.backend = "native-cuda-degree-subgradient-" + device.implementation;
  } else {
    CuOptSession session(options.cuopt_library);
    const LpSolution solution = session.Solve(model.epoch, false);
    dual = solution.dual;
    result.solve_ms = solution.solve_time_seconds * 1000.0;
    result.iterations = 0U;
    result.backend = "cuopt-baseline-degree-lp";
  }
  if (result.solve_ms == 0.0) {
    result.solve_ms = ElapsedMilliseconds(begin);
  }
  result.exact_bound = BuildExactModelBound(model.epoch, dual, options.fractional_bits);
  result.cpu_certified = result.exact_bound.certified;
  if (!result.cpu_certified) {
    throw std::runtime_error("PDLP multiplier 未通过 __int128 box-Lagrangian 门禁: " +
                             result.exact_bound.reason);
  }
  result.fractional_bits = options.fractional_bits;
  result.vertex_dual_numerator = QuantizeDual(dual, options.fractional_bits);
  FillScores(graph, model, result.vertex_dual_numerator, options.fractional_bits,
             &result.edge_scores);
  return result;
}

LpBoxVerificationData BuildLpBoxVerificationData(const GraphSnapshot& graph,
                                                 const LpBoxProof& proof) {
  LpBoxVerificationData result;
  if (proof.snapshot_hash != graph.ContentHash()) {
    result.reason = "LP box proof 的不可变快照哈希不匹配";
    return result;
  }
  if (proof.fractional_bits > 40U || proof.incumbent_cost < 0 ||
      proof.vertex_dual_numerator.size() != static_cast<std::size_t>(graph.dimension)) {
    result.reason = "LP box proof 的分母、incumbent 或 dual 维度非法";
    return result;
  }
  const __int128 denominator = static_cast<__int128>(1) << proof.fractional_bits;
  __int128 lower = 0;
  for (const std::int64_t dual : proof.vertex_dual_numerator) {
    __int128 term = 0;
    if (__builtin_mul_overflow(static_cast<__int128>(2), static_cast<__int128>(dual), &term) ||
        __builtin_add_overflow(lower, term, &lower)) {
      result.reason = "LP box proof 的 degree 常数项溢出";
      return result;
    }
  }
  result.reduced_cost_numerator.assign(graph.edges.size(), 0);
  for (std::size_t edge_id = 0U; edge_id < graph.edges.size(); ++edge_id) {
    const Edge& edge = graph.edges[edge_id];
    if (!edge.active) {
      continue;
    }
    __int128 objective = 0;
    if (__builtin_mul_overflow(static_cast<__int128>(edge.weight), denominator, &objective)) {
      result.reason = "LP box proof 的目标系数量化溢出";
      return result;
    }
    __int128 reduced = 0;
    if (__builtin_sub_overflow(
            objective,
            static_cast<__int128>(proof.vertex_dual_numerator[static_cast<std::size_t>(edge.u)]),
            &reduced) ||
        __builtin_sub_overflow(
            reduced,
            static_cast<__int128>(proof.vertex_dual_numerator[static_cast<std::size_t>(edge.v)]),
            &reduced)) {
      result.reason = "LP box proof 的 reduced cost 累加溢出";
      return result;
    }
    result.reduced_cost_numerator[edge_id] = reduced;
    if (reduced < 0 && __builtin_add_overflow(lower, reduced, &lower)) {
      result.reason = "LP box proof 的 box contribution 溢出";
      return result;
    }
  }
  result.lower_bound_numerator = lower;
  result.certified = true;
  result.reason = "degree equality + complete live-variable box bound 已用 __int128 重放";
  return result;
}

bool detail::VerifyLpBoxCandidateForSnapshot(const GraphSnapshot& graph, const LpBoxProof& proof,
                                             const LpBoxVerificationData& verification_data,
                                             const Candidate& candidate,
                                             const std::uint64_t actual_snapshot_hash,
                                             std::string* const reason) {
  const auto fail = [&](const std::string& message) {
    if (reason != nullptr) {
      *reason = message;
    }
    return false;
  };
  if (!verification_data.certified || proof.snapshot_hash != actual_snapshot_hash) {
    return fail("LP box proof 的共享验证数据未认证或快照已变化");
  }
  if (candidate.method != EliminationMethod::kLpBox || candidate.witness != -1 ||
      candidate.second_witness != -1 || candidate.edge_id < 0 ||
      static_cast<std::size_t>(candidate.edge_id) >= graph.edges.size()) {
    return fail("LP box 候选类型、见证或 edge id 非法");
  }
  const Edge& edge = graph.edges[static_cast<std::size_t>(candidate.edge_id)];
  if (!edge.active || graph.Degree(edge.u) <= 2 || graph.Degree(edge.v) <= 2) {
    return fail("LP box 候选边不活动或违反最小度提交前提");
  }
  __int128 forced = verification_data.lower_bound_numerator;
  const __int128 reduced =
      verification_data.reduced_cost_numerator[static_cast<std::size_t>(candidate.edge_id)];
  if (reduced > 0 && __builtin_add_overflow(forced, reduced, &forced)) {
    return fail("LP forced-one 下界溢出");
  }
  __int128 incumbent = 0;
  const __int128 denominator = static_cast<__int128>(1) << proof.fractional_bits;
  if (__builtin_mul_overflow(static_cast<__int128>(proof.incumbent_cost), denominator,
                             &incumbent)) {
    return fail("LP incumbent 定点乘法溢出");
  }
  if (forced <= incumbent) {
    return fail("LP forced-one 下界未严格超过 incumbent");
  }
  return true;
}

bool VerifyLpBoxCandidate(const GraphSnapshot& graph, const LpBoxProof& proof,
                          const LpBoxVerificationData& verification_data,
                          const Candidate& candidate, std::string* const reason) {
  return detail::VerifyLpBoxCandidateForSnapshot(graph, proof, verification_data, candidate,
                                                 graph.ContentHash(), reason);
}

EliminationResult RunLpBoxElimination(GraphSnapshot* const graph, const PdlpResult& pdlp,
                                      const std::int64_t incumbent_cost) {
  if (graph == nullptr || incumbent_cost < 0 || !pdlp.cpu_certified ||
      pdlp.vertex_dual_numerator.size() != static_cast<std::size_t>(graph->dimension)) {
    throw std::invalid_argument("LP box 消元缺少图、incumbent 或已认证量化 dual");
  }
  EliminationResult result;
  result.backend = pdlp.backend + "+cpu-int128-forced-one";
  result.initial_hash = graph->ContentHash();
  LpBoxProof proof{result.initial_hash, pdlp.fractional_bits, incumbent_cost,
                   pdlp.vertex_dual_numerator};
  const LpBoxVerificationData verification = BuildLpBoxVerificationData(*graph, proof);
  if (!verification.certified) {
    throw std::runtime_error("LP box 共享 proof 验证失败: " + verification.reason);
  }
  std::vector<Candidate> candidates;
  candidates.reserve(graph->ActiveEdgeCount() / 4U);
  for (std::size_t edge_id = 0U; edge_id < graph->edges.size(); ++edge_id) {
    Candidate candidate{static_cast<std::int32_t>(edge_id), -1, EliminationMethod::kLpBox, -1};
    if (detail::VerifyLpBoxCandidateForSnapshot(*graph, proof, verification, candidate,
                                                result.initial_hash, nullptr)) {
      candidates.push_back(candidate);
    }
  }
  const std::size_t proposed = candidates.size();
  std::vector<Candidate> committed =
      detail::CommitVerifiedCandidates(graph, std::move(candidates), result.initial_hash);
  if (!committed.empty()) {
    result.lp_box_proofs.push_back(proof);
  }
  for (const Candidate& candidate : committed) {
    const Edge& edge = graph->edges[static_cast<std::size_t>(candidate.edge_id)];
    result.proof.push_back({0U, result.initial_hash, candidate.edge_id, edge.u, edge.v, -1,
                            EliminationMethod::kLpBox, 0U, -1});
  }
  result.epochs.push_back({.epoch = 0U,
                           .edges_before = graph->ActiveEdgeCount() + committed.size(),
                           .proposed = proposed,
                           .verified = proposed,
                           .rejected = 0U,
                           .committed = committed.size()});
  result.final_hash = graph->ContentHash();
  return result;
}

} // namespace cudaee
