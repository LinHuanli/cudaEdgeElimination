#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

instance="${1:-}"
max_targets="${2:-8}"
if [[ -z "${instance}" || ! "${instance}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法：$0 INSTANCE [MAX_TARGETS]" >&2
  exit 2
fi
if [[ ! "${max_targets}" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：MAX_TARGETS 必须是正整数。" >&2
  exit 2
fi

cpu_cost_threads="${CUDAEE_CPU_COST_THREADS:-8}"
if [[ ! "${cpu_cost_threads}" =~ ^[1-8]$ ]]; then
  echo "错误：CUDAEE_CPU_COST_THREADS 必须位于 [1,8]。" >&2
  exit 2
fi
export OMP_NUM_THREADS="${cpu_cost_threads}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

reuse_reply_cuda_cache="${CUDAEE_REUSE_REPLY_CUDA_CACHE:-1}"
if [[ "${reuse_reply_cuda_cache}" != "0" && "${reuse_reply_cuda_cache}" != "1" ]]; then
  echo "错误：CUDAEE_REUSE_REPLY_CUDA_CACHE 必须为 0 或 1。" >&2
  exit 2
fi

deduplicate_reply_tasks="${CUDAEE_DEDUPLICATE_REPLY_TASKS:-1}"
if [[ "${deduplicate_reply_tasks}" != "0" && "${deduplicate_reply_tasks}" != "1" ]]; then
  echo "错误：CUDAEE_DEDUPLICATE_REPLY_TASKS 必须为 0 或 1。" >&2
  exit 2
fi

config="${repo_root}/configs/m5_jv_instances.tsv"
row="$(awk -F '\t' -v name="${instance}" '$1 == name { print; found = 1 } END { if (!found) exit 1 }' "${config}")" || {
  echo "错误：${config} 中没有实例 ${instance}。" >&2
  exit 2
}
IFS=$'\t' read -r _ tsp_relative edges_relative certified_optimum <<<"${row}"
tsp="$(realpath "${repo_root}/${tsp_relative}")"
edges="$(realpath "${repo_root}/${edges_relative}")"
if [[ ! -f "${tsp}" || ! -f "${edges}" || ! "${certified_optimum}" =~ ^[0-9]+$ ]]; then
  echo "错误：实例输入或 certified optimum 无效。" >&2
  exit 2
fi

available_kib="$(df -Pk "${repo_root}" | awk 'NR == 2 { print $4 }')"
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "错误：仓库文件系统可用空间不足 8 GiB。" >&2
  exit 2
fi

if [[ ! -f build/cpu-release/CMakeCache.txt ]]; then
  cmake --preset cpu-release
fi
if [[ ! -f build/cuda-release/CMakeCache.txt ]]; then
  cmake --preset cuda-release
fi
cmake --build --preset cpu-release --target cudaee --parallel
cmake --build --preset cuda-release --target cudaee --parallel

cpu_binary="${repo_root}/build/cpu-release/cudaee"
cuda_binary="${repo_root}/build/cuda-release/cudaee"
physical_gpu="${CUDAEE_BENCHMARK_GPU:-$(tools/select_gpu.sh)}"
if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "错误：CUDAEE_BENCHMARK_GPU 必须是物理 GPU 编号。" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${instance}-ht-scan-${timestamp}-$$"
run_dir="${repo_root}/artifacts/${run_id}"
mkdir -p "${run_dir}"

protected_tour_source="${CUDAEE_BENCHMARK_TOUR:-}"
protected_tour=""
protected_tour_sha256="none"
if [[ -n "${protected_tour_source}" ]]; then
  protected_tour_source="$(realpath "${protected_tour_source}")"
  if [[ ! -f "${protected_tour_source}" ]]; then
    echo "错误：CUDAEE_BENCHMARK_TOUR 不存在。" >&2
    exit 2
  fi
  protected_tour="${run_dir}/protected.opt.tour"
  cp -- "${protected_tour_source}" "${protected_tour}"
  protected_tour_sha256="$(sha256sum "${protected_tour}" | awk '{ print $1 }')"
fi

jv_edges="${run_dir}/jv-fixed.edg"
jv_proof="${run_dir}/jv-fixed.proof"
"${cpu_binary}" gpu-eliminate --tsp "${tsp}" --edges "${edges}" --output "${jv_edges}" \
  --proof "${jv_proof}" --backend cpu --max-rounds 100 \
  >"${run_dir}/jv.stdout" 2>"${run_dir}/jv.stderr"
"${cpu_binary}" verify --tsp "${tsp}" --edges "${edges}" --proof "${jv_proof}" \
  >"${run_dir}/jv.verify.stdout" 2>"${run_dir}/jv.verify.stderr"

tour_arguments=()
protected_tour_checked=0
protected_tour_hash="none"
if [[ -n "${protected_tour}" ]]; then
  tour_arguments=(--protected-tour "${protected_tour}" --expected-cost "${certified_optimum}")
  "${cpu_binary}" tour-check --tsp "${tsp}" --edges "${jv_edges}" \
    --tour "${protected_tour}" --expected-cost "${certified_optimum}" \
    >"${run_dir}/protected.jv.stdout" 2>"${run_dir}/protected.jv.stderr"
  protected_tour_hash="$(awk '{ for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == "tour_hash") print pair[2] } }' "${run_dir}/protected.jv.stdout")"
  protected_tour_checked=1
fi

target_offset="${CUDAEE_HT_TARGET_OFFSET:-0}"
if [[ ! "${target_offset}" =~ ^[0-9]+$ ]]; then
  echo "错误：CUDAEE_HT_TARGET_OFFSET 必须是非负整数。" >&2
  exit 2
fi

# 这是 M5 pilot 的固定有界协议；所有值都会写入 manifest。
ht_arguments=(
  --max-targets "${max_targets}"
  --target-offset "${target_offset}"
  --target-order weight-desc
  --cd-mode missing-or-incompatible
  --max-neighborhood 25
  --max-cd-candidates 5
  --max-candidate-degree 50
  --max-root-replies 10000
  --max-k 5
  --max-deletion-sets 100
  --max-depth 2
  --max-states 2000
  --max-total-replies 20000
  --max-replies-per-move 2000
  --max-point-candidates 3
  --max-end-candidates 3
  --reply-frontier-batch-states 256
  --leaf-frontier-batch-states 256
  --cost-batch-size 4096
  --cuda-min-cost-cells 128
  --reuse-reply-cuda-cache "${reuse_reply_cuda_cache}"
  --deduplicate-reply-tasks "${deduplicate_reply_tasks}"
)

run_scan() {
  local backend="$1"
  local binary="$2"
  local output="${run_dir}/${backend}.edg"
  local proof="${run_dir}/${backend}.proof"
  local report="${run_dir}/${backend}.report"
  local stdout_file="${run_dir}/${backend}.stdout"
  local stderr_file="${run_dir}/${backend}.stderr"
  local start_ns end_ns
  start_ns="$(date +%s%N)"
  if [[ "${backend}" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${binary}" ht-scan \
      --tsp "${tsp}" --edges "${jv_edges}" --output "${output}" --proof "${proof}" \
      --report "${report}" "${tour_arguments[@]}" "${ht_arguments[@]}" \
      --backend cuda --reply-backend cuda --path-append-backend cuda \
      --propagation-backend cuda >"${stdout_file}" 2>"${stderr_file}"
  elif [[ "${backend}" == "hybrid" ]]; then
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${binary}" ht-scan \
      --tsp "${tsp}" --edges "${jv_edges}" --output "${output}" --proof "${proof}" \
      --report "${report}" "${tour_arguments[@]}" "${ht_arguments[@]}" \
      --backend cpu --leaf-backend cuda --reply-backend cpu --path-append-backend cpu \
      --propagation-backend cpu >"${stdout_file}" 2>"${stderr_file}"
  elif [[ "${backend}" == "fused" ]]; then
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${binary}" ht-scan \
      --tsp "${tsp}" --edges "${jv_edges}" --output "${output}" --proof "${proof}" \
      --report "${report}" "${tour_arguments[@]}" "${ht_arguments[@]}" \
      --backend cpu --leaf-backend cuda --reply-backend cpu --path-append-backend cpu \
      --propagation-backend cpu --fuse-leaf-buckets 1 \
      >"${stdout_file}" 2>"${stderr_file}"
  elif [[ "${backend}" == "cpu-fused" ]]; then
    "${binary}" ht-scan \
      --tsp "${tsp}" --edges "${jv_edges}" --output "${output}" --proof "${proof}" \
      --report "${report}" "${tour_arguments[@]}" "${ht_arguments[@]}" \
      --backend cpu --reply-backend cpu --path-append-backend cpu \
      --propagation-backend cpu --fuse-leaf-buckets 1 \
      >"${stdout_file}" 2>"${stderr_file}"
  else
    "${binary}" ht-scan \
      --tsp "${tsp}" --edges "${jv_edges}" --output "${output}" --proof "${proof}" \
      --report "${report}" "${tour_arguments[@]}" "${ht_arguments[@]}" \
      --backend cpu --reply-backend cpu --path-append-backend cpu \
      --propagation-backend cpu --fuse-leaf-buckets 0 \
      >"${stdout_file}" 2>"${stderr_file}"
  fi
  end_ns="$(date +%s%N)"
  awk -v first="${start_ns}" -v last="${end_ns}" \
    'BEGIN { printf "%.3f\n", (last-first)/1000000 }' >"${run_dir}/${backend}.wall-ms"

  "${cpu_binary}" verify --tsp "${tsp}" --edges "${jv_edges}" --proof "${proof}" \
    >"${run_dir}/${backend}.verify.stdout" 2>"${run_dir}/${backend}.verify.stderr"
  if [[ -n "${protected_tour}" ]]; then
    "${cpu_binary}" tour-check --tsp "${tsp}" --edges "${output}" \
      --tour "${protected_tour}" --expected-cost "${certified_optimum}" \
      >"${run_dir}/${backend}.tour.stdout" 2>"${run_dir}/${backend}.tour.stderr"
  fi
}

echo "运行 ${instance} HT scan CPU，targets=${max_targets} offset=${target_offset}"
run_scan cpu "${cpu_binary}"
echo "运行 ${instance} HT scan CPU fused leaf buckets"
run_scan cpu-fused "${cpu_binary}"
echo "运行 ${instance} HT scan CUDA，物理 GPU ${physical_gpu}"
run_scan cuda "${cuda_binary}"
echo "运行 ${instance} HT scan hybrid，物理 GPU ${physical_gpu}"
run_scan hybrid "${cuda_binary}"
echo "运行 ${instance} HT scan fused leaf buckets，物理 GPU ${physical_gpu}"
run_scan fused "${cuda_binary}"

cmp "${run_dir}/cpu.edg" "${run_dir}/cuda.edg"
cmp "${run_dir}/cpu.edg" "${run_dir}/cpu-fused.edg"
cmp "${run_dir}/cpu.edg" "${run_dir}/hybrid.edg"
cmp "${run_dir}/cpu.edg" "${run_dir}/fused.edg"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/cpu.report" >"${run_dir}/cpu.work-signature"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/cpu-fused.report" >"${run_dir}/cpu-fused.work-signature"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/cuda.report" >"${run_dir}/cuda.work-signature"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/hybrid.report" >"${run_dir}/hybrid.work-signature"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/fused.report" >"${run_dir}/fused.work-signature"
cmp "${run_dir}/cpu.work-signature" "${run_dir}/cuda.work-signature"
cmp "${run_dir}/cpu.work-signature" "${run_dir}/cpu-fused.work-signature"
cmp "${run_dir}/cpu.work-signature" "${run_dir}/hybrid.work-signature"
cmp "${run_dir}/cpu.work-signature" "${run_dir}/fused.work-signature"

read_field() {
  awk -v key="$2" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' "$1"
}

cpu_search_ms="$(read_field "${run_dir}/cpu.report" search_ms)"
cpu_fused_search_ms="$(read_field "${run_dir}/cpu-fused.report" search_ms)"
cuda_search_ms="$(read_field "${run_dir}/cuda.report" search_ms)"
hybrid_search_ms="$(read_field "${run_dir}/hybrid.report" search_ms)"
fused_search_ms="$(read_field "${run_dir}/fused.report" search_ms)"
cpu_target_execution_ms="$(read_field "${run_dir}/cpu.report" target_execution_ms)"
cpu_fused_target_execution_ms="$(read_field "${run_dir}/cpu-fused.report" target_execution_ms)"
cuda_target_execution_ms="$(read_field "${run_dir}/cuda.report" target_execution_ms)"
hybrid_target_execution_ms="$(read_field "${run_dir}/hybrid.report" target_execution_ms)"
fused_target_execution_ms="$(read_field "${run_dir}/fused.report" target_execution_ms)"
cpu_candidate_ms="$(read_field "${run_dir}/cpu.report" candidate_ms)"
cpu_fused_candidate_ms="$(read_field "${run_dir}/cpu-fused.report" candidate_ms)"
cuda_candidate_ms="$(read_field "${run_dir}/cuda.report" candidate_ms)"
hybrid_candidate_ms="$(read_field "${run_dir}/hybrid.report" candidate_ms)"
cpu_work_graph_ms="$(read_field "${run_dir}/cpu.report" work_graph_ms)"
cpu_fused_work_graph_ms="$(read_field "${run_dir}/cpu-fused.report" work_graph_ms)"
cuda_work_graph_ms="$(read_field "${run_dir}/cuda.report" work_graph_ms)"
hybrid_work_graph_ms="$(read_field "${run_dir}/hybrid.report" work_graph_ms)"
fused_work_graph_ms="$(read_field "${run_dir}/fused.report" work_graph_ms)"
cpu_root_child_normalizations="$(read_field "${run_dir}/cpu.report" root_child_normalizations)"
cpu_fused_root_child_normalizations="$(read_field "${run_dir}/cpu-fused.report" root_child_normalizations)"
cuda_root_child_normalizations="$(read_field "${run_dir}/cuda.report" root_child_normalizations)"
hybrid_root_child_normalizations="$(read_field "${run_dir}/hybrid.report" root_child_normalizations)"
fused_root_child_normalizations="$(read_field "${run_dir}/fused.report" root_child_normalizations)"
if [[ "${cpu_root_child_normalizations}" != "${cpu_fused_root_child_normalizations}" ||
      "${cpu_root_child_normalizations}" != "${cuda_root_child_normalizations}" ||
      "${cpu_root_child_normalizations}" != "${hybrid_root_child_normalizations}" ||
      "${cpu_root_child_normalizations}" != "${fused_root_child_normalizations}" ]]; then
  echo "五路根 child 规范化次数不一致" >&2
  exit 1
fi
cpu_root_child_normalize_ms="$(read_field "${run_dir}/cpu.report" root_child_normalize_ms)"
cpu_fused_root_child_normalize_ms="$(read_field "${run_dir}/cpu-fused.report" root_child_normalize_ms)"
cuda_root_child_normalize_ms="$(read_field "${run_dir}/cuda.report" root_child_normalize_ms)"
hybrid_root_child_normalize_ms="$(read_field "${run_dir}/hybrid.report" root_child_normalize_ms)"
fused_root_child_normalize_ms="$(read_field "${run_dir}/fused.report" root_child_normalize_ms)"
cpu_point_candidate_scans="$(read_field "${run_dir}/cpu.report" point_candidate_scans)"
cpu_fused_point_candidate_scans="$(read_field "${run_dir}/cpu-fused.report" point_candidate_scans)"
cuda_point_candidate_scans="$(read_field "${run_dir}/cuda.report" point_candidate_scans)"
hybrid_point_candidate_scans="$(read_field "${run_dir}/hybrid.report" point_candidate_scans)"
fused_point_candidate_scans="$(read_field "${run_dir}/fused.report" point_candidate_scans)"
cpu_point_candidate_nodes_checked="$(read_field "${run_dir}/cpu.report" point_candidate_nodes_checked)"
cpu_fused_point_candidate_nodes_checked="$(read_field "${run_dir}/cpu-fused.report" point_candidate_nodes_checked)"
cuda_point_candidate_nodes_checked="$(read_field "${run_dir}/cuda.report" point_candidate_nodes_checked)"
hybrid_point_candidate_nodes_checked="$(read_field "${run_dir}/hybrid.report" point_candidate_nodes_checked)"
fused_point_candidate_nodes_checked="$(read_field "${run_dir}/fused.report" point_candidate_nodes_checked)"
cpu_point_candidate_nodes_ranked="$(read_field "${run_dir}/cpu.report" point_candidate_nodes_ranked)"
cpu_fused_point_candidate_nodes_ranked="$(read_field "${run_dir}/cpu-fused.report" point_candidate_nodes_ranked)"
cuda_point_candidate_nodes_ranked="$(read_field "${run_dir}/cuda.report" point_candidate_nodes_ranked)"
hybrid_point_candidate_nodes_ranked="$(read_field "${run_dir}/hybrid.report" point_candidate_nodes_ranked)"
fused_point_candidate_nodes_ranked="$(read_field "${run_dir}/fused.report" point_candidate_nodes_ranked)"
cpu_point_candidate_nodes_selected="$(read_field "${run_dir}/cpu.report" point_candidate_nodes_selected)"
cpu_fused_point_candidate_nodes_selected="$(read_field "${run_dir}/cpu-fused.report" point_candidate_nodes_selected)"
cuda_point_candidate_nodes_selected="$(read_field "${run_dir}/cuda.report" point_candidate_nodes_selected)"
hybrid_point_candidate_nodes_selected="$(read_field "${run_dir}/hybrid.report" point_candidate_nodes_selected)"
fused_point_candidate_nodes_selected="$(read_field "${run_dir}/fused.report" point_candidate_nodes_selected)"
if [[ "${cpu_point_candidate_scans}" != "${cpu_fused_point_candidate_scans}" ||
      "${cpu_point_candidate_scans}" != "${cuda_point_candidate_scans}" ||
      "${cpu_point_candidate_scans}" != "${hybrid_point_candidate_scans}" ||
      "${cpu_point_candidate_scans}" != "${fused_point_candidate_scans}" ||
      "${cpu_point_candidate_nodes_checked}" != "${cpu_fused_point_candidate_nodes_checked}" ||
      "${cpu_point_candidate_nodes_checked}" != "${cuda_point_candidate_nodes_checked}" ||
      "${cpu_point_candidate_nodes_checked}" != "${hybrid_point_candidate_nodes_checked}" ||
      "${cpu_point_candidate_nodes_checked}" != "${fused_point_candidate_nodes_checked}" ||
      "${cpu_point_candidate_nodes_ranked}" != "${cpu_fused_point_candidate_nodes_ranked}" ||
      "${cpu_point_candidate_nodes_ranked}" != "${cuda_point_candidate_nodes_ranked}" ||
      "${cpu_point_candidate_nodes_ranked}" != "${hybrid_point_candidate_nodes_ranked}" ||
      "${cpu_point_candidate_nodes_ranked}" != "${fused_point_candidate_nodes_ranked}" ||
      "${cpu_point_candidate_nodes_selected}" != "${cpu_fused_point_candidate_nodes_selected}" ||
      "${cpu_point_candidate_nodes_selected}" != "${cuda_point_candidate_nodes_selected}" ||
      "${cpu_point_candidate_nodes_selected}" != "${hybrid_point_candidate_nodes_selected}" ||
      "${cpu_point_candidate_nodes_selected}" != "${fused_point_candidate_nodes_selected}" ]]; then
  echo "五路 point candidate 的规范工作计数不一致" >&2
  exit 1
fi
cpu_point_candidate_scan_ms="$(read_field "${run_dir}/cpu.report" point_candidate_scan_ms)"
cpu_fused_point_candidate_scan_ms="$(read_field "${run_dir}/cpu-fused.report" point_candidate_scan_ms)"
cuda_point_candidate_scan_ms="$(read_field "${run_dir}/cuda.report" point_candidate_scan_ms)"
hybrid_point_candidate_scan_ms="$(read_field "${run_dir}/hybrid.report" point_candidate_scan_ms)"
fused_point_candidate_scan_ms="$(read_field "${run_dir}/fused.report" point_candidate_scan_ms)"
cpu_point_candidate_sort_ms="$(read_field "${run_dir}/cpu.report" point_candidate_sort_ms)"
cpu_fused_point_candidate_sort_ms="$(read_field "${run_dir}/cpu-fused.report" point_candidate_sort_ms)"
cuda_point_candidate_sort_ms="$(read_field "${run_dir}/cuda.report" point_candidate_sort_ms)"
hybrid_point_candidate_sort_ms="$(read_field "${run_dir}/hybrid.report" point_candidate_sort_ms)"
fused_point_candidate_sort_ms="$(read_field "${run_dir}/fused.report" point_candidate_sort_ms)"
cpu_leaf_ms="$(read_field "${run_dir}/cpu.report" leaf_ms)"
cpu_fused_leaf_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_ms)"
cuda_leaf_ms="$(read_field "${run_dir}/cuda.report" leaf_ms)"
hybrid_leaf_ms="$(read_field "${run_dir}/hybrid.report" leaf_ms)"
fused_leaf_ms="$(read_field "${run_dir}/fused.report" leaf_ms)"
hybrid_leaf_cuda_batches="$(read_field "${run_dir}/hybrid.report" leaf_cuda_cost_batches)"
fused_leaf_cuda_batches="$(read_field "${run_dir}/fused.report" leaf_cuda_cost_batches)"
hybrid_leaf_frontier_batches="$(read_field "${run_dir}/hybrid.report" leaf_frontier_batches)"
fused_leaf_frontier_batches="$(read_field "${run_dir}/fused.report" leaf_frontier_batches)"
cpu_fused_leaf_frontier_batches="$(read_field "${run_dir}/cpu-fused.report" leaf_frontier_batches)"
hybrid_leaf_bucket_count="$(read_field "${run_dir}/hybrid.report" leaf_bucket_count)"
fused_leaf_bucket_count="$(read_field "${run_dir}/fused.report" leaf_bucket_count)"
cpu_fused_leaf_bucket_count="$(read_field "${run_dir}/cpu-fused.report" leaf_bucket_count)"
hybrid_leaf_cells="$(read_field "${run_dir}/hybrid.report" leaf_cost_cells)"
fused_leaf_cells="$(read_field "${run_dir}/fused.report" leaf_cost_cells)"
cpu_leaf_cells="$(read_field "${run_dir}/cpu.report" leaf_cost_cells)"
cpu_fused_leaf_cells="$(read_field "${run_dir}/cpu-fused.report" leaf_cost_cells)"
cuda_leaf_cells="$(read_field "${run_dir}/cuda.report" leaf_cost_cells)"
cpu_leaf_tasks="$(read_field "${run_dir}/cpu.report" leaf_cost_tasks)"
cpu_fused_leaf_tasks="$(read_field "${run_dir}/cpu-fused.report" leaf_cost_tasks)"
cuda_leaf_tasks="$(read_field "${run_dir}/cuda.report" leaf_cost_tasks)"
hybrid_leaf_tasks="$(read_field "${run_dir}/hybrid.report" leaf_cost_tasks)"
fused_leaf_tasks="$(read_field "${run_dir}/fused.report" leaf_cost_tasks)"
if [[ "${cpu_leaf_tasks}" != "${cpu_fused_leaf_tasks}" ||
      "${cpu_leaf_tasks}" != "${cuda_leaf_tasks}" ||
      "${cpu_leaf_tasks}" != "${hybrid_leaf_tasks}" ||
      "${cpu_leaf_tasks}" != "${fused_leaf_tasks}" ||
      "${cpu_leaf_cells}" != "${cpu_fused_leaf_cells}" ||
      "${cpu_leaf_cells}" != "${cuda_leaf_cells}" ||
      "${cpu_leaf_cells}" != "${hybrid_leaf_cells}" ||
      "${cpu_leaf_cells}" != "${fused_leaf_cells}" ]]; then
  echo "leaf bucket 融合改变了 cost cell 工作量" >&2
  exit 1
fi
cpu_leaf_setup_ms="$(read_field "${run_dir}/cpu.report" leaf_setup_ms)"
cpu_leaf_proof_initialize_ms="$(read_field "${run_dir}/cpu.report" leaf_proof_initialize_ms)"
cpu_leaf_coverage_scan_ms="$(read_field "${run_dir}/cpu.report" leaf_coverage_scan_ms)"
cpu_leaf_cursor_construct_ms="$(read_field "${run_dir}/cpu.report" leaf_cursor_construct_ms)"
cpu_leaf_cursor_prepare_ms="$(read_field "${run_dir}/cpu.report" leaf_cursor_prepare_ms)"
cpu_leaf_cost_evaluate_ms="$(read_field "${run_dir}/cpu.report" leaf_cost_evaluate_ms)"
cpu_leaf_cost_cpu_certify_ms="$(read_field "${run_dir}/cpu.report" leaf_cost_cpu_certify_ms)"
cpu_leaf_cost_scatter_ms="$(read_field "${run_dir}/cpu.report" leaf_cost_scatter_ms)"
cpu_leaf_cursor_consume_ms="$(read_field "${run_dir}/cpu.report" leaf_cursor_consume_ms)"
cpu_leaf_candidate_recheck_ms="$(read_field "${run_dir}/cpu.report" leaf_candidate_recheck_ms)"
cpu_leaf_completeness_fallback_ms="$(read_field "${run_dir}/cpu.report" leaf_completeness_fallback_ms)"
cpu_leaf_scalar_search_ms="$(read_field "${run_dir}/cpu.report" leaf_scalar_search_ms)"
cpu_leaf_apply_ms="$(read_field "${run_dir}/cpu.report" leaf_apply_ms)"
cpu_leaf_proof_verify_ms="$(read_field "${run_dir}/cpu.report" leaf_proof_verify_ms)"
cpu_fused_leaf_setup_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_setup_ms)"
cpu_fused_leaf_proof_initialize_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_proof_initialize_ms)"
cpu_fused_leaf_coverage_scan_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_coverage_scan_ms)"
cpu_fused_leaf_cursor_construct_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_cursor_construct_ms)"
cpu_fused_leaf_cursor_prepare_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_cursor_prepare_ms)"
cpu_fused_leaf_cost_evaluate_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_cost_evaluate_ms)"
cpu_fused_leaf_cost_cpu_certify_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_cost_cpu_certify_ms)"
cpu_fused_leaf_cost_scatter_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_cost_scatter_ms)"
cpu_fused_leaf_cursor_consume_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_cursor_consume_ms)"
cpu_fused_leaf_candidate_recheck_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_candidate_recheck_ms)"
cpu_fused_leaf_completeness_fallback_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_completeness_fallback_ms)"
cpu_fused_leaf_scalar_search_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_scalar_search_ms)"
cpu_fused_leaf_apply_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_apply_ms)"
cpu_fused_leaf_proof_verify_ms="$(read_field "${run_dir}/cpu-fused.report" leaf_proof_verify_ms)"
cuda_leaf_setup_ms="$(read_field "${run_dir}/cuda.report" leaf_setup_ms)"
cuda_leaf_proof_initialize_ms="$(read_field "${run_dir}/cuda.report" leaf_proof_initialize_ms)"
cuda_leaf_coverage_scan_ms="$(read_field "${run_dir}/cuda.report" leaf_coverage_scan_ms)"
cuda_leaf_cursor_construct_ms="$(read_field "${run_dir}/cuda.report" leaf_cursor_construct_ms)"
cuda_leaf_cursor_prepare_ms="$(read_field "${run_dir}/cuda.report" leaf_cursor_prepare_ms)"
cuda_leaf_cost_evaluate_ms="$(read_field "${run_dir}/cuda.report" leaf_cost_evaluate_ms)"
cuda_leaf_cost_cpu_certify_ms="$(read_field "${run_dir}/cuda.report" leaf_cost_cpu_certify_ms)"
cuda_leaf_cost_scatter_ms="$(read_field "${run_dir}/cuda.report" leaf_cost_scatter_ms)"
cuda_leaf_cursor_consume_ms="$(read_field "${run_dir}/cuda.report" leaf_cursor_consume_ms)"
cuda_leaf_candidate_recheck_ms="$(read_field "${run_dir}/cuda.report" leaf_candidate_recheck_ms)"
cuda_leaf_completeness_fallback_ms="$(read_field "${run_dir}/cuda.report" leaf_completeness_fallback_ms)"
cuda_leaf_scalar_search_ms="$(read_field "${run_dir}/cuda.report" leaf_scalar_search_ms)"
cuda_leaf_apply_ms="$(read_field "${run_dir}/cuda.report" leaf_apply_ms)"
cuda_leaf_proof_verify_ms="$(read_field "${run_dir}/cuda.report" leaf_proof_verify_ms)"
hybrid_leaf_setup_ms="$(read_field "${run_dir}/hybrid.report" leaf_setup_ms)"
hybrid_leaf_proof_initialize_ms="$(read_field "${run_dir}/hybrid.report" leaf_proof_initialize_ms)"
hybrid_leaf_coverage_scan_ms="$(read_field "${run_dir}/hybrid.report" leaf_coverage_scan_ms)"
hybrid_leaf_cursor_construct_ms="$(read_field "${run_dir}/hybrid.report" leaf_cursor_construct_ms)"
hybrid_leaf_cursor_prepare_ms="$(read_field "${run_dir}/hybrid.report" leaf_cursor_prepare_ms)"
hybrid_leaf_cost_evaluate_ms="$(read_field "${run_dir}/hybrid.report" leaf_cost_evaluate_ms)"
hybrid_leaf_cost_cpu_certify_ms="$(read_field "${run_dir}/hybrid.report" leaf_cost_cpu_certify_ms)"
hybrid_leaf_cost_scatter_ms="$(read_field "${run_dir}/hybrid.report" leaf_cost_scatter_ms)"
hybrid_leaf_cursor_consume_ms="$(read_field "${run_dir}/hybrid.report" leaf_cursor_consume_ms)"
hybrid_leaf_candidate_recheck_ms="$(read_field "${run_dir}/hybrid.report" leaf_candidate_recheck_ms)"
hybrid_leaf_completeness_fallback_ms="$(read_field "${run_dir}/hybrid.report" leaf_completeness_fallback_ms)"
hybrid_leaf_scalar_search_ms="$(read_field "${run_dir}/hybrid.report" leaf_scalar_search_ms)"
hybrid_leaf_apply_ms="$(read_field "${run_dir}/hybrid.report" leaf_apply_ms)"
hybrid_leaf_proof_verify_ms="$(read_field "${run_dir}/hybrid.report" leaf_proof_verify_ms)"
fused_leaf_setup_ms="$(read_field "${run_dir}/fused.report" leaf_setup_ms)"
fused_leaf_proof_initialize_ms="$(read_field "${run_dir}/fused.report" leaf_proof_initialize_ms)"
fused_leaf_coverage_scan_ms="$(read_field "${run_dir}/fused.report" leaf_coverage_scan_ms)"
fused_leaf_cursor_construct_ms="$(read_field "${run_dir}/fused.report" leaf_cursor_construct_ms)"
fused_leaf_cursor_prepare_ms="$(read_field "${run_dir}/fused.report" leaf_cursor_prepare_ms)"
fused_leaf_cost_evaluate_ms="$(read_field "${run_dir}/fused.report" leaf_cost_evaluate_ms)"
fused_leaf_cost_cpu_certify_ms="$(read_field "${run_dir}/fused.report" leaf_cost_cpu_certify_ms)"
fused_leaf_cost_scatter_ms="$(read_field "${run_dir}/fused.report" leaf_cost_scatter_ms)"
fused_leaf_cursor_consume_ms="$(read_field "${run_dir}/fused.report" leaf_cursor_consume_ms)"
fused_leaf_candidate_recheck_ms="$(read_field "${run_dir}/fused.report" leaf_candidate_recheck_ms)"
fused_leaf_completeness_fallback_ms="$(read_field "${run_dir}/fused.report" leaf_completeness_fallback_ms)"
fused_leaf_scalar_search_ms="$(read_field "${run_dir}/fused.report" leaf_scalar_search_ms)"
fused_leaf_apply_ms="$(read_field "${run_dir}/fused.report" leaf_apply_ms)"
fused_leaf_proof_verify_ms="$(read_field "${run_dir}/fused.report" leaf_proof_verify_ms)"
cpu_leaf_cost_rows="$(read_field "${run_dir}/cpu.report" leaf_cost_rows_consumed)"
cpu_fused_leaf_cost_rows="$(read_field "${run_dir}/cpu-fused.report" leaf_cost_rows_consumed)"
cuda_leaf_cost_rows="$(read_field "${run_dir}/cuda.report" leaf_cost_rows_consumed)"
hybrid_leaf_cost_rows="$(read_field "${run_dir}/hybrid.report" leaf_cost_rows_consumed)"
fused_leaf_cost_rows="$(read_field "${run_dir}/fused.report" leaf_cost_rows_consumed)"
cpu_leaf_candidate_rechecks="$(read_field "${run_dir}/cpu.report" leaf_candidate_templates_rechecked)"
cpu_fused_leaf_candidate_rechecks="$(read_field "${run_dir}/cpu-fused.report" leaf_candidate_templates_rechecked)"
cuda_leaf_candidate_rechecks="$(read_field "${run_dir}/cuda.report" leaf_candidate_templates_rechecked)"
hybrid_leaf_candidate_rechecks="$(read_field "${run_dir}/hybrid.report" leaf_candidate_templates_rechecked)"
fused_leaf_candidate_rechecks="$(read_field "${run_dir}/fused.report" leaf_candidate_templates_rechecked)"
cpu_leaf_completeness_rows="$(read_field "${run_dir}/cpu.report" leaf_cpu_completeness_rows)"
cpu_fused_leaf_completeness_rows="$(read_field "${run_dir}/cpu-fused.report" leaf_cpu_completeness_rows)"
cuda_leaf_completeness_rows="$(read_field "${run_dir}/cuda.report" leaf_cpu_completeness_rows)"
hybrid_leaf_completeness_rows="$(read_field "${run_dir}/hybrid.report" leaf_cpu_completeness_rows)"
fused_leaf_completeness_rows="$(read_field "${run_dir}/fused.report" leaf_cpu_completeness_rows)"
cpu_leaf_completeness_templates="$(read_field "${run_dir}/cpu.report" leaf_cpu_completeness_templates)"
cpu_fused_leaf_completeness_templates="$(read_field "${run_dir}/cpu-fused.report" leaf_cpu_completeness_templates)"
cuda_leaf_completeness_templates="$(read_field "${run_dir}/cuda.report" leaf_cpu_completeness_templates)"
hybrid_leaf_completeness_templates="$(read_field "${run_dir}/hybrid.report" leaf_cpu_completeness_templates)"
fused_leaf_completeness_templates="$(read_field "${run_dir}/fused.report" leaf_cpu_completeness_templates)"
cpu_leaf_cpu_certified_cells="$(read_field "${run_dir}/cpu.report" leaf_cpu_certified_cost_cells)"
cpu_fused_leaf_cpu_certified_cells="$(read_field "${run_dir}/cpu-fused.report" leaf_cpu_certified_cost_cells)"
cuda_leaf_cpu_certified_cells="$(read_field "${run_dir}/cuda.report" leaf_cpu_certified_cost_cells)"
hybrid_leaf_cpu_certified_cells="$(read_field "${run_dir}/hybrid.report" leaf_cpu_certified_cost_cells)"
fused_leaf_cpu_certified_cells="$(read_field "${run_dir}/fused.report" leaf_cpu_certified_cost_cells)"
cpu_leaf_rows_scored="$(read_field "${run_dir}/cpu.report" leaf_cpu_cost_rows_scored)"
cpu_fused_leaf_rows_scored="$(read_field "${run_dir}/cpu-fused.report" leaf_cpu_cost_rows_scored)"
cuda_leaf_rows_scored="$(read_field "${run_dir}/cuda.report" leaf_cpu_cost_rows_scored)"
hybrid_leaf_rows_scored="$(read_field "${run_dir}/hybrid.report" leaf_cpu_cost_rows_scored)"
fused_leaf_rows_scored="$(read_field "${run_dir}/fused.report" leaf_cpu_cost_rows_scored)"
cpu_leaf_rows_reused="$(read_field "${run_dir}/cpu.report" leaf_cpu_cost_rows_reused)"
cpu_fused_leaf_rows_reused="$(read_field "${run_dir}/cpu-fused.report" leaf_cpu_cost_rows_reused)"
cuda_leaf_rows_reused="$(read_field "${run_dir}/cuda.report" leaf_cpu_cost_rows_reused)"
hybrid_leaf_rows_reused="$(read_field "${run_dir}/hybrid.report" leaf_cpu_cost_rows_reused)"
fused_leaf_rows_reused="$(read_field "${run_dir}/fused.report" leaf_cpu_cost_rows_reused)"
cpu_leaf_parallel_batches="$(read_field "${run_dir}/cpu.report" leaf_cpu_parallel_cost_batches)"
cpu_fused_leaf_parallel_batches="$(read_field "${run_dir}/cpu-fused.report" leaf_cpu_parallel_cost_batches)"
cuda_leaf_parallel_batches="$(read_field "${run_dir}/cuda.report" leaf_cpu_parallel_cost_batches)"
hybrid_leaf_parallel_batches="$(read_field "${run_dir}/hybrid.report" leaf_cpu_parallel_cost_batches)"
fused_leaf_parallel_batches="$(read_field "${run_dir}/fused.report" leaf_cpu_parallel_cost_batches)"
cpu_leaf_parallel_cells="$(read_field "${run_dir}/cpu.report" leaf_cpu_parallel_cost_cells)"
cpu_fused_leaf_parallel_cells="$(read_field "${run_dir}/cpu-fused.report" leaf_cpu_parallel_cost_cells)"
cuda_leaf_parallel_cells="$(read_field "${run_dir}/cuda.report" leaf_cpu_parallel_cost_cells)"
hybrid_leaf_parallel_cells="$(read_field "${run_dir}/hybrid.report" leaf_cpu_parallel_cost_cells)"
fused_leaf_parallel_cells="$(read_field "${run_dir}/fused.report" leaf_cpu_parallel_cost_cells)"
cpu_peak_leaf_threads="$(read_field "${run_dir}/cpu.report" peak_leaf_cpu_cost_threads)"
cpu_fused_peak_leaf_threads="$(read_field "${run_dir}/cpu-fused.report" peak_leaf_cpu_cost_threads)"
cuda_peak_leaf_threads="$(read_field "${run_dir}/cuda.report" peak_leaf_cpu_cost_threads)"
hybrid_peak_leaf_threads="$(read_field "${run_dir}/hybrid.report" peak_leaf_cpu_cost_threads)"
fused_peak_leaf_threads="$(read_field "${run_dir}/fused.report" peak_leaf_cpu_cost_threads)"
cpu_leaf_cursor_searches="$(read_field "${run_dir}/cpu.report" leaf_cursor_searches_started)"
cpu_fused_leaf_cursor_searches="$(read_field "${run_dir}/cpu-fused.report" leaf_cursor_searches_started)"
cuda_leaf_cursor_searches="$(read_field "${run_dir}/cuda.report" leaf_cursor_searches_started)"
hybrid_leaf_cursor_searches="$(read_field "${run_dir}/hybrid.report" leaf_cursor_searches_started)"
fused_leaf_cursor_searches="$(read_field "${run_dir}/fused.report" leaf_cursor_searches_started)"
if [[ "${cpu_leaf_cursor_searches}" != "${cpu_fused_leaf_cursor_searches}" ||
      "${cpu_leaf_cursor_searches}" != "${cuda_leaf_cursor_searches}" ||
      "${cpu_leaf_cursor_searches}" != "${hybrid_leaf_cursor_searches}" ||
      "${cpu_leaf_cursor_searches}" != "${fused_leaf_cursor_searches}" ||
      "${cpu_leaf_cost_rows}" != "${cpu_fused_leaf_cost_rows}" ||
      "${cpu_leaf_cost_rows}" != "${cuda_leaf_cost_rows}" ||
      "${cpu_leaf_cost_rows}" != "${hybrid_leaf_cost_rows}" ||
      "${cuda_leaf_cost_rows}" != "${fused_leaf_cost_rows}" ||
      "${cpu_leaf_candidate_rechecks}" != "${cuda_leaf_candidate_rechecks}" ||
      "${cpu_leaf_candidate_rechecks}" != "${hybrid_leaf_candidate_rechecks}" ||
      "${cuda_leaf_candidate_rechecks}" != "${hybrid_leaf_candidate_rechecks}" ||
      "${cuda_leaf_candidate_rechecks}" != "${fused_leaf_candidate_rechecks}" ||
      "${cpu_leaf_candidate_rechecks}" != "${cpu_fused_leaf_candidate_rechecks}" ||
      "${cpu_leaf_completeness_rows}" != "${cuda_leaf_completeness_rows}" ||
      "${cuda_leaf_completeness_rows}" != "${hybrid_leaf_completeness_rows}" ||
      "${cuda_leaf_completeness_rows}" != "${fused_leaf_completeness_rows}" ||
      "${cpu_leaf_completeness_rows}" != "${cpu_fused_leaf_completeness_rows}" ||
      "${cpu_leaf_completeness_templates}" != "${cuda_leaf_completeness_templates}" ||
      "${cuda_leaf_completeness_templates}" != "${hybrid_leaf_completeness_templates}" ||
      "${cuda_leaf_completeness_templates}" != "${fused_leaf_completeness_templates}" ||
      "${cpu_leaf_completeness_templates}" != "${cpu_fused_leaf_completeness_templates}" ||
      "${cpu_leaf_cpu_certified_cells}" != "${cuda_leaf_cpu_certified_cells}" ||
      "${cuda_leaf_cpu_certified_cells}" != "${hybrid_leaf_cpu_certified_cells}" ||
      "${cuda_leaf_cpu_certified_cells}" != "${fused_leaf_cpu_certified_cells}" ||
      "${cpu_leaf_cpu_certified_cells}" != "${cpu_fused_leaf_cpu_certified_cells}" ||
      "${cpu_leaf_cpu_certified_cells}" != "${cpu_leaf_cells}" ]]; then
  echo "五路 leaf consume 的规范工作计数不一致" >&2
  exit 1
fi
if (( cpu_leaf_rows_scored + cpu_leaf_rows_reused != cpu_leaf_tasks ||
      cpu_fused_leaf_rows_scored + cpu_fused_leaf_rows_reused != cpu_fused_leaf_tasks ||
      cuda_leaf_rows_scored + cuda_leaf_rows_reused != cuda_leaf_tasks ||
      hybrid_leaf_rows_scored + hybrid_leaf_rows_reused != hybrid_leaf_tasks ||
      fused_leaf_rows_scored + fused_leaf_rows_reused != fused_leaf_tasks )); then
  echo "五路 CPU cost row 评分/复用计数不闭合" >&2
  exit 1
fi
cpu_path_append_ms="$(read_field "${run_dir}/cpu.report" path_append_ms)"
cpu_fused_path_append_ms="$(read_field "${run_dir}/cpu-fused.report" path_append_ms)"
cuda_path_append_ms="$(read_field "${run_dir}/cuda.report" path_append_ms)"
hybrid_path_append_ms="$(read_field "${run_dir}/hybrid.report" path_append_ms)"
fused_path_append_ms="$(read_field "${run_dir}/fused.report" path_append_ms)"
cpu_path_append_parent_prepare_ms="$(read_field "${run_dir}/cpu.report" path_append_parent_prepare_ms)"
cpu_fused_path_append_parent_prepare_ms="$(read_field "${run_dir}/cpu-fused.report" path_append_parent_prepare_ms)"
cuda_path_append_parent_prepare_ms="$(read_field "${run_dir}/cuda.report" path_append_parent_prepare_ms)"
hybrid_path_append_parent_prepare_ms="$(read_field "${run_dir}/hybrid.report" path_append_parent_prepare_ms)"
fused_path_append_parent_prepare_ms="$(read_field "${run_dir}/fused.report" path_append_parent_prepare_ms)"
cpu_path_append_child_normalize_ms="$(read_field "${run_dir}/cpu.report" path_append_child_normalize_ms)"
cpu_fused_path_append_child_normalize_ms="$(read_field "${run_dir}/cpu-fused.report" path_append_child_normalize_ms)"
cuda_path_append_child_normalize_ms="$(read_field "${run_dir}/cuda.report" path_append_child_normalize_ms)"
hybrid_path_append_child_normalize_ms="$(read_field "${run_dir}/hybrid.report" path_append_child_normalize_ms)"
fused_path_append_child_normalize_ms="$(read_field "${run_dir}/fused.report" path_append_child_normalize_ms)"
cpu_path_append_child_edges_ms="$(read_field "${run_dir}/cpu.report" path_append_child_edges_ms)"
cpu_fused_path_append_child_edges_ms="$(read_field "${run_dir}/cpu-fused.report" path_append_child_edges_ms)"
cuda_path_append_child_edges_ms="$(read_field "${run_dir}/cuda.report" path_append_child_edges_ms)"
hybrid_path_append_child_edges_ms="$(read_field "${run_dir}/hybrid.report" path_append_child_edges_ms)"
fused_path_append_child_edges_ms="$(read_field "${run_dir}/fused.report" path_append_child_edges_ms)"
cpu_path_append_cuda_evaluate_ms="$(read_field "${run_dir}/cpu.report" path_append_cuda_evaluate_ms)"
cpu_fused_path_append_cuda_evaluate_ms="$(read_field "${run_dir}/cpu-fused.report" path_append_cuda_evaluate_ms)"
cuda_path_append_cuda_evaluate_ms="$(read_field "${run_dir}/cuda.report" path_append_cuda_evaluate_ms)"
hybrid_path_append_cuda_evaluate_ms="$(read_field "${run_dir}/hybrid.report" path_append_cuda_evaluate_ms)"
fused_path_append_cuda_evaluate_ms="$(read_field "${run_dir}/fused.report" path_append_cuda_evaluate_ms)"
cpu_path_append_cuda_compare_ms="$(read_field "${run_dir}/cpu.report" path_append_cuda_compare_ms)"
cpu_fused_path_append_cuda_compare_ms="$(read_field "${run_dir}/cpu-fused.report" path_append_cuda_compare_ms)"
cuda_path_append_cuda_compare_ms="$(read_field "${run_dir}/cuda.report" path_append_cuda_compare_ms)"
hybrid_path_append_cuda_compare_ms="$(read_field "${run_dir}/hybrid.report" path_append_cuda_compare_ms)"
fused_path_append_cuda_compare_ms="$(read_field "${run_dir}/fused.report" path_append_cuda_compare_ms)"
cpu_hamilton_reply_ms="$(read_field "${run_dir}/cpu.report" hamilton_reply_ms)"
cpu_fused_hamilton_reply_ms="$(read_field "${run_dir}/cpu-fused.report" hamilton_reply_ms)"
cuda_hamilton_reply_ms="$(read_field "${run_dir}/cuda.report" hamilton_reply_ms)"
hybrid_hamilton_reply_ms="$(read_field "${run_dir}/hybrid.report" hamilton_reply_ms)"
fused_hamilton_reply_ms="$(read_field "${run_dir}/fused.report" hamilton_reply_ms)"
cpu_hamilton_reply_validation_ms="$(read_field "${run_dir}/cpu.report" hamilton_reply_validation_ms)"
cuda_hamilton_reply_validation_ms="$(read_field "${run_dir}/cuda.report" hamilton_reply_validation_ms)"
hybrid_hamilton_reply_validation_ms="$(read_field "${run_dir}/hybrid.report" hamilton_reply_validation_ms)"
fused_hamilton_reply_validation_ms="$(read_field "${run_dir}/fused.report" hamilton_reply_validation_ms)"
cpu_hamilton_reply_enumerate_ms="$(read_field "${run_dir}/cpu.report" hamilton_reply_cpu_enumerate_ms)"
cuda_hamilton_reply_enumerate_ms="$(read_field "${run_dir}/cuda.report" hamilton_reply_cpu_enumerate_ms)"
hybrid_hamilton_reply_enumerate_ms="$(read_field "${run_dir}/hybrid.report" hamilton_reply_cpu_enumerate_ms)"
fused_hamilton_reply_enumerate_ms="$(read_field "${run_dir}/fused.report" hamilton_reply_cpu_enumerate_ms)"
cpu_hamilton_reply_cuda_ms="$(read_field "${run_dir}/cpu.report" hamilton_reply_cuda_evaluate_ms)"
cuda_hamilton_reply_cuda_ms="$(read_field "${run_dir}/cuda.report" hamilton_reply_cuda_evaluate_ms)"
hybrid_hamilton_reply_cuda_ms="$(read_field "${run_dir}/hybrid.report" hamilton_reply_cuda_evaluate_ms)"
fused_hamilton_reply_cuda_ms="$(read_field "${run_dir}/fused.report" hamilton_reply_cuda_evaluate_ms)"
cpu_hamilton_reply_compare_ms="$(read_field "${run_dir}/cpu.report" hamilton_reply_cuda_compare_ms)"
cuda_hamilton_reply_compare_ms="$(read_field "${run_dir}/cuda.report" hamilton_reply_cuda_compare_ms)"
hybrid_hamilton_reply_compare_ms="$(read_field "${run_dir}/hybrid.report" hamilton_reply_cuda_compare_ms)"
fused_hamilton_reply_compare_ms="$(read_field "${run_dir}/fused.report" hamilton_reply_cuda_compare_ms)"
cpu_hamilton_reply_batches="$(read_field "${run_dir}/cpu.report" hamilton_reply_batches)"
cpu_fused_hamilton_reply_batches="$(read_field "${run_dir}/cpu-fused.report" hamilton_reply_batches)"
cuda_hamilton_reply_batches="$(read_field "${run_dir}/cuda.report" hamilton_reply_batches)"
hybrid_hamilton_reply_batches="$(read_field "${run_dir}/hybrid.report" hamilton_reply_batches)"
fused_hamilton_reply_batches="$(read_field "${run_dir}/fused.report" hamilton_reply_batches)"
cpu_hamilton_reply_centers="$(read_field "${run_dir}/cpu.report" hamilton_reply_centers)"
cpu_fused_hamilton_reply_centers="$(read_field "${run_dir}/cpu-fused.report" hamilton_reply_centers)"
cuda_hamilton_reply_centers="$(read_field "${run_dir}/cuda.report" hamilton_reply_centers)"
hybrid_hamilton_reply_centers="$(read_field "${run_dir}/hybrid.report" hamilton_reply_centers)"
fused_hamilton_reply_centers="$(read_field "${run_dir}/fused.report" hamilton_reply_centers)"
cpu_hamilton_reply_unique_centers="$(read_field "${run_dir}/cpu.report" hamilton_reply_unique_centers)"
cpu_fused_hamilton_reply_unique_centers="$(read_field "${run_dir}/cpu-fused.report" hamilton_reply_unique_centers)"
cuda_hamilton_reply_unique_centers="$(read_field "${run_dir}/cuda.report" hamilton_reply_unique_centers)"
hybrid_hamilton_reply_unique_centers="$(read_field "${run_dir}/hybrid.report" hamilton_reply_unique_centers)"
fused_hamilton_reply_unique_centers="$(read_field "${run_dir}/fused.report" hamilton_reply_unique_centers)"
cpu_hamilton_reply_pairs="$(read_field "${run_dir}/cpu.report" hamilton_reply_neighbor_pairs_tested)"
cpu_fused_hamilton_reply_pairs="$(read_field "${run_dir}/cpu-fused.report" hamilton_reply_neighbor_pairs_tested)"
cuda_hamilton_reply_pairs="$(read_field "${run_dir}/cuda.report" hamilton_reply_neighbor_pairs_tested)"
hybrid_hamilton_reply_pairs="$(read_field "${run_dir}/hybrid.report" hamilton_reply_neighbor_pairs_tested)"
fused_hamilton_reply_pairs="$(read_field "${run_dir}/fused.report" hamilton_reply_neighbor_pairs_tested)"
cpu_hamilton_replies_generated="$(read_field "${run_dir}/cpu.report" hamilton_replies_generated)"
cpu_fused_hamilton_replies_generated="$(read_field "${run_dir}/cpu-fused.report" hamilton_replies_generated)"
cuda_hamilton_replies_generated="$(read_field "${run_dir}/cuda.report" hamilton_replies_generated)"
hybrid_hamilton_replies_generated="$(read_field "${run_dir}/hybrid.report" hamilton_replies_generated)"
fused_hamilton_replies_generated="$(read_field "${run_dir}/fused.report" hamilton_replies_generated)"
if [[ "${cpu_hamilton_reply_batches}" != "${cpu_fused_hamilton_reply_batches}" ||
      "${cpu_hamilton_reply_batches}" != "${cuda_hamilton_reply_batches}" ||
      "${cpu_hamilton_reply_batches}" != "${hybrid_hamilton_reply_batches}" ||
      "${cpu_hamilton_reply_batches}" != "${fused_hamilton_reply_batches}" ||
      "${cpu_hamilton_reply_centers}" != "${cuda_hamilton_reply_centers}" ||
      "${cpu_hamilton_reply_centers}" != "${hybrid_hamilton_reply_centers}" ||
      "${cpu_hamilton_reply_centers}" != "${fused_hamilton_reply_centers}" ||
      "${cpu_hamilton_reply_centers}" != "${cpu_fused_hamilton_reply_centers}" ||
      "${cpu_hamilton_reply_unique_centers}" != "${cuda_hamilton_reply_unique_centers}" ||
      "${cpu_hamilton_reply_unique_centers}" != "${hybrid_hamilton_reply_unique_centers}" ||
      "${cpu_hamilton_reply_unique_centers}" != "${fused_hamilton_reply_unique_centers}" ||
      "${cpu_hamilton_reply_unique_centers}" != "${cpu_fused_hamilton_reply_unique_centers}" ||
      "${cpu_hamilton_reply_pairs}" != "${cuda_hamilton_reply_pairs}" ||
      "${cpu_hamilton_reply_pairs}" != "${hybrid_hamilton_reply_pairs}" ||
      "${cpu_hamilton_reply_pairs}" != "${fused_hamilton_reply_pairs}" ||
      "${cpu_hamilton_reply_pairs}" != "${cpu_fused_hamilton_reply_pairs}" ||
      "${cpu_hamilton_replies_generated}" != "${cuda_hamilton_replies_generated}" ||
      "${cpu_hamilton_replies_generated}" != "${hybrid_hamilton_replies_generated}" ||
      "${cpu_hamilton_replies_generated}" != "${fused_hamilton_replies_generated}" ||
      "${cpu_hamilton_replies_generated}" != "${cpu_fused_hamilton_replies_generated}" ]]; then
  echo "五路 Hamilton reply 的规范工作计数不一致" >&2
  exit 1
fi
cpu_reply_cuda_batches="$(read_field "${run_dir}/cpu.report" reply_cuda_batches)"
cpu_fused_reply_cuda_batches="$(read_field "${run_dir}/cpu-fused.report" reply_cuda_batches)"
cuda_reply_cuda_batches="$(read_field "${run_dir}/cuda.report" reply_cuda_batches)"
cpu_reply_cuda_tasks_submitted="$(read_field "${run_dir}/cpu.report" reply_cuda_tasks_submitted)"
cpu_fused_reply_cuda_tasks_submitted="$(read_field "${run_dir}/cpu-fused.report" reply_cuda_tasks_submitted)"
cuda_reply_cuda_tasks_submitted="$(read_field "${run_dir}/cuda.report" reply_cuda_tasks_submitted)"
hybrid_reply_cuda_batches="$(read_field "${run_dir}/hybrid.report" reply_cuda_batches)"
fused_reply_cuda_batches="$(read_field "${run_dir}/fused.report" reply_cuda_batches)"
hybrid_reply_cuda_tasks_submitted="$(read_field "${run_dir}/hybrid.report" reply_cuda_tasks_submitted)"
fused_reply_cuda_tasks_submitted="$(read_field "${run_dir}/fused.report" reply_cuda_tasks_submitted)"
cuda_reply_graph_cache_hits="$(read_field "${run_dir}/cuda.report" reply_cuda_graph_cache_hits)"
cuda_reply_workspace_cache_hits="$(read_field "${run_dir}/cuda.report" reply_cuda_workspace_cache_hits)"
cuda_peak_reply_device_cache_bytes="$(read_field "${run_dir}/cuda.report" peak_reply_device_cache_bytes)"
if [[ "${cpu_reply_cuda_batches}" != "0" || "${cpu_fused_reply_cuda_batches}" != "0" ||
      "${hybrid_reply_cuda_batches}" != "0" || "${fused_reply_cuda_batches}" != "0" ||
      "${cpu_reply_cuda_tasks_submitted}" != "0" ||
      "${cpu_fused_reply_cuda_tasks_submitted}" != "0" ||
      "${hybrid_reply_cuda_tasks_submitted}" != "0" ||
      "${fused_reply_cuda_tasks_submitted}" != "0" ||
      "${cuda_reply_cuda_batches}" == "0" || "${cuda_peak_reply_device_cache_bytes}" == "0" ||
      "${cuda_reply_workspace_cache_hits}" -gt "${cuda_reply_cuda_batches}" ]]; then
  echo "HT reply CUDA 驻留缓存计数非法" >&2
  exit 1
fi
cpu_end_reply_batches="$(read_field "${run_dir}/cpu.report" end_reply_batches)"
cpu_fused_end_reply_batches="$(read_field "${run_dir}/cpu-fused.report" end_reply_batches)"
cuda_end_reply_batches="$(read_field "${run_dir}/cuda.report" end_reply_batches)"
hybrid_end_reply_batches="$(read_field "${run_dir}/hybrid.report" end_reply_batches)"
fused_end_reply_batches="$(read_field "${run_dir}/fused.report" end_reply_batches)"
cpu_end_reply_tasks="$(read_field "${run_dir}/cpu.report" end_reply_tasks)"
cpu_fused_end_reply_tasks="$(read_field "${run_dir}/cpu-fused.report" end_reply_tasks)"
cuda_end_reply_tasks="$(read_field "${run_dir}/cuda.report" end_reply_tasks)"
hybrid_end_reply_tasks="$(read_field "${run_dir}/hybrid.report" end_reply_tasks)"
fused_end_reply_tasks="$(read_field "${run_dir}/fused.report" end_reply_tasks)"
cpu_end_reply_unique_tasks="$(read_field "${run_dir}/cpu.report" end_reply_unique_tasks)"
cpu_fused_end_reply_unique_tasks="$(read_field "${run_dir}/cpu-fused.report" end_reply_unique_tasks)"
cuda_end_reply_unique_tasks="$(read_field "${run_dir}/cuda.report" end_reply_unique_tasks)"
hybrid_end_reply_unique_tasks="$(read_field "${run_dir}/hybrid.report" end_reply_unique_tasks)"
fused_end_reply_unique_tasks="$(read_field "${run_dir}/fused.report" end_reply_unique_tasks)"
cpu_end_replies_generated="$(read_field "${run_dir}/cpu.report" end_replies_generated)"
cpu_fused_end_replies_generated="$(read_field "${run_dir}/cpu-fused.report" end_replies_generated)"
cuda_end_replies_generated="$(read_field "${run_dir}/cuda.report" end_replies_generated)"
hybrid_end_replies_generated="$(read_field "${run_dir}/hybrid.report" end_replies_generated)"
fused_end_replies_generated="$(read_field "${run_dir}/fused.report" end_replies_generated)"
if [[ "${cpu_end_reply_batches}" != "${cpu_fused_end_reply_batches}" ||
      "${cpu_end_reply_batches}" != "${cuda_end_reply_batches}" ||
      "${cpu_end_reply_batches}" != "${hybrid_end_reply_batches}" ||
      "${cpu_end_reply_batches}" != "${fused_end_reply_batches}" ||
      "${cpu_end_reply_tasks}" != "${cpu_fused_end_reply_tasks}" ||
      "${cpu_end_reply_tasks}" != "${cuda_end_reply_tasks}" ||
      "${cpu_end_reply_tasks}" != "${hybrid_end_reply_tasks}" ||
      "${cpu_end_reply_tasks}" != "${fused_end_reply_tasks}" ||
      "${cpu_end_reply_unique_tasks}" != "${cpu_fused_end_reply_unique_tasks}" ||
      "${cpu_end_reply_unique_tasks}" != "${cuda_end_reply_unique_tasks}" ||
      "${cpu_end_reply_unique_tasks}" != "${hybrid_end_reply_unique_tasks}" ||
      "${cpu_end_reply_unique_tasks}" != "${fused_end_reply_unique_tasks}" ||
      "${cpu_end_replies_generated}" != "${cpu_fused_end_replies_generated}" ||
      "${cpu_end_replies_generated}" != "${cuda_end_replies_generated}" ||
      "${cpu_end_replies_generated}" != "${hybrid_end_replies_generated}" ||
      "${cpu_end_replies_generated}" != "${fused_end_replies_generated}" ||
      "${cuda_end_reply_unique_tasks}" -gt "${cuda_end_reply_tasks}" ]]; then
  echo "五路 end reply 的规范工作计数不一致" >&2
  exit 1
fi
if [[ "${deduplicate_reply_tasks}" == "1" ]]; then
  expected_cuda_reply_tasks=$((cuda_hamilton_reply_unique_centers + cuda_end_reply_unique_tasks))
else
  expected_cuda_reply_tasks=$((cuda_hamilton_reply_centers + cuda_end_reply_tasks))
fi
if [[ "${cuda_reply_cuda_tasks_submitted}" != "${expected_cuda_reply_tasks}" ]]; then
  echo "HT reply CUDA 物理任务计数与去重配置不一致" >&2
  exit 1
fi
if [[ "${reuse_reply_cuda_cache}" == "1" ]]; then
  if (( cuda_reply_graph_cache_hits + 1 != cuda_reply_cuda_batches )); then
    echo "HT reply CUDA 驻留图没有保持一批冷启动、后续全命中" >&2
    exit 1
  fi
elif [[ "${cuda_reply_graph_cache_hits}" != "0" ||
        "${cuda_reply_workspace_cache_hits}" != "0" ]]; then
  echo "禁用 HT reply CUDA 驻留缓存后仍报告命中" >&2
  exit 1
fi
cpu_end_reply_ms="$(read_field "${run_dir}/cpu.report" end_reply_ms)"
cpu_fused_end_reply_ms="$(read_field "${run_dir}/cpu-fused.report" end_reply_ms)"
cuda_end_reply_ms="$(read_field "${run_dir}/cuda.report" end_reply_ms)"
hybrid_end_reply_ms="$(read_field "${run_dir}/hybrid.report" end_reply_ms)"
fused_end_reply_ms="$(read_field "${run_dir}/fused.report" end_reply_ms)"
cpu_propagation_ms="$(read_field "${run_dir}/cpu.report" propagation_ms)"
cuda_propagation_ms="$(read_field "${run_dir}/cuda.report" propagation_ms)"
hybrid_propagation_ms="$(read_field "${run_dir}/hybrid.report" propagation_ms)"
cpu_proof_extract_ms="$(read_field "${run_dir}/cpu.report" proof_extract_ms)"
cuda_proof_extract_ms="$(read_field "${run_dir}/cuda.report" proof_extract_ms)"
hybrid_proof_extract_ms="$(read_field "${run_dir}/hybrid.report" proof_extract_ms)"
cpu_proof_verify_ms="$(read_field "${run_dir}/cpu.report" proof_verify_ms)"
cuda_proof_verify_ms="$(read_field "${run_dir}/cuda.report" proof_verify_ms)"
hybrid_proof_verify_ms="$(read_field "${run_dir}/hybrid.report" proof_verify_ms)"
cpu_immediate_verify_ms="$(read_field "${run_dir}/cpu.report" immediate_verify_ms)"
cuda_immediate_verify_ms="$(read_field "${run_dir}/cuda.report" immediate_verify_ms)"
hybrid_immediate_verify_ms="$(read_field "${run_dir}/hybrid.report" immediate_verify_ms)"
cpu_commit_ms="$(read_field "${run_dir}/cpu.report" commit_ms)"
cuda_commit_ms="$(read_field "${run_dir}/cuda.report" commit_ms)"
hybrid_commit_ms="$(read_field "${run_dir}/hybrid.report" commit_ms)"
cpu_total_ms="$(read_field "${run_dir}/cpu.report" total_ms)"
cpu_fused_total_ms="$(read_field "${run_dir}/cpu-fused.report" total_ms)"
cuda_total_ms="$(read_field "${run_dir}/cuda.report" total_ms)"
hybrid_total_ms="$(read_field "${run_dir}/hybrid.report" total_ms)"
fused_total_ms="$(read_field "${run_dir}/fused.report" total_ms)"
cpu_wall_ms="$(<"${run_dir}/cpu.wall-ms")"
cpu_fused_wall_ms="$(<"${run_dir}/cpu-fused.wall-ms")"
cuda_wall_ms="$(<"${run_dir}/cuda.wall-ms")"
hybrid_wall_ms="$(<"${run_dir}/hybrid.wall-ms")"
fused_wall_ms="$(<"${run_dir}/fused.wall-ms")"
attempted="$(read_field "${run_dir}/cuda.report" attempted_targets)"
proven="$(read_field "${run_dir}/cuda.report" proven_targets)"
unresolved="$(read_field "${run_dir}/cuda.report" unresolved_targets)"
committed="$(read_field "${run_dir}/cuda.report" committed_targets)"
states="$(read_field "${run_dir}/cuda.report" states_expanded)"
replies="$(read_field "${run_dir}/cuda.report" replies_expanded)"
leaf_calls="$(read_field "${run_dir}/cuda.report" leaf_calls)"
leaf_cells="$(read_field "${run_dir}/cuda.report" leaf_cost_cells)"
peak_cache="$(read_field "${run_dir}/cuda.report" peak_leaf_device_cache_bytes)"
initial_hash="$(read_field "${run_dir}/cuda.report" initial_hash)"
final_hash="$(read_field "${run_dir}/cuda.report" final_hash)"
search_speedup="$(awk -v cpu="${cpu_search_ms}" -v cuda="${cuda_search_ms}" 'BEGIN { printf "%.3f", cpu/cuda }')"
cpu_fused_search_speedup="$(awk -v cpu="${cpu_search_ms}" -v fused="${cpu_fused_search_ms}" 'BEGIN { printf "%.3f", cpu/fused }')"
wall_speedup="$(awk -v cpu="${cpu_wall_ms}" -v cuda="${cuda_wall_ms}" 'BEGIN { printf "%.3f", cpu/cuda }')"
cpu_fused_wall_speedup="$(awk -v cpu="${cpu_wall_ms}" -v fused="${cpu_fused_wall_ms}" 'BEGIN { printf "%.3f", cpu/fused }')"
leaf_speedup="$(awk -v cpu="${cpu_leaf_ms}" -v cuda="${cuda_leaf_ms}" 'BEGIN { printf "%.3f", cpu/cuda }')"
cpu_fused_leaf_speedup="$(awk -v cpu="${cpu_leaf_ms}" -v fused="${cpu_fused_leaf_ms}" 'BEGIN { printf "%.3f", cpu/fused }')"
hybrid_search_speedup="$(awk -v cpu="${cpu_search_ms}" -v hybrid="${hybrid_search_ms}" 'BEGIN { printf "%.3f", cpu/hybrid }')"
hybrid_wall_speedup="$(awk -v cpu="${cpu_wall_ms}" -v hybrid="${hybrid_wall_ms}" 'BEGIN { printf "%.3f", cpu/hybrid }')"
hybrid_leaf_speedup="$(awk -v cpu="${cpu_leaf_ms}" -v hybrid="${hybrid_leaf_ms}" 'BEGIN { printf "%.3f", cpu/hybrid }')"
fused_search_speedup="$(awk -v cpu="${cpu_search_ms}" -v fused="${fused_search_ms}" 'BEGIN { printf "%.3f", cpu/fused }')"
fused_wall_speedup="$(awk -v cpu="${cpu_wall_ms}" -v fused="${fused_wall_ms}" 'BEGIN { printf "%.3f", cpu/fused }')"
fused_leaf_speedup="$(awk -v hybrid="${hybrid_leaf_ms}" -v fused="${fused_leaf_ms}" 'BEGIN { printf "%.3f", hybrid/fused }')"
cpu_leaf_residual_ms="$(awk -v total="${cpu_leaf_ms}" -v setup="${cpu_leaf_setup_ms}" \
  -v prepare="${cpu_leaf_cursor_prepare_ms}" -v cost="${cpu_leaf_cost_evaluate_ms}" \
  -v scatter="${cpu_leaf_cost_scatter_ms}" -v consume="${cpu_leaf_cursor_consume_ms}" \
  -v scalar="${cpu_leaf_scalar_search_ms}" -v apply="${cpu_leaf_apply_ms}" \
  -v verify="${cpu_leaf_proof_verify_ms}" \
  'BEGIN { printf "%.6f", total-setup-prepare-cost-scatter-consume-scalar-apply-verify }')"
cpu_fused_leaf_residual_ms="$(awk -v total="${cpu_fused_leaf_ms}" \
  -v setup="${cpu_fused_leaf_setup_ms}" -v prepare="${cpu_fused_leaf_cursor_prepare_ms}" \
  -v cost="${cpu_fused_leaf_cost_evaluate_ms}" -v scatter="${cpu_fused_leaf_cost_scatter_ms}" \
  -v consume="${cpu_fused_leaf_cursor_consume_ms}" \
  -v scalar="${cpu_fused_leaf_scalar_search_ms}" -v apply="${cpu_fused_leaf_apply_ms}" \
  -v verify="${cpu_fused_leaf_proof_verify_ms}" \
  'BEGIN { printf "%.6f", total-setup-prepare-cost-scatter-consume-scalar-apply-verify }')"
cuda_leaf_residual_ms="$(awk -v total="${cuda_leaf_ms}" -v setup="${cuda_leaf_setup_ms}" \
  -v prepare="${cuda_leaf_cursor_prepare_ms}" -v cost="${cuda_leaf_cost_evaluate_ms}" \
  -v scatter="${cuda_leaf_cost_scatter_ms}" -v consume="${cuda_leaf_cursor_consume_ms}" \
  -v scalar="${cuda_leaf_scalar_search_ms}" -v apply="${cuda_leaf_apply_ms}" \
  -v verify="${cuda_leaf_proof_verify_ms}" \
  'BEGIN { printf "%.6f", total-setup-prepare-cost-scatter-consume-scalar-apply-verify }')"
hybrid_leaf_residual_ms="$(awk -v total="${hybrid_leaf_ms}" -v setup="${hybrid_leaf_setup_ms}" \
  -v prepare="${hybrid_leaf_cursor_prepare_ms}" -v cost="${hybrid_leaf_cost_evaluate_ms}" \
  -v scatter="${hybrid_leaf_cost_scatter_ms}" -v consume="${hybrid_leaf_cursor_consume_ms}" \
  -v scalar="${hybrid_leaf_scalar_search_ms}" -v apply="${hybrid_leaf_apply_ms}" \
  -v verify="${hybrid_leaf_proof_verify_ms}" \
  'BEGIN { printf "%.6f", total-setup-prepare-cost-scatter-consume-scalar-apply-verify }')"
fused_leaf_residual_ms="$(awk -v total="${fused_leaf_ms}" -v setup="${fused_leaf_setup_ms}" \
  -v prepare="${fused_leaf_cursor_prepare_ms}" -v cost="${fused_leaf_cost_evaluate_ms}" \
  -v scatter="${fused_leaf_cost_scatter_ms}" -v consume="${fused_leaf_cursor_consume_ms}" \
  -v scalar="${fused_leaf_scalar_search_ms}" -v apply="${fused_leaf_apply_ms}" \
  -v verify="${fused_leaf_proof_verify_ms}" \
  'BEGIN { printf "%.6f", total-setup-prepare-cost-scatter-consume-scalar-apply-verify }')"
cpu_leaf_setup_residual_ms="$(awk -v total="${cpu_leaf_setup_ms}" \
  -v initialize="${cpu_leaf_proof_initialize_ms}" -v coverage="${cpu_leaf_coverage_scan_ms}" \
  -v cursor="${cpu_leaf_cursor_construct_ms}" \
  'BEGIN { printf "%.6f", total-initialize-coverage-cursor }')"
cpu_fused_leaf_setup_residual_ms="$(awk -v total="${cpu_fused_leaf_setup_ms}" \
  -v initialize="${cpu_fused_leaf_proof_initialize_ms}" \
  -v coverage="${cpu_fused_leaf_coverage_scan_ms}" \
  -v cursor="${cpu_fused_leaf_cursor_construct_ms}" \
  'BEGIN { printf "%.6f", total-initialize-coverage-cursor }')"
cuda_leaf_setup_residual_ms="$(awk -v total="${cuda_leaf_setup_ms}" \
  -v initialize="${cuda_leaf_proof_initialize_ms}" -v coverage="${cuda_leaf_coverage_scan_ms}" \
  -v cursor="${cuda_leaf_cursor_construct_ms}" \
  'BEGIN { printf "%.6f", total-initialize-coverage-cursor }')"
hybrid_leaf_setup_residual_ms="$(awk -v total="${hybrid_leaf_setup_ms}" \
  -v initialize="${hybrid_leaf_proof_initialize_ms}" \
  -v coverage="${hybrid_leaf_coverage_scan_ms}" -v cursor="${hybrid_leaf_cursor_construct_ms}" \
  'BEGIN { printf "%.6f", total-initialize-coverage-cursor }')"
fused_leaf_setup_residual_ms="$(awk -v total="${fused_leaf_setup_ms}" \
  -v initialize="${fused_leaf_proof_initialize_ms}" -v coverage="${fused_leaf_coverage_scan_ms}" \
  -v cursor="${fused_leaf_cursor_construct_ms}" \
  'BEGIN { printf "%.6f", total-initialize-coverage-cursor }')"
cuda_leaf_consume_residual_ms="$(awk -v total="${cuda_leaf_cursor_consume_ms}" \
  -v candidate="${cuda_leaf_candidate_recheck_ms}" \
  -v completeness="${cuda_leaf_completeness_fallback_ms}" \
  'BEGIN { printf "%.6f", total-candidate-completeness }')"
hybrid_leaf_consume_residual_ms="$(awk -v total="${hybrid_leaf_cursor_consume_ms}" \
  -v candidate="${hybrid_leaf_candidate_recheck_ms}" \
  -v completeness="${hybrid_leaf_completeness_fallback_ms}" \
  'BEGIN { printf "%.6f", total-candidate-completeness }')"
fused_leaf_consume_residual_ms="$(awk -v total="${fused_leaf_cursor_consume_ms}" \
  -v candidate="${fused_leaf_candidate_recheck_ms}" \
  -v completeness="${fused_leaf_completeness_fallback_ms}" \
  'BEGIN { printf "%.6f", total-candidate-completeness }')"
cpu_path_append_residual_ms="$(awk -v total="${cpu_path_append_ms}" \
  -v parent="${cpu_path_append_parent_prepare_ms}" \
  -v normalize="${cpu_path_append_child_normalize_ms}" \
  -v edges="${cpu_path_append_child_edges_ms}" -v cuda="${cpu_path_append_cuda_evaluate_ms}" \
  -v compare="${cpu_path_append_cuda_compare_ms}" \
  'BEGIN { printf "%.6f", total-parent-normalize-edges-cuda-compare }')"
cpu_fused_path_append_residual_ms="$(awk -v total="${cpu_fused_path_append_ms}" \
  -v parent="${cpu_fused_path_append_parent_prepare_ms}" \
  -v normalize="${cpu_fused_path_append_child_normalize_ms}" \
  -v edges="${cpu_fused_path_append_child_edges_ms}" \
  -v cuda="${cpu_fused_path_append_cuda_evaluate_ms}" \
  -v compare="${cpu_fused_path_append_cuda_compare_ms}" \
  'BEGIN { printf "%.6f", total-parent-normalize-edges-cuda-compare }')"
cuda_path_append_residual_ms="$(awk -v total="${cuda_path_append_ms}" \
  -v parent="${cuda_path_append_parent_prepare_ms}" \
  -v normalize="${cuda_path_append_child_normalize_ms}" \
  -v edges="${cuda_path_append_child_edges_ms}" -v cuda="${cuda_path_append_cuda_evaluate_ms}" \
  -v compare="${cuda_path_append_cuda_compare_ms}" \
  'BEGIN { printf "%.6f", total-parent-normalize-edges-cuda-compare }')"
hybrid_path_append_residual_ms="$(awk -v total="${hybrid_path_append_ms}" \
  -v parent="${hybrid_path_append_parent_prepare_ms}" \
  -v normalize="${hybrid_path_append_child_normalize_ms}" \
  -v edges="${hybrid_path_append_child_edges_ms}" \
  -v cuda="${hybrid_path_append_cuda_evaluate_ms}" \
  -v compare="${hybrid_path_append_cuda_compare_ms}" \
  'BEGIN { printf "%.6f", total-parent-normalize-edges-cuda-compare }')"
fused_path_append_residual_ms="$(awk -v total="${fused_path_append_ms}" \
  -v parent="${fused_path_append_parent_prepare_ms}" \
  -v normalize="${fused_path_append_child_normalize_ms}" \
  -v edges="${fused_path_append_child_edges_ms}" \
  -v cuda="${fused_path_append_cuda_evaluate_ms}" \
  -v compare="${fused_path_append_cuda_compare_ms}" \
  'BEGIN { printf "%.6f", total-parent-normalize-edges-cuda-compare }')"
cpu_hamilton_reply_residual_ms="$(awk -v total="${cpu_hamilton_reply_ms}" \
  -v validation="${cpu_hamilton_reply_validation_ms}" \
  -v enumerate="${cpu_hamilton_reply_enumerate_ms}" -v cuda="${cpu_hamilton_reply_cuda_ms}" \
  -v compare="${cpu_hamilton_reply_compare_ms}" \
  'BEGIN { printf "%.6f", total-validation-enumerate-cuda-compare }')"
cuda_hamilton_reply_residual_ms="$(awk -v total="${cuda_hamilton_reply_ms}" \
  -v validation="${cuda_hamilton_reply_validation_ms}" \
  -v enumerate="${cuda_hamilton_reply_enumerate_ms}" -v cuda="${cuda_hamilton_reply_cuda_ms}" \
  -v compare="${cuda_hamilton_reply_compare_ms}" \
  'BEGIN { printf "%.6f", total-validation-enumerate-cuda-compare }')"
hybrid_hamilton_reply_residual_ms="$(awk -v total="${hybrid_hamilton_reply_ms}" \
  -v validation="${hybrid_hamilton_reply_validation_ms}" \
  -v enumerate="${hybrid_hamilton_reply_enumerate_ms}" -v cuda="${hybrid_hamilton_reply_cuda_ms}" \
  -v compare="${hybrid_hamilton_reply_compare_ms}" \
  'BEGIN { printf "%.6f", total-validation-enumerate-cuda-compare }')"
fused_hamilton_reply_residual_ms="$(awk -v total="${fused_hamilton_reply_ms}" \
  -v validation="${fused_hamilton_reply_validation_ms}" \
  -v enumerate="${fused_hamilton_reply_enumerate_ms}" -v cuda="${fused_hamilton_reply_cuda_ms}" \
  -v compare="${fused_hamilton_reply_compare_ms}" \
  'BEGIN { printf "%.6f", total-validation-enumerate-cuda-compare }')"
cpu_host_build_ms="$(awk -v work="${cpu_work_graph_ms}" -v leaf="${cpu_leaf_ms}" \
  -v path="${cpu_path_append_ms}" -v hamilton="${cpu_hamilton_reply_ms}" \
  -v end="${cpu_end_reply_ms}" 'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"
cpu_fused_host_build_ms="$(awk -v work="${cpu_fused_work_graph_ms}" \
  -v leaf="${cpu_fused_leaf_ms}" -v path="${cpu_fused_path_append_ms}" \
  -v hamilton="${cpu_fused_hamilton_reply_ms}" -v end="${cpu_fused_end_reply_ms}" \
  'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"
cuda_host_build_ms="$(awk -v work="${cuda_work_graph_ms}" -v leaf="${cuda_leaf_ms}" \
  -v path="${cuda_path_append_ms}" -v hamilton="${cuda_hamilton_reply_ms}" \
  -v end="${cuda_end_reply_ms}" 'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"
hybrid_host_build_ms="$(awk -v work="${hybrid_work_graph_ms}" -v leaf="${hybrid_leaf_ms}" \
  -v path="${hybrid_path_append_ms}" -v hamilton="${hybrid_hamilton_reply_ms}" \
  -v end="${hybrid_end_reply_ms}" 'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"
fused_host_build_ms="$(awk -v work="${fused_work_graph_ms}" -v leaf="${fused_leaf_ms}" \
  -v path="${fused_path_append_ms}" -v hamilton="${fused_hamilton_reply_ms}" \
  -v end="${fused_end_reply_ms}" 'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"
cpu_host_build_unprofiled_ms="$(awk -v total="${cpu_host_build_ms}" \
  -v root="${cpu_root_child_normalize_ms}" -v scan="${cpu_point_candidate_scan_ms}" \
  -v sort="${cpu_point_candidate_sort_ms}" \
  'BEGIN { printf "%.6f", total-root-scan-sort }')"
cpu_fused_host_build_unprofiled_ms="$(awk -v total="${cpu_fused_host_build_ms}" \
  -v root="${cpu_fused_root_child_normalize_ms}" \
  -v scan="${cpu_fused_point_candidate_scan_ms}" \
  -v sort="${cpu_fused_point_candidate_sort_ms}" \
  'BEGIN { printf "%.6f", total-root-scan-sort }')"
cuda_host_build_unprofiled_ms="$(awk -v total="${cuda_host_build_ms}" \
  -v root="${cuda_root_child_normalize_ms}" -v scan="${cuda_point_candidate_scan_ms}" \
  -v sort="${cuda_point_candidate_sort_ms}" \
  'BEGIN { printf "%.6f", total-root-scan-sort }')"
hybrid_host_build_unprofiled_ms="$(awk -v total="${hybrid_host_build_ms}" \
  -v root="${hybrid_root_child_normalize_ms}" -v scan="${hybrid_point_candidate_scan_ms}" \
  -v sort="${hybrid_point_candidate_sort_ms}" \
  'BEGIN { printf "%.6f", total-root-scan-sort }')"
fused_host_build_unprofiled_ms="$(awk -v total="${fused_host_build_ms}" \
  -v root="${fused_root_child_normalize_ms}" -v scan="${fused_point_candidate_scan_ms}" \
  -v sort="${fused_point_candidate_sort_ms}" \
  'BEGIN { printf "%.6f", total-root-scan-sort }')"

manifest="${run_dir}/run-manifest-v1"
{
  echo "CUDAEE_HT_SCAN_BENCHMARK_MANIFEST_V1"
  echo "run_id ${run_id}"
  echo "instance ${instance}"
  echo "utc_timestamp ${timestamp}"
  echo "git_commit $(git rev-parse HEAD)"
  echo "git_dirty $(git status --porcelain | awk 'END { print NR == 0 ? 0 : 1 }')"
  echo "tsp ${tsp}"
  echo "tsp_sha256 $(sha256sum "${tsp}" | awk '{ print $1 }')"
  echo "source_edges ${edges}"
  echo "source_edges_sha256 $(sha256sum "${edges}" | awk '{ print $1 }')"
  echo "jv_fixed_edges ${jv_edges}"
  echo "jv_fixed_edges_sha256 $(sha256sum "${jv_edges}" | awk '{ print $1 }')"
  echo "certified_optimum ${certified_optimum}"
  echo "protected_tour ${protected_tour:-none}"
  echo "protected_tour_sha256 ${protected_tour_sha256}"
  echo "target_offset ${target_offset}"
  echo "max_targets ${max_targets}"
  echo "target_order weight-desc"
  echo "target_devices auto"
  echo "benchmark_modes cpu,cpu-fused,cuda-all,hybrid-leaf-cuda,fused-leaf-buckets"
  echo "cd_mode missing-or-incompatible"
  echo "max_neighborhood 25"
  echo "max_cd_candidates 5"
  echo "max_candidate_degree 50"
  echo "max_root_replies 10000"
  echo "max_k 5"
  echo "max_deletion_sets 100"
  echo "max_depth 2"
  echo "max_states 2000"
  echo "max_total_replies 20000"
  echo "max_replies_per_move 2000"
  echo "max_point_candidates 3"
  echo "max_end_candidates 3"
  echo "reply_frontier_batch_states 256"
  echo "leaf_frontier_batch_states 256"
  echo "cost_batch_size 4096"
  echo "cuda_min_cost_cells 128"
  echo "reuse_reply_cuda_cache ${reuse_reply_cuda_cache}"
  echo "deduplicate_reply_tasks ${deduplicate_reply_tasks}"
  echo "cpu_cost_threads ${cpu_cost_threads}"
  echo "omp_dynamic FALSE"
  echo "omp_proc_bind spread"
  echo "omp_places cores"
  echo "physical_gpu ${physical_gpu}"
  echo "compiler $(c++ --version | awk 'NR == 1')"
  echo "nvcc $(nvcc --version | awk '/release/ { print; exit }')"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | sed 's/^/gpu /'
  echo "END"
} >"${manifest}"

summary="${run_dir}/summary.txt"
{
  echo "CUDAEE_HT_SCAN_BENCHMARK_SUMMARY_V20"
  echo "instance ${instance}"
  echo "attempted_targets ${attempted}"
  echo "proven_targets ${proven}"
  echo "unresolved_targets ${unresolved}"
  echo "committed_targets ${committed}"
  echo "states_expanded ${states}"
  echo "replies_expanded ${replies}"
  echo "leaf_calls ${leaf_calls}"
  echo "cuda_leaf_cost_cells ${leaf_cells}"
  echo "cuda_peak_leaf_device_cache_bytes ${peak_cache}"
  echo "cpu_candidate_ms ${cpu_candidate_ms}"
  echo "cpu_fused_candidate_ms ${cpu_fused_candidate_ms}"
  echo "cuda_candidate_ms ${cuda_candidate_ms}"
  echo "hybrid_candidate_ms ${hybrid_candidate_ms}"
  echo "cpu_work_graph_ms ${cpu_work_graph_ms}"
  echo "cpu_fused_work_graph_ms ${cpu_fused_work_graph_ms}"
  echo "cuda_work_graph_ms ${cuda_work_graph_ms}"
  echo "hybrid_work_graph_ms ${hybrid_work_graph_ms}"
  echo "fused_work_graph_ms ${fused_work_graph_ms}"
  echo "cpu_host_build_residual_ms ${cpu_host_build_ms}"
  echo "cpu_fused_host_build_residual_ms ${cpu_fused_host_build_ms}"
  echo "cuda_host_build_residual_ms ${cuda_host_build_ms}"
  echo "hybrid_host_build_residual_ms ${hybrid_host_build_ms}"
  echo "fused_host_build_residual_ms ${fused_host_build_ms}"
  echo "root_child_normalizations ${cpu_root_child_normalizations}"
  echo "cpu_root_child_normalize_ms ${cpu_root_child_normalize_ms}"
  echo "cpu_fused_root_child_normalize_ms ${cpu_fused_root_child_normalize_ms}"
  echo "cuda_root_child_normalize_ms ${cuda_root_child_normalize_ms}"
  echo "hybrid_root_child_normalize_ms ${hybrid_root_child_normalize_ms}"
  echo "fused_root_child_normalize_ms ${fused_root_child_normalize_ms}"
  echo "point_candidate_scans ${cpu_point_candidate_scans}"
  echo "point_candidate_nodes_checked ${cpu_point_candidate_nodes_checked}"
  echo "point_candidate_nodes_ranked ${cpu_point_candidate_nodes_ranked}"
  echo "point_candidate_nodes_selected ${cpu_point_candidate_nodes_selected}"
  echo "cpu_point_candidate_scan_ms ${cpu_point_candidate_scan_ms}"
  echo "cpu_fused_point_candidate_scan_ms ${cpu_fused_point_candidate_scan_ms}"
  echo "cuda_point_candidate_scan_ms ${cuda_point_candidate_scan_ms}"
  echo "hybrid_point_candidate_scan_ms ${hybrid_point_candidate_scan_ms}"
  echo "fused_point_candidate_scan_ms ${fused_point_candidate_scan_ms}"
  echo "cpu_point_candidate_sort_ms ${cpu_point_candidate_sort_ms}"
  echo "cpu_fused_point_candidate_sort_ms ${cpu_fused_point_candidate_sort_ms}"
  echo "cuda_point_candidate_sort_ms ${cuda_point_candidate_sort_ms}"
  echo "hybrid_point_candidate_sort_ms ${hybrid_point_candidate_sort_ms}"
  echo "fused_point_candidate_sort_ms ${fused_point_candidate_sort_ms}"
  echo "cpu_host_build_unprofiled_ms ${cpu_host_build_unprofiled_ms}"
  echo "cpu_fused_host_build_unprofiled_ms ${cpu_fused_host_build_unprofiled_ms}"
  echo "cuda_host_build_unprofiled_ms ${cuda_host_build_unprofiled_ms}"
  echo "hybrid_host_build_unprofiled_ms ${hybrid_host_build_unprofiled_ms}"
  echo "fused_host_build_unprofiled_ms ${fused_host_build_unprofiled_ms}"
  echo "cpu_leaf_ms ${cpu_leaf_ms}"
  echo "cpu_fused_leaf_ms ${cpu_fused_leaf_ms}"
  echo "cuda_leaf_ms ${cuda_leaf_ms}"
  echo "hybrid_leaf_ms ${hybrid_leaf_ms}"
  echo "fused_leaf_ms ${fused_leaf_ms}"
  echo "hybrid_leaf_cuda_batches ${hybrid_leaf_cuda_batches}"
  echo "fused_leaf_cuda_batches ${fused_leaf_cuda_batches}"
  echo "hybrid_leaf_frontier_batches ${hybrid_leaf_frontier_batches}"
  echo "fused_leaf_frontier_batches ${fused_leaf_frontier_batches}"
  echo "cpu_fused_leaf_frontier_batches ${cpu_fused_leaf_frontier_batches}"
  echo "hybrid_leaf_bucket_count ${hybrid_leaf_bucket_count}"
  echo "fused_leaf_bucket_count ${fused_leaf_bucket_count}"
  echo "cpu_fused_leaf_bucket_count ${cpu_fused_leaf_bucket_count}"
  echo "leaf_cursor_searches_started ${cpu_leaf_cursor_searches}"
  echo "cpu_leaf_setup_ms ${cpu_leaf_setup_ms}"
  echo "cpu_leaf_proof_initialize_ms ${cpu_leaf_proof_initialize_ms}"
  echo "cpu_leaf_coverage_scan_ms ${cpu_leaf_coverage_scan_ms}"
  echo "cpu_leaf_cursor_construct_ms ${cpu_leaf_cursor_construct_ms}"
  echo "cpu_leaf_setup_residual_ms ${cpu_leaf_setup_residual_ms}"
  echo "cpu_leaf_cursor_prepare_ms ${cpu_leaf_cursor_prepare_ms}"
  echo "cpu_leaf_cost_evaluate_ms ${cpu_leaf_cost_evaluate_ms}"
  echo "cpu_leaf_cost_cpu_certify_ms ${cpu_leaf_cost_cpu_certify_ms}"
  echo "cpu_leaf_cost_scatter_ms ${cpu_leaf_cost_scatter_ms}"
  echo "cpu_leaf_cursor_consume_ms ${cpu_leaf_cursor_consume_ms}"
  echo "cpu_leaf_candidate_recheck_ms ${cpu_leaf_candidate_recheck_ms}"
  echo "cpu_leaf_completeness_fallback_ms ${cpu_leaf_completeness_fallback_ms}"
  echo "cpu_leaf_scalar_search_ms ${cpu_leaf_scalar_search_ms}"
  echo "cpu_leaf_apply_ms ${cpu_leaf_apply_ms}"
  echo "cpu_leaf_proof_verify_ms ${cpu_leaf_proof_verify_ms}"
  echo "cpu_leaf_residual_ms ${cpu_leaf_residual_ms}"
  echo "cpu_fused_leaf_setup_ms ${cpu_fused_leaf_setup_ms}"
  echo "cpu_fused_leaf_proof_initialize_ms ${cpu_fused_leaf_proof_initialize_ms}"
  echo "cpu_fused_leaf_coverage_scan_ms ${cpu_fused_leaf_coverage_scan_ms}"
  echo "cpu_fused_leaf_cursor_construct_ms ${cpu_fused_leaf_cursor_construct_ms}"
  echo "cpu_fused_leaf_setup_residual_ms ${cpu_fused_leaf_setup_residual_ms}"
  echo "cpu_fused_leaf_cursor_prepare_ms ${cpu_fused_leaf_cursor_prepare_ms}"
  echo "cpu_fused_leaf_cost_evaluate_ms ${cpu_fused_leaf_cost_evaluate_ms}"
  echo "cpu_fused_leaf_cost_cpu_certify_ms ${cpu_fused_leaf_cost_cpu_certify_ms}"
  echo "cpu_fused_leaf_cost_scatter_ms ${cpu_fused_leaf_cost_scatter_ms}"
  echo "cpu_fused_leaf_cursor_consume_ms ${cpu_fused_leaf_cursor_consume_ms}"
  echo "cpu_fused_leaf_candidate_recheck_ms ${cpu_fused_leaf_candidate_recheck_ms}"
  echo "cpu_fused_leaf_completeness_fallback_ms ${cpu_fused_leaf_completeness_fallback_ms}"
  echo "cpu_fused_leaf_scalar_search_ms ${cpu_fused_leaf_scalar_search_ms}"
  echo "cpu_fused_leaf_apply_ms ${cpu_fused_leaf_apply_ms}"
  echo "cpu_fused_leaf_proof_verify_ms ${cpu_fused_leaf_proof_verify_ms}"
  echo "cpu_fused_leaf_residual_ms ${cpu_fused_leaf_residual_ms}"
  echo "cuda_leaf_setup_ms ${cuda_leaf_setup_ms}"
  echo "cuda_leaf_proof_initialize_ms ${cuda_leaf_proof_initialize_ms}"
  echo "cuda_leaf_coverage_scan_ms ${cuda_leaf_coverage_scan_ms}"
  echo "cuda_leaf_cursor_construct_ms ${cuda_leaf_cursor_construct_ms}"
  echo "cuda_leaf_setup_residual_ms ${cuda_leaf_setup_residual_ms}"
  echo "cuda_leaf_cursor_prepare_ms ${cuda_leaf_cursor_prepare_ms}"
  echo "cuda_leaf_cost_evaluate_ms ${cuda_leaf_cost_evaluate_ms}"
  echo "cuda_leaf_cost_cpu_certify_ms ${cuda_leaf_cost_cpu_certify_ms}"
  echo "cuda_leaf_cost_scatter_ms ${cuda_leaf_cost_scatter_ms}"
  echo "cuda_leaf_cursor_consume_ms ${cuda_leaf_cursor_consume_ms}"
  echo "cuda_leaf_candidate_recheck_ms ${cuda_leaf_candidate_recheck_ms}"
  echo "cuda_leaf_completeness_fallback_ms ${cuda_leaf_completeness_fallback_ms}"
  echo "cuda_leaf_consume_residual_ms ${cuda_leaf_consume_residual_ms}"
  echo "cuda_leaf_scalar_search_ms ${cuda_leaf_scalar_search_ms}"
  echo "cuda_leaf_apply_ms ${cuda_leaf_apply_ms}"
  echo "cuda_leaf_proof_verify_ms ${cuda_leaf_proof_verify_ms}"
  echo "cuda_leaf_residual_ms ${cuda_leaf_residual_ms}"
  echo "hybrid_leaf_setup_ms ${hybrid_leaf_setup_ms}"
  echo "hybrid_leaf_proof_initialize_ms ${hybrid_leaf_proof_initialize_ms}"
  echo "hybrid_leaf_coverage_scan_ms ${hybrid_leaf_coverage_scan_ms}"
  echo "hybrid_leaf_cursor_construct_ms ${hybrid_leaf_cursor_construct_ms}"
  echo "hybrid_leaf_setup_residual_ms ${hybrid_leaf_setup_residual_ms}"
  echo "hybrid_leaf_cursor_prepare_ms ${hybrid_leaf_cursor_prepare_ms}"
  echo "hybrid_leaf_cost_evaluate_ms ${hybrid_leaf_cost_evaluate_ms}"
  echo "hybrid_leaf_cost_cpu_certify_ms ${hybrid_leaf_cost_cpu_certify_ms}"
  echo "hybrid_leaf_cost_scatter_ms ${hybrid_leaf_cost_scatter_ms}"
  echo "hybrid_leaf_cursor_consume_ms ${hybrid_leaf_cursor_consume_ms}"
  echo "hybrid_leaf_candidate_recheck_ms ${hybrid_leaf_candidate_recheck_ms}"
  echo "hybrid_leaf_completeness_fallback_ms ${hybrid_leaf_completeness_fallback_ms}"
  echo "hybrid_leaf_consume_residual_ms ${hybrid_leaf_consume_residual_ms}"
  echo "hybrid_leaf_scalar_search_ms ${hybrid_leaf_scalar_search_ms}"
  echo "hybrid_leaf_apply_ms ${hybrid_leaf_apply_ms}"
  echo "hybrid_leaf_proof_verify_ms ${hybrid_leaf_proof_verify_ms}"
  echo "hybrid_leaf_residual_ms ${hybrid_leaf_residual_ms}"
  echo "fused_leaf_setup_ms ${fused_leaf_setup_ms}"
  echo "fused_leaf_proof_initialize_ms ${fused_leaf_proof_initialize_ms}"
  echo "fused_leaf_coverage_scan_ms ${fused_leaf_coverage_scan_ms}"
  echo "fused_leaf_cursor_construct_ms ${fused_leaf_cursor_construct_ms}"
  echo "fused_leaf_setup_residual_ms ${fused_leaf_setup_residual_ms}"
  echo "fused_leaf_cursor_prepare_ms ${fused_leaf_cursor_prepare_ms}"
  echo "fused_leaf_cost_evaluate_ms ${fused_leaf_cost_evaluate_ms}"
  echo "fused_leaf_cost_cpu_certify_ms ${fused_leaf_cost_cpu_certify_ms}"
  echo "fused_leaf_cost_scatter_ms ${fused_leaf_cost_scatter_ms}"
  echo "fused_leaf_cursor_consume_ms ${fused_leaf_cursor_consume_ms}"
  echo "fused_leaf_candidate_recheck_ms ${fused_leaf_candidate_recheck_ms}"
  echo "fused_leaf_completeness_fallback_ms ${fused_leaf_completeness_fallback_ms}"
  echo "fused_leaf_consume_residual_ms ${fused_leaf_consume_residual_ms}"
  echo "fused_leaf_scalar_search_ms ${fused_leaf_scalar_search_ms}"
  echo "fused_leaf_apply_ms ${fused_leaf_apply_ms}"
  echo "fused_leaf_proof_verify_ms ${fused_leaf_proof_verify_ms}"
  echo "fused_leaf_residual_ms ${fused_leaf_residual_ms}"
  echo "leaf_cost_rows_consumed ${cpu_leaf_cost_rows}"
  echo "leaf_candidate_templates_rechecked ${cpu_leaf_candidate_rechecks}"
  echo "leaf_cpu_completeness_rows ${cpu_leaf_completeness_rows}"
  echo "leaf_cpu_completeness_templates ${cpu_leaf_completeness_templates}"
  echo "leaf_cpu_certified_cost_cells ${cpu_leaf_cpu_certified_cells}"
  echo "cpu_leaf_cost_rows_scored ${cpu_leaf_rows_scored}"
  echo "cpu_leaf_cost_rows_reused ${cpu_leaf_rows_reused}"
  echo "cpu_fused_leaf_cost_rows_scored ${cpu_fused_leaf_rows_scored}"
  echo "cpu_fused_leaf_cost_rows_reused ${cpu_fused_leaf_rows_reused}"
  echo "cuda_leaf_cost_rows_scored ${cuda_leaf_rows_scored}"
  echo "cuda_leaf_cost_rows_reused ${cuda_leaf_rows_reused}"
  echo "hybrid_leaf_cost_rows_scored ${hybrid_leaf_rows_scored}"
  echo "hybrid_leaf_cost_rows_reused ${hybrid_leaf_rows_reused}"
  echo "fused_leaf_cost_rows_scored ${fused_leaf_rows_scored}"
  echo "fused_leaf_cost_rows_reused ${fused_leaf_rows_reused}"
  echo "cpu_leaf_parallel_cost_batches ${cpu_leaf_parallel_batches}"
  echo "cpu_fused_leaf_parallel_cost_batches ${cpu_fused_leaf_parallel_batches}"
  echo "cuda_leaf_parallel_cost_batches ${cuda_leaf_parallel_batches}"
  echo "hybrid_leaf_parallel_cost_batches ${hybrid_leaf_parallel_batches}"
  echo "fused_leaf_parallel_cost_batches ${fused_leaf_parallel_batches}"
  echo "cpu_leaf_parallel_cost_cells ${cpu_leaf_parallel_cells}"
  echo "cpu_fused_leaf_parallel_cost_cells ${cpu_fused_leaf_parallel_cells}"
  echo "cuda_leaf_parallel_cost_cells ${cuda_leaf_parallel_cells}"
  echo "hybrid_leaf_parallel_cost_cells ${hybrid_leaf_parallel_cells}"
  echo "fused_leaf_parallel_cost_cells ${fused_leaf_parallel_cells}"
  echo "cpu_peak_leaf_cpu_cost_threads ${cpu_peak_leaf_threads}"
  echo "cpu_fused_peak_leaf_cpu_cost_threads ${cpu_fused_peak_leaf_threads}"
  echo "cuda_peak_leaf_cpu_cost_threads ${cuda_peak_leaf_threads}"
  echo "hybrid_peak_leaf_cpu_cost_threads ${hybrid_peak_leaf_threads}"
  echo "fused_peak_leaf_cpu_cost_threads ${fused_peak_leaf_threads}"
  echo "leaf_speedup ${leaf_speedup}"
  echo "cpu_fused_leaf_speedup ${cpu_fused_leaf_speedup}"
  echo "hybrid_leaf_speedup ${hybrid_leaf_speedup}"
  echo "fused_leaf_speedup_vs_hybrid ${fused_leaf_speedup}"
  echo "cpu_path_append_ms ${cpu_path_append_ms}"
  echo "cpu_path_append_parent_prepare_ms ${cpu_path_append_parent_prepare_ms}"
  echo "cpu_path_append_child_normalize_ms ${cpu_path_append_child_normalize_ms}"
  echo "cpu_path_append_child_edges_ms ${cpu_path_append_child_edges_ms}"
  echo "cpu_path_append_cuda_evaluate_ms ${cpu_path_append_cuda_evaluate_ms}"
  echo "cpu_path_append_cuda_compare_ms ${cpu_path_append_cuda_compare_ms}"
  echo "cpu_path_append_residual_ms ${cpu_path_append_residual_ms}"
  echo "cpu_fused_path_append_ms ${cpu_fused_path_append_ms}"
  echo "cpu_fused_path_append_parent_prepare_ms ${cpu_fused_path_append_parent_prepare_ms}"
  echo "cpu_fused_path_append_child_normalize_ms ${cpu_fused_path_append_child_normalize_ms}"
  echo "cpu_fused_path_append_child_edges_ms ${cpu_fused_path_append_child_edges_ms}"
  echo "cpu_fused_path_append_cuda_evaluate_ms ${cpu_fused_path_append_cuda_evaluate_ms}"
  echo "cpu_fused_path_append_cuda_compare_ms ${cpu_fused_path_append_cuda_compare_ms}"
  echo "cpu_fused_path_append_residual_ms ${cpu_fused_path_append_residual_ms}"
  echo "cuda_path_append_ms ${cuda_path_append_ms}"
  echo "cuda_path_append_parent_prepare_ms ${cuda_path_append_parent_prepare_ms}"
  echo "cuda_path_append_child_normalize_ms ${cuda_path_append_child_normalize_ms}"
  echo "cuda_path_append_child_edges_ms ${cuda_path_append_child_edges_ms}"
  echo "cuda_path_append_cuda_evaluate_ms ${cuda_path_append_cuda_evaluate_ms}"
  echo "cuda_path_append_cuda_compare_ms ${cuda_path_append_cuda_compare_ms}"
  echo "cuda_path_append_residual_ms ${cuda_path_append_residual_ms}"
  echo "hybrid_path_append_ms ${hybrid_path_append_ms}"
  echo "hybrid_path_append_parent_prepare_ms ${hybrid_path_append_parent_prepare_ms}"
  echo "hybrid_path_append_child_normalize_ms ${hybrid_path_append_child_normalize_ms}"
  echo "hybrid_path_append_child_edges_ms ${hybrid_path_append_child_edges_ms}"
  echo "hybrid_path_append_cuda_evaluate_ms ${hybrid_path_append_cuda_evaluate_ms}"
  echo "hybrid_path_append_cuda_compare_ms ${hybrid_path_append_cuda_compare_ms}"
  echo "hybrid_path_append_residual_ms ${hybrid_path_append_residual_ms}"
  echo "fused_path_append_ms ${fused_path_append_ms}"
  echo "fused_path_append_parent_prepare_ms ${fused_path_append_parent_prepare_ms}"
  echo "fused_path_append_child_normalize_ms ${fused_path_append_child_normalize_ms}"
  echo "fused_path_append_child_edges_ms ${fused_path_append_child_edges_ms}"
  echo "fused_path_append_cuda_evaluate_ms ${fused_path_append_cuda_evaluate_ms}"
  echo "fused_path_append_cuda_compare_ms ${fused_path_append_cuda_compare_ms}"
  echo "fused_path_append_residual_ms ${fused_path_append_residual_ms}"
  echo "cpu_hamilton_reply_ms ${cpu_hamilton_reply_ms}"
  echo "cpu_fused_hamilton_reply_ms ${cpu_fused_hamilton_reply_ms}"
  echo "cuda_hamilton_reply_ms ${cuda_hamilton_reply_ms}"
  echo "hybrid_hamilton_reply_ms ${hybrid_hamilton_reply_ms}"
  echo "fused_hamilton_reply_ms ${fused_hamilton_reply_ms}"
  echo "cpu_hamilton_reply_validation_ms ${cpu_hamilton_reply_validation_ms}"
  echo "cpu_hamilton_reply_enumerate_ms ${cpu_hamilton_reply_enumerate_ms}"
  echo "cpu_hamilton_reply_cuda_ms ${cpu_hamilton_reply_cuda_ms}"
  echo "cpu_hamilton_reply_compare_ms ${cpu_hamilton_reply_compare_ms}"
  echo "cpu_hamilton_reply_residual_ms ${cpu_hamilton_reply_residual_ms}"
  echo "cuda_hamilton_reply_validation_ms ${cuda_hamilton_reply_validation_ms}"
  echo "cuda_hamilton_reply_enumerate_ms ${cuda_hamilton_reply_enumerate_ms}"
  echo "cuda_hamilton_reply_cuda_ms ${cuda_hamilton_reply_cuda_ms}"
  echo "cuda_hamilton_reply_compare_ms ${cuda_hamilton_reply_compare_ms}"
  echo "cuda_hamilton_reply_residual_ms ${cuda_hamilton_reply_residual_ms}"
  echo "hybrid_hamilton_reply_validation_ms ${hybrid_hamilton_reply_validation_ms}"
  echo "hybrid_hamilton_reply_enumerate_ms ${hybrid_hamilton_reply_enumerate_ms}"
  echo "hybrid_hamilton_reply_cuda_ms ${hybrid_hamilton_reply_cuda_ms}"
  echo "hybrid_hamilton_reply_compare_ms ${hybrid_hamilton_reply_compare_ms}"
  echo "hybrid_hamilton_reply_residual_ms ${hybrid_hamilton_reply_residual_ms}"
  echo "fused_hamilton_reply_validation_ms ${fused_hamilton_reply_validation_ms}"
  echo "fused_hamilton_reply_enumerate_ms ${fused_hamilton_reply_enumerate_ms}"
  echo "fused_hamilton_reply_cuda_ms ${fused_hamilton_reply_cuda_ms}"
  echo "fused_hamilton_reply_compare_ms ${fused_hamilton_reply_compare_ms}"
  echo "fused_hamilton_reply_residual_ms ${fused_hamilton_reply_residual_ms}"
  echo "hamilton_reply_batches ${cpu_hamilton_reply_batches}"
  echo "hamilton_reply_centers ${cpu_hamilton_reply_centers}"
  echo "hamilton_reply_unique_centers ${cpu_hamilton_reply_unique_centers}"
  echo "hamilton_reply_neighbor_pairs_tested ${cpu_hamilton_reply_pairs}"
  echo "hamilton_replies_generated ${cpu_hamilton_replies_generated}"
  echo "reuse_reply_cuda_cache ${reuse_reply_cuda_cache}"
  echo "deduplicate_reply_tasks ${deduplicate_reply_tasks}"
  echo "cuda_reply_cuda_batches ${cuda_reply_cuda_batches}"
  echo "cuda_reply_cuda_tasks_submitted ${cuda_reply_cuda_tasks_submitted}"
  echo "cuda_reply_graph_cache_hits ${cuda_reply_graph_cache_hits}"
  echo "cuda_reply_workspace_cache_hits ${cuda_reply_workspace_cache_hits}"
  echo "cuda_peak_reply_device_cache_bytes ${cuda_peak_reply_device_cache_bytes}"
  echo "end_reply_batches ${cpu_end_reply_batches}"
  echo "end_reply_tasks ${cpu_end_reply_tasks}"
  echo "end_reply_unique_tasks ${cpu_end_reply_unique_tasks}"
  echo "end_replies_generated ${cpu_end_replies_generated}"
  echo "cpu_end_reply_ms ${cpu_end_reply_ms}"
  echo "cpu_fused_end_reply_ms ${cpu_fused_end_reply_ms}"
  echo "cuda_end_reply_ms ${cuda_end_reply_ms}"
  echo "hybrid_end_reply_ms ${hybrid_end_reply_ms}"
  echo "fused_end_reply_ms ${fused_end_reply_ms}"
  echo "cpu_propagation_ms ${cpu_propagation_ms}"
  echo "cuda_propagation_ms ${cuda_propagation_ms}"
  echo "hybrid_propagation_ms ${hybrid_propagation_ms}"
  echo "cpu_proof_extract_ms ${cpu_proof_extract_ms}"
  echo "cuda_proof_extract_ms ${cuda_proof_extract_ms}"
  echo "hybrid_proof_extract_ms ${hybrid_proof_extract_ms}"
  echo "cpu_proof_verify_ms ${cpu_proof_verify_ms}"
  echo "cuda_proof_verify_ms ${cuda_proof_verify_ms}"
  echo "hybrid_proof_verify_ms ${hybrid_proof_verify_ms}"
  echo "cpu_immediate_verify_ms ${cpu_immediate_verify_ms}"
  echo "cuda_immediate_verify_ms ${cuda_immediate_verify_ms}"
  echo "hybrid_immediate_verify_ms ${hybrid_immediate_verify_ms}"
  echo "cpu_commit_ms ${cpu_commit_ms}"
  echo "cuda_commit_ms ${cuda_commit_ms}"
  echo "hybrid_commit_ms ${hybrid_commit_ms}"
  echo "cpu_target_execution_ms ${cpu_target_execution_ms}"
  echo "cpu_fused_target_execution_ms ${cpu_fused_target_execution_ms}"
  echo "cuda_target_execution_ms ${cuda_target_execution_ms}"
  echo "hybrid_target_execution_ms ${hybrid_target_execution_ms}"
  echo "fused_target_execution_ms ${fused_target_execution_ms}"
  echo "cpu_search_ms ${cpu_search_ms}"
  echo "cpu_fused_search_ms ${cpu_fused_search_ms}"
  echo "cpu_fused_search_speedup ${cpu_fused_search_speedup}"
  echo "cuda_search_ms ${cuda_search_ms}"
  echo "search_speedup ${search_speedup}"
  echo "hybrid_search_ms ${hybrid_search_ms}"
  echo "hybrid_search_speedup ${hybrid_search_speedup}"
  echo "fused_search_ms ${fused_search_ms}"
  echo "fused_search_speedup ${fused_search_speedup}"
  echo "cpu_total_ms ${cpu_total_ms}"
  echo "cpu_fused_total_ms ${cpu_fused_total_ms}"
  echo "cuda_total_ms ${cuda_total_ms}"
  echo "hybrid_total_ms ${hybrid_total_ms}"
  echo "fused_total_ms ${fused_total_ms}"
  echo "cpu_wall_ms ${cpu_wall_ms}"
  echo "cpu_fused_wall_ms ${cpu_fused_wall_ms}"
  echo "cpu_fused_wall_speedup ${cpu_fused_wall_speedup}"
  echo "cuda_wall_ms ${cuda_wall_ms}"
  echo "wall_speedup ${wall_speedup}"
  echo "hybrid_wall_ms ${hybrid_wall_ms}"
  echo "hybrid_wall_speedup ${hybrid_wall_speedup}"
  echo "fused_wall_ms ${fused_wall_ms}"
  echo "fused_wall_speedup ${fused_wall_speedup}"
  echo "initial_hash ${initial_hash}"
  echo "final_hash ${final_hash}"
  echo "verified_edge_sha256 $(sha256sum "${run_dir}/cuda.edg" | awk '{ print $1 }')"
  echo "protected_tour_checked ${protected_tour_checked}"
  echo "protected_tour_hash ${protected_tour_hash}"
  echo "END"
} >"${summary}"

echo "完成：${run_dir}"
sed -n '1,240p' "${summary}"
