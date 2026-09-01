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
  else
    "${binary}" ht-scan \
      --tsp "${tsp}" --edges "${jv_edges}" --output "${output}" --proof "${proof}" \
      --report "${report}" "${tour_arguments[@]}" "${ht_arguments[@]}" \
      --backend cpu --reply-backend cpu --path-append-backend cpu \
      --propagation-backend cpu >"${stdout_file}" 2>"${stderr_file}"
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
echo "运行 ${instance} HT scan CUDA，物理 GPU ${physical_gpu}"
run_scan cuda "${cuda_binary}"
echo "运行 ${instance} HT scan hybrid，物理 GPU ${physical_gpu}"
run_scan hybrid "${cuda_binary}"

cmp "${run_dir}/cpu.edg" "${run_dir}/cuda.edg"
cmp "${run_dir}/cpu.edg" "${run_dir}/hybrid.edg"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/cpu.report" >"${run_dir}/cpu.work-signature"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/cuda.report" >"${run_dir}/cuda.work-signature"
awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' \
  "${run_dir}/hybrid.report" >"${run_dir}/hybrid.work-signature"
cmp "${run_dir}/cpu.work-signature" "${run_dir}/cuda.work-signature"
cmp "${run_dir}/cpu.work-signature" "${run_dir}/hybrid.work-signature"

read_field() {
  awk -v key="$2" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' "$1"
}

cpu_search_ms="$(read_field "${run_dir}/cpu.report" search_ms)"
cuda_search_ms="$(read_field "${run_dir}/cuda.report" search_ms)"
hybrid_search_ms="$(read_field "${run_dir}/hybrid.report" search_ms)"
cpu_candidate_ms="$(read_field "${run_dir}/cpu.report" candidate_ms)"
cuda_candidate_ms="$(read_field "${run_dir}/cuda.report" candidate_ms)"
hybrid_candidate_ms="$(read_field "${run_dir}/hybrid.report" candidate_ms)"
cpu_work_graph_ms="$(read_field "${run_dir}/cpu.report" work_graph_ms)"
cuda_work_graph_ms="$(read_field "${run_dir}/cuda.report" work_graph_ms)"
hybrid_work_graph_ms="$(read_field "${run_dir}/hybrid.report" work_graph_ms)"
cpu_leaf_ms="$(read_field "${run_dir}/cpu.report" leaf_ms)"
cuda_leaf_ms="$(read_field "${run_dir}/cuda.report" leaf_ms)"
hybrid_leaf_ms="$(read_field "${run_dir}/hybrid.report" leaf_ms)"
cpu_path_append_ms="$(read_field "${run_dir}/cpu.report" path_append_ms)"
cuda_path_append_ms="$(read_field "${run_dir}/cuda.report" path_append_ms)"
hybrid_path_append_ms="$(read_field "${run_dir}/hybrid.report" path_append_ms)"
cpu_hamilton_reply_ms="$(read_field "${run_dir}/cpu.report" hamilton_reply_ms)"
cuda_hamilton_reply_ms="$(read_field "${run_dir}/cuda.report" hamilton_reply_ms)"
hybrid_hamilton_reply_ms="$(read_field "${run_dir}/hybrid.report" hamilton_reply_ms)"
cpu_end_reply_ms="$(read_field "${run_dir}/cpu.report" end_reply_ms)"
cuda_end_reply_ms="$(read_field "${run_dir}/cuda.report" end_reply_ms)"
hybrid_end_reply_ms="$(read_field "${run_dir}/hybrid.report" end_reply_ms)"
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
cuda_total_ms="$(read_field "${run_dir}/cuda.report" total_ms)"
hybrid_total_ms="$(read_field "${run_dir}/hybrid.report" total_ms)"
cpu_wall_ms="$(<"${run_dir}/cpu.wall-ms")"
cuda_wall_ms="$(<"${run_dir}/cuda.wall-ms")"
hybrid_wall_ms="$(<"${run_dir}/hybrid.wall-ms")"
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
wall_speedup="$(awk -v cpu="${cpu_wall_ms}" -v cuda="${cuda_wall_ms}" 'BEGIN { printf "%.3f", cpu/cuda }')"
leaf_speedup="$(awk -v cpu="${cpu_leaf_ms}" -v cuda="${cuda_leaf_ms}" 'BEGIN { printf "%.3f", cpu/cuda }')"
hybrid_search_speedup="$(awk -v cpu="${cpu_search_ms}" -v hybrid="${hybrid_search_ms}" 'BEGIN { printf "%.3f", cpu/hybrid }')"
hybrid_wall_speedup="$(awk -v cpu="${cpu_wall_ms}" -v hybrid="${hybrid_wall_ms}" 'BEGIN { printf "%.3f", cpu/hybrid }')"
hybrid_leaf_speedup="$(awk -v cpu="${cpu_leaf_ms}" -v hybrid="${hybrid_leaf_ms}" 'BEGIN { printf "%.3f", cpu/hybrid }')"
cpu_host_build_ms="$(awk -v work="${cpu_work_graph_ms}" -v leaf="${cpu_leaf_ms}" \
  -v path="${cpu_path_append_ms}" -v hamilton="${cpu_hamilton_reply_ms}" \
  -v end="${cpu_end_reply_ms}" 'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"
cuda_host_build_ms="$(awk -v work="${cuda_work_graph_ms}" -v leaf="${cuda_leaf_ms}" \
  -v path="${cuda_path_append_ms}" -v hamilton="${cuda_hamilton_reply_ms}" \
  -v end="${cuda_end_reply_ms}" 'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"
hybrid_host_build_ms="$(awk -v work="${hybrid_work_graph_ms}" -v leaf="${hybrid_leaf_ms}" \
  -v path="${hybrid_path_append_ms}" -v hamilton="${hybrid_hamilton_reply_ms}" \
  -v end="${hybrid_end_reply_ms}" 'BEGIN { printf "%.6f", work-leaf-path-hamilton-end }')"

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
  echo "benchmark_modes cpu,cuda-all,hybrid-leaf-cuda"
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
  echo "physical_gpu ${physical_gpu}"
  echo "compiler $(c++ --version | awk 'NR == 1')"
  echo "nvcc $(nvcc --version | awk '/release/ { print; exit }')"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | sed 's/^/gpu /'
  echo "END"
} >"${manifest}"

summary="${run_dir}/summary.txt"
{
  echo "CUDAEE_HT_SCAN_BENCHMARK_SUMMARY_V3"
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
  echo "cuda_candidate_ms ${cuda_candidate_ms}"
  echo "hybrid_candidate_ms ${hybrid_candidate_ms}"
  echo "cpu_work_graph_ms ${cpu_work_graph_ms}"
  echo "cuda_work_graph_ms ${cuda_work_graph_ms}"
  echo "hybrid_work_graph_ms ${hybrid_work_graph_ms}"
  echo "cpu_host_build_residual_ms ${cpu_host_build_ms}"
  echo "cuda_host_build_residual_ms ${cuda_host_build_ms}"
  echo "hybrid_host_build_residual_ms ${hybrid_host_build_ms}"
  echo "cpu_leaf_ms ${cpu_leaf_ms}"
  echo "cuda_leaf_ms ${cuda_leaf_ms}"
  echo "hybrid_leaf_ms ${hybrid_leaf_ms}"
  echo "leaf_speedup ${leaf_speedup}"
  echo "hybrid_leaf_speedup ${hybrid_leaf_speedup}"
  echo "cpu_path_append_ms ${cpu_path_append_ms}"
  echo "cuda_path_append_ms ${cuda_path_append_ms}"
  echo "hybrid_path_append_ms ${hybrid_path_append_ms}"
  echo "cpu_hamilton_reply_ms ${cpu_hamilton_reply_ms}"
  echo "cuda_hamilton_reply_ms ${cuda_hamilton_reply_ms}"
  echo "hybrid_hamilton_reply_ms ${hybrid_hamilton_reply_ms}"
  echo "cpu_end_reply_ms ${cpu_end_reply_ms}"
  echo "cuda_end_reply_ms ${cuda_end_reply_ms}"
  echo "hybrid_end_reply_ms ${hybrid_end_reply_ms}"
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
  echo "cpu_search_ms ${cpu_search_ms}"
  echo "cuda_search_ms ${cuda_search_ms}"
  echo "search_speedup ${search_speedup}"
  echo "hybrid_search_ms ${hybrid_search_ms}"
  echo "hybrid_search_speedup ${hybrid_search_speedup}"
  echo "cpu_total_ms ${cpu_total_ms}"
  echo "cuda_total_ms ${cuda_total_ms}"
  echo "hybrid_total_ms ${hybrid_total_ms}"
  echo "cpu_wall_ms ${cpu_wall_ms}"
  echo "cuda_wall_ms ${cuda_wall_ms}"
  echo "wall_speedup ${wall_speedup}"
  echo "hybrid_wall_ms ${hybrid_wall_ms}"
  echo "hybrid_wall_speedup ${hybrid_wall_speedup}"
  echo "initial_hash ${initial_hash}"
  echo "final_hash ${final_hash}"
  echo "verified_edge_sha256 $(sha256sum "${run_dir}/cuda.edg" | awk '{ print $1 }')"
  echo "protected_tour_checked ${protected_tour_checked}"
  echo "protected_tour_hash ${protected_tour_hash}"
  echo "END"
} >"${summary}"

echo "完成：${run_dir}"
sed -n '1,240p' "${summary}"
