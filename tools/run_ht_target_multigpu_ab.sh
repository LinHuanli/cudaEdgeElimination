#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

instance="${1:-}"
physical_first="${2:-}"
physical_second="${3:-}"
pairs="${4:-7}"
if [[ -z "${instance}" || ! "${instance}" =~ ^[A-Za-z0-9._-]+$ ||
      ! "${physical_first}" =~ ^[0-9]+$ || ! "${physical_second}" =~ ^[0-9]+$ ||
      "${physical_first}" == "${physical_second}" || ! "${pairs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "用法：$0 INSTANCE FIRST_PHYSICAL_GPU SECOND_PHYSICAL_GPU [PAIRS]" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" && "${CUDAEE_ALLOW_DIRTY_BENCHMARK:-0}" != "1" ]]; then
  echo "错误：正式多 GPU A/B 要求 clean worktree；调试时可显式设置 CUDAEE_ALLOW_DIRTY_BENCHMARK=1。" >&2
  exit 2
fi

max_targets="${CUDAEE_HT_MAX_TARGETS:-8}"
target_offset="${CUDAEE_HT_TARGET_OFFSET:-0}"
cpu_cost_threads="${CUDAEE_CPU_COST_THREADS:-4}"
if [[ ! "${max_targets}" =~ ^[1-9][0-9]*$ || ! "${target_offset}" =~ ^[0-9]+$ ||
      ! "${cpu_cost_threads}" =~ ^[1-8]$ ]]; then
  echo "错误：MAX_TARGETS/CPU_COST_THREADS 必须为正整数，offset 必须为非负整数。" >&2
  exit 2
fi
export OMP_NUM_THREADS="${cpu_cost_threads}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
# 让 CUDA_VISIBLE_DEVICES 中的物理 ordinal 与 nvidia-smi 的 PCI 顺序稳定对应。
export CUDA_DEVICE_ORDER=PCI_BUS_ID

config="${repo_root}/configs/m5_jv_instances.tsv"
row="$(awk -F '\t' -v name="${instance}" \
  '$1 == name { print; found = 1 } END { if (!found) exit 1 }' "${config}")" || {
  echo "错误：${config} 中没有实例 ${instance}。" >&2
  exit 2
}
IFS=$'\t' read -r _ tsp_relative edges_relative certified_optimum <<<"${row}"
tsp="$(realpath "${repo_root}/${tsp_relative}")"
edges="$(realpath "${repo_root}/${edges_relative}")"
tour_source="${CUDAEE_BENCHMARK_TOUR:-}"
if [[ ! -f "${tsp}" || ! -f "${edges}" || ! "${certified_optimum}" =~ ^[0-9]+$ ||
      -z "${tour_source}" ]]; then
  echo "错误：实例输入无效，且正式 A/B 必须通过 CUDAEE_BENCHMARK_TOUR 提供最优 tour。" >&2
  exit 2
fi
tour_source="$(realpath "${tour_source}")"
if [[ ! -f "${tour_source}" ]]; then
  echo "错误：CUDAEE_BENCHMARK_TOUR 不存在。" >&2
  exit 2
fi

max_gpu_util="${CUDAEE_MAX_GPU_UTILIZATION:-10}"
max_gpu_memory_mib="${CUDAEE_MAX_GPU_MEMORY_USED_MIB:-512}"
if [[ ! "${max_gpu_util}" =~ ^[0-9]+$ || ! "${max_gpu_memory_mib}" =~ ^[0-9]+$ ]]; then
  echo "错误：GPU 空闲阈值必须为非负整数。" >&2
  exit 2
fi
check_physical_gpu() {
  local device="$1"
  local stats used util
  stats="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F ',' -v target="${device}" \
    '{ gsub(/ /, "", $1); if ($1 == target) { gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3 } }')"
  if [[ -z "${stats}" ]]; then
    echo "错误：找不到物理 GPU ${device}。" >&2
    exit 2
  fi
  read -r used util <<<"${stats}"
  if (( used > max_gpu_memory_mib || util > max_gpu_util )); then
    echo "错误：物理 GPU ${device} 当前非空闲（memory=${used} MiB, util=${util}%）。" >&2
    exit 2
  fi
}
if [[ "${CUDAEE_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  check_physical_gpu "${physical_first}"
  check_physical_gpu "${physical_second}"
fi
gpu_uuid_for_index() {
  local device="$1"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F ',' \
    -v target="${device}" \
    '{ gsub(/ /, "", $1); gsub(/^ +| +$/, "", $2); if ($1 == target) print $2 }'
}
physical_first_uuid="$(gpu_uuid_for_index "${physical_first}")"
physical_second_uuid="$(gpu_uuid_for_index "${physical_second}")"
if [[ -z "${physical_first_uuid}" || -z "${physical_second_uuid}" ]]; then
  echo "错误：无法解析所选物理 GPU 的 UUID。" >&2
  exit 2
fi
visible_gpu_uuids="${physical_first_uuid},${physical_second_uuid}"
gpu_initial_snapshot="$(nvidia-smi \
  --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits)"

available_kib="$(df -Pk "${repo_root}" | awk 'NR == 2 { print $4 }')"
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "错误：仓库文件系统可用空间不足 8 GiB。" >&2
  exit 2
fi

cmake --preset cpu-release
cmake --preset cuda-release
cmake --build --preset cpu-release --target cudaee --parallel
cmake --build --preset cuda-release --target cudaee --parallel
cpu_binary="${repo_root}/build/cpu-release/cudaee"
cuda_binary="${repo_root}/build/cuda-release/cudaee"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${instance}-ht-target-multigpu-ab-${timestamp}-$$"
run_dir="${repo_root}/artifacts/${run_id}"
mkdir -p "${run_dir}"
tour="${run_dir}/protected.opt.tour"
cp -- "${tour_source}" "${tour}"

jv_edges="${run_dir}/jv-fixed.edg"
jv_proof="${run_dir}/jv-fixed.proof"
"${cpu_binary}" gpu-eliminate --tsp "${tsp}" --edges "${edges}" --output "${jv_edges}" \
  --proof "${jv_proof}" --backend cpu --max-rounds 100 \
  >"${run_dir}/jv.stdout" 2>"${run_dir}/jv.stderr"
"${cpu_binary}" verify --tsp "${tsp}" --edges "${edges}" --proof "${jv_proof}" \
  >"${run_dir}/jv.verify.stdout" 2>"${run_dir}/jv.verify.stderr"
"${cpu_binary}" tour-check --tsp "${tsp}" --edges "${jv_edges}" --tour "${tour}" \
  --expected-cost "${certified_optimum}" \
  >"${run_dir}/jv.tour.stdout" 2>"${run_dir}/jv.tour.stderr"

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
  --reuse-reply-cuda-cache 1
  --deduplicate-reply-tasks 1
  --fuse-leaf-buckets 0
  --backend cuda
  --leaf-backend cuda
  --reply-backend cuda
  --path-append-backend cuda
  --propagation-backend cuda
)

metrics="${run_dir}/metrics.csv"
echo "mode,run,target_execution_ms,search_ms,total_ms,wall_ms,candidate_ms,work_graph_ms,leaf_ms,hamilton_reply_ms,end_reply_ms,immediate_verify_ms,commit_ms,target_workers,target_parallel,states,replies,leaf_calls,committed" >"${metrics}"

read_field() {
  awk -v key="$2" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' "$1"
}

validate_device_affinity() {
  local mode="$1"
  local stdout_file="$2"
  local expected_rows="$3"
  awk -v mode="${mode}" -v expected_rows="${expected_rows}" '
    /^target_index=/ {
      assigned = selected = leaf = path = hamilton = end = -2
      for (i = 1; i <= NF; ++i) {
        split($i, pair, "=")
        if (pair[1] == "assigned_device") assigned = pair[2] + 0
        if (pair[1] == "selected_device") selected = pair[2] + 0
        if (pair[1] == "leaf_cost_device") leaf = pair[2] + 0
        if (pair[1] == "path_append_device") path = pair[2] + 0
        if (pair[1] == "hamilton_reply_device") hamilton = pair[2] + 0
        if (pair[1] == "end_reply_device") end = pair[2] + 0
      }
      expected = mode == "one" ? 0 : rows % 2
      if (assigned != expected) exit 10
      if ((selected >= 0 && selected != assigned) || (leaf >= 0 && leaf != assigned) ||
          (path >= 0 && path != assigned) || (hamilton >= 0 && hamilton != assigned) ||
          (end >= 0 && end != assigned)) exit 11
      ++rows
    }
    END { if (rows != expected_rows) exit 12 }
  ' "${stdout_file}"
}

run_scan() {
  local mode="$1"
  local run="$2"
  if [[ "${CUDAEE_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
    # 每次独立进程启动前复核，避免 warmup 之后有外部任务进入而污染正式样本。
    check_physical_gpu "${physical_first}"
    check_physical_gpu "${physical_second}"
  fi
  local target_devices="0"
  if [[ "${mode}" == "two" ]]; then
    target_devices="0,1"
  fi
  local prefix="${run_dir}/${mode}.${run}"
  local start_ns end_ns wall_ms report
  report="${prefix}.report"
  start_ns="$(date +%s%N)"
  CUDA_VISIBLE_DEVICES="${visible_gpu_uuids}" "${cuda_binary}" ht-scan \
    --tsp "${tsp}" --edges "${jv_edges}" --output "${prefix}.edg" \
    --proof "${prefix}.proof" --report "${report}" --target-devices "${target_devices}" \
    --protected-tour "${tour}" --expected-cost "${certified_optimum}" \
    "${ht_arguments[@]}" >"${prefix}.stdout" 2>"${prefix}.stderr"
  end_ns="$(date +%s%N)"
  wall_ms="$(awk -v first="${start_ns}" -v last="${end_ns}" \
    'BEGIN { printf "%.3f", (last-first)/1000000 }')"
  echo "${wall_ms}" >"${prefix}.wall-ms"

  "${cpu_binary}" verify --tsp "${tsp}" --edges "${jv_edges}" --proof "${prefix}.proof" \
    >"${prefix}.verify.stdout" 2>"${prefix}.verify.stderr"
  "${cpu_binary}" tour-check --tsp "${tsp}" --edges "${prefix}.edg" --tour "${tour}" \
    --expected-cost "${certified_optimum}" \
    >"${prefix}.tour.stdout" 2>"${prefix}.tour.stderr"
  grep -v '^metrics ' "${prefix}.proof" >"${prefix}.proof.canonical"
  awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10,$11 }' "${report}" \
    >"${prefix}.work-signature"
  validate_device_affinity "${mode}" "${prefix}.stdout" \
    "$(read_field "${report}" attempted_targets)"

  if [[ "${run}" != "warm" ]]; then
    echo "${mode},${run},$(read_field "${report}" target_execution_ms),$(read_field "${report}" search_ms),$(read_field "${report}" total_ms),${wall_ms},$(read_field "${report}" candidate_ms),$(read_field "${report}" work_graph_ms),$(read_field "${report}" leaf_ms),$(read_field "${report}" hamilton_reply_ms),$(read_field "${report}" end_reply_ms),$(read_field "${report}" immediate_verify_ms),$(read_field "${report}" commit_ms),$(read_field "${report}" target_workers),$(read_field "${report}" target_parallel),$(read_field "${report}" states_expanded),$(read_field "${report}" replies_expanded),$(read_field "${report}" leaf_calls),$(read_field "${report}" committed_targets)" >>"${metrics}"
  fi
}

if [[ "${CUDAEE_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  check_physical_gpu "${physical_first}"
  check_physical_gpu "${physical_second}"
fi
echo "预热：单 worker 使用物理 GPU ${physical_first}"
run_scan one warm
echo "预热：双 worker 使用物理 GPU ${physical_first},${physical_second}"
run_scan two warm
cmp "${run_dir}/one.warm.edg" "${run_dir}/two.warm.edg"
cmp "${run_dir}/one.warm.proof.canonical" "${run_dir}/two.warm.proof.canonical"
cmp "${run_dir}/one.warm.work-signature" "${run_dir}/two.warm.work-signature"

for ((pair = 1; pair <= pairs; ++pair)); do
  if (( pair % 2 == 1 )); then
    order=(one two)
  else
    order=(two one)
  fi
  for mode in "${order[@]}"; do
    echo "计时 pair=${pair}/${pairs} mode=${mode}"
    run_scan "${mode}" "${pair}"
  done
  cmp "${run_dir}/one.${pair}.edg" "${run_dir}/two.${pair}.edg"
  cmp "${run_dir}/one.${pair}.proof.canonical" "${run_dir}/two.${pair}.proof.canonical"
  cmp "${run_dir}/one.${pair}.work-signature" "${run_dir}/two.${pair}.work-signature"
  cmp "${run_dir}/one.warm.edg" "${run_dir}/one.${pair}.edg"
  cmp "${run_dir}/one.warm.proof.canonical" "${run_dir}/one.${pair}.proof.canonical"
  cmp "${run_dir}/one.warm.work-signature" "${run_dir}/one.${pair}.work-signature"
done

analysis="${run_dir}/analysis.txt"
awk -F ',' '
  function sort_values(values, count, i, j, temporary) {
    for (i = 1; i <= count; ++i) {
      for (j = i + 1; j <= count; ++j) {
        if (values[j] < values[i]) {
          temporary = values[i]; values[i] = values[j]; values[j] = temporary
        }
      }
    }
  }
  function median(values, count, copy, i) {
    delete copy
    for (i = 1; i <= count; ++i) copy[i] = values[i]
    sort_values(copy, count)
    return count % 2 ? copy[(count + 1) / 2] : (copy[count / 2] + copy[count / 2 + 1]) / 2
  }
  function improvement(first, second) { return first == 0 ? 0 : (first - second) * 100 / first }
  NR == 1 { next }
  $1 == "one" {
    run = $2 + 0; ++one_count
    one_execution[one_count] = $3; one_search[one_count] = $4
    one_total[one_count] = $5; one_wall[one_count] = $6
    execution_by_run[run] = $3; search_by_run[run] = $4
    total_by_run[run] = $5; wall_by_run[run] = $6
  }
  $1 == "two" {
    run = $2 + 0; ++two_count
    two_execution[two_count] = $3; two_search[two_count] = $4
    two_total[two_count] = $5; two_wall[two_count] = $6
    two_execution_by_run[run] = $3; two_search_by_run[run] = $4
    two_total_by_run[run] = $5; two_wall_by_run[run] = $6
  }
  END {
    if (one_count != two_count || one_count == 0) exit 20
    for (run = 1; run <= one_count; ++run) {
      paired_execution[run] = improvement(execution_by_run[run], two_execution_by_run[run])
      paired_search[run] = improvement(search_by_run[run], two_search_by_run[run])
      paired_total[run] = improvement(total_by_run[run], two_total_by_run[run])
      paired_wall[run] = improvement(wall_by_run[run], two_wall_by_run[run])
    }
    one_execution_median = median(one_execution, one_count)
    two_execution_median = median(two_execution, two_count)
    one_search_median = median(one_search, one_count)
    two_search_median = median(two_search, two_count)
    one_total_median = median(one_total, one_count)
    two_total_median = median(two_total, two_count)
    one_wall_median = median(one_wall, one_count)
    two_wall_median = median(two_wall, two_count)
    print "CUDAEE_HT_TARGET_MULTIGPU_AB_ANALYSIS_V1"
    print "metric one_median two_median median_improvement_pct paired_median_improvement_pct speedup"
    printf "target_execution_ms %.6f %.6f %.3f %.3f %.3f\n", one_execution_median,
      two_execution_median, improvement(one_execution_median, two_execution_median),
      median(paired_execution, one_count), one_execution_median / two_execution_median
    printf "search_sum_ms %.6f %.6f %.3f %.3f %.3f\n", one_search_median,
      two_search_median, improvement(one_search_median, two_search_median),
      median(paired_search, one_count), one_search_median / two_search_median
    printf "total_ms %.6f %.6f %.3f %.3f %.3f\n", one_total_median, two_total_median,
      improvement(one_total_median, two_total_median), median(paired_total, one_count),
      one_total_median / two_total_median
    printf "wall_ms %.6f %.6f %.3f %.3f %.3f\n", one_wall_median, two_wall_median,
      improvement(one_wall_median, two_wall_median), median(paired_wall, one_count),
      one_wall_median / two_wall_median
    print "END"
  }
' "${metrics}" >"${analysis}"

manifest="${run_dir}/manifest.txt"
{
  echo "CUDAEE_HT_TARGET_MULTIGPU_AB_MANIFEST_V1"
  echo "run_id ${run_id}"
  echo "instance ${instance}"
  echo "git_commit $(git rev-parse HEAD)"
  echo "git_dirty $(git status --porcelain | awk 'END { print NR == 0 ? 0 : 1 }')"
  echo "physical_first_gpu ${physical_first}"
  echo "physical_first_gpu_uuid ${physical_first_uuid}"
  echo "physical_second_gpu ${physical_second}"
  echo "physical_second_gpu_uuid ${physical_second_uuid}"
  echo "visible_device_order ${physical_first},${physical_second}"
  echo "visible_device_uuid_order ${visible_gpu_uuids}"
  echo "cuda_device_order ${CUDA_DEVICE_ORDER}"
  echo "one_worker_devices 0"
  echo "two_worker_devices 0,1"
  echo "pairs ${pairs}"
  echo "timed_order odd=one/two,even=two/one"
  echo "target_offset ${target_offset}"
  echo "max_targets ${max_targets}"
  echo "cpu_cost_threads_per_worker ${cpu_cost_threads}"
  echo "report_version $(head -n 1 "${run_dir}/one.1.report")"
  echo "tsp_sha256 $(sha256sum "${tsp}" | awk '{ print $1 }')"
  echo "source_edges_sha256 $(sha256sum "${edges}" | awk '{ print $1 }')"
  echo "jv_fixed_edges_sha256 $(sha256sum "${jv_edges}" | awk '{ print $1 }')"
  echo "protected_tour_sha256 $(sha256sum "${tour}" | awk '{ print $1 }')"
  echo "verified_edge_sha256 $(sha256sum "${run_dir}/one.1.edg" | awk '{ print $1 }')"
  echo "canonical_proof_sha256 $(sha256sum "${run_dir}/one.1.proof.canonical" | awk '{ print $1 }')"
  echo "work_signature_sha256 $(sha256sum "${run_dir}/one.1.work-signature" | awk '{ print $1 }')"
  echo "timed_proof_replays $((2 * pairs))"
  echo "timed_protected_tour_checks $((2 * pairs))"
  echo "total_independent_proof_replays $((2 * pairs + 3))"
  echo "total_independent_tour_checks $((2 * pairs + 3))"
  echo "compiler $(c++ --version | awk 'NR == 1')"
  echo "nvcc $(nvcc --version | awk '/release/ { print; exit }')"
  while IFS= read -r gpu_row; do echo "gpu_initial ${gpu_row}"; done <<<"${gpu_initial_snapshot}"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | sed 's/^/gpu_final /'
  echo "END"
} >"${manifest}"

echo "完成：${run_dir}"
cat "${analysis}"
