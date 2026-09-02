#!/usr/bin/env bash
set -euo pipefail

# V3 单 GPU正式 A/B：同一 JV 固定点、同一目标切片和规范 proof，比较
# 当前最佳 CPU wavefront 与候选/leaf CUDA + CPU 控制流的混合后端。

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

instance="${1:-}"
physical_gpu="${2:-}"
pairs="${3:-7}"
if [[ -z "${instance}" || ! "${instance}" =~ ^[A-Za-z0-9._-]+$ ||
      ! "${physical_gpu}" =~ ^[0-9]+$ || ! "${pairs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "用法：$0 INSTANCE PHYSICAL_GPU [PAIRS]" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" && "${CUDAEE_ALLOW_DIRTY_BENCHMARK:-0}" != "1" ]]; then
  echo "错误：正式 V3 A/B 要求 clean worktree；调试时可设置 CUDAEE_ALLOW_DIRTY_BENCHMARK=1。" >&2
  exit 2
fi

max_targets="${CUDAEE_HT_MAX_TARGETS:-32}"
target_offset="${CUDAEE_HT_TARGET_OFFSET:-0}"
cpu_cost_threads="${CUDAEE_CPU_COST_THREADS:-8}"
hybrid_cost_threads="${CUDAEE_HYBRID_CPU_COST_THREADS:-2}"
hybrid_target_workers="${CUDAEE_HYBRID_TARGET_WORKERS:-4}"
cuda_preset="${CUDAEE_CUDA_PRESET:-cuda-sm86-release}"
if [[ ! "${max_targets}" =~ ^[1-9][0-9]*$ || ! "${target_offset}" =~ ^[0-9]+$ ||
      ! "${cpu_cost_threads}" =~ ^[1-8]$ || ! "${hybrid_cost_threads}" =~ ^[1-8]$ ||
      ! "${hybrid_target_workers}" =~ ^[1-9][0-9]*$ ||
      ! "${cuda_preset}" =~ ^cuda(-sm[0-9]+)?-release$ ]]; then
  echo "错误：target、线程数或 CUDA preset 参数非法。" >&2
  exit 2
fi
if (( hybrid_target_workers > 32 )); then
  echo "错误：CUDAEE_HYBRID_TARGET_WORKERS 不得超过 32。" >&2
  exit 2
fi

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
# 让物理 ordinal 与 nvidia-smi 的 PCI 顺序保持一致；程序内仅暴露一张卡。
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
  echo "错误：实例输入无效，且正式 A/B 必须设置 CUDAEE_BENCHMARK_TOUR。" >&2
  exit 2
fi
tour_source="$(realpath "${tour_source}")"
if [[ ! -f "${tour_source}" ]]; then
  echo "错误：CUDAEE_BENCHMARK_TOUR 不存在。" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误：当前主机找不到 nvidia-smi。" >&2
  exit 2
fi
max_gpu_util="${CUDAEE_MAX_GPU_UTILIZATION:-10}"
max_gpu_memory_mib="${CUDAEE_MAX_GPU_MEMORY_USED_MIB:-512}"
gpu_cooldown_seconds="${CUDAEE_GPU_COOLDOWN_SECONDS:-30}"
if [[ ! "${max_gpu_util}" =~ ^[0-9]+$ || ! "${max_gpu_memory_mib}" =~ ^[0-9]+$ ||
      ! "${gpu_cooldown_seconds}" =~ ^[0-9]+$ ]] || (( gpu_cooldown_seconds > 120 )); then
  echo "错误：GPU 空闲阈值必须为非负整数，冷却等待须位于 [0,120] 秒。" >&2
  exit 2
fi

gpu_stats() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F ',' -v target="$1" '
      { gsub(/ /, "", $1); if ($1 == target) {
          gsub(/ /, "", $2); gsub(/ /, "", $3); print $2, $3
        }
      }'
}

check_gpu_idle() {
  local stats used util
  stats="$(gpu_stats "${physical_gpu}")"
  if [[ -z "${stats}" ]]; then
    echo "错误：找不到物理 GPU ${physical_gpu}。" >&2
    return 2
  fi
  read -r used util <<<"${stats}"
  if (( used > max_gpu_memory_mib || util > max_gpu_util )); then
    echo "错误：GPU ${physical_gpu} 非空闲（memory=${used} MiB, util=${util}%）。" >&2
    return 1
  fi
}

wait_for_gpu() {
  local waited=0
  while (( waited < gpu_cooldown_seconds )); do
    if check_gpu_idle >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    ((++waited))
  done
  check_gpu_idle
}

if [[ "${CUDAEE_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  check_gpu_idle
fi
gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | \
  awk -F ',' -v target="${physical_gpu}" \
    '{ gsub(/ /, "", $1); gsub(/^ +| +$/, "", $2); if ($1 == target) print $2 }')"
if [[ -z "${gpu_uuid}" ]]; then
  echo "错误：无法解析 GPU ${physical_gpu} 的 UUID。" >&2
  exit 2
fi
gpu_initial_snapshot="$(nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits)"

available_kib="$(df -Pk "${repo_root}" | awk 'NR == 2 { print $4 }')"
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "错误：仓库文件系统可用空间不足 8 GiB。" >&2
  exit 2
fi

cmake --preset cpu-release
cmake --preset "${cuda_preset}"
cmake --build --preset cpu-release --target cudaee --parallel
cmake --build --preset "${cuda_preset}" --target cudaee --parallel
cpu_binary="${repo_root}/build/cpu-release/cudaee"
cuda_binary="${repo_root}/build/${cuda_preset}/cudaee"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${instance}-v3-single-gpu-ab-${timestamp}-$$"
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

ht_common=(
  --max-targets "${max_targets}"
  --target-offset "${target_offset}"
  --target-order weight-desc
  --scheduler wavefront
  --speculation 1
  --cd-mode missing-or-incompatible
  --max-neighborhood 25
  --max-cd-candidates 5
  --max-candidate-degree 50
  --max-root-replies 10000
  --max-k 5
  --max-deletion-sets 100
  --exact-blocks 0
  --exact-backend cpu
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
)

metrics="${run_dir}/metrics.csv"
echo "mode,run,target_execution_ms,search_ms,total_ms,wall_ms,candidate_ms,work_graph_ms,leaf_ms,immediate_verify_ms,commit_ms,states,replies,leaf_calls,committed" >"${metrics}"

read_field() {
  awk -v key="$2" '$1 == key { print $2; found = 1 } END { if (!found) exit 1 }' "$1"
}

run_scan() {
  local mode="$1"
  local run="$2"
  local prefix="${run_dir}/${mode}.${run}"
  local binary omp_threads start_ns end_ns wall_ms report
  local -a backend_arguments
  report="${prefix}.report"
  if [[ "${mode}" == "cpu" ]]; then
    binary="${cpu_binary}"
    omp_threads="${cpu_cost_threads}"
    backend_arguments=(
      --target-workers 1
      --backend cpu
      --leaf-backend cpu
      --reply-backend cpu
      --path-append-backend cpu
      --propagation-backend cpu
      --fuse-leaf-buckets 1
    )
  else
    if [[ "${CUDAEE_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
      wait_for_gpu
    fi
    binary="${cuda_binary}"
    omp_threads="${hybrid_cost_threads}"
    backend_arguments=(
      --target-devices 0
      --target-workers "${hybrid_target_workers}"
      --backend cuda
      --leaf-backend cuda
      --reply-backend cpu
      --path-append-backend cpu
      --propagation-backend cpu
      --fuse-leaf-buckets 0
    )
  fi

  start_ns="$(date +%s%N)"
  CUDA_VISIBLE_DEVICES="${gpu_uuid}" OMP_NUM_THREADS="${omp_threads}" \
    "${binary}" ht-scan --tsp "${tsp}" --edges "${jv_edges}" \
    --output "${prefix}.edg" --proof "${prefix}.proof" --report "${report}" \
    --protected-tour "${tour}" --expected-cost "${certified_optimum}" \
    "${ht_common[@]}" "${backend_arguments[@]}" \
    >"${prefix}.stdout" 2>"${prefix}.stderr"
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
  awk '$1 == "attempt" { print $2,$3,$4,$5,$6,$7,$8,$9,$10 }' "${report}" \
    >"${prefix}.work-signature"

  if [[ "${mode}" == "cpu" ]]; then
    [[ "$(read_field "${report}" target_workers)" == "1" ]]
    [[ "$(read_field "${report}" target_parallel)" == "0" ]]
  else
    [[ "$(read_field "${report}" target_workers)" == "${hybrid_target_workers}" ]]
    if (( hybrid_target_workers > 1 )); then
      [[ "$(read_field "${report}" target_parallel)" == "1" ]]
    fi
    awk '
      /^target_index=/ {
        assigned = leaf = -2
        for (i = 1; i <= NF; ++i) {
          split($i, pair, "=")
          if (pair[1] == "assigned_device") assigned = pair[2] + 0
          if (pair[1] == "leaf_cost_device") leaf = pair[2] + 0
        }
        if (assigned != 0 || (leaf >= 0 && leaf != 0)) exit 10
        if (leaf == 0) observed_leaf = 1
        ++rows
      }
      END { if (rows == 0 || !observed_leaf) exit 11 }
    ' "${prefix}.stdout"
  fi

  if [[ "${run}" != "warm" ]]; then
    echo "${mode},${run},$(read_field "${report}" target_execution_ms),$(read_field "${report}" search_ms),$(read_field "${report}" total_ms),${wall_ms},$(read_field "${report}" candidate_ms),$(read_field "${report}" work_graph_ms),$(read_field "${report}" leaf_ms),$(read_field "${report}" immediate_verify_ms),$(read_field "${report}" commit_ms),$(read_field "${report}" states_expanded),$(read_field "${report}" replies_expanded),$(read_field "${report}" leaf_calls),$(read_field "${report}" committed_targets)" >>"${metrics}"
  fi
}

echo "预热：CPU wavefront"
run_scan cpu warm
echo "预热：单 GPU hybrid（物理 GPU ${physical_gpu}, UUID ${gpu_uuid}）"
run_scan hybrid warm
cmp "${run_dir}/cpu.warm.edg" "${run_dir}/hybrid.warm.edg"
cmp "${run_dir}/cpu.warm.proof.canonical" "${run_dir}/hybrid.warm.proof.canonical"
cmp "${run_dir}/cpu.warm.work-signature" "${run_dir}/hybrid.warm.work-signature"

for ((pair = 1; pair <= pairs; ++pair)); do
  if (( pair % 2 == 1 )); then
    order=(cpu hybrid)
  else
    order=(hybrid cpu)
  fi
  for mode in "${order[@]}"; do
    echo "计时 pair=${pair}/${pairs} mode=${mode}"
    run_scan "${mode}" "${pair}"
  done
  cmp "${run_dir}/cpu.${pair}.edg" "${run_dir}/hybrid.${pair}.edg"
  cmp "${run_dir}/cpu.${pair}.proof.canonical" "${run_dir}/hybrid.${pair}.proof.canonical"
  cmp "${run_dir}/cpu.${pair}.work-signature" "${run_dir}/hybrid.${pair}.work-signature"
  cmp "${run_dir}/cpu.warm.edg" "${run_dir}/cpu.${pair}.edg"
  cmp "${run_dir}/cpu.warm.proof.canonical" "${run_dir}/cpu.${pair}.proof.canonical"
  cmp "${run_dir}/cpu.warm.work-signature" "${run_dir}/cpu.${pair}.work-signature"
done

analysis="${run_dir}/analysis.txt"
awk -F ',' '
  function sort_values(values, count, i, j, temporary) {
    for (i = 1; i <= count; ++i) for (j = i + 1; j <= count; ++j) {
      if (values[j] < values[i]) {
        temporary = values[i]; values[i] = values[j]; values[j] = temporary
      }
    }
  }
  function quantile(count, fraction, position, lower, weight) {
    position = 1 + (count - 1) * fraction
    lower = int(position)
    weight = position - lower
    return lower == count ? sorted[lower] : sorted[lower] * (1 - weight) + sorted[lower + 1] * weight
  }
  function summary(mode, column, label, count, i, key) {
    delete sorted
    key = mode SUBSEP column
    count = counts[key]
    for (i = 1; i <= count; ++i) sorted[i] = values[key SUBSEP i]
    sort_values(sorted, count)
    medians[mode SUBSEP column] = quantile(count, 0.5)
    p25s[mode SUBSEP column] = quantile(count, 0.25)
    p75s[mode SUBSEP column] = quantile(count, 0.75)
  }
  NR == 1 { next }
  {
    mode = $1
    for (column = 3; column <= 11; ++column) {
      key = mode SUBSEP column
      ++counts[key]
      values[key SUBSEP counts[key]] = $column + 0
    }
    run = $2 + 0
    if (mode == "cpu") {
      for (column = 3; column <= 6; ++column) cpu_by_run[run SUBSEP column] = $column + 0
    } else {
      for (column = 3; column <= 6; ++column) hybrid_by_run[run SUBSEP column] = $column + 0
    }
    strength[mode] = $12 "," $13 "," $14 "," $15
  }
  END {
    labels[3] = "target_execution_ms"; labels[4] = "search_sum_ms"
    labels[5] = "total_ms"; labels[6] = "wall_ms"
    labels[7] = "candidate_ms"; labels[8] = "work_graph_ms"
    labels[9] = "leaf_ms"; labels[10] = "immediate_verify_ms"; labels[11] = "commit_ms"
    print "CUDAEE_V3_SINGLE_GPU_AB_ANALYSIS_V1"
    print "metric cpu_median cpu_p25 cpu_p75 hybrid_median hybrid_p25 hybrid_p75 speedup"
    for (column = 3; column <= 11; ++column) {
      summary("cpu", column, labels[column]); summary("hybrid", column, labels[column])
      cpu_median = medians["cpu" SUBSEP column]
      hybrid_median = medians["hybrid" SUBSEP column]
      printf "%s %.6f %.6f %.6f %.6f %.6f %.6f %.3f\n", labels[column],
        cpu_median, p25s["cpu" SUBSEP column], p75s["cpu" SUBSEP column],
        hybrid_median, p25s["hybrid" SUBSEP column], p75s["hybrid" SUBSEP column],
        hybrid_median == 0 ? 0 : cpu_median / hybrid_median
    }
    print "paired_metric median_speedup p25 p75"
    run_count = counts["cpu" SUBSEP 3]
    for (column = 3; column <= 6; ++column) {
      delete sorted
      for (run = 1; run <= run_count; ++run) {
        cpu_value = cpu_by_run[run SUBSEP column]
        hybrid_value = hybrid_by_run[run SUBSEP column]
        if (cpu_value <= 0 || hybrid_value <= 0) exit 21
        sorted[run] = cpu_value / hybrid_value
      }
      sort_values(sorted, run_count)
      printf "%s %.6f %.6f %.6f\n", labels[column], quantile(run_count, 0.5),
        quantile(run_count, 0.25), quantile(run_count, 0.75)
    }
    print "strength_fields states,replies,leaf_calls,committed"
    print "cpu_strength " strength["cpu"]
    print "hybrid_strength " strength["hybrid"]
    print "END"
  }
' "${metrics}" >"${analysis}"

manifest="${run_dir}/manifest.txt"
{
  echo "CUDAEE_V3_SINGLE_GPU_AB_MANIFEST_V1"
  echo "run_id ${run_id}"
  echo "instance ${instance}"
  echo "git_commit $(git rev-parse HEAD)"
  echo "git_dirty $(git status --porcelain | awk 'END { print NR == 0 ? 0 : 1 }')"
  echo "hostname $(hostname)"
  echo "physical_gpu ${physical_gpu}"
  echo "gpu_uuid ${gpu_uuid}"
  echo "visible_device 0"
  echo "cuda_device_order ${CUDA_DEVICE_ORDER}"
  echo "pairs ${pairs}"
  echo "timed_order odd=cpu/hybrid,even=hybrid/cpu"
  echo "target_offset ${target_offset}"
  echo "max_targets ${max_targets}"
  echo "cpu_cost_threads ${cpu_cost_threads}"
  echo "cpu_target_workers 1"
  echo "hybrid_cpu_cost_threads ${hybrid_cost_threads}"
  echo "hybrid_target_workers ${hybrid_target_workers}"
  echo "hybrid_gpu_phases candidate,leaf"
  echo "hybrid_cpu_phases reply,path-append,propagation,exact,verification,commit"
  echo "cuda_preset ${cuda_preset}"
  echo "max_gpu_utilization ${max_gpu_util}"
  echo "max_gpu_memory_used_mib ${max_gpu_memory_mib}"
  echo "gpu_cooldown_seconds ${gpu_cooldown_seconds}"
  echo "report_version $(head -n 1 "${run_dir}/cpu.1.report")"
  echo "tsp_sha256 $(sha256sum "${tsp}" | awk '{ print $1 }')"
  echo "source_edges_sha256 $(sha256sum "${edges}" | awk '{ print $1 }')"
  echo "jv_fixed_edges_sha256 $(sha256sum "${jv_edges}" | awk '{ print $1 }')"
  echo "protected_tour_sha256 $(sha256sum "${tour}" | awk '{ print $1 }')"
  echo "verified_edge_sha256 $(sha256sum "${run_dir}/cpu.1.edg" | awk '{ print $1 }')"
  echo "canonical_proof_sha256 $(sha256sum "${run_dir}/cpu.1.proof.canonical" | awk '{ print $1 }')"
  echo "work_signature_sha256 $(sha256sum "${run_dir}/cpu.1.work-signature" | awk '{ print $1 }')"
  echo "timed_proof_replays $((2 * pairs))"
  echo "timed_protected_tour_checks $((2 * pairs))"
  echo "total_independent_proof_replays $((2 * pairs + 3))"
  echo "total_independent_tour_checks $((2 * pairs + 3))"
  echo "compiler $(c++ --version | awk 'NR == 1')"
  echo "nvcc $(nvcc --version | awk '/release/ { print; exit }')"
  while IFS= read -r gpu_row; do echo "gpu_initial ${gpu_row}"; done <<<"${gpu_initial_snapshot}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | sed 's/^/gpu_final /'
  echo "END"
} >"${manifest}"

echo "完成：${run_dir}"
cat "${analysis}"
