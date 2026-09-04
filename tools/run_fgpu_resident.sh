#!/usr/bin/env bash
set -euo pipefail

# FGPU-Elim 单卡全常驻复现实验：全部搜索阶段在一张 GPU 上完成，随后单独
# CPU 精确审计并写证书。外部数据只读，所有构建与实验产物都留在本仓库。

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

instance="${1:-pcb442}"
physical_gpu="${2:-}"
repetitions="${3:-7}"
if [[ ! "${instance}" =~ ^[A-Za-z0-9._-]+$ ||
      ! "${repetitions}" =~ ^[1-9][0-9]*$ ]] || (( repetitions > 21 )); then
  echo "用法：$0 [INSTANCE，默认 pcb442] [PHYSICAL_GPU] [REPETITIONS，默认 7]" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" &&
      "${CUDAEE_ALLOW_DIRTY_BENCHMARK:-0}" != "1" ]]; then
  echo "错误：正式基准要求 clean worktree；调试可设置 CUDAEE_ALLOW_DIRTY_BENCHMARK=1。" >&2
  exit 2
fi

config="${repo_root}/configs/e2e_single_gpu_instances.tsv"
row="$(awk -F '\t' -v name="${instance}" \
  '$1 == name { print; found = 1 } END { if (!found) exit 1 }' "${config}")" || {
  echo "错误：${config} 中没有实例 ${instance}。" >&2
  exit 2
}
IFS=$'\t' read -r _ dimension tsp_relative tsp_sha tour_relative tour_sha optimum \
  complete_edge_count reference_remaining <<<"${row}"
baseline_config="${repo_root}/configs/fgpu_resident_baselines.tsv"
baseline_row="$(awk -F '\t' -v name="${instance}" '$1 == name { print; exit }' \
  "${baseline_config}")"

tsp="$(realpath "${repo_root}/${tsp_relative}")"
tour="$(realpath "${repo_root}/${tour_relative}")"
if [[ "$(sha256sum "${tsp}" | awk '{print $1}')" != "${tsp_sha}" ||
      "$(sha256sum "${tour}" | awk '{print $1}')" != "${tour_sha}" ]]; then
  echo "错误：实例或最优 tour 的 SHA-256 与锁定配置不一致。" >&2
  exit 3
fi

if [[ -z "${physical_gpu}" ]]; then
  physical_gpu="$("${repo_root}/tools/select_gpu.sh")"
fi
if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "错误：PHYSICAL_GPU 必须是 nvidia-smi 中的非负设备号。" >&2
  exit 4
fi

gpu_row="$(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ',' -v target="${physical_gpu}" '
    { gsub(/ /, "", $1); if ($1 == target) {
        gsub(/^ +| +$/, "", $2); gsub(/ /, "", $3); gsub(/ /, "", $4);
        print $2, $3, $4
      }
    }')"
if [[ -z "${gpu_row}" ]]; then
  echo "错误：找不到物理 GPU ${physical_gpu}。" >&2
  exit 4
fi
read -r gpu_uuid gpu_memory_used gpu_utilization <<<"${gpu_row}"
if [[ "${CUDAEE_ALLOW_BUSY_GPU:-0}" != "1" ]] &&
   (( gpu_memory_used > 512 || gpu_utilization > 10 )); then
  echo "错误：GPU ${physical_gpu} 非空闲（memory=${gpu_memory_used} MiB, util=${gpu_utilization}%）。" >&2
  exit 4
fi

cuda_preset="${CUDAEE_CUDA_PRESET:-cuda-release}"
potential_candidates="${CUDAEE_FGPU_RESIDENT_POTENTIALS:-32}"
pdlp_iterations="${CUDAEE_FGPU_RESIDENT_PDLP_ITERATIONS:-5000}"
pdlp_epochs="${CUDAEE_FGPU_RESIDENT_PDLP_EPOCHS:-2}"
hs_epochs="${CUDAEE_FGPU_RESIDENT_HS_EPOCHS:-100}"
jv_rounds="${CUDAEE_FGPU_RESIDENT_JV_ROUNDS:-100}"
omp_threads="${CUDAEE_FGPU_RESIDENT_OMP_THREADS:-16}"
for value in "${potential_candidates}" "${pdlp_iterations}" "${pdlp_epochs}" \
             "${hs_epochs}" "${jv_rounds}" "${omp_threads}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：resident 数值配置必须是正整数。" >&2
    exit 5
  fi
done

cmake --preset "${cuda_preset}"
cmake --build --preset "${cuda_preset}" --target fgpu-elim --parallel
exe="${repo_root}/build/${cuda_preset}/fgpu-elim"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${repo_root}/artifacts/${instance}-fgpu-resident-${timestamp}-$$"
mkdir -p "${run_dir}"
measurements="${run_dir}/measurements.tsv"
printf 'run\tprocess_wall_s\tgpu_solve_ms\tcpu_audit_ms\ttrusted_total_ms\tfinal_edges\tfinal_hash\n' \
  >"${measurements}"

reference_edge_sha=""
reference_certificate_sha=""
for ((run = 1; run <= repetitions; ++run)); do
  output_prefix="${run_dir}/run-${run}"
  mkdir -p "${output_prefix}"
  output_edges="${output_prefix}/${instance}.edg"
  fixed="${output_prefix}/${instance}.fix"
  nonpairs="${output_prefix}/${instance}.nonpairs"
  certificate="${output_prefix}/${instance}.fgcert"
  app_manifest="${output_prefix}/${instance}.manifest"
  command=(
    "${exe}" resident
    --instance "${tsp}"
    --tour "${tour}"
    --tour-role known-optimum
    --expected-cost "${optimum}"
    --device 0
    --potential-candidates "${potential_candidates}"
    --pdlp-iterations "${pdlp_iterations}"
    --max-pdlp-epochs "${pdlp_epochs}"
    --max-hs-epochs "${hs_epochs}"
    --max-jv-rounds "${jv_rounds}"
    --enable-geometry 1
    --enable-pdlp 1
    --enable-quick-hs 1
    --enable-jv 1
    --output-edges "${output_edges}"
    --fixed "${fixed}"
    --nonpairs "${nonpairs}"
    --certificate "${certificate}"
    --manifest "${app_manifest}"
  )

  /usr/bin/time -f '%e' -o "${output_prefix}/process.wall-seconds" \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
      OMP_NUM_THREADS="${omp_threads}" "${command[@]}" \
      >"${output_prefix}/run.stdout" 2>"${output_prefix}/run.stderr"
  "${exe}" verify --instance "${tsp}" --tour "${tour}" --tour-role known-optimum \
    --expected-cost "${optimum}" --output-edges "${output_edges}" --fixed "${fixed}" \
    --nonpairs "${nonpairs}" --certificate "${certificate}" \
    >"${output_prefix}/verify.stdout" 2>"${output_prefix}/verify.stderr"

  read -r output_dimension output_count <"${output_edges}"
  if [[ "${output_dimension}" != "${dimension}" ||
        "${output_count}" -gt "${complete_edge_count}" ]]; then
    echo "错误：第 ${run} 次最终边文件规模门禁失败。" >&2
    exit 6
  fi
  if ! grep -q 'status=OK mode=resident.*converged=1' "${output_prefix}/run.stdout" ||
     ! grep -q 'status=VERIFIED' "${output_prefix}/verify.stdout"; then
    echo "错误：第 ${run} 次运行未收敛或证书未通过独立重放。" >&2
    exit 6
  fi

  edge_sha="$(sha256sum "${output_edges}" | awk '{print $1}')"
  certificate_sha="$(sha256sum "${certificate}" | awk '{print $1}')"
  if (( run == 1 )); then
    reference_edge_sha="${edge_sha}"
    reference_certificate_sha="${certificate_sha}"
  elif [[ "${edge_sha}" != "${reference_edge_sha}" ||
          "${certificate_sha}" != "${reference_certificate_sha}" ]]; then
    echo "错误：第 ${run} 次输出与第 1 次不确定。" >&2
    exit 7
  fi

  process_wall="$(<"${output_prefix}/process.wall-seconds")"
  gpu_solve="$(awk '$1 == "gpu_solve_wall_ms" { print $2 }' "${app_manifest}")"
  cpu_audit="$(awk '$1 == "cpu_audit_ms" { print $2 }' "${app_manifest}")"
  trusted_total="$(awk '$1 == "trusted_total_ms" { print $2 }' "${app_manifest}")"
  final_hash="$(awk '$1 == "final_hash" { print $2 }' "${app_manifest}")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${run}" "${process_wall}" \
    "${gpu_solve}" "${cpu_audit}" "${trusted_total}" "${output_count}" "${final_hash}" \
    >>"${measurements}"
  cat "${output_prefix}/run.stdout"
done

median_column() {
  local column="$1"
  tail -n +2 "${measurements}" | cut -f "${column}" | sort -n |
    awk '{ value[NR] = $1 } END {
      if (NR % 2) printf "%.6f", value[(NR + 1) / 2];
      else printf "%.6f", (value[NR / 2] + value[NR / 2 + 1]) / 2
    }'
}

process_median="$(median_column 2)"
gpu_median="$(median_column 3)"
audit_median="$(median_column 4)"
trusted_median="$(median_column 5)"
final_edges="$(awk 'NR == 2 { print $6 }' "${measurements}")"
author_single_edges="NA"
author_single_wall="NA"
author_fixed_edges="NA"
author_fixed_wall="NA"
author_speedup="NA"
fixed_point_speedup="NA"
if [[ -n "${baseline_row}" ]]; then
  IFS=$'\t' read -r _ author_single_edges author_single_wall author_fixed_edges \
    author_fixed_wall <<<"${baseline_row}"
  author_speedup="$(awk -v baseline="${author_single_wall}" -v current="${process_median}" \
    'BEGIN { printf "%.6f", baseline / current }')"
  fixed_point_speedup="$(awk -v baseline="${author_fixed_wall}" -v current="${process_median}" \
    'BEGIN { printf "%.6f", baseline / current }')"
fi

{
  echo "CUDAEE_FGPU_RESIDENT_BENCHMARK_V1"
  echo "git_commit $(git rev-parse HEAD)"
  echo "git_dirty $([[ -n "$(git status --porcelain)" ]] && echo 1 || echo 0)"
  echo "instance ${instance}"
  echo "dimension ${dimension}"
  echo "complete_edges ${complete_edge_count}"
  echo "tsp_sha256 ${tsp_sha}"
  echo "tour_sha256 ${tour_sha}"
  echo "optimum ${optimum}"
  echo "gpu_physical_index ${physical_gpu}"
  echo "gpu_uuid ${gpu_uuid}"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "${physical_gpu}" |
    sed 's/^/gpu /'
  echo "repetitions ${repetitions}"
  echo "process_wall_median_s ${process_median}"
  echo "gpu_solve_median_ms ${gpu_median}"
  echo "cpu_audit_median_ms ${audit_median}"
  echo "trusted_total_median_ms ${trusted_median}"
  echo "final_edges ${final_edges}"
  echo "edge_sha256 ${reference_edge_sha}"
  echo "certificate_sha256 ${reference_certificate_sha}"
  echo "configured_strength_ceiling ${reference_remaining}"
  echo "author_single_pass_baseline_s ${author_single_wall}"
  echo "author_single_pass_reference_edges ${author_single_edges}"
  echo "author_single_pass_e2e_speedup ${author_speedup}"
  echo "author_fixed_point_baseline_s ${author_fixed_wall}"
  echo "author_fixed_point_reference_edges ${author_fixed_edges}"
  echo "author_fixed_point_e2e_speedup ${fixed_point_speedup}"
  echo "END"
} >"${run_dir}/benchmark.manifest"

echo "完成：${run_dir}"
echo "可信进程端到端中位数 ${process_median}s；最终边 ${final_edges}；相对作者固定点 ${fixed_point_speedup}x。"
