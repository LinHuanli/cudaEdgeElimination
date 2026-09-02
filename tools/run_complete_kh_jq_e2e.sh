#!/usr/bin/env bash
set -euo pipefail

# 完整单 GPU 端到端：完全图 -> CUDA JV -> 作者 KH -q/HS -> CUDA JV -> 门禁。
# KH OpenMP 的扫描顺序可能随调度变化；重复运行只用于报告方差，最终选择的每一份
# 边集都必须单独通过作者算法、格式校验和最优 tour 门禁。

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

instance="${1:-}"
physical_gpu="${2:-}"
repetitions="${3:-1}"
if [[ -z "${instance}" || ! "${instance}" =~ ^[A-Za-z0-9._-]+$ ||
      ! "${repetitions}" =~ ^[1-9][0-9]*$ ]] || (( repetitions > 9 )); then
  echo "用法：$0 INSTANCE [PHYSICAL_GPU] [REPETITIONS，默认 1]" >&2
  exit 2
fi

if [[ -z "${physical_gpu}" ]]; then
  physical_gpu="$("${repo_root}/tools/select_gpu.sh")"
fi
if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "错误：PHYSICAL_GPU 必须是 nvidia-smi 中的非负设备号。" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" && "${CUDAEE_ALLOW_DIRTY_BENCHMARK:-0}" != "1" ]]; then
  echo "错误：正式端到端要求 clean worktree；调试时可设置 CUDAEE_ALLOW_DIRTY_BENCHMARK=1。" >&2
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
if [[ ! "${dimension}" =~ ^[1-9][0-9]*$ || ! "${optimum}" =~ ^[0-9]+$ ||
      ! "${complete_edge_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：实例配置中的维度、最优值或完全图边数非法。" >&2
  exit 2
fi

tsp_source="$(realpath "${repo_root}/${tsp_relative}")"
tour_source="$(realpath "${repo_root}/${tour_relative}")"
if [[ ! -f "${tsp_source}" || ! -f "${tour_source}" ]]; then
  echo "错误：TSPLIB 或最优 tour 不存在。" >&2
  exit 2
fi
actual_tsp_sha="$(sha256sum "${tsp_source}" | awk '{print $1}')"
actual_tour_sha="$(sha256sum "${tour_source}" | awk '{print $1}')"
if [[ "${actual_tsp_sha}" != "${tsp_sha}" || "${actual_tour_sha}" != "${tour_sha}" ]]; then
  echo "错误：外部输入 SHA-256 与锁定配置不一致。" >&2
  exit 2
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
  exit 2
fi
read -r gpu_uuid gpu_memory_used gpu_utilization <<<"${gpu_row}"
if [[ "${CUDAEE_ALLOW_BUSY_GPU:-0}" != "1" ]] &&
   (( gpu_memory_used > 512 || gpu_utilization > 10 )); then
  echo "错误：GPU ${physical_gpu} 非空闲（memory=${gpu_memory_used} MiB, util=${gpu_utilization}%）。" >&2
  exit 2
fi

cuda_preset="${CUDAEE_CUDA_PRESET:-cuda-release}"
omp_threads="${CUDAEE_KH_OMP_THREADS:-16}"
max_fixed_point_rounds="${CUDAEE_MAX_FIXED_POINT_ROUNDS:-20}"
if [[ ! "${cuda_preset}" =~ ^cuda(-sm[0-9]+)?-release$ ||
      ! "${omp_threads}" =~ ^[1-9][0-9]*$ ||
      ! "${max_fixed_point_rounds}" =~ ^[1-9][0-9]*$ ]] ||
   (( omp_threads > 256 || max_fixed_point_rounds > 100 )); then
  echo "错误：CUDA preset、KH OpenMP 线程数或固定点轮数非法。" >&2
  exit 2
fi

cmake --preset "${cuda_preset}"
cmake --build --preset "${cuda_preset}" --target cudaee cudaee_kh_elim_omp --parallel
cudaee="${repo_root}/build/${cuda_preset}/cudaee"
kh_omp="${repo_root}/build/${cuda_preset}/cudaee_kh_elim_omp"
if [[ ! -x "${cudaee}" || ! -x "${kh_omp}" ]]; then
  echo "错误：端到端可执行文件构建失败。" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${instance}-complete-kh-jq-${timestamp}-$$"
run_dir="${repo_root}/artifacts/${run_id}"
mkdir -p "${run_dir}"

copy_uncompressed() {
  local source="$1"
  local destination="$2"
  if [[ "${source}" == *.gz ]]; then
    cp -- "${source}" "${destination}.gz"
    gzip -df "${destination}.gz"
  else
    cp -- "${source}" "${destination}"
  fi
}

tsp="${run_dir}/${instance}.tsp"
tour="${run_dir}/${instance}.opt.tour"
copy_uncompressed "${tsp_source}" "${tsp}"
copy_uncompressed "${tour_source}" "${tour}"

complete_edges="${run_dir}/complete.edg"
jv_edges="${run_dir}/jv.edg"
jv_proof="${run_dir}/jv.proof"

/usr/bin/time -f '%e' -o "${run_dir}/complete.wall-seconds" \
  "${cudaee}" complete-graph --tsp "${tsp}" --output "${complete_edges}" \
  >"${run_dir}/complete.stdout" 2>"${run_dir}/complete.stderr"
read -r complete_dimension actual_complete_edges <"${complete_edges}"
if [[ "${complete_dimension}" != "${dimension}" ||
      "${actual_complete_edges}" != "${complete_edge_count}" ]]; then
  echo "错误：生成的完全图规模不符合锁定配置。" >&2
  exit 3
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
/usr/bin/time -f '%e' -o "${run_dir}/jv.wall-seconds" \
  env CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${cudaee}" gpu-eliminate \
    --tsp "${tsp}" --edges "${complete_edges}" --output "${jv_edges}" \
    --proof "${jv_proof}" --backend cuda --max-rounds 100 \
    >"${run_dir}/jv.stdout" 2>"${run_dir}/jv.stderr"
"${cudaee}" verify --tsp "${tsp}" --edges "${complete_edges}" --proof "${jv_proof}" \
  >"${run_dir}/jv.verify.stdout" 2>"${run_dir}/jv.verify.stderr"
"${cudaee}" tour-check --tsp "${tsp}" --edges "${jv_edges}" --tour "${tour}" \
  --expected-cost "${optimum}" >"${run_dir}/jv.tour.stdout" 2>"${run_dir}/jv.tour.stderr"
read -r jv_dimension jv_edge_count <"${jv_edges}"
if [[ "${jv_dimension}" != "${dimension}" || ! "${jv_edge_count}" =~ ^[0-9]+$ ||
      "${jv_edge_count}" -gt "${complete_edge_count}" ]]; then
  echo "错误：CUDA JV 输出边文件头非法。" >&2
  exit 3
fi

best_run=0
best_remaining="${jv_edge_count}"
declare -a run_round_count run_first_remaining run_total_wall
for ((run = 1; run <= repetitions; ++run)); do
  round_input="${jv_edges}"
  previous_count="${jv_edge_count}"
  rounds_file="${run_dir}/kh.${run}.rounds.tsv"
  : >"${rounds_file}"
  total_wall="0"
  converged=0
  for ((round = 1; round <= max_fixed_point_rounds; ++round)); do
    prefix="${run_dir}/kh.${run}.round.${round}"
    kh_hs_edges="${prefix}.hs.edg"
    kh_edges="${prefix}.edg"
    kh_proof="${prefix}.jv.proof"
    /usr/bin/time -f '%e' -o "${prefix}.hs.wall-seconds" \
      env OMP_NUM_THREADS="${omp_threads}" OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread \
        "${kh_omp}" -q -t "${tour}" -o "${kh_hs_edges}" -T "${tsp}" "${round_input}" \
        >"${prefix}.hs.stdout" 2>"${prefix}.hs.stderr"
    read -r kh_dimension kh_hs_remaining <"${kh_hs_edges}"
    if [[ "${kh_dimension}" != "${dimension}" || ! "${kh_hs_remaining}" =~ ^[0-9]+$ ||
          "${kh_hs_remaining}" -gt "${previous_count}" ]]; then
      echo "错误：KH 第 ${run} 次第 ${round} 轮 HS 输出边文件头非法。" >&2
      exit 3
    fi

    /usr/bin/time -f '%e' -o "${prefix}.jv.wall-seconds" \
      env CUDA_VISIBLE_DEVICES="${gpu_uuid}" "${cudaee}" gpu-eliminate \
        --tsp "${tsp}" --edges "${kh_hs_edges}" --output "${kh_edges}" \
        --proof "${kh_proof}" --backend cuda --max-rounds 100 \
        >"${prefix}.jv.stdout" 2>"${prefix}.jv.stderr"
    "${cudaee}" verify --tsp "${tsp}" --edges "${kh_hs_edges}" --proof "${kh_proof}" \
      >"${prefix}.jv.verify.stdout" 2>"${prefix}.jv.verify.stderr"
    read -r kh_dimension kh_remaining <"${kh_edges}"
    if [[ "${kh_dimension}" != "${dimension}" || ! "${kh_remaining}" =~ ^[0-9]+$ ||
          "${kh_remaining}" -gt "${kh_hs_remaining}" ]]; then
      echo "错误：KH 第 ${run} 次第 ${round} 轮 GPU JV 输出边文件头非法。" >&2
      exit 3
    fi
    "${cudaee}" tour-check --tsp "${tsp}" --edges "${kh_edges}" --tour "${tour}" \
      --expected-cost "${optimum}" >"${prefix}.tour.stdout" 2>"${prefix}.tour.stderr"

    hs_wall="$(<"${prefix}.hs.wall-seconds")"
    jv_wall="$(<"${prefix}.jv.wall-seconds")"
    total_wall="$(awk -v total="${total_wall}" -v hs="${hs_wall}" -v jv="${jv_wall}" \
      'BEGIN { printf "%.6f", total + hs + jv }')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${round}" "${previous_count}" "${kh_hs_remaining}" "${kh_remaining}" \
      "${hs_wall}" "${jv_wall}" \
      "$(sha256sum "${kh_hs_edges}" | awk '{print $1}')" \
      "$(sha256sum "${kh_edges}" | awk '{print $1}')" \
      "$(sha256sum "${kh_proof}" | awk '{print $1}')" >>"${rounds_file}"
    if (( round == 1 )); then
      run_first_remaining[run]="${kh_remaining}"
    fi
    if (( kh_hs_remaining == previous_count && kh_remaining == kh_hs_remaining )); then
      converged=1
      run_round_count[run]="${round}"
      break
    fi
    round_input="${kh_edges}"
    previous_count="${kh_remaining}"
  done
  if (( converged == 0 )); then
    echo "错误：KH 第 ${run} 次运行在 ${max_fixed_point_rounds} 轮内未达到固定点。" >&2
    exit 3
  fi
  run_total_wall[run]="${total_wall}"
  cp -- "${kh_edges}" "${run_dir}/kh.${run}.edg"
  if (( kh_remaining < best_remaining )); then
    best_remaining="${kh_remaining}"
    best_run="${run}"
  fi
done

if (( best_run == 0 )); then
  echo "错误：KH 没有产生任何有效删除。" >&2
  exit 3
fi
cp -- "${run_dir}/kh.${best_run}.edg" "${run_dir}/final.edg"
"${cudaee}" tour-check --tsp "${tsp}" --edges "${run_dir}/final.edg" --tour "${tour}" \
  --expected-cost "${optimum}" >"${run_dir}/final.tour.stdout" \
  2>"${run_dir}/final.tour.stderr"

strength_status="NOT_CONFIGURED"
if [[ "${reference_remaining}" =~ ^[0-9]+$ ]]; then
  strength_status="PASS"
  if (( best_remaining > reference_remaining )); then
    strength_status="FAIL"
  fi
fi

manifest="${run_dir}/manifest.txt"
{
  echo "CUDAEE_COMPLETE_KH_JQ_E2E_MANIFEST_V2"
  echo "instance ${instance}"
  echo "git_commit $(git rev-parse HEAD)"
  echo "git_dirty $([[ -n "$(git status --porcelain)" ]] && echo 1 || echo 0)"
  echo "dimension ${dimension}"
  echo "tsp_source ${tsp_relative}"
  echo "tsp_source_sha256 ${actual_tsp_sha}"
  echo "tour_source ${tour_relative}"
  echo "tour_source_sha256 ${actual_tour_sha}"
  echo "certified_optimum ${optimum}"
  echo "gpu_physical_index ${physical_gpu}"
  echo "gpu_uuid ${gpu_uuid}"
  echo "cuda_preset ${cuda_preset}"
  echo "omp_threads ${omp_threads}"
  echo "max_fixed_point_rounds ${max_fixed_point_rounds}"
  echo "complete_edges ${actual_complete_edges}"
  echo "complete_sha256 $(sha256sum "${complete_edges}" | awk '{print $1}')"
  echo "complete_wall_seconds $(<"${run_dir}/complete.wall-seconds")"
  echo "jv_edges ${jv_edge_count}"
  echo "jv_deleted $((complete_edge_count - jv_edge_count))"
  echo "jv_edges_sha256 $(sha256sum "${jv_edges}" | awk '{print $1}')"
  echo "jv_proof_sha256 $(sha256sum "${jv_proof}" | awk '{print $1}')"
  echo "jv_wall_seconds $(<"${run_dir}/jv.wall-seconds")"
  echo "kh_profile fixed-point_GPU-JV_KH-q_GPU-JV"
  echo "kh_repetitions ${repetitions}"
  for ((run = 1; run <= repetitions; ++run)); do
    read -r _ run_remaining <"${run_dir}/kh.${run}.edg"
    echo "kh_run ${run} ${run_round_count[run]} ${run_first_remaining[run]} ${run_remaining} $(sha256sum "${run_dir}/kh.${run}.edg" | awk '{print $1}') ${run_total_wall[run]}"
    awk -F '\t' -v run="${run}" \
      '{ print "kh_round", run, $1, $2, $3, $4, $5, $6, $7, $8, $9 }' \
      "${run_dir}/kh.${run}.rounds.tsv"
  done
  echo "selected_run ${best_run}"
  echo "final_edges ${best_remaining}"
  echo "final_deleted $((complete_edge_count - best_remaining))"
  echo "final_deletion_ratio $(awk -v before="${complete_edge_count}" -v after="${best_remaining}" 'BEGIN { printf "%.8f", (before-after)/before }')"
  echo "final_sha256 $(sha256sum "${run_dir}/final.edg" | awk '{print $1}')"
  echo "reference_remaining ${reference_remaining}"
  echo "strength_gate ${strength_status}"
  echo "jv_proof_replayed 1"
  echo "protected_tour_checked 1"
  echo "fixed_point_converged 1"
  echo "kh_authorization author-KH-exact-HS-algorithm"
  echo "END"
} >"${manifest}"

echo "run_dir=${run_dir}"
echo "instance=${instance} complete_edges=${complete_edge_count} jv_edges=${jv_edge_count} final_edges=${best_remaining} selected_run=${best_run} strength_gate=${strength_status}"
if [[ "${strength_status}" == "FAIL" ]]; then
  echo "错误：最终剩余边 ${best_remaining} 超过参考门槛 ${reference_remaining}。" >&2
  exit 4
fi
