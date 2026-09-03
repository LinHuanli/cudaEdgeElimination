#!/usr/bin/env bash
set -euo pipefail

# FGPU-Elim 单 GPU 可复现实验。外部 TSPLIB 只读，所有运行产物严格写入仓库 artifacts/。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

instance="${1:-pcb442}"
physical_gpu="${2:-}"
config="${repo_root}/configs/e2e_single_gpu_instances.tsv"
row="$(awk -F '\t' -v key="${instance}" '$1 == key { print; exit }' "${config}")"
if [[ -z "${row}" ]]; then
  echo "错误：configs/e2e_single_gpu_instances.tsv 中没有 ${instance}。" >&2
  exit 2
fi
IFS=$'\t' read -r _ dimension tsp_relative tsp_sha tour_relative tour_sha optimum complete_edges _ \
  <<<"${row}"
tsp="$(realpath "${repo_root}/${tsp_relative}")"
tour="$(realpath "${repo_root}/${tour_relative}")"
if [[ "$(sha256sum "${tsp}" | awk '{print $1}')" != "${tsp_sha}" ||
      "$(sha256sum "${tour}" | awk '{print $1}')" != "${tour_sha}" ]]; then
  echo "错误：实例或最优 tour 的 SHA-256 与锁定配置不一致。" >&2
  exit 3
fi

if [[ -z "${physical_gpu}" ]]; then
  physical_gpu="$(tools/select_gpu.sh)"
fi
if ! [[ "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "错误：物理 GPU ordinal 非法。" >&2
  exit 4
fi

preset="${CUDAEE_CUDA_PRESET:-cuda-release}"
potential_k="${CUDAEE_FGPU_POTENTIAL_K:-32}"
geometry_witnesses="${CUDAEE_FGPU_GEOMETRY_WITNESSES:-8}"
pdlp_iterations="${CUDAEE_FGPU_PDLP_ITERATIONS:-5000}"
pdlp_epochs="${CUDAEE_FGPU_PDLP_EPOCHS:-8}"
enable_ht="${CUDAEE_FGPU_ENABLE_HT:-0}"
# 完整 sweep 在每次 HT 提交后会从 offset=0 重排；16 轮对 pcb442
# 不足以越过“提交轮 + 最终无提交分片”。64 仍是安全预算：若耗尽会
# 明确输出 ht-epoch-limit，绝不冒充固定点。
ht_epochs="${CUDAEE_FGPU_HT_EPOCHS:-64}"
ht_targets="${CUDAEE_FGPU_HT_TARGETS:-512}"
ht_workers="${CUDAEE_FGPU_HT_WORKERS:-4}"
omp_threads="${CUDAEE_FGPU_OMP_THREADS:-8}"
cpu_cost_threads="${CUDAEE_FGPU_CPU_COST_THREADS:-2}"
allow_partial="${CUDAEE_FGPU_ALLOW_PARTIAL:-0}"

for value in "${potential_k}" "${geometry_witnesses}" "${pdlp_iterations}" "${pdlp_epochs}" \
             "${enable_ht}" "${ht_epochs}" "${ht_targets}" "${ht_workers}" \
             "${omp_threads}" "${cpu_cost_threads}" "${allow_partial}"; do
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "错误：FGPU 数值环境参数必须是非负整数。" >&2
    exit 5
  fi
done
if (( enable_ht > 1 || allow_partial > 1 )); then
  echo "错误：CUDAEE_FGPU_ENABLE_HT 和 CUDAEE_FGPU_ALLOW_PARTIAL 必须是 0 或 1。" >&2
  exit 5
fi

cmake --build --preset "${preset}" --target fgpu-elim --parallel
exe="${repo_root}/build/${preset}/fgpu-elim"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${repo_root}/artifacts/${instance}-fgpu-oneshot-${timestamp}-$$"
mkdir -p "${run_dir}"

output_edges="${run_dir}/${instance}.edg"
fixed="${run_dir}/${instance}.fix"
nonpairs="${run_dir}/${instance}.nonpairs"
certificate="${run_dir}/${instance}.fgcert"
app_manifest="${run_dir}/${instance}.manifest"

command=(
  "${exe}" run
  --instance "${tsp}"
  --tour "${tour}"
  --tour-role known-optimum
  --expected-cost "${optimum}"
  --device 0
  --numeric mixed-safe
  --verification epoch
  --pdlp native
  --pdlp-iterations "${pdlp_iterations}"
  --max-pdlp-epochs "${pdlp_epochs}"
  --potential-candidates "${potential_k}"
  --geometry-witnesses "${geometry_witnesses}"
  --max-jv-rounds 100
  --max-ht-epochs "${ht_epochs}"
  --ht-targets-per-epoch "${ht_targets}"
  --ht-target-workers "${ht_workers}"
  --max-paths 6
  --max-local-nodes 32
  --enable-geometry 1
  --enable-jv 1
  --enable-ht "${enable_ht}"
  --output-edges "${output_edges}"
  --fixed "${fixed}"
  --nonpairs "${nonpairs}"
  --certificate "${certificate}"
  --manifest "${app_manifest}"
)

start_epoch="$(date +%s)"
OMP_NUM_THREADS="${omp_threads}" CUDAEE_CPU_COST_THREADS="${cpu_cost_threads}" \
CUDA_VISIBLE_DEVICES="${physical_gpu}" "${command[@]}" | tee "${run_dir}/run.stdout"
end_epoch="$(date +%s)"

OMP_NUM_THREADS="${omp_threads}" "${exe}" verify \
  --instance "${tsp}" --tour "${tour}" --tour-role known-optimum \
  --expected-cost "${optimum}" --output-edges "${output_edges}" --fixed "${fixed}" \
  --nonpairs "${nonpairs}" --certificate "${certificate}" | tee "${run_dir}/verify.stdout"

termination="$(awk '{ for (i = 1; i <= NF; ++i) if ($i ~ /^termination=/) { sub(/^termination=/, "", $i); print $i; exit } }' "${run_dir}/run.stdout")"
if [[ -z "${termination}" ]]; then
  echo "错误：运行输出缺少 termination 字段。" >&2
  exit 6
fi
if (( allow_partial == 0 )) && [[ "${termination}" == *-limit || "${termination}" == *-partial ]]; then
  echo "错误：搜索以 ${termination} 结束，未达固定点；产物已验证但不冒充完整运行。" >&2
  exit 6
fi

read -r output_dimension output_count <"${output_edges}"
if [[ "${output_dimension}" != "${dimension}" || "${output_count}" -gt "${complete_edges}" ]]; then
  echo "错误：最终边文件规模门禁失败。" >&2
  exit 6
fi

{
  echo "CUDAEE_FGPU_ONESHOT_RUN_V1"
  echo "git_commit $(git rev-parse HEAD)"
  echo "git_dirty $(git status --porcelain | wc -l)"
  echo "instance ${instance}"
  echo "dimension ${dimension}"
  echo "tsp_sha256 ${tsp_sha}"
  echo "tour_sha256 ${tour_sha}"
  echo "optimum ${optimum}"
  echo "physical_gpu ${physical_gpu}"
  nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader -i "${physical_gpu}" \
    | sed 's/^/gpu /'
  echo "start_epoch ${start_epoch}"
  echo "end_epoch ${end_epoch}"
  echo "wall_seconds $((end_epoch - start_epoch))"
  echo "termination ${termination}"
  echo "final_edges ${output_count}"
  echo "edge_sha256 $(sha256sum "${output_edges}" | awk '{print $1}')"
  echo "fixed_sha256 $(sha256sum "${fixed}" | awk '{print $1}')"
  echo "nonpairs_sha256 $(sha256sum "${nonpairs}" | awk '{print $1}')"
  echo "certificate_sha256 $(sha256sum "${certificate}" | awk '{print $1}')"
  printf 'command'
  printf ' %q' "${command[@]}"
  printf '\nEND\n'
} >"${run_dir}/run-manifest.txt"

echo "完成：${run_dir}"
