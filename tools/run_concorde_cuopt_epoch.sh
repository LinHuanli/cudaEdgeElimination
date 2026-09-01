#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"
source tools/env.sh

nodes=20
seed=1
timeout_seconds=120
tamper_model_hash=0
tsp_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nodes)
      nodes="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --timeout)
      timeout_seconds="$2"
      shift 2
      ;;
    --tsp)
      tsp_file="$2"
      shift 2
      ;;
    --tamper-model-hash)
      tamper_model_hash=1
      shift
      ;;
    *)
      echo "错误：未知参数 $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! "${nodes}" =~ ^[0-9]+$ ]] || ((nodes < 10 || nodes > 10000)); then
  echo "错误：--nodes 必须在 [10, 10000]。" >&2
  exit 2
fi
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
  echo "错误：--seed 必须是非负整数。" >&2
  exit 2
fi
if [[ ! "${timeout_seconds}" =~ ^[0-9]+$ ]] ||
   ((timeout_seconds < 1 || timeout_seconds > 3600)); then
  echo "错误：--timeout 必须在 [1, 3600] 秒。" >&2
  exit 2
fi
if [[ -n "${tsp_file}" ]]; then
  tsp_file="$(realpath -e "${tsp_file}")"
  if [[ ! -f "${tsp_file}" ]]; then
    echo "错误：--tsp 不是普通文件。" >&2
    exit 2
  fi
fi
if [[ -z "${CUDAEE_CUOPT_LIBRARY:-}" ]]; then
  echo "错误：项目内尚未安装 cuOpt，请先运行 tools/bootstrap.sh。" >&2
  exit 2
fi
if [[ ! -x build/cuda-release/cudaee ]]; then
  echo "错误：缺少 build/cuda-release/cudaee，请先运行 tools/bootstrap.sh。" >&2
  exit 2
fi
if [[ ! -x build/concorde/TSP/concorde ]]; then
  echo "错误：缺少 Concorde，请先运行 tools/bootstrap_concorde.sh。" >&2
  exit 2
fi

mkdir -p artifacts
run_dir="$(mktemp -d "${repo_root}/artifacts/concorde-cuopt.XXXXXX")"
epoch="${run_dir}/model.lp-epoch"
solution="${run_dir}/model.lp-solution"
solution_tmp="${solution}.tmp.$$"
certificate="${run_dir}/model.exact-certificate"
log="${run_dir}/concorde.log"
concorde_pid=""

stop_concorde() {
  if [[ -n "${concorde_pid}" ]] && kill -0 "${concorde_pid}" 2>/dev/null; then
    kill "${concorde_pid}" 2>/dev/null || true
    wait "${concorde_pid}" 2>/dev/null || true
  fi
}
trap stop_concorde EXIT

# Concorde 持有 cut loop；导出稳定 epoch 后等待 sidecar 原子发布解。
if [[ -n "${tsp_file}" ]]; then
  concorde_arguments=(-s "${seed}" -B "${tsp_file}")
else
  concorde_arguments=(-k "${nodes}" -s "${seed}" -B)
fi
(
  cd "${run_dir}"
  CUDAEE_LP_EPOCH_OUT="${epoch}" \
  CUDAEE_LP_SOLUTION_IN="${solution}" \
  CUDAEE_EXACT_CERT_OUT="${certificate}" \
  CUDAEE_SIDECAR_TIMEOUT="${timeout_seconds}" \
    "${repo_root}/build/concorde/TSP/concorde" \
      "${concorde_arguments[@]}" >"${log}" 2>&1
) &
concorde_pid=$!

epoch_ready=0
for ((attempt = 0; attempt < timeout_seconds * 10; attempt++)); do
  if [[ -s "${epoch}" ]] && rg -q '^END$' "${epoch}"; then
    epoch_ready=1
    break
  fi
  if ! kill -0 "${concorde_pid}" 2>/dev/null; then
    break
  fi
  sleep 0.1
done
if ((epoch_ready != 1)); then
  echo "错误：Concorde 未在时限内导出完整 epoch；日志位于 ${log}" >&2
  tail -n 80 "${log}" >&2 || true
  exit 3
fi

physical_gpu="$(tools/select_gpu.sh)"
CUDA_VISIBLE_DEVICES="${physical_gpu}" build/cuda-release/cudaee lp-solve \
  --input "${epoch}" \
  --output "${solution_tmp}" \
  --cuopt-library "${CUDAEE_CUOPT_LIBRARY}"
if ((tamper_model_hash == 1)); then
  # 负向门禁测试：只篡改待发布副本，原始 epoch 保持不变。
  sed -i 's/^model_hash .*/model_hash 0000000000000000/' "${solution_tmp}"
fi
mv "${solution_tmp}" "${solution}"

if wait "${concorde_pid}"; then
  concorde_status=0
else
  concorde_status=$?
fi
concorde_pid=""

if ((tamper_model_hash == 1)); then
  if ((concorde_status == 0)); then
    echo "错误：Concorde 接受了被篡改的模型哈希。" >&2
    exit 5
  fi
  rg -q '^cudaee exact price: solution gate or hash failed$' "${log}"
  if [[ -e "${certificate}" ]]; then
    echo "错误：哈希拒绝路径不应生成精确定价证书。" >&2
    exit 5
  fi
  printf 'Concorde hash rejection passed: gpu=%s artifacts=%s\n' \
    "${physical_gpu}" "${run_dir}"
  exit 0
fi

if ((concorde_status != 0)); then
  concorde_pid=""
  echo "错误：Concorde 精确定价失败；日志位于 ${log}" >&2
  tail -n 120 "${log}" >&2 || true
  exit 4
fi

epoch_hash="$(awk '$1 == "hash" {print $2}' "${epoch}")"
solution_hash="$(awk '$1 == "model_hash" {print $2}' "${solution}")"
certificate_hash="$(awk '$1 == "model_hash" {print $2}' "${certificate}")"
cuopt_objective="$(awk '$1 == "objective" {print $2}' "${solution}")"
exact_bound="$(awk '$1 == "exact_lower_bound" {print $2}' "${certificate}")"
upper_bound="$(awk '$1 == "upper_bound" {print $2}' "${certificate}")"

[[ -n "${epoch_hash}" && "${epoch_hash}" == "${solution_hash}" &&
   "${epoch_hash}" == "${certificate_hash}" ]]
rg -q '^status OPTIMAL$' "${solution}"
rg -q '^numerically_accepted 1$' "${solution}"
rg -q '^status EXACT_PRICED$' "${certificate}"
rg -q '^complete_graph 1$' "${certificate}"
rg -q '^Pricing COMPLETE GRAPH$' "${log}"
awk -v exact="${exact_bound}" -v model="${cuopt_objective}" \
  -v upper="${upper_bound}" \
  'BEGIN { exit !((exact <= model + 1e-7) && (exact <= upper + 1e-7)) }'

printf 'Concorde/cuOpt epoch passed: gpu=%s hash=%s model_obj=%s exact_lb=%s upper=%s\n' \
  "${physical_gpu}" "${epoch_hash}" "${cuopt_objective}" \
  "${exact_bound}" "${upper_bound}"
printf 'artifacts=%s\n' "${run_dir}"
