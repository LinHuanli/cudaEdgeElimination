#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

source_config="${repo_root}/configs/m5_opt_tours.tsv"
instance_config="${repo_root}/configs/m5_jv_instances.tsv"
destination="${repo_root}/artifacts/lkh-tours"
mkdir -p "${repo_root}/.tmp" "${destination}"
temp_dir="$(mktemp -d "${repo_root}/.tmp/m5-opt-tours.XXXXXX")"
cleanup() {
  rm -rf -- "${temp_dir}"
}
trap cleanup EXIT

if [[ ! -f build/cpu-release/CMakeCache.txt ]]; then
  cmake --preset cpu-release
fi
cmake --build --preset cpu-release --target cudaee --parallel
binary="${repo_root}/build/cpu-release/cudaee"

while IFS=$'\t' read -r instance url expected_sha256 filename; do
  if [[ -z "${instance}" || "${instance}" == \#* ]]; then
    continue
  fi
  if [[ ! "${instance}" =~ ^[A-Za-z0-9._-]+$ ||
        ! "${filename}" =~ ^[A-Za-z0-9._-]+\.tour$ ||
        ! "${expected_sha256}" =~ ^[0-9a-f]{64}$ ||
        ! "${url}" =~ ^https?:// ]]; then
    echo "错误：${source_config} 中的 tour 来源记录无效。" >&2
    exit 2
  fi

  instance_row="$(awk -F '\t' -v name="${instance}" \
    '$1 == name { print; found = 1 } END { if (!found) exit 1 }' "${instance_config}")" || {
    echo "错误：${instance_config} 中没有实例 ${instance}。" >&2
    exit 2
  }
  IFS=$'\t' read -r _ tsp_relative edges_relative certified_optimum <<<"${instance_row}"
  tsp="$(realpath "${repo_root}/${tsp_relative}")"
  edges="$(realpath "${repo_root}/${edges_relative}")"
  output="${destination}/${filename}"
  candidate="${output}"

  actual_sha256=""
  if [[ -f "${output}" ]]; then
    actual_sha256="$(sha256sum "${output}" | awk '{ print $1 }')"
  fi
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    candidate="${temp_dir}/${filename}"
    echo "下载 ${instance} 最优 tour"
    curl --fail --location --silent --show-error --max-time 120 \
      --proto '=http,https' --proto-redir '=http,https' \
      "${url}" --output "${candidate}"
    actual_sha256="$(sha256sum "${candidate}" | awk '{ print $1 }')"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
      echo "错误：${instance} tour SHA-256 与锁定值不一致。" >&2
      exit 1
    fi
  fi

  # 不信任外部文件的声明成本：重新检查节点置换、TSPLIB 整数距离和稀疏图边完整性。
  "${binary}" tour-check --tsp "${tsp}" --edges "${edges}" --tour "${candidate}" \
    --expected-cost "${certified_optimum}"
  if [[ "${candidate}" != "${output}" ]]; then
    install -m 0644 -- "${candidate}" "${output}"
  fi
  echo "已锁定 ${output} sha256=${expected_sha256}"
done <"${source_config}"
