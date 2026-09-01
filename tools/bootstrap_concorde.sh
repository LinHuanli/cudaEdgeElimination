#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_root="${repo_root}/../references/concorde_code"
qsopt_dir="${repo_root}/.deps/qsopt"
patch_dir="${repo_root}/integrations/concorde/patches"

if [[ ! -f "${source_root}/configure" ]]; then
  echo "错误：未找到外部 Concorde 源码 ${source_root}" >&2
  exit 2
fi
mkdir -p "${repo_root}/.deps" "${repo_root}/.tmp" "${qsopt_dir}"

expected_tree_hash="f9caef4b41140a48cec906cf0456c266a2e83e5a1392b355694a99b151a819f6"
actual_tree_hash="$(cd "${source_root}" && find . -type f -print0 | sort -z |
  xargs -0 sha256sum | sha256sum | awk '{print $1}')"
if [[ "${actual_tree_hash}" != "${expected_tree_hash}" ]]; then
  echo "错误：外部 Concorde 源码哈希不匹配，拒绝应用补丁。" >&2
  exit 3
fi

download_and_verify() {
  local url="$1"
  local output="$2"
  local expected="$3"
  if [[ ! -f "${output}" ]]; then
    curl --fail --location --retry 3 --output "${output}" "${url}"
  fi
  printf '%s  %s\n' "${expected}" "${output}" | sha256sum --check --status
}

download_and_verify \
  "https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.a" \
  "${qsopt_dir}/qsopt.a" \
  "5dcf323c7fce85e8b9de7ce79aabc17b672e224b77e2a89370c4e35da07434ee"
download_and_verify \
  "https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt.h" \
  "${qsopt_dir}/qsopt.h" \
  "647729f1bd77e1263ecf35e1897c705ef1cb45e2d65dbd9cb8fdf5df5ae65624"

patch_series_hash="$(find "${patch_dir}" -maxdepth 1 -type f -name '*.patch' \
  -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | awk '{print $1}' | sha256sum | \
  awk '{print $1}')"
overlay="${repo_root}/.deps/concorde-03.12.19-${patch_series_hash:0:12}"

# 补丁集合采用内容寻址；新集合总从受校验的原始源码构建，避免重叠 hunk
# 让“是否已应用”的反向探测产生歧义。
if [[ ! -d "${overlay}" ]]; then
  overlay_tmp="$(mktemp -d "${repo_root}/.deps/concorde-overlay.XXXXXX")"
  cp -a "${source_root}/." "${overlay_tmp}/"
  while IFS= read -r patch_file; do
    if ! (cd "${overlay_tmp}" &&
          patch --fuzz=0 --forward -p1 < "${patch_file}"); then
      echo "错误：Concorde patch 不能严格应用：${patch_file}" >&2
      exit 5
    fi
  done < <(find "${patch_dir}" -maxdepth 1 -type f -name '*.patch' -print | \
    LC_ALL=C sort)
  mv "${overlay_tmp}" "${overlay}"
fi

build_dir="${repo_root}/build/concorde"
mkdir -p "${build_dir}"
cd "${build_dir}"
CC="${CC:-gcc}" CFLAGS="${CFLAGS:--O2 -std=gnu89}" /bin/sh "${overlay}/configure" \
  --host=i686-pc-linux-gnu \
  --with-qsopt="${qsopt_dir}"
cmake -E env TMPDIR="${repo_root}/.tmp" make -j"$(nproc)"
