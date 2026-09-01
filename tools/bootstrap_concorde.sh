#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_root="${repo_root}/../references/concorde_code"
overlay="${repo_root}/.deps/concorde-03.12.19"
qsopt_dir="${repo_root}/.deps/qsopt"

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

if [[ ! -d "${overlay}" ]]; then
  cp -a "${source_root}" "${overlay}"
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

patch_file="${repo_root}/integrations/concorde/patches/0001-add-lp-epoch-export-hook.patch"
if [[ -s "${patch_file}" ]]; then
  if (cd "${overlay}" && patch --fuzz=0 --dry-run --forward -p1 < "${patch_file}" >/dev/null); then
    (cd "${overlay}" && patch --fuzz=0 --forward -p1 < "${patch_file}")
  elif ! (cd "${overlay}" && patch --fuzz=0 --dry-run --reverse -p1 < "${patch_file}" >/dev/null); then
    echo "错误：Concorde patch 既不能应用，也不是已应用状态。" >&2
    exit 5
  fi
fi

build_dir="${repo_root}/build/concorde"
mkdir -p "${build_dir}"
cd "${build_dir}"
CC="${CC:-gcc}" CFLAGS="${CFLAGS:--O2 -std=gnu89}" /bin/sh "${overlay}/configure" \
  --host=i686-pc-linux-gnu \
  --with-qsopt="${qsopt_dir}"
cmake -E env TMPDIR="${repo_root}/.tmp" make -j"$(nproc)"
