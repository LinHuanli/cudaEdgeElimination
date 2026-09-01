#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

mkdir -p .deps/pip-cache .tmp
free_kib="$(df -Pk "${repo_root}" | awk 'NR==2 {print $4}')"
required_kib=$((8 * 1024 * 1024))
if (( free_kib < required_kib )); then
  echo "错误：仓库所在文件系统可用空间不足 8 GiB，拒绝安装依赖。" >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  TMPDIR="${repo_root}/.tmp" python3 -m venv .venv
fi

export TMPDIR="${repo_root}/.tmp"
export PIP_CACHE_DIR="${repo_root}/.deps/pip-cache"
.venv/bin/python -m pip install "pip==26.2.1"
.venv/bin/python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  --requirement configs/requirements-cuopt.lock

library="$(find .venv -type f -name 'libcuopt.so*' -print -quit)"
if [[ -z "${library}" ]]; then
  echo "错误：安装完成但没有找到 libcuopt.so。" >&2
  exit 3
fi

echo "cuOpt 已安装：${library}"
echo "运行 LP 前请执行：source tools/env.sh"
