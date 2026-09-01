#!/usr/bin/env bash
# 本文件必须由当前 shell source，所有路径均限定在仓库内。
if [[ -n "${ZSH_VERSION:-}" ]]; then
  script_source="${(%):-%N}"
else
  script_source="${BASH_SOURCE[0]}"
fi
repo_root="$(cd "$(dirname "${script_source}")/.." && pwd -P)"

cuopt_library="$(find "${repo_root}/.venv" -type f -name 'libcuopt.so*' -print -quit 2>/dev/null || true)"
if [[ -n "${cuopt_library}" ]]; then
  # NVIDIA wheels 把依赖共享库分散在多个包目录；统一加入运行时搜索路径。
  dependency_directories="$(find "${repo_root}/.venv" -type f -name '*.so*' -printf '%h\n' |
    sort -u | paste -sd ':' -)"
  export LD_LIBRARY_PATH="${dependency_directories}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export CUDAEE_CUOPT_LIBRARY="${cuopt_library}"
fi
export CUDAEE_REPOSITORY_ROOT="${repo_root}"
export TMPDIR="${repo_root}/.tmp"
export PIP_CACHE_DIR="${repo_root}/.deps/pip-cache"
