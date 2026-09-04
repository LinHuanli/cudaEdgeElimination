#!/usr/bin/env bash
set -euo pipefail

# 该 bridge 只用于评估 GPU cutting-plane 尚缺失的分离强度，不进入正式产品链路。
# Concorde 源码和目标文件均保持在仓库内的 .deps/build 忽略目录中。
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

source_root="$(awk -F ' = ' '$1 == "SRCROOT" { print $2; exit }' build/concorde/CUT/Makefile)"
if [[ -z "${source_root}" || ! -f "${source_root}/INCLUDE/cut.h" ||
      ! -f build/concorde/CUT/cut.a || ! -f build/concorde/UTIL/util.a ]]; then
  echo "错误：请先按项目 bootstrap 流程构建 Concorde。" >&2
  exit 2
fi
mkdir -p .tmp
gcc -O2 -std=gnu89 -Wall -Wextra \
  -Ibuild/concorde/INCLUDE -I"${source_root}/INCLUDE" \
  tools/concorde_mincut_bridge.c build/concorde/CUT/cut.a build/concorde/UTIL/util.a \
  -liberty -lbfd -lm -o .tmp/concorde_mincut_bridge
echo "完成：${repo_root}/.tmp/concorde_mincut_bridge"
