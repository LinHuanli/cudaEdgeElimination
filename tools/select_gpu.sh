#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误：找不到 nvidia-smi。" >&2
  exit 2
fi

# 先按利用率升序，再按空闲显存降序，最后按设备号稳定选择。
nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
  --format=csv,noheader,nounits |
  awk -F',' '{gsub(/ /, "", $0); print $1, $2, $3}' |
  sort -k2,2n -k3,3nr -k1,1n |
  awk 'NR==1 {print $1}'
