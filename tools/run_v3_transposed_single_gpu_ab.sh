#!/usr/bin/env bash
set -euo pipefail

# V3 C1 正式 A/B：比较各自最佳的 CPU 规范短路与单 GPU 跨目标 leaf broker。
# 两路 speculation 可不同，但下层脚本会强制边集、规范 proof 与已消费工作签名相同。
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

export CUDAEE_HT_SCHEDULER=transposed
export CUDAEE_CPU_TARGET_WORKERS="${CUDAEE_CPU_TARGET_WORKERS:-32}"
export CUDAEE_HYBRID_TARGET_WORKERS="${CUDAEE_HYBRID_TARGET_WORKERS:-32}"
export CUDAEE_CPU_SPECULATION="${CUDAEE_CPU_SPECULATION:-1}"
export CUDAEE_HYBRID_SPECULATION="${CUDAEE_HYBRID_SPECULATION:-4}"
export CUDAEE_CPU_COST_THREADS="${CUDAEE_CPU_COST_THREADS:-2}"
export CUDAEE_HYBRID_CPU_COST_THREADS="${CUDAEE_HYBRID_CPU_COST_THREADS:-2}"

exec "${script_dir}/run_v3_single_gpu_ab.sh" "$@"
