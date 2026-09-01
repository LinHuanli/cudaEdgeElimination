#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"
if [[ ! -f build/cuda-debug/CMakeCache.txt ]]; then
  cmake --preset cuda-debug
fi
cmake --build --preset cuda-debug --target cudaee --parallel
physical_gpu="$(tools/select_gpu.sh)"
CUDA_VISIBLE_DEVICES="${physical_gpu}" compute-sanitizer --tool memcheck --error-exitcode=99 \
  build/cuda-debug/cudaee gpu-eliminate \
  --tsp third_party/ElimTSP/data/pr299.tsp \
  --edges third_party/ElimTSP/data/pr299.edg \
  --output artifacts/pr299.sanitized.edg \
  --proof artifacts/pr299.sanitized.proof \
  --backend cuda --max-rounds 100
