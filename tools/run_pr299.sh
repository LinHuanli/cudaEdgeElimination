#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"
mkdir -p artifacts

binary="${repo_root}/build/cuda-release/cudaee"
if [[ ! -x "${binary}" ]]; then
  cmake --preset cuda-release
  cmake --build --preset cuda-release
fi

physical_gpu="$(tools/select_gpu.sh)"
echo "选择物理 GPU ${physical_gpu}"
CUDA_VISIBLE_DEVICES="${physical_gpu}" "${binary}" gpu-eliminate \
  --tsp "${repo_root}/third_party/ElimTSP/data/pr299.tsp" \
  --edges "${repo_root}/third_party/ElimTSP/data/pr299.edg" \
  --output "${repo_root}/artifacts/pr299.filtered.edg" \
  --proof "${repo_root}/artifacts/pr299.proof" \
  --manifest "${repo_root}/artifacts/pr299.manifest" \
  --backend cuda --max-rounds 100

"${binary}" verify \
  --tsp "${repo_root}/third_party/ElimTSP/data/pr299.tsp" \
  --edges "${repo_root}/third_party/ElimTSP/data/pr299.edg" \
  --proof "${repo_root}/artifacts/pr299.proof"
