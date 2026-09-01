#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"
source tools/env.sh

if [[ -z "${CUDAEE_CUOPT_LIBRARY:-}" ]]; then
  echo "错误：项目内尚未安装 cuOpt，请先运行 tools/bootstrap.sh。" >&2
  exit 2
fi
if [[ ! -x build/cuda-release/cudaee ]]; then
  cmake --preset cuda-release
  cmake --build --preset cuda-release
fi

mkdir -p artifacts/cuopt-smoke
physical_gpu="$(tools/select_gpu.sh)"
build/cuda-release/cudaee lp-example \
  --output artifacts/cuopt-smoke/tiny.lp-epoch
CUDA_VISIBLE_DEVICES="${physical_gpu}" build/cuda-release/cudaee lp-solve \
  --input artifacts/cuopt-smoke/tiny.lp-epoch \
  --output artifacts/cuopt-smoke/tiny.lp-solution \
  --cuopt-library "${CUDAEE_CUOPT_LIBRARY}"

rg -q '^status OPTIMAL$' artifacts/cuopt-smoke/tiny.lp-solution
rg -q '^objective 1$' artifacts/cuopt-smoke/tiny.lp-solution
rg -q '^numerically_accepted 1$' artifacts/cuopt-smoke/tiny.lp-solution
rg -q '^exact_model_bound_numerator 16777216$' artifacts/cuopt-smoke/tiny.lp-solution
rg -q '^exact_model_bound_denominator 16777216$' artifacts/cuopt-smoke/tiny.lp-solution
echo "cuOpt C API smoke test passed on physical GPU ${physical_gpu}"
