#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${repo_root}"

instance="${1:-}"
runs="${2:-5}"
if [[ -z "${instance}" || ! "${instance}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "用法：$0 INSTANCE [RUNS]" >&2
  exit 2
fi
if [[ ! "${runs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误：RUNS 必须是正整数。" >&2
  exit 2
fi

config="${repo_root}/configs/m5_jv_instances.tsv"
row="$(awk -F '\t' -v name="${instance}" '$1 == name { print; found = 1 } END { if (!found) exit 1 }' "${config}")" || {
  echo "错误：${config} 中没有实例 ${instance}。" >&2
  exit 2
}
IFS=$'\t' read -r _ tsp_relative edges_relative certified_optimum <<<"${row}"
tsp="$(realpath "${repo_root}/${tsp_relative}")"
edges="$(realpath "${repo_root}/${edges_relative}")"
if [[ ! -f "${tsp}" || ! -f "${edges}" ]]; then
  echo "错误：实例输入不存在。" >&2
  exit 2
fi
if [[ ! "${certified_optimum}" =~ ^[0-9]+$ ]]; then
  echo "错误：实例 certified optimum 不是非负整数。" >&2
  exit 2
fi

# 大实例开始前保留至少 8 GiB；失败运行仍保留已产生的诊断文件。
available_kib="$(df -Pk "${repo_root}" | awk 'NR == 2 { print $4 }')"
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "错误：仓库文件系统可用空间不足 8 GiB。" >&2
  exit 2
fi

binary="${repo_root}/build/cuda-release/cudaee"
cpu_verifier="${repo_root}/build/cpu-release/cudaee"
benchmark_binary="${repo_root}/build/cuda-release/cudaee_jv_benchmark"
if [[ ! -f "${repo_root}/build/cuda-release/CMakeCache.txt" ]]; then
  cmake --preset cuda-release
fi
if [[ ! -f "${repo_root}/build/cpu-release/CMakeCache.txt" ]]; then
  cmake --preset cpu-release
fi
if [[ ! -f "${repo_root}/build/cuda-release/CMakeCache.txt" ]] ||
   ! grep -q '^CUDAEE_BUILD_BENCHMARKS:BOOL=ON$' "${repo_root}/build/cuda-release/CMakeCache.txt"; then
  cmake --preset cuda-release -DCUDAEE_BUILD_BENCHMARKS=ON
fi
cmake --build --preset cuda-release --target cudaee cudaee_jv_benchmark --parallel
cmake --build --preset cpu-release --target cudaee --parallel

physical_gpu="${CUDAEE_BENCHMARK_GPU:-$(tools/select_gpu.sh)}"
if [[ ! "${physical_gpu}" =~ ^[0-9]+$ ]]; then
  echo "错误：CUDAEE_BENCHMARK_GPU 必须是物理 GPU 编号。" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${instance}-jv-${timestamp}-$$"
run_dir="${repo_root}/artifacts/${run_id}"
mkdir -p "${run_dir}"
metrics="${run_dir}/metrics.csv"
summary="${run_dir}/summary.txt"
manifest="${run_dir}/run-manifest-v1"
protected_tour_source="${CUDAEE_BENCHMARK_TOUR:-}"
protected_tour=""
protected_tour_sha256="none"
if [[ -n "${protected_tour_source}" ]]; then
  protected_tour_source="$(realpath "${protected_tour_source}")"
  if [[ ! -f "${protected_tour_source}" ]]; then
    echo "错误：CUDAEE_BENCHMARK_TOUR 不存在。" >&2
    exit 2
  fi
  protected_tour="${run_dir}/protected.opt.tour"
  cp -- "${protected_tour_source}" "${protected_tour}"
  protected_tour_sha256="$(sha256sum "${protected_tour}" | awk '{ print $1 }')"
fi

read_edge_header() {
  if [[ "$1" == *.gz ]]; then
    gzip -cd -- "$1" | awk 'NR == 1 { print $1, $2; exit }'
  else
    awk 'NR == 1 { print $1, $2; exit }' "$1"
  fi
}

read -r dimension initial_edges <<<"$(read_edge_header "${edges}")"
if [[ ! "${dimension}" =~ ^[0-9]+$ || ! "${initial_edges}" =~ ^[0-9]+$ ]]; then
  echo "错误：无法解析边文件头。" >&2
  exit 2
fi

{
  echo "CUDAEE_BENCHMARK_MANIFEST_V1"
  echo "run_id ${run_id}"
  echo "instance ${instance}"
  echo "utc_timestamp ${timestamp}"
  echo "git_commit $(git rev-parse HEAD)"
  echo "git_dirty $(git status --porcelain | awk 'END { print NR == 0 ? 0 : 1 }')"
  echo "tsp ${tsp}"
  echo "tsp_sha256 $(sha256sum "${tsp}" | awk '{ print $1 }')"
  echo "edges ${edges}"
  echo "edges_sha256 $(sha256sum "${edges}" | awk '{ print $1 }')"
  echo "dimension ${dimension}"
  echo "initial_edges ${initial_edges}"
  echo "certified_optimum ${certified_optimum}"
  echo "optimum_source https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html"
  echo "protected_tour ${protected_tour:-none}"
  echo "protected_tour_sha256 ${protected_tour_sha256}"
  echo "timed_runs ${runs}"
  echo "measurement_modes cli_process,in_process"
  echo "jv_cuda_cache exact_static_key,dynamic_edge_ids,growth_workspace"
  echo "physical_gpu ${physical_gpu}"
  echo "cuda_visible_devices ${physical_gpu}"
  echo "cmake_preset cuda-release"
  echo "compiler $(c++ --version | awk 'NR == 1')"
  echo "nvcc $(nvcc --version | awk '/release/ { print; exit }')"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | sed 's/^/gpu /'
  echo "END"
} >"${manifest}"

echo "backend,run,wall_ms,replay_ms,edges_scanned,propose_ms,verify_ms,committed,active_edges,edges_per_second,final_hash,snapshot_ms,commit_ms,static_cache_hits,workspace_cache_hits,peak_resident_bytes,h2d_ms,kernel_ms,d2h_ms" >"${metrics}"

run_elimination() {
  local backend="$1"
  local label="$2"
  local output="${run_dir}/${backend}.${label}.edg"
  local proof="${run_dir}/${backend}.${label}.proof"
  local stdout_file="${run_dir}/${backend}.${label}.stdout"
  local stderr_file="${run_dir}/${backend}.${label}.stderr"
  local verify_stdout="${run_dir}/${backend}.${label}.verify.stdout"
  local verify_stderr="${run_dir}/${backend}.${label}.verify.stderr"
  local start_ns end_ns replay_start_ns replay_end_ns

  start_ns="$(date +%s%N)"
  if [[ "${backend}" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="${physical_gpu}" "${binary}" gpu-eliminate \
      --tsp "${tsp}" --edges "${edges}" --output "${output}" --proof "${proof}" \
      --backend cuda --max-rounds 100 >"${stdout_file}" 2>"${stderr_file}"
  else
    "${binary}" gpu-eliminate \
      --tsp "${tsp}" --edges "${edges}" --output "${output}" --proof "${proof}" \
      --backend cpu --max-rounds 100 >"${stdout_file}" 2>"${stderr_file}"
  fi
  end_ns="$(date +%s%N)"

  replay_start_ns="$(date +%s%N)"
  "${cpu_verifier}" verify --tsp "${tsp}" --edges "${edges}" --proof "${proof}" \
    >"${verify_stdout}" 2>"${verify_stderr}"
  replay_end_ns="$(date +%s%N)"

  if [[ "${label}" != "warmup" ]]; then
    local wall_ms replay_ms parsed
    wall_ms="$(awk -v first="${start_ns}" -v last="${end_ns}" 'BEGIN { printf "%.3f", (last-first)/1000000 }')"
    replay_ms="$(awk -v first="${replay_start_ns}" -v last="${replay_end_ns}" 'BEGIN { printf "%.3f", (last-first)/1000000 }')"
    parsed="$(awk '
      /^epoch=/ {
        for (i = 1; i <= NF; ++i) {
          split($i, pair, "=")
          if (pair[1] == "edges_before") scanned += pair[2]
          if (pair[1] == "snapshot_ms") snapshot += pair[2]
          if (pair[1] == "propose_ms") propose += pair[2]
          if (pair[1] == "verify_ms") verify += pair[2]
          if (pair[1] == "commit_ms") commit += pair[2]
          if (pair[1] == "jv_static_cache_hit") static_hits += pair[2]
          if (pair[1] == "jv_workspace_cache_hit") workspace_hits += pair[2]
          if (pair[1] == "jv_resident_bytes" && pair[2] > resident) resident = pair[2]
          if (pair[1] == "jv_h2d_ms") h2d += pair[2]
          if (pair[1] == "jv_kernel_ms") kernel += pair[2]
          if (pair[1] == "jv_d2h_ms") d2h += pair[2]
        }
      }
      /^status=OK/ {
        for (i = 1; i <= NF; ++i) {
          split($i, pair, "=")
          if (pair[1] == "committed") committed = pair[2]
          if (pair[1] == "active_edges") active = pair[2]
          if (pair[1] == "final_hash") hash = pair[2]
        }
      }
      END { printf "%d,%.6f,%.6f,%d,%d,%s,%.6f,%.6f,%d,%d,%d,%.6f,%.6f,%.6f", scanned, propose, verify, committed, active, hash, snapshot, commit, static_hits, workspace_hits, resident, h2d, kernel, d2h }
    ' "${stdout_file}")"
    local scanned propose verify committed active hash snapshot commit static_hits workspace_hits resident h2d kernel d2h throughput
    IFS=',' read -r scanned propose verify committed active hash snapshot commit static_hits workspace_hits resident h2d kernel d2h <<<"${parsed}"
    if [[ -z "${hash}" || ! "${scanned}" =~ ^[0-9]+$ || ! "${committed}" =~ ^[0-9]+$ ||
          ! "${active}" =~ ^[0-9]+$ ]]; then
      echo "错误：${backend}.${label} 输出缺少规范指标。" >&2
      exit 2
    fi
    throughput="$(awk -v count="${scanned}" -v elapsed="${wall_ms}" 'BEGIN { printf "%.3f", 1000*count/elapsed }')"
    echo "${backend},${label},${wall_ms},${replay_ms},${scanned},${propose},${verify},${committed},${active},${throughput},${hash},${snapshot},${commit},${static_hits},${workspace_hits},${resident},${h2d},${kernel},${d2h}" >>"${metrics}"
  fi
}

echo "预热 ${instance}：CPU 与物理 GPU ${physical_gpu}"
run_elimination cpu warmup
run_elimination cuda warmup
cmp "${run_dir}/cpu.warmup.edg" "${run_dir}/cuda.warmup.edg"

protected_tour_checked=0
protected_tour_hash="none"
if [[ -n "${protected_tour}" ]]; then
  "${cpu_verifier}" tour-check --tsp "${tsp}" --edges "${edges}" \
    --tour "${protected_tour}" --expected-cost "${certified_optimum}" \
    >"${run_dir}/protected.initial.stdout" 2>"${run_dir}/protected.initial.stderr"
  "${cpu_verifier}" tour-check --tsp "${tsp}" --edges "${run_dir}/cpu.warmup.edg" \
    --tour "${protected_tour}" --expected-cost "${certified_optimum}" \
    >"${run_dir}/protected.final.stdout" 2>"${run_dir}/protected.final.stderr"
  protected_tour_hash="$(awk '{ for (i = 1; i <= NF; ++i) { split($i, pair, "="); if (pair[1] == "tour_hash") print pair[2] } }' "${run_dir}/protected.final.stdout")"
  if [[ -z "${protected_tour_hash}" ]]; then
    echo "错误：受保护 tour 门禁没有输出哈希。" >&2
    exit 2
  fi
  protected_tour_checked=1
fi

for ((run = 1; run <= runs; ++run)); do
  echo "计时 ${instance} ${run}/${runs}"
  run_elimination cpu "${run}"
  run_elimination cuda "${run}"
  cmp "${run_dir}/cpu.${run}.edg" "${run_dir}/cuda.${run}.edg"
  cmp "${run_dir}/cpu.warmup.edg" "${run_dir}/cpu.${run}.edg"
done

echo "进程内预热基准 ${instance}"
CUDA_VISIBLE_DEVICES="${physical_gpu}" "${benchmark_binary}" "${tsp}" "${edges}" "${runs}" \
  >"${run_dir}/inprocess.csv" 2>"${run_dir}/inprocess.stderr"

mapfile -t cpu_walls < <(awk -F',' '$1 == "cpu" { print $3 }' "${metrics}" | sort -n)
mapfile -t cuda_walls < <(awk -F',' '$1 == "cuda" { print $3 }' "${metrics}" | sort -n)
mapfile -t cpu_proposes < <(awk -F',' '$1 == "cpu" { print $6 }' "${metrics}" | sort -n)
mapfile -t cuda_proposes < <(awk -F',' '$1 == "cuda" { print $6 }' "${metrics}" | sort -n)
mapfile -t inprocess_cpu < <(awk -F',' '$1 == "cpu" { print $3 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda < <(awk -F',' '$1 == "cuda" { print $3 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cpu_propose < <(awk -F',' '$1 == "cpu" { print $4 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_propose < <(awk -F',' '$1 == "cuda" { print $4 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cpu_replay < <(awk -F',' '$1 == "cpu" { print $6 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_replay < <(awk -F',' '$1 == "cuda" { print $6 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cpu_snapshot < <(awk -F',' '$1 == "cpu" { print $11 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_snapshot < <(awk -F',' '$1 == "cuda" { print $11 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cpu_commit < <(awk -F',' '$1 == "cpu" { print $12 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_commit < <(awk -F',' '$1 == "cuda" { print $12 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_static_hits < <(awk -F',' '$1 == "cuda" { print $13 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_workspace_hits < <(awk -F',' '$1 == "cuda" { print $14 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_resident < <(awk -F',' '$1 == "cuda" { print $15 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_h2d < <(awk -F',' '$1 == "cuda" { print $16 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_kernel < <(awk -F',' '$1 == "cuda" { print $17 }' "${run_dir}/inprocess.csv" | sort -n)
mapfile -t inprocess_cuda_d2h < <(awk -F',' '$1 == "cuda" { print $18 }' "${run_dir}/inprocess.csv" | sort -n)

# 所有数据列必须完整，避免部分失败的结果被误汇总为成功。
for count in "${#cpu_walls[@]}" "${#cuda_walls[@]}" "${#cpu_proposes[@]}" \
  "${#cuda_proposes[@]}" "${#inprocess_cpu[@]}" "${#inprocess_cuda[@]}" \
  "${#inprocess_cpu_propose[@]}" "${#inprocess_cuda_propose[@]}" \
  "${#inprocess_cpu_replay[@]}" "${#inprocess_cuda_replay[@]}" \
  "${#inprocess_cpu_snapshot[@]}" "${#inprocess_cuda_snapshot[@]}" \
  "${#inprocess_cpu_commit[@]}" "${#inprocess_cuda_commit[@]}" \
  "${#inprocess_cuda_static_hits[@]}" "${#inprocess_cuda_workspace_hits[@]}" \
  "${#inprocess_cuda_resident[@]}" "${#inprocess_cuda_h2d[@]}" \
  "${#inprocess_cuda_kernel[@]}" "${#inprocess_cuda_d2h[@]}"; do
  if (( count != runs )); then
    echo "错误：基准数据列不完整（期望 ${runs} 行，实际 ${count} 行）。" >&2
    exit 2
  fi
done

median_index="$(( (runs - 1) / 2 ))"
p95_index="$(( (95 * runs + 99) / 100 - 1 ))"
wall_speedup="$(awk -v cpu="${cpu_walls[median_index]}" \
  -v gpu="${cuda_walls[median_index]}" 'BEGIN { printf "%.3f", cpu/gpu }')"
propose_speedup="$(awk -v cpu="${cpu_proposes[median_index]}" \
  -v gpu="${cuda_proposes[median_index]}" 'BEGIN { printf "%.3f", cpu/gpu }')"
inprocess_speedup="$(awk -v cpu="${inprocess_cpu[median_index]}" \
  -v gpu="${inprocess_cuda[median_index]}" 'BEGIN { printf "%.3f", cpu/gpu }')"

{
  echo "CUDAEE_BENCHMARK_SUMMARY_V1"
  echo "instance ${instance}"
  echo "cpu_wall_median_ms ${cpu_walls[median_index]}"
  echo "cpu_wall_p95_ms ${cpu_walls[p95_index]}"
  echo "cuda_wall_median_ms ${cuda_walls[median_index]}"
  echo "cuda_wall_p95_ms ${cuda_walls[p95_index]}"
  echo "wall_speedup ${wall_speedup}"
  echo "cpu_propose_median_ms ${cpu_proposes[median_index]}"
  echo "cpu_propose_p95_ms ${cpu_proposes[p95_index]}"
  echo "cuda_propose_median_ms ${cuda_proposes[median_index]}"
  echo "cuda_propose_p95_ms ${cuda_proposes[p95_index]}"
  echo "propose_speedup ${propose_speedup}"
  echo "inprocess_cpu_algorithm_median_ms ${inprocess_cpu[median_index]}"
  echo "inprocess_cpu_algorithm_p95_ms ${inprocess_cpu[p95_index]}"
  echo "inprocess_cuda_algorithm_median_ms ${inprocess_cuda[median_index]}"
  echo "inprocess_cuda_algorithm_p95_ms ${inprocess_cuda[p95_index]}"
  echo "inprocess_cpu_propose_median_ms ${inprocess_cpu_propose[median_index]}"
  echo "inprocess_cuda_propose_median_ms ${inprocess_cuda_propose[median_index]}"
  echo "inprocess_cpu_replay_median_ms ${inprocess_cpu_replay[median_index]}"
  echo "inprocess_cuda_replay_median_ms ${inprocess_cuda_replay[median_index]}"
  echo "inprocess_cpu_snapshot_median_ms ${inprocess_cpu_snapshot[median_index]}"
  echo "inprocess_cuda_snapshot_median_ms ${inprocess_cuda_snapshot[median_index]}"
  echo "inprocess_cpu_commit_median_ms ${inprocess_cpu_commit[median_index]}"
  echo "inprocess_cuda_commit_median_ms ${inprocess_cuda_commit[median_index]}"
  echo "inprocess_cuda_static_cache_hits_median ${inprocess_cuda_static_hits[median_index]}"
  echo "inprocess_cuda_workspace_cache_hits_median ${inprocess_cuda_workspace_hits[median_index]}"
  echo "inprocess_cuda_peak_resident_bytes ${inprocess_cuda_resident[p95_index]}"
  echo "inprocess_cuda_h2d_median_ms ${inprocess_cuda_h2d[median_index]}"
  echo "inprocess_cuda_kernel_median_ms ${inprocess_cuda_kernel[median_index]}"
  echo "inprocess_cuda_d2h_median_ms ${inprocess_cuda_d2h[median_index]}"
  echo "inprocess_algorithm_speedup ${inprocess_speedup}"
  echo "protected_tour_checked ${protected_tour_checked}"
  echo "protected_tour_hash ${protected_tour_hash}"
  echo "verified_edge_sha256 $(sha256sum "${run_dir}/cpu.warmup.edg" | awk '{ print $1 }')"
  echo "END"
} >"${summary}"

echo "完成：${run_dir}"
sed -n '1,240p' "${summary}"
