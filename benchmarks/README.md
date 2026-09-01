# Benchmarks

首期只记录性能，不设最低加速门槛。pr299 正确性入口为 `tools/run_pr299.sh`；M5 JV 中大型入口为 `tools/run_jv_benchmark.sh INSTANCE 5`。后者使用 `configs/m5_jv_instances.tsv`，检查至少 8 GiB 空间，记录独立 CLI 与进程内稳态两种口径，并逐次执行 CPU proof 重放和边集比较。若已有最优 TSPLIB tour，可通过 `CUDAEE_BENCHMARK_TOUR=FILE` 启用成本与受保护边门禁。CUDA 行同时记录动态 H2D、同步 kernel 和 D2H；报告必须包含 CPU 复核时间，不能只展示 kernel 时间。

缓存后的 k-opt cost 稳态交叉点可用项目内目标测量：

```bash
cmake --preset cuda-release -DCUDAEE_BUILD_BENCHMARKS=ON
cmake --build --preset cuda-release --target cudaee_kopt_cost_benchmark
CUDA_VISIBLE_DEVICES=1 build/cuda-release/cudaee_kopt_cost_benchmark \
  > artifacts/kopt-cost-benchmark.csv
```

该基准包含完整 CPU/CUDA 矩阵逐单元比较，并计入同步调用、task H2D、cost D2H 和驻留 cache 键检查；不把首次分配/上传混入稳态中位数。
