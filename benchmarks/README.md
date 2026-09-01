# Benchmarks

首期只记录性能，不设最低加速门槛。标准入口为 `tools/run_pr299.sh`；扩展到 pcb3038 前需补齐实例清单和至少 8 GiB 空间门禁。报告必须同时包含 CPU 复核时间，不能只展示 kernel 时间。

缓存后的 k-opt cost 稳态交叉点可用项目内目标测量：

```bash
cmake --preset cuda-release -DCUDAEE_BUILD_BENCHMARKS=ON
cmake --build --preset cuda-release --target cudaee_kopt_cost_benchmark
CUDA_VISIBLE_DEVICES=1 build/cuda-release/cudaee_kopt_cost_benchmark \
  > artifacts/kopt-cost-benchmark.csv
```

该基准包含完整 CPU/CUDA 矩阵逐单元比较，并计入同步调用、task H2D、cost D2H 和驻留 cache 键检查；不把首次分配/上传混入稳态中位数。
