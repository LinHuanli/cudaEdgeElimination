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

M5 的有界全图 HT pilot 使用 JV 固定点作为不可变输入，对同一确定性目标切片分别执行 CPU/CUDA wavefront，比较工作签名和最终边文件，并用独立 CPU 进程重放两份 V2 proof：

```bash
CUDAEE_BENCHMARK_GPU=1 \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/pcb3038.tour \
tools/run_ht_scan_benchmark.sh pcb3038 8
```

脚本固定记录搜索深度、state/reply/deletion-set 预算；资源耗尽是 `UNRESOLVED`，不是失败或删除授权。四路对照分别是纯 CPU、所有候选器均显式 CUDA、CPU c,d/reply/path/propagation + CUDA leaf cost 的混合路径，以及在混合路径上启用复杂度桶融合；四者必须拥有相同工作签名、最终边集和可独立重放的 proof。

V6 报告将工作图总时间进一步拆成 leaf、path-append、Hamilton reply 与 end reply，并单列传播、proof 抽取、三层 CPU 重放和最终 commit；它还记录 leaf frontier/bucket/cost batch 数、融合开关、leaf 内部 setup/cursor/cost/consume/apply/verifier、CPU 精确 cost-matrix 认证，以及 consume 内的候选复核和旧 CPU completeness fallback。`work_graph_ms`、`leaf_ms`、`leaf_cost_evaluate_ms` 和 `leaf_cursor_consume_ms` 都是包含式总量，不能再与各自子阶段相加。V8 summary 对纯 CPU matrix 路径也输出完整 leaf 子阶段，并要求四路认证 cells 与规范工作计数完全一致；residual 已按对应包含关系扣除。

`CUDAEE_HT_TARGET_OFFSET` 只选择当前不可变输入上的目标切片；若使用一个已提交的新图开始下一 epoch，必须重新从 offset 0 排序。
