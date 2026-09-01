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

脚本固定记录搜索深度、state/reply/deletion-set 预算；资源耗尽是 `UNRESOLVED`，不是失败或删除授权。五路对照分别是纯 CPU、启用 leaf 复杂度桶融合的 CPU、所有候选器均显式 CUDA、CPU c,d/reply/path/propagation + CUDA leaf cost 的混合路径，以及在混合路径上启用复杂度桶融合；五者必须拥有相同工作签名、最终边集和可独立重放的 proof。

V9 报告将工作图总时间进一步拆成 leaf、path-append、Hamilton reply 与 end reply，并单列传播、proof 抽取、三层 CPU 重放和最终 commit；它还记录 leaf frontier/bucket/cost batch 数、融合开关、leaf 内部 setup/cursor/cost/consume/apply/verifier、CPU 精确 cost-matrix 认证，以及 Hamilton reply 的 validation/CPU enumerate/CUDA evaluate/compare。setup 又细分为 proof 初始化、coverage 扫描和 cursor 构造；这些是包含于 setup 的子项，不能重复相加。报告同时记录启用 CPU row 并行的 batches/cells 和实际峰值线程数。`work_graph_ms`、`leaf_ms`、`leaf_setup_ms`、`leaf_cost_evaluate_ms`、`leaf_cursor_consume_ms` 和 `hamilton_reply_ms` 都是包含式总量。V12 summary 要求五路 leaf cursor、cost/consume 与 Hamilton reply 的规范工作计数完全一致，并单列 CPU 融合路径的完整 leaf 子阶段和相对非融合 CPU 的加速比；residual 已按对应包含关系扣除。正式脚本默认固定 8 个 CPU cost threads、关闭动态线程并按物理 core spread；可用 `CUDAEE_CPU_COST_THREADS=1..8` 做串行或缩放实验。

`CUDAEE_HT_TARGET_OFFSET` 只选择当前不可变输入上的目标切片；若使用一个已提交的新图开始下一 epoch，必须重新从 offset 0 排序。
