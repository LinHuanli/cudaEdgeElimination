# M5 HT CPU 精确成本矩阵行并行

## 1. 研究问题

[leaf setup 快照哈希复用](34_M5_HT_Leaf_Setup_Snapshot_Hash.md)后，pcb3038 8-target CPU leaf 为 2.981 秒，其中 `cost_evaluate_ms` 为 2.345 秒，逐 cell CPU 精确认证占 2.285 秒。成本矩阵的每个 task row 在生成阶段互不依赖，适合做不改变证明顺序的有界并行。

本切片不并行 cursor 消费、witness 选择或 proof 写入；这些阶段仍严格串行，因而首次改善模板和规范 proof 字节不受线程调度影响。

## 2. 实现与安全边界

提交 `34cf918` 增加可选 OpenMP C++ 路径：

- 输出矩阵先按 `[task][template]` 的既定规模分配；
- 只在矩阵不少于 8,192 cells 时并行，否则保持原串行循环；
- 最多使用 8 个线程，并同时受 `omp_get_max_threads()` 和当前可用处理器数约束；
- `schedule(static)` 按 task row 分片，每个线程只写自己的连续输出区间；
- 每个 task 的端口距离 cache 仍是栈上私有 `KOptCostTaskCpuScorer`；
- task 校验、CUDA 全矩阵比较、cursor 消费、完整 witness 重建和 proof verifier 均保持不变。

整数坐标安全门禁和端口范围校验在进入并行区前完成。没有 OpenMP runtime、显式关闭 `CUDAEE_ENABLE_OPENMP`、线程上限为 1 或 batch 小于门槛时，都走同一串行实现。项目内 `cpu-noomp` 构建和 k-opt 单测验证了无 OpenMP 回退。

大型单测用 40 个 5-opt tasks 形成 8,320 cells，逐 row 检查规范布局，并要求 CUDA 输出通过同一并行 CPU 全矩阵认证。并行只改变 cell 的完成时序，不能把 GPU 结果直接变成删除授权。

## 3. 微基准

RTX 4000 Ada 所在节点有 8 个物理 Xeon Gold 5122 cores。相同二进制分别固定 `OMP_NUM_THREADS=1/8`，8 线程使用 `OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=spread`、`OMP_PLACES=cores`。代表性稳态中位数如下：

| k / tasks / cells | 1 thread（µs） | 8 threads（µs） | 加速 |
|---|---:|---:|---:|
| 3 / 4,096 / 16,384 | 1,646.300 | 682.537 | 2.41× |
| 4 / 4,096 / 102,400 | 5,389.374 | 1,322.776 | 4.07× |
| 5 / 4,096 / 851,968 | 36,814.587 | 10,433.435 | 3.53× |

门槛以下的行保持原串行耗时。该结果支持保留 row 并行，但也表明距离计算、批次启动和双路 NUMA/SMT 不会得到理想 8×。

## 4. V9/V11 可观测性

HT scan report 升级为 V9，benchmark summary 升级为 V11，新增：

- `leaf_cpu_parallel_cost_batches`；
- `leaf_cpu_parallel_cost_cells`；
- `peak_leaf_cpu_cost_threads`。

正式脚本默认固定 8 threads，并把线程数、dynamic、bind 和 places 写入 manifest。CPU、全 CUDA和非融合 hybrid 必须报告相同的并行 batches/cells/峰值线程；融合调度可以因 batch 边界不同而拥有不同并行批次数和 cells，但规范总 cells 仍必须相同。

## 5. clean-commit 串行/并行 A/B

两次运行均绑定 clean commit `34cf918`、相同 pcb3038/JV 固定点、相同最高权重 8-target 切片和物理 GPU 1：

- 1 thread：`artifacts/pcb3038-ht-scan-20260901T221443Z-2622871`；
- 8 threads：`artifacts/pcb3038-ht-scan-20260901T221504Z-2623373`。

| CPU 指标（ms） | 1 thread | 8 threads | 加速 |
|---|---:|---:|---:|
| CPU exact certify | 2,340.357 | 700.932 | 3.34× |
| cost evaluate | 2,400.569 | 764.575 | 3.14× |
| leaf | 3,028.903 | 1,441.934 | 2.10× |
| search | 3,444.287 | 1,866.604 | 1.85× |
| 进程 wall | 3,480.162 | 1,904.274 | 1.83× |

8-thread run 有 726 个并行 cost batches，覆盖 48,879,635/51,309,996 cells（95.26%），峰值线程为 8。融合调度把 51,047,847 cells（99.49%）放入 208 个并行 batch，显示跨桶合批在 CPU 并行阶段可能重新具有价值。

8-thread 的 CPU/全 CUDA/hybrid/fused search 分别为 `1,866.604 / 2,401.521 / 2,293.673 / 2,098.027 ms`；同步 CPU 完整认证下，显式 CPU backend 继续最快。

两路均为 2 PROVEN、6 UNRESOLVED、提交相同 2 条边；9,891 cursors、727,635 consumed rows、987 candidates 和 51,309,996 certified cells 完全一致。最终图哈希为 `fe11f98414b04c0e`，边文件 SHA-256 为 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`；成本 137,694 的受保护 tour 保持 0 缺边。

CPU Debug/Release、CUDA Release、GPU 大矩阵差分、无 OpenMP 回退和 compute-sanitizer memcheck 均通过。

## 6. 下一切片

并行后 CPU leaf 中 cost evaluate 仍占 53.02%，但 setup、cursor prepare、consume 和矩阵 scatter 合计也已不可忽略。下一步先增加“CPU backend + leaf bucket fusion”对照：它可以把 frontier batches 从 500 降到 86、并行 cost batches 从 726 降到约 208，并复用更少的 batch 级快照哈希。只有同一 clean commit 上证明端到端稳定获益，才考虑把融合用于 CPU 默认路径；否则保持现有默认调度并转向 scorer 数据布局。
