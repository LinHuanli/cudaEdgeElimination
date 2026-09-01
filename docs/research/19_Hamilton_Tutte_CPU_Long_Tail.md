# Hamilton–Tutte leaf GPU/CPU long-tail

## 范围

M4.3b3b2b2b2b2b1 为 `auto` k-opt cost backend 加入显式、可测试的矩阵规模阈值。它解决驻留缓存之后仍然存在的小矩阵 kernel launch/synchronize 开销，不裁剪 deletion set、template、outside matching 或 leaf state。

策略按跨 cursor 融合后的完整 cost matrix 决策：

- `cells < cuda_min_cost_cells`：用 CPU 精确 cost matrix；
- `cells >= cuda_min_cost_cells`：`auto` 尝试 CUDA，不可用或运行失败时按既有规则回退 CPU；
- 显式 `cpu`/`cuda` 后端忽略该阈值；
- `cuda_min_cost_cells=0` 表示 `auto` 对所有非空矩阵尝试 CUDA。

默认阈值为 128 cells，CLI 参数是 `--cuda-min-cost-cells N`。调度发生在 rows 跨 leaf 融合之后，因此许多单独很小的 cursor block 仍可一起越过阈值，而不是被永久拆成 CPU 小任务。

## 基准依据

项目内 `cudaee_kopt_cost_benchmark` 使用 4096 点整数 `EUC_2D` 图，在 RTX 4000 Ada 上测量驻留 cache 预热后的同步调用中位数。时间包含 cache 键检查、task H2D、kernel、同步和 cost D2H，并在每一行逐单元比较 CPU/CUDA 矩阵。

交叉点附近结果如下：

| k | tasks | cells | CPU μs | CUDA μs | CPU/CUDA |
|---:|---:|---:|---:|---:|---:|
| 3 | 16 | 64 | 19.626 | 26.154 | 0.750 |
| 3 | 64 | 256 | 77.250 | 32.406 | 2.384 |
| 4 | 1 | 25 | 14.285 | 31.727 | 0.450 |
| 4 | 4 | 100 | 42.690 | 32.532 | 1.312 |
| 5 | 1 | 208 | 153.919 | 85.865 | 1.793 |

128 是单一 workload 单位下的保守边界：它保留 3-opt 64-cell 与 4-opt 25-cell 小矩阵在 CPU，同时让所有 5-opt 矩阵和更大的融合任务进入 CUDA。该数值是当前硬件基线，不是跨架构性能承诺；完整复现实验入口见 `benchmarks/README.md`。

## 规范计数

原候选路径把完整 cost cells 和随后 CPU `TryReconnect` 的尝试都累加到 `reconnect_matchings_tested`，因此相同 witness 可能因 CPU/GPU 路由不同而产生不同 V1 文本。long-tail 会在 batch 边界附近主动改变路由，所以该指标现规范化为“CPU 模板顺序中检查到的位置”：

- 找到第 `i` 个模板的 witness 时计 `i+1`；
- 完整无改善时计全部 proper templates；
- CUDA 没有 CPU 接受的候选时仍调用全模板 `TryReconnect`，直接采用其计数和结果。

cost matrix 的实际工作量继续由独立 `cost_cells` 记录。这样 CPU 精确筛选、CUDA 候选筛选和阈值两侧都生成相同 witness、逻辑计数及 V1 proof；GPU 仍不能用“无命中”授权结论。

## 指标与回归

`PathSystemKOptBatchResult` 新增 `cpu_long_tail_batches/tasks/cells`，wavefront/CLI 对应公开 `leaf_cpu_long_tail_*`。这些指标只统计因为 `auto` 阈值而主动选择 CPU 的矩阵；CUDA 不可用后的普通 auto fallback 不混入 long-tail 数。

固定 5 点无改善 leaf 回归使用 3-opt 单模板块：

- 31 个 leaf 融合为 124 cells，记录 1 个 CPU long-tail batch；
- 32 个 leaf 融合为 128 cells，在 CUDA 可用时记录 1 个 CUDA batch；
- 两侧每份 proof 都与显式 CPU scalar proof 逐字节一致。

固定 recursive-point 的 6 个融合 3-opt batches 均小于 128 cells，`auto` 因而全部进入 CPU long-tail；其 34-state 工作图与显式全 CUDA leaf 运行生成完全相同的 4-node V1 proof。显式 CUDA 仍用于 cache、差分与 memcheck 门禁，不受默认阈值影响。

## 后续

当前 policy 只分流 leaf cost matrix；CPU witness 构造和 exact Held–Karp 本来就在主机。下一阶段是 continuation 的多 block GPU 调度与深层/超大状态资源策略，之后才设计验证成功候选到不可变 epoch 的确定性 commit。
