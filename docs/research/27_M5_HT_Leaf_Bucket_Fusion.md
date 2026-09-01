# M5 HT leaf 复杂度桶融合

## 1. 研究问题

[阶段画像](26_M5_HT_Phase_Profiling_and_Hybrid.md)显示 pcb3038 的 8-target CPU search 约 80% 位于 leaf，而显式 CUDA leaf 把 51,309,996 个 cost cells 拆成 1,835 个同步 batches。本切片先验证这些 batches 能否仅靠现有参数消除，再实现同一 target、同一 reply chunk 内的跨复杂度桶合批。

它不是跨目标融合：每个 target 的工作图、required edge、预算和提交顺序仍完全独立。公开开关为：

```bash
--fuse-leaf-buckets 0|1
```

默认保持 `0`。当前数据证明它显著减少调用次数，但没有证明稳定 wall-time 收益，因此不得静默改变默认基线。

## 2. 参数假设被排除

在 commit `7ee529f` 的相同 JV 固定点和混合后端上进行了两个单变量诊断：

- `leaf_frontier_batch_states: 256 -> 0`：cost cells 仍为 51,309,996，CUDA cost batches 仍为 1,835；
- `cost_batch_size: 4096 -> 16384`：cost cells 和 CUDA cost batches 同样完全不变。

两份输出均与基线边文件逐字节一致，外层 proof 均由 CPU 独立重放到 `fe11f98414b04c0e`。这说明 1,835 个调用不是由单桶 256-state 上限或 cursor 4,096-task 上限造成，而是不同复杂度桶/调用各自启动增量 cursor 调度。

## 3. 实现语义

提交 `b5fde24` 把原 leaf 逻辑分为：

1. `EvaluateLeafFrontierChunk` 仍按 `(depth,path_count,node_count,max_k,reply_bucket)` 建立有序桶；
2. 未启用融合时逐桶调用，与旧行为相同；
3. 启用时按规范桶序、桶内 state index 顺序拼接索引；
4. `EvaluateLeafStateIndices` 按既有 `leaf_frontier_batch_states` 上限统一调用 `ProvePathSystemsByKOpt`；
5. proof 按保存的 state index 散布回原工作图。

融合不改变任何 state/reply/deletion-set 预算，不投机生成新的 cost row，不改变单 cursor 的 k、删除集合或 reconnect template 顺序。显式 CUDA 失败仍令受影响搜索失败关闭；PROVEN 仍经过 wavefront 内部、scan 即时与 batch commit 三层 CPU 重放。

CPU 与 CUDA 单元测试都要求融合前后：

- 搜索状态相同；
- `CUDAEE_HT_RECURSIVE_PROOF_V1` 字节相同；
- leaf cost tasks/cells 相同；
- frontier/cost batch 数不增加。

`CUDAEE_HT_SCAN_REPORT_V3` 新增 `fuse_leaf_buckets`、leaf frontier batches/states、桶数、峰值 batch 与全部 cost batch 数，使报告自身足以解释调度配置。

## 4. pcb3038 clean-commit 四路门禁

正式 run id 为 `pcb3038-ht-scan-20260901T202351Z-2558943`，clean commit 为 `b5fde2433eaa1ed9d477a8e06aec45371d8ab83a`，物理 GPU 1。输入哈希、目标切片与预算和前两次 8-target pilot 相同。

CPU、全 CUDA、混合、桶融合四路均得到：

- 12,383 states、14,285 replies、9,120 leaf calls、5,085 moves；
- 2 PROVEN、6 UNRESOLVED，提交相同两条边；
- 最终 6,702 条边、图哈希 `fe11f98414b04c0e`；
- 活动边 SHA-256 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`；
- 四份外层 V2 proof 均独立 CPU 重放，最优 tour 成本 137,694、哈希 `ca0238497c090a3c`、0 缺边。

| 指标 | CPU | 全 CUDA | 混合 | 混合 + 桶融合 |
|---|---:|---:|---:|---:|
| leaf frontier batches | 500 | 500 | 500 | 86 |
| leaf complexity buckets | 500 | 500 | 500 | 500 |
| CUDA cost batches | 0 | 1,835 | 1,835 | 382 |
| leaf ms | 27,179.349 | 26,524.886 | 25,898.284 | 26,074.319 |
| search ms | 34,003.964 | 33,933.522 | 32,671.735 | 32,905.281 |
| process wall ms | 34,047.458 | 34,038.294 | 32,775.193 | 33,005.548 |
| 相对 CPU search | 1.000× | 1.002× | 1.041× | 1.033× |

桶融合将 frontier batches 减少 `82.8%`，CUDA cost batches 减少 `79.2%`，但正式单次运行中 leaf 增加 176.036 ms，search 增加 233.546 ms。四份 proof SHA-256 因运行计时不同而不同；这不影响其独立语义重放与相同最终边集。

## 5. 重复性判断

为了避免用一次共享节点抖动决定默认值，另做了 `baseline -> fused -> fused -> baseline` 交错诊断。两次 baseline 的 leaf 均值为 26,654.481 ms、search 均值 33,492.696 ms；两次 fused 的 leaf 均值为 26,434.289 ms、search 均值 33,421.077 ms，分别仅 `1.008×` 与 `1.002×`。其中一个 fused run 的 Hamilton reply 阶段出现约 300 ms 抖动。

clean run 与交错诊断对 wall time 的方向并不一致，但都确认 batch 数稳定下降且 proof/工作量不变。因此当前只能得出：

- 桶边界确实是 launch 碎片的来源；
- 减少 1,453 个 CUDA cost batches 仍不足以稳定降低端到端时间；
- kernel launch latency 不是 26 秒 leaf 的主导部分；
- 开关应保留用于下一层融合实验，但默认继续关闭。

## 6. 下一步

在重构跨目标状态机前，先把 `ProvePathSystemsByKOpt` 的 leaf wall 拆成：

1. path-system 初始化与 outside/inside 枚举；
2. cursor `PrepareNextBlock`；
3. CPU/CUDA cost matrix 同步调用；
4. cursor `ConsumeBlock`、CPU witness 构造与 completeness fallback；
5. proof finalize 与 `VerifyPathSystemKOptProof`。

若 cost matrix 占主导，才继续跨目标/跨 frontier 融合或 kernel 优化；若 cursor consume 与 CPU completeness 主导，则应先优化可验证的主机算法。无论哪条路径，都继续以 proof 字节、完整 CPU 重放、最终图和受保护 tour 为硬门禁。
