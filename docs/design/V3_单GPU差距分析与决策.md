# V3 单 GPU 差距分析与决策

## 目标

本分支只研究单 GPU 的 Local Elimination 主线。目标是在不改变“同一不可变快照上由 CPU 精确验证后才能删边”的前提下，将递归 Hamilton–Tutte 搜索从完整 wavefront 改造成可短路、可跨目标合批的转置执行，并为收缩后的精确 DP 提供紧凑 CPU 与 CUDA 后端。

多 GPU、LP–Local 固定点、完整 reduced-cost pricing 和 LKH 集成本轮不实现。已有多 GPU 路径保留为兼容功能和回归对象。

## 当前实现与 V3 的对应关系

| V3 能力 | 当前状态 | 本分支决策 |
| --- | --- | --- |
| 不可变 epoch 与 CPU 证书链 | 已实现 | 直接复用 |
| 3/4/5-opt、matching/coverage CUDA 批处理 | 已实现 | 直接复用 |
| Hamilton/end reply CUDA 批处理 | 已实现 | 纳入统一调度指标 |
| 完整 AND/OR wavefront | 已实现，但先生成整棵有界工作树 | 保留为 oracle 与回退 |
| 严格短路 Trace/replay | 缺失 | 首先实现，作为调度决策依据 |
| stackless/transposed scheduler | 缺失 | 新增显式 opt-in 后端 |
| 紧凑滚动 exact DP | 缺失，当前保存完整 cost 与 predecessor | 先做 CPU oracle，再做 CUDA `k<=13` |
| 单次验证授权 token | 缺失，热路径存在重复 proof 重放 | 在正确性门禁后合并 |

## 已测基线

提交 `93825c3`、d15112、相同 32 个目标的现有结果如下：

- RTX A5000 单 GPU wavefront：进程 wall 中位数约 `2.845 s`。
- 8 线程 CPU fused wavefront：进程 wall 中位数约 `1.608 s`。
- CPU 递归 DFS：状态数由 wavefront 的 `40,044` 降至 `19,498`，已证明目标由 `11` 增至 `18`，但标量执行约 `285.4 s`。

结论是短路次序确实同时改善工作量和证明强度，但必须保留批处理；仅把 DFS 搬到 GPU 或继续优化完整 wavefront 都不足以达到目标。

## 不变量

1. GPU 结果只能生成候选；每一条删除必须由同一快照上的 CPU 精确 verifier 授权。
2. speculation 只允许增加物理执行量，不得改变规范 child 次序、逻辑预算或规范 proof。
3. 队列溢出、设备错误、数值范围不受支持或证书不完整时保留边。
4. `wavefront` 保持兼容默认值；`transposed` 在研究分支中显式启用。
5. 所有代码、依赖、构建缓存和实验产物只允许位于本仓库内。

## 阶段门禁

1. Trace/replay 能复现完整 wavefront 与 `s=1/2/4/8` 的逻辑工作量。
2. 紧凑 CPU DP 与旧完整 DP 在随机和穷举样本上完全一致。
3. CUDA DP 在 `k<=13` 上逐任务与 CPU 精确结果一致；成功 witness 仍由 CPU 重建和验证。
4. `transposed,s=1` 与递归 DFS 的结论和规范 proof 一致。
5. d15112 32 目标不少于 18 个 PROVEN，且单 GPU wall 至少比当前最佳 CPU 快 1.25 倍，才考虑改变默认后端。
