# M5 HT leaf 子阶段画像

## 1. 研究问题

[leaf 复杂度桶融合](27_M5_HT_Leaf_Bucket_Fusion.md)把 CUDA cost batches 从 1,835 降到 382，却没有得到稳定的 wall-time 收益。本切片因此不再假定 kernel launch 是主瓶颈，而是把 `ProvePathSystemsByKOpt` 的同步 wall time 拆为：

- path proof 初始化和 cursor 构造；
- cursor block 准备；
- CPU/CUDA cost matrix 计算；
- cost row 散布；
- cursor 消费、CPU witness 构造和全模板 completeness fallback；
- scalar CPU 搜索、proof 应用及 leaf proof 独立复核。

这些字段仅用于诊断，不进入 `CUDAEE_PATH_KOPT_PROOF_V1` 或 HT proof，也不参与任何删除决策。

## 2. 实现与报告契约

提交 `fe99f38` 在 `PathSystemKOptBatchResult`、wavefront、scan attempt 和 scan 总计之间传递以下包含于 `leaf_ms` 的字段：

| 字段 | 同步计时边界 |
|---|---|
| `leaf_setup_ms` | path proof 初始化、活动 work 扫描和 cursor 构造 |
| `leaf_cursor_prepare_ms` | `PrepareNextBlock` |
| `leaf_cost_evaluate_ms` | `EvaluateKOptTemplateCosts`，包含同步 CUDA API 和 auto 回退 |
| `leaf_cost_scatter_ms` | 融合矩阵向单 cursor row 的确定性复制 |
| `leaf_cursor_consume_ms` | `ConsumeBlock`，包含 CPU 重连和 CUDA 无命中后的完整兜底 |
| `leaf_scalar_search_ms` | CPU leaf 的 `FindKOptWitness` |
| `leaf_apply_ms` | witness 复核、inside coverage 和 proof record 更新 |
| `leaf_proof_verify_ms` | 成功 leaf 的 `VerifyPathSystemKOptProof` |

`CUDAEE_HT_SCAN_REPORT_V4` 保存全局和逐 target 数值；四路 benchmark summary 升为 V5，并按包含关系计算 leaf residual。测试要求所有子阶段非负且总和不超过 `leaf_ms`。CPU Debug/Release 各 17/17、GPU 1 上 CUDA Release 20/20 均通过。

## 3. pcb3038 clean-commit 四路门禁

正式 run id 为 `pcb3038-ht-scan-20260901T203636Z-2565629`，clean commit 为 `fe99f38`，使用物理 GPU 1（RTX 4000 Ada）。输入、目标切片和预算与此前 8-target pilot 相同。

四路均产生相同工作签名：12,383 states、14,285 replies、9,120 leaf calls、5,085 moves，结果均为 2 PROVEN、6 UNRESOLVED，并提交相同两条边。最终图哈希为 `fe11f98414b04c0e`，活动边 SHA-256 为 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`。全部 proof 经独立 CPU 重放；成本 137,694、哈希 `ca0238497c090a3c` 的受保护最优巡回仍为 0 缺边。

| 指标（ms） | CPU | 全 CUDA | 混合 | 混合 + 桶融合 |
|---|---:|---:|---:|---:|
| leaf 总计 | 27,231.421 | 26,354.767 | 26,489.332 | 26,318.716 |
| setup | 5,680.111 | 5,854.297 | 5,864.498 | 5,851.563 |
| cursor prepare | 0 | 194.977 | 192.979 | 219.707 |
| cost evaluate | 0 | 429.518 | 583.746 | 499.407 |
| cost scatter | 0 | 40.795 | 41.769 | 41.527 |
| cursor consume | 0 | 19,722.249 | 19,695.247 | 19,571.942 |
| scalar search | 21,474.526 | 0 | 0 | 0 |
| apply + leaf verifier | 72.588 | 74.619 | 74.937 | 73.653 |
| search 总计 | 34,094.769 | 33,487.773 | 33,334.581 | 33,184.469 |
| 相对 CPU search | 1.000× | 1.018× | 1.023× | 1.027× |

混合 leaf 中，setup 占 `22.139%`，cost evaluate 仅占 `2.204%`，cursor consume 占 `74.352%`。即使忽略 GPU cost 的全部时间，理论上也只影响约 2%；继续跨桶或跨目标合并 kernel 不是当前最高收益项。

CPU scalar search 比混合 cursor consume 多 1,779.280 ms，但混合还要支付 cursor prepare、cost 和 scatter，因此 leaf 净收益只有 `1.028×`。这解释了为什么 5,130 万个精确 cost cells 已在 GPU 上执行，端到端仍只有约 2% 改善。

## 4. 热点的代码含义

代码检查发现两个可保持 proof 语义不变的重复构造：

1. 每个 path-system work 都重新枚举仅由 `path_count` 决定的 outside/inside matchings，并重新生成完整兼容表；
2. CPU scalar、CUDA cost 和每个 cursor consume 都反复生成仅由 `k` 决定的 proper reconnect templates。

此外，CUDA cost row 无可接受候选时，`TryReconnectFromCostRow` 会调用 CPU `TryReconnect` 穷举整行。这是 `cursor_consume` 的主要候选热点，但它同时承载当前“GPU 只作候选 oracle”的 completeness 契约，不能在没有单独安全论证与门禁时直接删除。

## 5. 下一实现切片

先将 path-count matching catalog 和 `k=3/4/5` reconnect table 改为线程安全、延迟构造的进程内不可变缓存。缓存键必须由完整结构参数决定，调用方只能持有 const 引用；proof 中的生成器哈希和规范顺序保持不变。验收要求：

1. CPU scalar、CUDA、不同 frontier/bucket 配置的 proof 字节及工作计数不变；
2. CPU/CUDA 全量测试和独立 proof/tour 门禁通过；
3. 同一 8-target 协议中 setup 与 cursor consume 分别重新计量；
4. 只有重复运行确认收益后才保留优化。

完成不可变表缓存后，再把 `cursor_consume` 拆成模板准备、GPU 候选复核与 CPU completeness fallback。若要实验“GPU 漏报只导致 UNRESOLVED”的候选模式，必须作为显式非默认策略，并证明任何 GPU 错误都不能把未知状态变为 PROVEN；当前默认完整 CPU 兜底不变。
