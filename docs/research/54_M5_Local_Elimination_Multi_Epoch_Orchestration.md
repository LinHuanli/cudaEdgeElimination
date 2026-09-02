# M5 JV—HT 多 epoch Local Elimination 编排

## 1. 结论与范围

提交 `e74b197` 增加库入口 `RunLocalElimination` 和 CLI `local-eliminate`，把原先相互独立的 JV 固定点与单快照 `ht-scan` 接成一个有界状态机。它完成以下闭环：

1. 先把 JV 跑到固定点；
2. 在不可变快照上扫描一个有界 HT 目标切片；
3. 有删除时发布一次 HT epoch，在新图上重新达到 JV 固定点，并从 offset 0 重排 HT 目标；
4. 无删除时保持同一快照，推进 offset 扫描下一切片；
5. 把所有 JV records 与 HT sidecars 合并为一个可独立 CPU 重放的 V1/V2 消元证明。

这是**多 epoch 调度与证明组合基线**。它没有实现跨目标 GPU 工作图或多 GPU，也没有把尚未完成的 M3.1 LP 删除授权接入固定点；活动 edge-id 紧凑 launch 已另行评测并因端到端回退而撤销。

## 2. 调度状态机

```text
复制调用方图
    |
    v
JV(max_jv_rounds) --预算耗尽且仍有提交--> jv-round-limit
    |
    | JV 固定点
    v
HT scan(offset, max_targets)
    |                           |
    | committed > 0             | committed == 0
    v                           v
offset = 0                  offset += attempted
重新跑 JV 固定点           到达 eligible 末尾？
    |                           | yes
    +--------> 新快照重排       v
                             converged
```

调度器拥有 `target_offset`，因此 `LocalEliminationOptions::ht_scan_options.target_offset` 必须为 0，CLI 也拒绝 `--target-offset`：

- HT 有提交时，活动图和目标严格次序都已改变，下一轮必须从 offset 0 重新选择；
- HT 无提交时，图哈希不变，才允许按 `offset += attempted` 继续当前 sweep；
- 一个完整无删除 sweep 才返回 `converged`，单个无删除切片不能冒充固定点；
- `max_ht_epochs` 用尽返回 `ht-epoch-limit`，结果仍是安全、可重放的部分消元，不标成收敛；
- JV 在最后一个允许 round 仍有提交时返回 `jv-round-limit`，并且不进入下一次 HT scan。

每次 HT 提交后即使 HT epoch 预算已经用完，也会先跑完随之产生的 JV 固定点，再返回 `ht-epoch-limit`。这样不会留下“HT 已提交、可立即发现的 JV 候选却未处理”的半个组合阶段。

## 3. 原子性与证明组合

整个 `RunLocalElimination` 在 `GraphSnapshot` 工作副本上执行。参数非法、HT `INVALID`、CUDA/CPU 差分失败、sidecar 复核失败或 proof 聚合不变量失败都会抛错，调用方图不变。正常的预算终止会发布当前安全部分结果。

每个子阶段仍走原有授权链：

- JV 的 CPU 路径直接产生候选；CUDA 路径的每个候选仍逐条通过 `VerifyJvCandidate`；
- 每个 HT `PROVEN` sidecar 在 scan 中即时完整重放一次；
- `CommitHtProofEpoch` 在同一不可变快照上再次整批重放，然后才在图副本上执行度数门禁；
- 联合调度器不解释浮点值，也不新增任何删除规则。

聚合器只做容器级变换：

1. 要求子阶段 `initial_hash` 等于前一阶段 `final_hash`；
2. 将有实际 records 的子阶段 epoch 连续重编号；
3. 将 HT `certificate_index` 加上已聚合 sidecar 数，并检查 32 位与一百万条安全上限；
4. JV record 必须保持 `kNoEliminationCertificate`；
5. 零提交轮的计时不写入 proof metrics，以免在 record epoch 中制造空洞；它们完整保留在 Local Elimination stage report；
6. 输出 proof 最终仍由现有 `ReplayProof` 根据每轮快照哈希、方法专属 CPU verifier、确定性提交顺序和最终哈希独立重放。

如果联合过程没有 HT 删除，proof 继续写 `CUDAEE_PROOF_V1`；出现任一 HT 删除时自动写自包含 `CUDAEE_PROOF_V2`。

## 4. CLI 与报告

核心入口为：

```bash
build/cuda-release/cudaee local-eliminate \
  --tsp INSTANCE.tsp --edges INPUT.edg \
  --output OUTPUT.edg --proof OUTPUT.proof --report OUTPUT.report \
  --backend auto --max-jv-rounds 100 \
  --max-ht-epochs 100 --max-targets 8 \
  --target-order weight-desc \
  [HT wavefront budgets]
```

可选的 `--protected-tour FILE --expected-cost COST` 在入口和全部阶段结束后分别检查成本、节点置换、规范 tour hash 与活动边缺失数；失败时不写出本次结果。该门禁是额外的已知 witness 保留约束，不替代方法专属证明。

`CUDAEE_LOCAL_ELIMINATION_REPORT_V1` 为每个 stage 记录：

- `JV/HT` 类型、实际 backend、初末快照哈希和边数；
- proposed/verified/rejected/committed；
- JV round 数；
- HT eligible/offset/attempted/proven/unresolved；
- stage wall time；
- 全局 termination、proof record/sidecar 数及 protected-tour 结果。

细粒度 leaf/path/reply 性能画像仍使用单快照 `ht-scan` V16 报告；联合报告只承担调度和哈希链审计，避免复制体积巨大的逐目标 sidecar/attempt 数据。`local-eliminate` 会透传 reply CUDA 驻留与精确任务去重开关；同一进程的 graph/增长 workspace 可跨 targets 和 stages 复用，但任务结果只在当前 batch 内按完整 key 折叠，新 epoch 的完整坐标/CSR 键不相等时仍必须重新上传。

## 5. 固定 8 点穷举门禁

`unit.hamilton_tutte` 在 8 点完整图上覆盖两种预算路径：

- `max_ht_epochs=1`：JV 两轮提交 7 条，HT 对重排后的首目标 `2-4` 提交 1 条，再次 JV 达到固定点；联合 proof 含 8 records、1 个 HT sidecar 和 2 个连续 record epochs，最终 20 条活动边；
- `max_ht_epochs=100, max_targets=1`：无提交切片逐一推进，任何提交后重新从 offset 0 排序，最终以 `converged` 结束；共提交 7 条 JV 和 4 条 HT，活动边 `28 -> 17`，proof 含 11 records、4 个 sidecars 和 5 个连续提交 epochs。

测试固定首节点后穷举全部 7! 个 Hamilton 巡回，最优值为 251。11 条被删边各自“强制包含该边的最短巡回”都严格大于 251，差值为 8–44；随后从原始完整图独立重放联合 proof，得到与调度器完全相同的最终图哈希。另有 `max_jv_rounds=1` 门禁确认 JV 预算耗尽时只返回 7 条 JV 的可重放部分结果，不启动 HT。

测试目录中的成本 395 tour 只用于单个有界 epoch 的额外 protected-tour 集成门禁，并不是该 8 点实例的最优标签；完整收敛正确性以上述 7! 穷举为准。

## 6. pcb3038 干净提交门禁

artifact：`artifacts/pcb3038-local-elimination-20260902-OiRUCG`。运行绑定干净提交 `e74b197627422f76ebdc0e79a54ff43e67ccf83e`，CPU exact-cost 使用 8 线程，CUDA 使用物理 GPU 1。输入、最优 tour、输出边和规范 proof 的 SHA-256 均记录在 `run.meta`。

固定 M5 pilot 预算下执行 3 个 HT epochs，每个最多 8 targets：

| stage | 输入边 | offset / attempted | proven / unresolved | committed | 输出边 |
|---|---:|---:|---:|---:|---:|
| JV 固定点 | 6,883 | — | — | 179 | 6,704 |
| HT epoch 0 | 6,704 | 0 / 8 | 2 / 6 | 2 | 6,702 |
| JV 固定点 | 6,702 | — | — | 0 | 6,702 |
| HT epoch 1 | 6,702 | 0 / 8 | 0 / 8 | 0 | 6,702 |
| HT epoch 2 | 6,702 | 8 / 8 | 1 / 7 | 1 | 6,701 |
| JV 固定点 | 6,701 | — | — | 0 | 6,701 |

结果因显式 3-epoch 上限以 `ht-epoch-limit` 结束，不标记为全图收敛。CPU 与全 CUDA 路径均得到：

- 182 records（179 JV + 3 HT）和 3 个 HT sidecars；
- 最终内容哈希 `dce8912b10c3736e`；
- 完全相同的活动边文件，SHA-256 为 `235dc6845d73ec3cbfb5dcf54e8b6e9992fc58860f3741e2048050f084367205`；
- 去掉非授权的 backend/timing metrics 后，完全相同的规范 proof，SHA-256 为 `2a07a1da2462728dfebc208a320a153a4062621f526d703068b8f94df1f91c09`；
- 两份 proof 均由 CPU Release 从 6,883 条原始边独立重放到 6,701 条；
- 官方最优值 137,694、tour hash `3d014f3fdfa4cd64` 的 witness 在两份输出中均为 0 缺边。

这是调度与正确性门禁，不是 CPU/CUDA 性能对比；两条路径的上下文初始化、同步和候选后端不同，单次 wall time 不用于加速结论。

## 7. 完整验证

- CPU Debug：20/20；
- CPU Release：20/20；
- CUDA Release（物理 GPU 1）：23/23，含 pr299 CPU/CUDA 差分；
- `local-eliminate` pcb3038 CPU/CUDA：边文件一致、规范 proof 一致、两份独立重放、两份最优 tour 门禁；
- CUDA Debug `local-eliminate`（物理 GPU 2）compute-sanitizer memcheck：0 errors；
- 非法 caller offset 保持调用方图哈希不变；
- JV/HT 两类预算终止均保持可重放的安全部分结果。

## 8. 下一步

多 epoch 重排已经成为可执行基线，但 pcb3038 的三个 8-target epochs 只覆盖目标序列的一小部分。下一阶段仍需：

1. 根 `c,d` screen 跨 target 融合已因端到端回退而排除；reply CUDA 静态图和增长 workspace 驻留已由后续 d15112 七对 A/B 保留。下一步只画像 leaf/reply 语义结果与 work-graph 子结构的实质重复，同时保持每目标预算与 proof 顺序；
2. 活动 edge-id 紧凑 launch 已在 d15112 上因端到端回退而排除；只有 inactive 比例显著提高或设备端能免费维护索引时才重新评估；
3. 单卡闭环稳定后再做多 GPU 静态切片和 CPU 汇总复核；
4. M3.1 能输出逐边安全 LP 授权后，再评测 LP—JV—HT 交替固定点；
5. 在 rl5915/d15112 上先跑有界多 epoch 正确性门禁，再决定完整 sweep 的资源预算。
