# M5 有界全图 HT 目标扫描

## 1. 范围与结论

提交 `5ef3c63` 增加 `RunHtScanEpoch` 与 `ht-scan`，把原来必须手工指定一条边的 Hamilton–Tutte wavefront 接到确定性目标选择、显式资源预算、sidecar 汇集和不可变 epoch 提交链。提交 `cd5ec3e` 固化 CPU/CUDA 对照脚本。

这完成的是**单进程、单快照、有界目标切片基线**，不是完整 Local Elimination 流水线。`gpu-eliminate` 仍只自动执行 JV；`ht-scan` 是独立入口。多 epoch 重新排序、跨目标 GPU 融合、rl5915/d15112 资源门禁与多 GPU 仍未完成。

pcb3038 pilot 在 JV 固定点的 6,704 条边上扫描 8 个最高权重且度数安全的目标，CPU/CUDA 都证明并提交同两条边，最终剩余 6,702 条。CUDA 搜索为 33.646 秒，CPU 为 34.103 秒，仅 `1.014×`；该结果主要证明闭环和工作量一致，不能作为显著加速结论。

## 2. 确定性目标序列

`SelectHtTargetEdgeIds` 只选择：

- 当前 snapshot 中活动的边；
- 两个端点当前度数都大于 2 的边。

后者与最终 commit 的最小度门禁一致，避免搜索必然无法提交的目标。支持两种稳定顺序：

- `canonical`：按 `(u,v,edge_id)` 升序；
- `weight-desc`：先按整数边权降序，再用 `(u,v,edge_id)` 升序消除并列歧义。

`target_offset` 与 `max_targets` 在过滤、排序后的序列上定义；`max_targets=0` 或超过一百万会被拒绝。offset 恰好等于 eligible 数量表示空切片，并产生可重放的零删除 proof；越过末尾则失败。

offset 只用于**同一个不可变输入**的抽样或未来分片。一个 batch 提交后活动图和 eligible 序列已经改变；在新图上开始下一 epoch 时必须从 offset 0 重新排序，不能沿用旧图的 end offset。

## 3. 搜索与提交边界

每个切片严格执行：

```text
immutable graph
  -> deterministic target slice
  -> wavefront search per target
  -> immediate CPU replay of every PROVEN sidecar
  -> collect all successful sidecars
  -> CommitHtProofEpoch replays the complete batch again
  -> degree gate on graph copy
  -> publish once
```

- `PROVEN`：proof 立即经 `VerifyHtRecursiveProof` 完整重放，最后还会在 batch commit 中再次重放；
- `UNRESOLVED`：包括 state/reply/deletion 预算耗尽，不产生删除授权，扫描继续；
- `INVALID`：整批抛错；此前搜索全部只读，尚未进入 commit，调用方图保持不变；
- 多个合法 proof 同时触发度数门禁时，`proven_targets` 可以大于 `committed_targets`，只提交规范顺序允许的子集。

成功删除写入自包含 V2，内嵌原始 HT V1 sidecar。`ht-scan` 可选接受 `--protected-tour` 与 `--expected-cost`，在搜索前和 commit 后分别重新计算成本、规范 tour 哈希和活动边缺失数；最终门禁失败时不会写出结果。

## 4. 报告与复现

`CUDAEE_HT_SCAN_REPORT_V1` 记录总目标数、切片、状态、回复、叶调用、move 数、搜索 wall time 和每个目标的：

- `PROVEN/UNRESOLVED` 与原因；
- states/replies/leaves/moves/peak frontier；
- propagation backend、设备、cooperative block 数及 CPU 认证位；
- leaf backend、cost cells、CUDA batches、CPU long-tail 和驻留 cache 峰值；
- path-append tasks、Hamilton replies 与 end replies。

正式入口先由 CPU JV 生成并重放固定点，再对同一图运行纯 CPU 和显式 CUDA scan；脚本比较前 10 个与算法工作图相关的 attempt 字段、逐字节比较最终边文件，并用独立 CPU 进程重放两份 V2 proof：

```bash
CUDAEE_BENCHMARK_GPU=1 \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/pcb3038.tour \
tools/run_ht_scan_benchmark.sh pcb3038 8
```

固定 pilot 预算为：`missing-or-incompatible` c,d，neighborhood 25，最多 5 个根候选，根 replies 10,000，3/4/5-opt、每叶 100 个 deletion sets，深度 2，每目标最多 2,000 states、20,000 total replies、每 move 2,000 replies，point/end 各最多 3 个候选。资源上限只影响强度，不影响 soundness。

## 5. pcb3038 clean-commit pilot

run id 为 `pcb3038-ht-scan-20260901T194637Z-2540418`，clean commit 为 `cd5ec3e55680ab168b6a0bb6d048434b66c7b405`，物理 GPU 1（RTX 4000 Ada）。输入由同一运行中的 CPU JV 固定点产生：

- TSPLIB SHA-256：`0b2229669b5d2916e812c36eaf76fb7b2bcb7ea09c6828e2599e0733ba18e933`；
- 原始边集 SHA-256：`9e6e2e612c0333c00d04cd2f91e3358e6f11508da1c56453da6e45da674cdcc7`；
- JV 固定点边集 SHA-256：`9b2229bc047f4fa107a474bf8201b7b086db11df64485217d36c67e53c326a54`；
- HT 输入 snapshot hash：`863392eda4798d1a`。

| 指标 | CPU | CUDA |
|---|---:|---:|
| attempted / proven / unresolved | 8 / 2 / 6 | 8 / 2 / 6 |
| states / replies / leaf calls | 12,383 / 14,285 / 9,120 | 12,383 / 14,285 / 9,120 |
| moves generated | 5,085 | 5,085 |
| search time | 34,102.630 ms | 33,646.432 ms |
| process wall | 34,142.217 ms | 33,753.721 ms |
| 相对加速 | 1.000× | 1.014× search / 1.012× wall |

CUDA 处理 51,309,996 个 leaf cost cells、1,835 个 cost batches，leaf device cache 峰值 8,948,163 bytes。CPU scalar 路径不构造 CUDA cost matrix，因此其报告中的 `leaf_cost_cells=0` 不是工作量为零；跨后端等价性由目标、状态、回复、叶、move 和 peak frontier 签名确认。

成功目标为 `511-513`（edge id 1194）和 `155-156`（edge id 372）。6 个 unresolved 目标中 5 个达到 2,000-state 上限，另一个在扩张中耗尽全局资源预算；均未被误记为失败证明。最终结果：

- active edges：`6704 -> 6702`；
- final hash：`fe11f98414b04c0e`；
- edge SHA-256：`0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`；
- CPU/CUDA 两份 V2 均由独立 CPU `verify` 重放到相同哈希；
- 成本 137,694、tour hash `ca0238497c090a3c` 的最优巡回在两份输出中均为 0 缺边。

## 6. 性能判断与下一步

单次结果只有约 1% 差异，不足以声称 CUDA HT 已显著加速。当前每个目标独立建立主机 BFS 工作图；GPU 只在一个目标内部融合叶成本、reply 和 continuation，随后 CPU 又逐项完整认证。33 秒中大量时间仍属于建图、规范化和 verifier，1,835 个 leaf launches 也说明 batch 粒度不足。

下一阶段应优先：

1. 将多个目标的同层 leaf cost rows 合为跨目标批次，同时保留每目标独立预算和 proof 顺序；
2. 分段统计 host graph build、CPU certification、各 CUDA 候选器与 proof extraction；
3. 在相同预算下扩展目标数和 pcb3038 重复运行，再进入 rl5915/d15112；
4. 新图的下一 epoch 从 offset 0 重新排序，测量强度固定点；
5. 单卡跨目标融合稳定后再按不可变 snapshot 切片到多 GPU。
