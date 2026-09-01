# M5 HT point-candidate 全维选择画像

## 1. 可观测性

[根 child 排除实验](39_M5_HT_Root_Child_Normalization_Profile.md)后，提交 `5944476` 将 report/summary 升级为 V12/V15，并对 `BuildPointCandidateNodes` 记录：

- `point_candidate_scans`：实际选择 point 候选的 frontier state 数；
- `nodes_checked`：完整维度循环检查的节点总数；
- `nodes_ranked`：通过 target/path/degree 过滤后进入排序的节点总数；
- `nodes_selected`：按 `max_neighborhood` 裁剪后交给 Hamilton reply 的节点总数；
- `scan_ms` 与 `sort_ms`：过滤/中点整数评分，以及全量排序/裁剪/节点物化。

四个整数计数均为规范工作量，V15 要求五路完全一致。host-build unprofiled residual 继续扣除 root normalize、point scan 和 point sort；所有计时均不进入 proof。

## 2. clean-commit 三实例画像

正式 runs 均绑定 clean commit `5944476cde98186b61a1b6d6ed56c13facdf23d3`、物理 GPU 1、8 个 CPU cost threads 和公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T231701Z-2662849`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T231713Z-2663420`；
- d15112：`artifacts/d15112-ht-scan-20260901T231720Z-2662848`。

CPU fused 画像如下：

| 实例 | state scans | nodes checked | nodes ranked | nodes selected | scan（ms） | 全量 sort（ms） | host residual（ms） | scan+sort 占比 | 未画像（ms） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pcb3038 | 1,102 | 3,347,876 | 3,337,754 | 27,550 | 41.514 | 137.639 | 191.374 | 93.613% | 11.406 |
| rl5915 | 1,059 | 6,263,985 | 6,254,906 | 26,475 | 71.287 | 298.303 | 381.194 | 96.956% | 9.318 |
| d15112 | 957 | 14,462,184 | 14,354,406 | 23,925 | 166.146 | 858.648 | 1,075.813 | 95.258% | 22.390 |

三个实例都恰好选择 `25 × state scans` 个节点，说明正式协议的 `max_neighborhood=25` 每次都触发裁剪。当前实现却先对几乎整个实例的候选执行全量 `std::sort`，然后只保留前 25 项；d15112 的排序单项已占 host residual 79.817%。

五路工作签名、proof 重放、最终边文件和受保护 tour 均通过；最终哈希和活动边 SHA-256 保持不变。

## 3. 下一单变量实验

先只把有界分支的全量排序替换为 `std::partial_sort(begin, begin + 25, end)`，随后裁剪；比较键仍是严格全序 `(midpoint_score,node)`，因此前 25 个节点及其顺序应逐项不变。`max_neighborhood=0` 或候选数不超过上限时继续全量排序。

门禁要求：

1. CPU Debug/Release、CUDA Release 和 HT compute-sanitizer 全部通过；
2. 三实例 V15 五路规范计数、proof、边文件和 tour 不变；
3. baseline 与优化后的每一路规范 proof 和活动边逐字节比较；
4. 只报告 sort、work graph、search 和 wall 的实测收益，不把未改变的 scan 宣传为优化收益。
