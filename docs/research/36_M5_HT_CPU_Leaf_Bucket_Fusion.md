# M5 HT CPU leaf 复杂度桶融合

## 1. 研究问题

[CPU 精确成本矩阵行并行](35_M5_HT_CPU_Cost_Row_Parallelism.md)把 pcb3038 8-target CPU search 降至约 1.87 秒，但非融合调度仍产生 500 个 leaf frontier batches。早期在 CUDA leaf 上评估过复杂度桶融合，当时 kernel 不是主瓶颈，所以默认关闭。行并行后条件发生变化：更大的 CPU cost batch 能让更多 cells 越过 8,192-cell 并行门槛，并且减少批级快照哈希和 OpenMP 团队启动。

本切片不改变 leaf 搜索器、成本矩阵、改善 witness 选择或证明格式；只在既有 `--fuse-leaf-buckets 1` 调度开关上增加 CPU backend 对照。

## 2. V12 五路门禁

`tools/run_ht_scan_benchmark.sh` 现在在同一 JV 固定点和目标切片上运行：

1. 非融合 CPU；
2. 复杂度桶融合 CPU；
3. 全 CUDA；
4. CPU 主路径 + CUDA leaf cost；
5. 融合的 CPU 主路径 + CUDA leaf cost。

summary 升级为 `CUDAEE_HT_SCAN_BENCHMARK_SUMMARY_V12`，并输出 CPU 融合路径的完整 leaf 子阶段、并行 batches/cells/峰值线程和相对 CPU 基线的 leaf/search/wall 加速。脚本在产生 summary 前必须通过：

- 五份最终边文件逐字节相同；
- 五份 target 工作签名相同；
- 每份 proof 由独立 CPU 进程重放；
- cursor 数、消费 rows、改善候选、completeness 计数、总 cost cells 和 Hamilton reply 规范计数相同；
- 融合路径可以改变 batch 边界和越过并行门槛的 cells，但不能改变总 cells 或 proof 语义；
- 提供最优 tour 时，五份结果都必须保留其全部边和精确成本。

1-target 冒烟 `artifacts/pcb3038-ht-scan-20260901T222304Z-2627218` 先通过上述全部门禁。

## 3. pcb3038 clean-commit 结果

正式 run 为 `artifacts/pcb3038-ht-scan-20260901T222415Z-2628279`，绑定 clean commit `63133c7`，使用物理 GPU 1、8 个固定 CPU cost threads 和成本 137,694 的受保护最优 tour。

| CPU 指标（ms） | 非融合 | 融合 | 变化 |
|---|---:|---:|---:|
| frontier batches | 500 | 86 | 5.814× 更少 |
| parallel cost batches | 726 | 208 | 3.490× 更少 |
| parallel cells | 48,879,635 | 51,047,847 | 95.263% → 99.489% |
| proof initialize | 100.273 | 26.206 | 3.826× |
| cursor construct | 130.123 | 125.891 | 1.034× |
| leaf setup | 232.973 | 154.647 | 1.506× |
| CPU exact certify | 690.959 | 572.607 | 1.207× |
| cost evaluate | 750.808 | 630.373 | 1.191× |
| leaf | 1,420.648 | 1,235.915 | 1.149× |
| search | 1,840.895 | 1,654.602 | 1.113× |
| 进程 wall | 1,878.249 | 1,691.490 | 1.110× |

融合使 99.489% 的 cells 进入并行 CPU 矩阵，同时把批级 proof 初始化降低 73.87%。`cursor_prepare_ms` 因更大批次从 175.795 ms 增至 187.055 ms，但 cost 和 setup 收益更大，最终 leaf/search 分别下降 13.00%/10.12%。

五路均为 2 PROVEN、6 UNRESOLVED，提交相同 2 条边；它们均消费 51,309,996 个已 CPU 认证 cells、9,891 个 cursors、727,635 个 rows 和 987 个改善候选。最终图哈希为 `fe11f98414b04c0e`，活动边 SHA-256 为 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`，最优 tour 哈希为 `ca0238497c090a3c`且缺边数为 0。

同一 run 的 CPU 融合 search 为 1,654.602 ms，仍快于全 CUDA、非融合 hybrid 和融合 hybrid 的 2,482.606/2,260.692/2,093.695 ms。这些 CUDA 路径仍要求同步 CPU 全矩阵认证，所以不将 kernel 时间单独宣传为端到端收益。

## 4. 决策与下一切片

CPU 融合在 pcb3038 的冒烟和正式 8-target 两个规模上均稳定获益，因此保留 V12 的长期对照。但开关仍默认关闭：目前只有一个中型实例具有可用的最优 tour 安全见证，且早期 CUDA 路径曾出现无收益。在 rl5915/d15112 补齐受保护 tour 并通过同样五路门禁前，不把单实例结果改成全局默认策略。

融合后 CPU leaf 仍占 search 的 74.70%，其中 cost evaluate 占 leaf 的 51.00%。下一性能切片应优先用多实例/多次重复确认调度收益，再针对 scorer 的距离数据布局或 batch 间 OpenMP 团队复用做单变量实验。任何后续优化都必须保留 CPU 逐 cell 整数认证与 proof 重放。
