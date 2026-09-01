# M5 HT leaf setup 画像与快照哈希复用

## 1. 研究问题

[Hamilton reply 主机优化](33_M5_HT_Hamilton_Reply_Host_Cache.md)后，pcb3038 的 CPU search 已降到约 4.9 秒，其中 leaf 占 91% 以上。旧报告只知道 leaf setup 约 1.76 秒，却无法区分 proof 初始化、覆盖扫描和 `KOptSearchCursor` 构造。本切片先补足诊断边界，再只消除被正式画像证实的重复工作。

计时字段不进入 `CUDAEE_PATH_KOPT_PROOF_V1`、HT proof 或任何删除判断；GPU 仍只提供候选/差分结果，所有删除继续由 CPU 精确证书授权。

## 2. V8/V10 诊断契约

提交 `f71b472` 将 `leaf_setup_ms` 细分为三个包含式子阶段：

| 字段 | 边界 |
|---|---|
| `leaf_proof_initialize_ms` | 建立 batch proof work、绑定快照/路径哈希并取得不可变 matching catalog |
| `leaf_coverage_scan_ms` | 检查 proof 是否已覆盖全部 outside matching，并取得首个未覆盖项 |
| `leaf_cursor_construct_ms` | 构造增量 k-opt cursor，包括 tour context |

`leaf_setup_ms` 仍是以上阶段及少量容器管理的包含式总量。报告升级为 `CUDAEE_HT_SCAN_REPORT_V8`，四路 summary 升级为 V10；新增 `leaf_cursor_searches_started`，要求 CPU、全 CUDA、混合和融合四路的实际 cursor 数完全相同。测试还要求 setup 大于等于三个子阶段之和。

## 3. clean-commit 基线画像

正式基线 run 为 `artifacts/pcb3038-ht-scan-20260901T215758Z-2610757`，clean commit 为 `f71b472`，使用物理 GPU 1。8 个目标的四路规范工作量完全一致：

- 12,383 states、14,285 replies、9,120 leaf calls；
- 9,891 个 cursor、727,635 个已消费 cost rows；
- 51,309,996 个精确 cost cells，全部由 CPU 认证；
- 987 个严格改善候选，通用 completeness fallback 为 0。

CPU setup 的组成如下：

| 子阶段 | 时间（ms） | setup 占比 |
|---|---:|---:|
| proof 初始化 | 1,629.824 | 92.57% |
| coverage 扫描 | 0.342 | 0.02% |
| cursor 构造 | 128.371 | 7.29% |
| residual | 2.180 | 0.12% |
| setup 总计 | 1,760.717 | 100.00% |

代码审计确认 `InitializeBatchedPathProof` 对 batch 中每个 path system 都调用一次 `GraphSnapshot::ContentHash()`。该哈希遍历同一不可变 graph 的完整坐标和活动 CSR；在 9,120 个 leaf states 上重复执行，与 1.63 秒画像吻合。coverage 扫描不是瓶颈，贸然线程化 cursor 也只能触及约 7%。

## 4. 安全优化

提交 `c968b01` 在 `ProvePathSystemsByKOpt` 进入 proof 初始化时计算一次快照哈希，并把该值写入同一 batch 的全部 proof work。安全边界如下：

1. API 接收 `const GraphSnapshot&`，一个同步 batch 内不存在提交或 CSR 重建；
2. 调用方不能注入任意哈希，值仍由证明实现直接调用 `ContentHash()` 得到；
3. 每份 proof 的序列化字段、规范搜索计数和 witness 均未改变；
4. 独立 `VerifyPathSystemKOptProof` 仍自行重算 graph 哈希，错误复用会失败关闭。

该优化只把相同纯函数调用从“每个 path system 一次”改为“每个 batch 一次”。不同 frontier batch 之间仍各算一次，因此融合模式因 batch 更少而保留更低的初始化时间；没有增加跨 epoch 缓存或失效协议。

## 5. clean-commit 收益与正确性门禁

优化后正式 run 为 `artifacts/pcb3038-ht-scan-20260901T220033Z-2613149`，clean commit 为 `c968b01`，协议、GPU 和输入与基线相同。

| 指标（ms） | `f71b472` CPU | `c968b01` CPU | 加速 |
|---|---:|---:|---:|
| proof 初始化 | 1,629.824 | 100.026 | 16.29× |
| setup | 1,760.717 | 232.023 | 7.59× |
| leaf | 4,496.767 | 2,981.133 | 1.51× |
| search | 4,914.625 | 3,398.001 | 1.45× |
| 进程 wall | 4,955.334 | 3,433.921 | 1.44× |

全 CUDA、混合、融合三路 search 分别为 `4,022.534 / 3,860.466 / 3,772.367 ms`，CPU 仍最快。融合模式因 86 个 leaf frontier batches 而非 500 个，proof 初始化进一步降到 28.011 ms，但较慢的 CUDA cost 路径仍抵消了这项收益。

优化前后均得到 2 PROVEN、6 UNRESOLVED、2 条提交边；最终图哈希保持 `fe11f98414b04c0e`，活动边 SHA-256 保持 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`。成本 137,694、哈希 `ca0238497c090a3c` 的受保护最优 tour 仍为 0 缺边。CPU Debug/Release 各 17/17、CUDA Release 20/20 均通过，GPU 差分通过，compute-sanitizer memcheck 为 0 errors。

## 6. 下一切片

优化后 CPU leaf 的 `cost_evaluate_ms` 为 2,345.263 ms，占 leaf 的 78.67%；其中逐 cell CPU 精确认证为 2,285.220 ms。setup 已不再适合优先做复杂线程化。下一切片应在不改变 cell 顺序、整数距离语义和 proof 计数的前提下评估 CPU cost rows 的有界并行；必须保留串行基线、逐 cell 差分和小 batch 避免线程开销的门禁。若并行调度不能在重复 clean run 中稳定获益，则撤回而转向 scorer 数据布局。
