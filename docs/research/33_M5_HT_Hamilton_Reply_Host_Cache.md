# M5 HT Hamilton reply 主机去重与过滤缓存

## 1. 研究问题与结论

[CPU matrix 公平基线](32_M5_HT_CPU_Matrix_Baseline.md)把 pcb3038 8-target CPU search 降至 11.015 s 后，Hamilton reply 占 `6.079 s`、即 `55.19%`。代码审计发现 CPU batch 路径存在三层重复：

1. `EvaluateHtHamiltonReplies` 已验证完整图和 target，却为每个 center 再调用公开单 center API，后者又完整扫描并验证 CSR；
2. 同一 batch 中大量状态选择相同 center，却逐次重新枚举相同邻边对；
3. 对一个 center 的每对邻点都重复计算两条 center-edge 的 2-opt 生存条件和固定 target/center 距离。

提交 `f12c181` 消除这些重复。正式 8-target clean run 中：

- 27,598 个逻辑 centers 在 72 个 batches 内只需实际枚举 1,395 次，批内重复约 `94.95%`；
- CPU Hamilton reply 从 `6,078.790 ms` 降为 `18.914 ms`，约 `321.39×`；
- CPU search 从 `11,014.960 ms` 降为 `4,869.906 ms`，约 `2.262×`；
- 四路仍生成相同 245,965 replies，拥有相同工作图、证明结果、最终边集和最优 tour。

该优化只复用同一不可变 `(graph snapshot, target, center)` 的确定性枚举结果，不改变 reply 集合或顺序。

## 2. 实现

### 2.1 单次验证与内部可信枚举器

公开 `EnumerateHtHamiltonReplies` 仍完整验证图、target 和 center。批 API 则：

1. 入口只调用一次 `ValidateHtGraph` 和 `ValidateTarget`；
2. 一次验证所有 centers 的范围及 target-endpoint 排除条件；
3. 随后调用仅在当前翻译单元可见的 `AppendHtHamiltonRepliesUnchecked`。

“unchecked”只表示不重复扫描已验证 CSR；它不对外公开，也不绕过 batch 入口验证。CUDA 输出仍与完整 CPU offsets/replies 数组逐元素比较。

### 2.2 batch 内 center 去重

每个 batch 建立 `dimension` 长度的 center-to-cache-index 表。一个 center 首次出现时才生成规范 reply vector，后续出现直接复制该 vector 到逻辑输出 slice。每个输入 center 仍获得独立的 `[offsets[i], offsets[i+1])` 区间，因此父状态映射、排序和 reply 预算语义不变。

缓存生命周期仅限一次 API 调用：

- 不跨 target；
- 不跨 graph snapshot；
- 不跨线程或异步 stream；
- 不保存到 proof。

因此不存在过期图状态复用问题。

### 2.3 每邻边预计算 quick filter

对一个 center，先按规范 CSR 顺序为每个邻点计算：

- `Distance(center, neighbor)`；
- 对 `target` 的 2-opt 生存布尔值。

邻边对循环随后只读取两个布尔值和两条 center costs，再计算 pair-specific 的 `Distance(first, second)`。固定的 target cost 与 `Distance(target.u, center)+Distance(target.v, center)` 也只计算一次。所有比较继续使用 `__int128`，输出循环仍是原 CSR 的 `first_offset < second_offset` 顺序。

独立 verifier 中故意展开的原始公式没有复用这条优化路径；测试侧 `ReferenceReplies` 也继续逐邻边对直接计算，从而避免生产 helper 自证。

## 3. 可观测性与门禁

`HtHamiltonReplyBatchResult` 和 wavefront/scan 聚合新增：

- batch 数、逻辑 center 数、每 batch 去重后 center 数之和；
- 实际枚举的邻边对数；
- validation、CPU enumerate、CUDA evaluate 和 CUDA full-array compare 时间。

`hamilton_reply_ms` 是上述子阶段的包含式总量。HT scan report 升级为 V7，benchmark summary 升级为 V9；四路必须拥有相同 batches、逻辑 centers、批内 unique-center 累计、实际邻边对和输出 replies，否则脚本失败。

验证包括：

- 30 组随机 8 点完整图、所有 target 和合法 center，与独立 LocalElimination 公式逐 reply 比较；
- `EUC_2D/CEIL_2D`、重复 center 和非法 center；
- CUDA offsets/replies 全数组与优化 CPU batch 一致；
- CPU Debug/Release 各 17/17、CUDA Release 20/20；
- `compute-sanitizer` 0 errors；
- V9 1-target 冒烟和正式四路门禁。

## 4. pcb3038 clean-commit 结果

正式 run id 为 `pcb3038-ht-scan-20260901T214535Z-2603074`，clean commit 为 `f12c181`，物理 GPU 1（RTX 4000 Ada）。输入、目标切片和预算与前一公平基线相同。

四路规范工作量完全一致：

| 指标 | 数值 |
|---|---:|
| attempted / proven / unresolved / committed | 8 / 2 / 6 / 2 |
| states / replies expanded / leaf calls | 12,383 / 14,285 / 9,120 |
| Hamilton reply batches | 72 |
| 逻辑 centers | 27,598 |
| batch 内 unique centers 累计 | 1,395 |
| 实际测试邻边对 | 11,515 |
| 输出 replies | 245,965 |
| leaf cost cells（全 CPU 认证） | 51,309,996 |

最终图哈希仍为 `fe11f98414b04c0e`，边文件 SHA-256 仍为 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`。成本 137,694 的 pcb3038 最优 tour 为 0 缺边、哈希 `ca0238497c090a3c`；四份 proof 均由独立 CPU verifier 重放成功。

## 5. 性能结果

| 模式 | Hamilton reply ms | leaf ms | search ms | 相对 CPU search |
|---|---:|---:|---:|---:|
| CPU matrix | 18.914 | 4,449.941 | 4,869.906 | 1.000× |
| all CUDA | 78.604 | 4,880.205 | 5,562.180 | 0.876× |
| hybrid leaf CUDA | 18.614 | 4,926.090 | 5,344.660 | 0.911× |
| hybrid + bucket fusion | 19.157 | 4,934.452 | 5,354.220 | 0.910× |

CPU reply 子阶段：

| 子阶段 | ms | reply 占比 |
|---|---:|---:|
| graph/target/center validation | 16.175 | 85.52% |
| CPU 去重枚举与输出拼接 | 2.640 | 13.96% |
| residual | 0.100 | 0.53% |

CUDA reply 的 78.604 ms 中，CPU validation/enumerate 为 `15.867/2.754 ms`，设备 evaluate 为 `57.867 ms`，全数组比较为 `0.493 ms`。在当前任务规模与同步完整 CPU 认证下，CUDA reply 约为 CPU reply 总时间的 `4.16×`，继续保留为正确性差分路径而非默认性能路径。

优化同时让 c,d candidate 阶段从前一 clean run 的 `51.658 ms` 降为 `4.125 ms`，因为其 reply-product 计算也复用了内部单 center 快速枚举器。

## 6. 新瓶颈与下一步

CPU search 的阶段占比现在约为：

| 阶段 | ms | search 占比 |
|---|---:|---:|
| leaf | 4,449.941 | 91.38% |
| path append | 190.643 | 3.92% |
| host build residual | 188.976 | 3.88% |
| Hamilton reply | 18.914 | 0.39% |
| end reply | 10.791 | 0.22% |

Hamilton validation 即使全部消除也只能再节省约 16 ms，不应继续优先。研究焦点返回 leaf：setup `1,757.315 ms` 与 CPU matrix evaluate `2,320.345 ms` 合计占 leaf 的约 `91.64%`。

下一切片优先量化 CPU matrix 的 task/row 并行性与 setup 中 path normalization/catalog/cursor 构造的比例。任何线程化必须保持每个 cursor 的规范消费顺序、首次 witness 和 proof 字节不变；并行只允许发生在互不依赖的 cost cells/tasks 生成阶段。

该 setup 细分及首轮优化已完成，结果见 [M5 HT leaf setup 画像与快照哈希复用](34_M5_HT_Leaf_Setup_Snapshot_Hash.md)。
