# M5 HT leaf 不可变组合表缓存

## 1. 动机

[leaf 子阶段画像](28_M5_HT_Leaf_Subphase_Profiling.md)显示，pcb3038 8-target 的混合 leaf 有 5.864 s 位于 setup、19.695 s 位于 cursor consume，而 GPU cost 只有 0.584 s。代码审计进一步发现，两类仅由小整数参数决定的组合结构在 9,120 个 leaf calls 中被反复生成：

- `path_count=1..7` 的 outside/inside matching 规范枚举，以及 `path_count<=5` 的兼容 bit table；
- `k=3/4/5` 的 proper reconnect templates 及生成器哈希。

这些结构不依赖图、required edge、路径节点、搜索预算或后端，重复构造既不增加验证强度，也不改变候选集合。

## 2. 实现

提交 `589bca0` 增加两个 translation-unit 内部缓存：

1. `CachedPathMatchingCatalog(path_count)` 保存 outside、inside 和可选兼容表；
2. `CachedKOptReconnectTable(k)` 保存 proper reconnect templates。

每个合法参数在首次使用时构造一次。函数内 `static const` 由 C++20 保证线程安全延迟初始化；调用方只获得 `const` 引用。批量 proof work 只保存指向静态 catalog 的只读指针，不再复制完整向量和 bit table。

公开的 `BuildKOptReconnectTable` 与 `BuildPathCompatibilityTable` API 保持原样，仍可构造独立值。proof 中的 path-system hash、兼容表 hash、outside source index、template 顺序、测试计数及 witness 均未改变。CUDA 仍只是 cost 候选器；无命中时的 CPU 全模板 completeness fallback 完整保留。

## 3. 正确性门禁

- CPU Debug 17/17；
- CPU Release 17/17；
- GPU 1 上 CUDA Release 20/20，包含 pr299 CPU/CUDA 差分；
- compute-sanitizer memcheck 0 errors；
- 1-target 和正式 8-target 四路 benchmark 均通过工作签名、独立 proof 重放、最终边文件及受保护最优巡回门禁。

静态缓存初始化失败会抛出异常，不会返回部分表；显式 CUDA leaf 因异常保持 fail-closed。缓存不保存图相关数据，因此不存在跨 snapshot 复用陈旧坐标、边或 required edge 的路径。

## 4. clean-commit 8-target 结果

正式 run id 为 `pcb3038-ht-scan-20260901T204639Z-2570897`，clean commit 为 `589bca0`，物理 GPU 1。基线为同节点、同协议的 `fe99f38` run `pcb3038-ht-scan-20260901T203636Z-2565629`。

优化前后四路都保持：12,383 states、14,285 replies、9,120 leaf calls、51,309,996 个 CUDA cost cells，2 PROVEN、6 UNRESOLVED，并提交相同两条边。最终图哈希 `fe11f98414b04c0e`、活动边 SHA-256 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810` 与最优 tour hash `ca0238497c090a3c` 均不变。

| 路径 | 优化前 leaf ms | 优化后 leaf ms | leaf 加速 | 优化前 search ms | 优化后 search ms | search 加速 |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 27,231.421 | 17,760.428 | 1.533× | 34,094.769 | 24,137.501 | 1.413× |
| 全 CUDA | 26,354.767 | 17,421.598 | 1.513× | 33,487.773 | 24,184.402 | 1.385× |
| 混合 | 26,489.332 | 17,684.036 | 1.498× | 33,334.581 | 24,208.211 | 1.377× |
| 混合 + 桶融合 | 26,318.716 | 17,487.774 | 1.505× | 33,184.469 | 24,023.297 | 1.381× |

关键子阶段变化：

| 子阶段 | 优化前 ms | 优化后 ms | 降幅 |
|---|---:|---:|---:|
| CPU setup | 5,680.111 | 1,624.651 | 71.398% |
| CPU scalar search | 21,474.526 | 16,073.739 | 25.150% |
| 混合 setup | 5,864.498 | 1,759.227 | 70.002% |
| 混合 cost evaluate | 583.746 | 509.530 | 12.714% |
| 混合 cursor consume | 19,695.247 | 15,110.667 | 23.278% |

matching catalog 复用主要消除了 setup；reconnect template 复用同时降低 scalar、cost 和 consume。优化后的 CPU search 比原 CPU 基线减少 9.957 s，进程 wall 减少 9.962 s。

## 5. 性能结论与下一步

缓存使所有后端都受益，CPU 受益略大。优化后全 CUDA、混合和桶融合相对 CPU search 分别为 `0.998×`、`0.997×` 和 `1.005×`，不足以声称 GPU 加速。当前混合 leaf 的 cursor consume 为 15.111 s，占 leaf 的约 `85.45%`；setup 已降到约 `9.95%`，GPU cost 约 `2.88%`。

下一步应把 cursor consume 拆为：

1. cost row 中低于删除成本的 GPU 候选扫描；
2. 候选的 CPU `TryReconnect` 与 witness 复核；
3. CUDA 无可接受候选后的 CPU 全模板 completeness fallback；
4. cursor 状态与计数更新。

只有量化 CPU completeness 后，才决定是优化 CPU 精确 cost/reconnect 数据布局，还是设计显式、默认关闭且“漏报只能 UNRESOLVED”的候选策略。默认 CPU 全模板兜底在新安全论证完成前保持不变。
