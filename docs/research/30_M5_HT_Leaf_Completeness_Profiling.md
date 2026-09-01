# M5 HT leaf CPU completeness 画像

## 1. 研究问题

[不可变组合表缓存](29_M5_HT_Immutable_Leaf_Tables.md)将 pcb3038 8-target CPU search 缩短约 29%，但混合 leaf 中仍有约 15.1 s 位于 `KOptSearchCursor::ConsumeBlock`。本切片区分：

- GPU cost row 中严格低于 deleted cost 的候选及其 CPU `TryReconnect`；
- 没有 CPU 接受候选时，对整行 proper templates 执行的 CPU completeness fallback；
- cursor 计数与状态更新等 residual。

目标是定位重复计算，不改变“GPU 只是候选器、CPU 保证 completeness”的默认契约。

## 2. 实现与门禁

提交 `ee4f3aa` 为每个 cursor consume 累计：

- `leaf_cost_rows_consumed`；
- `leaf_candidate_templates_rechecked`；
- `leaf_cpu_completeness_rows/templates`；
- `leaf_candidate_recheck_ms`；
- `leaf_completeness_fallback_ms`。

它们只用于诊断，不进入 leaf/HT proof。`CUDAEE_HT_SCAN_REPORT_V5` 保存全局和逐 target 指标，V6 benchmark summary 另计算：

```text
consume_residual = cursor_consume - candidate_recheck - completeness_fallback
```

all-CUDA、hybrid 与 fused 三路必须拥有完全相同的四个规范工作计数，否则 benchmark 失败。CPU Debug/Release 各 17/17、GPU 1 上 CUDA Release 20/20 和 compute-sanitizer 0 errors 均通过。

## 3. pcb3038 clean-commit 结果

正式 run id 为 `pcb3038-ht-scan-20260901T205936Z-2577686`，clean commit 为 `ee4f3aa`，物理 GPU 1。四路仍保持 12,383 states、14,285 replies、9,120 leaf calls、2 PROVEN、6 UNRESOLVED，提交相同两条边。最终图哈希 `fe11f98414b04c0e`、活动边 SHA-256 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810` 和受保护最优 tour 均不变。

三条 CUDA leaf 路径的规范 consume 工作量完全一致：

| 指标 | 数值 | 占比 |
|---|---:|---:|
| cost rows consumed | 727,635 | 100% |
| 低成本候选 CPU 重建 | 987 | 每个非 fallback row 恰好 1 个 |
| CPU completeness rows | 726,648 | rows 的 99.864% |
| CUDA cost cells | 51,309,996 | 100% |
| CPU completeness templates | 51,179,094 | cells 的 99.745% |

| hybrid consume 子阶段 | ms | consume 占比 |
|---|---:|---:|
| candidate recheck | 56.554 | 0.371% |
| CPU completeness fallback | 15,107.389 | 99.033% |
| residual | 91.011 | 0.597% |
| cursor consume 总计 | 15,254.955 | 100% |

987 个 GPU 低成本候选全部被 CPU 接受并直接形成 witness；其余 726,648 行进入完整 fallback。fallback 不是罕见的错误保护，而是稳态主路径。

绝对 search 时间会受新增逐行计时开销与共享节点抖动影响，本切片只用比例定位热点。输出等价门禁证明计数插桩没有改变搜索结果。

## 4. 代码层原因

`EvaluateKOptTemplateCosts` 已计算 `[task][template]` 精确整数成本矩阵；但 CUDA 输出按现有安全契约只作为候选 oracle。若一行没有可接受候选，`TryReconnectFromCostRow` 再调用通用 `TryReconnect`：

1. 重建删除后 component matching 和多个 `std::set`；
2. 重新计算 deleted cost；
3. 对全部 proper templates 重建 added edge set并重新计算距离；
4. 仅在严格改善时才构建 cycle、inside matching 和 witness。

绝大多数模板在第 3 步因成本不改善而退出。因此 15 秒主要不是 proof verifier，也不是成功 witness 重建，而是用通用拓扑路径重复认证“该模板成本不改善”。

## 5. 下一实现切片

实现 CPU 精确矩阵认证快路径，同时保持默认 completeness：

1. 每个 task 只计算一次端口对距离和 deleted edge 集；
2. 用固定大小数组而非逐 cell `std::set` 生成 CPU `[task][template]` 精确成本；
3. CUDA 路径逐 cell 与 CPU 矩阵比较；任何不一致显式 CUDA 失败关闭，`auto` 可使用已生成的 CPU 矩阵；
4. 由已通过 CPU 认证的行确定严格改善模板，只对这些模板调用完整 `TryReconnect` 和 witness verifier；
5. CPU backend 也可复用同一批量精确矩阵，但在默认切换前必须与 scalar proof 字节、计数及性能做门禁。

该路径不会用 GPU 的“无命中”直接证明任何事实。completeness 来自 CPU 对每个 cell 的独立精确计算，成功 witness 仍由既有通用 CPU verifier 重建；因此优化目标是更换 CPU 完整认证的数据布局，而不是放宽证明边界。
