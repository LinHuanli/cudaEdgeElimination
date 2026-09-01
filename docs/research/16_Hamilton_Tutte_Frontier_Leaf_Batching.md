# Hamilton–Tutte frontier leaf batching

## 范围

M4.3b3b2b2b2a 把 wavefront 的 path-system leaf 从“按状态同步调用”改为“先按复杂度分桶，再跨状态融合首个 k-opt cost row”。本文件记录首期 `max_deletion_sets=1` 基线；M4.3b3b2b2b2b1 已在相同接口下扩展为一般预算的[增量 leaf 游标](17_Hamilton_Tutte_Incremental_Leaf_Cursors.md)。

公开入口 `ProvePathSystemsByKOpt` 接收多个规范路径系统，返回与输入一一对应的完整 `PathSystemKOptProof`，并记录实际 cost batches/tasks/cells 与 scalar searches。它不改变单状态 `ProvePathSystemByKOpt` 的接口或证书格式。

## 确定性复杂度桶

wavefront 为当前 reply chunk 的每个状态建立键：

```text
(depth, path_count, node_count, max_k, floor_log2(incoming_reply_count))
```

桶使用有序 map，桶内保持原 state index 顺序。`--leaf-frontier-batch-states N` 只限制一个桶内每次调用的状态数：`0` 表示完整桶，`1` 是无跨状态融合基线，默认 `256`。该选项不裁剪状态、outside matching、删除集合或 reconnect template。

leaf 计算完成后仍按原 state index 回填 proof，候选 move 和 child 的生成顺序完全不变。分桶次序只改变互相独立的 cost 计算顺序，不参与全局 state/reply 预算消费。

## 跨 leaf cost 融合

对每个尚未完成的 path-system proof，调度器保持独立的 `covered[outside]`。每一轮：

1. 按原规范顺序选择每个 proof 的第一个未覆盖 outside matching；
2. 首期在 `max_deletion_sets=1` 输入域构造该搜索会访问的首个删除集合及 `KOptCostTask`；
3. 按相同 `k` 展平所有 rows，一次计算完整 proper template 矩阵；
4. 把每行结果映射回原 path/outside；
5. 继续原 `FindKOptWitness` 的 template 顺序、CPU `TryReconnect`、inside coverage 和 exact fallback；
6. 已覆盖全部 outside 的 proof 完成，其余 proof 进入下一轮。

对于 `k=3`，每个 row 固定有 4 个 proper templates。融合只合并 kernel 输入，不改变每个 proof 统计的 deletion sets、reconnect cells、witness 选择或 records 顺序。

## GPU 候选与 CPU 认证

CUDA cost matrix 仍不是删除授权。任何低于 deleted cost 的 cell 都必须由 CPU 重建完整 tour、检查现存/重复边、重新计算整数成本、提取 inside matching，并调用独立 witness verifier。每个 CUDA row 在候选扫描后还运行 CPU 全模板 completeness fallback，因此 GPU 漏报只增加工作，不会把未穷尽搜索解释成 leaf 失败。

批量 API 对每个成功 proof 再调用 `VerifyPathSystemKOptProof`，全部通过才设置 `cpu_verified=true`。显式 CUDA 批次失败保留该 outside 为 unresolved；`auto` 批次失败时一次转为 CPU matrix。unproven leaf 不能独立授权，只会让 HT 继续递归或最终保留边。

## 指标与回归

`HtWavefrontResult` 新增：

- `leaf_frontier_batches/states`、`leaf_bucket_count`、`peak_leaf_frontier_batch`；
- `leaf_cost_backend/device` 与 `leaf_cpu_verified`；
- `leaf_cost_batches/tasks/cells` 和 `leaf_scalar_searches`。

三份相同的固定 5 点 leaf 在一个 batch 中产生 3 tasks / 12 cells；三份输出与 scalar proof 的 V1 序列化逐字节相同，CPU-only 与 CUDA build 均覆盖。

固定 recursive-point 全 CUDA wavefront 的结果为：

| 指标 | `N=1` | `N=256` |
|---|---:|---:|
| leaf frontier/cost batches | 34 | 6 |
| leaf states / cost tasks | 34 / 34 | 34 / 34 |
| cost cells | 136 | 136 |
| complexity buckets | 6 | 6 |
| peak leaf batch | 1 | 16 |

两种配置保持同一 34-state/18-move/84-reply 工作图，序列化 4 节点 V1 proof 逐字节相同并通过 `ht-verify`。这些数字证明 launch 融合和映射等价，不代表中大型实例加速比。

## 当前限制与后续

M4.3b3b2b2b2b1 已解除单 deletion-set 限制；`leaf_scalar_searches` 现在主要记录显式 CPU backend 或不能建立 cost cursor 的输入。exact Held–Karp 始终在 CPU。

后续已加入[设备快照/模板与 workspace 驻留缓存](18_Hamilton_Tutte_GPU_Leaf_Cache.md)；路径规范对象、outside/inside 枚举及 witness 构造仍在主机。接下来加入明确的 GPU/CPU long-tail 阈值和多 block continuation。HT epoch commit 继续保持待实现。
