# Hamilton–Tutte GPU 规范 child edge SoA

## 范围与权限边界

M4.3b3b2b2b1 把 path-append 从“GPU 只返回可行 flags”扩展为“GPU 同时写出每个可行 child 的完整规范边集”。公开结果使用 append-only CSR：

```text
child_edge_offsets[task_count + 1]
child_edges[child_edge_offsets[i] : child_edge_offsets[i + 1]]
```

不可行 task 的 slice 必须为空；可行 slice 包含父路径森林的全部边和本次 point/end 新增边，并按 `(u,v)` 字典序严格递增。无向边始终规范成 `u < v`。对于不交简单链森林，完整边集唯一确定规范路径系统，因此该 SoA 可作为后续 GPU leaf batch 的无指针输入。

这仍是候选输出。CPU 对每个 task 独立调用 `NormalizePathSystem`，真正用于工作图的 `children` 始终来自 CPU。只有 flags、全部 offsets 和全部边逐项相等时，结果才设置 `device_children_verified=true`；任何差异都 fail-closed，不会形成证明或删边授权。

## 父状态输入

每个父状态同时携带两个连续区间：

- 稀疏 `(node,component,degree)` records，用于常数规则判断 degree 冲突和同分量成环；
- 严格排序的规范 `NodeEdge` records，用于 child 写出。

launch 前检查 node/edge spans 连续覆盖输入数组、节点唯一、边合法且严格排序、边的两个端点属于同一分量，以及边计算出的度数与 node record 一致。point 中心必须尚未出现；end 起点必须是度为 1 的已有端点。

## Count/write 流程

第一阶段一个线程处理一个 task，输出可行 flag 和边数：

```text
infeasible: 0
point:      parent_edge_count + 2
end:        parent_edge_count + 1
```

主机以 `uint64_t` 做完整前缀和并检查地址空间溢出，随后一次分配精确长度的设备输出。第二阶段每个线程只写自己的 slice：复制父边、加入一条或两条新边、在 slice 内作确定性插入排序，并检查重复边。kernel 同时验证实际写入量与 CSR 区间一致；写出错误通过设备错误位返回并拒绝整个 batch。整个过程不使用设备动态分配，也不存在跨 task 写竞争。

当前插入排序和“每个 child 复制完整父边”是正确性基线，复杂度约为每 task `O(E_parent^2)` 排序工作和 `O(E_parent)` 输出空间。后续 leaf kernel 可直接消费这些 slices；在大状态上是否改用分桶并行 merge 或转入 CPU long-tail，必须由基准决定。

## CPU 独立认证

CPU 不复用 GPU 的 count 规则生成期望值，而是先把父路径与 reply 组成原始路径集合，再运行完整规范化器。仅对规范化成功的 child 遍历其规范路径、生成无向边并独立排序；失败 child 生成空区间。CUDA 返回后比较：

1. `feasible[task]`；
2. `child_edge_offsets[0..task_count]`；
3. `child_edges[0..offsets.back())`。

公开 API 在 CPU、CUDA 和 auto fallback 下都返回同一 CPU 认证的 SoA 语义。`HtWavefrontResult` 额外记录 `path_append_child_edges` 和 `path_append_device_children_verified`，前者是实际批次中全部可行候选边记录数，不属于证明字段。

## 回归结果

双父状态 11-task 表继续得到 flags `[1,0,0,1,1,0,0,1,1,0,1]`，其 offsets 固定为 `[0,5,5,5,10,14,14,14,18,23,23,27]`，共 27 条规范 child 边。测试从 CPU children 独立重建期望边集，并要求 CUDA 全数组相等。

固定 recursive-point 全 CUDA wavefront 产生 3 个 path-append batches、84 个 tasks 和 172 条 child edge records；`path_append_device_children_verified=1`。工作图仍为 34 states、18 moves、84 replies，最终 4 节点 V1 proof 通过独立 `ht-verify`。这些是确定性与正确性证据，不构成性能结论。

## 后续

M4.3b3b2b2b2a 已把 frontier leaf 按 `(depth,path_count,node_count,k_max,reply bucket)` 分桶，并跨 leaf 融合规则 k-opt cost rows；后续一般游标、设备驻留缓存、GPU/CPU long-tail、cooperative multi-block continuation 和 [HT epoch commit](21_Hamilton_Tutte_Epoch_Commit.md)也已完成。CPU 仍对每个成功 leaf、最终 proof 与提交 sidecar 完整复核。
