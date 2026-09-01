# Hamilton–Tutte GPU path append

## 范围与安全边界

M4.3b3b1 迁移递归 point/end reply 中最规则的一步：判断把一条或两条边加入规范路径系统后，是否立刻产生度数冲突、重复边或提前成环。CUDA 只返回候选 `feasible[]`；CPU 对同一批每项运行完整 `NormalizePathSystem`，逐字节比较 flags，并且工作图只接收 CPU 生成的规范 child。GPU 结果因此不会直接进入证明或授权删边。

公开入口为：

```cpp
EvaluateHtPathAppends(dimension, parents, tasks, backend)
```

`backend=auto` 在 CUDA 不可用或运行时异常时返回 `cpu-fallback`；显式 `cuda` 不可用会失败；任何 CUDA/CPU flag 不一致都会抛出内部逻辑错误。父状态必须重新规范化后与输入完全一致，point 中心必须尚未出现，end 起点必须是现有路径端点。未知枚举、越界节点和伪造状态均在 launch 前拒绝。

## 等价判定

规范路径系统是若干节点不交简单链。CUDA 为每个出现的节点保存：

```text
(node, component, degree)
```

未出现节点的度数视为 0、分量视为 absent。对 `point(c)` reply `x-c-y`，搜索器保证 `c` 是新节点且三点互异。加入两条边合法当且仅当：

- `degree(x) < 2` 且 `degree(y) < 2`；
- `x,y` 不同时属于同一个已有分量。

第一项排除度数超过 2；第二项排除用新中心闭合一条已有链。若一个或两个连接点尚未出现，则不会闭合已有分量。

对 `end(p,q)` reply，`p` 必须是已有链端点。加入边合法当且仅当：

- `degree(p) == 1` 且 `degree(q) < 2`；
- `q` 未出现，或 `q` 属于另一个分量。

同分量另一端会成环；原内部邻居会触发重复边或度数冲突；其他内部节点会触发度数冲突。因此该常数条件与完整规范化器在合法 HT 输入域上的结果相同，最终仍由 CPU 全量差分确认。

## 批处理与预算

`HtPathStateSpan[]` 指向连续 node records，`HtPathAppendTask[]` 指向父状态并携带 point/end 参数。kernel 一个线程扫描一个父状态的稀疏 records，无需分配 `dimension × states` 的稠密度数表。

wavefront 在每个未解决父状态上：

1. 按既有确定性顺序生成候选，并在 launch 前检查单 move、总 reply 与硬上限；
2. 把所有可预算 point 候选合为一个 batch；只有 point 未形成全 infeasible 短路时，才把 end 候选合为第二个 batch；
3. 按原候选/reply 顺序使用 CPU children 建图，保持 proof 和 DFS 语义不变；
4. 已预判但因较早 OR move 成功而未物化的 task 只计入 `path_append_tasks`，不计入 `replies_expanded`。

结果记录实际 backend、device、batch/task 数和 `cpu_verified`。这些计数用于后续决定最小 GPU bucket，不是证明字段。

## 回归与当前结果

- 固定两个父状态的 11-task 表期望 flags 为 `[1,0,0,1,1,0,0,1,1,0,1]`，覆盖不同分量合并、同分量成环、内部节点度数冲突、重复边和新节点；
- CPU-only build 验证 `auto` 回退与显式 CUDA 失败；CUDA build 对全部 flags 与规范 children 做逐项差分；
- 固定递归 point 实例使用 CUDA c,d、path append 和 continuation 三段候选器，得到 34 states、18 moves、84 replies、9 append batches、84 append tasks，最终 V1 arena 为 4 个节点并通过独立 `ht-verify`；
- 包含 path-append 与 persistent continuation kernel 的完整 Hamilton–Tutte 单测经 compute-sanitizer memcheck 为 0 error；
- 单 move/总 reply 预算在 batch 前限制任务数，状态预算在物化 CPU child 时再次检查；预算不足、设备异常和内存不足都只会保留目标边。

## 后续

当前 node records 和规范 children 仍由 CPU 构造，每个父状态最多发起 point/end 两个 batch。M4.3b3b2 将先按 `(depth,path_count,reply bucket)` 合并整个 frontier，再在 GPU 上完成 reply 计数、prefix-sum 写出和规范 child SoA；小桶、超大 reply、深层状态与精确 DP 进入 CPU long-tail。只有完整小实例差分、显存峰值和端到端收益均过门禁后，HT 证明才会接入不可变 epoch commit。
