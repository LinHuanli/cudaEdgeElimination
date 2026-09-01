# Hamilton–Tutte frontier path append

## 范围

M4.3b3b2b2a 将 path-append 从每个父状态一次 point/end 调用改为每个 frontier chunk 最多两次调用。它复用 `EvaluateHtPathAppends` 已支持的多父状态输入：规范父路径分别展平为稀疏 `(node,component,degree)` spans，每个 task 用 `parent_index` 指向自己的 span。

M4.3b3b2b2b1 已在同一接口上加入设备端规范 child edge SoA：GPU 除冲突 flags 外还输出每个可行 task 的完整规范边 slice；CPU 为每个展平 task 独立运行 `NormalizePathSystem` 并比较 flags、offsets 和全部边。真正用于建图的 child 仍由 CPU 保存，跨父状态合批不扩大 GPU 的证明权限。

## 两阶段流水线

每个 reply frontier chunk 执行：

1. 批量生成并按规范顺序恢复所有 point candidates；
2. 展平所有未超过单 move reply 上限的 point tasks，记录每个 candidate 的 `append_begin`；
3. 一次执行 point path-append batch；
4. 若某个可尝试 point candidate 的全部 flags 都是 infeasible，则该父状态必有 vacuous-success point move，不生成 end replies；
5. 对其余父状态批量生成 end replies，再一次执行 end path-append batch；
6. 最后按原父状态、candidate、reply 顺序消费全局预算并物化 CPU children。

步骤 4 不会越过资源预算错误地选择 shortcut：若全局 reply/state 预算在到达该 candidate 前耗尽，原流程会直接返回 `unresolved`，本流程同样不会进入 end move；若预算允许到达，则全 infeasible candidate 按定义成功。超过单 move reply 上限的 candidate 不参与 shortcut 判定。

## 索引与资源边界

point/end task 分别形成一个展平数组；candidate 保存数组起点，reply 的 CPU child 位于 `children[append_begin+i]`。任务的 `parent_index` 是 chunk 内局部索引，不是全局 state index；chunk 状态数在转换到 `uint32_t` 前检查。

候选生成可以早于最终全局预算消费，因此 batch 可能包含后来因总 reply/state 预算而未物化的 tasks。这只增加候选计算，不改变证明；所有硬预算仍在原确定性顺序中执行。设备或 CPU/GPU 差分在较晚父状态失败会让整个 chunk fail-closed。

## 回归与当前结果

固定 recursive-point 样例在 `N=1` 与 `N=256` 下保持同一 34-state/18-move/84-reply 工作图，序列化 V1 proof 逐字节相同并通过独立重放：

| 指标 | `N=1` | `N=256` |
|---|---:|---:|
| path-append batches | 9 | 3 |
| path-append tasks | 84 | 84 |
| end reply batches | 2 | 1 |
| end tasks / edges | 8 / 48 | 8 / 48 |
| peak parents/chunk | 1 | 4 |

全 CUDA CLI 同样得到 3 个 path-append batches，全部 84 flags 由 CPU 规范化认证；Hamilton/end/continuation 三段 CUDA 后端共同产生的 4 节点 proof 通过 `ht-verify`。recursive-end 固定实例继续要求真实 end move 在全 CUDA 模式成功。

这些数字只证明跨父状态合批减少 launch 且没有投机 end 膨胀，不构成性能结论。图和父状态 records 仍在每个调用中重新分配、复制。

## 后续

设备端 append-only [规范 child edge SoA](15_Hamilton_Tutte_GPU_Child_Edge_SoA.md)、后续 leaf batching/游标、GPU 驻留缓存、CPU long-tail 与 [cooperative multi-block continuation](20_Hamilton_Tutte_Multi_Block_Continuation.md)均已完成，并通过独立 V1 proof 等价门禁。HT epoch commit 仍待实现。
