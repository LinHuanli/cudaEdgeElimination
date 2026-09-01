# M5 HT point-candidate 静态次序缓存

## 1. 等价性边界

`BuildPointCandidateNodes` 的原选择键为 `(midpoint_score,node)`。其中中点评分、target 端点排除、候选度数门禁及 node tie-break 只依赖不可变的 `(graph,target,options)`；逐 frontier state 唯一变化的是 `ContainsNode(state,node)`。

提交 `ea85ffa` 因此在每次 `ProveEdgeByWavefrontHt` 内：

1. 首次 point selection 对静态 eligible 节点计算一次评分并建立严格全序；
2. 用按维度分配的 generation marks 标记当前 state 的路径节点，避免逐 state 清零；
3. 沿静态次序跳过已标记节点，输出前 `max_neighborhood` 项；无界分支仍输出全部剩余节点；
4. `nodes_checked` 继续记完整维度，`nodes_ranked` 用静态 eligible 总数减去 state 中唯一 eligible 节点，`nodes_selected` 仍是实际规范前缀。

因此缓存没有缩减候选集合、改变预算或改变计数定义。严格全序过滤后的前缀与“先过滤、后排序、再裁剪”逐项相同。

## 2. clean-commit 三实例结果

正式 runs 均绑定 clean commit `ea85ffa189e5c73eb0d81c47d55d818ed3a9c8b9`、物理 GPU 1、8 个 CPU cost threads 和锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T233338Z-2676126`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T233352Z-2676761`；
- d15112：`artifacts/d15112-ht-scan-20260901T233402Z-2677328`。

相对前一 clean Top-K commit `4e4f8e3`，CPU fused 单变量结果如下：

| 实例 | scan：Top-K → cache（ms） | scan 加速 | order/select：Top-K → cache（ms） | order/select 加速 | host residual 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pcb3038 | 40.342 → 0.214 | 188.364× | 11.901 → 1.188 | 10.016× | 4.616× | 1.045× | 1.041× |
| rl5915 | 70.826 → 0.297 | 238.242× | 14.314 → 2.214 | 6.465× | 7.060× | 1.176× | 1.155× |
| d15112 | 165.022 → 0.563 | 293.326× | 28.784 → 4.774 | 6.029× | 4.368× | 1.131× | 1.106× |

相对最初全量排序画像 commit `5944476`，两步优化的累计 CPU-fused search 加速为 `1.135×/1.858×/1.756×`，wall 加速为 `1.131×/1.757×/1.617×`；host residual 累计加速 `13.815×/27.935×/19.241×`。

## 3. 正确性与资源门禁

- CPU Debug/Release 为 17/17，CUDA Release 为 20/20；
- CUDA Debug HT 单元在 compute-sanitizer 下为 0 error；
- 三实例五路的 scans、checked、ranked、selected 与前一 commit 完全相同；
- 三实例五路活动边、工作签名和去除计时行后的规范 proof 逐字节一致；
- JV 固定点、V2 重放、最终活动边和受保护最优 tour 全部通过。

缓存的额外内存是每 target 一份 `ordered_nodes`、eligible byte map 与 32-bit generation marks，均为 `O(n)`；在 `d15112` 规模仍远低于 leaf/device 工作区。缓存只活到单个 wavefront 调用结束，不跨 graph snapshot 复用。

## 4. 后续决策

优化后的 point scan + order/select 只占 host residual 的 `10.123%/18.405%/9.545%`，host residual 本身只占 CPU-fused search 的 `1.031%/3.219%/4.116%`。继续微调该路径收益上限很低。

下一阶段回到整体工作图：pcb3038/rl5915/d15112 的 CPU-fused leaf 占 search 约 `92.989%/63.457%/54.858%`，大实例还保留 path-append、Hamilton reply 与 proof verify 成本。优先评估跨目标共享不可变 leaf 准备数据与调度批次；任何跨目标融合仍必须按 target 独立保留完整 CPU 认证、proof 与原子提交边界。
