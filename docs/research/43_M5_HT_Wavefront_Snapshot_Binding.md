# M5 HT wavefront leaf 快照绑定复用

## 1. 目标与安全边界

批内 proof 绑定复用消除了同一 k-opt batch 内成功 witness 的重复整图哈希，但一个 HT wavefront 会发起多个 leaf batch；此前每个 batch 仍在初始化时重新执行 `graph.ContentHash()`。这些 batch 共享 `ProveEdgeByWavefrontHt` 入口处的同一只读 `GraphSnapshot`，因此哈希扫描仍是纯重复工作。

提交 `00c0156` 引入内部强类型 `KOptSnapshotBinding`：

- binding 只能由实际 `GraphSnapshot` 构造，同时保存对象身份和当时计算的内容哈希；
- wavefront 入口只构造一次 binding，所有同步 leaf batch 复用其哈希；
- bound leaf API 会检查 graph 对象身份，不接受由另一个快照对象构造的 binding；
- 公开 `ProvePathSystemsByKOpt`、公开 proof verifier、HT 最终 verifier、scan 即时复核和 epoch 独立重放仍各自重新计算图哈希；
- 单元测试逐项比较 bound/public API 的 proof 与工作计数，并验证内容相同但对象不同的 graph copy 会被拒绝。

该优化只复用一次调用内的不可变输入绑定，不缓存搜索结论，也没有扩大任何删除授权边界。

## 2. clean-commit 三实例结果

正式 runs 均绑定 clean commit `00c0156`、物理 GPU 1、8 个 CPU cost threads 和锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T235035Z-2690984`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T235050Z-2691614`；
- d15112：`artifacts/d15112-ht-scan-20260901T235059Z-2692184`。

相对前一 clean commit `0530ff4`，CPU-fused 单变量结果如下：

| 实例 | proof init：基线 → binding（ms） | init 加速 | leaf 加速 | work graph 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 26.605 → 11.016 | 2.415× | 0.998× | 0.997× | 0.998× | 1.001× |
| rl5915 | 8.678 → 0.196 | 44.302× | 1.088× | 1.069× | 1.063× | 1.059× |
| d15112 | 29.163 → 0.292 | 99.936× | 1.112× | 1.034× | 1.028× | 1.019× |

pcb3038 的 51,309,996 个 cost cells 仍主导 leaf，约 15.6 ms 的 init 降幅被整段波动覆盖，因此不能宣称端到端加速。rl5915/d15112 的 init 几乎降至计时噪声，并分别获得 `1.063×/1.028×` search 加速。

## 3. 正确性与资源门禁

- CPU Debug/Release 为 17/17，CUDA Release 为 20/20；
- CUDA Debug HT 单元在 compute-sanitizer 下为 0 error；
- 三实例五路的活动边、工作签名和去除 `metrics` 行后的规范 proof 逐字节一致；
- JV 固定点、V2 重放、最终活动边与三份受保护最优 tour 全部通过；
- 新 binding 不拥有 graph，不跨 wavefront、线程或 epoch 保存，也不进入公开序列化格式。

## 4. 新画像与下一切片

绑定后的 CPU-fused work graph 分解如下：

| 实例 | leaf（ms） | path append（ms） | Hamilton reply（ms） | end reply（ms） | host residual（ms） |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 1202.741 | 38.682 | 19.306 | 10.962 | 13.896 |
| rl5915 | 141.797 | 61.593 | 37.262 | 11.688 | 13.433 |
| d15112 | 210.090 | 202.646 | 146.762 | 66.917 | 55.439 |

d15112 的 path append 已成为最大单项，其中 child normalize 为 `180.448 ms`，占 path append 的 `89.04%`。下一单变量实验直接从已规范化 parent 增量合并新增的一条或两条边，避免每个 task 重建整份 map/set 邻接；generic sparse 和独立 dense normalizer 继续作为差分 oracle。只有 valid、reason、edge_count、规范 paths 和 child edges 全部逐项一致，才允许替换生产 fast path。
