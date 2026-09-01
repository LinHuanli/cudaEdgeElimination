# Hamilton–Tutte frontier reply batching

## 目标与非目标

M4.3b3b2b1 将 reply 生成从“每个父状态一次调用”改为“每个 frontier chunk 一次调用”。它同时展平 point centers 和 end endpoint tasks，复用既有 Hamilton/end count-write API；CPU 仍完整枚举并认证每一个区间。

在 M4.3b3b2b1 交付时，本阶段不改 leaf engine、不把规范 child 写到设备，也不跨父状态合并 path-append。目标是先证明跨状态索引、回填与资源切块不会改变工作图或证书，再复用同一映射扩展后续 kernel；后续 path-append 扩展见 [frontier path append](14_Hamilton_Tutte_Frontier_Path_Append.md)。

## chunk 调度

`HtWavefrontOptions::reply_frontier_batch_states` 和 CLI `--reply-frontier-batch-states` 控制一个 chunk 最多包含多少个父状态：

- 默认 `256`；
- `1` 是逐父状态差分基线；
- `0` 表示把当前完整 frontier 放入一个 chunk；
- 该参数只切分工作，不跳过状态、候选或 replies，因此不是搜索预算。

每个 chunk 先完成其中所有 leaf calls。对未被 leaf 证明且深度未达上限的状态，按原状态顺序记录 point-node span 与 end-task span，再分别发起一次展平 batch。返回后按 span 重建每个父状态的候选列表，并使用原排序键：point 为 `(reply_count,node)`，end 为 `(reply_count,endpoint,internal_neighbor)`。

## 确定性回填

GPU batch 完成后，主机仍按以下顺序物化工作图：

```text
frontier state order
  -> point candidate order
     -> point reply CSR order
  -> end candidate order
     -> end reply CSR order
```

单 move、总 reply、state 和 move 预算也在这个原顺序中消费；只有预取发生得更早。child indices、OR shortcut、成功 move 选择和最终 continuation arena 因而与单状态 chunk 相同。固定 point 实例对 `N=1` 与 `N=256` 序列化后的完整 V1 proof 做逐字节比较，而不只比较根真值。

一个 chunk 内较晚状态的显式 CUDA 失败会让整个 chunk fail-closed，可能损失本可由较早状态得到的证明，但不会错误证明。`auto` 的运行时失败由公开 batch API 回退 CPU；缩小 chunk 可降低显存峰值与失败影响范围。

## 指标与当前结果

新增指标：

- `reply_frontier_batches`：实际包含待扩展状态的 chunk 数；
- `reply_frontier_states`：这些 chunk 覆盖的状态总数；
- `peak_reply_frontier_batch`：单个 chunk 的最大待扩展状态数。

固定 recursive-point 全 CUDA 运行保持 34 states、18 moves、84 replies、4 节点 proof。`N=1` 与 `N=256` 的对比如下：

| 指标 | `N=1` | `N=256` |
|---|---:|---:|
| frontier chunks | 7 | 2 |
| Hamilton batches | 9 | 4 |
| end batches | 2 | 1 |
| path-append batches | 9 | 3 |
| V1 proof | 相同 | 相同 |

当前数据只能说明 launch 合并和语义等价，不能说明端到端加速。样例规模很小，CPU 全量复核与设备图重复传输仍可能占主导。

## point-first end 筛选

初版 frontier reply batching 为保持父状态的 child 插入顺序，在知道 point move 是否 vacuous-success 之前预取 end replies，固定样例一度从 8 tasks/48 edges 增至 18 tasks/108 edges。M4.3b3b2b2a 已先跨 chunk 计算 point path-append flags，筛掉必然 shortcut 的 states 后才生成 end replies；当前工作量恢复为 8/48，且 child 仍按原顺序物化。

M4.3b3b2b2b1 已按相同 span 映射批量写出规范 child edge SoA，并继续保持 `N=1`/批量 proof 逐字节一致。下一步是 leaf 分桶与批处理。
