# M5 HT 跨 batch reply 结果缓存排除画像

## 1. 结论

当前不实现跨 batch/target 的 Hamilton/end reply 语义结果缓存。上一切片已把 d15112 的
28,497 个逻辑 CUDA tasks 缩为 418 个 batch-local 唯一 tasks；把缓存生命周期继续扩大后，
理论最小值仅为 282，再减少 136 个（`32.536%`，上界 `1.4823×`）。更关键的是 23 个 reply
batches 中没有任何一批能达到“全部 key 已命中”，因此同步 CUDA 调用数仍是 23，无法消除
当前 reply 路径的主要固定开销。

引入 scan/epoch 生命周期的结果所有权、精确图失效键、容量策略和 CPU/CUDA 认证状态传播，
只能减少少量 count/write rows，却不能减少一次 launch/H2D/D2H/synchronize 边界。收益上界
不足以覆盖新增复杂度，本原型只保留画像 artifact，生产源码不加入结果缓存。

## 2. 画像方法

画像基于生产提交 `589caf5129b34e0c36a12924259efbd066afe5ed`。临时 observer 在
Hamilton/end batch 调用前把逻辑 key、target 和 batch id 写入当前命令的 stderr：

```text
CUDAEE_REPLY_TASK_KEY H target_u target_v hamilton_batch center
CUDAEE_REPLY_TASK_KEY E target_u target_v end_batch endpoint internal_neighbor
```

observer 不改变任务、结果或控制流，但同步文本输出会污染计时，因此本实验只使用离散 key
计数，不引用 report 中的阶段耗时。画像后临时代码已通过补丁完整移除，工作树恢复到生产
提交；不存在待合并的缓存原型。

运行条件与正式任务去重 A/B 相同：

- d15112 CPU JV 固定点，159,187 条输入边；
- `weight-desc` 前 8 个 targets，每 target 最多 2,000 states；
- reply CUDA 图/workspace 驻留和 batch-local 任务去重均开启；
- c,d、reply、path append、leaf cost 与 propagation 均显式 CUDA；
- 物理 GPU 1。

原始目录为 `artifacts/ht-reply-key-profile-20260902-RO5lcF`：

- 原始 stderr SHA-256：
  `8b5e95780055c3b54ddfc96bdd9629678de7f9a0b6afc767c14648ec040ca967`；
- `manifest.txt` SHA-256：
  `9f34320118e4f07108e018793629d4d2d8958c61121addd81bb41fbd3ec36599`；
- `analysis.txt` SHA-256：
  `b0b91066e6a7d1e8149a19d607c6a842bbbc405e26a9e4f2d0f1555451a1c3fb`。

## 3. 精确生命周期 key

Hamilton reply 依赖 graph、target 和 center。graph 在本 scan 内固定，因此可行的最长安全 key
为 `(target_u,target_v,center)`，缓存生命周期最多覆盖同一个 target 的多个 batches；不同
target 不能共享 center 结果。

End reply 只依赖 graph 和有向 `(endpoint,internal_neighbor)`。同一不可变 scan 内可以把该
有向二元组作为全局 key；方向仍不能规范成无向边。新 epoch 的活动 CSR 改变后必须整体失效。

本画像分别按这两个最长安全生命周期做精确 `sort -u`，没有概率哈希或近似等价。

## 4. 重复率上界

| 类型与最长安全作用域 | 逻辑任务 | batch-local 唯一 | 长生命周期唯一 | 还能移除 | 减少率 | 理论上界 |
|---|---:|---:|---:|---:|---:|---:|
| Hamilton：同 target | 23,939 | 242 | 160 | 82 | 33.884% | 1.5125× |
| end：整个只读 scan | 4,558 | 176 | 122 | 54 | 30.682% | 1.4426× |
| 合计 | 28,497 | 418 | 282 | 136 | 32.536% | 1.4823× |

Hamilton 的 160 个 target-lifetime keys 中，78 个只出现在一个 batch，82 个恰好出现在两个
batches；没有 key 出现三次。End 的 122 个 scan-lifetime keys 中，出现 1/2/3/4/5 个 batches
的 key 数分别为 `79/38/1/2/2`。这说明剩余重复是浅而分散的，并非少数高热 key。

## 5. Batch 消除能力

15 个 Hamilton batches 和 8 个 end batches 逐批按首次出现顺序模拟缓存：

- Hamilton 的后续 batch 仍各有 2–29 个新 target key；
- end 的每批仍各有 1–35 个新 scan key；
- Hamilton 全命中 batch：0/15；
- end 全命中 batch：0/8；
- 合计可跳过的 CUDA reply 调用：0/23。

因此即使缓存命中结果可以直接展开到逻辑 offsets/replies，每批仍必须为新 key 执行一次
CUDA count/write。上一切片开启 batch-local 去重后，d15112 的 15 个 Hamilton CUDA 调用仍
累计约 178–180 ms，8 个 end 调用约 68–70 ms；这些同步调用的固定成本不会按移除 136 个小
tasks 的比例下降。observer 运行自身的计时因 stderr 输出无效，这里只引用上一切片的正式
无 observer A/B 范围说明固定成本量级。

## 6. 正确性复核

画像运行仍产生与正式 A/B 相同的结果：

- 10,003 states、11,851 replies、1,114 leaf calls；
- 提交相同 2 条边，最终活动边为 159,185；
- 活动边 SHA-256：
  `39abcae8832b5eca8cee278237eafe52ed2f053fbe2abefd7f8593e746a189b4`；
- 去掉 V2 `metrics` 行后的规范 proof SHA-256：
  `f5bd5565f2f9dfe3c01fc0bc5c419dae648b252f65725433a8880060913cbe64`；
- 工作签名 SHA-256：
  `6a15c91380df021d507eeb2721cfba9ea2f32eb4e38c12b6c5c39c141a3bf9be`；
- 独立 CPU Release 重放得到最终哈希 `29c3b8fccaf1a3fc`；
- d15112 最优 tour 成本 1,573,084，缺边数 0，规范哈希 `4495654253f2318e`。

这些检查只证明 observer 未改变语义；它们不把被排除的缓存方案提升为已实现功能。

## 7. 决策与重启条件

当前决策：

1. 保留默认的 batch-local 精确任务去重，不增加跨 batch 结果所有权；
2. 不为 136 个潜在命中引入 scan/epoch cache、全图失效比较或额外 report 格式；
3. 不把结果缓存与跨 target 根候选融合混合评测；后者已独立证实回退；
4. artifact 中只保留 key 画像，所有 observer 计时明确标为无效。

只有以下任一条件成立时才重启：

- 新工作负载出现可完全命中的 batches，能实际消除同步 CUDA 调用；
- reply 调度改为持久 kernel/设备队列，使命中可以减少设备队列项而不新增 launch；
- 多 epoch 中同一精确 graph snapshot 被重复扫描，且结果缓存无需跨 CSR 失效；
- 三实例 clean A/B 显示目标阶段和端到端收益稳定，且 proof/tour 门禁全部不变。

下一研究边界应转向 target 级并行或真正主导的 leaf/work-graph 计算，而不是继续压缩已经降至
418 个的 reply rows。多 GPU 只能静态分配同一不可变快照上的独立 targets，最终仍必须按原
target 顺序复核、预算、选择 proof 并原子提交。
