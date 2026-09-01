# Hamilton–Tutte 增量 leaf cost 游标

## 范围

M4.3b3b2b2b2b1 将 frontier leaf batching 从单 deletion-set 扩展到任意 `max_deletion_sets`，包括 `0` 表示的完整枚举。目标不是预先展开全部组合，而是让每个 `(path system, outside matching)` 保存一个增量游标，每次只暴露原搜索下一块最多 `cost_batch_size` 个 deletion tasks。

同一调度轮中、同一 `k` 的多个 cursor blocks 展平为一个 cost matrix 调用。CPU 消费完整 block 后，只有尚未得到 witness 且预算未耗尽的游标才推进下一块。因此默认 100000 或无界预算不会造成投机任务、组合数组或 cost matrix 膨胀。

## 游标状态

每个游标私有保存：

- 已验证的 `TourContext` 与规范 outside matching；
- 当前 `k`、proper reconnect table 和组合索引；
- `has_combination`、当前 deletion/reconnect 计数；
- 一个待消费 block 及“消费后预算耗尽”标记；
- 最终 `KOptSearchResult`。

`PrepareNextBlock` 严格复现 scalar 搜索的循环顺序：`k=3,4,5`，每阶按组合字典序，block 内最多 `cost_batch_size` 项。预算检查发生在构造下一项之前；若一个 block 已含合法前缀，先完整消费前缀，再返回与 scalar 相同的“预算耗尽”。

`ConsumeBlock` 按原 work/template 顺序处理 cost cells。CUDA block 进入 consumer 前必须与独立 CPU 精确矩阵逐 cell 相等；低于 deleted cost 的候选仍逐项进入 CPU `TryReconnect`。CPU 全矩阵已经覆盖该 block 的全部 proper templates，因此无改善 row 不再调用通用 completeness fallback。找到 witness、出现 fatal 输入或计数溢出时，游标立即完成，不再生成后续组合。

## 跨游标调度

对当前 leaf batch 的每个未覆盖 outside 建立一个游标。每轮执行：

1. 向所有未完成游标请求至多一个当前 block；
2. 按 `k` 把 blocks 的 rows 拼接，同时记录 `(cursor,row_begin,row_count)`；
3. 一次调用 `EvaluateKOptTemplateCosts`；
4. 按 slice 拆回每个游标并由 CPU 消费；
5. 重复，直到本轮全部 outside 搜索完成；
6. 更新各 path proof 的 inside coverage，再选择下一未覆盖 outside。

任何时刻每个游标最多保留一个 block。不同游标可以处于不同 `k`；调度器每轮最多产生 k=3/4/5 三个矩阵，不会为了对齐而跳过或重排单游标工作。

CPU cost backend 现已使用相同增量 cursor 和精确矩阵；直接单 path API 的 CPU scalar 路径保留为独立 proof oracle。block 可以投机计算尾部 rows，但 `deletion_sets_tested` 只在 row 实际进入 consumer 时增加，预算边界和首次 witness 因而与 scalar 逐字节一致。完整门禁见 [CPU matrix 公平基线](32_M5_HT_CPU_Matrix_Baseline.md)。`auto` 的 CUDA batch 失败会整批转 CPU matrix；显式 CUDA 失败使对应游标 unresolved，之后仍允许配置的 CPU exact fallback。成功 proof 最终再次进入独立 `VerifyPathSystemKOptProof`。

## 等价与回归

固定 7 点无改善 leaf 使用 `max_k=5`、`cost_batch_size=2` 时，每个 scalar proof 依次测试 25 个 deletion sets：10 个 3-opt、10 个 4-opt、5 个 5-opt。两个相同 leaf 的增量游标得到：

| 指标 | 值 |
|---|---:|
| cost tasks | 50 |
| cost cells | 2660 |
| fused cost batches | 13 |
| scalar searches | 0 |

两份输出均与各自 scalar V1 proof 逐字节一致。另一个 `max_deletion_sets=3` 样例把每游标的 `2+1` tasks 合为两轮，共 6 tasks/24 cells；预算在第二 block 内的停止点和 proof 字节保持一致。

随机差分使用固定种子的 12 组 7 点整数坐标，每组包含两条不同单路径布局和一个双路径布局，并轮换 `max_k=3/4/5`、无界或 1–7 deletion 预算、`cost_batch_size=1–4`。CPU-only 与 CUDA build 均要求 batch 中每个 proof 的完整 V1 序列化等于独立 scalar 调用。

固定 recursive-point 在更强的 `max_deletion_sets=8,cost_batch_size=2` 配置下，4 个 leaf states 生成 20 tasks/80 cells，由 5 个 CUDA batches 完成；产生的 5 节点全局 proof 通过独立重放。该配置与单 deletion-set 搜索强度不同，不能拿 proof 规模直接作调度性能比较。

## 后续

后续已完成设备驻留缓存、128-cell long-tail policy、[cooperative multi-block continuation](20_Hamilton_Tutte_Multi_Block_Continuation.md)与[HT epoch commit](21_Hamilton_Tutte_Epoch_Commit.md)。路径上下文、组合游标、CPU witness 构造和 sidecar 重放仍位于主机。
