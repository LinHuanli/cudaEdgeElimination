# M5 HT 根 child 规范化排除实验

## 1. 假设与计时

[path-append 稀疏化](38_M5_HT_Path_Append_Profile.md)后，rl5915/d15112 的 CPU-fused host-build residual 仍约占 work graph 一半。根 `c,d` move 的笛卡尔积 reply 仍通过 `AppendChild -> AddPaths -> NormalizePathSystem` 使用 dense 规范化，因此先验证它是否为同类维度瓶颈。

提交 `6b2b8ad` 将 report/summary 升级为 V11/V14，新增：

- `root_child_normalizations`：实际执行根 child 规范化的规范次数；
- `root_child_normalize_ms`：只包含 `AddPaths`/dense 规范化，不含 child 写入；
- `host_build_unprofiled_ms`：既有 host-build residual 再扣除根规范化。

五路必须拥有相同根规范化次数；计时不进入 proof。

## 2. clean-commit 三实例结果

正式 runs 均绑定 clean commit `6b2b8ad`、物理 GPU 1、8 个 CPU cost threads 和相同公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T230917Z-2656770`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T230928Z-2657271`；
- d15112：`artifacts/d15112-ht-scan-20260901T230936Z-2656762`。

| 实例 | 根规范化次数 | CPU-fused root normalize（ms） | host residual（ms） | 占比 | 扣除后未画像（ms） |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 72 | 0.773 | 190.582 | 0.406% | 189.808 |
| rl5915 | 151 | 2.174 | 386.906 | 0.562% | 384.733 |
| d15112 | 647 | 28.642 | 1,099.359 | 2.605% | 1,070.717 |

三实例五路工作签名、proof 重放、最终边文件和受保护 tour 门禁全部通过。CPU Debug/Release、CUDA Release 全套测试通过；直接运行 Hamilton–Tutte 单元程序的 compute-sanitizer memcheck 为 0 errors。

## 3. 决策

根 child dense 规范化不是当前主瓶颈。即使将它理想地降为零，d15112 的 work graph 上界收益也只有约 1.3%。因此不把 path-append 的内部稀疏实现提升为共享 API，也不为了低占比路径扩大改动面。

剩余最可疑路径是 `BuildPointCandidateNodes`：它对每个尚未由 leaf 证明、且深度未达上限的 frontier state 扫描完整 `graph.dimension`，计算中点距离并排序。下一切片先记录被扫描状态数、检查节点数、入选节点数和包含式 CPU 时间，再决定采用 top-k 选择、候选预计算还是维度无关索引。
