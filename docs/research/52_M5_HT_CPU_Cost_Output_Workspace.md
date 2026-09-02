# M5 HT CPU cost 输出 workspace

## 1. 画像与被否决方案

cursor prepare 优化后，pcb3038 CPU-fused cost evaluate 仍为 `341.039 ms`。临时画像保存在：

- `artifacts/kopt-cost-phase-profile-OlI5Qe`：距离表构造、row scoring 与跨 k 节点集合；
- `artifacts/kopt-validation-profile-YyvoGM`：公开 task 验证的独立计时。

跨 k 共享距离表的假设被数据否决：382 个 frontier rounds 中没有一个 round 同时包含两个 k，`sum_nodes == sum_union == 7,070`，节点重用为 `1.000×`。因此没有实现跨 k binding，也没有扩大距离缓存生命周期。

第二次稳定画像中，batch 距离表构造合计 `10.333 ms`，row scoring `213.704 ms`，739,641 个 tasks 的公开验证合计 `11.980 ms`；但 leaf cost evaluate 为 `343.080 ms`、CPU certify 为 `329.740 ms`。剩余约 100 ms 的主要来源是每次构造 `vector<int64_t>(cell_count)` 时先把输出矩阵清零，随后 scorer 又覆盖全部 51,309,996 个 cells。

## 2. 实现

提交 `900f5d9` 把 CPU scorer 核心拆为写入调用方 `std::span<int64_t>` 的 `EvaluateKOptTemplateCostsCpuInto`。边界如下：

- 公开 `EvaluateKOptTemplateCosts` 仍返回 owning `std::vector<int64_t>`，输入验证和外部语义不变；
- 仅内部显式 CPU leaf cursor 使用 `KOptCpuCostWorkspace`；
- workspace 以 `make_unique_for_overwrite<int64_t[]>` 按需增长，不预清零；
- 所有 k 共用一个 storage，因为每份矩阵在下一次 evaluate 前已经被全部 cursor slices 同步消费；
- workspace 不保存图、tasks、模板或证明状态，并在一次 `ProvePathSystemsByKOpt` 返回时释放；
- 峰值容量等于本 proof batch 遇到的最大单份逻辑矩阵，不是三阶容量之和。

CPU scorer 仍逐 task、逐 canonical template 写入每个逻辑 cell。Release 在写入完成前不读取 storage；Debug 先填充不可能成为合法成本的 `-1` 哨兵，并在返回前检查无任何哨兵残留。矩阵形状、并行 cell 统计、CPU long-tail 统计和 cursor slice 均改用逻辑 span 大小，不读取 workspace 容量或旧值。

CUDA 与 hybrid/fused 的 owning host 结果未修改；CUDA 返回的完整矩阵仍由原独立 CPU owning vector 逐 cell 认证。严格改善候选的通用重建和独立 proof verifier 也完全不变。

## 3. 正确性门禁

内部 workspace 路径由显式 CPU random cursor 差分覆盖，包括不同 k、不同 cost batch size、预算中断和连续 workspace 重用。提交前通过：

- CPU Debug 17/17，包含全覆盖哨兵；
- CPU Release 17/17；
- CUDA Release 20/20；
- CUDA Debug k-opt 与 Hamilton–Tutte compute-sanitizer memcheck，0 errors。

三次 pcb3038 dirty 试跑均处理 51,309,996 cells，cost evaluate 为 `247.505/252.819/233.701 ms`，去 `metrics` proof、工作签名和最终边集完全相同。

正式结果相对 `611c701` 执行 54 项跨提交精确比较：三实例五路最终边、工作签名、规范 V2 proof、受保护最优 tour、JV 固定点边集和 proof 全部相同，所有 proof/tour 均独立复核成功。

## 4. clean-commit 正式结果

正式 runs 绑定 clean commit `900f5d9`、物理 GPU 1、8 个 CPU cost threads 和锁定公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T014640Z-2763322`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T014648Z-2763857`；
- d15112：`artifacts/d15112-ht-scan-20260902T014652Z-2763321`。

CPU-fused 单变量结果如下；加速均为 `611c701 / 900f5d9`：

| 实例 | cost evaluate：基线 → workspace（ms） | cost 加速 | CPU certify 加速 | leaf 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 341.039 → 242.433 | 1.407× | 1.424× | 1.156× | 1.148× | 1.132× |
| rl5915 | 23.743 → 18.796 | 1.263× | 1.272× | 1.156× | 1.134× | 1.174× |
| d15112 | 21.727 → 16.633 | 1.306× | 1.318× | 1.060× | 1.032× | 1.011× |

pcb3038 CPU-fused search/wall 从 `728.263/763.032 ms` 降至 `634.582/674.254 ms`；rl5915 search/wall 从 `146.429/242.128 ms` 降至 `129.119/206.256 ms`；d15112 search 从 `293.553 ms` 降至 `284.330 ms`。目标阶段及三实例端到端方向一致。

## 5. 下一切片

pcb3038 CPU-fused leaf 的主要阶段现为 cost evaluate `242.433 ms`、cursor consume `117.587 ms`、prepare `94.024 ms`、setup `60.001 ms`。cost row 的 canonical templates 已固定化，但每个 cell 仍为最多 5 条 added edges 做线性重复边检查。下一步计量 active port nodes 全部不同的 task 比例；仅在端口互异时，proper matching 的“一端口一次”性质才允许跳过该重复检查，端口重合的 task 必须保留完整 scorer。
