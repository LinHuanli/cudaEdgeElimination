# M5 HT cursor prepare 路径边成本缓存

## 1. 根因

零复制分发后，pcb3038 CPU-fused leaf 的 cursor prepare 仍为 `185.747 ms`。每个 3/4/5-opt deletion work 此前执行两类重复操作：

- 为仅 3–5 个删除位置创建并释放一个堆 `vector<size_t>`；
- 构造 `KOptCostTask` 时创建临时删除边向量，并对相同 cursor/path 中反复出现的路径边重新调用整数 `Distance`。

同一 `KOptSearchCursor` 的图快照与规范 tour 在整个组合枚举期间不变。删除集合不同，但其成员都来自固定 `selectable_positions`，所以每条可删除路径边的精确成本只需计算一次。

## 2. 实现

提交 `611c701` 在 `TourContext` 构造成功后建立 `path_edge_costs_by_tour_position`：

- 只为规范 tour 中属于 path system 的边调用一次 `GraphSnapshot::Distance`；
- 非路径边保持 `-1`，cost task 只允许读取 selectable positions；
- 负距离、位置越界、非路径位置、删除数不在 3–5 或 `int64_t` 求和溢出均立即失败关闭。

`KOptCursorWork::deleted_positions` 改为固定 5 元数组。当前 k 决定有效前缀，映射后仍按原巡回位置排序；内部重连入口改收只读 `std::span`，组合枚举顺序、端口编号和 canonical template 顺序不变。

缓存只生成 cost task 的 `deleted_cost` 阈值。严格改善 cell 仍进入原通用 `TryReconnect`：重新构造删除边、重新调用精确距离、重建完整 tour/inside matching，并由独立 witness verifier 复核。因此缓存错误不能绕过候选重建和最终 proof 授权边界。

## 3. 正确性门禁

既有测试覆盖 3/4/5-opt、随机 path systems、多 cursor 融合、预算截断、batch/scalar proof 字节差分、CPU/CUDA 全矩阵认证和完整 witness replay。提交前通过：

- CPU Debug 17/17；
- CPU Release 17/17；
- CUDA Release 20/20；
- CUDA Debug k-opt 与 Hamilton–Tutte compute-sanitizer memcheck，0 errors。

三次 pcb3038 dirty 试跑中，cursor construct 为 `31.856/30.005/30.169 ms`，没有把 prepare 工作整体转移到 setup；prepare 为 `110.379/103.983/102.458 ms`。三次去 `metrics` proof、工作签名和最终边集相同。

正式结果相对 `45098d3` 进行了 54 项跨提交精确比较：每实例五路最终边、工作签名、规范 V2 proof、受保护最优 tour、JV 固定点边集和 proof 全部相同，所有 proof/tour 均独立复核成功。

## 4. clean-commit 正式结果

正式 runs 绑定 clean commit `611c701`、物理 GPU 1、8 个 CPU cost threads 和锁定公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T013037Z-2754256`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T013045Z-2754848`；
- d15112：`artifacts/d15112-ht-scan-20260902T013049Z-2754255`。

CPU-fused 单变量结果如下；加速均为 `45098d3 / 611c701`：

| 实例 | prepare：基线 → 缓存（ms） | prepare 加速 | leaf 加速 | search 加速 | total 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 185.747 → 94.438 | 1.967× | 1.138× | 1.127× | 1.125× | 1.129× |
| rl5915 | 17.888 → 11.904 | 1.503× | 1.049× | 0.992× | 1.008× | 0.967× |
| d15112 | 16.059 → 9.925 | 1.618× | 1.050× | 1.030× | 1.022× | 1.025× |

pcb3038 CPU-fused search/wall 从 `820.532/861.406 ms` 降至 `728.263/763.032 ms`；d15112 search/wall 从 `302.383/627.562 ms` 降至 `293.553/612.480 ms`。rl5915 prepare 与 leaf 分别改善 `33.45%/4.69%`，但 search 回退 `0.86%`、CLI wall 回退 `3.45%`，属于小实例其他 host 阶段和进程开销波动，不能宣称端到端收益。

## 5. 下一切片

pcb3038 CPU-fused leaf 的下一主要阶段为 cost evaluate `341.039 ms`、cursor consume `120.444 ms` 和 prepare `94.438 ms`。当前 batch distance cache 会为同一 frontier round 的 k=3/4/5 cost calls 分别建表，而画像显示三阶 active node 集高度重叠。下一步先拆分距离表构造与 row scoring 时间，并计量同轮跨 k 的节点集合交集；只有共享表能保持图对象绑定、512 节点/内存上限和完整 CPU/CUDA cell 认证时才实现跨 k 复用。
