# M5 HT CPU batch 精确距离表复用

## 1. 问题与测量

固定 reconnect plan 已消除逐 cell 模板解码，但每个 task 仍独立调用 `GraphSnapshot::Distance`。融合 leaf batch 中，不同 task 的端口节点高度重叠，因此 task-local pair cache 只能消除同一 task 内的重复，不能消除跨 task 的相同节点对。

临时只读画像保存在 `artifacts/kopt-pair-profile-fj2x5i`。`task_pairs` 是各 task 实际首次解析的合法端口对次数，`batch unique` 是每个同步 cost batch 内去重后的规范节点对数；下表的 unique 为逐 batch 求和，不是跨 batch 全局缓存：

| 实例 | k | tasks | task pairs | batch unique 总和 | 重用上界 | 单 batch 最大 unique |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 3 | 208,902 | 1,674,973 | 16,618 | 100.793× | 333 |
| pcb3038 | 4 | 327,428 | 5,840,006 | 16,358 | 357.012× | 337 |
| pcb3038 | 5 | 203,311 | 6,237,422 | 15,628 | 399.118× | 335 |
| d15112 | 3 | 16,911 | 132,104 | 2,018 | 65.463× | 309 |
| d15112 | 4 | 16,192 | 280,283 | 1,891 | 148.219× | 294 |
| d15112 | 5 | 8,720 | 263,982 | 1,246 | 211.864× | 271 |

pcb3038 合计 `13,752,401 / 48,604 = 282.948×`，d15112 合计 `676,369 / 5,155 = 131.206×`。这证明同步 batch 是合适的复用边界：它既有足够重复，又与一次 CPU 精确矩阵认证的生命周期完全一致，无需建立跨图或跨 epoch 状态。

## 2. 实现与资源边界

提交 `0d506ab` 在每次同步 CPU cost 调用入口收集全部 active port nodes，并一次构造对称的精确整数距离表。每个 task 只保存端口到表内节点的局部下标；OpenMP row workers 共享只读表，矩阵布局、template 顺序、`INT64_MAX` 无效值、整数距离和溢出语义均不变。

缓存必须同时满足以下门禁，否则沿用原 task-local 惰性 scorer：

- cost matrix 至少有 8,192 cells，与现有 CPU 行并行门槛一致；
- batch 最多包含 512 个不同端口节点，距离表最多约 2 MiB；
- graph 最多 1,048,576 个节点，稠密 `node -> local` 索引最多约 4 MiB；
- 完整局部节点对数不超过理论 task pair 请求的一半，即预计至少减少 2× 距离计算；
- 任一容量、乘法溢出或索引条件不满足时不建表。

总额外内存硬上界约 6 MiB，生命周期仅覆盖一次 cost 调用。节点收集使用稠密线性索引，不排序，也不把坐标距离近似为图边权；表内每项仍直接调用同一 `GraphSnapshot::Distance`。CUDA kernel 未修改，CUDA 返回的每个 cell 仍由该完整 CPU 矩阵认证。

首个基于排序/二分映射的原型在 pcb3038 dirty run 中使 cost evaluate 从约 `418 ms` 回退到约 `457 ms`，因此未提交。改为稠密线性索引后，dirty 重复运行约为 `315–317 ms`；正式 clean run 为 `350.125 ms`。这个否决结果说明高重用率本身不足以保证收益，建表和映射成本也必须受常数与容量门禁约束。

`KOptCostBatchResult::cpu_distance_cache_nodes` 暴露本次调用实际缓存的节点数，用于确定性单元门禁，不参与 proof 或消元授权。

## 3. 正确性与回退门禁

新增测试覆盖：

- 40 个重复 k=5 tasks 必须启用 10 节点距离表，并把所有规范 templates 与独立 edge-list oracle 逐项比较；
- CUDA 路径的完整 CPU certification 必须报告同样的 10 个缓存节点，且 CPU/CUDA 矩阵逐 cell 相同；
- 520 节点、52 tasks 超过容量上限，必须返回 0 cache nodes 并走原 scorer；
- 既有随机图、3/4/5-opt、batch/scalar proof、预算中断、CUDA 全矩阵认证和独立 witness replay 全部保留。

clean 提交的完整门禁为：CPU Debug 17/17、CPU Release 17/17、CUDA Release 20/20；CUDA Debug 的 k-opt 与 Hamilton–Tutte 单元均经 compute-sanitizer memcheck，0 errors。

三实例相对 `273ac9d` 进行了 54 项跨提交精确比较：每实例五路最终 `.edg`、工作签名、去除 `metrics` 行后的 V2 proof、受保护最优 tour，以及 JV 固定点边集/proof 全部相同；所有 proof 均由独立进程重放成功。

## 4. clean-commit 正式结果

正式 runs 绑定 clean commit `0d506ab`、物理 GPU 1、8 个 CPU cost threads 和锁定公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T010948Z-2741113`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T011001Z-2741715`；
- d15112：`artifacts/d15112-ht-scan-20260902T011013Z-2742284`。

CPU-fused 单变量结果如下；加速均为 `273ac9d / 0d506ab`：

| 实例 | cost evaluate：基线 → 距离表（ms） | cost 加速 | CPU certify 加速 | leaf 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 418.325 → 350.125 | 1.195× | 1.203× | 1.084× | 1.083× | 1.078× |
| rl5915 | 25.160 → 23.799 | 1.057× | 1.062× | 0.991× | 0.972× | 0.979× |
| d15112 | 24.083 → 19.799 | 1.216× | 1.226× | 1.034× | 1.012× | 1.004× |

pcb3038 的 CPU-fused search/wall 从 `913.156/950.567 ms` 降至 `843.266/881.674 ms`；d15112 search 从 `301.853 ms` 降至 `298.238 ms`。rl5915 的 cost 和 CPU certify 分别缩短 `5.41%/5.79%`，但 leaf/search 在单次正式 run 中回退 `0.91%/2.89%`，说明其端到端结果已小于共享节点噪声和其他 host 阶段波动，不能宣称整体加速。优化的直接目标阶段在三实例均改善，且容量门禁使不适合的 batch 自动回退，因此保留该实现。

## 5. 下一切片

pcb3038 正式 run 中，CPU-fused cost scatter 仍为 `37.841 ms`，cursor prepare/consume 也保持可见。当前 scatter 为每个 cursor slice 从融合矩阵复制一段 `added_costs` 后再消费；下一实验改为只读 `std::span` 消费，消除中间向量分配和复制，同时保持 cursor 顺序、投机 block 计数、候选重建及 proof 字节不变。若收益只落在噪声范围，则转向 cursor prepare 的组合展开，不扩大公共 API。
