# M5 HT leaf cost 零复制分发

## 1. 根因

跨 cursor 的 leaf bucket fusion 已经把同阶 tasks 合并成一份连续精确成本矩阵，但消费阶段此前为每个 `CursorSlice` 新建 `KOptCostBatchResult`，再用 `vector::assign` 复制对应 rows。pcb3038 的一次 CPU-fused 8-target scan 包含 51,309,996 个 `int64_t` cells，因此仅矩阵分片就累计复制约 391.5 MiB；数据没有发生变换，副本只为适配 cursor 的旧入口。

该复制位于 CPU/CUDA 完整矩阵已经生成和认证之后，不增加正确性信息。标量 k-opt 路径也早已按 row 使用只读 `std::span`，因此融合路径可以采用相同的借用语义。

## 2. 实现

提交 `45098d3` 增加仅在 `kopt_search.cpp` 内可见的 `KOptCostBlockView`，携带：

- `k` 与规范 template 数；
- 融合矩阵中连续 cursor slice 的 `std::span<const int64_t>`；
- cost backend 的 `std::string_view`；
- 完整 CPU 矩阵认证标志。

每个 cursor 仍先验证 k、template 数和子矩阵形状，再逐 row 调用原 `TryReconnectFromCostRow`。融合 `KOptCostBatchResult` 在全部 slices 同步消费完毕后才离开作用域，期间不修改 `added_costs`，因此 span 与 string view 生命周期完整覆盖消费者。`selected_device` 等只用于 batch 级统计、不参与 cursor 语义的字段不再复制。

`cost_scatter_ms` 指标保留，但现在只计量 view 绑定，不再计量内存分配与矩阵复制。公共 API、矩阵布局、cursor 顺序、投机尾部 row 计数、候选重建、proof 格式和删除授权链均未修改。

## 3. 正确性门禁

现有门禁直接覆盖这次生命周期与切片边界改写：

- 两个及多个 cursors 跨 3/4/5-opt block 融合，proof 与 scalar 逐字节一致；
- cost batch size、预算截断和随机 path systems 的 batch/scalar 差分；
- CPU、auto/CUDA 全矩阵认证与候选 witness 独立重建；
- HT scan 的工作签名、即时公开 verifier 和 V2 独立重放。

clean 提交前通过 CPU Debug 17/17、CPU Release 17/17、CUDA Release 20/20；CUDA Debug 的 k-opt 与 Hamilton–Tutte 单元分别在物理 GPU 1/2 上通过 compute-sanitizer memcheck，均为 0 errors。

三次 pcb3038 dirty 试跑的 scatter 为 `0.522/0.543/0.549 ms`，去除计时 `metrics` 行后的 proof、工作签名和最终边集完全一致；独立 verifier 与成本 137,694 的受保护最优 tour 均通过。

## 4. clean-commit 正式结果

正式 runs 绑定 clean commit `45098d3`、物理 GPU 1、8 个 CPU cost threads 和锁定公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T012039Z-2747970`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T012047Z-2748508`；
- d15112：`artifacts/d15112-ht-scan-20260902T012052Z-2747969`。

CPU-fused 单变量结果如下；加速均为 `0d506ab / 45098d3`：

| 实例 | scatter：基线 → view（ms） | scatter 加速 | leaf 加速 | search 加速 | total 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 37.841 → 0.534 | 70.871× | 1.032× | 1.028× | 1.027× | 1.024× |
| rl5915 | 2.327 → 0.076 | 30.817× | 1.045× | 1.052× | 1.025× | 1.008× |
| d15112 | 1.865 → 0.066 | 28.441× | 0.993× | 0.986× | 0.990× | 0.983× |

pcb3038 CPU-fused search/wall 从 `843.266/881.674 ms` 降至 `820.532/861.406 ms`；rl5915 search 从 `152.678 ms` 降至 `145.188 ms`。d15112 的目标阶段节省约 `1.800 ms`，但 cost evaluate 在本次单次正式运行中增加 `1.731 ms`，其他 host 阶段也有小幅波动，因此 search 回退 `1.39%`，不能宣称端到端加速。

相对 `0d506ab` 的三实例五路最终 `.edg`、工作签名、去 `metrics` 的 V2 proof、受保护 tour、JV 固定点边集和 proof 共 54 项精确比较全部相同；所有 proof 与 tour 均由独立进程复核。

## 5. 下一切片

复制消除后，pcb3038 CPU-fused leaf 的主要可见阶段为 cost evaluate `338.482 ms`、cursor prepare `185.747 ms` 和 cursor consume `133.506 ms`。下一步先审计 prepare 中每个组合重复分配 `deleted_positions`、映射并排序，以及构造 `KOptCostTask` 的成本；只有在能保持组合枚举顺序和 proof 计数逐字节一致时才引入固定容量 work 或增量组合状态。若准备阶段改写需要扩大公共 API，则先做更细画像，不以结构复杂度换取不确定收益。
