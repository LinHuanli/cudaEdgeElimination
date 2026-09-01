# Hamilton–Tutte GPU leaf 驻留缓存

## 范围

M4.3b3b2b2b2b2a 消除增量 leaf cost 调度中每个 CUDA batch 重复分配并上传图坐标和 proper reconnect templates 的开销。该切片只改变设备数据生命周期，不改变 deletion-set、template、outside matching 的枚举顺序，也不改变当时的 CPU `TryReconnect`、completeness fallback 或 proof verifier；后续 M5 以 CPU 精确矩阵认证替换了通用 fallback。

每个主机线程维护一个首选 CUDA 设备；该线程在曾使用的每个设备上最多保留一份 k-opt cache。首次选择仍使用“可见设备中空闲显存最多者”，随后保持设备亲和性，避免少量缓存分配导致多个等容量 GPU 之间来回迁移。

## 驻留对象与精确键

缓存包含三类对象：

- 距离快照：`integer_x[]`、`integer_y[]`、维度和距离类型；
- 模板：分别保存 `k=3/4/5` 的完整 `EndpointMatching[]`、`k` 与生成器哈希；
- 工作区：可增长的 `KOptCostTask[]` 与 cost-cell 数组。

k-opt cost kernel 不读取活动边集合，因此距离快照的键只覆盖 kernel 的完整依赖，而不扫描无关边表。命中时逐个比较所有整数坐标以及维度、`EUC_2D/CEIL_2D` 类型；它不是只凭哈希接受缓存。模板命中同时要求固定生成器哈希和完整模板数组相等。任何一个字段变化都会先构造并上传新 buffer，成功后才替换旧缓存。

工作区采用只增容策略。容量不足时至少增长到请求规模，并尽量按两倍容量扩展；容量足够时只上传当前 task 前缀并下载当前 cost 前缀。每个 `DeviceBuffer` 记录 owner device，释放前切回该设备，防止跨 CUDA context 释放指针。`ClearKOptCostCudaCache` 可在当前主机线程的 epoch 边界显式释放所有驻留对象。

## 可观测指标

每次 `KOptCostBatchResult` 返回：

- `snapshot_hit`、`template_hit`、`workspace_hit`；
- 当前设备 cache 的 `resident_bytes`。

`PathSystemKOptBatchResult` 聚合 CUDA batch 数、三类 hit 数和设备驻留字节峰值；wavefront 再公开为：

- `leaf_cuda_cost_batches`；
- `leaf_snapshot_cache_hits`；
- `leaf_template_cache_hits`；
- `leaf_workspace_cache_hits`；
- `peak_leaf_device_cache_bytes`。

miss 数等于 `leaf_cuda_cost_batches - hits`。CPU 或 auto fallback 到 CPU 的 batch 不计入 CUDA cache 分母。

## 回归证据

直接 cost-matrix 单元测试在清空 cache 后依次覆盖：

1. 首次 `k=3`：snapshot、template、workspace 均 miss；
2. 原样重复：三类均 hit，矩阵逐单元相同，驻留字节不变；
3. 首次 `k=4/5`：snapshot hit，各自 template miss，扩大 workspace；
4. 把同一坐标图从 `EUC_2D` 改为 `CEIL_2D`：snapshot miss，`k=5` template 与已有 workspace hit；CPU/CUDA 矩阵仍逐单元一致。

固定双 7 点完整 3/4/5-opt cursor 在清空 cache 后产生 13 个 CUDA batches：snapshot 命中 12 次，三个模板首次各上传一次，工作区在每个 `k` 的首批扩大一次后命中 10 次；proof 仍与 scalar V1 逐字节一致。

固定 recursive-point 的 `N=256,max_deletion_sets=1` 全 CUDA wavefront 记录：

| 指标 | 值 |
|---|---:|
| leaf CUDA batches | 6 |
| snapshot/template hits | 5 / 5 |
| workspace hits | 4 |
| peak resident bytes | 1468 |
| leaf tasks/cells | 34 / 136 |

因此同一进程中只有首个 batch 上传距离快照和 3-opt 模板；两次 workspace miss 来自容量增长。34-state 工作图和 4-node V1 proof 保持原结果，并通过独立 `ht-verify`。

## 安全边界与后续

缓存内容仍只服务 GPU 候选成本；任何 cache 命中都不构成删边授权。设备异常或 CPU/CUDA 矩阵差异使显式 CUDA 搜索 unresolved，`auto` 仍可回退 CPU matrix；GPU 无候选时由逐 cell CPU 精确矩阵提供 completeness，详见 [M5 矩阵认证](31_M5_HT_CPU_Exact_Cost_Matrix.md)。

当前缓存为线程本地、同步 kernel 的正确性基线，没有跨线程共享或异步 stream 生命周期。后续 GPU/CPU long-tail、[cooperative multi-block continuation](20_Hamilton_Tutte_Multi_Block_Continuation.md)与[HT epoch commit](21_Hamilton_Tutte_Epoch_Commit.md)均已保持完整 proof 字节等价或独立重放。
