# M5 HT leaf CPU matrix 公平基线

## 1. 研究问题与结论

[CPU 精确成本矩阵认证](31_M5_HT_CPU_Exact_Cost_Matrix.md)把 CUDA leaf 的通用 completeness fallback 消除后，hybrid search 相对旧 CPU scalar 基线达到 `2.097×`。但该比较不公平：CUDA 路径使用固定数组 CPU matrix scorer，纯 CPU 路径仍对每个 template 调用通用 `TryReconnect`。

提交 `9fa301d` 让显式 CPU backend 使用同一个增量 cursor 和 CPU 精确矩阵。pcb3038 8-target clean run 的结论是：

- CPU matrix leaf/search 为 `4.491/11.015 s`，相对旧 CPU scalar 的 `17.896/24.298 s` 分别缩短约 `3.984×/2.206×`；
- all-CUDA、hybrid 和 fused search 分别为 `11.730/11.430/11.594 s`，都没有超过 CPU matrix；
- 51,309,996 个 cells、727,635 个实际消费 rows 和 987 个改善候选在四路完全一致；
- 最终图、内嵌 HT proof 语义、独立重放和受保护最优 tour 均不变。

因此，在“CUDA 全矩阵必须由独立 CPU 全矩阵认证”的同步契约下，当前 GPU leaf 是有价值的差分 oracle，但不是净性能加速器。后续性能结论必须以 CPU matrix 为基线，不能继续引用 CPU scalar 得出的 `2×` GPU 加速。

## 2. 实现

### 2.1 显式 CPU backend 进入 batch cursor

`ProvePathSystemsByKOpt` 过去只为 `auto/cuda` 建立 `KOptSearchCursor`；`cpu` 会逐 path/outside 调用 scalar 搜索。现在三种合法 cost backend 都使用相同 cursor：

1. 按 `k=3,4,5` 和组合字典序生成最多 `cost_batch_size` 个 tasks；
2. 同轮同 k 的多个 leaf blocks 合为一张矩阵；
3. CPU backend 用固定数组 scorer 生成完整精确矩阵并标记 `cpu_verified`；
4. consumer 按原 row/template 顺序检查，只有严格改善 cell 进入通用 CPU witness 重建；
5. proof 完成后仍运行独立 `VerifyPathSystemKOptProof`。

直接 `ProvePathSystemByKOpt(..., cost_backend=cpu)` 的 scalar 路径保留，作为 batch/cursor 的独立规范 oracle。

### 2.2 修复投机 block 的 proof 计数

强化随机差分时发现一个既有计数问题：matrix 搜索会先生成整个 cost block，并在生成时增加 `deletion_sets_tested`。若 block 中间已找到 witness，尾部 rows 虽未消费却进入 proof 计数，导致 witness 完全相同但 V1 proof 不与 scalar 逐字节一致。

修复后：

- block 生成只推进组合 cursor，不增加“已测试”计数；
- 每个 row 真正进入 consumer 前才增加 `deletion_sets_tested`；
- 预算检查使用“已消费 rows + 当前待生成 block 前缀”，仍只生成预算内任务；
- block 尾部可以为矩阵合批而投机计算，但不会伪装成已测试的 deletion set；
- 计数溢出显式返回 `unresolved`。

12 组固定随机图、3 种路径布局、多种 `k`、预算和 block size 现在同时比较直接 matrix、auto batch、CPU matrix batch 与 CPU scalar，全部 V1 proof 逐字节一致。该修复强化确定性和可审计性，不改变 witness 接受条件。

### 2.3 V8 benchmark 门禁

V8 summary 为 CPU 路径增加完整的 cursor/cost/consume 子阶段，并要求四路同时满足：

- cost cells 相等；
- 实际消费 rows 相等；
- 严格改善候选 CPU 重建数相等；
- 旧 completeness fallback rows/templates 相等；
- CPU-certified cells 相等，且等于全部 cost cells。

四份输出边文件继续逐字节比较，四份 proof 分别由独立 CPU Release 进程重放，最优 tour 在每份输出上重新检查。

## 3. 验证门禁

提交 `9fa301d2c76acb0b41612b7058cff4d326001f7d` 完成：

- CPU Debug：17/17 CTest；
- CPU Release：17/17 CTest；
- CUDA Release（物理 GPU 1）：20/20 CTest；
- CPU/CUDA k-opt 单测中的独立 cell oracle 与随机 scalar/matrix proof 差分；
- `compute-sanitizer`：0 errors；
- V8 1-target 冒烟及四路规范计数门禁；
- clang-format、shell 语法和 `git diff --check`。

正式 run id 为 `pcb3038-ht-scan-20260901T212700Z-2593290`，manifest 为 clean commit，物理 GPU 1（RTX 4000 Ada）。输入、目标切片和资源预算与文档 31 相同。

## 4. 正确性与工作量

| 指标 | CPU / all-CUDA / hybrid / fused |
|---|---:|
| attempted / proven / unresolved / committed | 8 / 2 / 6 / 2 |
| states expanded | 12,383 |
| replies expanded | 14,285 |
| leaf calls | 9,120 |
| cost cells（全 CPU 认证） | 51,309,996 |
| 实际消费 rows | 727,635 |
| 严格改善模板 CPU 重建 | 987 |
| 旧 fallback rows/templates | 0 / 0 |

四路最终图哈希均为 `fe11f98414b04c0e`，边文件 SHA-256 均为 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`。pcb3038 成本 137,694 的最优 tour 在四份输出上均为 0 缺边，规范 tour 哈希为 `ca0238497c090a3c`。

V2 proof 文件的数学 payload 相同；文件 SHA-256 不同仅因为 `metrics` 行包含各次运行的 commit wall time。四份文件均由独立 verifier 成功重放。

## 5. 公平性能结果

| 模式 | leaf ms | search ms | wall ms | 相对 CPU leaf | 相对 CPU search |
|---|---:|---:|---:|---:|---:|
| CPU matrix | 4,491.042 | 11,014.960 | 11,136.581 | 1.000× | 1.000× |
| all CUDA | 4,847.875 | 11,730.037 | 11,835.509 | 0.926× | 0.939× |
| hybrid leaf CUDA | 5,034.474 | 11,429.995 | 11,540.997 | 0.892× | 0.964× |
| hybrid + bucket fusion | 4,974.233 | 11,594.470 | 11,713.258 | 0.903× | 0.950× |

CPU matrix leaf 分解：

| 子阶段 | ms | leaf 占比 |
|---|---:|---:|
| setup | 1,755.290 | 39.08% |
| cursor prepare | 176.940 | 3.94% |
| CPU matrix evaluate | 2,333.507 | 51.96% |
| 其中 scorer/certify | 2,273.327 | 50.62% |
| cost scatter | 16.367 | 0.36% |
| cursor consume | 113.393 | 2.52% |
| apply + proof verify | 61.413 | 1.37% |

hybrid 的 cost evaluate 为 `2,830.208 ms`，其中 CPU certify `2,343.087 ms`。相对 CPU matrix，同步 CUDA 计算、传输和差分增加约 0.5 s，不能被已经必须执行的 CPU scorer 隐藏。fused 相对普通 hybrid 的 leaf 仅 `1.012×`，且 search 反而更慢；默认关闭的决定不变。

## 6. 新瓶颈与下一决策

CPU matrix 把 search 从约 24.3 s 降到约 11.0 s 后，主要时间重新排序为：

| CPU search 阶段 | ms | search 占比 |
|---|---:|---:|
| Hamilton reply | 6,078.790 | 55.19% |
| leaf | 4,491.042 | 40.77% |
| path append | 188.559 | 1.71% |
| host build residual | 187.981 | 1.71% |

下一切片应先细分 CPU Hamilton reply 的 count/enumerate、距离筛选、容器分配和跨 state 重复工作。CUDA reply 当前仍需 CPU 完整列表逐元素认证，all-CUDA 的 reply 时间 `6,234.941 ms` 并未胜过 CPU；在没有主机画像前继续增加 GPU batching 缺乏依据。

leaf 侧后续可评估 CPU scorer 并行化或 CPU/CUDA 并发差分，但同步 CUDA 不再作为默认性能路径。任何并行方案都必须保持：

1. CPU 精确矩阵完整覆盖所有 cells；
2. canonical row/template 消费顺序和首次 witness 不变；
3. 线程异常、CUDA 差异或资源失败只产生 `unresolved`/CPU fallback；
4. scalar proof 字节、最终图和最优 tour 门禁不变。
