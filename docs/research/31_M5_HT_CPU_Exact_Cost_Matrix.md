# M5 HT leaf CPU 精确成本矩阵认证

## 1. 研究问题与结论

[CPU completeness 画像](30_M5_HT_Leaf_Completeness_Profiling.md)表明，pcb3038 的 8-target 混合路径把 `51,179,094/51,309,996` 个 cost cells 再交给通用 `TryReconnect` 穷举；该 fallback 占 cursor consume 的 `99.033%`。这些调用几乎都只是在重复证明“新增边成本不严格改善”，却仍反复创建集合、重算距离和重建拓扑。

提交 `48d68dc` 将这条稳态主路径改为 CPU 精确成本矩阵认证：CUDA 矩阵只有与独立 CPU 整数矩阵逐 cell 相等后才可用于筛选，只有严格改善的模板才进入原有完整 witness 重建与 verifier。正式 clean run 中：

- `51,309,996` 个 cost cells 全部获得 CPU 认证；
- 旧通用 completeness fallback 从 `51,179,094` 个模板降为 `0`；
- hybrid cursor consume 从 `15,254.955 ms` 降为 `125.605 ms`；
- hybrid leaf/search 从 `17.797/24.196 s` 降为 `5.011/11.586 s`；
- 相对当前纯 CPU scalar 基线，hybrid leaf/search 分别达到 `3.571×/2.097×`。

这不是用 GPU “无命中”授权证明，而是把逐 cell CPU completeness 改写成更紧凑的等价整数计算。

## 2. 安全契约

新路径分为三层，任何一层失败都关闭当前显式 CUDA 搜索；`auto` 调用方可退回 CPU：

1. **CPU 精确矩阵**：对每个 deletion task 和 proper reconnect template，检查自环、复用被删除边、重复新增边和整数加法溢出，并计算新增边的精确 TSPLIB 整数距离和。
2. **CUDA 全矩阵差分**：CUDA 返回值与上述 CPU 矩阵逐 cell 比较；规模或任意 cell 不一致均抛出错误，不能进入 row consumer。
3. **改善 witness 重放**：仅当认证后的 added cost 严格小于 deleted cost，才调用既有 `TryReconnect` 重建完整巡回、inside matching 和交换 witness；随后依次运行 path witness verifier、完整 leaf proof verifier、HT proof verifier 和 epoch commit verifier。

因此一个非改善 row 的完备性来自 CPU 对该 row **全部 proper templates** 的精确成本计算。GPU 结果只提供并行候选值，既不单独证明无改善，也不能绕过 CPU verifier。proof 格式、template 顺序、deletion-set 顺序和预算停止点均未改变。

## 3. 实现

### 3.1 每任务固定容量 scorer

`KOptCostTaskCpuScorer` 利用当前研究范围 `k<=5`：

- 用 `std::array<NodeEdge, 5>` 保存 deleted/added edges；
- 用最多 `10×10` 的固定数组缓存端口对距离，首次访问时才调用 `GraphSnapshot::Distance`；
- 用线性小数组检查替代每个 cell 的动态 `std::set`；
- 用显式上界检查保证 `int64_t` 求和不溢出。

缓存只活在一个 task 的评分期间，不保存图状态，也不跨 snapshot 复用。测试侧另写动态 edge-list oracle，对 `k=3/4/5` 的每个矩阵 cell 独立复算，避免用生产 scorer 自证。

### 3.2 认证状态与计时

`KOptCostBatchResult` 新增：

- `cpu_verified`：整个矩阵是否已通过 CPU 精确认证；
- `cpu_certify_ms`：CPU 矩阵生成和差分耗时。

该状态在矩阵切片进入各 leaf cursor 时显式传播。consumer 拒绝未认证的 CPU row；未认证 CUDA row 仍保留旧 fallback 作为防御分支，但正常新路径不会产生这种 row。

HT scan report 升级为 V6，benchmark summary 升级为 V7，并增加：

- `leaf_cpu_certified_cost_cells`；
- `leaf_cost_cpu_certify_ms`。

`leaf_cost_cpu_certify_ms` 是 `leaf_cost_evaluate_ms` 的子阶段，不能重复相加。三条 CUDA leaf 路径的认证 cell 数必须相同且等于 cost cells，否则 benchmark 失败。

## 4. 验证门禁

提交前后完成：

- CPU Debug：17/17 CTest；
- CPU Release：17/17 CTest；
- CUDA Release：20/20 CTest；
- CUDA 单元测试逐 cell 比较 CUDA、生产 CPU scorer 和独立测试 oracle；
- 随机 path-system 的 scalar/batch proof 字节、预算停止点和规范计数保持一致；
- `compute-sanitizer`：0 errors；
- shell 语法、clang-format 与 `git diff --check` 通过。

正式 run id 为 `pcb3038-ht-scan-20260901T211338Z-2585561`，clean commit 为 `48d68dc3b487fcfe70d69bcb9fd4ff97caed47c3`，使用物理 GPU 1（RTX 4000 Ada）。输入和预算与前一 completeness 画像相同。

四路均保持：

| 正确性/工作指标 | 数值 |
|---|---:|
| attempted / proven / unresolved / committed | 8 / 2 / 6 / 2 |
| states / replies / leaf calls | 12,383 / 14,285 / 9,120 |
| leaf cost rows / cells | 727,635 / 51,309,996 |
| 严格改善候选 CPU 重建 | 987 |
| 初始 / 最终图哈希 | `863392eda4798d1a` / `fe11f98414b04c0e` |
| 最终边文件 SHA-256 | `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810` |
| pcb3038 最优 tour | 成本 137,694；0 缺边；哈希 `ca0238497c090a3c` |

## 5. 性能结果

| 模式 | leaf ms | search ms | 相对 CPU leaf | 相对 CPU search |
|---|---:|---:|---:|---:|
| CPU scalar | 17,895.509 | 24,298.323 | 1.000× | 1.000× |
| all CUDA | 4,845.929 | 11,672.944 | 3.693× | 2.082× |
| hybrid leaf CUDA | 5,011.246 | 11,586.380 | 3.571× | 2.097× |
| hybrid + bucket fusion | 4,982.144 | 11,499.427 | 3.592× | 2.113× |

三条 CUDA 路径的 leaf cost/consume 画像：

| 模式 | cost evaluate ms | 其中 CPU certify ms | cursor consume ms | 旧 fallback ms |
|---|---:|---:|---:|---:|
| all CUDA | 2,666.096 | 2,324.154 | 126.245 | 0.000 |
| hybrid | 2,824.569 | 2,340.033 | 125.605 | 0.000 |
| fused | 2,764.473 | 2,326.468 | 121.537 | 0.000 |

与前一 clean run `ee4f3aa` 对比，hybrid 的旧 fallback `15,107.389 ms` 被约 `2,340.033 ms` 的固定数组 CPU 全矩阵认证取代。GPU kernel/传输的剩余成本约为 `cost evaluate - CPU certify`，仍只有数百毫秒；新的 CUDA leaf 主成本是 CPU 精确认证和 setup，而不再是 cursor consume。

bucket fusion 只比普通 hybrid leaf 快约 `1.006×`，仍不足以更改默认值。正式结果证明了认证数据布局的收益，不证明跨 GPU 型号或更大实例具有相同比例。

## 6. 下一切片

当前纯 CPU 模式仍走 scalar `TryReconnect`，所以表中的 CPU 基线没有复用已经验证的矩阵 scorer。下一步应先让 `--leaf-backend cpu` 使用相同增量 cursor 和 CPU 精确矩阵，并以以下门禁决定是否成为默认基线：

1. scalar 与 CPU matrix 的 proof 字节、deletion-set 计数、预算边界和最终图完全一致；
2. CPU Debug/Release、CUDA Release 与 sanitizer 全部通过；
3. 相同 8-target clean run 重新比较 CPU matrix、all CUDA、hybrid 和 fused；
4. 只有在公平 CPU matrix 基线下仍有收益，才继续优化 GPU 或跨目标融合。

之后的主要候选是减少每次 leaf 的 path proof/cursor setup、让 CPU 认证与 CUDA 执行重叠，或在更大任务上重新测量 GPU 交叉点；任何异步化都必须保留矩阵级 CPU 差分和 fail-closed 语义。
