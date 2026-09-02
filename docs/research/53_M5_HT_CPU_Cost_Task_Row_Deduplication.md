# M5 HT CPU cost 完全相同 task row 去重

## 1. 画像与被否决方案

CPU cost 输出 workspace 落地后，pcb3038 CPU-fused 的 `cost_evaluate_ms` 仍为
`242.433 ms`。本切片先验证两条互斥假设，没有从结果反推实现。

端口互异 fast path 的画像保存在 `artifacts/kopt-distinct-port-profile-mR8heE`。pcb3038
的 3/4/5-opt task 端口全互异比例为 `58.57%/27.43%/2.18%`，按 template cells
加权后仅约 `7.1%`；rl5915 与 d15112 分别约为 `2.9%/4.5%`。该分支覆盖不足，且会在
scorer 热循环引入第二套冲突判定，因此没有保留。

完全相同 task 的画像保存在 `artifacts/kopt-duplicate-task-profile-Z6Qp7Z`。精确 key
由全部 10 个端口和 `deleted_cost` 组成；每个 batch 内的逻辑 rows/唯一 rows 比例如下：

| 实例 | 3-opt | 4-opt | 5-opt |
|---|---:|---:|---:|
| pcb3038 | 6.158× | 3.885× | 2.710× |
| rl5915 | 4.358× | 2.476× | 1.735× |
| d15112 | 3.502× | 2.163× | 1.659× |

第一个原型只计算唯一 rows，随后把结果展开回完整连续矩阵。它在 pcb3038 将 cost
evaluate 从基线 `242.433 ms` 恶化到 `305.337 ms`，故被否决。子阶段画像
`artifacts/kopt-task-dedup-phase-xW5NZp` 显示 5-opt 的 unique scoring 为
`71.265 ms`，但 row 展开本身达到 `100.217 ms`；问题不是去重，而是重新物化逻辑矩阵。

## 2. 零展开实现

提交 `a4afa29` 只优化内部显式 CPU leaf cursor：

- batch 逻辑矩阵少于 8,192 cells 时不建哈希表；
- 以完整 `port_nodes[10] + deleted_cost` 做相等判断，哈希冲突仍由 key 相等运算消解；
- 唯一 row 严格按首次出现次序保存，不遍历 `unordered_map` 产生输出次序；
- 只有唯一 rows 不超过原 rows 的 75% 时启用映射，否则丢弃去重结果并走原连续矩阵；
- CPU workspace 只写唯一矩阵，cursor 通过只读 `row_to_unique` view 定位成本 row；
- 不展开、不复制，也不改变每个 cursor 的逻辑消费顺序。

完整映射的大小和所有索引会在任何 cursor proof 计数推进前验证；每个 slice 在消费入口再做
一次局部形状与越界检查。损坏映射只会得到 `unresolved` 或内部错误，不能授权删除。

公开 `EvaluateKOptTemplateCosts` 仍返回完整 owning matrix。CUDA、hybrid 和 fused 的 CUDA
成本路径也不去重：设备矩阵继续由 CPU 对每个逻辑 row、每个 cell 做独立完整认证。严格改善
候选仍走通用重连路径重新取距并调用独立 witness verifier。

## 3. 逻辑认证与可观测性

CPU 去重路径只复用字节级相等 task 的纯函数结果，因此唯一 row 的精确整数评分同时认证其
全部等价逻辑 rows。以下规范量保持原语义：

- `cost_tasks`、`cost_cells` 和 `cpu_certified_cost_cells` 仍计逻辑工作；
- `cost_rows_consumed` 仍按规范 cursor 次序计数；
- proof、工作签名和删除集合计数不记录物理评分优化。

新增 `cpu_cost_rows_scored/reused`，并贯通到 wavefront、scan、CLI 和报告。HT scan report
升级为 V13，benchmark summary 升级为 V16；脚本要求每一路满足
`scored + reused == logical tasks`。`cpu_parallel_cost_batches/cells` 描述实际物理矩阵是否启用
OpenMP，去重后可以合理地因后端不同而不同，不再错误地作为跨后端规范等式；所有逻辑证明量
仍执行五路严格比较。

单元回归构造恰好 8,192 cells 的 2,048 个相同 3-opt rows，要求只评分 1 row、复用
2,047 rows，并确认首尾 proof 与标量 proof 逐字节一致。Debug 继续用 `-1` 哨兵证明唯一
物理矩阵的每个 cell 都被 scorer 覆盖。

## 4. 正确性门禁

提交前通过：

- CPU Debug 17/17；
- CPU Release 17/17；
- CUDA Release 20/20；
- CUDA Debug k-opt 与 Hamilton–Tutte 单元分别在物理 GPU 1/2 上通过
  compute-sanitizer memcheck，均为 0 errors。

选定的三实例正式结果对 `900f5d9` 执行 54 项跨提交精确比较：每实例五路最终边、五路工作
签名、去除 `metrics` 行的五路规范 proof、受保护最优 tour、JV 固定点边集和 JV proof
全部相同。所有 proof 和 tour 也由 benchmark 中的独立 CPU 进程重放。另一个 pcb3038 clean
抖动 run 同样通过额外 18 项比较。

## 5. clean-commit 正式结果

正式 runs 绑定 clean commit `a4afa29`、物理 GPU 1、8 个 CPU cost threads 和锁定来源的
最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T021216Z-2781508`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T021108Z-2779906`；
- d15112：`artifacts/d15112-ht-scan-20260902T021119Z-2780492`。

CPU-fused 的物理 row 复用为：

| 实例 | 逻辑 rows | 实际评分 | 精确复用 | 逻辑/评分 |
|---|---:|---:|---:|---:|
| pcb3038 | 739,641 | 209,918 | 529,723 | 3.523× |
| rl5915 | 45,938 | 21,433 | 24,505 | 2.143× |
| d15112 | 41,823 | 21,577 | 20,246 | 1.938× |

相对 `900f5d9` 的单变量结果如下；加速均为基线/当前：

| 实例 | cost evaluate（ms） | cost 加速 | CPU certify 加速 | leaf 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 242.433 → 180.396 | 1.344× | 1.362× | 1.193× | 1.186× | 1.181× |
| rl5915 | 18.796 → 17.632 | 1.066× | 1.070× | 1.002× | 0.991× | 0.974× |
| d15112 | 16.633 → 15.100 | 1.102× | 1.106× | 1.042× | 1.021× | 1.013× |

rl5915 的目标阶段稳定改善，但 search/wall 单次分别回退约 0.95%/2.71%，属于短任务共享节点
噪声，不标记为端到端收益。pcb3038 首个 clean run
`artifacts/pcb3038-ht-scan-20260902T021054Z-2779296` 的 CPU-fused 非目标阶段发生同步抖动，
cost/search 为 `224.106/752.065 ms`；同 commit 重复为 `180.396/534.939 ms`。两份 run
均保留且正确性完全相同，表中采用重复 run，不把首轮异常删除或伪装成算法回退。

## 6. 紧凑指纹链排除实验

在 `a4afa29` 之后又做了一次未提交的可回退实验：哈希表只保存 64-bit 指纹和唯一 row
索引，相同指纹通过完整 task 链逐项比较，因此即使发生指纹碰撞也不会错误复用。该结构减少了
哈希节点 key 大小，却增加了 collision-chain 数组写入。

三次 pcb3038 dirty A/B 保存在：

- `artifacts/pcb3038-ht-scan-20260902T021825Z-2784845`；
- `artifacts/pcb3038-ht-scan-20260902T021838Z-2785545`；
- `artifacts/pcb3038-ht-scan-20260902T021850Z-2786172`。

CPU-fused cost evaluate 为 `175.177/189.148/189.773 ms`，中位数 `189.148 ms`；完整 key
实现的三次稳定画像为 `182.096/181.687/178.707 ms`，中位数 `181.687 ms`。指纹链中位数
回退约 4.1%，首轮偶发改善不足以满足保留门槛，因此源码已完整回退，未产生代码提交。后续不再
仅凭哈希节点尺寸推断收益。

## 7. 下一切片

pcb3038 CPU-fused 现有主要 leaf 阶段为 cost evaluate `180.396 ms`、cursor consume
`106.344 ms`、prepare `85.654 ms` 和 setup `57.806 ms`。下一步若继续优化 cost，必须先以
独立子阶段计时区分完整 key 建表、唯一 task 搬运、距离表和 row scoring；不再直接尝试另一种
哈希容器。端口互异双 scorer 与紧凑指纹链都继续保持否决状态。
