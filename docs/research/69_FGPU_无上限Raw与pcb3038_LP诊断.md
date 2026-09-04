# FGPU 无上限 raw 路径与 pcb3038 LP 强度诊断

## 1. 本轮结论

资源不是当前瓶颈。`pcb3038` 完全图 raw 运行常驻显存约 308.3 MiB；取消 CPU
逐边审计和证书后，最终边数仍为 23,720。与论文 Table 7 的 6,466 条相比，差距
来自 LP 松弛和 Local Elimination 深度，而不是内存、显存、证书空间或外层轮数。

本轮因此做了四项改动：

1. `fgpu-elim resident` 和复现实验脚本默认使用 `cpu-audit=0`，不生成 trace 或证书；
2. LP/local 的三个外层 `max` 默认均为 `0`，语义为一直运行到自然固定点；
3. GPU 使用稳定 edge-id 活动列表，删除后由 CUB 在设备端压缩，后续阶段不再扫描
   已删除的数百万条边；
4. manifest 新增 Geometry、PDLP、JV、Quick-HS 和压缩的独立计时。

raw 输出继续强制标记 `result_trust gpu-raw`。它可用于全 GPU 性能与强度实验，不能
被称为认证删边结果；需要回归安全性时仍可显式传 `--cpu-audit 1`。

## 2. 无上限 fixed point 的准确含义

三个参数均接受 `0`：

```text
--max-pdlp-epochs 0
--max-jv-rounds 0
--max-hs-epochs 0
```

每个 JV/Quick-HS epoch 若有删除就继续，否则自然终止；LP 与 local 交错后，只要
整个 orchestration 没有新增删除就终止。因此没有按 epoch 数截断，也不再设节点数
人为上限；可用规模由实际主机内存、GPU 显存和 `cudaMalloc` 结果决定。仍保留的检查是
`int32` edge id 可表示性、整数运算不溢出等正确性前提，不是资源预算。

PDLP 内部的 5,000 次更新不是导致强度不足的原因。下一节用 cuOpt 将完全相同的
度约束模型解到最优，直接排除了这一可能性。

## 3. pcb3038 完整 raw 画像与正式基准

运行使用 `pcb3038`、成本 137,694 的已知最优 tour 和单张 RTX 4000 Ada，
从完整图 4,613,203 条边开始。输出固定为：

| 阶段 | 删除 | 剩余 |
|---|---:|---:|
| Geometry | 4,544,561 | 68,642 |
| degree-box PDLP | 20 | 68,622 |
| JV | 841 | 67,781 |
| Quick-HS | 44,061 | 23,720 |

最终 content hash 为 `824cfe92e7345428`，公开最优 tour 缺边为 0。在 clean
commit `e102216` 上独占物理 GPU 2 连续复跑 7 次，七份边文件 SHA-256 均为
`ac3b0b9c1e4eb500269fad649b026e8ff3beef92e1ea0e9e070ab21e11ee114f`。正式中位数为：

| 阶段 | wall |
|---|---:|
| Geometry | 4.484 s |
| PDLP | 0.083 s |
| JV | 0.268 s |
| Quick-HS | 50.230 s |
| 活动列表压缩 | 0.002 s |
| GPU solve | 55.404 s |
| 应用端到端 | 56.255 s |
| 进程 wall | 56.330 s |

对应 artifact 为
`artifacts/pcb3038-fgpu-resident-20260904T082102Z-400033/`，manifest 记录
`git_dirty 0`。七次进程 wall 为 56.08–57.06 s；常驻显存 323,309,391 bytes
（约 308.3 MiB）。

旧 clean 基准 61.10 s 在 RTX A5000 上完成，与本次 RTX 4000 Ada 不是同卡，
不将 `61.10/56.33=1.085x` 当作代码加速。同一 RTX 4000 Ada 上的提交前探索运行
为 56.02–56.26 s，与 clean 结果一致；要报告严格优化倍数，仍需两个 clean
commit 在同卡上交错 A/B。

## 4. Quick-HS warp 并行否决

画像显示 Quick-HS 占 GPU solve 的约 90.6%。曾评估“一条目标边一个 warp、同一
`d_first` 的 Hamilton replies 并行”原型。最终边数和哈希完全不变，但 Quick-HS
从约 50.0 s 退化到 86.7 s，进程 wall 达 93.0 s。

原因是原 KH `-q` 谓词存在很强的首失败短路；warp 同时展开 32 个 reply 增加的
无效工作超过并行收益。初次撤销后遗留的 helper 抽取还使同卡 Quick-HS 增至
73.58–73.72 s；再次画像后已将谓词完整恢复为原单体内联形态。正式路径为一条边
一个线程的规范短路顺序，clean 中位数恢复为 50.23 s。

## 5. LP 强度逐层诊断

### 5.1 精确度约束 LP

新增只读命令 `fgpu-elim pdlp-inspect`，它不写边文件或证书。在 Geometry 输出的
68,642 条边上，cuOpt 把 degree equality + box LP 解到最优：

```text
objective = 135391
incumbent = 137694
gap = 2303
forced-one = 22
solve = 0.299 s
```

当前常驻 subgradient 删除 20 条，精确模型也只能删除 22 条，说明增加 PDLP 迭代
不会解决论文强度差距。

### 5.2 Held–Karp 1-tree oracle

独立 CPU 强度诊断运行 500 次 Held–Karp subgradient，最佳下界为
136,563.296，gap 仍为 1,130.704；按最佳 1-tree 的非树边替换代价只能提出约
268 条 forced-one。它比度约束强，但仍远不足以把 68,642 条边压到论文量级，
所以没有把这一 CPU oracle 混入全 GPU 主计时。

### 5.3 subtour cutting plane

新增研究 API `RunFgpuSubtourPdlp`：cuOpt 反复求解，host 诊断分离器加入 support
连通分量、多个阈值分量和最大生成树 fundamental cuts，直到没有新违反割。结果为：

```text
epochs = 12
cuts = 639
objective = 136587.21666667
forced-one = 289
cuOpt solve = 13.098 s
total = 16.118 s
```

再用外部 Concorde `CCcut_violated_cuts` 作为 exact-mincut CPU oracle，得到 650 个
割、目标值 136,587.5 和 292 个 forced-one，仍无数量级变化。bridge 只用于判断
缺失算法强度，显式标记 `concorde-cpu-oracle`，不会进入全 GPU 结果。

这说明论文的 6,883 条 LP 边不是仅靠 degree、1-tree 或 subtour cuts 得到的；论文
命令还运行 local cuts、comb 和 domino-parity 等 separator，并做完整列 pricing。

## 6. 与论文当前可比结论

| 输入/流程 | 最终边 | wall | 结论 |
|---|---:|---:|---|
| 论文 LP 图 → 作者 `KH -Jq` | 6,466 | 约 0.03 s（当前主机粗粒度） | 论文 Local 基线 |
| 同一论文 LP 图 → resident raw | 6,461 | 约 0.66 s | 强度对齐，当前 GPU 更慢 |
| 完整图 → 当前 resident raw | 23,720 | 约 56 s | 快，但强度不等价 |

因此不能用论文 Concorde 5,460.2 s 除以当前约 56 s 宣称约 97x：两者最终边强度
不同。也不能把 6,461 与 23,720 混成一个端到端结果。

## 7. 下一步

下一主线不是恢复证书、限制显存或增加迭代，而是补齐强 LP：

1. 用 Concorde 完整 root LP 作为只读 oracle，归档每类 cut 对 bound/删边的增量；
2. 优先实现 GPU 可批处理的 connect/subtour 与低阶 local-cut separator；
3. comb/domino 先保持明确缺口，不用启发式 KNN 裁边冒充 LP 删除；
4. LP 强度达到论文输入边量级后，再运行 Quick-HS 固定点并报告真正等强度端到端；
5. 同时按画像重构 Quick-HS 的跨目标短路调度，不再在单目标内部盲目展开 replies。

在完成第 2–4 项前，当前结果应称为“单 GPU raw 候选图实验”，不能称为论文完整
流程的 GPU 替代。
