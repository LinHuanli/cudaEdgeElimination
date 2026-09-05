# FGPU 强度升级 P0–P8：实现边界、正确性修复与论文对齐

> 本文保留 `563d78a` 及此前实验的历史口径。后续发现并修复了 fixed 提前写入的
> 快照事务问题，新增稀疏 PDHG、独立 SEC incidence 校验和完整四端点 OR；最新状态
> 见 [71 号实现记录](71_FGPU_Transaction_PDHG_and_PathEnd.md)。下文的“cutting-plane
> dual”实为静态窗口/有限族 connectivity SEC 次梯度，不是完整强 LP cutting-plane。

本文是 [FGPU 强度升级与纯 GPU 架构方案](../design/FGPU-Elim_Strength_Upgrade_and_Pure_GPU_Architecture.md)
的实现记录。设计中的 P0–P8 是依赖优先级，不是可以跳过正确性门禁的发布阶段。

## 1. 当前结论

分支 `research/fgpu-strength-upgrade-p0-p8` 已把原有 resident 快速前端扩展为一条
正式的单 GPU 联合固定点主链：

```text
sparse resident graph
  -> interval Geometry
  -> GPU cutting-plane dual / Signed128 精确授权
  -> LP edge delete / fix / non-pair / path closure
  -> JV / Quick-HS
  -> Main-Edge / metric-excess
  -> KH -e2 endpoint continuation
  -> 一层 point non-pair
  -> Direct Fix / non-pair implied fix / fixed propagation
  -> device replay
  -> parallel commit
  -> edge / pair / fixed 联合固定点
```

正式 `solve --mode gpu-safe` 不使用 CPU 逐边审计。CPU 只解析输入、发起 CUDA、
校验最终已知 tour 并写文件；候选生成、授权重算和状态提交都在 GPU 上完成。replay
在独立 kernel pass 中从不可变 snapshot 重算，不信任 proposal bit，但部分 proof
类型会复用同一组 device 数学谓词，因此这里不把它描述成异构独立 verifier。

这仍不是设计文档 P0–P8 的全部最终形态。完整 local-cut/comb LP、一般
Direct/Close Point、深度 4–6 continuation、`m<=5` 多输出 path-cover、persistent
ready queue 和 CUDA Graph 尚未完成。当前应称为“P0–P8 核心子集”，不能称为完整
复刻 2014 Edge Elimination 或 2023 Local Elimination。

经 `-e2` 正确性修复后，同论文 6,883-edge LP 输入上的可靠 `pcb3038` 结果是：

- 6,883 → 6,326 条边，删除 557 条；
- 424 条 fixed edges；
- 最终 22,760 个邻边对中有 1,404 个 non-pairs，即 6.16872%；
- 2,836 次 device replay、0 rejected；
- 三次隔离 clean E2E 中位数 153.220 s；
- 三次 edge/fixed/non-pair 文件逐字节一致，图 hash
  `88098fbab9b930d3`，全状态 hash `a97ccf56515068ec`；
- 成本 137,694 的已知最优 tour 零缺边。

2023 论文完整 bootstrap 为 5,548 条边、934 fixed、49.4% non-pairs、497 s。
当前墙钟数值是论文的约 `1/3.244`，但终图仍多 778 条边、强度明显不同，故
**3.244x 只是同输入墙钟比，不是等强度加速比**。

从完全图开始的正式 `pcb3038` run 得到 `4,613,203 -> 17,872`，19 fixed、
32,041 non-pairs，5,162,470 次 GPU replay、0 rejected，E2E 为 7,495.652 s
（2:04:55.652）。按边数它比 2014 Step 2 的 17,940 条少 68 条，但用时是论文
累计约 673 s 的 11.14 倍；相对 2014 Step 3 的 14,869 条仍多 3,003 条，虽然墙钟
是论文总计 21,322 s 的 `1/2.845`，仍然不是等强度加速。该长基准目前只有一次完整
计时；正确性门禁有效，性能数值应视为单次测量而非中位数。

## 2. 正式入口与运行契约

```bash
build/cuda-release/fgpu-elim solve \
  --instance .tmp/lkh-tours/pcb3038.tsp \
  --input-edges third_party/ElimTSP/data/pcb3038.all.edg.gz \
  --tour artifacts/lkh-tours/pcb3038.opt.tour \
  --tour-role known-optimum --expected-cost 137694 \
  --mode gpu-safe --device auto \
  --output-edges artifacts/pcb3038.gpu-safe.edg \
  --fixed artifacts/pcb3038.gpu-safe.fix \
  --nonpairs artifacts/pcb3038.gpu-safe.nonpairs \
  --manifest artifacts/pcb3038.gpu-safe.json
```

`solve` 不暴露阶段开关、epoch 上限、节点上限或 CPU audit 开关。edge、non-pair、
fixed 三类状态都不再变化时才返回 `termination=fixed-point`。显存分配失败、距离类型
不支持、计数溢出、snapshot 不一致或 replay 冲突都会安全失败，不把部分结果伪装成
成功终态。默认不生成大体积逐 proof 证书文件。

`gpu-fast-raw` 仅用于性能消融；它跳过 replay，manifest 明确标为 unaudited，不能
作为正式强度结果。本文所有论文对齐结果均使用 `gpu-safe`。

## 3. P0–P8 落地矩阵

| 优先级 | 状态 | 已进入正式 `solve` | 尚缺能力 |
|---:|---|---|---|
| P0 | 核心完成 | stable edge SoA、稀疏 CSR、按需整数距离、active edge compact IDs、dirty-root sweep、并行 degree-floor commit | 明确的双缓冲 snapshot 类型、完整 reverse dependency |
| P1 | 部分完成 | degree equality、16/24/32/48 静态 local SEC、三阈值动态 connectivity SEC、跨 epoch dual ensemble、GPU Signed128 bound、delete/fix/non-pair/path authorization | 精确 mincut separator、一般 local cuts、blossom/comb、domino-parity、长期 cut pool |
| P2 | 部分完成 | compatible/three-compatible pair、两中心 surviving-pair product、Main-Edge、低度数 metric-excess、全位置固定点 | 完整 2014 Direct/Close Point 和全度数 metric-excess 表 |
| P3 | 部分完成 | persistent triangular pair mask、LP/fixed-anchor/point non-pair、LP fix、Direct Fix、non-pair-implied fix、显式 fixed bits、degree-2 propagation | 深度 2–4 pair elimination、一般 endpoint-product fixing HT |
| P4 | 部分完成 | Quick-HS fast root、基础 replies、`-e1/-e2` endpoint reveal、q10 exact subset DP、LP leaf closure | 通用 F/A state pool、lazy OR/windowed AND、depth 4–6、generation cancellation、transposition |
| P5 | 部分完成 | 三路径精确 path ordering、warp distance cache、q10 exact DP | `m<=5` multi-output inside matching、coverage bitset、traceback path-cover DP |
| P6 | 部分完成 | surviving-pair product 的规范顺序、短路顺序、几何候选排序 | 动态候选评分、failure history、自适应 speculation window |
| P7 | 核心完成 | Signed128、proposal/replay pass 分离、parallel deterministic commit、冲突 fail-closed | 通用扁平 proof DAG 和异构 replay 实现 |
| P8 | 部分完成 | active Quick-HS sweep、resident arrays、增长型 compact workspace | persistent ready queues、CUDA Graph、stream overlap、完整 device termination、多 GPU |

实现中仍保留算法覆盖范围：Quick-HS/`-e2` 使用 16 个两跳候选中心，Main-Edge
使用 23 个插值位置、每位置 11 个几何候选，metric-excess 只在有证明的低度数路径
启用。这些不是运行预算或规模截断，但确实是相对设计文档完整算法的能力上限。

## 4. 已实现的关键能力

### 4.1 稀疏、全常驻状态

正式链路不再依赖 `n×n` active/distance/neighbor 表。稳定边 ID 与活动 bit 分离；
每轮提交后用 CUB scan/select 重建紧凑 CSR，并按 `(cost,node,edge-id)` 确定性排序。
non-pair 使用每个顶点邻边对的三角索引，CSR 压缩时按 stable edge ID 映射继承。
图结构校验、degree/fixed-degree、pair offsets、dirty vertices 和 active roots 都在设备端
维护。`pcb3038` 论文 LP 输入的 resident 状态为 2,884,756 bytes。

### 4.2 精确 GPU LP 授权

设备端为 degree rows、静态 local SEC 和动态 connectivity SEC 寻找 dual。浮点
迭代只负责找 multiplier；正式授权先量化，再用两 limb `Signed128` 重算：

```text
L                         box-Lagrangian lower bound
L + max(0, r_e)           强制使用边 e，超过 U 则删除
L + max(0,-r_e)           禁止边 e，超过 U 则固定
L + sum max(0,r_f), f∈F   强制 path system F，超过 U 则关闭 reply
```

所有 cut multiplier 非负性、incidence、常数项和 reduced cost 都在 device replay
重新构造。任一溢出或不一致均拒绝授权。当前 `pcb3038` 下界为 135,986，距最优
137,694 仍差 1,708（1.2404%）；这解释了该输入上 LP 没有直接删边，也解释了后续
point 枚举仍然很重。

### 4.3 Main-Edge、`-e2` 与精确局部叶

正式路径加入 compatible/universal-three-compatible、surviving pair product、两个
Main-Edge 严格 3-opt 方向、低度数 strong metric-excess，以及 KH `-e1/-e2`
endpoint reveal。局部叶覆盖 `Opt23/24/33/34/233/243/244/253/333/343`，q10
使用 cooperative warp subset DP，并支持 fixed endpoint merge 和 LP path closure。

所有 candidate 都针对同一不可变 snapshot 生成；只有 replay bit 通过后才进入并行
commit。论文 LP 输入上的 `pcb3038` 分阶段删除为 Geometry 2、JV 193、Quick-HS 231、
Main-Edge 68、`-e2` 15、fixed propagation 48。

### 4.4 non-pair 与 fixing 是一等状态

non-pair 来源包括 LP 二边路径下界、fixed-anchor 冲突和完整一层 point move。
point move 对路径外点枚举当前允许的全部邻边对，每个 reply 都必须由局部精确谓词
或 LP path bound 关闭。新增 non-pair 在本轮 replay/commit 后才可被下一 snapshot
消费，避免同 epoch 循环依赖。

fixing 覆盖 LP force-zero、degree-2、non-pair implied fixing 和 Direct Fix endpoint
pair product。Direct Fix proposal 与 replay 都穷举两个端点的 surviving pair 笛卡尔积，
同时记录在全状态 hash 中。终止条件联合比较 edge、non-pair 和 fixed 三类状态。

## 5. 关键正确性修复：旧强结果为什么作废

早期版本在论文 LP 图上曾得到 `6,268 edges / 439 fixed / 5,304 non-pairs /
62.635 s`。小图全最优 tour 穷举随后发现 `-e2` 可删除属于最优 tour 的边，因此
该组数据不具备正确性资格，不能用于论文对比。

同一缺陷也影响旧完全图产物 `17,644 edges / 2,258.580 s`；它不能代替本节新的
`17,872 edges / 7,495.652 s` 正式结果，也不能用来声称更强或更快。

根因有两项：

1. q10 path-order DP 被直接当作 KH 高阶谓词的充分条件，遗漏了作者实现中先行的
   `Opt23` 必要门禁；
2. endpoint reveal 可能让逻辑角色重合。只对顶点互异路径成立的快速筛选被错误地
   用于内部节点重叠形状，把“不支持”误当成“reply 已关闭”。

修复后：

- `Opt244/253/343` 必须先通过原 KH 对应的全部 `Opt23` 门禁；
- q10 高阶形状要求逻辑角色互异；发生歧义重叠时保守保持 reply 开放；
- `Opt33` 只在角色互异时使用快速必要门禁，共享端点交给通用 path normalization；
- proposal 与 replay 同时采用上述语义；
- 新增 `e2-overlap-regression` 和 `e2-distinct-regression`，并将正式 solve 接入小图
  全最优 tour 穷举门禁。

修复使 `pcb3038` 从旧的 6,268 条回升到可靠的 6,326 条，同时 point non-pair
明显减少。这不是性能回退掩饰，而是删除了 58 条没有充分安全依据的结果，并纠正了
由不安全 non-pair 连锁产生的过强 fixing/删边。

## 6. 验证门禁

### 6.1 全最优解与差分测试

`tests/exhaustive/check_fgpu_solve.py` 对小图枚举所有 Hamilton tours，验证：

- 删除边不属于任何最优 tour；
- fixed edge 属于所有最优 tours；
- non-pair 不出现在任何最优 tour；
- 两次正式 solve 的 edge/fixed/non-pair/state hash 一致；
- `proof_rejected == 0`。

另外覆盖 unprotected one-shot、non-pair 跨 epoch 持久化、Direct Fix、重叠/互异
`-e2`。64 组 q8/q10/fixed endpoint/path overlap 由 GPU warp DP 与 host permutation
oracle 逐值差分；128 组确定性非度量矩阵验证专用局部谓词与通用 path oracle 一致。

### 6.2 最终源码实测

| 门禁 | 结果 |
|---|---:|
| CPU Release ctest | 30/30 passed |
| CUDA Release ctest, sm89 | 53/53 passed |
| CUDA Release ctest, sm86 fresh build | 53/53 passed |
| `cudaee_unit` memcheck | 0 errors |
| `cudaee_unit` racecheck | 0 hazards / 0 errors / 0 warnings |
| formal overlap solve memcheck | 0 errors |
| formal overlap solve racecheck | 0 hazards / 0 errors / 0 warnings |

formal overlap solve 在 memcheck/racecheck 下均得到 `28 -> 8`、170 replay、0 reject，
全状态 hash `a596b9b314331737`。sanitizer 时间不能用于性能比较。

## 7. 中小实例端到端结果

### 7.1 `pr299` 完全图

成本 48,191 的已知最优 tour 作为保护门禁：

| 指标 | 结果 |
|---|---:|
| 初始 → 最终边 | 44,551 → 1,415 |
| fixed / non-pairs | 25 / 2,982 |
| 最终邻边对 / non-pair ratio | 14,353 / 20.7761% |
| replay / rejected | 150,828 / 0 |
| GPU solve / E2E | 200.682 / 200.848 s |
| 图 / 全状态 hash | `5d3efe6f4c95c89f` / `d5aafa0c208805ee` |

该结果证明完整图入口、LP path closure 和大量 replay 能端到端运行；它不等价于作者
已经过预处理的 1,208-edge 输入。这里 replay 为 51.644 s，占 E2E 25.7%，说明
完整图上的 proof 批处理仍有优化价值。

### 7.2 `pr299` 作者稀疏图

使用 `third_party/ElimTSP/KH-elim/pr299.edg`：

| 指标 | 结果 |
|---|---:|
| 初始 → 最终边 | 1,208 → 828 |
| fixed / non-pairs | 38 / 298 |
| replay / rejected | 859 / 0 |
| GPU solve / E2E | 9.938 / 10.107 s |
| 图 / 全状态 hash | `eccecf5bcd25a8c7` / `b5189fdf7268be1e` |

两组结果都通过最优 tour 零缺边门禁。

## 8. `pcb3038` 与 2023 Local Elimination 对齐

### 8.1 协议

```text
instance: .tmp/lkh-tours/pcb3038.tsp
input:    third_party/ElimTSP/data/pcb3038.all.edg.gz
tour:     artifacts/lkh-tours/pcb3038.opt.tour
cost:     137694
GPU:      NVIDIA RTX 4000 Ada Generation
mode:     gpu-safe
```

输入就是论文所用的 6,883 条 Concorde LP edges，避免把完全图前端与 LP bootstrap
混为一谈。

### 8.2 强度

| 指标 | 当前 GPU | 2023 full bootstrap | 差距 |
|---|---:|---:|---:|
| 初始边 | 6,883 | 6,883 | 0 |
| 最终边 | 6,326 | 5,548 | +778（相对论文终图 +14.02%） |
| fixed edges | 424 | 934 | -510 |
| non-pair ratio | 6.1687% | 49.4% | -43.2313 pp |
| 已删输入边 | 557 | 1,335 | 当前覆盖论文删边量的 41.72% |

non-pair 的当前分母是最终图中 22,760 个邻边对。各来源字段是跨 epoch 累计提案，
会重复命中同一最终 pair，不能相加当成最终 non-pair 数。

### 8.3 三次隔离 clean 计时

| run | E2E (ms) | GPU solve (ms) | LP + point (ms) | Main (ms) | `-e2` (ms) | replay (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 153,297 | 153,123 | 115,647 | 28,311.5 | 5,501.1 | 210.5 |
| 3 | 153,220 | 153,016 | 115,576 | 28,275.6 | 5,489.6 | 207.9 |
| 4 | 153,172 | 153,004 | 115,564 | 28,282.5 | 5,488.2 | 207.2 |
| median | **153,220** | **153,016** | **115,576** | **28,282.5** | **5,489.6** | **207.9** |

三次输出逐字节一致；图 hash 与全状态 hash 也一致。按中位数，LP + point 占
E2E 75.43%，Main-Edge 18.46%，`-e2` 3.58%，replay 仅 0.136%。因此在这张
稀疏输入上，去掉 replay 几乎没有有意义的总加速，反而失去正式授权边界。

额外的同机 `gpu-fast-raw` 消融为 153.153 s，图仍为 6,326 条且 graph hash 相同；
相对 safe 中位数只快 0.068 s（0.044%），小于 clean run 抖动。raw 不重放，manifest
标记 `unaudited=true`，并且输出层有意不导出未经 replay 的 94 条一般 fixed edges，
所以其 330 fixed 和全状态 hash 不能与正式结果混用。该消融实测确认：在论文 LP
稀疏图上移除 replay 不是有效的加速方向。

### 8.4 能说与不能说的加速

2023 论文在 48 cores、4 台服务器上报告 497 s。当前单 GPU 153.220 s，单看墙钟
为 `497 / 153.220 = 3.244x`。但当前多保留 778 条边、少 510 fixed，并且 non-pair
密度差 43.23 个百分点。因此严谨表述只能是：

> 在相同 6,883-edge 输入上，当前单 GPU 方法用论文约 30.83% 的墙钟得到一个更弱
> 的安全固定点；尚未得到与论文 5,548-edge 终态等强度的加速比。

## 9. `pcb3038` 与 2014 Edge Elimination 对齐

2014 论文从 4,613,203 条完全图边开始，报告：

```text
Step 1: 95,576 edges, 8 s
Step 2: 17,940 edges, 11:05
Step 3: 14,869 edges, 5:44:08
Total:  5:55:22 = 21,322 s
```

### 9.1 正式完全图协议与结果

```text
instance: .tmp/lkh-tours/pcb3038.tsp
input:    complete graph generated from TSPLIB, no --input-edges
tour:     artifacts/lkh-tours/pcb3038.opt.tour
cost:     137694
GPU:      NVIDIA RTX 4000 Ada Generation
mode:     gpu-safe
```

| 指标 | 当前单 GPU | 2014 Step 2 | 2014 Step 3 |
|---|---:|---:|---:|
| 初始边 | 4,613,203 | 4,613,203 | 4,613,203 |
| 最终边 | **17,872** | 17,940 | 14,869 |
| 初始边删除率 | 99.6126% | 99.6111% | 99.6777% |
| 阶段/累计墙钟 | **7,495.652 s** | 11:13 约 673 s | 5:55:22 = 21,322 s |

当前按**数量**比 Step 2 少保留 68 条（0.379%），但没有论文 Step 2 的具体边集，
不能据此声称两个终图具有包含关系或证明能力完全等价。当前完整主链和 GPU replay
合计比论文 Step 2 累计时间慢 11.14 倍；这说明当前实现虽然达到相近边数，却还没有
把 2014 的中等强度路径加速。

相对 Step 3，当前多保留 3,003 条（相对论文终图 +20.20%）；墙钟数值为
`21,322 / 7,495.652 = 2.845x`，只能表述为“同起点得到较弱安全固定点时快 2.845x”，
不能表述为等强度加速。

正式状态如下：

| 状态指标 | 结果 |
|---|---:|
| fixed edges | 19 |
| 最终邻边对 / non-pairs | 211,766 / 32,041（15.1304%） |
| replay / rejected | 5,162,470 / 0 |
| LP connectivity cuts / path replies closed | 4,133 / 73,917,580 |
| LP lower bound / optimum gap | 135,966 / 1,728（1.2550%） |
| resident bytes | 1,050,497,540（1,001.833 MiB） |
| 图 / 全状态 hash | `c0d80eb7a9b717ce` / `895b49f61309475c` |

输出前的 host 端否定门禁再次检查了已知最优 tour：成本 137,694、零缺边、零
tour non-pair 冲突，且全部 19 条 fixed edges 都在该 tour 中。它不替 GPU 授权删边；
任一条件失败只会让整次运行报错。edge/fixed/non-pair 文件 SHA-256 分别为：

```text
025271b411eaf52f32e832eef4fc0039ea2916d52755fef379bff5d3bfbd779c
0265972b6dfa250fe5790e620e317c38ce645b2f62b6090c2d4af997a9b1e418
193ed53e065d28d87f5cd2f2e8ff0f01467aa9f88662e539e33b15e9b28f4aff
```

这是一次完整正式 run，不是多次中位数。所选 GPU 没有其他计算进程，但节点上有
独立 CPU 作业，因此计时可用于当前数量级和热点判断，不能冒充跨硬件严格公平复现。

### 9.2 完整图阶段画像

| 阶段 | 删除边 | 时间 (s) | E2E 占比 |
|---|---:|---:|---:|
| Geometry | 4,544,561 | 1.365 | 0.018% |
| LP + point | 120 | 3,596.350 | 47.98% |
| JV | 884 | 5.744 | 0.077% |
| Quick-HS | 48,759 | 1,913.890 | 25.53% |
| Main-Edge | 3 | 34.688 | 0.463% |
| KH `-e2` | 999 | 180.898 | 2.413% |
| fixed propagation | 5 | 并入 commit | — |
| GPU replay | — | 1,760.220 | 23.48% |
| **E2E** | **4,595,331** | **7,495.652** | **100%** |

与论文 LP 稀疏图不同，完全图上 replay 已占 23.48%，后续值得做 proof compaction、
形状分桶和 replay kernel 专门化；但当前正式路径已经没有 CPU 逐边精确审计，继续
删除 host 检查不会回收这部分时间。最大热点仍是 LP + 完整 point（47.98%）以及
255-register 的 Quick-HS（25.53%）。

### 9.3 相对旧 resident raw 的净变化

旧 resident raw 在同一完全图上为 `4,613,203 -> 23,720`，七次中位 56.330 s；
当前正式结果少保留 5,848 条，即终图边数下降 24.65%，但 E2E 是其 133.1 倍。
两者信任边界不同：旧值没有 GPU replay 且只覆盖 degree-LP/local raw，当前值包含
Signed128、non-pair/fixing、Main/`-e2` 和正式 replay。因此这组数字说明强度升级确实
有效，却也说明当前新增组合搜索远未实现“加速版”目标，不能用 Geometry 的短时间
掩盖后续固定点成本。

## 10. 性能画像与下一步

`cuobjdump --dump-resource-usage` 显示：

| kernel family | registers/thread | stack | static shared |
|---|---:|---:|---:|
| depth-2 Quick-HS proposal | 255 | 1,680 B | 44,388 B |
| depth-2 replay | 255 | 1,648 B | 33,492 B |
| Main proposal/replay | 128 | 864 B | 约 400 B |
| point proposal | 80 | 320 B | 1,468 B |
| point replay | 74 | 320 B | 很小 |

监控中的 100% GPU utilization 只说明采样窗口内一直有 kernel 工作，不代表 occupancy、
issue efficiency 或显存带宽已经用满。深层 kernel 的 255 registers 和大 shared memory
会限制 resident warps，低显存占用也不是“GPU 没工作”的证据。

已做过一个不保留的 GPU-only A/B：用 CUB radix sort 按 degree/node 排列全部 point
roots，替代四个 degree bucket scans。`pr299` 和 `pcb3038` 输出 hash 不变，但
`pcb3038` E2E 从 158.223 s 变为 160.112 s，略有回退，故原型已撤销。结论是外层
point 扫描不是瓶颈，真正成本是每个 root 的全 reply 检查。

后续收益按优先级排序：

1. 补齐一般 local cuts、mincut、comb/blossom，让 LP bound 更靠近 incumbent，先降
   平均度和 pair product；这是同时提高强度和减少 75% 热路径的最大杠杆。
2. 将 q10 DP 与 root/reply 调度拆成专门 kernel，降低 255-register 压力；大 workspace
   改为 global/pool-backed，按形状专门化 CTA。
3. 把 Main-Edge 23 个位置合批，跨位置复用 KD candidates、surviving-pair masks 和
   metric-excess reduction。
4. 实现深度 2–4 non-pair continuation 和一般 Direct/Close Point；当前 6.17% 对论文
   49.4% 的差距会乘法放大 edge HT 与 fixing 的 reply 数。
5. 完成 `m<=5` multi-output path-cover，再接通通用 depth 4–6 F/A queues。
6. 完整图上 replay 占比高时再做 proof compaction/batching；稀疏 LP 图上 replay 仅
   0.136%，不能把移除安全检查当作主要优化。

## 11. 可复现位置

正式实验产物位于忽略目录，不提交大文件：

```text
artifacts/strength-upgrade-safe-final/pcb3038-paper-lp/
artifacts/strength-upgrade-safe-final/pcb3038-complete/
artifacts/strength-upgrade-safe-final/pr299/
artifacts/strength-upgrade-safe-final/sanitizer/
```

关键实现与回归：

```text
src/cuda/fgpu_resident.cu
src/cuda/signed128.cuh
src/fgpu/main_edge_predicate.hpp
src/fgpu/quick_hs_predicate.hpp
src/fgpu/resident_pipeline.cpp
tests/exhaustive/check_fgpu_solve.py
tests/exhaustive/check_fgpu_e2.py
tests/exhaustive/check_fgpu_direct_fix.py
tests/exhaustive/check_fgpu_nonpair_persistence.py
tests/unit/test_core.cpp
```
