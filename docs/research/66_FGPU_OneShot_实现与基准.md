# FGPU-Elim One-Shot 实现与完整基准

## 1. 当前结论

`research/fgpu-elim-oneshot` 已有一个单命令、单 GPU、可独立重放的端到端程序 `fgpu-elim`。它把以下阶段串成同一条不可变快照哈希链：

```text
完整图 / 输入稀疏图
  -> CUDA Main-Edge 几何候选 + MPFR 区间认证
  -> native CUDA degree-subgradient
  -> __int128 LP-box forced-one 删除
  -> exhaustive CUDA JV 固定点
  -> 可选的 PDLP-score Hamilton–Tutte wavefront
  -> 全证明重放
  -> .edg / .fix / .nonpairs / V4/V5 .fgcert / .manifest
```

这不是设计文档中“所有最终授权也只在 GPU 上完成”的版本。当前研究选择 CPU exact replay 作为最终信任边界：GPU 或浮点结果只能提出候选，不能独立授权删除。这一差异是有意的安全收敛，不应在论文中写成 fully GPU-certified。

## 2. 已实现模块

| 计划模块 | 当前实现 | 授权边界 |
|---|---|---|
| 完全图几何 | GPU 扫描 stable edge IDs；每边最多 32 个中点候选、8 组见证 | 256-bit MPFR outward interval 重算 strongly-potential/Main-Edge 不等式 |
| 图与 epoch | `GraphSnapshot` stable edge array、CSR 重建、不可变 hash、批量 degree gate | 提交前后哈希链；副本上原子发布 |
| JV | quick 与 exhaustive 两种 GPU 候选；FGPU 使用 exhaustive | CPU 整数谓词逐候选认证 |
| Path matching | `m<=6` 完整 outside/inside coverage table | 固定生成器哈希和 CPU/CUDA 差分 |
| k-opt leaf | proper 3/4/5-opt、candidate mask、CUDA exact cost | CPU 重建 witness 和完整 proof verifier |
| HT wavefront | F/A 工作图、批量 reply/path append/leaf、CUDA continuation | 每个成功 HT sidecar 完整 CPU 重放 |
| native LP | 单 GPU degree-equality box relaxation；小图 persistent CSR、大图 edge-atomic | dual 量化为分母 `2^24`，CPU `__int128` 重算全部 live variables |
| LP 删除 | `L + max(0,r_e) > U` 的 forced-one 判定 | V4 sidecar 绑定 snapshot、incumbent、fractional bits 和全部 vertex dual |
| LP/local 编排 | LP 与 exhaustive JV 交错，图变化后重新求 dual，直到固定点或 epoch limit | 每轮单独 proof epoch；最后从初图顺序重放 |
| 输出 | 五类输出；degree-2 surviving edge 可安全推导为 fixed | verifier 可复核 `.edg`、`.fix` 和当前规范空 `.nonpairs` |

HT 支持 1–32 个 host target workers 共享同一张 GPU，不属于多 GPU 加速。pcb442 的 64-target 小实验中，4 workers 与 1 worker 得到相同最终哈希，wall time 从 8.35 秒降到 5.53 秒。

## 3. LP-box 精确式

当前 native 模型只有 degree equalities 和 box：

\[
\min c^Tx,\qquad Bx=2,\qquad 0\le x\le1.
\]

对任意 unrestricted multiplier \(\pi\)，量化后令公共分母为 \(D=2^{24}\)，整数 numerator 为 \(q_v\)。验证器逐活动边计算：

\[
\hat r_e=c_eD-q_u-q_v,
\]

\[
\hat L=2\sum_vq_v+\sum_{e\in E_k}\min(0,\hat r_e).
\]

强制目标边进入 tour 后：

\[
\hat L_e^{(1)}=\hat L+\max(0,\hat r_e).
\]

只有严格满足 \(\hat L_e^{(1)}>UD\) 才生成 `LP_BOX` record。全部乘加使用 `__int128` 和显式 overflow gate；任意异常保留边。

## 4. 首轮结果

环境为 RTX 4000 Ada、驱动 610.43.02，单 GPU，`OMP_NUM_THREADS=8`。参数为 `potential_k=32`、`geometry_witnesses=8`、native LP 每 epoch 5,000 iterations、最多 8 个 LP epochs、exhaustive JV，表中先关闭深 HT，以便锁定快速主链。wall time 已包含最终完整 proof replay 和五类文件写出。

| instance | 完全图边 | Geometry 删除 | LP 删除 | JV 删除 | 最终边 | 删除率 | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| recursive-point (8) | 28 | 0（LP-only test） | 20 | 0 | 8 | 71.43% | 约 0.20 s |
| pcb442 | 97,461 | 85,297 | 3,872 | 277 | 8,015 | 91.78% | 4.872 s |
| pr1002 | 501,501 | 477,266 | 0 | 947 | 23,288 | 95.36% | 18.439 s |

正确性门禁：

- pcb442 官方最优 tour 成本 50,778，最终缺边 0；89,446 条删除 record 独立重放通过；
- pr1002 官方最优 tour 成本 259,045，最终缺边 0；478,213 条删除 record 独立重放通过；
- pcb442 最终 degree bound 为 `840009976524 / 16777216 = 50068.496258497`，与上界仍差约 709.5；
- pr1002 最终 degree bound 为 `4040747142945 / 16777216 = 240847.298082411`，与上界差约 18,197.7，因此当前 degree-only LP 在该实例没有产生 forced-one 删除。

`artifacts/` 被 Git 忽略；本次临时原始结果目录分别为：

- `artifacts/fgpu-pcb442-oneshot-final/`；
- `artifacts/fgpu-pr1002-oneshot-fast/`。

正式复现应使用：

```bash
CUDAEE_FGPU_ENABLE_HT=0 tools/run_fgpu_oneshot.sh pcb442 2
CUDAEE_FGPU_ENABLE_HT=0 tools/run_fgpu_oneshot.sh pr1002 2
```

脚本会校验锁定的 TSP/tour SHA-256、记录 GPU UUID、命令、Git dirty 状态和全部输出哈希，并保证产物只进入仓库 `artifacts/`。

## 5. 与既有 pcb442 结果的关系

已有 V3 对照为：

- 作者单轮 `KH -Jq`：12,914 条，94.89 秒；
- 作者 KH-HS 与 CUDA JV 交错到固定点：4,016 条，99.68 秒。

当前快速 FGPU 主链为 8,015 条、4.872 秒。它同时比作者单轮留下更少边且约快 `94.89 / 4.872 = 19.48x`；但相对 4,016 条固定点仍多 3,999 条。若只计算 wall 比为 `99.68 / 4.872 = 20.46x`，这个数字不是等强度 speedup，不能单独作为论文主结论。

当前可信表述是：快速主链已明显越过作者单轮强度，并保留约 20x 的 wall 余量；它是当前唯一适合作为“加速配置”的结果。

pr1002 同环境作者 `KH -Jq` 从 501,501 条删到 21,651 条，最优 tour 缺边为 0；FGPU 快速主链为 23,288 条，比作者多 1,637 条（约 7.56%）。作者程序报告的 `Time=16432.86 s` 是 OpenMP 各线程累计 user CPU time，不是 wall；该次 wall 又受并发 HT 实验干扰，因此只用于确定强度，不计入正式 speedup。

### 5.1 从完全图开始的完整 one-shot

主强度实验只用物理 GPU 2 上的一张 RTX 4000 Ada，`OMP_NUM_THREADS=8`、`CUDAEE_CPU_COST_THREADS=2`、16 个 host target workers 共享该 GPU。从 pcb442 完全图以一条 `fgpu-elim run` 命令启动，native LP 每轮 5,000 iterations、最多 2 个 PDLP epochs，每个 HT epoch 尝试全部当前目标，最多 16 epochs。

| 初始边 | Geometry | LP | JV | HT | 最终边 | termination | wall | 峰值 RSS |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 97,461 | 85,297 | 3,856 | 289 | 4,780 | 3,239 | `local-pdlp-fixed-point` | 3,201.08 s | 19,888,500 KiB |

它在单命令内提交 94,222 条删除 record，删除率 96.6766%；最终 content hash 为 `ba92119b724b2a1c`，`.edg` SHA-256 为 `8f08ee895375d415da8b5d23743dc929878f5543f3ee8cc8013f6cda084268b9`。应用程序的内置全量 replay 通过；随后用最终 tight-cap 二进制独立重放同一份证书，1,351.77 秒内再次验证 94,222 条 record，峰值 RSS 9,082,504 KiB。官方 50,778 tour 在在线门禁和独立 verifier 中均为零缺边。

V5 证书 SHA-256 为 `f9b9b92bab246c4f4e81450f1b0205ace0dbbb902ae4af782b9b398579299d23`，共 421,564,264 bytes；其中 4,780 份 HT sidecar 原文合计 6,812,894,031 bytes，zlib payload 414,979,169 bytes，压缩比 `16.417x`，最大单份原文 13,189,196 bytes。该证书已用最终 8 GiB raw / 448 MiB compressed / 512 MiB file 读取门禁通过。原始产物保留在 `artifacts/fgpu-pcb442-oneshot-ht-full-v5-final/`（`artifacts/` 不进入 Git）。

与作者单轮 12,914 边/94.89 秒相比，one-shot 少留 9,675 边（稀疏 74.92%），但 wall 慢 `33.73x`；与作者旧固定点 4,016 边/99.68 秒相比，少留 777 边（稀疏 19.35%），但 wall 慢 `32.11x`。因此可信结论是：当前 one-shot 已达到并超过文章级删边强度，但不存在等强度 GPU 加速，方向是显著减速。

### 5.2 分段证书链交叉验证

另一条 pcb442 完整深 HT 路径以每段已认证 `.edg` 作为下一段输入，直到真实 `local-pdlp-fixed-point`，而不是把 `ht-epoch-limit` 当作结束：

| 段 | 输入边 | 输出边 | 本段删除 | termination | wall | 证书 bytes | 独立重放 |
|---|---:|---:|---:|---|---:|---:|---:|
| 快速主链 | 97,461 | 8,015 | Geometry 85,297 + LP 3,872 + JV 277 | `jv-pdlp-fixed-point` | 4.872 s | 6,055,405 | 通过 |
| HT 小切片 1 | 8,015 | 6,952 | HT 973 + JV 90 | `ht-epoch-limit` | 708.64 s | 33,961,540 | 39.43 s |
| HT 小切片 2 | 6,952 | 5,794 | LP 1 + HT 1,107 + JV 50 | `ht-epoch-limit` | 3,460.89 s | 73,139,366 | 107.30 s |
| HT 全目标收口 | 5,794 | 3,231 | HT 2,551 + JV 12 | `local-pdlp-fixed-point` | 1,124.32 s | 129,492,375 | 62.12 s |

整条链恰好有 94,230 条删除 record，最终删除率 96.6848%，四份证书共 242,648,686 bytes。最终 edge content hash 为 `dff4f155be4057d8`，`.edg` SHA-256 为 `8259939de2f817548f43f4744130d8c69fd7bbeebc990d625ee50256df75e0c7`；官方 50,778 tour 在每段在线门禁和最终独立重放中均为零缺边。

最后一段 8/16 workers 的 wall 分别为 1,538.94/1,124.32 秒（`1.369x`），而 `.edg` 和规范化 `.fgcert` 均逐字节相同；证书 SHA-256 同为 `037b6f6d2a8e06a5bc1db703ec037ea0cefa5c4408c8e0edbd3c1c38afdf4f30`。表中采用更快的 16-worker 结果，仍只使用一张 GPU。

强度上，3,231 比作者旧固定点 4,016 少 785 条（稀疏 19.55%）；性能上，累计运行 wall 为 5,298.722 秒（88.31 分钟），是作者 99.68 秒的 `53.16x`，方向是减速而非加速。one-shot 的 3,239 条边是这 3,231 条的严格超集，只多保留 8 条；one-shot wall 比分段链减少 39.59%（`1.655x`），但合并证书是四段总证书的 `1.737x`。

两条路径都有有效证明。差异来自有界 HT 的 snapshot、目标评分、批提交和 degree gate 路径依赖；`local-pdlp-fixed-point` 表示在当前预算和所到达图上无新提交，不表示所有分批路径都会得到唯一最小边集。分段链因此是独立强度交叉验证，one-shot 才是用户要求的单命令主结果。

### 5.3 外部 HS 诊断（不属于 FGPU 主方法）

为判断快速主链后的剩余边是否仍有大量可删结构，对其输出额外运行一轮作者 CPU `KH -q`，再运行 exhaustive CUDA JV：

| instance | FGPU 快速主链 | CPU HS 后 | exhaustive JV 后 | 附加 wall |
|---|---:|---:|---:|---:|
| pcb442 | 8,015 | 3,837 | 3,836 | 2.39 + 0.24 s |
| pr1002 | 23,288 | 7,198 | 7,110 | 1.99 + 0.25 s |

两个实例的官方最优 tour 都保持零缺边。这证明 GPU 几何/LP/JV 预稀疏化可以让作者 HS 在很小的剩余图上快速完成，但这两行不是 FGPU 主结果：HS 仍是 CPU 外部程序，且没有进入 `.fgcert` 的可移植见证链。它们只用于定位下一个算法瓶颈：需要把作者专用 HS predicate 转成 GPU 候选与独立证书，不能把这组数字当作 fully GPU speedup。

## 6. 已解决的性能问题

1. LP record 原先每条都重新计算整个 stable-edge hash，形成 \(O(k|E|)\)。改为 epoch 一次绑定后，同一 pcb442 3,449-record LP 重放由 3.53 秒降到 0.34 秒，最终哈希不变。
2. proof replay 原先串行验证同 epoch records。现在共享数据串行构造一次，数学谓词并行执行，仍按规范次序报告首错和提交。pr1002 的 478,213-record verifier 约为 8.25 秒。
3. native LP 的 persistent CSR kernel 只在 `n<=64` 自动采用。pcb442 实测 cooperative barrier 比普通 edge-atomic 路径慢，因此大图使用测得更快的实现。
4. HT 从 weight-desc 改成 PDLP reduced-score 排序；pcb442 的 64-target 试验提交 23 HT + 2 JV，wall 8.35 秒。共享单 GPU 的 4 workers 把相同工作降至 5.53 秒。
5. 完整 HT 扫描首次在写出阶段超过 256 MiB 原文 sidecar 上限并失败关闭。V5 现在逐 sidecar 做有界 zlib 压缩，记录 raw/compressed size 和 CRC32，全部预编码通过后才打开输出文件。pcb442 的 973 份 sidecar 原文合计 505,332,235 bytes，压缩 payload 为 33,848,952 bytes（`14.929x`），最终整个证书为 33,961,540 bytes。
6. CPU workers 或多 worker 共享单 GPU 时，target 从静态 stride 改为原子动态领取，但结果仍按规范 target 索引回填。pcb442 512-target、4-worker clean A/B 均提交 228 HT + 15 JV，最终边集 SHA-256 同为 `932159d53af50f9e47e6093f66f8a864a21118557326ce58eeb115f706641a52`；wall 从 48.63 降到 44.01 秒（`1.105x`）。64-target 小批次为 5.53/5.55 秒，基本持平。完整收口的 8/16-worker 单 GPU 结果为 1,538.94/1,124.32 秒（`1.369x`），且边集与证书逐字节相同。多张不同 GPU 仍使用确定性静态分片；相关回归单独及连续 5 次均通过。
7. proof 的 epoch metrics 原先携带 wall-clock，导致相同数学 proof 因计时抖动产生不同字节。现在证书中的两个历史时间槽规范写 0，真实时间只留在 report/manifest；单元测试要求仅计时不同的 V5 证书逐字节相同。
8. pcb442 的旧 2 GiB V5 累计原文门禁被真实深扫触发并安全失败；全目标批量同样在 1,544.79 秒搜索后触发，且因先写证书而没有留下伪正式边集。累计原文上限现为仍有界的 8 GiB；从完全图开始的单份 one-shot 证书又真实触发了旧 384 MiB 压缩 payload 门禁。该次保守的 2-thread 压力运行在 6,805.11 秒后失败，峰值 RSS 19,305,412 KiB，输出目录仍为空；这个 wall 只是失败关闭证据，不是性能基准。成功证书的实际 payload 为 414,979,169 bytes，因此累计压缩上限收紧为有约 13.2% 余量的 448 MiB，整体文件继续保留 512 MiB 上限，单份 256 MiB 上限不变。最终收口段 2,551 份 HT sidecar 原文为 2,061,801,234 bytes，压缩 payload 为 129,211,331 bytes（`15.957x`），说明旧 2 GiB 原文门禁只剩约 85.7 MB 余量。
9. 含 HT 的 replay epoch 原使用 `schedule(dynamic,64)`，完整 one-shot 实测出现其他线程已空闲、单线程处理最后重型 chunk 的长尾。现在 HT 逐 sidecar 动态领取，几何/LP/JV 仍每 64 条领取；最终二进制对 94,222-record 证书的 8 线程独立重放为 1,351.77 秒，运行中未再观测到 64-record 的单线程尾部。

## 7. 尚未达到设计终态的部分

- 几何仍先物化 `GraphSnapshot` 完全图，不是 100k 级 tile-streaming 输出；
- native LP 尚无 subtour cut pool，pr1002 的 degree bound 明显偏弱；
- HT continuation 控制和最终授权仍有 CPU 工作，不是 fully GPU-resident；
- 当前 `.nonpairs` 为经过 verifier 检查的规范空集合，尚未生成 LP/HT non-pair 证书；
- fixed 输出目前仅从最终证明图的 degree-2 节点推导；
- 尚未实现设计稿中的多输出 inside path-cover solver，现有 leaf 仍以 outside cursor 为搜索入口并用 inside coverage 批量覆盖；
- V5 已解决 HT sidecar 文本膨胀，但几何/LP/JV 仍是每条删除边一个外层 record；100k 完全图仍需要 tile/DAG 聚合证书；
- `max_local_nodes` 已有 CLI 契约，但当前浅层 HT profile 尚未以独立 ExtendedPool 执行该容量分桶。
- `ReplayProof` 为保留返回容器语义会复制已验证 HT sidecar；完整 one-shot 进程峰值约 18.97 GiB，独立 verifier 峰值约 8.66 GiB，尚需不实体化返回 proof 的轻量 replay API。

这些缺口都以保留边或关闭不适用模块处理，不会把 heuristic 阴性解释成证明。
