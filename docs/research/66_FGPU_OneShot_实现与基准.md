# FGPU-Elim One-Shot 实现与首轮基准

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

当前可信表述是：快速主链已明显越过作者单轮强度，并保留约 20x 的 wall 余量；是否能在深 HT 后接近 4,016 条，需要以完整 target sweep 的最终结果判断。

pr1002 同环境作者 `KH -Jq` 从 501,501 条删到 21,651 条，最优 tour 缺边为 0；FGPU 快速主链为 23,288 条，比作者多 1,637 条（约 7.56%）。作者程序报告的 `Time=16432.86 s` 是 OpenMP 各线程累计 user CPU time，不是 wall；该次 wall 又受并发 HT 实验干扰，因此只用于确定强度，不计入正式 speedup。

### 5.1 外部 HS 诊断（不属于 FGPU 主方法）

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
5. 完整 HT 扫描首次在写出阶段超过 256 MiB 原文 sidecar 上限并失败关闭。V5 现在逐 sidecar 做有界 zlib 压缩，记录 raw/compressed size 和 CRC32，全部预编码通过后才打开输出文件。失败半成品的 256 MiB 原文用 gzip level 1 只有 17,942,822 bytes，说明膨胀主要来自文本表示。

## 7. 尚未达到设计终态的部分

- 几何仍先物化 `GraphSnapshot` 完全图，不是 100k 级 tile-streaming 输出；
- native LP 尚无 subtour cut pool，pr1002 的 degree bound 明显偏弱；
- HT continuation 控制和最终授权仍有 CPU 工作，不是 fully GPU-resident；
- 当前 `.nonpairs` 为经过 verifier 检查的规范空集合，尚未生成 LP/HT non-pair 证书；
- fixed 输出目前仅从最终证明图的 degree-2 节点推导；
- 尚未实现设计稿中的多输出 inside path-cover solver，现有 leaf 仍以 outside cursor 为搜索入口并用 inside coverage 批量覆盖；
- V5 已解决 HT sidecar 文本膨胀，但几何/LP/JV 仍是每条删除边一个外层 record；100k 完全图仍需要 tile/DAG 聚合证书；
- `max_local_nodes` 已有 CLI 契约，但当前浅层 HT profile 尚未以独立 ExtendedPool 执行该容量分桶。

这些缺口都以保留边或关闭不适用模块处理，不会把 heuristic 阴性解释成证明。
