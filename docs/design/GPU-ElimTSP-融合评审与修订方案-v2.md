# GPU-ElimTSP 融合评审与修订方案 v2

> **目的**：综合《GPU-ElimTSP-设计方案》与既有 cuOpt–CUDA Local Elimination 方案，保留其中可证明、可复现、投入产出比高的部分，修正尚未充分验证的假设，并形成一条可执行的研究与工程路线。
>
> **基线**：Cook–Helsgaun–Hougardy–Schroeder (2023) Local Elimination；`bicobico2/ElimTSP`；NVIDIA cuOpt 26.08 官方公开接口。
>
> **结论先行**：融合方案应把“强制边收缩精确 DP”和“任意合法拉格朗日乘子的安全下界”作为两个新的核心模块；把外部匹配改为静态查表；把纯 BFS 改为有界 AND/OR wavefront 与局部 DFS 的混合调度；把 cuOpt 定位为数值候选生成器和 LP 后端，而不是未经定价与认证的 Concorde 即插即用替代品。

---

## 1. 总体评价

上传方案包含两项很强、值得优先实现的技术贡献：

1. **强制外部匹配边收缩后的精确 Held–Karp 型 DP**。它把 `CCelim_tsp_swap` 中带强制边的小 TSP，转化为规模
   \[
   k=N-m=|F|
   \]
   的带方向超点 Hamilton 回路问题，并以
   \[
   O(2^{k-1}k^2)
   \]
   时间求解。该变换在数学上成立，并将不规则 branch-and-bound 改造成规则的整数动态规划。
2. **任意非负乘子的拉格朗日下界**。对标准形
   \[
   \min c^\top x\quad\text{s.t.}\quad Ax\ge b,\;0\le x\le1,
   \]
   任意 \(y\ge0\) 均给出
   \[
   L(y)=b^\top y+\sum_j\min\{0,c_j-a_j^\top y\}.
   \]
   这意味着 PDLP 的近似乘子不必先达到传统意义上的对偶可行性，仍可经过符号映射、完整列定价和定向舍入后产生严格有效的下界。

但原方案中以下结论需要降级为“待验证假设”或直接修正：

- 单个 `pr299 -z4` 的 gprof 不能证明 67.8% 热点比例适用于全部目标实例和全部 bootstrap level；
- “局部 TSP 返回改善仅 2%”不能推出 Hamilton–Tutte DFS 的 early-exit 也极少，因此不能据此直接断言纯 BFS 冗余因子仅为 1.5–3；
- 只在稀疏 restricted master 上计算拉格朗日界，若未计入被省略的完整图列，则不能自动得到完整 TSP 的有效下界；
- GPU 1-tree 在一个未经证明包含最优 tour 的稀疏图上运行，可能产生对完整图无效的过高下界；
- `384×384` 每任务匹配比较不如预计算 `inside-ID → outside-bitset`；
- 论文表中的 LP 单核秒与消元 48 核墙钟折算为 core-seconds，可以衡量计算资源，却不能直接作为端到端 wall-clock 的 Amdahl 权重；
- `550×` 单核、`97×–295×` 端到端等数字目前缺少 GPU kernel 实测支持，应作为远期上界假设，而非项目承诺。

---

## 2. 取长补短决策表

| 模块 | 原方案观点 | 融合决策 | 理由 |
|---|---|---|---|
| 强制边收缩 DP | 替代局部 Held–Karp B&B | **直接吸收，最高优先级** | 数学等价、已有初步 CPU 结果、最利于 SIMT |
| 去除热路径 malloc | arena/定长工作区 | **直接吸收** | 低风险，同时为 CUDA 数据结构铺路 |
| 3/4/5-opt case 表 | 常量表 + warp 比较 | **直接吸收** | 固定拓扑、整数运算、易验证 |
| 外部匹配反秩 | 每线程 Lehmer 解码 | **保留为生成器/测试工具** | 主路径中 \(m\le5\)，静态表更快 |
| 匹配覆盖 | 每任务 384×384 比较 | **替换为预计算 bitset 表** | 约 45 KB 即可覆盖 \(m=5\)，每次只需少量 OR/AND |
| 纯层同步 BFS | 全 GPU 常驻 | **改为有界 wavefront + 局部 DFS** | 控制状态爆炸、保留 early-exit、减小证书 |
| 删除候选排序 | 全宽度展开 | **不采纳** | 排序同时控制分支数与证书规模；应改为 GPU top-k |
| kNN 替代 kd-tree | 预计算每点 kNN | **实验性采用** | edge-midpoint 查询不一定等价于 per-node kNN |
| PDLP 任意乘子界 | 近似乘子即可严格使用 | **吸收，但加入完整列定价与区间认证** | 定理正确，工程前提不可省略 |
| cuOpt 替代 Concorde simplex | 直接更换求解范式 | **先 sidecar，后混合后端** | cut loop、定价、basis、分离仍需处理 |
| GPU 1-tree 替代 LP | 首先实施 | **降级为辅助界/排序器** | 在未经认证的稀疏图上可能不是完整 TSP 下界 |
| cuOpt 内部 headers | 直接借用 | **仅借鉴范式，核心代码自实现** | 内部 API 不稳定，依赖链重 |
| 550× 模型 | 远期性能预测 | **改为阶段性测量模型** | 当前数据只支持“值得做”，不支持精确倍率 |

---

## 3. 强制边收缩 DP：应成为第一核心

### 3.1 等价性

给定路径系统 \(P_F\) 和 outside matching \(O\)。原基准 tour 为

\[
T_F=F\cup O,
\qquad c(T_F)=c(F)+c(O).
\]

局部 oracle 需要回答：是否存在 Hamilton 回路 \(T'\supseteq O\)，满足

\[
c(T')<c(F)+c(O)?
\]

将每条强制边 \(o=(u,v)\in O\) 收缩成一个带两个端口的超点，访问该超点时选择方向 \(u\to v\) 或 \(v\to u\)。其他内部点作为单端口超点。则

\[
c(T')=c(O)+c(T'\setminus O),
\]

因此

\[
c(T')<c(F)+c(O)
\iff
c(T'\setminus O)<c(F).
\]

这给出一个仅在 \(k=N-m=|F|\) 个超点上的精确问题。

### 3.2 更适合 GPU 的两阶段实现

不建议每个任务都保存 parent table。采用：

1. **cost-only pass**：仅计算最优值；
2. **witness pass**：只对成功任务重算并回溯 inside matching。

```text
Algorithm CONTRACTED-DP-COST(task):
    build oriented supernodes from O
    threshold <- c(F)
    D <- +infinity

    initialize one-edge states
    for subset size r = 1..k-2:
        process all states of layer r in parallel
        relax to layer r+1
        clamp every value >= threshold to +infinity

    best <- close cycle to the fixed start supernode
    return best < threshold
```

### 3.3 GPU 分桶

建议按 \(k\) 采用不同 kernel：

| \(k\) | 实现 |
|---:|---|
| 3–8 | 一个 warp 一个任务；寄存器/小 shared memory |
| 9 | 一个 block 一个或多个任务；shared memory |
| 10–11 | 一个 block 一个任务；约 40–90 KB shared memory，低 occupancy 但仍可批处理 |
| 12–13 | global scratch 或 CPU fallback；需根据实测决定 |
| >13 | 原 Held–Karp、CPU exact DP 或限时 heuristic |

RTX A5000 与 RTX 4000 Ada 对应的 compute capability 8.6/8.9 每 SM 最大 shared memory 为 100 KB、每 block 为 99 KB。因此 \(k=11\) 的约 90 KB DP 表只能做到接近一 block/SM，不能按 A100 的 164 KB 参数估计。

### 3.4 必须先复现

上传文档引用了 `dpswap.c` 与 `improve.c.patch`，但当前材料中没有这两个文件。因此当前的 1.6× 端到端结果只能视为待复现实验。合入主分支前应完成：

- 对所有 \(N\le12\) 局部任务与暴力枚举逐例比较；
- 对原 Held–Karp 返回 witness 的任务检查 DP 必须返回同等或更好 witness；
- 检查 DP traceback 得到的 inside matching 与 `check_match` 一致；
- 在 `pr299`、`pcb3038`、`fl3795` 和至少一个深参数 workload 上收集 \(k,m,N\) 分布。

---

## 4. 外部/内部匹配：静态表优于运行时反秩

对于 \(m\le5\)：

\[
|\mathcal O_m|=2^{m-1}(m-1)!\le384,
\]

而 10 个端点上的 perfect matchings 数为

\[
9!!=945.
\]

离线生成：

1. `outside_matchings[m][oid]`；
2. `perfect_matching_id[m][canonical_pairing]`；
3. `coverage[m][inside_id]`，其中每行是最多 384 bit。

\(m=5\) 的 coverage 表大小为

\[
945\times384\text{ bits}=45{,}360\text{ bytes},
\]

可放入 constant/global read-only memory。

```text
uncovered <- ALL_OUTSIDE_MASK[m]
while uncovered != 0:
    oid <- first_set_bit(uncovered)
    witness <- FIND_IMPROVEMENT(F, outside[oid])
    if witness == NONE:
        return UNRESOLVED
    iid <- CANONICAL_INSIDE_ID(witness)
    uncovered &= ~coverage[m][iid]
return PROVEN
```

Lehmer 反秩仍有两个用途：

- 生成静态表的离线工具；
- 将来若把 path-count 上限提高到 \(m>5\)，作为 fallback。

---

## 5. PDLP 安全下界：吸收定理，但修正工程前提

### 5.1 一般有界变量形式

对

\[
\min c^\top x,
\qquad Ax\ge b,
\qquad \ell\le x\le u,
\]

任意 \(y\ge0\) 定义

\[
\bar c=c-A^\top y.
\]

则严格的拉格朗日下界为

\[
L(y)=b^\top y+
\sum_j \min_{x_j\in[\ell_j,u_j]} \bar c_jx_j.
\]

在 \([0,1]\) 情形退化为

\[
L(y)=b^\top y+
\sum_j\min\{0,\bar c_j\}.
\]

强制一个变量 \(x_j=1\) 的额外代价为

\[
\pi_j^1=\bar c_j-\min\{0,\bar c_j\}=\max\{0,\bar c_j\},
\]

强制 \(x_j=0\) 的额外代价为

\[
\pi_j^0=0-\min\{0,\bar c_j\}=\max\{0,-\bar c_j\}.
\]

因此：

\[
L(y)+\pi_j^1>U \Longrightarrow x_j=0,
\]

\[
L(y)+\pi_j^0>U \Longrightarrow x_j=1.
\]

对 Hamilton 已揭示集合 \(F\)，更一般的安全叶判据为

\[
L(y)+\sum_{e\in F}\max\{0,\bar c_e\}>U.
\]

这比简单地累加可能为负的 raw reduced costs 更稳健，也直接适用于未完全对偶可行的 PDLP 乘子。

### 5.2 必须加入的“完整列”条件

上述定理对**模型中的全部变量**成立。若 cuOpt 只求解 restricted master \(E_R\subset E(K_n)\)，则

\[
L_R(y)=b^\top y+
\sum_{e\in E_R}\min\{0,\bar c_e\}
\]

只自动界定 restricted problem。要得到完整 TSP 下界，必须满足下列之一：

1. 当前稀疏图已有独立证书，保证包含全部最优 tours；
2. 对所有 omitted columns 完成精确 pricing，并把负项
   \[
   \sum_{e\notin E_R}\min\{0,\bar c_e\}
   \]
   纳入下界；
3. 使用可证明覆盖全部负 reduced-cost columns 的几何/组合定价器。

因此，cuOpt 的正确定位是：

```text
cuOpt PDLP/barrier
      -> approximate multipliers
      -> row-sign canonicalization
      -> full-column pricing / omitted-column accounting
      -> interval or rational evaluation
      -> certified L(y), edge penalties and path penalties
```

### 5.3 cuOpt 集成路线

**第一阶段：sidecar**

- Concorde 继续生成 cuts 与执行 pricing；
- 将 checkpoint LP 导出 CSR/MPS 给 cuOpt；
- cuOpt 返回 primal/dual；
- CPU 重新计算 \(A^\top y\)、完整列 pricing 与严格下界；
- 比较 cuOpt 与原 LP solver 的时间和删边强度。

**第二阶段：周期性混合后端**

- 小模型或频繁少量行更新：CPU dual simplex；
- 大模型 checkpoint：PDLP/barrier；
- 相邻 epoch 用 primal/dual warm start；
- crossover 用于 cut separation 与定价稳定性，而不是 soundness 的必要条件。

**第三阶段：稀疏图上的 LP–Local 固定点**

当图已有证书保证包含所有最优 tours 后，cuOpt 可直接对该稀疏模型产生有效 bound：

\[
\text{LP solve}
\to \text{RC/path penalties}
\to \text{Local elimination}
\to \text{sparser LP}
\to \text{warm restart}.
\]

### 5.4 GPU 1-tree 的安全定位

GPU 1-tree 可以用于：

- 候选边排序；
- Tutte move 排序；
- 已认证稀疏图上的快速辅助下界；
- cuOpt 不可用时的低成本 fallback。

它不应在未经证明的任意稀疏候选图上直接替代完整图 LP 下界，因为限制 1-tree 的边集合可能把最小 1-tree 值抬高，从而失去对完整 TSP 的下界有效性。

---

## 6. Hamilton–Tutte：混合 AND/OR 调度，而不是纯 BFS

原递归语义为

\[
HT(F)=Leaf(F)
\lor
\bigvee_{v\in\mathcal C(F)}
\bigwedge_{H\in\mathcal H(F,v)}HT(F\cup H).
\]

层同步 wavefront 在完整枚举时与 DFS 逻辑等价，但内存与冗余取决于**树级 early-exit**，而不是 `CCelim_tsp_swap` 的成功率。应新增下列 profiling：

- 每个 OR 节点尝试到第几个 Tutte candidate 才成功；
- 每个 AND candidate 在第几个 Hamilton reply 失败；
- 各深度 frontier 宽度；
- 成功证书大小与被投机展开状态数之比；
- cancellation 可节省的状态数。

### 6.1 推荐调度

```text
Algorithm HYBRID-HT(ROOT_BATCH):
    push roots into GPU frontier

    while unfinished roots exist:
        take a bounded wave W, grouped by depth and task shape
        run cheap leaf tests and contracted DP on GPU
        propagate resolved AND/OR records

        for each unresolved state:
            if depth <= d_gpu and bucket_size >= B_min:
                expand a bounded number of ranked Tutte candidates on GPU
            else:
                migrate state to CPU local DFS queue

        cancel descendants of resolved OR/AND records when safe
        compact frontier

    replay only the selected winning subgraph to emit certificates
```

### 6.2 候选排序不应删除

`rank_extra_points` 不仅减少单次评估成本，还控制：

- OR 节点分支数；
- BFS/wavefront 峰值；
- 证书大小；
- 找到早期成功 candidate 的概率。

应将其改为 GPU 批量评分 + warp/block top-k，而不是全宽展开。

---

## 7. GPU 数据面与正确性

### 7.1 图快照

使用 immutable CSR + active/fixed bitsets：

```text
read G_t
parallel prove candidates
CPU/GPU verify witnesses
commit bitsets
construct G_{t+1}
repeat
```

同步快照可能暂时少消边，但不会产生错误消边。

### 7.2 距离

对整数坐标的 EUC_2D/CEIL_2D，优先采用整数平方和 + integer sqrt + 精确取整规则，而不是依赖 CPU/GPU `sqrt(double)` 恰好一致。对每个局部任务先建立小型 pairwise distance matrix，避免 DP 中重复开方。

GEO 等复杂距离类型第一版由 CPU 预计算或逐 witness 复核。

### 7.3 硬件现实

- RTX A5000 与 RTX 4000 Ada 的 compute capability 8.6/8.9 都只有 100 KB shared memory/SM；\(k=11\) 的 DP 接近一 block/SM。
- RTX PRO 5000 Blackwell 官方公开规格确认 48/72 GB GDDR7 ECC 与 1,344 GB/s 带宽，但 L2、INT32 相对吞吐和具体 chip 数字应通过 `cudaGetDeviceProperties`、microbenchmark 与 Nsight 实测，而不应写死为项目假设。
- `usa115475` 在最初 LP reduced-cost 阶段可有 25,009,702 条边；约 30 万边是后续 bootstrap/final sparse graph 的量级。只有后期图可能整体接近大缓存容量，初始阶段不能按“全部常驻 L2”建模。

---

## 8. 融合后的实施路线

### Track A：精确局部 oracle 与 GPU leaf engine

#### A0：复现与 CPU 基线

- 获取 `dpswap.c`、patch 与完整命令；
- exact DP differential tests；
- arena allocation；
- static matching coverage；
- table-driven k-opt；
- 代表性 workload profiling。

**Go/No-Go**：DP 必须 100% 通过局部穷举；在至少三个实例上 `tsp_swap` 时间降低 \(\ge2\times\)。

#### A1：GPU contracted DP

- 按 \(k\) 分桶；
- cost-only + witness replay；
- CPU tree control；
- asynchronous hard-task fallback。

**Go/No-Go**：相对优化后的多核 CPU，leaf engine 至少 \(3\times\)；GPU fallback 率与传输占比可控。

#### A2：GPU k-opt、matching、path validation

- constant reconnect tables；
- static matching coverage；
- fixed-width path encoding；
- exact local distance matrix。

### Track B：cuOpt 与安全 LP 证书

#### B0：LP checkpoint corpus

保存多个 cutting-plane epoch 的：

- rows/columns/nnz；
- CSR/MPS；
- CPU primal/dual；
- omitted-column pricing 信息；
- solve/cut/pricing 时间。

#### B1：cuOpt sidecar

比较 PDLP、barrier、concurrent、warm start；输出未经认证和经认证的 bound/RC 两套数据。

**Go/No-Go**：认证后 bound 质量不能显著劣化；模型构建+H2D+solve 总时间优于 CPU checkpoint。

#### B2：周期性混合后端

仅在模型规模与 row-batch 阈值满足时调用 cuOpt；其余 epoch 留在 CPU simplex。

### Track C：搜索调度与系统集成

#### C0：KH-elim GPU correctness harness

KH-elim 适合验证：

- 图快照；
- 精确距离；
- bitset commit；
- 任务分桶；
- 多 GPU 分片。

它不是主要时间热点的最终解决方案，但风险最低。

#### C1：bounded AND/OR wavefront

先对浅层与大桶启用；保留 CPU DFS 长尾。

#### C2：LP–Local 交替固定点

当 A/B/C 各自通过验证后再集成，避免同时引入数值与组合两类错误。

---

## 9. 修订后的性能目标

必须维护两种基线：

1. **统一硬件 wall-clock**：实际端到端用户时间；
2. **core-hours/GPU-hours**：总计算资源与能耗。

不能把论文中 LP 单核秒与消元 48 核墙钟简单换成 core-seconds 后，再把 14%/86% 作为 wall-clock Amdahl 权重。

### 9.1 建议目标区间

| 阶段 | 相对优化后单核 Local Elimination | 相对强多核 CPU Local Elimination | 完整预处理 E2E |
|---|---:|---:|---:|
| CPU contracted DP + tables | 1.5–3× | 1.2–2× | 1.2–2× |
| CPU tree + GPU leaf engine | 20–100× | 2–6× | 2.5–6× |
| GPU path/k-opt + hybrid wavefront | 80–300× | 4–12× | 4–10× |
| cuOpt + certified pricing + LP–Local 固定点 | — | Local 部分 5–15× | 8–18×，有利实例更高 |

上传方案给出的“相对 48 核 6–17×”可保留为成熟 Local Elimination 的 stretch target；“相对单核 300–800×”只作为吞吐上界研究目标；`97×–295×` 端到端数字在重新建立统一基线前不采用。

### 9.2 必测项目

- `tasks/s` 按 \(k,m,N\) 分桶；
- GPU DP arithmetic intensity；
- shared-memory occupancy；
- frontier expansion/cancellation ratio；
- CPU verifier 吞吐；
- LP model-build/H2D/solve/pricing 分解；
- certified RC candidates / raw candidates；
- 最终 edge count、fixed count、certificate size。

---

## 10. 最终融合架构

```text
CPU incumbent + cut generation
             |
             v
cuOpt PDLP / barrier / concurrent
             |
             v
CPU row-sign mapping + full-column pricing + interval/rational certificate
             |
             v
certified L(y), per-edge penalties, path-system penalties
             |
             v
immutable GPU graph snapshot
             |
             +--> GPU KH/fast elimination
             |
             +--> GPU 2/3/4/5-opt + matching tables
             |
             +--> GPU contracted exact DP
             |
             +--> bounded AND/OR wavefront
                         |
                         +--> CPU DFS / Held-Karp long tail
             |
             v
CPU certificate replay and commit
             |
             v
sparser LP + warm restart
```

系统的安全不变量为：

> **数值 solver 和 GPU search 均可视为不可信候选生成器；只有完整列被计入的严格 LP 证书，或可重放的组合 witness，才允许改变 active/fixed edge sets。**

---

## 11. 近期最优行动序列

1. 补齐并审阅 `dpswap.c` 与 `improve.c.patch`；
2. 在当前 CPU 代码中合入 contracted DP、arena 与静态 matching coverage；
3. 建立局部任务 dump/replay 格式，使 GPU kernel 可以脱离完整搜索独立 benchmark；
4. 首先实现 `EXACT_DP_BUCKET<K=5..9>`；
5. 同步建立 cuOpt sidecar 与完整列定价验证器；
6. 收集树级 early-exit 数据后，再决定 bounded wavefront 的深度与预算；
7. 只有当 leaf engine 与 LP certificate 均稳定后，才实施 LP–Local 交替固定点。

这一路线吸收了上传方案中最强的算法重构，同时保留了混合 CPU–GPU 架构的稳健性，并避免把尚未测量的硬件吞吐与弱下界路线误写成确定的端到端收益。
