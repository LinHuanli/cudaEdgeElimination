# 实现状态（2026-09-02）

| 工作包 | 状态 | 已验证证据 |
|---|---|---|
| M0 仓库与复现 | 完成 | 项目内依赖/构建目录、固定子模块、CPU CI、路径门禁 |
| M1 JV CPU/CUDA 闭环 | 完成（首期 JV 范围） | n=6–12 全最优边检查；pr299 CPU/GPU 哈希一致；proof 重放；compute-sanitizer 0 error |
| M2 cuOpt sidecar | 完成（模型内证书） | cuOpt C API 26.8.0；手算小 LP objective=1；残差为 0；精确下界=1 |
| M3 Concorde 导出 | 完成 | 内容寻址受限 overlay；Concorde graph 目标映射复核；随机 20 点和 pr299 CSR 往返 |
| M3 完整图 exact pricing | 完成（安全下界桥接） | `CCbigguy` 注入；完整图负 reduced-cost penalty；三方哈希；错配拒绝 |
| M3.1 对偶稳定化与边集导出 | 待实现 | pr299 PDLP 完整图界偏弱；尚未导出每边 exact RC/Concorde 消元后边集 |
| M4.1 path-system 组合层 | 完成 | 路径规范化；固定哈希表；368,047 单元 CPU/CUDA 全量差分；`m=6,7` CPU fallback |
| M4.2a CPU k-opt 叶证明 | 完成 | proper 3/4/5-opt `4/25/208` 模板；ElimTSP oracle 差分；`path-kopt-proof-v1` 独立重放 |
| M4.2b CUDA k-opt cost | 完成（CPU 认证候选器） | 批量精确成本矩阵；逐 cell CPU/CUDA 一致；改善 witness 完整重建；memcheck 0 error |
| M4.3a 精确困难叶 | 完成（有界 CPU fallback） | 收缩 forced outside matching；Held–Karp 子集 DP；通用交换 witness 独立重放；block 超限为 unresolved |
| M4.3b1 浅层 HS AND–OR | 完成（研究 API） | `c,d` OR；完整邻边对 AND；嵌套 leaf 重放；CUDA flags 经 CPU 全量差分 |
| M4.3b2 递归 HT 语义与证书 | 完成（CPU 研究 API/CLI） | extra point/end；continuation arena；全局 proof V1；`ht-prove`/`ht-verify` 严格重放 |
| M4.3b3a 混合 GPU wavefront | 完成（研究 API/CLI） | 主机 BFS；CUDA 原子 continuation counters；single/cooperative persistent queue；CPU 全状态差分 |
| M4.3b3b1 GPU path append | 完成（候选器） | point/end 状态内合批；稀疏分量/度数 kernel；CPU 规范子状态逐项认证 |
| M4.3b3b2a1 GPU Hamilton replies | 完成（候选器） | 多中心 count/write；确定性 CSR 区间；CPU 完整列表逐元素认证 |
| M4.3b3b2a2 GPU end replies | 完成（候选器） | 多端点 count/write；空区间/重复 task；CPU 完整边列表认证 |
| M4.3b3b2b1 frontier reply batching | 完成（调度基线） | 可配置 chunk；跨父状态 spans；单状态/批量 V1 proof 逐字节一致 |
| M4.3b3b2b2a frontier path append | 完成（调度基线） | 多父状态稀疏 spans；point-first end 筛选；CPU 规范 child 全量认证 |
| M4.3b3b2b2b1 规范 child edge SoA | 完成（候选器） | CUDA count/write；不可行空 slice；CPU offsets/edges 全数组认证 |
| M4.3b3b2b2b2a frontier leaf batching | 完成（规则首行） | 确定性复杂度桶；跨 leaf cost rows；CPU witness/proof 复核 |
| M4.3b3b2b2b2b1 一般 leaf 游标 | 完成（增量融合） | 分段组合 cursor；3/4/5-opt；预算边界与随机 proof 字节差分 |
| M4.3b3b2b2b2b2a GPU leaf 驻留缓存 | 完成（线程/设备本地） | 精确坐标/模板键；增长型 workspace；命中与字节指标 |
| M4.3b3b2b2b2b2b1 CPU long-tail | 完成（128-cell 基线） | 缓存后交叉点；融合矩阵分流；CPU/CUDA proof 规范计数 |
| M4.3b3b2b2b2b2b2 multi-block continuation | 完成（cooperative 基线） | grid barrier；residency 门禁；512-way AND 跨 block 差分 |
| M4.3b3b2b2b2c HT epoch commit | 完成 | 整批 CPU 重放；V2 内嵌 sidecar；图副本原子提交；旧 V1 兼容 |
| M5 有界全图 HT scan | 完成（单快照 pilot） | 稳定目标切片；预算 unresolved；V2 阶段计时；三路工作签名；V2 原子提交；最优 tour 门禁 |
| M5 中大型调优 | 进行中（JV 三轮 + HT host fast paths） | leaf 哈希复用 + CPU cost row 并行；CPU search `1.867 s` |

## 当前基准结果

pr299 输入 1208 条边；JV 两个 epoch 后保留 1122 条，提交 86 条删除。CPU 与 CUDA 的最终内容哈希均为 `b9b67e9981518177`。这些数字是正确性回归结果，不构成论文性能结论。

M5 JV 正式基准绑定 `cac180f`：pcb3038、rl5915、d15112 分别从 `6883/29143/166499` 条边删除 `179/550/7312` 条，最终哈希为 `90d13888e351df17`、`0174cf46124ce870`、`76e196dd53d887d5`。进程内 CUDA 算法中位数为 `2.396/7.681/74.829 ms`，相对 CPU 为 `8.132×/25.679×/37.426×`；独立 CLI wall 加速为 `0.186×/0.940×/6.139×`。所有运行逐份 CPU 重放并比较输出；pcb3038 的 137,694 最优 tour 还通过 0 缺边门禁。另两实例尚缺本地最优 tour witness，不能标成 tour-checked。

M5 JV 驻留优化绑定 `25590af`：精确比较坐标、边端点和边权后复用静态 device arrays，每轮仍完整上传 active/CSR/witness。三实例 CUDA 算法中位数降为 `1.741/6.639/70.097 ms`，相对 CPU 为 `11.117×/29.788×/40.119×`；所有 timed epochs 均命中静态键和增长 workspace，峰值驻留为 `391148/1517168/8294196 bytes`。最终图哈希和输出 SHA-256 均未改变。

M5 JV 动态 edge-id 优化绑定 `41feceb`：CUDA CSR 不再重复上传 64 位权重，而用 32 位稳定 edge id 读取驻留权重；三实例算法中位数降为 `1.728/6.503/67.651 ms`，相对 CPU 为 `11.194×/30.517×/41.695×`，峰值驻留降为 `336084/1284024/6962204 bytes`。d15112 的同步 H2D/kernel/D2H 中位数为 `2.965/6.628/0.513 ms`；proof、最终图哈希与输出 SHA-256 均未改变。

M5 HT scan pilot 绑定 `cd5ec3e`：pcb3038 的 CPU JV 固定点有 6,704 条边和 6,476 个度数安全目标；最高权重 8-target 切片的 CPU/CUDA 工作签名均为 12,383 states、14,285 replies、9,120 leaf calls 和 5,085 moves，证明并提交相同 2 条边。CUDA/CPU search 为 `33.646/34.103 s`，仅 `1.014×`；最终 6,702 条边、哈希 `fe11f98414b04c0e`，两份 V2 均独立重放且 pcb3038 最优 tour 为 0 缺边。这是功能与资源 pilot，不是显著性能结论。

M5 HT profiling 绑定 `65f9488`：V2 报告把 work graph 拆成 leaf/path/reply 与 host residual，并用 `--leaf-backend` 建立第三条混合路径。相同 8-target clean run 中，CPU/全 CUDA/混合 search 为 `33.987/33.599/33.401 s`；混合仅 `1.018×`。CPU leaf 为 27.160 s（search 的 `79.914%`），Hamilton reply 为 6.245 s（`18.375%`），host residual 只有 0.228 s。三路 proof 均独立重放、边文件 SHA-256 一致且最优 tour 零缺边。

M5 HT bucket fusion 绑定 `b5fde24`：V3 报告加入 leaf frontier/bucket/cost batch 计数，显式融合将 8-target 的 frontier batches 从 500 降到 86、CUDA cost batches 从 1,835 降到 382，cost cells 与 proof 不变；但 clean run 的混合/fused search 为 `32.672/32.905 s`，交错重复也仅约 `1.002×`。因此默认保持关闭，并把下一门禁改为 leaf 内部 cost/cursor/certification 计时。

M5 HT leaf 子阶段画像绑定 `fe99f38`：V4 报告和 V5 summary 把 leaf 拆成 setup、cursor prepare、cost、scatter、consume、apply 与 verifier。相同 8-target clean run 中，混合 leaf 为 `26.489 s`，其中 setup `5.864 s`（`22.139%`）、GPU cost `0.584 s`（`2.204%`）、CPU cursor consume `19.695 s`（`74.352%`）；四路工作签名、最终边集、proof 重放与最优 tour 门禁均一致。下一步先缓存只由 path count/k 决定的不可变组合表，再细分 CPU completeness。

M5 HT 不可变表缓存绑定 `589bca0`：线程安全延迟缓存复用 path-count matching catalog 和 `k=3/4/5` reconnect templates，不缓存任何图相关状态。相对 `fe99f38`，CPU leaf/search 从 `27.231/34.095 s` 降至 `17.760/24.138 s`（`1.533×/1.413×`）；混合 setup 与 cursor consume 分别下降 `70.002%/23.278%`。四路工作签名、51,309,996 cost cells、最终边集、proof 重放和受保护 tour 不变；剩余混合 consume 为 `15.111 s`，是下一画像对象。

M5 HT completeness 画像绑定 `ee4f3aa`：V5 report/V6 summary 显示 727,635 个 CUDA cost rows 中 726,648 个（`99.864%`）进入 CPU 全模板 fallback，51,179,094/51,309,996 cells（`99.745%`）被通用重连器重新穷举。混合 consume 中 candidate/fallback 为 `0.057/15.107 s`，fallback 占 `99.033%`。三条 CUDA leaf 路径计数、proof、图和 tour 均一致；下一步以 CPU 独立精确矩阵替换昂贵的通用无改善认证，不降低 completeness。

M5 HT CPU 精确矩阵认证绑定 `48d68dc`：固定数组 scorer 对 CUDA cost matrix 的 51,309,996 cells 全部进行独立 CPU 整数认证，差异立即失败关闭；987 个严格改善模板仍由通用 CPU 路径重建完整 witness，旧 completeness fallback 降为 0。相同 8-target clean run 中，hybrid cursor consume 从 `15.255 s` 降至 `0.126 s`，leaf/search 为 `5.011/11.586 s`，相对 CPU scalar 达到 `3.571×/2.097×`；all-CUDA/fused search 分别为 `11.673/11.499 s`。四路工作签名、最终图哈希 `fe11f98414b04c0e`、proof 重放和 pcb3038 最优 tour 门禁完全一致。下一步让 CPU backend 复用相同矩阵 fast path，建立公平基线。

M5 HT CPU matrix 公平基线绑定 `9fa301d`：显式 CPU leaf 进入相同增量 cursor，并修复 cost block 尾部未消费 rows 被计入 proof 的既有规范计数问题。随机 direct/auto/CPU-matrix 三路现与 CPU scalar proof 逐字节一致。8-target clean run 中 CPU matrix leaf/search 为 `4.491/11.015 s`，优于 all-CUDA `4.848/11.730 s`、hybrid `5.034/11.430 s` 和 fused `4.974/11.594 s`；四路 51,309,996 cells、727,635 consumed rows、987 个候选、最终图和 tour 均一致。同步全量 CPU 认证下 GPU 没有净 leaf 加速，下一画像转向占 CPU search `55.19%` 的 Hamilton reply。

M5 HT Hamilton reply 主机优化绑定 `f12c181`：批 API 只验证一次图，同批重复 center 复用规范回复，并把 2-opt quick filter 提升为每邻边一次。8-target 的 72 batches/27,598 逻辑 centers 只实际枚举 1,395 个 batch-unique centers 和 11,515 个邻边对，仍输出相同 245,965 replies。CPU reply 从 `6.079 s` 降至 `0.019 s`（约 `321×`），CPU search 从 `11.015 s` 降至 `4.870 s`（约 `2.26×`）；all-CUDA/hybrid/fused search 为 `5.562/5.345/5.354 s`。四路工作图、51,309,996 leaf cells、最终图、proof 重放和 tour 门禁均一致；下一瓶颈是占 CPU search `91.38%` 的 leaf。

M5 HT leaf setup 画像与快照哈希复用绑定 `f71b472/c968b01`：V8 report/V10 summary 将 setup 拆为 proof 初始化、coverage 扫描和 cursor 构造，并要求四路 cursor 数一致。基线的 9,891 个 cursor 中，proof 初始化占 setup `92.57%`，根因是每个 leaf state 都重算同一不可变 graph 哈希。改为每 batch 计算一次后，CPU setup/leaf/search 从 `1.761/4.497/4.915 s` 降至 `0.232/2.981/3.398 s`，分别加速 `7.59×/1.51×/1.45×`。四路 51,309,996 cells、proof、最终图和 tour 均不变；下一瓶颈是占 CPU leaf `78.67%` 的精确 cost matrix。

M5 HT CPU cost row 并行与 leaf 桶融合绑定 `34cf918/63133c7/623f167`：8,192 cells 以上按 task row 静态分片，最多 8 线程；无 OpenMP、小 batch 和单线程保持串行。clean 1/8-thread A/B 使 CPU certify/leaf/search 从 `2.340/3.029/3.444 s` 降至 `0.701/1.442/1.867 s`。V12 再加入 CPU 融合为第五路；pcb3038 leaf/search 为 `1.149×/1.113×`，rl5915/d15112 leaf 为 `1.085×/1.180×`、search 均为 `1.016×`。三份锁定公开 tour 及三实例五路 proof、规范工作量和最终图全部通过。CLI 现对 CPU leaf 默认融合，auto/CUDA 默认不变，显式 0/1 可覆盖。大实例下一瓶颈是 path-append。

M5 HT path-append 画像与稀疏规范化绑定 `ba15919/b551a2e`：V10 report/V13 summary 将包含式总量拆为 parent prepare、child normalize、child edges、CUDA evaluate 和 compare。内部 fast path 只为实际路径节点构造有序邻接，通用 dense 实现继续独立 proof 重放；308-task 单元差分逐项比较完整规范结果。三实例 path-append 加速 `5.503×/9.538×/25.561×`，search 加速 `1.115×/1.679×/3.133×`；五路规范 proof、最终边和受保护 tour 全部不变。下一瓶颈是大实例 host-build residual。

M5 HT 根 child 规范化排除实验绑定 `6b2b8ad`：V11/V14 记录 72/151/647 次根 child dense 规范化，CPU-fused 仅为 0.773/2.174/28.642 ms，占三实例 host residual `0.406%/0.562%/2.605%`。未为低占比路径扩大共享 fast-path API；下一画像目标是逐 frontier state 的全维 point-candidate 扫描。

M5 HT point-candidate 画像与 Top-K 优化绑定 `5944476/4e4f8e3`：V12/V15 记录 state scans、checked/ranked/selected 节点、扫描与排序时间；严格全序的 `partial_sort(top 25)` 将三实例排序从 `137.639/298.303/858.648 ms` 降至 `11.901/14.314/28.784 ms`，CPU-fused search 加速 `1.086×/1.580×/1.552×`。三实例五路活动边、工作签名、规范 proof 和受保护 tour 全部不变。下一步验证同一 target 的静态评分顺序复用。

M5 HT point-candidate 静态次序缓存绑定 `ea85ffa`：每个 target 只计算一次中点评分和严格全序，逐 state 用 generation marks 过滤已有路径节点，同时保持完整维度 checked 与原 ranked 计数。相对 Top-K，三实例 scan 再加速 `188.364×/238.242×/293.326×`，CPU-fused search 加速 `1.045×/1.176×/1.131×`；相对全量排序画像的累计 search 加速为 `1.135×/1.858×/1.756×`。三实例五路规范计数、边、工作签名、proof 和 tour 全部不变。下一步转向跨目标 leaf 准备数据与调度共享。

M5 HT leaf proof 批内快照绑定复用绑定 `0530ff4`：同一同步 k-opt batch 内生成和复核使用一次入口 graph hash；公开 `VerifyPathSystemKOptProof`、HT 最终 verifier、scan 即时复核和 epoch 重放仍独立计算。三实例 leaf proof verify 加速 `8.882×/17.591×/27.449×`，CPU-fused search 加速 `1.037×/1.355×/1.618×`。三实例五路规范工作、边、proof 和 tour 全部不变。下一步复用同一 wavefront 的 snapshot binding。

cuOpt 手算 LP：状态 `OPTIMAL`，objective/dual objective 均为 `1`，primal violation 与 reduced-cost residual 均为 `0`，定点模型下界为 `16777216/16777216`。

Concorde 随机 20 点 epoch：25 行、43 列；QSopt 与 cuOpt 模型目标均为 `88`。cuOpt primal violation 为 `4.44e-15`，reduced-cost residual 为 `1.57e-14`；完整图 exact lower bound 为 `87.3932819641`，上界为 `88`。

pr299 Concorde epoch：454 行、888 列、8561 个非零元；cuOpt 状态 `OPTIMAL`，模型目标 `48187.777777780764`，primal violation `8.37e-11`，reduced-cost residual `4.87e-10`。完整图 exact lower bound 为 `43977.2693797`，合法但较弱；这说明后续需要对偶稳定化/迭代补列，而不是跳过负 reduced-cost penalty。

路径兼容表：`m=5` 为 45,360 字节，生成器哈希 `f6bccacc5c1fa84f`；362,880 个 `m=5` 单元和全部较小表均通过 CPU/CUDA 差分，CUDA memcheck 为 0 error。`m=6,7` 不建立完整表，按契约使用 CPU 直接判定。

CPU k-opt 叶证明：自动生成的 proper 3/4/5-opt 模板数为 `4/25/208`，每阶 2,000 个阈值定向随机矩阵与固定子模块 `swap.c` oracle 一致；成功 witness、全 outside coverage、预算 unresolved 和篡改拒绝均有回归。

CUDA k-opt cost：按删除集合与 proper template 形成精确成本矩阵，CUDA 全矩阵与独立 CPU 整数矩阵逐 cell 一致后才进入 consumer；非改善 completeness 由 CPU 矩阵认证，严格改善候选再由通用 CPU 路径重建并验证 witness。真实 HT 8-target 的 51,309,996 cells 全部认证，CUDA memcheck 为 0 error；相对 CPU scalar 的 hybrid search 为 `2.097×`，但仍需 CPU matrix 公平基线。

CPU 精确困难叶：将每条 forced outside edge 收缩为可双向访问的 block，其余节点为 singleton block；固定一个 block 的方向消除无向反转对称后，以 Held–Karp 子集 DP 穷举所有 block 次序和方向。比较时消去两条巡回共有的 outside 成本，并禁止候选重新使用 required path edge。成功结果转换为任意 `k>=2` 的交换 witness，再交给同一独立 verifier；默认关闭，启用时最多 18 个 block，内存不足或超限均返回 `unresolved`。60 组随机 7 点、每组两个 outside 的结果与直接 Hamilton 巡回枚举最优值一致，并覆盖 7-opt proof 往返。CPU Debug/ASan、CPU Release 与 CUDA Release 全量回归通过，GPU 2 上 k-opt memcheck 为 0 error。

浅层 HT：固定 `c,d` move 后，重新枚举两个中心所有通过严格 2/3-opt 快速筛选的邻边对，并要求其笛卡尔积中的每个 reply 都有 path-infeasibility 或完整 path-system leaf proof。固定 8 点实例覆盖 30 个非空 replies；删除 reply、篡改 move/snapshot/leaf 均被 verifier 拒绝。CUDA 在稀疏 32 点 EUC/CEIL 批次上输出与 CPU 完全一致的 `c,d` flags，memcheck 为 0 error。

递归 HT：CPU DFS 实现与后续 wavefront 相同的 `Leaf(F) OR ∨move ∧reply HT(F∪reply)` 真值。未解决状态可选择未出现在路径系统中的 point，或选择当前路径 endpoint；每个 move 必须记录其完整活动图 replies。成功子树保存为只向后引用的扁平 continuation arena，独立 verifier 从目标边重新规范化每个子状态并拒绝环、共享 child、遗漏 reply 和未引用节点。`CUDAEE_HT_RECURSIVE_PROOF_V1` 嵌入现有 path-k-opt V1 叶证明并严格拒绝非法计数、枚举值和尾随字段。固定 point/end 递归实例均由 8 点完整巡回穷举额外确认目标边不属于任何最优巡回；depth/budget fail-closed 与 proof 篡改已有回归。`ht-prove`/`ht-verify` 已覆盖固定实例的文件级端到端 CTest；未解决写入 `proven=0` 并返回退出码 3。DFS 调度本身仍在 CPU，GPU 版本使用下一段的独立 wavefront 路径。

混合 wavefront：主机 BFS 为每个状态先跑 leaf，再生成有界 point/end OR moves 及其完整 replies；所有 child 只指向下一层。CPU 从后向前得到规范状态真值；CUDA 从 leaf/vacuous/无 move 终态队列开始，每个完成 child 原子更新 move 的 `remaining_children/failed`，成功 move 立即完成父状态，失败 move 递减父状态的 `remaining_moves`，最后一个失败 move 才宣告 OR 失败。single-block 或 cooperative multi-block persistent kernel 在设备端冻结并消费动态队列批次，要求每个状态恰好入队一次；队列溢出或提前停滞均失败。CUDA/CPU 任一状态不同即返回 `invalid`。成功时仅复制第一个成功 move 的子树到既有 continuation arena，再运行完整 proof verifier。

GPU path append：规范父路径展平为 `(node,component,degree)` 与规范父边 spans，一个线程检查一个 point/end task。point 中心必须是新节点；两个连接点度数均小于 2 且不能来自同一分量。end 必须从现有端点出发，另一端只能是新节点或其他分量端点。CUDA count/write 为每个可行 task 输出父边加新增边的严格排序 CSR slice；CPU 对每项仍执行 `NormalizePathSystem`，独立重建 flags、offsets 和全部边，只有全数组相等才标记设备 child 已认证。固定双父状态 11-task 表得到 27 条 child edges；不可行 task 保持空 slice。工作图仍只使用 CPU 规范 child，这些是正确性样例，不是性能结论。

GPU Hamilton replies：CUDA 对每个中心先 count，再由主机建立 `uint64_t` 前缀区间，最后按排序 CSR 的确定顺序 write。整数平方根实现与 CPU 的 `EUC_2D`/`CEIL_2D` 精确边界一致；根 `c,d` 两中心合批，递归 point 的全部候选中心在单个父状态内合批。CPU 始终重新枚举完整 offsets 和 reply 列表并逐元素比较，只有 CPU 列表进入工作图。12 点完整图覆盖两种距离、重复中心和 CPU-only 回退；固定 point CLI 为 9 batches、16 centers、46 surviving pairs，证明规模和重放结果保持不变。这些仍是正确性指标，不代表端到端加速。

GPU end replies：同一父状态的全部路径 front/back 端点可合为一个 batch；每个线程从端点排序 CSR 排除内部邻点，经主机 `uint64_t` 前缀和后把规范活动边写入独占区间。CPU 重新扫描完整 offsets/edges，只有 CPU 列表进入候选排序和工作图。9 点完整图覆盖相反 endpoint 方向与重复 task，稀疏链覆盖 degree=1 的零长度区间；固定 recursive-end proof 同时要求 CPU 与全 CUDA wavefront 成功并由同一 V1 verifier 重放。

Frontier reply batching：当前层按最多 256 个父状态切成资源 chunk，point centers 与 end tasks 分别展平并一次生成，再按记录的 spans 恢复每个父状态候选。child 仍按原 state/candidate/reply 顺序物化，固定 point 实例的 `N=1` 与 `N=256` V1 proof 逐字节一致；Hamilton batches 从 9 降为 4。

Frontier path append：一个 chunk 的可尝试 point tasks 先共用一次多父状态 batch；根据 CPU 认证 flags 排除 vacuous-success states 后，才生成 end replies 和 end append batch。固定 point 实例的 path-append batches 从 9 降为 3，tasks 保持 84；end batches 从 2 降为 1，工作量在两种 chunk 大小下均为 8 tasks/48 edges。全局预算和 child 写入仍按原 state/candidate/reply 顺序执行，V1 proof 逐字节一致。

规范 child edge SoA：父状态新增严格排序的 edge spans。第一阶段输出可行 flags 与 child edge counts，主机建立 `uint64_t` offsets，第二阶段在 task 私有 slice 中复制父边、追加 reply 边并确定性排序；重复边、区间不一致或 CPU 全量差分失败均拒绝批次。固定 recursive-point 全 CUDA wavefront 为 3 batches/84 tasks/172 child edge records，34-state 工作图和 4 节点 V1 proof 保持不变并通过独立重放。

Frontier leaf batching：当前 reply chunk 先按 `(depth,path_count,node_count,max_k,incoming reply bucket)` 分桶。每轮把各 proof 首个未覆盖 outside 的同 k cost blocks 合并，CUDA matrix 先经 CPU 全矩阵逐 cell 认证，只有严格改善 cell 按原顺序执行 `TryReconnect`、inside coverage 与独立 proof verifier。固定 recursive-point 的 34 rows/136 cells 从 34 个 cost batches 降为 6 个，最大 batch 16；`N=1/256` 的 4 节点 V1 proof 逐字节一致。

一般 leaf 游标：每个 path/outside 保留当前 k、组合索引、统计计数和至多一个 pending block；主机只在 CPU 消费当前 block 后推进。固定双 7 点完整 3/4/5-opt 穷举把 50 tasks/2660 cells 合为 13 batches，预算 3 的 `2+1` block 停止点也与 scalar 一致；12 组随机图、三种路径布局和多种 k/预算/batch size 的 CPU/CUDA V1 proof 全部逐字节差分。

GPU leaf 驻留缓存：每个主机线程在首选设备上复用整数坐标、独立 3/4/5-opt 模板和增长型 task/cost buffers。距离快照逐坐标比较完整 kernel 依赖，模板同时比较生成器哈希与完整数组；owner-device buffer 可在 epoch 边界显式释放。固定 recursive-point 的 6 个 CUDA leaf batches 只有首批上传快照/模板，记录 5/5 次命中、4 次 workspace 命中和 1468-byte 驻留峰值，proof 保持不变。

CPU long-tail：项目内稳态基准在 RTX 4000 Ada 上定位 3-opt 64/256、4-opt 25/100 与 5-opt 208 cells 的 CPU/CUDA 交叉区间，默认用 128 cells 分流融合后的 `auto` 矩阵。31 个 3-opt leaf（124 cells）走 CPU、32 个（128 cells）走 CUDA；规范化的模板枚举计数使显式 CPU/CUDA、阈值两侧和不同 frontier batch size 都保持 V1 proof 字节一致。固定 recursive-point 的 auto leaf 六批全部进入 CPU long-tail，proof 与显式全 CUDA 相同。

Multi-block continuation：cooperative kernel 以 grid barrier 冻结并消费完成队列批次，只有 `queue_tail==state_count` 才正常终止；自动 block 数不超过 kernel 的实际 cooperative residency，显式越界闭门失败。固定 truth table 的 single/2-block 状态相同；512 个 child 跨两个 block 汇入同一 AND move 时，单失败和全成功真值均正确，513-state 自动模式选择 3 blocks。固定 recursive-point 显式 2 blocks 的 34-state 数组经 CPU 全量认证，V1 proof 与 single block 逐字节一致。

HT epoch commit：`ht-commit` 可重复接收同一不可变快照上的 recursive HT sidecar，先逐份 CPU 重放，再按 `(u,v,serialized-proof)` 规范化重复目标，并在图副本上执行共用最小度门禁与 CSR 重建。实际含 HT 删除时写自包含 `CUDAEE_PROOF_V2`，outer record 唯一引用 inner HT V1；通用 `verify` 检查绑定、规范顺序、完整证明、最终哈希后才发布重放图。固定 recursive-point 的两份等价 sidecar 从 28 条边提交 1 条，哈希由 `d7bfbec67ffc9a66` 变为 `78ce8b9a9dc29473`；坏 sidecar 混入时整批零修改。JV-only 仍写原 V1。

## 安全边界

`gpu-eliminate` 的自动候选器目前仍只实现 JV；HT 可使用显式 `ht-prove -> ht-commit`，或由 `ht-scan` 在一个不可变快照上扫描有界目标切片并原子提交。后者尚未接入 JV/LP 多 epoch orchestrator，也没有跨目标 GPU 工作图；pcb3038 pilot 不能外推为完整 Local Elimination 加速。`lp-solve` 始终不修改图。Concorde 桥接已能产生完整图安全下界，但测试 wrapper 使用 `-B`，尚不输出消元边集；M3.1 仍为 pending。仍严禁从未完整验证的局部结果、过期 HT sidecar 或 cuOpt 浮点 reduced cost 直接构造删除记录。
