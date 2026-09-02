# 路线图与门禁

## M0：仓库与复现基线

交付目录、Git 子模块、CMake presets、依赖固定脚本、文档和 CPU CI。完成条件：全新 checkout 不写出仓库，CPU build/test 一条命令通过。

## M1：JV 可验证 GPU 闭环

交付 TSPLIB/edge 读取、CSR、整数距离、CPU JV、CUDA JV、proof/verify、epoch 提交。门禁：

- n=6–12 穷举实例零错误删除；
- 固定快照 CPU/GPU 已验证候选一致；
- pr299 重放输出一致；
- ASan/UBSan 与 compute-sanitizer 无错误；
- 不支持输入保留全部边并非零退出。

## M2：cuOpt LP sidecar

交付 epoch 读写、C API solve、残差、对偶裁剪、定点精确模型下界。门禁：手算小 LP 目标与证书一致；失败状态不能产生删除授权；重复运行的证书分子/分母一致。

## M3：Concorde 完整图精确定价

交付受限 overlay 构建、QSopt 模型导出、行/列映射、`CCbigguy` 注入与 exact pricing。门禁分两层：

- M3 安全桥接：pr299 上 cuOpt 浮点对偶经完整精确定价得到合法下界；人为错配哈希被拒绝；已完成。
- M3.1 删除强度：迭代补列或稳定化后与原 QSopt 路径的安全删除结果等价，并导出可比边集；待完成。

## M4：HS `c,d` 与路径兼容

M4.1 组合层已完成：CPU 路径规范化与匹配穷举、`m<=5` GPU 表查找、固定生成器哈希，以及 `m=6,7` CPU fallback。全部小表单元由独立 CPU oracle 差分，GPU 结果只在 CPU 接受后返回。

M4.2a CPU 叶规范器已完成：组合生成 proper 3/4/5-opt 模板，与固定 ElimTSP `swap.c` oracle 差分，并生成绑定快照/路径/兼容表哈希的 `path-kopt-proof-v1`。

M4.2b 批量 CUDA k-opt cost 候选器已完成：GPU 输出完整 template 成本矩阵，并与独立 CPU 精确整数矩阵逐 cell 比较；只有严格改善模板由 CPU 重建完整 witness。非改善结论来自 CPU 全矩阵认证，GPU 不承担 completeness 授权。

M4.3a 有界困难叶 CPU fallback 已完成：收缩 forced outside matching 后用 Held–Karp 子集 DP 精确求解局部巡回，找到的任意阶交换由通用 verifier 重放。默认禁用，最多 18 个 block；规模或内存超限保持 `unresolved`，不会被解释为无改善。

M4.3b1 浅层 HS 根证明已完成：确定性生成 `c,d` OR moves，对固定 move 完整枚举两个中心的 Hamilton 邻边对笛卡尔积，并逐叶重放；CUDA `c,d` flags 返回前与 CPU 逐项核对。

M4.3b2 递归语义与全局证书已完成：CPU DFS 枚举 extra point/end OR moves 和每个 move 的完整 Hamilton replies；成功子树转为扁平 continuation arena，并用 `recursive-ht-proof-v1` 连同嵌套 path-k-opt 叶独立重放。资源上限只返回 `unresolved`。

M4.3b3a 混合 wavefront 已完成：主机按层生成完整有界 AND–OR 工作图；CUDA 用 `remaining_children/failed/remaining_moves` 原子 continuation counters 从已完成叶队列向根传播；single-block 基线及后续 cooperative multi-block persistent kernel 都在设备端消费完整动态队列，不逐轮返回主机。返回前 CPU 对全部状态逐项复算，只从成功状态提取既有 V1 continuation arena。DFS 与 wavefront 在固定 shallow/point/end 实例上均产生可验证证明。

M4.3b3b1 path-append 候选器已完成：同一父状态的 point replies 合为一个 batch、end replies 合为另一个 batch；CUDA 以稀疏 `(node,component,degree)` 记录判断度数冲突与同分量成环，CPU 对每个 task 运行完整规范化并逐项比较，同时返回真正用于建图的规范子状态。

M4.3b3b2a1 Hamilton reply count/write 已完成：根 `c,d` 在一个 batch 内枚举两个中心，递归 point 在每个父状态内枚举全部候选中心；CUDA 以 count、主机前缀和、write 两阶段生成确定性 CSR reply 列表，CPU 完整枚举并逐元素复核后才用于建图。

M4.3b3b2a2 end reply count/write 已完成：同一父状态的所有路径端点合批，排除各自路径内部邻点后按 CSR 顺序写出规范活动边；空 reply 区间、重复 task 和正反 endpoint 均由 CPU 完整列表认证。

M4.3b3b2b1 frontier reply batching 已完成：按可配置父状态数切分当前层，每个 chunk 分别把 point centers 和 end tasks 展平为一次生成调用，再按原 state/candidate/reply 顺序回填；chunk 大小不裁剪搜索。固定实例的 V1 proof 与单状态 chunk 逐字节一致。

M4.3b3b2b2a frontier path-append batching 已完成：一个 chunk 的全部 point tasks 共用多父状态稀疏 batch；取得 point flags 后排除 vacuous-success states，再批量生成和检查 end tasks。CPU 仍逐 task 规范化，child 按原顺序物化。

M4.3b3b2b2b1 规范 child edge SoA 已完成：CUDA 为每个可行 path-append task count/write 完整规范边集；不可行 slice 为空，主机前缀和使用 `uint64_t`。CPU 从完整 `NormalizePathSystem` 结果独立重建 offsets/edges 并逐元素认证，工作图仍只使用 CPU child。

M4.3b3b2b2b2a frontier leaf batching 已完成：状态按 `(depth,path_count,node_count,max_k,incoming reply bucket)` 确定性分桶；`max_deletion_sets=1` 时同轮、同 k 的首个 cost rows 跨状态融合。CUDA 矩阵经 CPU 全矩阵认证，改善候选再经 witness verifier；`N=1/256` 的完整 V1 proof 逐字节一致。

M4.3b3b2b2b2b1 一般 leaf 游标已完成：每个 path/outside 只生成当前 `cost_batch_size` block，同轮同 k 跨状态融合；CPU 消费后才推进下一组合。无界预算不预展开，3/4/5-opt、预算中断和随机路径的 batch/scalar proof 字节一致。

M4.3b3b2b2b2b2a GPU leaf 驻留缓存已完成：每主机线程/设备保留精确坐标快照、独立 3/4/5-opt 模板与增长型 task/cost workspace；完整键比较防止错误复用，命中和驻留字节均进入 wavefront 指标。

M4.3b3b2b2b2b2b1 GPU/CPU long-tail 已完成：缓存后微基准确定默认 128-cell 阈值，`auto` 按融合矩阵规模分流；CPU/GPU 路由使用规范模板计数，阈值两侧 V1 proof 字节一致。

M4.3b3b2b2b2b2b2 cooperative multi-block continuation 已完成：冻结 queue batches 通过 grid barrier 传播，自动 block 数受实际 residency 约束；512-way AND 跨 block 真值、single/multi-block 状态和完整 V1 proof 均一致。

M4.3b3b2b2b2c HT epoch commit 已完成：同一不可变 snapshot 上的多个递归 proof sidecar 先整批 CPU 重放，再按规范目标和最小度门禁在图副本上原子提交；自包含 `CUDAEE_PROOF_V2` 内嵌原 HT V1 并由通用 `verify` 重放。坏/旧 sidecar 使整批零修改，JV-only 输出继续保持 V1。

## M5：中大型评测与优化

依次 pcb3038、rl5915、d15112，定位 CSR 构建、带宽、分歧和复核瓶颈。图重排、多流、多 GPU 都必须在 M1–M4 证书不变量不变的前提下实验。

JV 基线与第三轮优化已完成：三实例各预热一次、计时五次，CPU/CUDA 输出逐字节一致且每次 proof 均由独立 CPU 进程重放。规范边 CSR 免排序、跨 epoch CUDA 驻留后，动态 CSR 再以 32 位 edge id 引用静态边权；进程内算法中位数加速达到 `11.194× / 30.517× / 41.695×`，驻留降为 `0.321/1.225/6.640 MiB`。分段计时确认 d15112 的 H2D/kernel/D2H 为 `2.965/6.628/0.513 ms`。独立 CLI 仍受 context/I/O 和共享节点抖动支配。`pcb3038` 另通过成本 137,694 的受保护最优 tour 门禁。

有界全图 HT 调度 baseline 已完成：`ht-scan` 在同一不可变图上按稳定权重/端点顺序选择显式切片，逐目标 wavefront 搜索，成功 sidecar 即时 CPU 重放后再整批复核和原子提交。pcb3038/JV 固定点的 8-target pilot 中，CPU/CUDA 工作签名一致，均证明并提交 2 条边；CUDA/CPU 搜索为 `33.646/34.103 s`，只有 `1.014×`。这说明下一瓶颈是跨目标融合与主机建图/认证，不是继续宣传单 kernel 加速。

阶段画像与混合后端基线也已完成：V2 报告显示 CPU search 的 `79.914%` 在 leaf、`18.375%` 在 Hamilton reply，纯 host build residual 仅 `0.670%`。解耦 `--leaf-backend` 后，CPU 小候选器 + CUDA leaf 的混合 search 为 33.401 s，相对 CPU 33.987 s 仅 `1.018×`；下一瓶颈明确位于 leaf batching/certification。

同 target 跨复杂度桶融合已完成受控实验：frontier batches `500 -> 86`、CUDA cost batches `1835 -> 382`，但 clean run 的 search 反而由 32.672 s 增至 32.905 s，交错重复也只有约 `1.002×`。开关默认关闭；在跨目标重构前先拆分 leaf 内部 cost/cursor/CPU certification 时间。

leaf 子阶段画像已完成：8-target 混合 leaf 的 GPU cost 仅占 `2.204%`，CPU cursor consume 占 `74.352%`，path proof/cursor setup 占 `22.139%`。因此跨目标 kernel 合批暂缓；先复用只依赖 path count/k 的不可变 matching/reconnect 表，再在不削弱 CPU completeness 契约的前提下细分并优化 consume。

不可变 matching/reconnect 表缓存已完成：相同 8-target 协议下 CPU search 从 34.095 s 降至 24.138 s，四路 leaf 均约 `1.50×` 加速，proof 与图结果不变。缓存后混合 cursor consume 占 leaf 约 `85.45%`；下一门禁是区分候选 CPU 重建与完整 fallback，而不是继续合并只占约 3% 的 GPU cost。

CPU 精确矩阵认证与公平基线均已完成：固定数组 scorer 逐 cell 复算 CUDA 成本，任何差异失败关闭；CPU backend 也复用相同 batch cursor。相同 8-target clean run 的 51,309,996 cells 在四路全部获 CPU 认证，旧 fallback 为 0。CPU matrix leaf/search 为 `4.491/11.015 s`，快于 hybrid 的 `5.034/11.430 s`；旧 CPU scalar 得出的 GPU `2×` 加速已被公平基线否定。当前最大阶段转为 Hamilton reply（CPU search 的 `55.19%`），下一切片先做主机细分画像。

Hamilton reply 主机优化已完成：batch 入口只验证一次图，重复 center 复用规范回复，每邻边的 2-opt 条件只计算一次。8-target 的 27,598 个逻辑 centers 在 batch 内归并为 1,395 次枚举，CPU reply `6.079 s -> 0.019 s`，CPU search `11.015 s -> 4.870 s`。当前 leaf 占 search `91.38%`；下一步拆分 setup 并评估 CPU cost task 并行化。

leaf setup 画像与快照哈希复用已完成：同一 8-target 协议中，9,891 个 cursor 的 setup 有 92.57% 来自每个 leaf state 重算不可变 graph 哈希。改为每个 leaf batch 计算一次后，CPU setup 从 1.761 秒降至 0.232 秒，leaf/search 从 `4.497/4.915 s` 降至 `2.981/3.398 s`；proof、规范工作计数、最终图和受保护 tour 均不变。当前优先级转为占 leaf 78.67% 的 CPU 精确 cost matrix。

CPU 精确 cost row 并行已完成：8,192 cells 以上按 task row 静态分片，最多使用 8 个可用线程；小 batch、无 OpenMP 和显式单线程均走相同串行实现。同 commit 的 1/8-thread A/B 中，CPU certify/leaf/search 从 `2.340/3.029/3.444 s` 降至 `0.701/1.442/1.867 s`。四路规范矩阵、proof、图和 tour 不变；下一步验证 CPU backend 与 leaf bucket fusion 的组合收益。

CPU leaf bucket fusion 组合实验与多实例门禁已完成：V12 增加 CPU 融合为第五路，pcb3038 leaf/search 从 `1.421/1.841 s` 降至 `1.236/1.655 s`。锁定公开最优 tour 后，rl5915/d15112 的 CPU 融合 leaf 为 `1.085×/1.180×`，search 均为 `1.016×`。三实例五路 proof、规范工作量、最终图和 tour 相同。CLI 因此对 CPU leaf 默认融合，auto/CUDA 不变；下一步先画像大实例中占 search 45.76%/70.92% 的 path-append。

path-append V10/V13 画像和稀疏规范化已完成：内部 fast path 按实际节点构造有序邻接，通用 dense 实现继续独立重放 proof。三实例 path-append 获得 `5.503×/9.538×/25.561×`，search 获得 `1.115×/1.679×/3.133×`；五路规范 proof 和最终边逐字节不变。rl5915/d15112 下一瓶颈是占 work graph 约一半的 host-build residual，先画像后优化。

根 child dense 规范化已由 V11/V14 排除：三实例只占 host residual `0.406%/0.562%/2.605%`，不值得扩大共享 fast-path API。下一步计量 `BuildPointCandidateNodes` 的 frontier 状态数、全维节点检查量和排序成本。

point-candidate V12/V15 画像、Top-K 与静态次序缓存已完成：target 级严格全序配合 state generation marks，使三实例 point scan 相对 Top-K 再加速 `188.364×/238.242×/293.326×`，累计 CPU-fused search 相对全量排序画像加速 `1.135×/1.858×/1.756×`。五路规范计数、活动边、工作签名和 proof 逐字节不变。该 host 路径已不再是主要矛盾，下一切片转向跨目标不可变 leaf 数据与调度共享。

leaf proof 批内快照绑定复用已完成：生成器内部逐成功 proof 的整图哈希改为复用同一同步 batch 的入口哈希，公开 verifier 与全部 HT/epoch 独立重放仍自行绑定图。三实例 leaf verify 加速 `8.882×/17.591×/27.449×`，CPU-fused search 加速 `1.037×/1.355×/1.618×`。下一步把相同只读 snapshot binding 上移到同一 wavefront 的多个 leaf batches。

wavefront leaf 快照绑定复用已完成：只能从实际 graph 对象构造的内部强类型 binding 在一次只读 wavefront 内供全部 leaf batches 使用，对象错配立即拒绝；公开生成器和所有独立 verifier 仍自行哈希。rl5915/d15112 的 proof init 加速 `44.302×/99.936×`，search 加速 `1.063×/1.028×`；pcb3038 端到端处于噪声范围。新画像显示 d15112 的 path child normalize 为 `180.448 ms`，下一切片以 sparse/dense 双 oracle 门禁增量 parent append。

path-append child 增量规范化已完成：batch 入口仍用通用 sparse 规范器认证 parent，生产 child 只合并新增边触及的至多两个规范链；992 个合法 point/end tasks 与独立 dense 规范器逐项比较完整结果和失败原因。三实例 child normalize 加速 `8.511×/9.752×/8.179×`，CPU-fused search 加速 `1.038×/1.220×/1.244×`，五路 proof、边和 tour 不变。下一热点是 d15112 Hamilton reply 中 `121.818 ms` 的重复 graph validation。

wavefront 图验证绑定复用已完成：强类型 binding 构造时完整验证 graph，c,d/Hamilton/end bound APIs 每次检查对象身份，公开 APIs 与独立 verifier 仍自行验证。三实例 Hamilton validation 加速 `305.600×/583.206×/3388.723×`，CPU-fused search 加速 `1.017×/1.146×/1.411×`；五路 proof、边和 tour 不变。下一切片把相同 binding 与 snapshot hash 上移到同一不可变 scan 的多个 targets。

scan 跨目标快照绑定复用已完成：`RunHtScanEpoch` 在只读 target 切片入口完整构造一次 graph/hash bindings，内部 wavefront 检查对象身份后复用；每个 target 后的独立 `ContentHash()` 变更守卫、即时 proof 重放和最终 V2 原子提交均保留。三实例 CPU-fused candidate 加速 `1.674×/3.162×/7.355×`，search 加速 `1.007×/1.135×/1.246×`，d15112 wall 加速 `1.123×`；54 项跨提交精确比较全部通过。下一切片审计 d15112 leaf cursor construct 的 `69.004 ms`。

leaf path 稀疏验证绑定复用已完成：batch 内每个 path 对象只执行一次实际节点规模的规范认证，多个 outside cursor 复用强类型 binding；公开 scalar/dense verifier、成功 proof 重放和所有 HT/V2 授权边界不变。三实例 leaf setup 加速 `2.487×/4.641×/6.814×`，CPU-fused search 加速 `1.051×/1.105×/1.196×`；2,000 组 sparse/dense 差分及 54 项正式跨提交比较全部通过。下一切片审计 CPU exact cost matrix 的数据布局和重复距离读取。

CPU exact cost 固定验证与重连计划已完成：3–5 条删除边用固定数组认证，规范 templates 一次编译为端口对计划，task 内规范边/冲突/整数距离按 pair 复用；CUDA 结果仍逐 cell 由完整 CPU 矩阵认证。三实例 cost evaluate 加速 `1.510×/1.580×/1.431×`，CPU-fused search 加速 `1.274×/1.131×/1.035×`；54 项跨提交精确比较全部通过。下一切片先测量融合 batch 的实际 node-pair 重复率，再决定 batch-local 距离缓存或 cursor prepare 优化。

CPU batch 精确距离表复用已完成：画像确认 pcb3038/d15112 的 batch 内 task-pair 对局部 unique pair 分别有 `282.948×/131.206×` 重用空间；同步 cost 调用在最多 512 个端口节点、约 6 MiB 总额外内存且预计至少 2× 减少距离计算时建表，其他情况安全回退。三实例 cost evaluate 加速 `1.195×/1.057×/1.216×`，pcb3038 CPU-fused search/wall 加速 `1.083×/1.078×`，d15112 search 加速 `1.012×`；rl5915 距离阶段改善但端到端处于负向噪声。54 项跨提交精确比较与完整 sanitizer 门禁通过。下一切片消除融合矩阵按 cursor slice 的 scatter 复制。

M5 仍未完成：跨目标 HT 融合与多 epoch 重新排序、活动 edge-id 紧凑 launch、多 GPU，以及 M3.1 完成后的 LP—组合消元固定点评测仍为 pending。

## 当前完成定义

当前已交付 M0、M1、M2、M3 安全桥接、M4.1、M4.2a、M4.2b 候选器、M4.3a 有界精确困难叶、M4.3b1 浅层 HT、M4.3b2 CPU 递归语义与全局证书、M4.3b3a 混合 wavefront、M4.3b3b1 path-append 候选器、M4.3b3b2a1/a2 reply count/write、M4.3b3b2b1 frontier reply batching、M4.3b3b2b2a frontier path-append、M4.3b3b2b2b1 规范 child edge SoA、M4.3b3b2b2b2a frontier leaf batching、M4.3b3b2b2b2b1 一般 leaf 游标、M4.3b3b2b2b2b2a GPU leaf 驻留缓存、M4.3b3b2b2b2b2b1 CPU long-tail、M4.3b3b2b2b2b2b2 cooperative multi-block continuation、M4.3b3b2b2b2c HT epoch commit，以及 M5 JV 三轮优化和有界全图 HT pilot。M3.1 未完成时必须在状态清单中标为 pending；不得把单切片 pilot 冒充完整多 epoch Local Elimination 性能结论。
