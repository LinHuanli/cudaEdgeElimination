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

M4.2b 批量 CUDA k-opt cost 候选器已完成：GPU 输出完整 template 成本矩阵，成功候选由 CPU 重建；任何未接受集合都由 CPU 全模板兜底，因此 GPU 不承担 completeness 授权。

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

M4.3b3b2b2b2a frontier leaf batching 已完成：状态按 `(depth,path_count,node_count,max_k,incoming reply bucket)` 确定性分桶；`max_deletion_sets=1` 时同轮、同 k 的首个 cost rows 跨状态融合。CUDA 候选仍经 CPU 全模板 completeness fallback 与 witness verifier；`N=1/256` 的完整 V1 proof 逐字节一致。

M4.3b3b2b2b2b1 一般 leaf 游标已完成：每个 path/outside 只生成当前 `cost_batch_size` block，同轮同 k 跨状态融合；CPU 消费后才推进下一组合。无界预算不预展开，3/4/5-opt、预算中断和随机路径的 batch/scalar proof 字节一致。

M4.3b3b2b2b2b2a GPU leaf 驻留缓存已完成：每主机线程/设备保留精确坐标快照、独立 3/4/5-opt 模板与增长型 task/cost workspace；完整键比较防止错误复用，命中和驻留字节均进入 wavefront 指标。

M4.3b3b2b2b2b2b1 GPU/CPU long-tail 已完成：缓存后微基准确定默认 128-cell 阈值，`auto` 按融合矩阵规模分流；CPU/GPU 路由使用规范模板计数，阈值两侧 V1 proof 字节一致。

M4.3b3b2b2b2b2b2 cooperative multi-block continuation 已完成：冻结 queue batches 通过 grid barrier 传播，自动 block 数受实际 residency 约束；512-way AND 跨 block 真值、single/multi-block 状态和完整 V1 proof 均一致。

M4.3b3b2b2b2c HT epoch commit 已完成：同一不可变 snapshot 上的多个递归 proof sidecar 先整批 CPU 重放，再按规范目标和最小度门禁在图副本上原子提交；自包含 `CUDAEE_PROOF_V2` 内嵌原 HT V1 并由通用 `verify` 重放。坏/旧 sidecar 使整批零修改，JV-only 输出继续保持 V1。

## M5：中大型评测与优化

依次 pcb3038、rl5915、d15112，定位 CSR 构建、带宽、分歧和复核瓶颈。图重排、多流、多 GPU 都必须在 M1–M4 证书不变量不变的前提下实验。

JV 基线与第三轮优化已完成：三实例各预热一次、计时五次，CPU/CUDA 输出逐字节一致且每次 proof 均由独立 CPU 进程重放。规范边 CSR 免排序、跨 epoch CUDA 驻留后，动态 CSR 再以 32 位 edge id 引用静态边权；进程内算法中位数加速达到 `11.194× / 30.517× / 41.695×`，驻留降为 `0.321/1.225/6.640 MiB`。分段计时确认 d15112 的 H2D/kernel/D2H 为 `2.965/6.628/0.513 ms`。独立 CLI 仍受 context/I/O 和共享节点抖动支配。`pcb3038` 另通过成本 137,694 的受保护最优 tour 门禁。

M5 仍未完成：`rl5915/d15112` 的最优 tour witness、全图 HT 自动目标调度、活动 edge-id 紧凑 launch、多 GPU，以及 M3.1 完成后的 LP—组合消元固定点评测仍为 pending。

## 当前完成定义

当前已交付 M0、M1、M2、M3 安全桥接、M4.1、M4.2a、M4.2b 候选器、M4.3a 有界精确困难叶、M4.3b1 浅层 HT、M4.3b2 CPU 递归语义与全局证书、M4.3b3a 混合 wavefront、M4.3b3b1 path-append 候选器、M4.3b3b2a1/a2 reply count/write、M4.3b3b2b1 frontier reply batching、M4.3b3b2b2a frontier path-append、M4.3b3b2b2b1 规范 child edge SoA、M4.3b3b2b2b2a frontier leaf batching、M4.3b3b2b2b2b1 一般 leaf 游标、M4.3b3b2b2b2b2a GPU leaf 驻留缓存、M4.3b3b2b2b2b2b1 CPU long-tail、M4.3b3b2b2b2b2b2 cooperative multi-block continuation、M4.3b3b2b2b2c HT epoch commit，以及 M5 JV 中大型基线、CSR 快路径、驻留与动态 edge-id 优化。M3.1 未完成时必须在状态清单中标为 pending；中大型全图 HT 调度属于 M5，不能用 JV 或小实例闭环冒充性能结论。
