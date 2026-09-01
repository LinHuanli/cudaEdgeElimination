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

M4.3b3a 混合 wavefront 已完成：主机按层生成完整有界 AND–OR 工作图；CUDA 从最深层反向计算每个状态的 `OR(move) / AND(reply)` 真值；返回前 CPU 对全部状态逐项复算，只从成功状态提取既有 V1 continuation arena。DFS 与 wavefront 在固定 shallow/point/end 实例上均产生可验证证明。

M4.3b3b 继续把状态/reply 生成和 leaf 批处理迁到 GPU，加入按路径数/深度/reply 数分桶、persistent continuation counters 和 CPU long-tail 队列。CPU/GPU 对完整小实例真值及证书必须一致；之后才把验证成功的 HT 候选接入 epoch commit。

## M5：中大型评测与优化

依次 pcb3038、rl5915、d15112，定位 CSR 构建、带宽、分歧和复核瓶颈。图重排、多流、多 GPU 都必须在 M1–M4 证书不变量不变的前提下实验。

## 当前完成定义

当前已交付 M0、M1、M2、M3 安全桥接、M4.1、M4.2a、M4.2b 候选器、M4.3a 有界精确困难叶、M4.3b1 浅层 HT、M4.3b2 CPU 递归语义与全局证书，以及 M4.3b3a 混合 wavefront。M3.1 与 M4.3b3b 未完成时必须在状态清单中标为 pending，不能用合法但偏弱的下界或仅传播阶段上 GPU 的原型替代删除强度与完整 GPU/提交链路。
