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

先做 CPU 规范实现，再做 `m<=5` GPU 表查找；`m=6,7` CPU fallback。门禁是所有小路径系统穷举等价、表生成可复现、GPU 只输出 CPU 接受的候选。

## M5：中大型评测与优化

依次 pcb3038、rl5915、d15112，定位 CSR 构建、带宽、分歧和复核瓶颈。图重排、多流、多 GPU 都必须在 M1–M4 证书不变量不变的前提下实验。

## 当前完成定义

当前已交付 M0、M1、M2 和 M3 安全桥接。M3.1 与 M4 未完成时必须在状态清单中标为 pending，不能用合法但偏弱的下界或计划文档替代删除强度与 HS 实现。
