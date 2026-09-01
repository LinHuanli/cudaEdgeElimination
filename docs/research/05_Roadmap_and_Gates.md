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

交付受限 overlay 构建、QSopt 模型导出、行/列映射、`CCbigguy` 注入与 exact pricing。门禁：pr299 上 cuOpt 浮点对偶经完整精确定价得到合法下界；与原 QSopt 路径的安全删除结果等价；人为错配哈希被拒绝。

## M4：HS `c,d` 与路径兼容

先做 CPU 规范实现，再做 `m<=5` GPU 表查找；`m=6,7` CPU fallback。门禁是所有小路径系统穷举等价、表生成可复现、GPU 只输出 CPU 接受的候选。

## M5：中大型评测与优化

依次 pcb3038、rl5915、d15112，定位 CSR 构建、带宽、分歧和复核瓶颈。图重排、多流、多 GPU 都必须在 M1–M4 证书不变量不变的前提下实验。

## 当前完成定义

本轮实现目标为 M0、M1、M2 的可运行版本以及 M3 的接口/overlay 脚手架。M3 精确定价和 M4 不完整时必须在状态清单中标为 pending，不能用计划文档替代实现。
