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

M4.2b 继续实现批量 CUDA k-opt cost、困难叶 CPU fallback、HS `c,d` AND–OR 搜索和全局证书重放。只有整条证书链完成后才把 HS 候选接入 epoch commit。

## M5：中大型评测与优化

依次 pcb3038、rl5915、d15112，定位 CSR 构建、带宽、分歧和复核瓶颈。图重排、多流、多 GPU 都必须在 M1–M4 证书不变量不变的前提下实验。

## 当前完成定义

当前已交付 M0、M1、M2、M3 安全桥接、M4.1 和 M4.2a。M3.1 与 M4.2b 未完成时必须在状态清单中标为 pending，不能用合法但偏弱的下界或局部叶 proof 替代删除强度与 HS 全链路。
