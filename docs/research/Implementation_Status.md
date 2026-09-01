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
| M4.2b CUDA k-opt cost | 完成（候选器） | 批量精确成本矩阵；CPU/CUDA 单元一致；坏/漏候选 CPU 全模板兜底；memcheck 0 error |
| M4.3a 精确困难叶 | 完成（有界 CPU fallback） | 收缩 forced outside matching；Held–Karp 子集 DP；通用交换 witness 独立重放；block 超限为 unresolved |
| M4.3b HS AND–OR | 待实现 | `c,d` 候选、全部 Hamilton replies、全局 proof 与 epoch 接线尚未完成 |
| M5 中大型调优 | 待开始 | 首期不设最低加速比；pcb3038 尚未形成认证运行记录 |

## 当前基准结果

pr299 输入 1208 条边；JV 两个 epoch 后保留 1122 条，提交 86 条删除。CPU 与 CUDA 的最终内容哈希均为 `b9b67e9981518177`。这些数字是正确性回归结果，不构成论文性能结论。

cuOpt 手算 LP：状态 `OPTIMAL`，objective/dual objective 均为 `1`，primal violation 与 reduced-cost residual 均为 `0`，定点模型下界为 `16777216/16777216`。

Concorde 随机 20 点 epoch：25 行、43 列；QSopt 与 cuOpt 模型目标均为 `88`。cuOpt primal violation 为 `4.44e-15`，reduced-cost residual 为 `1.57e-14`；完整图 exact lower bound 为 `87.3932819641`，上界为 `88`。

pr299 Concorde epoch：454 行、888 列、8561 个非零元；cuOpt 状态 `OPTIMAL`，模型目标 `48187.777777780764`，primal violation `8.37e-11`，reduced-cost residual `4.87e-10`。完整图 exact lower bound 为 `43977.2693797`，合法但较弱；这说明后续需要对偶稳定化/迭代补列，而不是跳过负 reduced-cost penalty。

路径兼容表：`m=5` 为 45,360 字节，生成器哈希 `f6bccacc5c1fa84f`；362,880 个 `m=5` 单元和全部较小表均通过 CPU/CUDA 差分，CUDA memcheck 为 0 error。`m=6,7` 不建立完整表，按契约使用 CPU 直接判定。

CPU k-opt 叶证明：自动生成的 proper 3/4/5-opt 模板数为 `4/25/208`，每阶 2,000 个阈值定向随机矩阵与固定子模块 `swap.c` oracle 一致；成功 witness、全 outside coverage、预算 unresolved 和篡改拒绝均有回归。

CUDA k-opt cost：按删除集合与 proper template 形成精确成本矩阵，CPU/CUDA 逐单元一致；候选成功和 GPU 无命中后的 CPU completeness fallback 均通过回归，CUDA memcheck 为 0 error。当前尚未以真实 HS 任务报告加速比。

CPU 精确困难叶：将每条 forced outside edge 收缩为可双向访问的 block，其余节点为 singleton block；固定一个 block 的方向消除无向反转对称后，以 Held–Karp 子集 DP 穷举所有 block 次序和方向。比较时消去两条巡回共有的 outside 成本，并禁止候选重新使用 required path edge。成功结果转换为任意 `k>=2` 的交换 witness，再交给同一独立 verifier；默认关闭，启用时最多 18 个 block，内存不足或超限均返回 `unresolved`。60 组随机 7 点、每组两个 outside 的结果与直接 Hamilton 巡回枚举最优值一致，并覆盖 7-opt proof 往返。CPU Debug/ASan、CPU Release 与 CUDA Release 全量回归通过，GPU 2 上 k-opt memcheck 为 0 error。

## 安全边界

`gpu-eliminate` 目前只实现 JV quick candidate search；path-system CPU leaf proof（含有界精确 fallback）尚未连接 HS 全局 AND–OR 证书，因此也不授权删边；`lp-solve` 始终不修改图。Concorde 桥接已能产生完整图安全下界，但测试 wrapper 使用 `-B`，尚不输出消元边集。HS 全链路与 M3.1 必须标为 pending；仍严禁从局部布尔结果或 cuOpt 浮点 reduced cost 直接构造删除记录。
