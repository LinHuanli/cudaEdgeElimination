# 实现状态（2026-09-02）

| 工作包 | 状态 | 已验证证据 |
|---|---|---|
| M0 仓库与复现 | 完成 | 项目内依赖/构建目录、固定子模块、CPU CI、路径门禁 |
| M1 JV CPU/CUDA 闭环 | 完成（首期 JV 范围） | n=6–12 全最优边检查；pr299 CPU/GPU 哈希一致；proof 重放；compute-sanitizer 0 error |
| M2 cuOpt sidecar | 完成（模型内证书） | cuOpt C API 26.8.0；手算小 LP objective=1；残差为 0；精确下界=1 |
| M3 Concorde 导出 | 完成 | 内容寻址受限 overlay；Concorde graph 目标映射复核；随机 20 点和 pr299 CSR 往返 |
| M3 完整图 exact pricing | 完成（安全下界桥接） | `CCbigguy` 注入；完整图负 reduced-cost penalty；三方哈希；错配拒绝 |
| M3.1 对偶稳定化与边集导出 | 待实现 | pr299 PDLP 完整图界偏弱；尚未导出每边 exact RC/Concorde 消元后边集 |
| M4 HS path-system GPU | 待实现 | `m<=5` 表与 `m=6,7` CPU fallback 尚未接入 CLI |
| M5 中大型调优 | 待开始 | 首期不设最低加速比；pcb3038 尚未形成认证运行记录 |

## 当前基准结果

pr299 输入 1208 条边；JV 两个 epoch 后保留 1122 条，提交 86 条删除。CPU 与 CUDA 的最终内容哈希均为 `b9b67e9981518177`。这些数字是正确性回归结果，不构成论文性能结论。

cuOpt 手算 LP：状态 `OPTIMAL`，objective/dual objective 均为 `1`，primal violation 与 reduced-cost residual 均为 `0`，定点模型下界为 `16777216/16777216`。

Concorde 随机 20 点 epoch：25 行、43 列；QSopt 与 cuOpt 模型目标均为 `88`。cuOpt primal violation 为 `4.44e-15`，reduced-cost residual 为 `1.57e-14`；完整图 exact lower bound 为 `87.3932819641`，上界为 `88`。

pr299 Concorde epoch：454 行、888 列、8561 个非零元；cuOpt 状态 `OPTIMAL`，模型目标 `48187.777777780764`，primal violation `8.37e-11`，reduced-cost residual `4.87e-10`。完整图 exact lower bound 为 `43977.2693797`，合法但较弱；这说明后续需要对偶稳定化/迭代补列，而不是跳过负 reduced-cost penalty。

## 安全边界

`gpu-eliminate` 目前只实现 JV quick candidate search；`lp-solve` 始终不修改图。Concorde 桥接已能产生完整图安全下界，但测试 wrapper 使用 `-B`，尚不输出消元边集。HS 与 M3.1 必须标为 pending；仍严禁从 cuOpt 浮点 reduced cost 直接构造删除记录。
