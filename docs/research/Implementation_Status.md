# 实现状态（2026-09-02）

| 工作包 | 状态 | 已验证证据 |
|---|---|---|
| M0 仓库与复现 | 完成 | 项目内依赖/构建目录、固定子模块、CPU CI、路径门禁 |
| M1 JV CPU/CUDA 闭环 | 完成（首期 JV 范围） | n=6–12 全最优边检查；pr299 CPU/GPU 哈希一致；proof 重放；compute-sanitizer 0 error |
| M2 cuOpt sidecar | 完成（模型内证书） | cuOpt C API 26.8.0；手算小 LP objective=1；残差为 0；精确下界=1 |
| M3 Concorde 导出 | 完成 | 受限 overlay 构建；随机 20 点模型 26×56/144 nnz 导出；哈希回读；cuOpt residual 通过 |
| M3 完整图 exact pricing | 待实现 | 当前明确禁止 LP 结果授权 TSP 删除 |
| M4 HS path-system GPU | 待实现 | `m<=5` 表与 `m=6,7` CPU fallback 尚未接入 CLI |
| M5 中大型调优 | 待开始 | 首期不设最低加速比；pcb3038 尚未形成认证运行记录 |

## 当前基准结果

pr299 输入 1208 条边；JV 两个 epoch 后保留 1122 条，提交 86 条删除。CPU 与 CUDA 的最终内容哈希均为 `b9b67e9981518177`。这些数字是正确性回归结果，不构成论文性能结论。

cuOpt 手算 LP：状态 `OPTIMAL`，objective/dual objective 均为 `1`，primal violation 与 reduced-cost residual 均为 `0`，定点模型下界为 `16777216/16777216`。

Concorde smoke epoch：26 行、56 列、144 个非零元；关闭 presolve 后 cuOpt 状态 `OPTIMAL`，primal violation 为 `0`，reduced-cost residual 为 `8.88e-16`。其模型目标为 0，只能证明导出/求解/残差链路，不能替代原 TSP 目标的完整图定价。

## 安全边界

`gpu-eliminate` 目前只实现 JV quick candidate search。`lp-solve` 不修改图。完整图精确定价和 HS 尚未完成，因此任何报告都必须把它们标为 pending；不允许从 cuOpt 的浮点 reduced cost 直接构造删除记录。
