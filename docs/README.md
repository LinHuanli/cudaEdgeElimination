# 研究文档索引

当前工作以[单 GPU 融合、超过 2014 的冻结执行计划](design/FGPU_Hybrid_Beat_HS2014_Execution.md)
为验收口径；最新结果见[近点预热与强度缺口](research/75_FGPU_Point_Priming_and_Strength_Gap.md)。
下方历史研究记录继续保留，不能把不同输入协议或版本的时间混作当前完整图结果。

原始方案保留在 [Local_Elimination_GPU_cuOpt_Design.md](design/Local_Elimination_GPU_cuOpt_Design.md)。以下文档把它拆成可执行、可验收的工作包：

1. [设计复核与决策](design/Design_Review_and_Decisions.md)：方案中的事实修正、范围与不变量。
2. [正确性与证书](research/01_Correctness_and_Certificates.md)：删除规则、LP 下界和 fail-closed 条件。
3. [GPU 快速消元](research/02_GPU_Fast_Elimination.md)：数据布局、epoch、kernel 和 CPU 复核。
4. [Concorde/cuOpt 集成](research/03_Concorde_cuOpt_LP_Integration.md)：模型导出、C API 与精确定价边界。
5. [数据与基准协议](research/04_Data_and_Benchmark_Protocol.md)：实例规范化、标签、指标与实验矩阵。
6. [路线图与门禁](research/05_Roadmap_and_Gates.md)：里程碑、完成定义和风险。
7. [环境与复现](development/Environment_and_Reproducibility.md)：固定依赖、GPU 选择和磁盘约束。
8. [仓库与 Git 工作流](development/Repository_and_Git_Workflow.md)：目录职责、分支、CI 与产物策略。
9. [实现状态](research/Implementation_Status.md)：已完成证据、当前结果和明确未完成项。
10. [路径系统与兼容表](research/06_Path_System_Compatibility.md)：规范化、穷举表、CUDA 查询与 CPU 回退契约。
11. [浅层 Hamilton–Tutte AND–OR](research/07_Hamilton_Tutte_Shallow_AND_OR.md)：`c,d` moves、完整 replies、嵌套叶证明与 CUDA 候选边界。
12. [递归 Hamilton–Tutte 证明](research/08_Hamilton_Tutte_Recursive_Proof.md)：point/end moves、continuation arena、全局 V1 证书与独立重放。
13. [Hamilton–Tutte GPU wavefront](research/09_Hamilton_Tutte_GPU_Wavefront.md)：BFS 工作图、CUDA continuation counters、CPU 全量差分与资源边界。
14. [Hamilton–Tutte GPU path append](research/10_Hamilton_Tutte_GPU_Path_Append.md)：point/end 批量冲突标记、规范子状态与 CPU 差分契约。
15. [Hamilton–Tutte GPU Hamilton replies](research/11_Hamilton_Tutte_GPU_Hamilton_Replies.md)：多中心邻边对 count/write、确定性 CSR 输出与 CPU 完整枚举复核。
16. [Hamilton–Tutte GPU end replies](research/12_Hamilton_Tutte_GPU_End_Replies.md)：端点活动边 count/write、零长度区间与 wavefront 集成契约。
17. [Hamilton–Tutte frontier reply batching](research/13_Hamilton_Tutte_Frontier_Reply_Batching.md)：跨父状态 chunk、确定性回填、资源边界与 point-first end 筛选。
18. [Hamilton–Tutte frontier path append](research/14_Hamilton_Tutte_Frontier_Path_Append.md)：多父状态稀疏 spans、point-first 两阶段批处理与无投机 end 生成。
19. [Hamilton–Tutte GPU 规范 child edge SoA](research/15_Hamilton_Tutte_GPU_Child_Edge_SoA.md)：设备端 count/write、规范边 CSR 与 CPU 全数组认证。
20. [Hamilton–Tutte frontier leaf batching](research/16_Hamilton_Tutte_Frontier_Leaf_Batching.md)：确定性复杂度桶、跨状态 k-opt cost row 融合与 scalar long-tail。
21. [Hamilton–Tutte 增量 leaf cost 游标](research/17_Hamilton_Tutte_Incremental_Leaf_Cursors.md)：一般 deletion 预算、分段组合游标与无投机跨 leaf 融合。
22. [Hamilton–Tutte GPU leaf 驻留缓存](research/18_Hamilton_Tutte_GPU_Leaf_Cache.md)：精确快照/模板键、增长型工作区与可观测命中指标。
23. [Hamilton–Tutte leaf GPU/CPU long-tail](research/19_Hamilton_Tutte_CPU_Long_Tail.md)：缓存后交叉点、128-cell 自动阈值与规范 proof 计数。
24. [Hamilton–Tutte cooperative multi-block continuation](research/20_Hamilton_Tutte_Multi_Block_Continuation.md)：冻结队列批次、grid barrier 与 residency 门禁。
25. [Hamilton–Tutte 不可变 epoch 提交](research/21_Hamilton_Tutte_Epoch_Commit.md)：整批 CPU 复核、V2 内嵌 sidecar 与原子确定性删边。
26. [M5 JV 中大型基准](research/22_M5_JV_Medium_Benchmark.md)：三组 clean-commit CPU/CUDA 门禁、最优 tour 检查、阶段计时与 CSR 快路径。
27. [M5 JV CUDA 驻留缓存](research/23_M5_JV_Device_Residency.md)：精确静态键、动态 CSR 刷新、增长 workspace 与三实例稳态收益。
28. [M5 JV 动态 CSR edge-id](research/24_M5_JV_Dynamic_CSR_Edge_Ids.md)：复用驻留边权、分段 CUDA 计时与三实例显存/耗时门禁。
29. [M5 有界全图 HT 目标扫描](research/25_M5_HT_Target_Scan.md)：确定性目标切片、预算失败关闭、V2 原子提交与 pcb3038 CPU/CUDA pilot。
30. [M5 HT 阶段画像与混合后端](research/26_M5_HT_Phase_Profiling_and_Hybrid.md)：V2 包含式计时、CPU/全 CUDA/混合三路门禁与跨目标 leaf 合批接口。
31. [M5 HT leaf 复杂度桶融合](research/27_M5_HT_Leaf_Bucket_Fusion.md)：参数排除、V3 批计数、四路等价门禁与 launch 非主瓶颈结论。
32. [M5 HT leaf 子阶段画像](research/28_M5_HT_Leaf_Subphase_Profiling.md)：V4/V5 细粒度计时、CPU consume/setup 主瓶颈与不可变表缓存门禁。
33. [M5 HT leaf 不可变组合表缓存](research/29_M5_HT_Immutable_Leaf_Tables.md)：matching/reconnect 延迟只读缓存、完整门禁与 8-target 约 29% search 降幅。
34. [M5 HT leaf CPU completeness 画像](research/30_M5_HT_Leaf_Completeness_Profiling.md)：V5/V6 consume 细分、99% fallback 主路径与 CPU 精确矩阵快路径门禁。
35. [M5 HT leaf CPU 精确成本矩阵认证](research/31_M5_HT_CPU_Exact_Cost_Matrix.md)：逐 cell CPU/CUDA 认证、固定数组 scorer、零通用 fallback 与 8-target 端到端收益。
36. [M5 HT leaf CPU matrix 公平基线](research/32_M5_HT_CPU_Matrix_Baseline.md)：显式 CPU cursor、投机 block 规范计数修复、V8 四路门禁与 GPU 净收益复评。
37. [M5 HT Hamilton reply 主机去重与过滤缓存](research/33_M5_HT_Hamilton_Reply_Host_Cache.md)：批内 center 去重、邻边预计算、V7/V9 画像与 8-target search 约 2.26× 降幅。
38. [M5 HT leaf setup 画像与快照哈希复用](research/34_M5_HT_Leaf_Setup_Snapshot_Hash.md)：V8/V10 setup 细分、批内快照哈希复用及 CPU search 约 1.45× 降幅。
39. [M5 HT CPU 精确成本矩阵行并行](research/35_M5_HT_CPU_Cost_Row_Parallelism.md)：8,192-cell 门槛、8-thread 静态 row 分片、V9/V11 门禁及 CPU search 约 1.85× 降幅。
40. [M5 HT CPU leaf 复杂度桶融合](research/36_M5_HT_CPU_Leaf_Bucket_Fusion.md)：V12 五路门禁、CPU 并行覆盖提升及 pcb3038 search 1.113× 收益。
41. [M5 HT 多实例融合门禁](research/37_M5_HT_Multi_Instance_Fusion_Gates.md)：锁定三份最优 tour、三实例 V12 门禁与 CPU leaf 后端感知默认值。
42. [M5 HT path-append 稀疏规范化](research/38_M5_HT_Path_Append_Profile.md)：V10/V13 画像、dense 差分门禁及三实例 path-append 5.503×–25.561× 加速。
43. [M5 HT 根 child 规范化排除实验](research/39_M5_HT_Root_Child_Normalization_Profile.md)：V11/V14 计时证明根 dense 规范化仅占 host residual 0.41%–2.61%。
44. [M5 HT point-candidate 全维选择画像与 Top-K 优化](research/40_M5_HT_Point_Candidate_Profile.md)：V12/V15 定位扫描/排序瓶颈，并以确定性 Top-K 获得 11.565×–29.831× 排序加速。
45. [M5 HT point-candidate 静态次序缓存](research/41_M5_HT_Point_Candidate_Order_Cache.md)：复用 target 级严格次序并以 generation marks 过滤 state，累计获得 1.135×–1.858× search 加速。
46. [M5 HT leaf proof 批内快照绑定复用](research/42_M5_HT_Leaf_Proof_Snapshot_Binding.md)：保持公开独立 verifier 不变，消除成功 leaf 的重复整图哈希并获得 8.882×–27.449× verify 加速。
47. [M5 HT wavefront leaf 快照绑定复用](research/43_M5_HT_Wavefront_Snapshot_Binding.md)：强类型绑定同一只读 graph 对象，消除跨 leaf batch 重复哈希并保持全部独立 verifier 不变。
48. [M5 HT path-append child 增量规范化](research/44_M5_HT_Path_Append_Incremental_Normalization.md)：从已认证 parent 直接合并规范链，以 992-task dense 差分获得 8.179×–9.752× child normalize 加速。
49. [M5 HT wavefront 图验证绑定复用](research/45_M5_HT_Wavefront_Graph_Validation_Binding.md)：一次完整 CSR 验证供 c,d/Hamilton/end batches 共用，并保持公开 API 与独立 verifier 不变。
50. [M5 HT scan 跨目标快照绑定复用](research/46_M5_HT_Scan_Snapshot_Binding.md)：在同一不可变 target 切片复用 graph/hash bindings，同时保留逐目标内容哈希守卫和全部独立重放。
51. [M5 HT leaf path 稀疏验证绑定复用](research/47_M5_HT_Leaf_Path_Validation_Binding.md)：每个 batch path 只做一次稀疏规范认证，成功 proof 仍由公开 dense verifier 独立重放。
52. [M5 HT CPU cost 固定验证与重连计划](research/48_M5_HT_CPU_Cost_Fixed_Plans.md)：消除小集合堆分配与逐 cell template 解码，并保持完整 CPU/CUDA 整数矩阵认证。
53. [M5 HT CPU batch 精确距离表复用](research/49_M5_HT_CPU_Batch_Distance_Cache.md)：在有界同步 batch 内复用跨 task 整数距离，并以容量/收益门禁安全回退。
54. [M5 HT leaf cost 零复制分发](research/50_M5_HT_Leaf_Cost_Zero_Copy_Scatter.md)：用生命周期受限的只读 view 分发融合矩阵，消除按 cursor slice 的重复分配与复制。
55. [M5 HT cursor prepare 路径边成本缓存](research/51_M5_HT_Cursor_Prepare_Edge_Cost_Cache.md)：固定化删除位置并在 cursor 内复用路径边精确成本，候选仍由通用路径重新取距复核。
56. [M5 HT CPU cost 输出 workspace](research/52_M5_HT_CPU_Cost_Output_Workspace.md)：以同步逻辑 span 覆盖可增长 storage，消除大矩阵先清零再全量覆写，并保留公开 owning API。
57. [M5 HT CPU cost 完全相同 task row 去重](research/53_M5_HT_CPU_Cost_Task_Row_Deduplication.md)：只评分精确相同 key 的首次 row，以零展开索引供 cursor 复用，并分别记录逻辑认证量与物理评分量。
58. [M5 JV—HT 多 epoch Local Elimination 编排](research/54_M5_Local_Elimination_Multi_Epoch_Orchestration.md)：JV 固定点、HT 无提交 sweep 推进、提交后重排，以及单一可重放 V2 证明组合。
59. [M5 JV 活动 edge-id 紧凑启动排除实验](research/55_M5_JV_Active_Edge_Compact_Launch_Rejection.md)：以 d15112 稳态 A/B 量化空线程节省与紧凑化开销，并撤销端到端回退的原型。
60. [M5 HT 跨目标根候选融合排除实验](research/56_M5_HT_Cross_Target_Root_Candidate_Fusion_Rejection.md)：以 d15112 七对 clean A/B 量化单次根 `c,d` launch 与逐目标路径，并把后续共享边界收窄到 leaf/reply/work-graph。
61. [M5 HT reply CUDA 图与工作区驻留复用](research/57_M5_HT_Reply_CUDA_Device_Residency.md)：跨 batches/targets 精确复用坐标、CSR 与增长 workspace，并以 d15112 七对 A/B、proof 和 tour 门禁确认收益。
62. [M5 HT reply 精确任务去重](research/58_M5_HT_Reply_Task_Deduplication.md)：按首次出现顺序折叠 batch-local Hamilton/end 任务，以完整 CPU 逻辑结果、开关差分和 d15112 七对 A/B 确认收益。
63. [M5 HT 跨 batch reply 结果缓存排除画像](research/59_M5_HT_Cross_Batch_Reply_Result_Cache_Profile.md)：量化长生命周期精确 key 只能再减少 136 个 tasks、不能消除任何 CUDA batch，因此不引入结果缓存。
64. [M5 HT 目标级多 GPU 静态切片](research/60_M5_HT_Target_Multi_GPU_Static_Slicing.md)：按可见设备固定 target workers，保持规范 proof/提交顺序，并以双 A4000 七对 A/B 验证正确性与扩展收益。
65. [GPU-ElimTSP 融合评审与修订方案 v2](design/GPU-ElimTSP-融合评审与修订方案-v2.md)：对原路线的融合评审、硬件假设修订与阶段建议。
66. [GPU-ElimTSP 设计方案 v3](design/GPU-ElimTSP-设计方案-v3.md)：短路转置执行、精确 leaf、验证边界与论文对齐的新版总设计。
67. [V3 单 GPU差距分析与决策](design/V3_单GPU差距分析与决策.md)：结合当前代码确定本分支范围、不变量、基线与采用门禁。
68. [V3 单 GPU实施路线](research/61_V3_单GPU实施路线.md)：A0/A1/C1/C1.5 的实现拆分与提交策略。
69. [V3 论文对齐与基准协议](research/62_V3_论文对齐与基准协议.md)：比较 scope、七对 A/B、强度门禁与 Table 7 对齐方法。
70. [V3 单 GPU原型实现与 Pilot](research/63_V3_单GPU原型实现与Pilot.md)：已落地能力、A5000 消融、转置调度限制和下一工作边界。
71. [V3 单 GPU 跨目标 Leaf Broker](research/64_V3_单GPU跨目标LeafBroker.md)：异构 required-edge 合批、CUDA 候选位图、机会式微批和 d15112 五对正式 A/B。
72. [pcb442 完全图固定点端到端](research/65_pcb442_完全图固定点端到端.md)：97,461 条完全图、GPU JV/KH-HS 固定点、4,016 条最终边、同环境作者对照与证书边界。
73. [FGPU One-Shot 实现与首轮基准](research/66_FGPU_OneShot_实现与基准.md)：geometry→LP-box→JV→HT 单 GPU 链路、pcb442/pr1002 结果、性能修复与剩余差距。
74. [FGPU 证书与数值边界](research/67_FGPU_证书与数值边界.md)：V4 sidecar、`__int128` 下界、原子 epoch、输出绑定与负向测试。
75. [FGPU 单卡全常驻实现与端到端基准](research/68_FGPU_单卡全常驻实现与端到端基准.md)：同卡 Geometry→PDLP→JV→Quick-HS 固定点、GPU 区间证明、pcb442 七次确定性复跑与可信端到端加速。
76. [FGPU 无上限 raw 与 pcb3038 LP 诊断](research/69_FGPU_无上限Raw与pcb3038_LP诊断.md)：默认无证书固定点、阶段画像、warp 并行否决，以及 degree/1-tree/subtour LP 强度分解。
77. [FGPU 强度升级与纯 GPU 架构方案](design/FGPU-Elim_Strength_Upgrade_and_Pure_GPU_Architecture.md)：P0–P8 总设计、数学授权边界、完整单 GPU 数据流与论文对齐门禁。
78. [FGPU 强度升级 P0–P8 实现与论文对齐](research/70_FGPU_Strength_Upgrade_P0_P8_Implementation.md)：纯 GPU-safe 固定点、Signed128 LP、Main/`-e2`、non-pair/fixing、正确性修复、pcb3038 论文 LP 图三次基准、完全图正式运行及尚未完成项。
79. [FGPU 快照事务、稀疏 PDHG 与完整端点 OR](research/71_FGPU_Transaction_PDHG_and_PathEnd.md)：同快照联合提交、独立 SEC replay、原生 PDHG、四端点与 Opt34／occupancy 消融、全最优回归及真实性能回退。
80. [FGPU 可复现交错基准](research/72_FGPU_Reproducible_Paired_Benchmarks.md)：冻结构建与输入身份、同卡 AB/BA、进程墙钟、终态一致性及 clean 边界。
81. [单 GPU 融合方案超过 2014](design/FGPU_Hybrid_Beat_HS2014_Execution.md)：四实例强度、历史及同机双速度门槛，完整无标签入口与实现依赖。
82. [GPU bootstrap 与全度数 metric](research/73_FGPU_Hybrid_Bootstrap_and_FullMetric.md)：GPU 完整图、精确距离、无标签上界、LP-off pair 服务与条件缓存。
83. [Point 复用与真实 frontier 调度](research/74_FGPU_Point_Reuse_and_Frontier_Scheduling.md)：排列复用、近点优先、延期与最终全扫，pr1002 同终态端到端提速及剩余强度差距。
84. [Point 近点预热与强度缺口](research/75_FGPU_Point_Priming_and_Strength_Gap.md)：预热开关、小图全最优检查、中间快照 CSR 修复、pr299 消融和 vm1084 长尾。

文档优先级为：数学安全不变量 > 本拆分计划 > 原始设计中的性能设想。发现冲突时必须保留边并记录原因。
