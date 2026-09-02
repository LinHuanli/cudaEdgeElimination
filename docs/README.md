# 研究文档索引

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

文档优先级为：数学安全不变量 > 本拆分计划 > 原始设计中的性能设想。发现冲突时必须保留边并记录原因。
