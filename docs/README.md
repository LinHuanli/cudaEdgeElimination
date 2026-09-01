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

文档优先级为：数学安全不变量 > 本拆分计划 > 原始设计中的性能设想。发现冲突时必须保留边并记录原因。
