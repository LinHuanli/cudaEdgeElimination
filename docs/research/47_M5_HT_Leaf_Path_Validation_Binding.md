# M5 HT leaf path 稀疏验证绑定复用

## 1. 根因与安全边界

d15112 的 8-target CPU-fused scan 启动 1,371 个 `KOptSearchCursor`。每个 cursor 构造都调用 `BuildTourContext`，并用公开 dense `NormalizePathSystem` 按完整 TSP 维度重新分配邻接/访问数组，尽管 leaf path system 通常只含少量实际节点。同一 batch 中一个 path proof 还可能因多个未覆盖 outside matching 重复启动 cursor，因此 d15112 的 cursor construct 累计达到 `69.004 ms`。

提交 `8c19740` 引入 batch 内部 `KOptPathValidationBinding`：

- 每个输入 path system 在 batch 初始化时只认证一次；
- binding 同时绑定实际 graph 和 `NormalizedPathSystem` 对象身份，错配立即失败；
- 认证使用只按实际节点分配的 sparse 规范器，并完整比较 `valid/paths/edge_count`；
- sparse 规范器从 path-append 私有副本收敛到 `detail` 内部实现，path-append 的行为不变；
- scalar/public verifier 继续使用原 dense `NormalizePathSystem`；
- batch 生成成功 proof 后仍调用独立 dense verifier，HT 即时复核、V2 replay 和 epoch commit 也完全不使用该 binding。

该 binding 不跨 leaf batch 缓存，不绑定指针以外的可变外部状态，也不缓存 outside matching、成本矩阵、witness 或证明结论。稀疏 fast path 即使错误接受输入，也无法绕过成功 proof 的 dense 重放；失败或资源耗尽仍只返回 unresolved，不授权删除。

## 2. 差分与提交门禁

新增测试包括：

- 固定种子生成 2,000 组合法/非法路径输入，逐项比较 sparse/dense 的 `valid`、失败原因、规范 paths 和 edge count；
- 篡改规范 path 的 edge count，要求 batch-sparse 与 scalar-dense 产生字节相同的失败 proof；
- 既有随机 cursor、预算边界、CPU/CUDA 矩阵、992-task path-append dense oracle 和完整 HT/V2 测试继续通过。

提交前完成 CPU Debug/Release 各 17/17、CUDA Release 20/20；k-opt 与 Hamilton–Tutte CUDA Debug 单元分别在 compute-sanitizer memcheck 下为 0 errors。

## 3. clean-commit 正式协议

正式 runs 均绑定 clean commit `8c19740`、物理 GPU 1、8 个 CPU cost threads，以及锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T004540Z-2725913`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T004555Z-2726539`；
- d15112：`artifacts/d15112-ht-scan-20260902T004603Z-2727087`。

相对 clean baseline `649f3f4`，三实例的 CPU、CPU-fused、CUDA、hybrid、fused 五路均逐字节比较最终边集和工作签名，并比较去除运行时 `metrics` 行后的 V2 proof；JV 固定点边集/proof、受保护最优 tour、独立 proof replay 和 tour-check 也全部通过。三实例共 54 项跨提交精确比较无差异。

## 4. 三实例性能结果

CPU-fused 单变量结果如下；加速均为 `649f3f4 / 8c19740`：

| 实例 | leaf setup：基线 → 新实现（ms） | setup 加速 | cursor construct 加速 | leaf 加速 | search 加速 | total 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pcb3038 | 143.255 → 57.603 | 2.487× | 4.546× | 1.053× | 1.051× | 1.051× | 1.053× |
| rl5915 | 33.193 → 7.152 | 4.641× | 6.528× | 1.193× | 1.105× | 1.085× | 1.045× |
| d15112 | 69.638 → 10.220 | 6.814× | 8.363× | 1.428× | 1.196× | 1.131× | 1.100× |

一次性 sparse 认证现在计入 proof initialization，因此该子阶段从 `11.259/0.204/0.300 ms` 增至 `26.225/1.811/1.646 ms`；与此同时 cursor construct 从 `129.501/32.654/69.004 ms` 降至 `28.487/5.002/8.251 ms`。setup、leaf、search、total 和 wall 均包含新增认证成本，表中收益不是计时搬移。

d15112 CPU-fused leaf 从 `206.669 ms` 降至 `144.750 ms`，work graph 从 `328.486 ms` 降至 `267.284 ms`，search 从 `373.588 ms` 降至 `312.481 ms`，CLI wall 从 `690.825 ms` 降至 `627.747 ms`。

## 5. 新画像与下一切片

优化后 d15112 leaf 的主要子阶段为 cursor consume `36.365 ms`、cost evaluate `34.454 ms`、apply `24.719 ms`、proof verify `19.112 ms` 和 cursor prepare `15.370 ms`；setup 已降至 `10.220 ms`。pcb3038 仍由 51,309,996 个 cost cells 主导，CPU-fused cost evaluate 为 `631.746 ms`。

下一单变量实验优先审计 CPU exact cost matrix 的任务布局、重复距离读取和行级写入，因为它同时支配 pcb3038，并在 d15112 位于最大两个 leaf 子阶段之一。任何优化仍须逐 cell 产生相同整数矩阵；CUDA 路径继续由 CPU 全矩阵认证，严格改善模板继续由通用路径重建并经独立 witness verifier 复核。
