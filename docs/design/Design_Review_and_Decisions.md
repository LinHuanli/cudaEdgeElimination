# 设计复核与已定决策

## 结论

原始设计的总体方向可行，但必须把“GPU/浮点结果”与“授权删除的证明”分开。首期采用同步双轨：GPU 快速消元与 cuOpt LP sidecar 同时落地；只有 CPU 精确复核或 Concorde 完整图精确定价可以提交删除。

## 环境事实修正

- 机器有 3 张 RTX 4000 Ada（20 GB，SM 8.9），不是假定型号；运行时动态选择空闲卡，不硬编码设备号。
- 本地 cuOpt 源快照是 26.10 开发版；可复现实验固定 `libcuopt-cu13==26.8.0`。
- ElimTSP 当前 `path.c` 的 `PMAX_TEST` 为 7。原设计 945×384-bit 兼容表只覆盖路径数 `m <= 5`；`m=6,7` 首期必须转 CPU。
- `references/concorde_code` 是可构建源码，但许可证限制学术研究；只能复制到 `.deps/` 覆盖构建，不能提交。
- `Datasets/TSP/test_dataset` 的 `.txt` 是归一化坐标和标签记录，不是 TSPLIB 文件；转换后距离度量发生变化，必须重新认证。
- `/vol` 空间紧张。依赖、构建和快照均留在本仓库，启动新快照前要求至少 8 GiB 可用，并限制保留数量。

## 首期范围

交付：

- JV 快速消元 CPU 基线、CUDA 候选 kernel、CPU 逐条复核、epoch 确定性提交；
- `EUC_2D` 与 `CEIL_2D` 整数坐标的精确距离；
- `proof-v1`、`lp-epoch-v1`、`run-manifest-v1` 的稳定接口；
- cuOpt C API sidecar，输出 primal/dual/reduced cost、残差和模型内定点精确下界；
- Concorde 导出与完整图精确定价的补丁接口和受限构建脚本；
- pr299 正确性集成与 pcb3038 中型实验入口。

后续项中，HS `c,d`/path-system GPU 闭环和目标级多 GPU 静态分片现已完成；更多距离类型、跨目标共享工作图和进一步性能调优仍未完成。未完成部分不伪装成已支持能力。

## 不变量

1. kernel 只读一个 epoch 快照，只写候选与见证。
2. CPU 在同一快照复算条件，拒绝不完整、溢出或不一致的见证。
3. 提交按 `(u,v,method,witness)` 排序，并保持每个顶点剩余度至少 2。
4. 一轮内不可见本轮删除；新图只在下一 epoch 生效。
5. 浮点下界、求解器 reduced cost、数据集标签不能单独授权删除。
6. unsupported/error/NaN/Inf/越界一律保留边并返回可诊断状态。

## 验收口径

首期不设最低加速比。验收只要求零错误删除、证明可重放、CPU/GPU 在固定快照上的已验证结果一致、同配置产出相同图哈希。性能数据必须记录，但不能为速度放松正确性门禁。
