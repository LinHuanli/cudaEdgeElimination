# M5 HT path-append 子阶段画像

## 1. 计时边界

提交 `ba15919` 将 HT scan report 升级为 V10、五路 benchmark summary 升级为 V13。`path_append_ms` 仍是包含式总量，新增五个互斥子阶段：

- `parent_prepare`：重新认证父路径系统的规范性，并构造 CPU/CUDA 共用的 state/node/edge 输入；
- `child_normalize`：逐 task 验证输入、复制父路径并调用 CPU `NormalizePathSystem`；
- `child_edges`：把有效 child 物化为规范 edge offsets/SoA；
- `cuda_evaluate`：显式 CUDA 后端的设备输入、kernel、同步和回传；
- `cuda_compare`：CUDA flags/offsets/edges 与 CPU 规范结果的全数组比较。

其余 reserve、后端选择和批级 bookkeeping 计入 residual。计时不进入 proof，也不改变候选顺序、CPU 认证或删除授权。

## 2. clean-commit 三实例 baseline

三次正式 run 均绑定 clean commit `ba159197572cc1b864d42916554f336b9b3c40b9`、物理 GPU 1、8 个 CPU cost threads、8 个相同目标和公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T225341Z-2645362`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T225359Z-2646015`；
- d15112：`artifacts/d15112-ht-scan-20260901T225414Z-2646539`。

CPU fused 路径画像如下，单位均为 ms：

| 实例 | path-append | parent prepare | child normalize | 占 path-append | child edges | residual |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 195.772 | 22.141 | 170.268 | 86.972% | 2.430 | 0.934 |
| rl5915 | 613.971 | 37.629 | 570.013 | 92.840% | 4.535 | 1.795 |
| d15112 | 5,306.771 | 87.379 | 5,193.507 | 97.866% | 19.609 | 6.276 |

对应 all-CUDA 的设备评估只有 12.904/4.942/11.101 ms；CUDA 路径仍必须执行相同 CPU child 规范化和全数组认证。因此问题不是 kernel 吞吐，而是 CPU 通用规范化随实例维度增长。

## 3. 根因

`NormalizePathSystem` 面向任意路径输入：每次调用都会按完整 `node_count` 构造 `vector<vector<int32_t>> adjacency` 和 visited/seen 位图，并两次扫描全部节点。HT path-append 的父路径通常只有少量边，但每个 reply task 都调用一次该通用实现。于是工作量接近 `tasks × TSP dimension`，d15112 即使 leaf work 少于 pcb3038，仍在 child normalization 上消耗 5.19 s。

父状态准备也会再次调用通用规范化，故同样带有维度项；但其调用粒度是 parent 而不是 task，目前只占 1.65%–11.31%。

## 4. 下一单变量实验

下一提交只替换 path-append 内部的 CPU 规范化实现：使用按实际出现节点存储的有序稀疏邻接表，保留相同的边唯一性、节点度数、回路、简单链、组件方向和字典序规则。通用 `NormalizePathSystem` 保持不动，继续作为 proof 重放的独立 dense 参考。

门禁要求：

1. 对批量 point/end tasks，稀疏 child 与 dense 参考逐项比较 `valid/reason/edge_count/paths`；
2. CPU Debug/Release、CUDA Release 和 compute-sanitizer 全部通过；
3. 三实例 V13 五路的工作签名、proof、最终边文件和受保护 tour 与本 baseline 完全一致；
4. 只在上述正确性门禁通过后报告 path-append/search 收益。
