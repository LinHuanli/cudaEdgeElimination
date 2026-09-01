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

## 4. 稀疏规范化设计与门禁

提交 `b551a2e` 只替换 path-append 内部的 CPU 规范化实现：使用按实际出现节点存储的有序稀疏邻接表，保留相同的边唯一性、节点度数、回路、简单链、组件方向和字典序规则。通用 `NormalizePathSystem` 保持不动，继续作为 proof 重放的独立 dense 参考。

门禁要求：

1. 对批量 point/end tasks，稀疏 child 与 dense 参考逐项比较 `valid/reason/edge_count/paths`；
2. CPU Debug/Release、CUDA Release 和 compute-sanitizer 全部通过；
3. 三实例 V13 五路的工作签名、proof、最终边文件和受保护 tour 与本 baseline 完全一致；
4. 只在上述正确性门禁通过后报告 path-append/search 收益。

单元测试批量枚举两个不同父系统上的 308 个合法 point/end tasks；稀疏结果与 dense 参考逐项比较 `valid/reason/edge_count/paths`。CPU Debug/Release、CUDA Release 全套测试通过，直接覆盖 HT path-append kernel 的 compute-sanitizer memcheck 为 0 errors。

## 5. clean-commit 优化结果

优化后的正式 runs 均绑定 clean commit `b551a2ee9832331c85f6a8455bfb3a925b4c8605`：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T230020Z-2650158`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T230037Z-2650784`；
- d15112：`artifacts/d15112-ht-scan-20260901T230044Z-2650783`。

CPU fused 前后对比如下：

| 实例 | path-append baseline→sparse（ms） | 加速 | search baseline→sparse（ms） | 加速 | wall baseline→sparse（ms） | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 195.772 → 35.578 | 5.503× | 1,657.283 → 1,486.697 | 1.115× | 1,696.659 → 1,522.066 | 1.115× |
| rl5915 | 613.971 → 64.369 | 9.538× | 1,337.652 → 796.794 | 1.679× | 1,417.586 → 872.240 | 1.625× |
| d15112 | 5,306.771 → 207.611 | 25.561× | 7,483.758 → 2,388.545 | 3.133× | 7,793.402 → 2,694.755 | 2.892× |

d15112 的 parent prepare 从 87.379 ms 降至 2.680 ms，child normalize 从 5,193.507 ms 降至 184.656 ms。剩余时间主要随实际路径节点和 task 数增长，不再随完整 TSP 维度为每个 task 付费。

三实例的 attempted/proven/unresolved/committed、states、replies、leaf calls 和全部五路规范工作计数保持不变。baseline 与优化后每一路的活动边文件逐字节相同；删除 outer metrics timing 行后，五路 V2/HT proof 也逐字节相同。最终图哈希、活动边 SHA-256 和三个受保护 tour 哈希仍分别为文档 37 固化的值。

优化后 rl5915/d15112 的 CPU-fused `work_graph_ms` 中，未细分 host-build residual 仍为 384.056/1,079.489 ms，占 49.99%/47.93%；它已超过 path-append。[后续根 child 排除实验](39_M5_HT_Root_Child_Normalization_Profile.md)证明 dense 根规范化只占 host residual 0.41%–2.61%，下一画像转向逐 frontier state 的全维 point-candidate 扫描。
