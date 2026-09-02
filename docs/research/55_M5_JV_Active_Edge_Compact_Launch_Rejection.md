# M5 JV 活动 edge-id 紧凑启动排除实验

## 1. 结论

当前 JV CUDA 路径**不采用**每个 epoch 在主机端构造活动 edge-id 紧凑数组的方案。d15112 稳态五次计时中，该原型把实际启动行数从 499,497 降到 484,885，仅减少 2.925%；kernel 中位数改善 1.272%，但新增的主机构造与传输成本使 `propose_ms` 回退 7.722%，最终算法中位数回退 1.052%。

原型源码已完整撤销，生产代码继续上传全长 active bitmap，并以稳定原始 edge id 写入候选和 proof。负结果保留在项目内 artifact 中，避免后续重复实现同一路径。

## 2. 假设与原型边界

基线的 JV kernel 每个 epoch 为 `graph.edges.size()` 个稳定槽位启动线程，inactive 线程读取 active bitmap 后立即退出。实验原型改为：

1. 主机按原始 edge id 严格递增地构造 `active_edge_ids`；
2. 只为活动行启动 kernel，并从逻辑行映射回稳定原始 edge id；
3. witness 使用紧凑逻辑行，收集候选时再映射回 proof edge id；
4. 动态 active-id/witness workspace 按活动边数增长；
5. 继续使用原有不可变静态图缓存、动态 CSR、CPU 候选全量差分与 proof 重放契约。

为排除明显的实现噪声，依次测试了三版：直接临时紧凑数组、复用主机 allocation，以及最终的复用 allocation + kernel 内初始化 witness。最后一版省去了 witness 的 H2D 初始化，仍未消除端到端回退。

## 3. 实验协议

- 基线：clean commit `5bf92d6bb4bbe1145ad6f0d538306e556c3659dd`；
- 原型：基于同一提交的未提交工作树，仅包含上述紧凑启动改动；
- 构建：CUDA Release；
- 设备：物理 GPU 1，NVIDIA RTX 4000 Ada Generation；
- 实例：`d15112`，输入 166,499 条边；
- 命令：`cudaee_jv_benchmark .../d15112.tsp.gz .../d15112.all.edg.gz 5`；
- 每个二进制先执行 CPU/CUDA warmup，再交错记录五次 CPU/CUDA；
- 每次运行都要求 CPU/CUDA 候选结果、最终 active 位、最终哈希和 proof 重放一致。

artifact：`artifacts/jv-active-edge-compact-20260902-STwMLc`。关键原始表：

- `baseline.d15112.csv`，SHA-256 `badbb5e3cb710ecc3041d5a217dbf9a51a5588bfcea1b281dec93db3f3e5be87`；
- `compact-device-init.d15112.csv`，SHA-256 `e42c5cde608c3798ee94f306aadc50a5f4ca308abe6848887d483d1b338073d4`。

## 4. 结果

下表均为五次 CUDA 运行中位数。阶段值分别取中位数，不能逐项相加推导算法总值。

| 指标 | 全长 active bitmap | 最终紧凑原型 | 变化 |
|---|---:|---:|---:|
| algorithm | 63.425767 ms | 64.093234 ms | **回退 1.052%** |
| propose | 11.408986 ms | 12.289997 ms | **回退 7.722%** |
| H2D | 1.952577 ms | 2.366408 ms | **回退 21.194%** |
| kernel | 6.614268 ms | 6.530124 ms | 改善 1.272% |
| D2H | 0.541182 ms | 0.622227 ms | 回退 14.976% |
| 三个 epoch 的实际启动行 | 499,497 | 484,885 | 减少 2.925% |

两条路径都扫描 484,885 个逻辑活动边、提交 7,312 条删除并保留 159,187 条活动边，最终内容哈希均为 `76e196dd53d887d5`。因此这里排除的是当前数据分布和实现边界下的性能方案，而不是正确性方案。

中间两版也没有出现收益：

| 原型 | algorithm | propose | H2D | kernel |
|---|---:|---:|---:|---:|
| 直接紧凑数组 | 65.371648 ms | 12.870832 ms | 2.935558 ms | 6.566806 ms |
| 复用主机 allocation | 66.204547 ms | 13.043369 ms | 2.928779 ms | 6.532002 ms |
| 再改为设备初始化 witness | 64.093234 ms | 12.289997 ms | 2.366408 ms | 6.530124 ms |

## 5. 原因判断

d15112 的三个 JV epochs 中，inactive 槽位只占 2.925%。全长路径的 inactive 线程只执行一个规则、连续的 bitmap 读取和分支；紧凑路径则必须在每轮：

- 遍历全部 host edge slots 并生成新的索引流；
- 传输 32 位 edge id，而不再传输可直接索引的 active bitmap；
- 在 kernel 和候选收集端各增加一次间接寻址；
- 处理不同长度的 pageable host 区间。

测量显示 0.084 ms 的 kernel 中位数节省无法覆盖约 0.881 ms 的 proposal 增量。H2D 和 D2H 单段值也受同步 pageable 传输抖动影响，但三种原型的总 proposal 都一致回退，足以支持当前排除决策；没有理由为这 2.925% 的行数缩减继续引入 pinned-memory 生命周期和额外缓存复杂度。

## 6. 重新评估门槛

只有同时满足以下条件时才重新打开该方案：

1. 真实多 epoch 工作负载的 inactive 比例显著高于本次 2.925%，并先用画像给出可覆盖紧凑化固定成本的下界；
2. 活动索引能在设备端增量维护，或由其他必需阶段免费产出，而不是每轮在主机全图扫描；
3. 不改变稳定原始 edge id、CPU verifier、proof 顺序和不可变 snapshot 契约；
4. 在 pcb3038、rl5915、d15112 至少三实例 clean-commit A/B 中获得稳健端到端收益，并通过现有差分、重放和 sanitizer 门禁。

在此之前，路线图不再把“活动 edge-id 紧凑 launch”列为 pending；后续资源优先投入跨目标 HT 工作共享、单卡完整 sweep 和多 GPU 静态切片。
