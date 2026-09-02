# M5 HT 跨目标根候选融合排除实验

## 1. 结论

当前 HT scan **不采用**跨 target 合并根 `c,d` 候选筛选的方案。d15112 的 8-target、七组交错 clean-commit A/B 中，该原型把 2,400 条 screen tasks 合成一次 CUDA launch，但候选阶段、search、total 和进程 wall 的中位数分别回退 `0.793%`、`0.764%`、`1.268%` 和 `2.514%`。配对差值的中位数也全部为正，不能把结果解释为单次噪声收益。

原型源码已由提交 `cca0b55df3e8a950bd1e0830aa58361547ce1ca7` 完整撤销。生产接口和报告格式恢复为 V13/V16，不保留实验性的 `--fuse-target-candidates`、V14 report 或 V17 benchmark 字段。路线图中的“跨目标 HT 融合”仍然存在，但范围收窄为 leaf/reply/work-graph 的实质性重复工作共享；根 `c,d` screen 不再作为候选融合边界。

## 2. 假设与原型边界

逐 target 基线每次都会构造至多 300 条根 `c,d` screen tasks，上传相同坐标并启动一次 CUDA kernel。实验假设是把同一不可变 scan 切片的 8 个目标展平为：

```text
(target_edge, c, d, active) × 2,400
                │
                ▼
       单次 CUDA flags 筛选
                │
                ▼
      按 target 稳定切片并规范收尾
```

原型只合并根候选的设备筛选，严格保留以下语义：

1. 每个 target 的原始顺序、候选上限、搜索预算和 proof 顺序不变；
2. 主机仍逐 target 构造完整 tasks 和 CPU flags，并用 CPU 结果全量认证 CUDA flags；
3. `FinalizeCdCandidates`、中点评分、稳定排序和 reply-product 计算仍逐 target 执行；
4. prepared candidates 绑定实际 graph 对象、规范 target 和所有根选项，错配即拒绝；
5. 批次有 4,096 targets 和 1,000,000 tasks 的硬上限，allocation/长度失败安全回退；
6. target 之间继续共用既有 graph/hash bindings，每个 target 后仍执行快照内容哈希守卫；
7. HT sidecar 即时公开重放、V2 整批原子提交和最优 tour 门禁不变。

这使实验只回答一个问题：减少根候选的 CUDA launch 与坐标上传，能否覆盖新增展平、较宽 task record、切片和调度成本。

## 3. 实验协议

- 原型 clean commit：`a520591f0679dece36ad2884835aa2b4e62b7674`；
- A/B 方式：同一个 clean 二进制内显式关闭/开启融合，消除跨提交构建差异；
- 设备：物理 GPU 1，NVIDIA RTX 4000 Ada Generation；
- 实例：d15112，先达到 JV 固定点，再按 `weight-desc` 取前 8 个 HT targets；
- 后端：根候选、Hamilton reply、leaf cost 与成功状态 propagation 使用既有 CUDA 候选路径，全部保持 CPU 完整认证；
- 预算：每 target 最多 2,000 states，其他预算与生产 V13 基线相同；
- 计时：先做污染检测和 clean 单次门禁，再按 `off/on` 与 `on/off` 交替执行七对；
- 原始目录：`artifacts/ht-target-candidate-clean-ab-20260902-cZkZ6M`；
- 原始表：`metrics.csv`，SHA-256 `ca133355e9b93af550904c5d1ecc8ed2f561d1227c99b7b9ed6c8689983f6b47`。

此前的一次污染 smoke 位于 `artifacts/ht-target-candidate-prototype-20260902-cdvh8v`，曾显示候选阶段 `173.06 -> 142.15 ms`。clean 单次复测立即变为 `171.774 -> 176.893 ms`，因此没有采用有利的单次样本，而是执行上述七对交错协议。

## 4. 正确性门禁

原型存在期间通过：

- CPU Debug 20/20、CPU Release 20/20；
- CUDA Release 23/23；
- CUDA Debug 在物理 GPU 2 上执行 compute-sanitizer memcheck，0 errors；
- on/off 每次均为 10,003 states、11,851 replies、1,114 leaf calls、751 moves；
- 每次均尝试 8 个 targets，证明并提交相同 2 条边；
- 14 份 V2 proof 均从原始图由 CPU 独立重放；去掉唯一允许变化的 epoch timing 后，规范 proof 内容一致；
- 14 份活动边文件 SHA-256 均为 `39abcae8832b5eca8cee278237eafe52ed2f053fbe2abefd7f8593e746a189b4`；
- 每份结果的 d15112 最优 tour 成本均为 1,573,084，活动图缺边数为 0；
- 每次最终内容哈希均为 `29c3b8fccaf1a3fc`。

因此这里排除的是该调度边界的性能方案，不是正确性方案。

## 5. 七对交错结果

下表分别对 off/on 的七次观测取中位数。阶段中位数不能逐项相加推导 total。

| 指标 | 逐 target 基线 | 跨 target 根候选融合 | 变化 |
|---|---:|---:|---:|
| candidate | 149.537028 ms | 150.722576 ms | **回退 0.793%** |
| search | 777.077075 ms | 783.015591 ms | **回退 0.764%** |
| total | 940.210696 ms | 952.131566 ms | **回退 1.268%** |
| process wall | 1168.401152 ms | 1197.777664 ms | **回退 2.514%** |

为避免两个总体中位数来自不同 pair，再对每个相同 run id 计算 `on - off`，七个配对差值的中位数为：

| 指标 | 配对差值中位数 |
|---|---:|
| candidate | +1.055 ms |
| search | +3.196 ms |
| total | +8.419 ms |
| process wall | +32.230 ms |

所有方向都一致回退，未达到“至少不劣”的保留门槛。

## 6. 原因判断

一次 CUDA launch 和一次坐标上传并不是当前根候选阶段的主要成本。融合后仍无法消除：

- 每个 target 的邻域选取与 300 条 task 构造；
- CPU 规范 flags 的完整计算；
- `FinalizeCdCandidates` 的中点评分、严格排序、截断与 reply-product；
- 每个 target 独立的 wavefront 资源预算、状态图和 proof 生命周期；
- 短命 CLI 进程的 CUDA context 与其他后端初始化成本。

同时，融合 record 从只含 `(c,d,active)` 扩宽为每条都携带 target edge；主机还要展平、累计 offset 并重新切片。省掉的七次小 launch 无法覆盖这些固定成本。现有 CPU-fused 路径的根候选阶段约为 10.3 ms，进一步说明把小规模根 flags 单独搬到跨目标 GPU 批次并非有效优化方向。

更重要的是，这个方案没有共享真正昂贵的工作：不同 targets 之间反复出现的 leaf cost rows、Hamilton/end replies 以及相似的 work-graph 子结构。后续设计必须先画像这些重复量，再决定缓存、去重或批处理边界。

## 7. 决策与重新评估门槛

当前决策如下：

1. 完整删除批量根候选 API、prepared-candidate binding、CUDA 宽记录 kernel、CLI 开关、实验报告字段和对应测试；
2. 保留逐 target 根 `c,d` 候选路径及其 CPU 全量认证；
3. 跨目标研究只继续 leaf/reply/work-graph 共享，必须保持每目标预算、顺序、快照守卫和 proof 字节语义；
4. 不用该负结果否定未来的设备常驻 scan，只否定当前“仅融合根 flags”方案。

只有同时满足以下条件时才重新打开根候选融合：

1. graph、坐标和 CUDA context 已在完整 scan 生命周期内常驻，融合不再依赖短命进程初始化；
2. 画像证明根候选 screen 本身成为 material hotspot，而不是 CPU 规范收尾或后续 work graph；
3. 批次能复用 target 数据而不为每条 task 重复携带宽记录；
4. pcb3038、rl5915、d15112 的 clean-commit 交错 A/B 获得稳定端到端收益；
5. CPU/CUDA 全量差分、独立 proof 重放、活动边逐字节比较、最优 tour 和 sanitizer 门禁全部通过。

在此之前，跨目标 HT 的下一实验切片应从 leaf/reply 重复度画像开始，而不是恢复本原型。
