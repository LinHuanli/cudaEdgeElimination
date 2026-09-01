# M5 HT 阶段画像与混合后端

## 1. 目标与交付

提交 `e612a12` 为 wavefront 与全图 scan 增加阶段级 wall time，提交 `65f9488` 再将 c,d 候选后端与 leaf cost 后端解耦。它们不改变搜索顺序、预算、proof 内容或删除授权，只回答两个性能问题：

1. pcb3038 的约 34 秒实际消耗在哪些阶段；
2. CPU 必须完整复核的轻量候选器是否值得强制再跑一遍 CUDA。

当前 CLI 可用 `--leaf-backend auto|cpu|cuda` 单独控制 leaf cost。未提供时仍继承 `--backend`，因此旧命令语义保持不变。推荐的研究型混合配置为：

```bash
--backend cpu --leaf-backend cuda \
--reply-backend cpu --path-append-backend cpu \
--propagation-backend cpu
```

这里的推荐只针对当前“GPU 结果必须由 CPU 完整差分复核”的实现；它不是放松 verifier，也不表示这些 CUDA 候选器不再保留为正确性 oracle。

## 2. V2 计时契约

`CUDAEE_HT_SCAN_REPORT_V2` 在每个 target 和 scan 总计中记录下表字段。所有时间均由 `steady_clock` 在同步 API 外围测量，只用于诊断，不序列化进 HT proof。

| 字段 | 边界 | 是否包含其他已列阶段 |
|---|---|---|
| `candidate_ms` | c,d task 构造、CPU flags、所选后端及差分复核 | 否 |
| `work_graph_ms` | root move 与全部 frontier expansion | 是，包含下面四个 build 子阶段 |
| `leaf_ms` | `ProvePathSystemsByKOpt` 全调用 | 否 |
| `path_append_ms` | path append CPU 规范化、所选后端及差分复核 | 否 |
| `hamilton_reply_ms` | root/point Hamilton replies 的 CPU 枚举、所选后端及差分复核 | 否 |
| `end_reply_ms` | end replies 的 CPU 枚举、所选后端及差分复核 | 否 |
| `propagation_ms` | CPU 规范传播、可选 CUDA propagation 及比较 | 否 |
| `proof_extract_ms` | 从成功工作图复制规范 proof tree | 否 |
| `proof_verify_ms` | wavefront 返回前的内部完整 CPU verifier | 否 |
| `immediate_verify_ms` | scan 收到 PROVEN sidecar 后的第二次 CPU 重放 | 否 |
| `commit_ms` | batch commit 的再次重放、度数门禁、图副本与一次发布 | 否 |
| `search_ms` | 所有 `ProveEdgeByWavefrontHt` 调用的外围 wall time | 包含 candidate/work graph/propagation/extract/internal verify |
| `total_ms` | `RunHtScanEpoch` 的选择、搜索、即时重放和 commit | 包含上述 scan 内阶段 |

因此不得把 `work_graph_ms` 与 leaf/reply/path 子阶段相加。benchmark summary 另给出：

```text
host_build_residual_ms = work_graph_ms
                       - leaf_ms
                       - path_append_ms
                       - hamilton_reply_ms
                       - end_reply_ms
```

该残差包括状态容器维护、规范 child 散布、move/reply 组装和 frontier 控制。阶段计时包含 CUDA context 建立、内存管理、同步和 CPU certification，不是只报 kernel 的理想化数字。

## 3. 三路等价性门禁

`tools/run_ht_scan_benchmark.sh` 当前对同一 JV 固定点、目标切片和预算运行：

- `cpu`：所有阶段显式 CPU；
- `cuda`：candidate、leaf、reply、path append 与 propagation 全部显式 CUDA；
- `hybrid`：candidate/reply/path/propagation 用 CPU，仅 leaf cost 用 CUDA。

脚本对三路执行以下硬门禁：

1. 比较 target index、稳定 edge id、端点、状态、states、replies、leaf calls、moves 和 peak frontier；
2. 逐字节比较最终活动边文件；
3. 用独立 CPU 可执行文件重放每份外层 V2 proof；
4. 若提供最优 tour，则逐份复核成本、规范 tour hash 与零缺边。

proof 文件的 SHA-256 可以不同，因为外层运行指标包含 wall time；删除记录、内嵌 HT 证明和最终图必须通过独立重放，而不是用 proof 文件逐字节相等代替语义验证。

## 4. pcb3038 clean-commit 结果

正式 run id 为 `pcb3038-ht-scan-20260901T200634Z-2550648`，clean commit 为 `65f9488a5e80d499af8279bfddd9653d349d9e8f`，使用物理 GPU 1（RTX 4000 Ada）。输入、预算与 [8-target 基线](25_M5_HT_Target_Scan.md)相同：JV 固定点 6,704 条边、6,476 个 eligible targets，扫描最高权重的前 8 个目标。

三路工作签名均为 12,383 states、14,285 replies、9,120 leaf calls 和 5,085 moves；均得到 2 PROVEN、6 UNRESOLVED，提交同两条边。最终均为 6,702 条边、图哈希 `fe11f98414b04c0e`，活动边 SHA-256 均为 `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`。成本 137,694、tour hash `ca0238497c090a3c` 的最优巡回均为 0 缺边。

| 阶段（ms） | CPU | 全 CUDA | 混合 |
|---|---:|---:|---:|
| candidate | 49.222 | 222.047 | 54.493 |
| work graph（包含式） | 33,930.145 | 33,364.388 | 33,338.588 |
| ├─ leaf | 27,159.861 | 26,496.459 | 26,562.864 |
| ├─ path append | 286.535 | 305.039 | 286.951 |
| ├─ Hamilton reply | 6,244.948 | 6,300.536 | 6,247.397 |
| ├─ end reply | 11.057 | 31.578 | 10.949 |
| └─ host build residual | 227.745 | 230.775 | 230.427 |
| propagation | 0.229 | 4.906 | 0.280 |
| proof internal verify | 3.063 | 3.089 | 3.087 |
| immediate verify | 3.019 | 3.032 | 3.037 |
| commit | 4.516 | 4.787 | 4.768 |
| search | 33,986.571 | 33,598.814 | 33,400.920 |
| `RunHtScanEpoch` total | 33,996.285 | 33,608.829 | 33,410.891 |
| process wall | 34,031.456 | 33,697.417 | 33,501.426 |

相对 CPU，全 CUDA search 为 `1.012×`，混合为 `1.018×`；混合比全 CUDA 节省 197.894 ms。CUDA leaf 相对 CPU 为 `1.025×`，混合 leaf 为 `1.022×`。这些差异仍很小，只能用于定位瓶颈，不能声称获得显著端到端加速。

## 5. 结论

CPU 基线中 leaf 占 search 的 `79.914%`，Hamilton reply 占 `18.375%`，path append 占 `0.843%`，纯 host build residual 只占 `0.670%`。propagation、proof 抽取和三次 CPU 重放合计远低于 1%，当前不值得优先优化。

全 CUDA candidate 的额外约 173 ms 主要是首个 CUDA context 冷启动被计入第一个可见 GPU 阶段；混合模式会把同一成本移动到第一次 leaf 调用，不能把单个字段差直接解释为 kernel 性能。更重要的是 candidate/reply/path/end 的实现先完成 CPU 规范结果，再运行 GPU 并逐项比较；强制 CUDA 只会增加正确性差分成本。混合模式保留 CUDA leaf 的高算术强度部分，并避免其余重复设备工作，因而是更合理的后续性能基线。

但混合模式仍只有 `1.018×`：51,309,996 个 leaf cost cells 被拆成 1,835 个 CUDA batches，每个 target 和 frontier chunk 都独立推进。单纯调后端选择已经接近收益上限。

## 6. 下一实现切片：跨目标 leaf 合批

下一阶段应把当前“一个 target 完整跑完再处理下一个”的接口拆成可恢复状态机，而不是继续微调 propagation：

1. 为每个目标保存独立的 root candidate 游标、frontier、state/reply 预算与累计指标；
2. 各目标只推进到同一轮 leaf barrier，汇集 `(target_slot, state_index, path_system, required_edge)`；
3. 扩展 k-opt 批接口，使每个 path system 携带自己的 required edge，并按 `k`/复杂度桶合并 cost rows；
4. 一次设备调用后按稳定 `target_slot,state_index` 顺序散布 proof，再分别推进 reply/path 与下一层；
5. 任一目标预算耗尽只令该目标 `UNRESOLVED`，不得影响其他目标或授权删除；
6. 最终仍逐目标内部 verifier、scan 即时 verifier、batch commit verifier，并在不可变 snapshot 上一次发布。

验收首先要求 CPU、旧串行 CUDA、新融合 CUDA 的工作签名、内嵌 HT proof 字节和最终图完全一致；性能门禁记录 CUDA batch 数、平均/分位 cost cells、峰值 device cache 与 search wall。只有在重复运行中稳定降低 leaf wall 和 launch 数后，才扩展到 rl5915/d15112 或多 GPU。
