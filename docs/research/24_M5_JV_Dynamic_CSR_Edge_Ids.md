# M5 JV 动态 CSR edge-id 传输

## 1. 目标与结论

跨 epoch 驻留版本仍在每轮上传 `int64 csr_weights`。这些值与按稳定 edge id 驻留的静态 `edge_weight` 完全相同，重复保存和传输没有提供新的 snapshot 信息。提交 `41fecebfbf05211c1b447fdf9bd7c78f41bd5a55` 改为上传 `int32 csr_edge_ids`，kernel 通过 edge id 读取已驻留权重，并增加同步 H2D、kernel、D2H 三段 wall time。

CPU verifier、proof V1/V2 格式、epoch 提交顺序和 exact 静态缓存键均未改变。三组正式运行的最终图哈希、输出 SHA-256 与前两轮完全相同。

## 2. 数据契约

`GraphSnapshot::RebuildCsr` 为每个活动无向边生成两个邻接项，并同步写入：

- `neighbors[offset]`：另一端节点；
- `csr_edge_ids[offset]`：稳定 edge id；
- `csr_weights[offset]`：对应 `edges[edge_id].weight`。

CPU 路径继续使用 `csr_weights`，作为独立实现和 verifier 的输入。CUDA 动态区只复制 `row_offsets/neighbors/csr_edge_ids`，并按

```text
edge_weight[csr_edge_ids[offset]]
```

取得相同整数权重。静态键仍逐元素比较所有边的 `(u,v,weight)`；只修改坐标或边权会使缓存失效，只修改 active 位则完整刷新 CSR。因此该优化不是用哈希或陈旧值替代 snapshot。

首次精确容量的驻留字节公式由单元测试固定为

```text
24 * edge_count + 20 * node_count + 4 + 8 * directed_adjacency_count
```

其中动态邻点与 edge id 均为 32 位。测试会拒绝退回 64 位 CSR 权重副本的实现。

## 3. 计时边界

新增指标只用于当前运行观测，不写入 proof：

- `jv_h2d_ms`：每轮五个动态数组的同步 H2D，包括 active、row offsets、neighbors、edge ids 和初始化为 `-1` 的 witnesses；
- `jv_kernel_ms`：launch、错误检查和 `cudaDeviceSynchronize`；
- `jv_d2h_ms`：完整 witness 数组的同步 D2H。

`propose_ms` 还包括 exact 静态键比较、workspace 检查、host staging 和候选收集，因此通常大于三段之和。静态图首次上传不计入动态 H2D；正式进程内计时在 warmup 后全部命中缓存。

## 4. 正式结果

物理 GPU 1（RTX 4000 Ada），clean commit `41feceb`，每实例预热一次、计时五次。

| 实例 | 原 CUDA algorithm (ms) | edge-id 后 (ms) | 改善 | 原/新 propose (ms) | 新 CPU/CUDA 加速 |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 1.741 | 1.728 | 0.7% | 0.479 / 0.467 | 11.194× |
| rl5915 | 6.639 | 6.503 | 2.1% | 1.790 / 1.671 | 30.517× |
| d15112 | 70.097 | 67.651 | 3.5% | 13.308 / 12.614 | 41.695× |

| 实例 | H2D (ms) | kernel (ms) | D2H (ms) | 原/新驻留 bytes | 显存下降 |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 0.177 | 0.201 | 0.034 | 391,148 / 336,084 | 14.1% |
| rl5915 | 0.450 | 0.882 | 0.077 | 1,517,168 / 1,284,024 | 15.4% |
| d15112 | 2.965 | 6.628 | 0.513 | 8,294,196 / 6,962,204 | 16.1% |

阶段值和总值分别取五次运行的中位数，不能逐行严格相加。`d15112` 的 H2D 仍占 proposal 的约 23.5%，kernel 约占 52.5%；剩余主要是 exact cache-key 比较、host staging 和候选收集。

独立 CLI wall 中位数为 `241.168/282.079/529.889 ms`，相对 CPU 为 `0.282×/0.880×/5.692×`，P95 还受到共享节点抖动影响。它继续证明短进程不受益于跨进程不存在的缓存，不能用来否定或夸大进程内稳态改动。

正式 run id：

- `pcb3038-jv-20260901T192137Z-2526909`；
- `rl5915-jv-20260901T192146Z-2527269`；
- `d15112-jv-20260901T192156Z-2527636`。

最终图哈希仍为 `90d13888e351df17 / 0174cf46124ce870 / 76e196dd53d887d5`，输出 SHA-256 仍为：

- pcb3038：`9b2229bc047f4fa107a474bf8201b7b086db11df64485217d36c67e53c326a54`；
- rl5915：`54472ebd8115ee84482b61276ade00bc9d24fe35d730e34d2cb4067783639f4d`；
- d15112：`8ab757da10cf5b19258620b0f87351baa0f260e1ee7d10e8caf3a78528a0cd14`。

pcb3038 继续通过成本 137,694、规范 tour 哈希 `ca0238497c090a3c` 的受保护最优巡回门禁；rl5915 与 d15112 仍没有来源明确且重新计算通过的本地最优 tour，必须标为 `protected_tour none`。

## 5. 验证与下一步

- CPU Debug/ASan 15/15；
- CPU Release 15/15；
- CUDA Release 16/16，包含 pr299 CPU/GPU 差分；
- `compute-sanitizer --tool memcheck` 0 error，覆盖首轮 miss 与次轮 hit；
- 正式运行逐份由独立 CPU 进程重放 proof 并逐字节比较 CPU/CUDA 输出。

动态 CSR 传输优化至此完成。下一步优先级为：

1. 对全活动 edge-id 列表做紧凑 launch 试验，比较节省空线程与新增 compaction/传输成本；
2. 为全图 HT 增加确定性目标选择、批次预算、sidecar 汇集和不可变 epoch 提交；
3. 让长驻 orchestrator 串联 JV/HT/LP，避免短 CLI 的 context 初始化；
4. 只有单卡端到端基线稳定后再做多 GPU 分片。
