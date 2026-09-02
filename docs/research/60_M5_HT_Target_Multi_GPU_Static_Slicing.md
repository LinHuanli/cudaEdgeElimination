# M5 HT 目标级多 GPU 静态切片

## 1. 结论与范围

提交 `3e9571d` 为 `RunHtScanEpoch`、`ht-scan` 和 `local-eliminate` 增加了显式
`target_devices` 配置。一个不可变 HT target 切片可以按目标相对序号轮转分配给多个固定 GPU
worker；所有 worker 完成后，主线程仍按原目标规范序执行成功 proof 的 CPU 重放，并通过原有
`CommitHtProofEpoch` 一次性提交。

这完成的是**单进程、单快照、目标级静态多 GPU 基线**。它不改变 Hamilton–Tutte 搜索语义、
逐目标预算、proof 字节顺序、最小度门禁或原子提交规则，也不实现以下能力：

- 跨 target 的 leaf/reply 语义结果共享；
- 跨 GPU 的单个 target 工作图拆分；
- 动态 work stealing 或按历史成本加权调度；
- 常驻跨 epoch worker 池；
- M3.1 LP 删除授权或 LP—JV—HT 固定点。

未传 `--target-devices` 时完全保留历史顺序路径和当前线程的 CUDA 自动选卡/驻留缓存语义。
因此多 GPU 是显式启用的研究能力，不会静默改变现有命令的资源占用。

## 2. 用户接口与可见设备

CLI 接受逗号分隔、无重复的进程内 CUDA ordinal：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-UUID-0,GPU-UUID-1 \
build/cuda-sm86-release/cudaee ht-scan \
  --tsp INSTANCE.tsp --edges INPUT.edg \
  --output OUTPUT.edg --proof OUTPUT.proof --report OUTPUT.report \
  --max-targets 32 --target-devices 0,1 \
  --backend cuda --leaf-backend cuda --reply-backend cuda \
  --path-append-backend cuda --propagation-backend cuda \
  [HT wavefront budgets]
```

`0,1` 指 `CUDA_VISIBLE_DEVICES` 映射后的**可见 ordinal**，不是未经映射的物理编号。正式实验用
UUID 构造可见顺序，避免重启、PCI 枚举或远端主机差异改变物理卡绑定。相同参数也由
`local-eliminate` 透传到每个 HT stage：

```bash
build/cuda-sm86-release/cudaee local-eliminate \
  --tsp INSTANCE.tsp --edges INPUT.edg \
  --output OUTPUT.edg --proof OUTPUT.proof --report OUTPUT.report \
  --max-jv-rounds 100 --max-ht-epochs 100 --max-targets 32 \
  --target-devices 0,1 [HT wavefront budgets]
```

配置约束如下：

- 空列表表示历史顺序路径；
- ordinal 必须非负、互异且位于当前可见设备范围；
- 最多 32 个 worker；
- 实际 worker 数为 `min(target_devices.size(), attempted_targets)`；
- 没有目标时 worker 数为 0；
- CPU-only 构建或 CUDA 构建运行时没有可见设备时，显式设备列表失败关闭；
- 只给一个设备时仍创建固定设备 worker，但 `target_parallel=0`。

## 3. 调度结构

入口先在调用线程完成目标选择和两个强类型只读绑定：

```text
GraphSnapshot（只读）
    |
    +-- 构造 KOptSnapshotBinding / HtGraphValidationBinding
    |
    +-- 稳定选择 targets[offset, offset + count)
             |
             +-- target 0, W, 2W ... -> worker 0 -> visible device D0
             +-- target 1, W+1 ...    -> worker 1 -> visible device D1
             +-- ...
             |
             v
       全部 worker join；异常按目标序检查
             |
             v
       按原 target 顺序消费结果和 CPU 重放成功 proof
             |
             v
       再查快照哈希 -> CommitHtProofEpoch 原子提交
```

静态分配公式为：

```text
worker(relative_target) = relative_target mod worker_count
```

结果槽位按 `relative_target` 预先确定，worker 只写自己的槽位。线程完成顺序不会改变 attempt
顺序、proof 顺序或提交顺序。默认空列表走原来的直接循环，不创建线程，避免为单卡已有用户引入
不必要的生命周期和性能变化。

## 4. CUDA 设备亲和性

新增的线程局部设备偏好由以下内部接口维护：

- `CudaDevicePreferenceForCurrentThread()`；
- `SetCudaDevicePreferenceForCurrentThread()`；
- `VisibleCudaDeviceCount()`。

worker 启动后先调用 `cudaSetDevice(assigned_device)`，随后 HT `c,d`、continuation
propagation、k-opt cost、path compatibility、path append、Hamilton reply 和 end reply 的 CUDA
选择入口都会优先服从该线程偏好；JV 的共用选卡入口也遵循同一约束。没有线程偏好时，各后端仍按
原规则自动选择设备。

leaf 和 reply 的驻留缓存原本就是线程/设备本地，因此不同 worker 不共享可变 CUDA buffer。
同一 worker 在一个 scan 内处理 `w, w+W, ...` 时可以复用自己的缓存。当前每次
`RunHtScanEpoch` 都新建并销毁 worker；所以显式多 GPU 的线程局部缓存不跨 Local Elimination
epochs 保留。这是后续常驻 worker 池可优化的固定开销，不影响正确性。

## 5. 正确性与失败关闭

多 GPU 路径保留以下授权链：

1. 所有 target 都读取同一个未修改的 `GraphSnapshot`；
2. GPU 仍只生成候选或传播结果，既有 CPU 全数组/整数矩阵认证不变；
3. 每个 `PROVEN` sidecar 在主线程按目标序调用公开 `VerifyHtRecursiveProof`；
4. 全部即时重放成功后，`CommitHtProofEpoch` 再整批 CPU 重放；
5. 只有随后通过规范目标、最小度和确定性 CSR 重建门禁的记录才发布。

并发路径还增加了以下防护：

- 每个 target 搜索结束后检查图内容哈希；
- 全部 worker join 后、提交前再次检查图内容哈希；
- worker 入口捕获全部异常，任何异常都不能逃出线程并触发 `std::terminate`；
- 一个 worker 失败后不再启动其后续 target，但会等待其他已启动的只读工作安全结束；
- 主线程按目标槽位顺序重抛首个已记录异常；
- 任一设备绑定、CUDA/CPU 差分、`INVALID`、proof 重放或内容守卫失败时都不进入 commit；
- `local-eliminate` 仍在外层图副本上执行，因此失败不会发布此前阶段的半成品。

这里没有并发 proof 提交，也没有用完成先后决定成功 move；并行只覆盖互相独立的 target 搜索。

## 6. 报告语义

`CUDAEE_HT_SCAN_REPORT_V17` 新增：

- `target_devices`：`auto` 或显式可见 ordinal 列表；
- `target_workers` 与 `target_parallel`；
- `target_execution_ms`：并行 target 搜索加主线程顺序即时 proof 重放的墙钟；
- 每个 attempt 的 `assigned_device`；
- propagation、leaf cost、path append、Hamilton reply、end reply 的实际 selected device。

某个 target 没有进入某条 CUDA 路径时，对应 selected device 为 `-1`，不能误报为使用了设备。
正式双卡审计要求每个非负 selected device 都等于 assigned device，并要求两名 worker 都至少有一条
CUDA 路径实际执行。

原有 `search_ms` 是逐 target `attempt.search_ms` 的**求和**。并行运行时它可大于
`target_execution_ms`，也可能因两个 worker 争用 CPU/内存带宽而上升；它不再表示 stage 墙钟。
多 GPU 加速必须比较 `target_execution_ms`、`total_ms` 和外部进程 `wall_ms`，不能用
`search_ms` 求和直接计算。

`CUDAEE_LOCAL_ELIMINATION_REPORT_V2` 在 HT stage 增加相同的 worker/并行/目标执行墙钟字段；
`CUDAEE_HT_SCAN_BENCHMARK_SUMMARY_V20` 也保留各路径的 `target_execution_ms`。这些计时只用于
诊断，不写入删除授权语义。

## 7. 自动化门禁

单元和集成测试覆盖：

- 默认空列表保持历史顺序、proof 字节和图结果；
- 单个显式设备使全部实际 CUDA 路径固定到 ordinal 0；
- 两个可见设备和三个 targets 的分配严格为 `0,1,0`；
- 双卡与顺序运行的 target、状态、states、replies、leaf calls、moves、proof 字节和最终图一致；
- 重复、负数、越界、超过上限的设备配置拒绝；
- CPU-only 构建和运行时无可见设备的显式列表拒绝且图不变；
- 空 target 切片报告 0 worker；
- Local Elimination stage 正确透传指标。

在 `cuda-small0` 的两张空闲 RTX A4000 上，绑定提交
`caed66059d024c514b0e3fcb017450f2e796c140` 的门禁结果为：

- `cuda-sm86-release` 完整 CTest：23/23；双卡可见，因此实际执行三 target 双 worker 分支；
- CUDA Debug `cudaee_hamilton_tutte_unit` 的 compute-sanitizer memcheck：0 errors；
- CPU Debug/ASan 和 CPU Release：各 20/20；
- CUDA 构建在 `CUDA_VISIBLE_DEVICES=''` 下的运行时失败关闭分支通过。

A4000 是 `sm_86`。构建系统因此用缓存变量 `CUDAEE_CUDA_ARCHITECTURES` 控制目标架构，并新增
`cuda-sm86-release` / `cuda-sm86-debug` presets；默认 `cuda-release` / `cuda-debug` 仍生成
`sm_89`。门禁用 `cuobjdump` 确认远端静态库只有 `sm_86` cubin，避免把“没有可加载 kernel”
误判成算法错误。

## 8. 正式 A/B 协议

`tools/run_ht_target_multigpu_ab.sh` 比较同一进程可见设备集合下的：

- 单 worker：`--target-devices 0`；
- 双 worker：`--target-devices 0,1`。

脚本要求 clean worktree 和显式最优 tour，按 GPU UUID 固定可见顺序，并执行：

1. 两张物理卡的显存/利用率空闲门禁；
2. 单卡和双卡各一次预热；
3. 奇数 pair 按单卡→双卡、偶数 pair 按双卡→单卡交替；
4. 每次运行后用 CPU Release 独立重放 proof；
5. 每次检查最优 tour 成本、节点置换和输出边完整性；
6. 比较活动边文件、去计时 proof、逐 target 工作签名；
7. 审计 assigned/selected devices 与两卡实际覆盖；
8. 输出中位数、配对中位改善、输入/输出哈希、Git 状态和硬件清单。

正式运行命令为：

```bash
CUDAEE_CUDA_PRESET=cuda-sm86-release \
CUDAEE_HT_MAX_TARGETS=32 \
CUDAEE_BENCHMARK_TOUR=artifacts/d15112-ht-scan-20260901T231720Z-2662848/protected.opt.tour \
tools/run_ht_target_multigpu_ab.sh d15112 0 1 7
```

## 9. d15112 双 A4000 结果

两张卡在首尾采样均为 `1 MiB / 0%`，型号为 NVIDIA RTX A4000 16 GiB，CUDA toolkit
12.6，driver 610.43.02。每组均为七对交替计时：

| targets | 指标 | 单卡中位数 (ms) | 双卡中位数 (ms) | 中位数加速 |
|---:|---|---:|---:|---:|
| 8 | target execution | 856.046 | 758.048 | 1.129× |
| 8 | algorithm total | 1,159.336 | 1,113.677 | 1.041× |
| 8 | process wall | 1,453.820 | 1,463.749 | 0.993× |
| 32 | target execution | 3,208.626 | 2,564.352 | 1.251× |
| 32 | algorithm total | 3,739.995 | 3,081.042 | 1.214× |
| 32 | process wall | 4,043.140 | 3,495.274 | 1.157× |

32-target 的配对中位改善分别为 `20.317% / 17.980% / 13.407%`。规范工作量在所有单/双卡
运行中固定为 40,044 states、52,917 replies、4,100 leaf calls 和 11 条提交边。七对计时、
两次预热和 JV 基线共执行 17 次独立 proof 重放及 17 次受保护 tour 检查；输出哈希为：

- 活动边：`3001cd0734256acfe31a2bc28480c403df48d4261d86818e2eafe28717d59905`；
- 规范 proof：`ddf89f79194f6590beb88317fdb6f50cf14fc3cc0fa4ab61fefdbc7818cd9b97`；
- 工作签名：`9a80d2ff86c5f96412bfcfdd37a412500f5496d72d8f81dbbd6e3918d10418bd`。

正式 artifact 为
`artifacts/d15112-ht-target-multigpu-ab-20260902T060009Z-917846`。8-target 的补充 artifact 为
`artifacts/d15112-ht-target-multigpu-ab-20260902T055841Z-916662`。

双卡的 `search_ms` 求和从 2,732.073 ms 增至 3,632.211 ms，说明并发 worker 会放大 CPU
认证、主机建图或共享内存带宽成本；但两条静态链重叠后，目标阶段墙钟仍下降 20.079%。8-target
规模的固定进程开销足以覆盖收益，32-target 才出现清楚的端到端改善。因此当前结论是：静态切片
值得保留，但不能把两张卡等同于 2× 加速。

## 10. 已知限制与下一步

1. 轮转只平衡 target 数，不平衡实际 states/replies/leaf 成本；下一步可用不影响规范消费顺序的
   预估权重做静态 LPT 分配，或引入只调度索引的动态队列。
2. 两个 worker 各自执行 CPU 精确认证；`OMP_NUM_THREADS × worker_count` 必须结合物理 CPU 核数
   设置，避免过量订阅。正式门禁固定每 worker 4 个 OpenMP threads。
3. 当前 worker 每个 scan 重建，跨 Local Elimination epochs 不保留线程局部 CUDA cache；常驻池必须
   先证明新快照键失配能完整刷新，再评测收益。
4. 32-target 结果来自共享服务器上的两张空闲 PCIe A4000；交替 A/B 已抑制时间漂移，但其他 GPU
   作业仍可能带来 CPU/PCIe 噪声。论文数字应在整机空闲时扩大 pairs，并补跑 pcb3038、rl5915。
5. 多 GPU 只并行独立 target，单个重 target 仍只能使用一张卡；跨 target work-graph/leaf 共享仍是
   独立研究项，不能通过本实现宣称完成。
6. `search_ms` 继续作为规范逐目标分解量保留；所有后续性能门禁必须以
   `target_execution_ms/total_ms/wall_ms` 为主指标。
