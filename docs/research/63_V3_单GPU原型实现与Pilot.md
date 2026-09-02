# V3 单 GPU 原型实现与 Pilot

## 结论

V3 中可独立验收的 A0、A1 和验证热路径已经落地；C1 完成的是保持 DFS 规范次序的 host-window 转置原型，还不是设计稿要求的跨目标 continuation broker。正式七对 A/B 表明，当前单 GPU 混合 wavefront 相对本分支之前的 A5000 全 CUDA 基线明显改善，但仍未超过当前 8 线程 CPU 基线，因此默认调度继续保持 `wavefront`，`transposed` 仅作为显式研究后端。

所有 GPU 路径仍是候选器。实际删除只发生在同一不可变快照上的完整 CPU sidecar 重放、最小度门禁和原子 epoch 提交之后。

## 已实现内容

### A0：短路 Trace 与 replay

- 新增 `CUDAEE_HT_SHORT_CIRCUIT_TRACE_V1`，记录版本、快照哈希、目标边、完整性、AND/OR/leaf 类型、规范 child ordinal、真值和工作量。
- reader 拒绝部分树、非法 parent、共享 child、孤立节点、逆向引用、connective 真值错误、计数超限和尾随数据。
- `ht-prove`、`ht-scan`、`local-eliminate` 的 wavefront 路径可用 `--trace-output` 导出；Trace 不进入 proof，也不能授权删边。
- `ht-trace-replay --speculation-widths 1,2,4,8,all` 按规范提交次序模拟窗口内已发出的投机任务，分别报告逻辑工作、物理工作、投机节点、短路数和峰值 ready width。

### A1：紧凑 exact DP

- CPU value pass 仅保留相邻 popcount 层；只有发现严格改善后才分配完整 cost/predecessor 表并重建 witness。
- 完整 traceback pass 与 compact value pass 必须得到相同最优值和规范终态；成功 witness 再交给原有通用 verifier。
- CUDA 后端支持 `2 <= block_count <= 13`，同一 block_count 的任务一 CTA 一个，EUC_2D/CEIL_2D 使用精确整数距离。
- `k=13` 动态共享内存为 `88,704 bytes`；launch 前检查 opt-in shared-memory 上限和 `uint32` 成本范围。
- 公共 batch API 会逐任务运行 CPU compact DP 对照；集成搜索中 CUDA 阴性只返回 `unresolved` 并保留边，CUDA 阳性才运行 CPU compact、traceback 和 witness verifier。
- CPU 后端仍支持原有最多 18 blocks；设备不可用、block 超限、数值超限、分配失败或 CUDA 错误均 fail closed。

### C1：转置调度原型

- 新增 `--scheduler transposed --speculation N|auto`。
- point/end 候选复用 target 级严格次序、generation marks、Hamilton reply cache 和 end reply cache。
- 每个 AND 窗口先批量计算 leaf，再严格按规范 reply 次序消费；后续窗口可短路，窗口内多发任务只计物理工作，不占逻辑预算。
- `s=1` 在随机小图上与递归 DFS 的结论、状态/reply/leaf 计数和序列化 proof 一致；`s=4` 保持相同规范 proof。
- 当前 path-append、Hamilton/end reply 和 continuation 控制流仍在 CPU；leaf 与根 `c,d` 可用 CUDA。显式要求 transposed 的 CUDA reply/path 会安全拒绝。

这还不是完整 C1：当前没有跨 target 的 ready queue、SoA continuation arena、generation cancellation token、异构 leaf broker 或 CUDA Graph 调度轮次。每个规范状态仍触发一次 leaf batch，因此只能验证短路语义和工作量，不能提供目标中的 GPU 吞吐。

### 单 GPU并发与验证热路径

- `--target-workers N --target-devices 0` 允许多个只读 target worker 共享同一张显式 GPU；结果仍按原 target 顺序消费和提交。
- worker 数硬上限为 32；每个 worker 绑定设备 preference，线程本地 CUDA cache 不跨设备混用。
- scan 内部不再对成功 proof 连续重放三次。bound scheduler 只生成候选 sidecar，`CommitHtProofEpoch` 是修改图之前唯一一次完整 CPU 授权重放。
- 同一 epoch 的 sidecars 最多用 32 个 `jthread` 并行只读验证；全部成功后才规范化、做 degree gate 并在图副本上原子发布。错误仍按输入索引确定性报告。
- 离线 `verify` 完全不使用内存内 binding 或任何验证缓存，继续独立重放序列化 proof。

## d15112 Pilot

环境为 `cuda08` 的单张 RTX A5000，物理 GPU 2，UUID `GPU-9e271973-f616-3c82-922d-792bb30eefc0`。输入先运行完整 CPU JV，得到 159,187 条活动边；随后固定最高权重 32 targets、相同预算和成本 1,573,084 的受保护最优 tour。

### 推测宽度扫描

单 target、CPU leaf 的 transposed 结果如下。逻辑工作始终为 901 states、1,754 replies 和 901 leaf calls；物理 leaf states 随推测宽度增加。

| 宽度 | 物理 leaf states | target execution (ms) |
|---:|---:|---:|
| 1 | 901 | 766.649 |
| 2 | 1,061 | 812.939 |
| 4 | 1,467 | 917.653 |
| 8 | 2,069 | 1,044.632 |
| 16 | 3,015 | 1,309.148 |
| 32 | 4,188 | 1,664.898 |
| 64 | 4,509 | 1,761.519 |

因此 `auto` 暂时解析为 1。只有真实跨目标 broker 能让批量收益大于投机空洞时，才重新扫描这个参数。

### 32-target 算法强度

| 调度 | target workers | states | replies | leaf batches | 提交边 | target execution |
|---|---:|---:|---:|---:|---:|---:|
| wavefront CPU | 1 | 40,044 | 52,917 | 38 | 11 | 1.162 s |
| transposed CPU/leaf，s=1 | 4 | 19,498 | 35,401 | 19,504 | 18 | 17.535 s |
| transposed CPU/leaf，s=1 | 8 | 19,498 | 35,401 | 19,504 | 18 | 17.133 s |

转置原型复现了 DFS 的 18 条证明，比 wavefront 的 11 条更强，逻辑 states 减少 51.3%；但 19,504 次小 leaf batch 使总时间约 24 秒，远慢于 wavefront。它证明了“短路次序值得保留”，同时也定位出下一实现边界必须是跨目标异构 leaf 合批，而不是继续增加 speculation。

### 单 GPU后端消融

同一 32-target wavefront 的一次性 pilot：

| 路径 | target workers | target execution (ms) | total (ms) | wall (s) |
|---|---:|---:|---:|---:|
| CPU，8 OpenMP threads | 1 | 1,164.203 | 1,403.940 | 1.57 |
| CUDA candidate，其他 CPU | 1 | 1,322.154 | 1,736.461 | 1.98 |
| CUDA candidate+leaf，其他 CPU | 1 | 1,366.032 | 1,697.088 | 1.92 |
| CUDA candidate+leaf，其他 CPU | 4，2 CPU threads/worker | 1,180.418 | 1,507.068 | 1.73 |
| CUDA candidate+leaf+reply | 1 | 2,479.601 | 2,806.645 | 3.03 |
| 全 CUDA | 4 | 1,955.751 | 2,283.815 | 2.52 |

六路均为 40,044 states、52,917 replies、4,100 leaf calls 和 11 条提交边；最终边文件 SHA-256 都是 `3001cd0734256acfe31a2bc28480c403df48d4261d86818e2eafe28717d59905`，规范 proof、工作签名和受保护 tour 均通过比较。

这组一次性消融中的最佳混合路径相对旧 A5000 全 CUDA wall `2.845 s` 约为 `1.64x`；但相对同轮 CPU pilot `1.57 s` 仍慢约 10.2%。一次性结果只用于选择正式 A/B 的后端组合，不作为最终速度结论。

### Clean-commit 七对 A/B

正式实验绑定提交 `a8faebb6ed7107f2ce50b9dd4264fa1613710bce`，`git_dirty=0`，在同一 `cuda08`/GPU UUID 上先各预热一次，再按奇数对 CPU→hybrid、偶数对 hybrid→CPU 交替运行七对。CPU 使用一个 target worker 和 8 个 OpenMP cost threads；hybrid 使用四个共享单 GPU 的 target workers、每 worker 2 个 CPU cost threads，只把 candidate 与 leaf 放到 CUDA，reply/path/propagation/exact/verification/commit 保持 CPU。

| 指标 | CPU 中位数 [P25,P75] (ms) | Hybrid 中位数 [P25,P75] (ms) | CPU/Hybrid |
|---|---:|---:|---:|
| target execution | 1,157.346 [1,155.321,1,158.722] | 1,223.906 [1,212.441,1,225.691] | 0.946x |
| algorithm total | 1,398.865 [1,394.984,1,401.637] | 1,595.436 [1,548.427,1,625.286] | 0.877x |
| process wall | 1,564.663 [1,558.524,1,567.197] | 1,858.134 [1,819.663,1,861.787] | 0.842x |
| commit | 204.902 [203.720,208.499] | 217.685 [216.528,218.768] | 0.941x |

混合路径的 target execution、algorithm total 和 process wall 分别比 CPU 慢约 `5.75%/14.05%/18.76%`。相对旧 A5000 全 CUDA wall `2.845 s`，正式混合中位数仍有 `1.53x` 改善；但它没有达到相对 CPU 的 `1.25x` Go 门槛。

按同一 pair 直接计算的 CPU/Hybrid speedup 中位数为 target execution `0.946x`、algorithm total `0.873x`、process wall `0.838x`；它们与两路分别取中位数的结论一致。

两路每次都保持 40,044 states、52,917 replies、4,100 leaf calls 和 11 条提交边。两次预热与 14 次计时输出的边 SHA-256 全部为 `3001cd0734256acfe31a2bc28480c403df48d4261d86818e2eafe28717d59905`；规范 proof SHA-256 为 `ddf89f79194f6590beb88317fdb6f50cf14fc3cc0fa4ab61fefdbc7818cd9b97`，工作签名 SHA-256 为 `0b09bf354d628a4bb4d20737ea0b4250ac7c71d0242cd0b0b81b62c9c3dc3108`。每次 proof 都由 CPU 可执行文件独立重放，每次结果都通过 1,573,084 最优 tour 门禁。

原始结果位于忽略提交的 `artifacts/d15112-v3-single-gpu-ab-20260902T102225Z-3499737/`；manifest 记录 CUDA 12.6、驱动 610.43.02、GCC 16.1.1、GPU UUID、全部输入哈希和 `CUDAEE_HT_SCAN_REPORT_V18`。phase `search_ms` 是并行 targets 的逐任务耗时求和，不能与 `target_execution_ms` 墙钟相加，也不能用来计算单 GPU 端到端加速。

## 验证状态

- CPU Debug CTest `23/23` 通过，包含真实 Trace→replay 和 transposed CLI→独立 verifier。
- A5000 `cuda-sm86-release` CTest `26/26` 通过，包含 pr299 CPU/CUDA 差分、单卡多 worker 与 `k=13` exact DP。
- `compute-sanitizer --tool memcheck` 分别覆盖 k-opt/exact 和 Hamilton–Tutte，均为 `0 errors`。
- `tools/check_workspace.py`、`git diff --check`、benchmark shell 语法和 clean-worktree 门禁通过。

## 与论文 Table 7 的关系

论文历史 d15112 数据是完整协议：`166,499 -> 73,840` 边用单核 `39.6 s`，并行版本到 `73,851` 边用 44 核 `1.9 s`。本 pilot 是完整 JV 后的 159,187 条边上只扫描 32 个 targets，最后仅删除 11 或 18 条边；输入阶段、目标覆盖和最终强度都不同。

因此目前不能声称相对论文有直接加速。可报告的是相同本项目切片上的 CPU/GPU A/B，以及“耗时—提交边数”两个点。直接论文对齐仍需完整 `kh-elim -Jq` 协议或达到约 73,850 条剩余边的等强度运行。

## 可复现实验与下一步

正式单 GPU A/B 入口是 `tools/run_v3_single_gpu_ab.sh INSTANCE PHYSICAL_GPU [PAIRS]`。脚本要求 clean worktree 和最优 tour，固定 GPU UUID，先重建/重放 JV 固定点，再交替执行七对 CPU 与单 GPU混合 wavefront；每次运行都独立验证 proof、tour、边文件和规范工作签名，并输出普通及配对 speedup 的中位数、P25/P75。

下一实现切片只有一个主目标：建立跨 target 的 heterogeneous leaf broker。它必须保留规范 child commit 顺序和逻辑预算，把多个 continuation 的 cost rows 合成少量大 batch，并用 generation token 丢弃短路后的迟到结果。达到 18 条证明且相对 CPU wall 至少 `1.25x` 前，`transposed` 保持 opt-in。
