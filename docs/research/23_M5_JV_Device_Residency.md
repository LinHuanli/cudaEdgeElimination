# M5 JV CUDA 跨 epoch 驻留缓存

## 1. 目标与结论

基线 CUDA JV 每个 epoch 都重新分配十个 device buffer，并重复上传坐标、边端点和边权。提交 `25590af85af55fb843026285313a30d93cae1bb5` 将这些不变量及增长型 workspace 保留在主机线程所选设备上；活动位、CSR 和 witness 每轮完整覆盖，CPU verifier 与 proof 语义不变。

在同一正式协议下，进程内算法中位数加速由 `8.132× / 25.679× / 37.426×` 提升为 `11.117× / 29.788× / 40.119×`。三个实例的最终图哈希、输出 SHA-256 和 proof 重放结果均与驻留前完全相同。

## 2. 缓存边界

每个 host thread、每个 CUDA device 独立持有一个 `JvDeviceCache`。静态键逐元素比较以下全部 kernel 依赖，不只依赖概率哈希：

- `dimension` 与 `distance_type`；
- 所有整数坐标 `(integer_x, integer_y)`；
- 每条边按稳定 edge id 对应的 `(u,v,weight)`。

键命中时复用坐标、端点和边权 device buffer。以下数据属于不可变 snapshot 的动态部分，每次调用仍完整复制：

- 每条边的 `active` 位；
- `row_offsets`、`neighbors` 与 `csr_weights`；
- 初始化为 `-1` 的 witness 输出。

动态 buffer 只增长不收缩，容量不足时取 `max(required, 2*current)`。缓存明确记录 owner device；析构或 `ClearJvCudaCache` 时先切回 owner，再释放指针。任何 CUDA 分配、复制、launch 或同步错误继续抛出异常，调用方不会提交删除。

## 3. 失效与正确性测试

CUDA 单元测试覆盖四条路径：

1. 首次调用必须 `static_hit=0, workspace_hit=0`，并与 CPU 候选逐项一致；
2. 完全相同图再次调用必须双命中，resident bytes 不变；
3. 只修改 active 位并重建 CSR 时静态键仍命中，但动态数组被完整刷新，CPU/CUDA 候选仍一致；
4. 修改坐标并重新计算全部边权时静态键必须失效，重新上传后仍与 CPU 一致。

正式三实例每次运行都继续执行 GPU 候选 CPU 复核、CPU/GPU 边文件比较、独立 proof 重放。CPU Debug 15/15、CPU Release 15/15、CUDA Release 16/16 均通过；包含首轮 miss 和第二轮 hit 的 `compute-sanitizer --tool memcheck` 为 0 error。

## 4. 正式结果

运行绑定 clean commit `25590af85af55fb843026285313a30d93cae1bb5`，物理 GPU 1，五次计时。

| 实例 | 驻留前 CUDA algorithm (ms) | 驻留后 CUDA algorithm (ms) | 改善 | 驻留前/后 propose (ms) | 新 CPU/CUDA 加速 |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 2.396 | 1.741 | 27.3% | 1.106 / 0.479 | 11.117× |
| rl5915 | 7.681 | 6.639 | 13.6% | 2.777 / 1.790 | 29.788× |
| d15112 | 74.829 | 70.097 | 6.3% | 20.632 / 13.308 | 40.119× |

| 实例 | timed epoch cache hits | peak resident bytes | peak resident MiB | 最终图哈希 |
|---|---:|---:|---:|---|
| pcb3038 | 2/2 | 391,148 | 0.373 | `90d13888e351df17` |
| rl5915 | 2/2 | 1,517,168 | 1.447 | `0174cf46124ce870` |
| d15112 | 3/3 | 8,294,196 | 7.910 | `76e196dd53d887d5` |

“timed epoch cache hits”同时适用于静态键和 workspace。基准在正式计时前运行一次 CUDA warmup，因此计时阶段应全部命中；独立 CLI 的每个新进程则在首个 epoch miss，只在同一进程后续 epoch 命中。

独立 CLI wall 中位数为 `239.466 / 263.195 / 498.197 ms`，相对 CPU 为 `0.176× / 0.944× / 6.019×`，没有显示驻留版的稳态收益。这是符合边界的结果：跨进程不能保留 device cache，CUDA context 和 I/O 仍占主导。不得用进程内数据声称单次短 CLI 同样加速。

正式 run id：

- `pcb3038-jv-20260901T191030Z-2520300`；
- `rl5915-jv-20260901T191039Z-2520782`；
- `d15112-jv-20260901T191055Z-2521277`。

`pcb3038` 继续通过成本 137,694、规范哈希 `ca0238497c090a3c` 的受保护最优 tour 门禁；另两实例仍明确记录 `protected_tour none`。

## 5. 下一步

`d15112` 驻留后的 proposal 仍为 13.3 ms。下一步应分别计量动态 H2D、kernel 和 D2H，再在不改变 CPU 完整复核的前提下：

1. 用动态 `csr_edge_ids` 引用已驻留的静态 edge weights，避免每轮上传 64-bit `csr_weights`；
2. 评估 active edge id 紧凑 launch，减少已删除边线程；
3. 将 JV/HT/LP 放入常驻 orchestrator，避免短任务反复创建 CUDA context；
4. 保留 exact cache key、owner-device 释放和所有错误 fail-closed 门禁。

第 1 项已由后续的 [动态 CSR edge-id 传输](24_M5_JV_Dynamic_CSR_Edge_Ids.md)完成；本文件继续保留 `25590af` 的驻留权重基线。
