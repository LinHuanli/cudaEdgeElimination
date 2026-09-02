# M5 HT reply CUDA 图与工作区驻留复用

## 1. 结论

提交 `4396489dbc4a01896f2a7d5f421ed76254d453ef` 为 Hamilton reply 与 end reply 的
CUDA count/write 路径增加同线程、同设备驻留缓存。它在多个 frontier batches、多个
targets 以及同一 `local-eliminate` 进程的多个只读 epochs 间复用：

- 完整整数坐标和活动图 CSR；
- Hamilton centers 与 end tasks 输入区；
- 两类 reply 输出区；
- 共用 counts、offsets 和设备错误码工作区。

缓存默认开启。d15112 的 8-target、七对交错 clean-commit A/B 中，Hamilton/end reply
阶段中位数分别下降 `4.324%/3.943%`，search/total/process wall 中位数分别下降
`1.027%/0.753%/0.450%`。23 个 reply CUDA batches 中，图命中 22 次、完整 workspace
命中 15 次；相对逐批释放基线只多驻留 805,596 bytes。14 份 proof 全部由独立 CPU
进程重放，14 次受保护最优 tour 检查均为零缺边，因此该切片保留为生产默认值。

这项工作只共享设备静态输入和分配，不缓存 Hamilton/end 的语义结果，也不融合不同 target
的工作图。GPU 输出仍是不可信候选，删除授权边界没有改变。

## 2. 原问题与共享边界

frontier reply batching 已把同一 chunk 的 centers/tasks 合成 count/write 调用，但旧实现每个
batch 仍执行完整生命周期：

```text
坐标 + CSR + tasks H2D
          |
          v
      count kernel
          |
          v
主机前缀和 + offsets H2D
          |
          v
      write kernel
          |
          v
 replies D2H + 释放全部 buffer
```

同一 HT scan 的 graph snapshot 在原子 epoch commit 前不可变，且 Hamilton/end kernels
读取相同 CSR；跨 target 重复上传和分配没有增加任何证明信息。d15112 正式切片固定产生
23 个 CUDA reply batches，因此把这些资源提升到 scan 进程生命周期是明确、可独立开关的
共享边界。

没有纳入缓存的对象包括 target edge、centers/tasks 内容、counts、offsets 和 replies 的有效
逻辑区间。它们每批仍完整覆写或清零；只有已分配容量可以复用。

## 3. 精确驻留键与内存所有权

每个主机线程维护按 CUDA device 区分的缓存，并记住首个可用设备，使后续 batch 能命中同一
device allocation。图命中必须逐项满足：

1. `dimension` 和 `distance_type` 相同；
2. 所有点的 `integer_x/integer_y` 相同；
3. 完整 `row_offsets` 和 `neighbors` 相同。

比较覆盖 reply kernels 的全部图依赖，不使用对象地址、指针生命周期或概率哈希作为正确性
条件。活动图在新 epoch 中改变时，即使复用同一个 `GraphSnapshot` 对象，也会得到 graph
miss 并重新上传。坐标或 CSR 任何一项变化都不能命中过期设备图。

graph miss 先在临时 host/device 对象中完成四个数组的构造和上传，再通过不可抛出的 move
替换缓存；中途 allocation/H2D 失败不会留下“新 host key + 旧 device 数据”的半更新状态。
设备 buffer 记录 owner device，移动和析构时先切回所属设备。析构路径不抛异常。

工作区容量不足时取 `max(required, 2 * current)`，并保持以下分区：

- Hamilton 专用：centers、neighbor-pair replies；
- end 专用：end tasks、edge replies；
- 两者共用：counts、offsets、error code；
- 图专用：x/y、CSR offsets/neighbors。

因此 `workspace_hit` 只在本次 batch 的输入、共用区和对应输出区都已有足够容量时为真。
图命中不等于 workspace 命中；pcb3038 单 target 的 5 个 batches 就可能因容量连续增长而只有
图命中。resident bytes 统计实际 allocation capacity，而不是本批逻辑长度。

## 4. A/B 基线与失败关闭

CLI 新增：

```text
--reuse-reply-cuda-cache 0|1
```

默认值为 1。值 0 仅用于同一二进制的单变量性能基线：每个 reply batch 前清除遗留状态，
batch 返回或抛错时再次清除。尾部释放发生在 phase timer 结束前，因此不会把最后一批释放
成本推迟到进程退出。公开的 `ClearHtReplyCudaCache()` 还用于单元测试隔离。

缓存路径保持原授权链：

1. CPU 先完整枚举规范 offsets/replies；
2. CUDA 使用驻留图和本批覆写的逻辑输入执行 count/write；
3. 主机检查设备错误码、前缀区间和输出边界；
4. CUDA offsets/replies 与 CPU 完整数组逐元素相等后，才把 batch 标成 `cpu_verified`；
5. 工作图始终使用 CPU 规范结果；最终 HT sidecar、scan 即时 verifier、epoch commit 与 V2
   proof 重放均不读取缓存命中信息。

allocation、设备选择、kernel、copy 或差分失败继续沿既有规则返回 `unresolved/invalid` 或在
显式 CUDA 模式抛错；缓存 miss 从不构成删除授权。

## 5. 可观测性与格式

wavefront、逐 target attempt 和 scan 总计新增：

- `reply_cuda_batches`；
- `reply_cuda_graph_cache_hits`；
- `reply_cuda_workspace_cache_hits`；
- `peak_reply_device_cache_bytes`。

命中计数只统计实际返回 CUDA 且已通过 CPU 完整比较的 batch；CPU/hybrid reply 后端必须为
零。scan 对 batch/hit 求和，对 resident bytes 取峰值。HT scan report 升级为 V15，五路
benchmark summary 升级为 V18；已撤销根候选融合实验使用过的 V14/V17 保持 tombstone，
不复用其格式编号。

`tools/run_ht_scan_benchmark.sh` 接受环境变量
`CUDAEE_REUSE_REPLY_CUDA_CACHE=0|1`，并把值写入 manifest/summary。脚本要求：

- 四条 CPU reply 路径的 CUDA cache batches 必须为零；
- 全 CUDA 路径必须至少有一个 reply batch 和非零驻留字节；
- 开启时同一只读 scan 恰有一个 graph cold batch，之后全部命中；
- 关闭时 graph/workspace 命中都为零；
- 五路规范工作签名、边、proof 重放和 tour 门禁继续全部通过。

## 6. 测试门禁

单元测试覆盖：

- Hamilton reply 首次 graph/workspace miss、第二次完整 hit；
- 修改一个整数坐标后强制 graph miss；
- end reply 首次 miss、第二次完整 hit；
- 同一 wavefront 中 Hamilton/end 共享一份图，除首批外 graph 全命中；
- 关闭缓存时 status、规范 proof 和 batch 数不变，两个命中计数均为零；
- CPU scan 的 CUDA cache 计数保持为零，attempt 与 scan 聚合一致。

提交门禁为 CPU Debug/Release 各 20/20、CUDA Release 23/23；CUDA Debug 的完整
Hamilton–Tutte 单元在物理 GPU 2 上通过 `compute-sanitizer --tool memcheck`，0 errors。
另外，pcb3038 单 target 的 V18 五路脚本已分别以缓存开启和关闭运行，两个模式均完成五路
边/工作签名比较、proof 独立重放和最优 tour 检查。

## 7. d15112 七对正式 A/B

协议如下：

- clean commit：`4396489dbc4a01896f2a7d5f421ed76254d453ef`；
- 设备：物理 GPU 1，NVIDIA RTX 4000 Ada Generation；
- 输入：d15112 先达到 JV 固定点，再按 `weight-desc` 扫描前 8 个 targets；
- 后端：c,d、reply、path append、leaf cost 与 propagation 均显式 CUDA，所有候选保持
  CPU 完整认证；
- 预算：每 target 最多 2,000 states，其余参数与固定 M5 pilot 一致；
- 预热：off/on 各一次，不计入统计；
- 顺序：七对 `off/on` 与 `on/off` 交替；
- 原始目录：`artifacts/ht-reply-cache-clean-ab-20260902-jFfmLj`；
- `metrics.csv` SHA-256：
  `fdd4228e0b277948a404e2cd8a7f9c829ec834d6da0876d61223cd8e2c547017`；
- `manifest.txt` SHA-256：
  `250e085d9d847da251a57825299e6566aeb00a4694fd0700fb0f76b147a232d4`；
- `analysis.txt` SHA-256：
  `303ae89772670a760a5e8d940622e77d8cc92fe034b343558e16ecb410f8a474`。

下表分别对 off/on 的七次观测取中位数；“配对改善”先按相同 run id 计算
`(off-on)/off`，再取七个百分比的中位数。各阶段中位数不能相加推导 total。

| 指标 | 逐批释放 | 驻留复用 | 中位数改善 | 配对改善中位数 |
|---|---:|---:|---:|---:|
| Hamilton reply | 238.188236 ms | 227.889106 ms | 4.324% | 4.039% |
| end reply | 73.257092 ms | 70.368873 ms | 3.943% | 4.446% |
| search | 785.739305 ms | 777.668380 ms | 1.027% | 1.449% |
| total | 949.960460 ms | 942.805780 ms | 0.753% | 1.220% |
| process wall | 1171.392 ms | 1166.116 ms | 0.450% | 0.661% |

每次运行都产生 10,003 states、11,851 replies、1,114 leaf calls，尝试 8 个 targets，
证明并提交相同 2 条边。off/on 的缓存指标固定为：

| 指标 | off | on |
|---|---:|---:|
| reply CUDA batches | 23 | 23 |
| graph cache hits | 0 | 22 |
| workspace cache hits | 0 | 15 |
| peak resident bytes | 13,377,980 | 14,183,576 |

14 份活动边文件 SHA-256 均为
`39abcae8832b5eca8cee278237eafe52ed2f053fbe2abefd7f8593e746a189b4`；去掉 proof 中唯一
允许变化的 `metrics` 行后，规范 SHA-256 均为
`f5bd5565f2f9dfe3c01fc0bc5c419dae648b252f65725433a8880060913cbe64`。全部 proof 从
159,187 条 JV 固定点边独立重放到 159,185 条，最终内容哈希均为
`29c3b8fccaf1a3fc`。d15112 的 1,573,084 最优 tour 在全部输出中均为零缺边。

wall 改善只有 0.450%，不能宣传为大幅端到端加速；保留依据是两个直接目标阶段及其配对
统计均稳定改善，同时内存增量只有约 0.768 MiB，且实现也为后续设备常驻 scan 提供必要基础。

## 8. 决策与下一切片

当前决策：

1. 默认开启精确图与增长 workspace 驻留；保留显式关闭开关作为长期单变量基线；
2. 不缓存 reply 结果，不跨 target 改变 state/move/reply 顺序、预算或 proof；
3. 不把 cache hit、GPU 时间或 resident bytes 写入任何授权证书；
4. 保持线程本地、同步调用基线，不在没有 stream 生命周期协议前跨线程共享 device pointer；
5. graph miss 继续做完整内容比较，不以地址或短哈希换取命中速度。

后续 [reply 精确任务去重](58_M5_HT_Reply_Task_Deduplication.md) 已先完成 batch-local 切片：
d15112 的 CUDA reply 提交量从 28,497 降为 418，且保持完整逻辑结果与 proof 不变；报告与
五路 summary 随之升级为 V16/V19。尚未实现的是跨 batch/target 的语义结果缓存和 work-graph
子结构共享。只有完整 key 相等且能保持逐 target 预算、稳定顺序、快照守卫和 proof 字节语义
时，才扩大复用生命周期。后续[目标级多 GPU 静态切片](60_M5_HT_Target_Multi_GPU_Static_Slicing.md)
已让每个 worker 保有独立线程/设备缓存；当前缓存只跨该 worker 在单次 scan 中的 targets，不跨
显式多 GPU 的 Local Elimination stages。
