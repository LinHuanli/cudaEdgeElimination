# M5 HT reply 精确任务去重

## 1. 结论

提交 `4428aa4f7a104232122fbc18999e114cba0a43c6` 在不改变逻辑 reply、工作图顺序、
预算或证明格式的前提下，将同一 Hamilton/end reply batch 内精确相同的任务只提交一次。
CPU 仍构造按原任务顺序展开的完整 offsets/replies；CUDA 只计算首次出现的唯一任务，并在
返回前逐项与 CPU 唯一结果比较。工作图只消费 CPU 完整逻辑结果，去重映射和性能计数都不
参与删除授权。

d15112 的固定 8-target 切片中，15 个 Hamilton batches 的 23,939 个逻辑 centers 缩为
242 个 batch-local 唯一 centers，8 个 end batches 的 4,558 个逻辑 tasks 缩为 176 个
唯一有向 tasks。实际 CUDA 提交总量因此从 28,497 降为 418，减少 `98.533%`，即
`68.175×`。七对交错 clean-commit A/B 的 Hamilton/search/total/process wall 中位数分别
改善 `11.851%/3.327%/2.582%/2.158%`，设备驻留峰值减少 12,529,904 bytes。

该优化默认开启，保留同一二进制的显式关闭开关。它只做 batch-local 精确任务去重，不缓存
跨 batch、跨 target 或跨 epoch 的语义结果；后者仍是独立研究项。

## 2. 重复度证据与边界

正式切片的规范工作量如下。唯一数是“每个 batch 内唯一数再求和”，不能解释为整个 scan
的全局唯一 key 数。

| 类型 | 逻辑任务 | batch-local 唯一任务 | 逻辑/唯一 |
|---|---:|---:|---:|
| Hamilton center | 23,939 | 242 | 98.921× |
| end 有向 task | 4,558 | 176 | 25.898× |
| 合计 CUDA reply task | 28,497 | 418 | 68.175× |

重复主要来自 frontier batching：多个父状态拥有相同的 point center，或多个路径状态生成相同
的有向 endpoint/internal-neighbor 组合。相同任务在同一不可变 graph、同一 batch 固定目标下
必然产生同一规范 CSR reply slice，重复 count/write 不增加证明信息。

当前没有扩大生命周期：

- Hamilton key 不跨 target 复用，因为结果还依赖当前目标边；
- end key 虽只依赖 graph 和有向 task，也不跨 batch 保存结果；
- graph snapshot 改变后不存在旧结果缓存或失效协议；
- 不合并 wavefront states、moves、children，也不改变每个 target 的资源预算。

## 3. 精确 key 与确定性展开

### 3.1 Hamilton reply

一次 Hamilton batch 已固定同一个只读 graph 和规范目标边，因此任务的完整可变部分就是
`center`。实现沿用 CPU 主机去重的 dense `cache_index[dimension]`：

1. 按逻辑 centers 的原顺序扫描；
2. 首次遇到 center 时加入 `unique_centers`，并由 CPU 枚举唯一 reply slice；
3. 重复 center 直接引用该 CPU slice，按原位置追加到完整逻辑结果；
4. 开启去重时只把 `unique_centers` 交给 CUDA；关闭时把原始 centers 全部交给 CUDA。

`unique_centers` 严格保持首次出现顺序，不依赖哈希容器遍历顺序。CPU 的批内 center 缓存早已
是生产语义；关闭本开关只恢复 Hamilton CUDA 的逻辑提交量，不撤销已有 CPU 缓存。

### 3.2 End reply

end task 的精确 key 是有向二元组：

```text
(endpoint, internal_neighbor)
```

方向不可忽略：`(u,v)` 与 `(v,u)` 从不同 CSR 行枚举，必须视为不同任务。入口先验证两个节点
均在范围内、互异，且该有向表示对应一条活动无向边；随后把两个非负 32-bit 节点号无损打包
为一个 64-bit key。该 key 没有概率碰撞，`unordered_map` 只用于精确整数相等查找，输出顺序
来自独立的首次出现向量。

开启时，CPU 为每个唯一 task 枚举一次规范活动边 slice，再通过 `logical_to_unique` 按原 task
顺序展开完整 offsets/replies；关闭时 CPU 和 CUDA 都逐个执行原始逻辑 tasks。两条路径产生
完全相同的逻辑数组。

## 4. CPU 认证与失败关闭

数据流保持以下边界：

```text
逻辑 tasks
   |-- 首次出现映射 --> CPU 唯一 slices --> 按原顺序展开完整逻辑结果 --> 工作图
   |
   `-- 唯一 tasks ------> CUDA count/write ------> 与 CPU 唯一 slices 逐项比较
```

共用比较器检查：

1. CUDA offsets 数量必须等于唯一任务数加一；
2. 首 offset 为 0，尾 offset 精确等于 reply 数量；
3. 每个区间单调、有界，长度与 CPU 唯一 slice 相同；
4. 每个 reply 按原 CSR 确定性顺序逐元素相等。

任一差异立即抛错，不把 CUDA 结果送入工作图。显式 CUDA 模式下，设备选择、allocation、
kernel 或 copy 失败仍向上传播；auto 模式沿既有规则回退 CPU，且
`cuda_tasks_submitted=0`。CPU 公开 API、scan 即时 proof verifier、V2 最终重放、最小度门禁
和原子提交均未缩减。

## 5. 配置、指标与格式

CLI 新增：

```text
--deduplicate-reply-tasks 0|1
```

默认值为 1。`HtWavefrontOptions::deduplicate_reply_tasks` 把选择传到 Hamilton/end bound batch
API；公开 batch API 也提供默认开启的布尔参数，便于单元差分。

新增或上移的规范/物理指标为：

- `reply_cuda_tasks_submitted`：成功返回且完成 CPU 比较的 CUDA 物理任务总数；
- `end_reply_batches`、`end_reply_tasks`、`end_reply_unique_tasks`；
- `end_replies_generated`：完整逻辑 end replies，不是唯一 slice 数；
- 已有 `hamilton_reply_centers/unique_centers` 继续分别表示逻辑量和 batch-local 唯一量。

全 CUDA reply 路径必须满足：

```text
开启：reply_cuda_tasks_submitted
    = hamilton_reply_unique_centers + end_reply_unique_tasks

关闭：reply_cuda_tasks_submitted
    = hamilton_reply_centers + end_reply_tasks
```

CPU/hybrid reply 后端的 `reply_cuda_tasks_submitted` 必须为 0。HT scan report 升为 V16，五路
benchmark summary 升为 V19；V15/V18 仍表示上一切片的 CUDA 驻留格式，不回写解释。

`tools/run_ht_scan_benchmark.sh` 接受
`CUDAEE_DEDUPLICATE_REPLY_TASKS=0|1`，把选择与全部逻辑/唯一/物理计数写入 manifest 和
summary，并检查五路逻辑计数一致及上述物理公式。

## 6. 自动化与冒烟门禁

单元测试覆盖：

- Hamilton 重复 centers 的默认唯一提交与关闭后的完整逻辑提交；
- end 重复 task、相反方向 task 和 degree-one 空 slice；
- 两个开关状态的 offsets/replies 完全一致；
- wavefront 默认/关闭状态的规范 proof 完全一致；
- scan attempt 与总计的逻辑、唯一、物理计数一致；
- reply 图/workspace 驻留命中在去重后继续满足精确缓存契约。

提交门禁结果：

- CPU Debug：20/20；
- CPU Release：20/20；
- CUDA Release（物理 GPU 1）：23/23；
- CUDA Debug Hamilton–Tutte 单元（物理 GPU 2）：
  `compute-sanitizer --tool memcheck`，0 errors；
- `git diff --check`、benchmark shell 语法和 workspace policy 全部通过。

pcb3038 单 target 的 V19 五路脚本分别运行了开关两侧：

- 开启：`artifacts/pcb3038-ht-scan-20260902T041659Z-2853438`；
- 关闭：`artifacts/pcb3038-ht-scan-20260902T041906Z-2855573`。

两次都通过五路边文件、工作签名、独立 proof 重放和最优 tour 门禁。逻辑量固定为 377 个
Hamilton centers 与 68 个 end tasks；开启时 CUDA 提交 `55+20=75`，关闭时提交
`377+68=445`，最终图哈希均为 `a69d7b45aff83cff`。

## 7. d15112 七对正式 A/B

协议：

- clean commit：`4428aa4f7a104232122fbc18999e114cba0a43c6`；
- 设备：物理 GPU 1，NVIDIA RTX 4000 Ada Generation；
- 输入：d15112 先由 CPU JV 达到 159,187 条边的固定点；
- 目标：`weight-desc` 的前 8 个 targets，每 target 最多 2,000 states；
- 后端：c,d、reply、path append、leaf cost 与 propagation 均显式 CUDA，CPU 认证保持开启；
- reply CUDA 图/workspace 驻留在两侧都开启，只切换任务去重；
- off/on 各预热一次；七对按 `off/on`、`on/off` 交替；
- 原始目录：`artifacts/ht-reply-task-dedup-clean-ab-20260902-0R8tlC`；
- `metrics.csv` SHA-256：
  `1219df0a8662089a60b06c50fb6fe8cccb27c7d36625c31a762e5f9e5e9f7b46`；
- `manifest.txt` SHA-256：
  `aba7afccacc9afff46750474a82b134816d29d2075e76dd6f39052a65993d6aa`；
- `analysis.txt` SHA-256：
  `d74ee8ce344e768900c29cf6b3b5ca64bc5c92e6b7a0945161ab142caf076795`。

下表对 off/on 各七次取中位数；配对改善先按相同 run id 计算 `(off-on)/off`，再取七个
百分比的中位数。包含式阶段不能相加推导 total。

| 指标 | 关闭去重 | 开启去重 | 中位数改善 | 配对改善中位数 |
|---|---:|---:|---:|---:|
| Hamilton reply | 227.839145 ms | 200.837564 ms | 11.851% | 11.833% |
| end reply | 70.297775 ms | 69.209819 ms | 1.548% | 1.973% |
| search | 758.117882 ms | 732.892852 ms | 3.327% | 4.418% |
| total | 922.757534 ms | 898.930295 ms | 2.582% | 3.506% |
| process wall | 1155.051 ms | 1130.125 ms | 2.158% | 1.687% |

调度与内存指标固定或按预期变化：

| 指标 | 关闭去重 | 开启去重 |
|---|---:|---:|
| reply CUDA batches | 23 | 23 |
| graph cache hits | 22 | 22 |
| workspace cache hits | 15 | 17 |
| CUDA tasks submitted | 28,497 | 418 |
| peak resident bytes | 14,183,576 | 1,653,672 |

14 次运行都产生 10,003 states、11,851 replies、1,114 leaf calls，尝试 8 个 targets，证明并
提交相同 2 条边。活动边 SHA-256 均为
`39abcae8832b5eca8cee278237eafe52ed2f053fbe2abefd7f8593e746a189b4`；移除唯一允许变化的
V2 `metrics` 行后，规范 proof SHA-256 均为
`f5bd5565f2f9dfe3c01fc0bc5c419dae648b252f65725433a8880060913cbe64`；工作签名 SHA-256
均为 `6a15c91380df021d507eeb2721cfba9ea2f32eb4e38c12b6c5c39c141a3bf9be`。

14 份 proof 都由独立 CPU Release 进程从 JV 固定点重放到 159,185 条边，最终内容哈希均为
`29c3b8fccaf1a3fc`。14 次 d15112 最优 tour 检查的成本均为 1,573,084、缺边数为 0，规范
tour 哈希为 `4495654253f2318e`。

## 8. 决策与下一切片

当前决策：

1. 默认开启 batch-local 精确 reply task 去重，保留显式关闭开关作为长期单变量基线；
2. 只复用完整 key 相等的确定性枚举，禁止近似 key、概率哈希或跨快照结果复用；
3. 保持完整逻辑 offsets/replies、state/move/reply 顺序、预算和 proof 不变；
4. 不把唯一率、CUDA 提交量、cache hit 或性能计时写入任何授权证书；
5. 设备 workspace 按唯一物理任务增长，因此本切片同时降低计算量和驻留峰值。

下一步先画像剩余 418 个物理 reply tasks 在不同 batches/targets 间的精确重复率，并与 leaf
任务/结果及 work-graph child 子结构重复率比较。只有收益能覆盖跨生命周期 key、失效与内存
管理成本，且逐 target 预算和确定顺序可证明保持时，才实现语义结果缓存。单卡闭环稳定后再
进入多 GPU 静态 target 切片；不得用跨 target 调度改变原子 epoch 提交边界。
