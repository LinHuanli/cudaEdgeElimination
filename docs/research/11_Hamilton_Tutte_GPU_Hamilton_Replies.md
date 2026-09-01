# Hamilton–Tutte GPU Hamilton replies

## 范围与证明边界

M4.3b3b2a1 将规则一致、计算量随节点度数二次增长的 Hamilton 邻边对枚举迁到 CUDA。给定目标边 `ab` 和多个中心 `c_i`，公开入口

```cpp
EvaluateHtHamiltonReplies(graph, target, centers, backend)
```

返回 `offsets` 与展平的 `HtNeighborPair`，中心 `i` 的区间为 `[offsets[i], offsets[i+1])`。根 move 把 `c,d` 放进同一批；递归 point move 在每个父状态内把全部可选中心放进同一批。重复中心是合法输入，并产生完全相同的独立区间。

CUDA 不是证明授权源。CPU 对每个中心重新运行规范枚举器，比较完整 offsets、每个 reply 的三个节点及顺序；工作图只使用这份 CPU 列表。设备不可用或 `auto` 运行失败时使用 `cpu-fallback`，显式 `cuda` 失败则整个搜索返回 `unresolved`，任何 CPU/CUDA 差异返回 `invalid`。

## 精确筛选语义

对中心 `c` 的排序邻接表中每个无序对 `x<order y`，设备执行与 CPU 相同的整数条件：

1. `xy` 不能等于目标边 `ab`；
2. `cx` 与 `ab`、`cy` 与 `ab` 分别至少有一个 2-opt 重连方向不是严格改善；
3. 三边替换满足

   ```text
   d(a,b) + d(c,x) + d(c,y)
     <= d(x,y) + d(a,c) + d(b,c)
   ```

距离只接受整数安全的 `EUC_2D`/`CEIL_2D` 快照。CUDA 使用整数平方根和 TSPLIB 的精确舍入条件，不把浮点误差带入边界判断。公开 CPU 层还会验证 CSR 已排序、无自环、对称且与活动边集合一致；CUDA detail 入口独立检查基本维度、边数、目标与中心范围。

## count/write 数据流

一个线程负责一个中心，并按 CSR 原顺序扫描所有邻边对：

1. `CountHtRepliesKernel` 只计算每个中心的 surviving 数；
2. 主机用检查溢出的 `uint64_t` 前缀和建立 offsets，并按精确总数分配输出；
3. `WriteHtRepliesKernel` 重走同一顺序，把 reply 写入中心独占区间；
4. 设备错误字检查越界以及 count/write 尾指针不一致；
5. 拷回后与 CPU 完整列表逐元素比较。

每个线程写独占区间，因此无需输出原子操作，且结果不依赖 block 调度顺序。当前前缀和在主机完成：这样先建立最小而可复核的实现，frontier 合批规模进一步增大后再评估 CUB scan 是否有端到端收益。

## wavefront 集成与预算

point 候选先按目标边中点距离形成候选中心列表，再批量枚举 replies，随后按 `(reply_count,node)` 排序并施加候选上限；这与旧 CPU 顺序一致。根候选已经由规范 CPU `c,d` 生成器记录 `reply_product`，在启动 reply batch 前先做预算筛选，设备返回后还必须重新验证两个区间大小的乘积完全一致。

结果记录实际 `reply_backend`、逻辑 device、batch/center/surviving reply 数和累计 `cpu_verified`。空候选不会初始化 CUDA 或计入 batch。指标只描述工作量，不属于证书，也不能单独支持性能结论。

## 回归证据

- 12 点完整图同时覆盖 `EUC_2D`、`CEIL_2D`、十个不同中心和一个重复中心；每个 GPU 区间与标量 CPU 枚举的内容和顺序完全一致；
- CPU-only build 覆盖 `auto` 安全回退与显式 CUDA 拒绝，非法目标端点中心在 launch 前失败；
- 固定递归 point CLI 全 CUDA 运行得到 9 batches、16 centers、46 surviving pairs；完整工作图仍为 34 states、18 moves、84 replies，最终 4 节点 V1 proof 通过独立 `ht-verify`；
- CUDA 单元测试还必须通过 compute-sanitizer memcheck，CPU Debug/ASan、CPU Release 与 CUDA Release 全套 CTest 均为提交门禁。

上述数字是确定性正确性样例，不是加速比。M4.3b3b2b1 已把当前层按父状态数切成 chunk，同一 chunk 的 point centers 共用一次 count/write；固定样例含根阶段在内的 Hamilton batches 从单状态 chunk 的 9 降到 4，详见 [frontier reply batching](13_Hamilton_Tutte_Frontier_Reply_Batching.md)。

## 未完成项

M4.3b3b2a2 已把 end move 的 CSR 提取迁到同一 CUDA 模块，见 [GPU end replies](12_Hamilton_Tutte_GPU_End_Replies.md)；M4.3b3b2b1/b2a 已把 reply 与 path-append 任务跨同层 chunk 合并。M4.3b3b2b2b1 又完成设备端规范 child edge SoA，M4.3b3b2b2b2a 已按桶融合规则 leaf cost rows；`NormalizedPathSystem` 和工作图所有权仍在主机。下一阶段扩展一般 deletion 游标、多 block continuation 和 CPU long-tail。所有设备输出继续先通过 CPU 全量差分，直至独立证书重放与 epoch commit 链路完成。
