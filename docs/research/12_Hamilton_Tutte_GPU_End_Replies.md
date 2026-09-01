# Hamilton–Tutte GPU end replies

## 范围与安全契约

M4.3b3b2a2 把 end move 的完整活动边 reply 枚举迁到 CUDA。每个任务为

```cpp
HtEndReplyTask { endpoint, internal_neighbor }
```

其中 `endpoint-internal_neighbor` 必须是当前规范路径中的活动边。完整 reply 集是 endpoint 的全部活动邻边减去这条内部边。公开入口 `EvaluateHtEndReplies(graph, tasks, backend)` 返回 offsets 和展平的规范 `NodeEdge`；任务 `i` 的区间为 `[offsets[i], offsets[i+1])`。

CPU 始终独立扫描排序 CSR 并比较全部 offsets、边和值顺序。wavefront 只使用 CPU 枚举的列表；显式 CUDA 不可用或运行失败返回 `unresolved`，`auto` 失败安全回退，任何列表差异返回 `invalid`。该候选器本身不产生证明，也不授权删除。

## count/write 实现

同一父状态的每条规范路径贡献 front/back 两个 task，一次提交到设备。一个线程负责一个 task：

1. count kernel 扫描 endpoint 的 CSR，确认内部邻点恰好可排除，并统计其余邻点；
2. 主机用检查溢出的 `uint64_t` 前缀和分配精确输出；
3. write kernel 重走 CSR，以邻点原顺序写入 task 的独占区间，并把边规范为 `(min,max)`；
4. 设备错误字拒绝内部邻点缺失、输出越界和 count/write 尾指针不一致；
5. CPU 对展平结果逐元素认证，再按 `(reply_count,endpoint,internal_neighbor)` 恢复原有候选顺序和上限。

不同 task 的区间不重叠，write 无需原子操作，输出不依赖 block 调度。degree=1 的 endpoint 合法地产生 `[k,k)` 空区间，设备输出缓冲可以为零长度；空区间仍表示完整枚举，而不是运行失败。

## 输入与失败模式

公开层要求与其他 HT 规则相同的整数安全、排序、无自环、对称 CSR 快照。每个 endpoint/internal 节点必须在范围内、互不相同且对应活动边。重复 task 和同一无向内部边的相反 endpoint 方向均合法，分别保留自己的确定性区间。

`--reply-backend auto|cpu|cuda` 同时控制 Hamilton 与 end reply 生成；两者分别记录 backend、device、batch/task/reply 数和累计 `cpu_verified`，便于定位哪一段发生回退。指标不写入 V1 proof，也不改变 verifier 语义。

## 回归门禁

- 9 点完整图覆盖多个 endpoint、相反方向和重复 task，CUDA offsets/edges 与 CPU 完全一致；
- 稀疏链图覆盖 degree=1 endpoint 的零长度输出；
- 非活动内部边、同点内部边和 CPU-only 显式 CUDA 请求在 launch 前失败；
- 固定 recursive-end 实例必须由 CPU 与全 CUDA wavefront 得到同一可验证 end proof，并确认 end batch 已实际执行；
- 固定 recursive-point CLI 的 frontier 预取路径实际触发 2 batches、18 tasks、108 end reply edges，最终 4 节点 V1 proof 独立重放通过；
- CPU Debug/ASan、CPU Release、CUDA Release 全套 CTest 与 Hamilton–Tutte compute-sanitizer memcheck 是提交门禁。

## 后续

M4.3b3b2b1 已把同一 frontier chunk 的端点 tasks 合并；为了不改变 child 插入顺序，当前会预取后来可能被 point vacuous-success shortcut 跳过的 end replies，所以固定样例从旧逐父状态路径的 8 tasks/48 edges 增至 18 tasks/108 edges。下一步先跨 chunk 批量得到 point path-append flags，再只为确实需要 end moves 的父状态生成端点任务。图 CSR 复用、规范 child SoA、leaf 分桶和 CPU long-tail 仍待实现。
