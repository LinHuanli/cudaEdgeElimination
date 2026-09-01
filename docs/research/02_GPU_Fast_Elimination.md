# GPU 快速消元实现计划

## 数据布局

每轮把活动无向边规范化为 `(min(u,v),max(u,v))`，按端点排序，并构造双向 CSR：

- `row_offsets[n+1]`；
- `neighbors[2m]`、`weights[2m]`、`edge_ids[2m]`；
- `edges[m]` 与整数坐标 `x[n],y[n]`；
- 快照 64 位内容哈希。

kernel 不修改 CSR。候选输出为定长结构 `(edge_id,witness,method)`，避免设备端动态分配。

## JV kernel

一个线程处理一条活动边：

1. 检查端点度数；
2. 从 `N(a) union N(b)` 生成至多 10 个唯一 `c`，按 `C(a,c)+C(c,b)`、再按点号排序；
3. 对每个 `c` 扫描全部活动邻点 `d`；
4. 找到第一个“无兼容 d”的 `c`，输出候选；
5. 无候选或任意算术检查失败则不输出。

CPU 基线使用同样的候选顺序与谓词，但独立实现距离和遍历，以便差分发现 kernel 错误。

## Host epoch 流程

`snapshot -> GPU/CPU propose -> CPU verify -> deterministic degree gate -> commit -> next snapshot`

停止条件为：没有已提交候选、达到 `--max-rounds`，或出现验证错误。默认 `backend=auto`：满足整数距离约束且 CUDA 可用时选择最空闲的可见 GPU，否则使用 CPU。

## HS 与兼容表

- triangle/diamond 先实现为 CPU 规则并单独测试，不与 JV 证明类型混用。
- HS `c,d` path-system 对 `m<=5` 才允许查预计算兼容表；表由 CPU 穷举生成并保存生成器哈希。
- `m=6,7` 直接调用 CPU 路径搜索。任何截断都只会少删边，绝不能把“未知”当“不兼容”。

首期 CLI 将 JV 标记为 `supported`；尚未接入的 HS 选项返回明确的 unsupported 错误，而不是静默降级为另一算法。

## 性能测量

每 epoch 记录 CSR 构建、H2D、kernel、D2H、CPU 复核与提交时间；同时记录候选数、接受数、拒绝数和边数。预热后至少 5 次，报告中位数与 P95。首期性能只观察，不设通过阈值。
