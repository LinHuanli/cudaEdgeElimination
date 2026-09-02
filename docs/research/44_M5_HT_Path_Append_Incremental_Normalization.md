# M5 HT path-append child 增量规范化

## 1. 根因与实现

上一版 sparse fast path 已避免按完整 TSP 维度分配邻接表，但每个 point/end task 仍把全部 parent paths 与新增边重新送入 `map/set` 规范器。d15112 的同一批次包含大量共享 parent 的短 child，重复建树、去重和全链遍历使 `path_append_child_normalize_ms` 达到 `180.448 ms`。

提交 `236022c` 把生产 child 路径改为增量链合并：

- batch 入口仍用通用 sparse 规范器完整认证每个 parent，非法或非规范 parent 继续拒绝；
- 为每个已认证 parent 建立按 node 排序的位置索引，记录 component、offset 和 endpoint；
- point 只连接 `first-center-second` 涉及的至多两个 parent components，end 只连接 `first-second`；
- 合并链按较小端点定向，再与未受影响 components 一起按字典序排序；
- 不可行判定严格保持 dense 规范器的原因优先级：point 为度数超限后成环，end 为重复边、度数超限、成环；
- child edge SoA 仍从规范 child 独立重建，CUDA 仍逐数组与 CPU 结果比较，proof 重放仍使用独立 dense 规范器。

因此优化没有把候选器结果提升为证书，也没有绕过任何 CPU 删除授权路径。

## 2. 差分与提交门禁

单元测试把原有两个 parent 的 308 个合法 API tasks 保持为锁定计数，并新增四类 parent 的 684 个 tasks，覆盖长链、多分量、空闲节点、全覆盖状态、方向翻转、字典序插入和全部冲突类型。共 992 个 tasks 均逐项比较：

- `valid`；
- 中文 `reason`；
- `edge_count`；
- 完整规范 `paths`。

提交前门禁结果：

- CPU Debug/Release 各 17/17；
- CUDA Release 20/20，包含 CPU/GPU 差分；
- 完整 Hamilton–Tutte CUDA Debug 单元在 compute-sanitizer memcheck 下为 0 errors。

## 3. clean-commit 三实例结果

正式 runs 均绑定 clean commit `236022c`、物理 GPU 1、8 个 CPU cost threads 和锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T000539Z-2700383`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T000549Z-2700950`；
- d15112：`artifacts/d15112-ht-scan-20260902T000554Z-2700380`。

相对 clean baseline `00c0156`，CPU-fused 单变量结果如下：

| 实例 | child normalize：基线 → 增量（ms） | normalize 加速 | path append 加速 | work graph 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 31.825 → 3.739 | 8.511× | 3.597× | 1.037× | 1.038× | 1.034× |
| rl5915 | 53.266 → 5.462 | 9.752× | 4.410× | 1.249× | 1.220× | 1.161× |
| d15112 | 180.448 → 22.063 | 8.179× | 4.673× | 1.308× | 1.244× | 1.158× |

索引构造令 parent prepare 增加 `0.230/0.308/0.312 ms`，但 child normalize 分别减少 `28.086/47.804/158.386 ms`，净收益稳定。三实例五路活动边、工作签名和去除 `metrics` 行后的 proof 均与 baseline 逐字节一致；JV 固定点、V2 重放和三份受保护最优 tour 也全部通过。

## 4. 新画像与下一切片

d15112 CPU-fused search 已由 `817.118 ms` 降至 `656.839 ms`。新的 work graph 主要分解为：

| 子阶段 | 时间（ms） |
|---|---:|
| leaf | 207.672 |
| Hamilton reply | 146.558 |
| end reply | 67.125 |
| host-build residual | 56.557 |
| path append | 43.367 |

Hamilton reply 中 validation 为 `121.818 ms`，占该阶段约 `83.1%`；同一 wavefront 的多个 reply batches 仍重复验证同一只读 graph。下一单变量实验应仿照 leaf binding：公开 reply API 继续独立验证，内部强类型 wavefront binding 只复用同一 graph 对象的入口验证，并以对象错配拒绝、公开/绑定结果全数组一致和三实例五路 proof 不变为门禁。
