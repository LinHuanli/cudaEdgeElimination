# 路径系统与兼容表实现

## 范围与安全边界

本工作包实现 Local Elimination 叶节点所需的组合基础，但**尚不构成 HS 删边证明**。已实现内容包括：

- 路径边并集的校验、合并与确定性规范化；
- `m=1..7` outside matching 和全部 inside perfect matching 的穷举；
- inside/outside 并是否形成单一交替 Hamilton 回路的 CPU 规范判定；
- `m<=5` 的压缩 coverage table、固定生成器哈希和 CUDA bitset 查询；
- `m=6,7` 的 CPU 直接判定回退。

当前没有新增消元方法，也不生成 proof 删除记录。只有后续 3/4/5-opt witness、困难叶 CPU fallback、HS AND–OR 搜索和独立证书重放全部接通后，才能让这套组合基础参与删边。

## 与参考实现的语义对应

`NormalizePathSystem` 对应 `CCelim_validate_paths` 的边并集语义：单条路径内不允许重复节点；合并图的节点度数不得超过 2；重复边、回路、越界节点和空路径系统均闭门失败。合法连通分量被重建为简单链，以较小端点定向，再按节点序列排序。因此相同边集得到相同规范形式，不依赖输入路径顺序或方向。

每条规范路径抽象为两个端点 `2p` 和 `2p+1`。outside matching 的枚举顺序与 `build_omatch_list` 一致：固定第 0 条路径正向，对其余路径按字典序枚举排列，再按低位到高位枚举方向位。数量为

\[
N_m=2^{m-1}(m-1)!.
\]

inside matching 通过“选择最小未匹配端点，再依次尝试伙伴”的递归穷举，数量为

\[
I_m=(2m-1)!!.
\]

兼容谓词与 `check_match` 等价：交替沿 inside、outside 边行走，恰好访问全部 `2m` 个端点并回到起点时为真。测试中的独立 oracle 不复用该遍历，而用并查集检查两组完美匹配的并是否连通；由于每个端点在并图中度数为 2，连通等价于单环。

## 表布局与固定哈希

coverage 按 `[inside][outside_word]` 排列，每个 inside matching 对应一个 outside bitset。`m=5` 的布局是 `945 × 6 × 8 = 45,360` 字节。哈希域为 `CUDAEE_PATH_COMPATIBILITY_V1`，整数按显式小端字节序写入 FNV-1a，避免依赖主机 ABI。

| m | outside | inside | packed bytes | generator hash |
|---:|---:|---:|---:|---|
| 1 | 1 | 1 | 8 | `4104b5c5658e8f3a` |
| 2 | 2 | 3 | 24 | `5fcd7fdac93b4fe9` |
| 3 | 8 | 15 | 120 | `9642d8a1cb6bf1ee` |
| 4 | 48 | 105 | 840 | `1853eb4cc99dd217` |
| 5 | 384 | 945 | 45,360 | `f6bccacc5c1fa84f` |

机器可读副本位于 `configs/path_compatibility_hashes_v1.txt`。单元测试把这五个哈希作为回归门禁；枚举顺序、bit 布局或谓词的任何变化都会显式触发失败。

## CPU/CUDA 执行契约

`EvaluatePathCompatibility` 的行为如下：

1. CPU 独立生成 outside/inside matching，并检查所有查询索引；
2. `m<=5` 时由 CPU 生成压缩表，CUDA kernel 每线程查询一个 bit；
3. CUDA 输出逐项重新调用 CPU 交替环谓词复核，数量或值不一致即报错；
4. `m=6,7` 时不构建可能急剧膨胀的完整表，明确返回 `cpu-fallback-m>5` 并直接判定；
5. 自动模式在没有可见 CUDA 设备时使用 CPU，显式 CUDA 模式在 `m<=5` 且设备不可用时非零失败。

可复现清单示例：

```bash
build/cuda-release/cudaee path-table \
  --paths 5 --backend cuda \
  --output artifacts/path-table-m5.cuda.manifest
```

当前 GPU 回归对 `m=1..5` 的 368,047 个 compatibility 单元做了全量差分；其中 `m=5` 有 362,880 个查询、147,456 个兼容单元。compute-sanitizer memcheck 为 0 error。

## proper 3/4/5-opt CPU 规范器

CPU 叶搜索没有复制 `swap.c` 的 1,900 余行展开判断，而是从相同组合定义生成 proper reconnect templates：删去巡回中的 `k` 条路径边后，将 `2k` 个端口重新完美匹配；新匹配与剩余 `k` 条路径分量必须形成一个交替单环，并且不得重新加入任何被删抽象边。得到的模板数与参考生成代码完全一致：

| k | proper templates | generator hash |
|---:|---:|---|
| 3 | 4 | `e58af5e08d290d04` |
| 4 | 25 | `03179e3ca191ce82` |
| 5 | 208 | `6696dde548591bce` |

机器可读哈希位于 `configs/kopt_reconnect_hashes_v1.txt`。测试把固定 ElimTSP 子模块中的 `src/swap.c` 编译为仅测试 oracle，在每个 `k` 上用 2,000 个确定性随机成本矩阵把判断阈值放在生成模板的最小重连成本附近，逐例比较严格改善结果，并验证 oracle 返回的重连属于生成模板集合。生产库不链接参考实现。

`FindKOptWitness` 的 CPU 规范流程为：

1. 重新规范化路径并验证 outside matching；
2. 从路径边与 outside 边独立重建简单巡回；
3. 固定 required edge（未提供时固定确定性 anchor），穷举包含它的删除集合；
4. 按原巡回次序构造端口和路径分量，枚举固定 proper templates；
5. 用精确整数距离比较删除成本与加入成本；
6. 重建改善巡回，提取 inside matching，再调用独立 verifier。

`VerifyKOptWitness` 不信任搜索器保存的成本或 matching，而是从原巡回重新检查：删除边全部属于路径、required edge 确实被删除、删除/加入集合互斥、改善后仍是覆盖全部局部节点的单巡回、成本严格下降、inside matching 与巡回端点次序一致。

`ProvePathSystemByKOpt` 对未覆盖 outside matching 依次保存 witness，并利用 inside coverage 合并其余方向。proof 同时绑定 graph snapshot hash、规范路径系统 hash 和兼容表 hash；独立 verifier 逐条重建 witness，并直接用 CPU 交替环谓词检查最终全覆盖。搜索预算耗尽返回 `unresolved`，绝不返回“无改善”或证明成功。

`CUDAEE_PATH_KOPT_PROOF_V1` 提供稳定文本往返，保存严格顺序的头字段、每条 source outside、删/加边、精确成本与 inside mate。读取器限制路径数、outside/record 数、`k`、边数、16 位十六进制哈希和完美匹配结构，并拒绝 `END` 后的多余字段；读回的 proof 仍必须经过 graph/path 绑定和完整 witness verifier，解析成功本身不授权证明。

回归覆盖 proper 3/4/5-opt 的全部模板路径：一个无严格改善的 7 节点实例会检查 25 个 anchor 删除集合和 1,330 个 proper reconnect 单元；另有 `m=1`、`m=2` 的成功 proof、成本篡改和 snapshot-hash 篡改拒绝测试。

## 批量 CUDA k-opt cost 候选器

`EvaluateKOptTemplateCosts` 接收同一 `k` 桶内的删除集合，每个任务保存按原巡回次序排列的至多 10 个端口。CUDA kernel 的一个线程计算一个 `(task, reconnect template)` 单元：

- 检查加边自环、实际重复边和重新加入的实际删除边；
- 用与 JV kernel 相同的 64 位整数平方根规则计算 `EUC_2D/CEIL_2D` 距离；
- 用 `int64_t` 检查累加溢出；
- 输出完整 `[task][template]` 成本矩阵，非法单元为 `INT64_MAX`。

CPU 有独立的成本矩阵实现，CUDA 回归逐单元比较三阶模板及包含相邻删除边端口的任务；批量搜索成功路径也会重建 witness 并再次调用 `VerifyKOptWitness`。compute-sanitizer memcheck 为 0 error。

这张 GPU 成本矩阵被严格定义为**候选 oracle**，不是 completeness certificate。`FindKOptWitness` 按 `cost_batch_size` 分批调用它：

1. 对 GPU 标出的低成本模板逐个做 CPU 完整重连与 witness 复核；
2. 若某个候选成功，可安全提前返回；
3. 若 GPU 对一个删除集合没有给出可接受 witness，CPU 对该集合重新穷举全部 proper templates；
4. `auto` 模式的 CUDA 运行错误转 CPU；显式 CUDA 失败返回 `unresolved`；
5. 因而 kernel 漏报只会损失性能，不会把“未知”变成证明。

## 后续接线

M4 后续必须按顺序完成：

1. 对未被 k-opt 解决的叶转入 CPU Held–Karp/原 LocalElimination fallback；
2. 实现 HS `c,d` 候选与 AND–OR 状态传播；
3. 缓存坐标/模板设备常驻数据，并用真实任务桶评估 kernel 粒度；
4. 将 leaf proof 嵌入完整 HS 证书，从不可变 epoch 快照递归重放；
5. 将完整 HS 证书链接入候选阶段，继续由 epoch commit 的 CPU 门禁授权。

在这些门禁完成前，`gpu-eliminate` 仍只授权现有 JV 证明。
