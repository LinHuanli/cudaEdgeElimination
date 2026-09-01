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

## 后续接线

M4 后续必须按顺序完成：

1. 对 3/4/5-opt reconnect template 建立独立 CPU 规范实现与穷举 oracle；
2. 让叶证明返回可重放的 inside matching 与具体换边 witness，而不是布尔值；
3. 对未被 k-opt 解决的叶转入 CPU Held–Karp/原 LocalElimination fallback；
4. 实现 HS `c,d` 候选与 AND–OR 状态传播；
5. 扩展 proof 格式并由 CPU 从不可变 epoch 快照完整重放。

在这些门禁完成前，`gpu-eliminate` 仍只授权现有 JV 证明。
