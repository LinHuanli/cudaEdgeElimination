# 递归 Hamilton–Tutte continuation arena 与全局证书

## 目标与边界

M4.3b2 先在 CPU 上固定递归真值和证书语义，再把执行顺序改造成 GPU wavefront。对目标边 `a-b`，状态 `F` 表示 Hamilton 已揭示的规范路径系统，求值式为

\[
\operatorname{HT}(F)
=
\operatorname{Leaf}(F)
\lor
\bigvee_{v\in\mathcal C(F)}
\bigwedge_{H\in\mathcal H(F,v)}
\operatorname{HT}(F\cup H).
\]

CPU DFS 和未来 GPU wavefront 必须计算同一表达式；只有调度顺序可以不同。当前实现是单目标研究 API，不修改图，也不向 `gpu-eliminate` 提交删除。

研究 CLI `ht-prove` 从 TSPLIB/Concorde 边快照生成 V1 文件，`ht-verify` 从原快照独立重放。`--scheduler dfs|wavefront` 选择求值顺序，二者输出同一证书类型。`ht-prove` 的成功、未解决和非法退出码分别为 0、3 和 2；即使未解决也覆盖写入一个 `proven=0` 文件，避免误用同路径下的旧成功证书。

## Move 与完整 replies

根节点只允许浅层阶段已经验证的 `c,d` move。它选择一对与目标边严格 2-opt 不兼容的活动边端点（或按配置允许缺失边），Hamilton replies 是两个中心 surviving 邻边对的完整笛卡尔积。

合法根 reply 若叶证明未解决，递归层可以依次尝试两类 OR move：

- `point(n)`：`n` 尚未出现在 `F` 中；AND replies 是 `n` 的所有 surviving 无序邻边对，每个 reply 加入三节点路径；
- `end(p,q)`：`p` 是当前规范路径端点，`q` 是其路径内邻居；AND replies 是 `p` 的所有活动邻边（排除 `p-q`），每个 reply 加入一条端点边。

每个 move 的候选数、单 move reply 数、总 reply 数、状态数和深度都可设预算。候选排序或截断只影响证明率；任何状态、reply 或深度预算耗尽均返回 `unresolved`，绝不能解释为证明成功。

## Continuation arena

成功 proof 使用 `HtTreeNode[]` 扁平 arena。节点保存规范路径系统、move 类型/参数、完整 replies，以及 leaf 节点的 path-k-opt proof。合法 reply 只向更大的 `child_index` 引用；路径冲突 reply 使用 `path_infeasible=true` 且没有 child。

搜索采用 checkpoint/rollback：一个 OR move 只有在所有合法 replies 都得到成功 child 后才写入父节点；任一 child 未解决就回滚该 move 的临时子树。最终 arena 因而只含成功证明所需的 continuation，不含失败分支。空 reply 集遵循 AND 的逻辑真值，允许形成 vacuous success。

虽然当前生成器使用 DFS，arena 不携带调用栈，后续 wavefront 可直接把 `child_index` 替换为 continuation counter 传播：任一 child 失败使候选失败，全部 child 成功使候选成功，任一候选成功使父状态成功。

## 独立 verifier

`VerifyHtRecursiveProof` 不读取搜索配置，也不信任保存的工作计数。它从目标边重新构造根路径，并以显式 worklist 检查：

1. 快照哈希、目标活动边和根 `c,d` admissibility；
2. 每个节点路径系统必须与父 reply 重新规范化的结果完全一致；
3. 根、point 和 end move 的 replies 从当前 CSR 重新完整枚举并逐序比较；
4. path-infeasibility 必须可由规范化器重新得到；合法 reply 必须有唯一、向后的有效 child；
5. leaf 的 path-system hash、outside coverage、交换成本和 witness 全部交给既有独立 k-opt verifier；
6. arena 不得含环、共享 child、越界引用或未引用节点。

证明元数据中的候选数、状态数和 leaf 调用数只用于 profiling，不参与授权。

## `CUDAEE_HT_RECURSIVE_PROOF_V1`

V1 是确定性文本格式，固定记录：

- `snapshot_hash`、规范目标边和 `cd_mode`；
- 搜索工作计数；
- 按 arena 顺序记录的规范路径、move 和全部 replies；
- leaf 节点中以十六进制封装的 `CUDAEE_PATH_KOPT_PROOF_V1`。

读取器限制文件、节点、reply 和路径节点总量；严格解析有符号/无符号整数、布尔值、枚举 token、连续索引和 16 位十六进制哈希，并拒绝 `END` 后的任何字段。读取成功只说明结构可解析，调用方仍必须针对原图运行 verifier。

## 回归证据

- 固定 point 实例：浅层证明在受限 3-opt 叶下为 `unresolved`；递归深度 1 产生 4 节点 arena，根为 `c,d=(0,3)`，三个 continuation 均真实使用 point move；
- 固定 end 实例：禁用 point 后，递归深度 1 产生 4 节点 arena，并真实使用端点 `3`、内部邻居 `6` 的 6-reply end move；
- 两个 8 点实例都独立穷举固定节点 0 后的全部 Hamilton 巡回，确认“包含目标边的最短巡回”严格劣于全局最优；
- 深度 0、状态预算耗尽保持 `unresolved`；遗漏 reply、错 point、错内部邻居、截断 arena、错 child 和篡改嵌套叶均被拒绝；
- 内存 proof、字符串 V1 和文件 V1 往返后均重新验证，规范序列化字节一致；尾随字段被解析器拒绝；
- CUDA 构建下可用 GPU 只筛选根 `c,d` flags，随后仍由 CPU 构造和验证完整递归 proof。

## M4.3b3 衔接

M4.3b3a 已在不改变 verifier 和文件格式的前提下加入主机 BFS 工作图、CUDA 原子 continuation counters 与单-block device-persistent queue，详见 [GPU wavefront](09_Hamilton_Tutte_GPU_Wavefront.md)。M4.3b3b1 又加入经 CPU 全量认证的 point/end [GPU path append](10_Hamilton_Tutte_GPU_Path_Append.md)。仍需 GPU 规范子状态/reply 写出、按深度/路径数/reply 数分桶、批量叶调用和 CPU long-tail 队列；完成后才能把 HT 删除接入不可变 epoch 的确定性 commit。
