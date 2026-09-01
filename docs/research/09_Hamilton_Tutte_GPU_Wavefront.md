# Hamilton–Tutte 混合 GPU wavefront

## 本阶段范围

M4.3b3a 把递归 DFS 的求值顺序改为显式工作图，验证 AND–OR continuation 可以在 GPU 上稳定传播。当前主机仍负责状态规范化、候选/reply 生成和 leaf proof；GPU 只计算已展平工作图的布尔真值。这个边界先隔离最容易出错的 continuation 语义，并为下一阶段的 GPU 状态生成和批量叶提供可差分基线。

`ht-prove --scheduler wavefront --propagation-backend auto|cpu|cuda` 使用该路径。它生成与 DFS 相同的 `CUDAEE_HT_RECURSIVE_PROOF_V1`，`ht-verify` 不区分证书来自哪一种调度器。

## 主机 BFS 工作图

每个 `WaveState` 保存规范路径系统、递归深度、leaf proof 和按确定性顺序排列的 OR moves。每个 move 保存完整 AND replies；非法路径 reply 立即成功，合法 reply 分配一个只位于下一层的新状态，工作图不做去重，因此没有共享 child。

根 `c,d` 单独占第 0 层，其合法 replies 形成第 1 层。主机逐层执行：

1. 对当前层每个状态运行现有 path-system leaf engine；
2. 未解决且未达深度上限时生成 point moves，必要时再生成 end moves；
3. 对每个 move 枚举全部 replies，规范化合法 child 并追加到下一层；
4. 达到状态、总 reply、单 move reply、move 或硬安全上限时停止并返回 `unresolved`。

候选数量上限仍只损失证明率。工作图中的“失败”表示在当前有界规则内未证明，不会形成删除授权。

## SoA 展平与 CUDA 传播

主机将工作图展平为三个连续数组：

- `HtWavefrontStateTask { move_begin, move_count, leaf_proven }`；
- `HtWavefrontMoveTask { reply_begin, reply_count }`；
- `HtWavefrontReplyTask { child_index, path_infeasible }`。

CUDA 输入校验要求 state/move/reply 区间连续且无未引用记录，布尔字段只能为 0/1，合法 child 必须落在严格后续层。kernel 按最深层到根层依次 launch，一个线程计算一个状态：

\[
S_s=L_s\lor\bigvee_{m\in M_s}\bigwedge_{r\in R_m}
\left(I_r\lor S_{child(r)}\right).
\]

层间 kernel 位于同一 CUDA stream，下一次 launch 只读取已经完成的更深层状态。当前没有使用跨层原子 counter；persistent queue 与 counter propagation 留到 M4.3b3b。

## CPU 差分与 proof 提取

CPU 对同一工作图按反向状态序独立计算完整 `status[]`。CUDA 返回的向量必须逐字节相同；显式 CUDA 不可用/异常返回 `unresolved`，任何不一致返回 `invalid`，`auto` 的运行时失败转 CPU。

只有根状态成功时才提取证书。提取器在每个状态选择第一个成功 OR move，只复制该 move 的全部 AND child，形成紧凑的前序 continuation arena；失败候选和未选成功候选不进入证书。随后运行既有独立 verifier，重新枚举 replies、重建路径和复核所有叶。GPU 状态本身从不直接授权目标边。

## 回归证据

- shallow、递归 point、递归 end 三个固定实例分别由 DFS 与 CPU wavefront 求值，结果都通过同一全局 verifier，成功 arena 的节点数和关键 move 类型一致；
- 8 个固定种子随机 7 点完整图在无用户预算截断时逐例比较 DFS/wavefront 状态；成功实例同时重放两份 proof，并以直接巡回穷举确认目标边不属于最优巡回；
- 独立 4 状态 CUDA truth table：根的第一个 move 包含一真一假 child，第二个 move 含一个真 child，期望状态为 `[1,1,0,1]`，覆盖 AND 失败与 OR 回退；
- 非向后层依赖在 kernel launch 前被拒绝；
- 固定 point CLI 的 CUDA 路径产生 34 states、18 moves、84 replies，峰值 frontier 为 27，压缩后是 4 节点可重放 proof；
- GPU 2 上 Hamilton–Tutte 单元测试经 compute-sanitizer memcheck 为 0 error。

## M4.3b3b 待办

下一阶段把收益较大的规则计算迁移到 GPU：批量 path normalization/冲突标记、point/end reply 计数与写出、按 `(depth,path_count,reply bucket)` 分桶的 leaf batches，以及 candidate `remaining_children/failed` 原子 counter。罕见深层、超大 reply 和精确 DP 留给 CPU long-tail。完成 CPU/GPU 无截断差分与显存峰值门禁后，才设计 HT proof sidecar 与不可变 epoch 的确定性提交。
