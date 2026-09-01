# Hamilton–Tutte 混合 GPU wavefront

## 本阶段范围

M4.3b3a 把递归 DFS 的求值顺序改为显式工作图，验证 AND–OR continuation 可以在 GPU 上稳定传播。主机仍负责工作图所有权和 leaf proof；M4.3b3b1 已把 point/end reply 的路径冲突标记加入 GPU，CPU 保留规范子状态生成与逐项认证。这个边界隔离最容易出错的 continuation 与路径语义，并为后续设备端状态写出和批量叶提供可差分基线。

`ht-prove --scheduler wavefront --path-append-backend auto|cpu|cuda --propagation-backend auto|cpu|cuda` 使用该路径。两个后端独立选择；它生成与 DFS 相同的 `CUDAEE_HT_RECURSIVE_PROOF_V1`，`ht-verify` 不区分证书来自哪一种调度器。

## 主机 BFS 工作图

每个 `WaveState` 保存规范路径系统、递归深度、leaf proof 和按确定性顺序排列的 OR moves。每个 move 保存完整 AND replies；非法路径 reply 立即成功，合法 reply 分配一个只位于下一层的新状态，工作图不做去重，因此没有共享 child。

根 `c,d` 单独占第 0 层，其合法 replies 形成第 1 层。主机逐层执行：

1. 对当前层每个状态运行现有 path-system leaf engine；
2. 未解决且未达深度上限时生成 point moves，必要时再生成 end moves；
3. 对同一状态的 point/end replies 分别合批，以 CPU 或 CUDA 标记冲突；CPU 规范化合法 child 并追加到下一层；
4. 达到状态、总 reply、单 move reply、move 或硬安全上限时停止并返回 `unresolved`。

候选数量上限仍只损失证明率。工作图中的“失败”表示在当前有界规则内未证明，不会形成删除授权。

## SoA 展平与 CUDA 传播

主机将工作图展平为三个连续数组：

- `HtWavefrontStateTask { parent_move, move_begin, move_count, leaf_proven }`；
- `HtWavefrontMoveTask { parent_state, reply_begin, reply_count, child_count }`；
- `HtWavefrontReplyTask { child_index, path_infeasible }`。

CUDA 输入校验要求 state/move/reply 区间连续且无未引用记录，布尔字段只能为 0/1，每个非根状态必须被唯一 reply 引用，合法 child 必须落在严格后续层。其目标真值仍为：

\[
S_s=L_s\lor\bigvee_{m\in M_s}\bigwedge_{r\in R_m}
\left(I_r\lor S_{child(r)}\right).
\]

实现以 leaf、含零合法 child 的 vacuous-success move，以及无 move 的 failure state 初始化完成队列。一个线程消费一个完成 state，并原子更新其唯一 parent move：失败 child 设置 `failed`，最后一个 child 根据 `remaining_children` 完成 move；成功 move 用 CAS 立即把 parent state 置为成功，失败 move 递减 `remaining_moves`，只有最后一个失败 move 才把 parent 置为失败。

传播使用单-block device-persistent kernel。每轮在设备端冻结当前 `queue_tail`，整个 block 并行消费该批次；新完成的父状态追加到同一全局队列并由下一轮消费。合法有限 DAG 中每个状态恰好入队一次，因此 `queue_tail` 最终必须精确等于 state 数；溢出或未完成前停滞都会报错。单 block 是先保证终止判定无跨 block 竞态的正确性基线，多 block/cooperative 版本仍需性能实测。

## CPU 差分与 proof 提取

CPU 对同一工作图按反向状态序独立计算完整 `status[]`。CUDA 返回的向量必须逐字节相同；显式 CUDA 不可用/异常返回 `unresolved`，任何不一致返回 `invalid`，`auto` 的运行时失败转 CPU。

只有根状态成功时才提取证书。提取器在每个状态选择第一个成功 OR move，只复制该 move 的全部 AND child，形成紧凑的前序 continuation arena；失败候选和未选成功候选不进入证书。随后运行既有独立 verifier，重新枚举 replies、重建路径和复核所有叶。GPU 状态本身从不直接授权目标边。

## 回归证据

- shallow、递归 point、递归 end 三个固定实例分别由 DFS 与 CPU wavefront 求值，结果都通过同一全局 verifier，成功 arena 的节点数和关键 move 类型一致；
- 8 个固定种子随机 7 点完整图在无用户预算截断时逐例比较 DFS/wavefront 状态；成功实例同时重放两份 proof，并以直接巡回穷举确认目标边不属于最优巡回；
- 独立 4 状态 CUDA truth table：根的第一个 move 包含一真一假 child，第二个 move 含一个真 child，期望状态为 `[1,1,0,1]`，覆盖 AND 失败与 OR 回退；另一个表验证一真一假 child 使唯一 move 失败，根只能在全部 moves 失败后置为失败；
- 非向后层依赖在 kernel launch 前被拒绝；
- 固定 point CLI 的全 CUDA 候选路径产生 34 states、18 moves、84 replies，峰值 frontier 为 27；point/end append 为 9 batches、84 tasks，压缩后是 4 节点可重放 proof；
- 空闲 RTX 4000 Ada 上完整 Hamilton–Tutte 单元测试经 compute-sanitizer memcheck 为 0 error，覆盖 persistent queue 与 path-append kernel。

## M4.3b3b 后续

M4.3b3b1 已完成 device-persistent 完成队列和批量 path 冲突标记，具体契约见 [GPU path append](10_Hamilton_Tutte_GPU_Path_Append.md)。下一阶段迁移 point/end reply 计数与写出、规范子状态 SoA，并加入按 `(depth,path_count,reply bucket)` 分桶的 leaf batches、多 block 调度和 CPU long-tail。完成 CPU/GPU 无截断差分、显存峰值与收益门禁后，才设计 HT proof sidecar 与不可变 epoch 的确定性提交。
