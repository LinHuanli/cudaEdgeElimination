# Concorde / cuOpt LP 集成

## 组件边界

Concorde 与 QSopt 受研究用途许可证约束，源码和二进制只存在 `.deps/concorde-*`。仓库提交的是：

- 可重复应用的补丁；
- `lp-epoch-v1` C 导出辅助模块；
- 构建脚本、固定 SHA-256 与接口测试。

cuOpt 固定 Python wheel 提供共享库；自有 sidecar 通过 C API 动态加载，不依赖 Python 求解接口，也不把 cuOpt 链入公开制品。

## `lp-epoch-v1`

文本格式按固定顺序保存：版本、维度、目标方向/偏置、CSR、行方向/RHS、变量上下界/类型、每列对应边端点以及模型哈希。读取器检查：

- `row_offsets[0]=0`、单调且末项等于 `nnz`；
- 所有列索引在范围内；
- 数组长度与有限值；
- 边映射数组维度与列数一致；需要跨 epoch warm start 时，再要求每列都是唯一合法的规范边；
- 内容哈希与规范化重算一致。

Concorde patch 在 cut epoch 结束、模型稳定时调用 QSopt getters 导出；不得从 LP 文本反向猜测列身份。列端点与整数目标来自 `lp->graph.edges[i]`，并在导出前用当前 QSopt primal 验证

\[
\sum_i c_i x_i = \texttt{QSget\_objval}.
\]

该检查解决了实测中 `QSget_obj` 返回数组不能复现当前 LP 目标的问题，同时把列重排错误变成 fail-closed。CSR、行方向和变量边界仍直接来自当前 QSopt 模型。

## cuOpt sidecar

调用序列：

1. `cuOptGetFloatSize/GetIntSize/GetVersion` 做 ABI 门禁；
2. `cuOptCreateProblem` 构造连续 LP；
3. 创建 settings，固定随机种子、双精度 PDLP、关闭 presolve 并设置容差；
4. `cuOptSolve`；
5. 获取 termination、objective、primal、dual、reduced costs 和 solve time；
6. 独立重算 primal residual、目标、reduced cost residual；
7. 量化符号合法对偶并生成模型内精确下界。

输出 `lp-solution-v1`。只有 `status=OPTIMAL`、维度/残差可接受且精确下界成功时标记模型证书有效，但仍不能跳过 Concorde 完整图定价。

### 跨 epoch 稳定身份与 warm start

`CuOptSession` 保留上一轮通过数值门禁的 primal/dual，并在下一轮重建 CSR 后按稳定身份投影：

- 列身份由规范 Concorde 边端点 `(min(u,v), max(u,v))` 构造；重复、缺失或自环身份使 warm start 失败关闭；
- 行身份由方向、RHS，以及按稳定列身份排序的精确系数位模式构造；重复行身份不做位置猜测；
- 旧值只映射到身份相同的目标；新列使用投影到边界内的 0，新行 dual 为 0；
- 列和行覆盖率都达到默认 80% 才调用 `cuOptSetInitialPrimalSolution` 与 `cuOptSetInitialDualSolution`；
- API 缺失、覆盖不足、初值被拒绝或上一轮数值验收失败时自动冷启动/清空缓存；presolve 始终关闭。

`lp-sequence` 用同一会话连续求解两个 epoch，并把是否尝试、是否应用、覆盖率和稳定身份哈希写入两份 solution。26.8.0 小 LP 实测第二轮 `warm_start_attempted=1`、`warm_start_applied=1`、行列覆盖率均为 1。该能力只减少 PDLP 迭代准备成本，不改变证明边界；warm-start 浮点向量仍不能直接授权删边。

## Concorde exact pricing 接口

补丁新增一个显式入口，将外部对偶按 Concorde 行顺序写入 `lp->exact_dual`。入口要求 solution magic、`END`、模型哈希、`OPTIMAL`、`numerically_accepted=1` 和对偶行数全部匹配；非有限值、越界值和错误符号会被拒绝或按不等式方向裁剪。随后调用现有 `CCtsp_exact_price` 路径，强制遍历完整图并计入所有负 reduced cost 惩罚。

`CUDAEE_CONCORDE_EXACT_V1` 当前输出：

- exact lower bound；
- `CCbigguy` 原始 words；
- 完整图标记、对偶行数、模型哈希与当前上界。

证书证明“本次受信 Concorde 进程已执行完整图定价”，尚不是脱离 Concorde 状态即可独立重放的证明；每边 exact reduced cost、版本清单和消元后边集仍属于 M3.1。`tools/run_concorde_cuopt_epoch.sh` 负责唯一目录、等待完整 epoch、临时解原子发布以及三方哈希检查。

## 已验证结果与数值边界

- 随机 20 点：25 行、43 列；cuOpt 与 QSopt 当前 LP 目标均为 `88`，primal violation `4.44e-15`，reduced-cost residual `1.57e-14`，完整图下界 `87.3932819641`，上界 `88`。
- pr299：454 行、888 列、8561 nnz；cuOpt 模型目标 `48187.777777780764`，primal violation `8.37e-11`，reduced-cost residual `4.87e-10`；完整图下界 `43977.2693797 <= 48191`。
- 人为把 solution 哈希替换为零：Concorde 非零退出，日志命中 hash gate，且不生成 exact certificate。

pr299 的完整图下界明显弱于稀疏模型目标。这不是可行性残差问题，而是退化最优对偶在未激活完整图边上产生大量负 reduced cost。直接 dual simplex 实验可把下界提高到 `47891.4521198`，仍未与 QSopt 删除强度等价；PDLP crossover 在 cuOpt 26.8 的该模型上触发堆损坏，因此默认关闭。下一步必须做迭代补列或显式对偶稳定化，不能绕过 exact penalty。

## 集成测试

- 2×3 整数小 LP：手算最优值与有理下界；
- 同一 epoch 由 QSopt 与 cuOpt 求解，比较可行性和目标容差；已覆盖随机 20 点与 pr299；
- 人为翻转一行方向，确保符号裁剪后下界仍有效；
- 删除一个未激活列造成负 reduced cost，确保模型内证书不能被标为完整图证书；
- pr299 一个 cut epoch 的导出—求解—精确定价往返；已覆盖；
- 错配模型哈希必须拒绝且不落证书；已覆盖。
- 稳定列重排后 primal/dual 按身份而非位置映射；新增行列覆盖不足、重复边身份和未验收解均失败关闭；已覆盖。
- cuOpt 26.8.0 同会话第二轮实际接受 primal/dual warm start；已由 `tools/run_cuopt_smoke.sh` 覆盖。

当前完成的是 M3.1 所需的稳定编号和 warm-start 基础设施。完整图负 reduced-cost 列的“定价—补列—重解”循环尚未接入 Concorde，因此 M3.1 删除强度仍是 pending，不能把一次 restricted-master warm start 误报为完整 LP 消元。
