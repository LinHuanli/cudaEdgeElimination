# 浅层 Hamilton–Tutte AND–OR 根证明

## 范围

M4.3b1 实现目标边 `a-b` 的一个可独立重放根层：Tutte 从若干 `c,d` moves 中选择一个，形成 OR；对固定 move，Hamilton 在 `c` 和 `d` 处所有仍可能属于最优巡回的邻边对组合必须全部关闭，形成 AND。当前不递归选择 extra point/end，也不向 epoch commit 提交删除；它是后续 wavefront 的已验证根节点语义。

固定 `c,d` 时计算：

\[
\operatorname{ShallowHT}(ab,c,d)
=
\bigwedge_{(c_0,c_1)\in R_c}
\bigwedge_{(d_0,d_1)\in R_d}
\operatorname{Leaf}
\bigl(\{ab,c_0c,cc_1,d_0d,dd_1\}\bigr).
\]

搜索只要找到一个成功的 `c,d` 即可返回证明；任一 reply 未解决就放弃该 move。预算耗尽和没有成功 move 都返回 `unresolved`。

## `c,d` Tutte move

实现支持两种与参考代码对应的安全模式：

- `kActiveIncompatible`：`c-d` 是活动边，且与 `a-b` 的两个 2-opt 方向都严格改善，因此二者不能同时属于最优巡回；
- `kMissingOrIncompatible`：`c-d` 不在当前安全图快照中，或满足上述严格不兼容条件。

四个节点必须互异。候选节点按目标边中点的精确整数平方距离排序，`max_neighborhood`、度数和候选数上限只改变尝试哪些 OR moves，不参与 proof verifier 的真值判断，因此只能损失证明率，不能制造错误证明。

候选筛选可使用 CUDA。kernel 的一个线程处理一个 `c,d` task，用与其他 CUDA 模块相同的整数 `EUC_2D/CEIL_2D` 距离规则计算两个 2-opt 方向。返回的完整 flags 在主机逐项与 CPU 规范结果比较；不一致或显式 CUDA 失败返回 `unresolved`，`auto` 失败转 CPU。GPU 不生成或省略 AND replies。

## Hamilton replies

对中心点 `n` 枚举活动 CSR 中所有无序邻居对 `n0,n1`。实现逐式对应参考 `CCelim_check_neighbors_three_swap`：

1. `n-n0` 与 `n-n1` 分别必须至少有一个与 `a-b` 的 2-opt 方向不严格改善；
2. 邻居对不能恰为 `a,b`；
3. 必须满足非改善 3-opt 不等式

\[
d(a,b)+d(n,n_0)+d(n,n_1)
\le
d(n_0,n_1)+d(a,n)+d(b,n).
\]

被这些严格改善测试排除的邻边或邻边对不可能与目标边同时出现在最优巡回。其余 reply 必须完整取 `R_c\times R_d`；若其中一个集合为空，AND 节点按逻辑真值封闭。

## 叶证明与 verifier

每个 reply 先把目标边和两条三节点 Hamilton 路径交给 `NormalizePathSystem`。度数冲突、重复边或提前成环直接构成 path-infeasibility leaf；合法路径系统进入 3/4/5-opt，并可按配置转入 18-block 精确 DP。叶 proof 仍要求 outside matching 全覆盖和独立 witness 重放。

`VerifyHtShallowProof` 不信任搜索器保存的候选计数、reply 数量或路径结论。它重新检查：

1. graph snapshot hash、目标活动边和 `c,d` 不兼容条件；
2. 从 CSR 重新枚举两个中心的全部 surviving neighbor pairs；
3. proof records 与确定性笛卡尔积逐项、逐序一致；
4. 每个无效路径重新规范化后确实无效；
5. 每个合法路径的嵌套 leaf proof 重新绑定 graph/path/compatibility hashes 并完整验证。

因此删除一个 reply、替换 move、修改快照或篡改叶哈希都会失败。

## 回归证据

- 30 组随机 8 点完整图上，对每条目标边和每个非端点中心，把 reply 列表与直接翻译自 LocalElimination 的公式 oracle 比较；
- 固定 8 点完整图的目标边 `0-5`，确定选择 `c,d=1,2`，完整覆盖 30 个 replies，其中包含真实 k-opt/exact leaf proof；
- 另有零 surviving reply 的 vacuous AND、reply 预算 `unresolved`、缺 reply、错 move、错 snapshot 和错 leaf hash 门禁；
- 稀疏 32 点图的 EUC/CEIL `c,d` flags 与 CUDA 逐项一致；GPU 2 上 compute-sanitizer memcheck 为 0 error。

## 后续衔接

M4.3b2 已在浅层根语义上加入 extra-point/extra-end 递归、continuation arena 和版本化全局证书，详见[递归 Hamilton–Tutte 证明](08_Hamilton_Tutte_Recursive_Proof.md)。`gpu-eliminate` 的自动候选仍只提交 JV；完整递归 proof 已可通过 [`ht-commit`](21_Hamilton_Tutte_Epoch_Commit.md) 进入独立的不可变 epoch，浅层 proof 本身仍不直接授权删除。
