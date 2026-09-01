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
- 边映射要么完整，要么全部为 `-1`；
- 内容哈希与规范化重算一致。

Concorde patch 在 cut epoch 结束、模型稳定时调用 QSopt getters 导出；不得从 LP 文本反向猜测列身份。

## cuOpt sidecar

调用序列：

1. `cuOptGetFloatSize/GetIntSize/GetVersion` 做 ABI 门禁；
2. `cuOptCreateProblem` 构造连续 LP；
3. 创建 settings，固定随机种子、双精度 PDLP 和容差；
4. `cuOptSolve`；
5. 获取 termination、objective、primal、dual、reduced costs 和 solve time；
6. 独立重算 primal residual、目标、reduced cost residual；
7. 量化符号合法对偶并生成模型内精确下界。

输出 `lp-solution-v1`。只有 `status=OPTIMAL`、维度/残差可接受且精确下界成功时标记模型证书有效，但仍不能跳过 Concorde 完整图定价。

## Concorde exact pricing 接口

补丁新增一个显式入口，将外部对偶按 Concorde 行顺序写入 `lp->exact_dual`。写入前比较 epoch 行哈希；写入后调用现有 `CCtsp_exact_price` 路径，遍历完整边域并减去所有负 reduced cost。输出：

- exact lower bound；
- 每个活动边的 exact reduced cost；
- 上界来源及其证书哈希；
- Concorde/QSopt/补丁版本。

若 row mapping、cut 顺序或 complete-edge generator 不一致，入口必须拒绝继续。

## 集成测试

- 2×3 整数小 LP：手算最优值与有理下界；
- 同一 epoch 由 QSopt 与 cuOpt 求解，比较可行性和目标容差；
- 人为翻转一行方向，确保符号裁剪后下界仍有效；
- 删除一个未激活列造成负 reduced cost，确保模型内证书不能被标为完整图证书；
- pr299 一个 cut epoch 的导出—求解—精确定价往返。
