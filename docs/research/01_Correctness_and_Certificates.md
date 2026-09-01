# 正确性与证书

## 图消元证明链

设 `G_k` 是第 `k` 轮不可变稀疏图。GPU 对边 `ab` 输出方法、见证点和快照哈希；CPU 只在哈希匹配的 `G_k` 上复算：

- JV：存在候选点 `c`，使 `c` 的所有活动邻点 `d != a,b` 都不满足
  `C_ab + C_cd <= C_ac + C_db` 或 `C_ab + C_cd <= C_ad + C_bc`。
- 度数前提：处理 `ab` 时 `deg(a),deg(b)>2`，受保护 tour/fixed edge 不参与删除。
- 提交前再次执行最小度门禁；多个独立有效候选也不能令顶点度数降到 2 以下。

JV-only `proof-v1` 记录规范化图哈希、epoch、边、方法和见证。含 HT 删除时使用自包含 `proof-v2`：outer record 绑定同一快照和目标边，并唯一引用内嵌的 recursive HT V1 continuation arena。验证器按 epoch 在图副本上重放“验证全部候选—排序—度数门禁—提交”，任何缺失、额外、重复或过期 sidecar 都失败且不发布部分图。

## 距离精确性

GPU 首期只接受整数坐标，平方距离必须能在无符号 64 位中安全表示。整数平方根后：

- `EUC_2D`：若 `S-r^2 > r` 则取 `r+1`，否则取 `r`；
- `CEIL_2D`：若 `r^2 < S` 则取 `r+1`，否则取 `r`。

这避免 CPU/GPU `sqrt` 舍入差异。非整数坐标可由 CPU 研究模式读取，但 CUDA 路径安全回退。

## 任意对偶向量的严格 LP 下界

对最小化模型

`min c^T x + o,  A x {<=,=,>=} b,  l <= x <= u`

选取符号合法的 `y`（`<=` 行要求 `y<=0`，`>=` 行要求 `y>=0`，等式自由），令 `r=c-A^T y`，则

`L = o + b^T y + sum_i min_{l_i<=x_i<=u_i}(r_i x_i)`

是严格下界。cuOpt 对偶先按行方向裁剪，再量化为分母 `2^24` 的有理数；当 `A,b,c,l,u,o` 均为整数且中间结果不溢出 128 位时，sidecar 输出精确分子与分母。否则只输出浮点诊断，`certified=false`。

这个证书只覆盖序列化模型中的变量。TSP 完整图中未进入稀疏 LP 的边也可能有负 reduced cost，因此模型内证书不能直接替代 Concorde 的完整图精确定价。

## Concorde 授权链

允许 LP reduced-cost 删除需要：

1. epoch 模型及列到 TSP 边的映射哈希匹配；
2. cuOpt 返回可用对偶，无 NaN/Inf；
3. 对偶符号归一化并转换为 `CCbigguy`；
4. Concorde 对完整候选边域执行 exact pricing，并计入所有负 reduced cost 惩罚；
5. 以严格 `L`、已认证上界 `U` 判断 `r_e > U-L`；
6. 输出可重放证书后才提交。

在第 3–5 步尚未接通前，`lp-solve` 只负责求解和导出候选信息。

## 失败策略

- 哈希、维度、CSR、索引、距离、上界或证书任一校验失败：本轮删除数为 0。
- cuOpt 非 optimal/可接受迭代终止：保留结果用于诊断，但不生成可授权证书。
- CPU 与 GPU 分歧：保存最小重现快照，停止当前 epoch。
- 验证器不会“修复”证明；它明确返回非零退出码。
