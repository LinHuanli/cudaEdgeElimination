# Concorde overlay integration

Concorde 03.12.19 和 QSopt 只允许学术研究用途，因此本仓库不包含其源码或二进制。`tools/bootstrap_concorde.sh` 校验外部源码树与 QSopt SHA-256，再按全部补丁的内容哈希，在 `.deps/concorde-03.12.19-<hash>/` 构造不可变 overlay。补丁重叠不会依赖反向试打判断，重复运行也不会修改原始源码。

补丁序列：

- `0001`：从 QSopt 导出 CSR、行方向、边界和 Concorde 列—边映射；
- `0002`：校验 sidecar 状态/哈希/完整结尾，注入 `CCbigguy` 对偶并调用 `CCtsp_exact_price(..., complete_price=1, ...)`；
- `0003`：以 Concorde graph 的整数边长为目标权威来源，并用当前 QSopt 解验证映射目标等于 `QSget_objval`。

可重复握手测试：

```bash
./tools/bootstrap_concorde.sh
./tools/run_concorde_cuopt_epoch.sh
./tools/run_concorde_cuopt_epoch.sh --tamper-model-hash
./tools/run_concorde_cuopt_epoch.sh \
  --tsp third_party/ElimTSP/data/pr299.tsp --timeout 600
```

脚本使用唯一的仓库内产物目录；Concorde 导出完整 `END` 后暂停，cuOpt 解写入临时文件并原子发布。正常路径要求三个模型哈希一致、`OPTIMAL`、独立残差门禁通过、`EXACT_PRICED` 且完整图下界不超过模型目标与上界。负向路径证明错配哈希被拒绝且不生成证书。

当前测试以 `-B` 只验证根节点下界，不导出 Concorde 消元后的边集。cuOpt 在退化稀疏 LP 上可能给出完整图意义下较弱的对偶；完整定价保证安全，但后续仍需迭代补列/对偶稳定化来恢复删除强度。
