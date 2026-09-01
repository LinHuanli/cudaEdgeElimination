# Concorde overlay integration

Concorde 03.12.19 和 QSopt 只允许学术研究用途，因此本仓库不包含其源码或二进制。运行 `tools/bootstrap_concorde.sh` 会把外部 `references/concorde_code` 复制到 `.deps/`、校验 QSopt 下载，并应用 `patches/0001-add-lp-epoch-export-hook.patch`。

补丁增加 `CClp_dump_lp_epoch`，直接从 QSopt 取出 CSR、目标、行方向、边界和列名对应的 TSP 边端点。设置环境变量 `CUDAEE_LP_EPOCH_OUT` 后，成功完成 cutting loop 时会原子语义地覆盖指定 epoch 文件（调用方应把路径放在本仓库内）。导出失败会令该 cutting loop 失败，避免继续使用不完整模型。

当前补丁只完成模型与列边映射导出。cuOpt 对偶注入 `CCbigguy` 和完整图 exact pricing 仍是 M3 的待完成门禁；在此之前 `lp-solve` 结果不能授权 TSP 边删除。
