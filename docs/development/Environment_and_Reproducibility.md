# 环境与可复现性

## 固定环境

- Linux x86_64，NVIDIA 驱动支持所选 CUDA toolkit；
- CMake >= 3.27、Ninja、C++20 编译器、`pkg-config`、MPFR 与 OpenSSL Crypto 开发包，
  以及可选的 OpenMP C++ runtime；OpenSSL 用于流式 SHA-256 构建／输入身份，不参与数值求解；
- 主环境 CUDA toolkit 13.x、目标 `sm_89`；远端门禁另验证 CUDA 12.6、`sm_86`；
- 项目内 `.venv` 安装 `libcuopt-cu13==26.8.0`；
- ElimTSP 子模块固定提交 `d7bacf0d...`；
- 外部 Concorde 完整源码树哈希 `f9caef4b41140a48cec906cf0456c266a2e83e5a1392b355694a99b151a819f6`；
- QSopt `qsopt.a` SHA-256 `5dcf323c7fce85e8b9de7ce79aabc17b672e224b77e2a89370c4e35da07434ee`；
- QSopt `qsopt.h` SHA-256 `647729f1bd77e1263ecf35e1897c705ef1cb45e2d65dbd9cb8fdf5df5ae65624`。

`tools/bootstrap.sh` 只在仓库内创建环境；不使用 sudo，不修改 shell profile。cuOpt wheel 较大，安装前运行空间门禁。

`tools/bootstrap_concorde.sh` 以全部 `.patch` 内容的 SHA-256 前缀选择 `.deps/concorde-03.12.19-<hash>`。补丁集合变化时创建新 overlay，旧目录保持可恢复且不会污染新构建；所有目录均被 Git 忽略。

## GPU 选择

`tools/select_gpu.sh` 从 `nvidia-smi` 查询每卡利用率和空闲显存，选择利用率最低、空闲显存最大的卡并打印索引。调用者设置 `CUDA_VISIBLE_DEVICES`；记录原始物理索引和查询结果。无法查询时不猜测，CPU 自动回退。

上述 CPU 回退仅适用于旧的自动候选器入口；正式 `fgpu-elim solve --mode gpu-safe`
没有 CPU 求解回退。性能对照优先按 UUID 固定设备，并使用
[`tools/benchmark_fgpu.py`](../../tools/benchmark_fgpu.py) 检查运行边界、记录真实
进程 wall 和可执行文件 SHA-256。当前 native sparse PDHG 不依赖 cuOpt wheel；
cuOpt 仍供独立的历史 LP 实验入口使用。

## 构建档位

- `cpu-debug`：无 CUDA，Debug，ASan/UBSan；
- `cuda-release`：CUDA `sm_89`，Release，关闭 fast-math；
- `cuda-debug`：CUDA 调试，供 compute-sanitizer；
- `cuda-sm86-release` / `cuda-sm86-debug`：RTX A4000 等 Ampere 卡的远端 Release/Debug 门禁；
- `lp-release`：与 cuda-release 相同，运行时从 `.venv` 动态加载 cuOpt。

所有 preset 把输出放在 `build/<preset>`。CUDA 实码架构由缓存变量 `CUDAEE_CUDA_ARCHITECTURES` 控制，preset 必须设置该变量，不能依赖与 target property 脱节的全局默认。`compile_commands.json` 只建立仓库内符号链接或由编辑器直接读取 build 文件。

`CUDAEE_ENABLE_OPENMP=ON` 时，8,192 个 cost cells 以上的 CPU 精确矩阵按 task row 静态分片；找不到 OpenMP 时安全退回串行。正式 HT benchmark 固定 `OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=spread`、`OMP_PLACES=cores`，并把 `CUDAEE_CPU_COST_THREADS`（默认 8，范围 1–8）写入 manifest。线程只改变独立 row 的计算时序，不改变矩阵布局和 proof 消费顺序。

显式 `--target-devices` 使用 `CUDA_VISIBLE_DEVICES` 映射后的 ordinal。正式多 GPU A/B 先以 UUID 固定可见顺序，再检查每张物理卡的显存和利用率；`tools/run_ht_target_multigpu_ab.sh` 默认每 worker 使用 4 个 OpenMP threads，并记录 `worker_count × CPU_COST_THREADS`，避免把 CPU 过量订阅误当作 GPU 扩展性。

M5 三份最优 tour 不入 Git；先运行 `tools/fetch_m5_opt_tours.sh`。来源 URL、SHA-256 和生成文件名锁定于 `configs/m5_opt_tours.tsv`；即使哈希匹配，脚本仍用本项目重算节点置换、精确 tour 成本和稀疏图边完整性。所有下载与临时文件都位于仓库内的 `artifacts/` 和 `.tmp/`。

## 运行清单

`run-manifest-v1` 至少包含 Git commit/dirty 状态、子模块提交、编译选项、GPU/驱动、cuOpt 版本、输入哈希、命令行、随机种子、开始结束时间和输出哈希。没有这些字段的性能数字只能作为临时观察。
