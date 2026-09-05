# FGPU 交错基准与结果身份

## 运行工具

`tools/benchmark_fgpu.py` 在指定单 GPU 上按 AB/BA 顺序运行完整求解，不设置目标数、
epoch、回复数或时间限制。`--warmups` 和 `--repetitions` 仅控制完整求解的重复次数。

```bash
env PYTHONDONTWRITEBYTECODE=1 python3 tools/benchmark_fgpu.py \
  --variant baseline=build/baseline-563d78a/fgpu-elim \
  --variant current=build/cuda-release/fgpu-elim \
  --instance third_party/ElimTSP/KH-elim/pr299.tsp \
  --input-edges third_party/ElimTSP/KH-elim/pr299.edg \
  --tour tests/data/pr299.opt.tour --expected-cost 48191 \
  --gpu-uuid GPU-8ce2c8bb-d2fd-214d-c9ce-b22a66f38bbb \
  --output-root artifacts/strength-complete/pr299-new-paired \
  --repetitions 3 --warmups 1
```

GPU UUID 需按实际空闲设备选择；不能把示例 UUID 当作空闲承诺。省略 `--input-edges`
就是完整图输入。对照 worktree 的单次产物写入其自身项目根下，以通过 CLI 的输出
边界约束；汇总仍写在 `--output-root` 下。已有输出目录不会被覆盖。

算法/执行消融通过 `--variant-args 'NAME=["--lp-backend","sec-dual"]'` 等显式传入，
工具拒绝借此覆盖输入、输出、设备或目标范围。正式比较始终使用 GPU-safe。

同一二进制可以声明为两个变体，然后分别传
`--variant-args 'cta2=["--point-cta-blocks","2"]'` 与
`--variant-args 'cta4=["--point-cta-blocks","4"]'`。这只改变编译期寄存器／驻留
策略，不改变搜索范围。Opt34 后端可选 `permutation`、`prescreen-permutation`、
`prescreen-subset-dp`；最终默认选项依据见
[同卡消融结果](71_FGPU_Transaction_PDHG_and_PathEnd.md)。

## 记录哪些身份

每个正式 solve JSON 包含：git commit、dirty 状态、diff/source-tree SHA-256、实际
`/proc/self/exe` SHA-256、编译器/CUDA/架构，以及 GPU UUID/PCI/驱动/运行时版本。
输入 instance、edges、tour 分别计算 SHA-256；完整图的 input-edges hash 为 null。

基准工具还在每次运行前后检查输入与实际二进制未变，记录真实命令、UTC 开始时间、
stdout/stderr、GPU 计算进程快照和子进程墙钟。非零退出或身份不一致立即停止汇总。
性能实验使用冻结的可执行文件副本时，副本也必须位于本项目内。

## 时间口径

- `process_wall_ms`：完整子进程，包括启动、解析、GPU、输出及 manifest/hash。
- `end_to_end_ms`：求解器内部的端到端计时，旧实现不包含 manifest 写入；不能与上项混用。
- `lp_ms`：LP service 总时间，包含 point/fixing 等组合服务，不能称为纯 LP 求解时间。
- `lp_solver_ms / lp_cut_separation_ms / lp_point_ms / lp_fixing_ms /
  lp_pair_filter_ms / pdhg_model_ms`：细分阶段。
- `pdhg_ms`：`lp_solver_ms` 的子集，不能再重复相加。

`point_registers`、`point_active_blocks_per_sm` 与 `point_local_bytes_per_thread`
来自实际选择 kernel 的 CUDA 属性／occupancy 查询，不是运行期间 achieved
occupancy 或 SM 利用率的采样。后两者需要另外做 profiler 实验。

主结果比较终态 edge/fixed/non-pair 和 `final_state_hash`。reply/proposal 计数可能因
短路窗口中已在执行的线程不同而略有变化，不能据此判定终态不确定。

## Clean 的边界

工具默认在发现同卡其他计算进程时拒绝运行。`--allow-busy` 只用于开发观测，汇总会
明确标记。当前工具检查的是运行边界，不是全程独占资源证明：
`boundary_gpu_checks_clear` 不能改称“全程无干扰”。同节点 CPU 负载、时钟、温度、
其他卡工作以及 profiler 的影响仍须结合实验记录判断。

不同算法终态强度不同时，仅报告墙钟比与强度差，不使用“等强度加速”措辞。
sanitizer 和 profiler 运行只用于正确性/诊断，不能并入正常性能样本。

## 本次正确性检查的复现边界

先构建、运行 `ctest --preset cpu-release` 和 `ctest --preset cuda-release`。CUDA
检查需要先设置 `CUDA_VISIBLE_DEVICES` 为所选空闲 GPU 的 UUID；`TMPDIR`、
`CUDA_CACHE_PATH` 均指向项目内 `.tmp/`，并设置 `PYTHONDONTWRITEBYTECODE=1`。
CTest 中 `exhaustive.fgpu_variants` 会在 build 目录内生成全部小图和逐变体输出。

对当前 CUDA build 中的 `cudaee_transaction_cuda_unit`、`cudaee_sec_replay_unit`、
`cudaee_unit` 分别运行：

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 build/cuda-release/tests/TEST_NAME
compute-sanitizer --tool racecheck --error-exitcode 99 build/cuda-release/tests/TEST_NAME
```

PDHG 单元使用 `build/cuda-release/tests/cudaee_sparse_pdhg_unit --sanitizer`；该选项
只缩短单元测试中的迭代批次，不是正式 solve 的截断选项。完整 LP 精度／非法模型
门禁仍由未带该选项的 CTest 执行。

正式小图 solve 的 memcheck 不过滤 kernel。为避免把大量 PDHG 重复更新与同一
单元的竞态检查重复执行，完整 solve 的 racecheck 使用以下过滤器：

```text
--kernel-name 'regex=Point|QuickHs|MainEdge|DirectFix|Nonpair|resident_transaction|resident_sec_replay|ApplyCommit|MarkDegreeTwoFixed'
```

这覆盖选定的候选／replay／提交内核，**不是所有 solve 内核的 racecheck 通过声明**。
最终检查的二进制身份、三个完整求解样例和日志前缀见
[实现报告的正确性门禁](71_FGPU_Transaction_PDHG_and_PathEnd.md#正确性门禁)。
