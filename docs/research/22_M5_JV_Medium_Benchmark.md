# M5 JV 中大型基准与首轮 CSR 优化

## 1. 结论

M5 已完成 `pcb3038`、`rl5915`、`d15112` 的 JV 基线。所有正式运行都绑定 clean commit `cac180f99a332b1cef72695f2f949f79692dbeb4`，CPU/CUDA 输出边集逐字节一致，每份 proof 均由独立 CPU Release 进程从初始图重放成功。

进程内稳态算法中位数加速为 `8.132× / 25.679× / 37.426×`；每次重新启动 CLI 的端到端加速为 `0.186× / 0.940× / 6.139×`。因此，中大型计算本身已经明显受益于 CUDA，但 `pcb3038` 和 `rl5915` 的短命令行运行仍被 CUDA context、输入解析、proof/边文件写出与进程启动成本覆盖。两种口径必须同时报告。

这组数据只评估已经闭环的 JV 规则，不代表完整 Hamilton–Tutte 自动扫描或 cuOpt/Concorde 交替流程的性能。

## 2. 输入与环境

正式运行时间为 2026-09-01 UTC（新西兰本地日期 2026-09-02），使用物理 GPU 1；`CUDA_VISIBLE_DEVICES=1` 后程序内设备号为 0。GPU 是 RTX 4000 Ada Generation，驱动 `610.43.02`，显存 20,475 MiB，开始运行时空闲 20,043 MiB。编译器为 GCC 16.1.1，CUDA toolkit 为 13.3，构建 preset 为 `cuda-release`。

| 实例 | n | 初始 m | TSPLIB SHA-256 | 边集 SHA-256 | 官方最优值 |
|---|---:|---:|---|---|---:|
| pcb3038 | 3,038 | 6,883 | `0b2229669b5d2916e812c36eaf76fb7b2bcb7ea09c6828e2599e0733ba18e933` | `9e6e2e612c0333c00d04cd2f91e3358e6f11508da1c56453da6e45da674cdcc7` | 137,694 |
| rl5915 | 5,915 | 29,143 | `1fa53f8f81192442f3671fb912b50d4762387b5e5666a438181d3a61d310b460` | `17abfd7a18a5b1ae28a6ff549e9ebc509ee63a2f56f033df58227de2b85f1599` | 565,530 |
| d15112 | 15,112 | 166,499 | `24b1f14a65da0c6ac2dc4001ad2eb2144f1cb6b0969c8abf90bb52a297229c8a` | `3d1769d87e2e742601a3298d34389d1cf6d34ffee00493f5c88c186976565da8` | 1,573,084 |

最优值来源为 [TSPLIB 官方最优解表](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/TSP-BEST.html)。完整哈希、绝对输入路径、GPU 列表和工具版本保存在各次 `run-manifest-v1` 中。

## 3. 正确性结果

| 实例 | epoch | 扫描边次 | 删除 | 最终 m | 删边率 | 最终图哈希 | 输出边文件 SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| pcb3038 | 2 | 13,587 | 179 | 6,704 | 2.601% | `90d13888e351df17` | `9b2229bc047f4fa107a474bf8201b7b086db11df64485217d36c67e53c326a54` |
| rl5915 | 2 | 57,736 | 550 | 28,593 | 1.887% | `0174cf46124ce870` | `54472ebd8115ee84482b61276ade00bc9d24fe35d730e34d2cb4067783639f4d` |
| d15112 | 3 | 484,885 | 7,312 | 159,187 | 4.392% | `76e196dd53d887d5` | `8ab757da10cf5b19258620b0f87351baa0f260e1ee7d10e8caf3a78528a0cd14` |

每个 backend 先预热一次，再独立计时五次。脚本对预热和每次计时运行都执行 proof 重放，并检查：

- GPU 候选逐条通过同一不可变 snapshot 上的 CPU JV verifier；三实例的拒绝数均为 0；
- CPU 与 CUDA 最终边文件逐字节相同；同一 backend 的五次输出也逐字节相同；
- 独立 CPU Release `verify` 重建出的最终边数与哈希一致；
- GPU 差分回归通过，`compute-sanitizer --tool memcheck` 报告 0 error。

`pcb3038` 另由项目外只读 LKH-3.0.13 生成成本 137,694 的 TSPLIB tour。`tour-check` 重新计算全部整数距离，确认初始与消元后边集都缺失 0 条 tour 边。原 tour SHA-256 为 `1b4e9f3c4da450f89af4a1457924bc5892b748844b0cd2f6f094a566817b4537`，与旋转/反向无关的规范 tour 哈希为 `ca0238497c090a3c`。

本地和 TSPLIB 官方索引没有提供 `rl5915`、`d15112` 的最优 tour witness；两者的 manifest 因而明确记录 `protected_tour none`。这不削弱逐条数学 proof 的 soundness，但“已知最优 tour 数据绑定”仍是 M5 的未完成数据门禁。一个已知最优 tour 本来也只能证明该 tour 未受损，不能替代规则证明。

## 4. 性能结果

### 4.1 独立 CLI 进程

该口径包括进程启动、输入解析、CUDA context、图/proof 写出；proof 重放作为单独阶段记录，不计入 wall。

| 实例 | CPU median / P95 (ms) | CUDA median / P95 (ms) | wall 加速 |
|---|---:|---:|---:|
| pcb3038 | 42.919 / 42.953 | 230.946 / 240.368 | 0.186× |
| rl5915 | 245.198 / 248.253 | 260.934 / 290.348 | 0.940× |
| d15112 | 3026.664 / 3037.380 | 493.028 / 505.365 | 6.139× |

短任务不应按“一次调用一个新进程”的方式部署 CUDA。正式管线应保持进程与 CUDA context 常驻，或者先按预计工作量路由到 CPU。

### 4.2 进程内稳态算法

图只加载一次；CPU 和 CUDA 各预热一次。每次计时前复制不可变初始快照，复制本身不计入算法时间；每次结果仍在计时后独立重放。

| 实例 | CPU median / P95 (ms) | CUDA median / P95 (ms) | 算法加速 | CPU replay median (ms) | CUDA-result replay median (ms) |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 19.482 / 19.611 | 2.396 / 2.439 | 8.132× | 1.112 | 1.105 |
| rl5915 | 197.245 / 198.884 | 7.681 / 7.690 | 25.679× | 5.597 | 5.299 |
| d15112 | 2800.545 / 2804.730 | 74.829 / 79.725 | 37.426× | 56.657 | 55.348 |

### 4.3 CUDA 算法阶段中位数

| 实例 | snapshot/hash (ms) | GPU propose (ms) | CPU verify (ms) | commit/CSR (ms) | algorithm (ms) |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 0.367 | 1.106 | 0.040 | 0.495 | 2.396 |
| rl5915 | 1.327 | 2.777 | 0.297 | 1.929 | 7.681 |
| d15112 | 10.591 | 20.632 | 11.025 | 26.007 | 74.829 |

在 `d15112` 上，commit/CSR 占算法时间约 34.8%，GPU propose 约 27.6%，snapshot/hash 约 14.2%，CPU verifier 约 14.7%，其余控制与最终哈希约 8.7%。因此下一轮首先优化提交/CSR 与快照扫描，而不是继续压缩已经很短的 kernel。

## 5. 首轮工程优化

原 `GraphSnapshot::RebuildCsr` 对每个顶点都构造临时 `tuple` 数组并排序。规范边已经按 `(u,v)` 全局排序；对任一顶点，较小邻点来自 `(u,vertex)`，较大邻点来自 `(vertex,v)`，两段及其拼接天然递增，删除活动边也只会得到有序子序列。

提交 `cac180f` 增加以下快路径：

1. 重建前验证所有边均为 `u<v` 且全局按 `(u,v)` 排序；
2. 满足不变量时直接保留填充顺序，跳过逐 row 临时分配和排序；
3. 任何手工构造、非规范或乱序图继续执行原逐行排序；单元测试专门打乱边数组并验证所有 CSR 二分查询。

该变化不修改边编号、proof 格式、图哈希或 CPU/GPU 候选语义。与 `2a653e4` 基线相比，三实例最终哈希完全不变；五次样本的整体变化较小，不能据此宣称独立的统计显著加速。它的主要价值是移除已证明冗余的工作，并让 `snapshot/propose/verify/commit` 四段可被后续优化逐项量化。

## 6. 复现

```bash
cmake --preset cuda-release -DCUDAEE_BUILD_BENCHMARKS=ON
cmake --build --preset cuda-release --target cudaee cudaee_jv_benchmark

CUDAEE_BENCHMARK_GPU=1 \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/pcb3038.tour \
  tools/run_jv_benchmark.sh pcb3038 5
CUDAEE_BENCHMARK_GPU=1 tools/run_jv_benchmark.sh rl5915 5
CUDAEE_BENCHMARK_GPU=1 tools/run_jv_benchmark.sh d15112 5
```

本报告使用的 run id 为：

- `pcb3038-jv-20260901T185804Z-2513316`；
- `rl5915-jv-20260901T185813Z-2513646`；
- `d15112-jv-20260901T185823Z-2513968`。

`artifacts/` 按仓库策略不提交；结果身份由 clean commit、输入 SHA-256、参数、GPU/工具版本、proof、最终图哈希与输出 SHA-256 联合绑定。

## 7. 下一步门禁

1. 为 CUDA JV 建立跨 epoch 的设备驻留 workspace，只更新活动位和变化后的 CSR；返回候选后仍执行完整 CPU verifier。
2. 评估不改变既有内容哈希格式的快照扫描优化；若无法保持 proof 兼容，宁可保留当前 O(m) 哈希。
3. 为 `rl5915`、`d15112` 补齐来源明确、成本重新计算通过的最优 tour witness。
4. 增加全图 HT 目标调度、预算与 sidecar 批量 commit；JV 基准不能替代 HT 吞吐结论。
5. 完成 M3.1 对偶稳定化和 exact reduced-cost 边集导出后，再测 LP—JV—HT 交替固定点的端到端性能。

第 1 项已在后续提交完成，结果见 [M5 JV CUDA 跨 epoch 驻留缓存](23_M5_JV_Device_Residency.md)；本文件保留 `cac180f` 基线数字用于可比性。
