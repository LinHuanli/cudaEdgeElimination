# FGPU 单卡全常驻实现与端到端基准

## 1. 结论

`research/fgpu-fully-resident-single-gpu` 已实现并实测一条单 GPU 主求解链：

```text
完整图
  -> CUDA FP64 有向舍入区间 Geometry
  -> CUDA degree-box PDLP 与量化下界
  -> CUDA exhaustive JV fixed point
  -> CUDA Quick-HS fixed point
  -> 最终 device mask
```

在 `pcb442` 上，单张 RTX 4000 Ada 将 97,461 条完全图边确定性地降到
3,008 条，删除 94,453 条，删除率 96.9136%。7 次进程端到端 wall
中位数为 11.88 秒；GPU 主求解 wall 中位数为 7.927 秒。官方最优 tour
成本 50,778，最终缺边为 0，94,453 条删除记录全部通过独立 CPU 重放。

相同环境中的作者对照为：

| 对照 | 最终边 | wall | 本实现最终边 | 本实现进程 wall 中位数 | 等强度方向的结论 |
|---|---:|---:|---:|---:|---:|
| 作者 `KH -Jq` 单轮 | 12,914 | 94.89 s | 3,008 | 11.88 s | 更强且 `7.99x` |
| 作者 KH-HS/CUDA-JV 固定点 | 4,016 | 99.68 s | 3,008 | 11.88 s | 更强且 `8.39x` |
| 旧 FGPU 深 HT one-shot | 3,239 | 3,201.08 s | 3,008 | 11.88 s | 更强且 `269.45x` |

这里的 `7.99x/8.39x` 使用 `/usr/bin/time` 的完整进程 wall，而不是只取
kernel 时间。相对作者两项，本实现还分别少保留 76.71% 和 25.10% 的边，
因此已经越过同强度门槛。

## 2. “全 GPU”的准确边界

本实现称为**全 GPU 主求解**，含义是所有会决定搜索轨迹的阶段均在同一张
GPU 上运行，CPU 不逐边回答 GPU，也没有 CPU DFS/KH 回退：

- 图的 active mask、度数、定长邻接行和距离矩阵在整次搜索中常驻显存；
- Geometry、PDLP、JV、Quick-HS 候选生成和 stable edge-id 提交均为 CUDA kernel；
- 每个 epoch 只把已提交位图及紧凑见证复制回主机，CPU 结果不反馈给设备搜索；
- 达到局部与 LP 外层固定点后，才回传最终 mask。

仓库 `AGENTS.md` 同时要求任何正式删边必须经过 CPU 精确验证。因此发布路径
保留一个**搜索后的 CPU 安全审计**：它从初始图按 epoch 重放 GPU 紧凑见证，
任何一条失败都会在写边文件前终止。本实现不是“GPU 独立信任根”，也不应写成
`fully GPU-certified`。为了不隐藏这部分成本，报告同时给出：

- `gpu_solve_wall_ms`：已解析图到设备固定点和最终 mask 回传；
- `cpu_audit_ms`：CPU 精确证明审计；
- `trusted_total_ms`：输入读取、GPU 求解、CPU 审计和输出的应用内总时间；
- `process_wall_s`：外部计时的完整进程 wall，论文 speedup 采用这一列。

TSPLIB 解析、CPU 审计和文件写出仍在主机。它们是可信端到端时间的一部分，
不属于 GPU 搜索回退。

## 3. 实现结构

### 3.1 常驻图与确定性 epoch

`src/cuda/fgpu_resident.cu` 一次分配并上传：

- stable edge SoA：端点、权重、active/protected/proposed/committed 位图；
- `n*n` 距离矩阵和 active matrix；
- 每轮由 GPU 重建的 `(cost,node)` 严格排序定长邻接行与 degree；
- Geometry、PDLP 和紧凑见证 workspace。

同一 epoch 的 kernel 只读同一 active snapshot。单线程 `CommitKernel` 按 stable
edge ID 执行 degree `>2` 与 protected-tour 门禁，确保 GPU、CPU 和重复运行的
提交顺序一致。pcb442 常驻空间为 6,073,116 bytes。

### 3.2 GPU 有向舍入 Geometry

旧版 CUDA Geometry 使用普通 FP64 候选，再由 CPU MPFR 过滤；真实运行中有
44,685 个近似候选被拒绝，不能直接在 GPU 上提交。本分支把 Main Edge 的完整
代数谓词改成 CUDA FP64 interval arithmetic：

- 整数到 double 使用 `__ll2double_rd/ru`；
- 加、乘、除、平方根分别使用向下/向上舍入 intrinsic；
- 严格比较只有在一个区间的下界已越过另一个区间上界时成立；
- 除数含 0、平方根下界为负、余弦越界或非有限值时一律保留边。

Geometry 单阶段实测 97,461 → 11,032，GPU solve wall 0.106 秒；86,429 条
记录全部通过 256-bit MPFR 独立复核。相比旧 CPU-filter 路径的 85,297 条，
候选选择变化多证明了 1,132 条边，但没有放宽数值授权条件。

### 3.3 GPU PDLP 与 LP-box

degree equality + box relaxation 使用 edge-parallel subgradient/PDHG 风格更新。
最终 multiplier 量化为分母 `2^24` 的 `int64`，GPU 重算

\[
\hat L=2\sum_vq_v+\sum_e\min(0,c_e2^{24}-q_u-q_v),
\]

仅当 `L + max(0,r_e) > U` 时提出 forced-one 删除。CPU audit 使用 `__int128`
对同一量化 sidecar 重算。负距离、乘加安全余量不足、dual 非有限或任何
累加器可能溢出时直接失败关闭。

### 3.4 GPU Quick-HS

`src/fgpu/quick_hs_predicate.hpp` 是 CPU/CUDA 共用的 KH `-q` 浅层谓词：

- 每条目标边选取至多 10 个严格排序的 `c,d` 候选并尝试前 10 对；
- 完整枚举 `c`、`d` 的活动邻边对；
- 用整数 `opt22/opt23/opt222/opt232/opt233` 关闭每个 Hamilton reply；
- `m<=3`、最多 8 个节点的 path system 使用固定数组 Held–Karp DP；
- 默认行为对齐作者 `strong_3_opt=0, extra_edges=0, extra_nodes=0,
  ab_stretch=1, max_cd_count=10`。

证书 record 只保存 `(target edge, c, d, snapshot hash)`。独立 replayer 在相同
不可变快照上重新枚举全部 replies，不信任 GPU 返回的布尔值。

## 4. pcb442 分阶段结果

参数：`potential_candidates=32`、每个 PDLP epoch 5,000 iterations、最多
2 个 LP epoch、JV/Quick-HS 各最多 100 轮。实际运行 2 个 LP epoch、20 个
JV rounds 和 11 个 Quick-HS epochs 后收敛。

| 阶段 | 删除边 | 阶段后剩余边 |
|---|---:|---:|
| GPU interval Geometry | 86,429 | 11,032 |
| GPU LP-box（两轮合计） | 2,714 | 8,318 |
| GPU exhaustive JV（交错合计） | 300 | 8,018 |
| GPU Quick-HS（交错合计） | 5,010 | 3,008 |
| 合计 | 94,453 | 3,008 |

最终边 content hash 为 `f5d359ca9b12e193`，边文件 SHA-256 为
`cad2ee09b4b690ba6dfd5e95558140b8f4dcdee3b696f1f5d617f1e90777ec6e`，
规范证书 SHA-256 为
`6542aaf38a77543d27cd1b6c03635c079bdeb1fb082f2c2969366f1867eb5f1d`。

## 5. 七次重复计时

主机 `cuda-small1.ecs.vuw.ac.nz`，单张 NVIDIA RTX 4000 Ada 20 GiB，驱动
610.43.02；运行期间使用物理 GPU 1。7 次的最终边、content hash、边文件与
证书均逐字节一致。

| run | GPU solve | CPU audit | trusted total | process wall |
|---:|---:|---:|---:|---:|
| 1 | 8.226 s | 3.400 s | 11.921 s | 11.98 s |
| 2 | 7.189 s | 2.463 s | 10.017 s | 10.08 s |
| 3 | 7.927 s | 3.539 s | 11.809 s | 11.88 s |
| 4 | 7.328 s | 3.794 s | 11.449 s | 11.52 s |
| 5 | 8.481 s | 3.653 s | 12.472 s | 12.54 s |
| 6 | 7.658 s | 3.722 s | 11.692 s | 11.76 s |
| 7 | 8.081 s | 3.759 s | 12.185 s | 12.26 s |
| **中位数** | **7.927 s** | **3.653 s** | **11.809 s** | **11.88 s** |

只用 GPU solve wall 计算，相对作者单轮/固定点分别为 11.97x/12.57x；这两项
只用于定位剩余 CPU 审计成本，不作为端到端主结论。

旧快速链为 8,015 条、4.872 秒。当前版本少保留 62.47% 的边并真正跑到
Quick-HS 固定点，但可信进程 wall 是其 2.44 倍。因此若下游只要求约 8k 边，
旧快速配置仍更快；若要求文章固定点级约 4k 或更少边，当前 resident 配置才是
公平对照。

## 6. 正确性证据

1. 在线 CPU audit 重放 94,453 条记录并要求最终 mask 与设备逐 bit 相同。
2. 新进程 `fgpu-elim verify` 再次重放通过；单次独立验证约 4.15 秒、峰值
   RSS 39,564 KiB。
3. 官方最优 tour 成本为 50,778，在线门禁和独立验证均为零缺边。
4. 7 次输出的最终哈希、`.edg` SHA-256 与 `.fgcert` SHA-256 全部一致。
5. `pr299` 回归从 1,208 条边经 90 条 JV 和 252 条 Quick-HS 删除收敛到
   866 条；证书重放通过，并有篡改 `c` witness 的负向测试。
6. 小型完整图回归同时覆盖 resident Geometry、LP、固定点与输出重放。
7. CUDA 提交前运行 GPU 差分测试与 `compute-sanitizer`；CPU-only stub 和
   sanitizer 构建也必须通过。

## 7. 复现

正式七次运行：

```bash
# 物理 GPU 1；若显存被驱动上下文占用但利用率可接受，按实验约定允许使用。
CUDAEE_ALLOW_BUSY_GPU=1 tools/run_fgpu_resident.sh pcb442 1 7
```

脚本会检查锁定的 TSP/tour SHA-256、要求 clean worktree、构建 CUDA Release、
每次独立验证证书、检查 7 次输出逐字节一致，并在仓库 `artifacts/` 内生成
`measurements.tsv` 与 `benchmark.manifest`。不传 GPU ordinal 时使用
`tools/select_gpu.sh` 选择本机卡；跨服务器先用 `gpu-free` 找空闲节点，再在
目标节点执行同一脚本。

单命令调试形式：

```bash
CUDA_VISIBLE_DEVICES=1 build/cuda-release/fgpu-elim resident \
  --instance pcb442.tsp --tour pcb442.opt.tour \
  --tour-role known-optimum --expected-cost 50778 --device 0 \
  --potential-candidates 32 --pdlp-iterations 5000 --max-pdlp-epochs 2 \
  --max-hs-epochs 100 --max-jv-rounds 100 \
  --enable-geometry 1 --enable-pdlp 1 --enable-quick-hs 1 --enable-jv 1 \
  --output-edges out/pcb442.edg --fixed out/pcb442.fix \
  --nonpairs out/pcb442.nonpairs --certificate out/pcb442.fgcert \
  --manifest out/pcb442.manifest
```

## 8. 尚未覆盖的设计终态

这次实现解决的是“单 GPU 主链是否真的能形成端到端加速”问题，不等于
`FGPU-Elim_Method_OneShot_Plan.md` 中所有远期模块均已完成：

- 完整图目前仍由 CPU `GraphSnapshot` 物化后一次上传，尚未做 100k 节点 tile streaming；
- LP 是 degree-box relaxation，尚无 subtour cut pool 与异步 cuOpt service；
- Quick-HS 对齐作者默认 `-q` 浅层规则，尚未实现 extra-edge/extra-node 与完整
  wavefront Hamilton–Tutte proof DAG；
- 当前只输出删边与 degree-2 fixed，non-pair 仍为空；
- CPU 后置审计仍占端到端中位数约 3.65 秒。后续可优化 verifier，但不能删除
  仓库规定的独立信任边界。

这些缺口均以保留边或禁用未实现能力处理，不影响本次 3,008 条结果的证书有效性。
