# cudaEdgeElimination

面向对称 TSP 的可验证 GPU 局部边消元研究实现。首期把 GPU 作为候选生成器，并由 CPU 在不可变图快照上逐条复核；LP sidecar 使用 cuOpt 求近似解，但只有独立的精确定价/定点证书可以授权删除。

当前实现范围：

- TSPLIB `EUC_2D` / `CEIL_2D` 与 Concorde 稀疏边文件读取；
- Jonker–Volgenant（JV）快速消元的 CPU 基线与 CUDA 候选生成；
- JV 跨 epoch 精确静态键、动态 CSR edge-id 刷新与增长型 CUDA 驻留 workspace；
- epoch 快照、CPU 复核、最小度保护、确定性提交，以及兼容 JV V1 的自包含 HT V2 证明日志；
- `lp-epoch-v1` CSR 模型、cuOpt C API 动态 sidecar、残差与精确定点下界，以及按稳定边/行身份投影且带覆盖率门禁的 PDLP warm start；
- Concorde 受限 overlay、列—边映射、`CCbigguy` 对偶注入与完整图精确定价证书；
- 路径系统规范化、`m<=6` CPU 生成/CUDA 查询兼容表与 `m=7` CPU 回退；
- proper 3/4/5-opt CPU 叶 witness、inside coverage 与 `path-kopt-proof-v1` 重放；
- 批量 CUDA k-opt 精确成本矩阵的逐 cell CPU 差分接口，以及 broker 专用的紧凑 candidate mask；严格改善 witness 仍由 CPU 完整重建；
- 收缩 forced outside matching 的 CPU 精确 Held–Karp 子集 DP 困难叶回退；
- 相邻 popcount 层的紧凑 exact value DP，以及 `block_count<=13` 的单 CTA CUDA 候选后端；CUDA 阴性保留边，阳性仍由 CPU traceback/witness verifier 认证；
- 浅层 Hamilton–Tutte `c,d` AND–OR 根证明与 CPU 复核的 CUDA 候选筛选；
- CPU 递归 Hamilton–Tutte point/end moves、continuation arena 与全局 `recursive-ht-proof-v1`；
- 主机 BFS 工作图、cooperative multi-block CUDA continuation 传播、跨父状态 Hamilton/end reply count/write、point/end path-append、规范 child edge SoA、增量 k-opt leaf cost block 融合、GPU 驻留缓存与 128-cell CPU long-tail，并由 CPU 完整差分复核；
- `ht-prove` sidecar 的整批 CPU 重放、不可变快照绑定、规范度数门禁和 `ht-commit` 原子删边；
- `ht-scan` 的确定性有界目标切片、允许多 worker 共享单张固定 GPU，以及提交前唯一完整 CPU 重放与 V2 原子提交；
- `local-eliminate` 的 JV 固定点、HT 无提交 sweep 推进、提交后目标重排、可选目标级多 GPU，以及单一可重放 V2 证明组合；
- HT scan V2 阶段计时，以及用 `--leaf-backend` 将 CUDA leaf cost 与 CPU 候选器解耦的混合路径；
- CPU leaf CLI 默认开启、auto/CUDA 默认关闭且可显式 0/1 覆盖的 `--fuse-leaf-buckets` 调度；
- 版本化短路 AND/OR Trace、`ht-trace-replay` 推测窗口模拟，以及保持 DFS 规范 proof 的 opt-in `transposed` host-window；
- 单 GPU 跨 target heterogeneous leaf broker：两请求机会式微批、独立 required edges、candidate masks 与 CPU witness 精确复核；
- HT scan V19 leaf/setup/cost/reply/path/root/point-candidate/cache/target-worker/broker/调度子阶段计时及 V20 五路 benchmark summary；
- point-candidate 严格全序的有界 Top-K 选择，并保留无界分支的完整排序语义；
- target 级 point-candidate 静态次序缓存与逐 state generation-mark 过滤；
- 同步 leaf batch 内复用 snapshot binding，同时保持公开 proof verifier 独立哈希；
- 同一只读 wavefront 内用强类型 graph binding 复用 leaf batch 快照哈希，并拒绝对象错配；
- 同一只读 wavefront 内让 c,d/Hamilton/end batches 复用一次完整 CSR 图验证；
- 同一不可变 HT scan 的 targets 间复用 graph/hash bindings，并保留逐目标内容哈希变更守卫；
- path-append 的 sparse parent 认证与增量规范链合并，以及保留的 dense proof 重放认证；
- leaf k-opt batch 的一次性 sparse path 认证与对象绑定，以及保留的 scalar/dense proof 重放；
- path-count matching catalog 与 3/4/5-opt reconnect templates 的线程安全不可变缓存；
- CPU/CUDA 共用的增量 leaf 精确成本矩阵路径，以及与 CPU scalar 逐字节一致的规范 proof 计数；
- CPU exact cost 的固定数组 task 验证、缓存端口对计划与 task 内 pair-cost 复用；
- 有容量与收益门禁的 CPU cost batch 精确距离表，以及超限时的 task-local 安全回退；
- 融合 leaf cost matrix 按 cursor slice 的生命周期受限只读 view，避免重复分配和复制；
- cursor deletion work 的固定 5 元位置数组与路径边精确成本缓存，并保留候选通用重算；
- 内部 CPU cost 的可增长输出 workspace 与 Debug 全覆盖哨兵，同时保留公开 owning vector API；
- CPU leaf batch 的完全相同 task row 精确去重、零展开 cursor 映射，以及逻辑/物理 row 分离计数；
- Hamilton reply 的批内 center 去重、每邻边 quick-filter 缓存与 CPU/CUDA 全数组差分；
- Hamilton/end reply 跨 batches/targets 的精确 CUDA 图驻留与增长 workspace 复用，以及逐批释放 A/B 开关；
- Hamilton/end reply 的 batch-local 精确任务去重、完整逻辑 CPU 展开与物理 CUDA 提交量诊断；
- leaf setup 的 proof/coverage/cursor 画像，以及同一批次只计算一次的不可变快照哈希；
- 8,192-cell 门槛以上按 task row 静态分片的有界 OpenMP CPU 精确成本矩阵；
- TSPLIB 最优 tour 的严格成本、节点置换、活动边完整性与规范哈希门禁；
- 带锁定来源 SHA-256 和本地精确复核的 pcb3038/rl5915/d15112 最优 tour 获取工具；
- CPU 单元测试、CUDA 差分测试入口和 pr299 集成脚本。
- TSPLIB 完全图构造、`kh-jq` 锁定 profile、完整 target sweep，以及单 GPU CUDA-JV/作者 KH-HS 固定点端到端复现实验。
- `fgpu-elim` 单命令链路：CUDA Main-Edge 几何筛选、MPFR 区间认证、native CUDA degree-subgradient、`__int128` LP-box 强制边证书、exhaustive CUDA JV、PDLP 排序的 HT wavefront，以及 `.edg/.fix/.nonpairs/.fgcert/.manifest` 五类输出；LP/JV 会跨不可变快照交错到固定点。
- FGPU V4 证书把量化 vertex dual 与快照哈希绑定；V5 对 HT sidecar 做有界 zlib 压缩并绑定原长/CRC32。在线提交和独立重放均重算完整数学条件。几何、LP 与 JV 同 epoch records 可并行复核，但提交次序、首错和最终哈希保持确定。
- `fgpu-elim resident` 的单卡全常驻主链：CUDA FP64 有向舍入 Geometry、device-resident degree-box PDLP、exhaustive JV 与 Quick-HS 固定点；CPU 只在搜索结束后做不反馈设备的精确安全审计。pcb442 clean-commit 七次端到端中位数为 9.23 秒、最终 3,008 条边，详见 [全常驻基准](docs/research/68_FGPU_单卡全常驻实现与端到端基准.md)。
- `resident --cpu-audit 0` 是全量性能实验路径：不回传逐边 trace、不生成证书、不做 CPU 精确重放，直接将 GPU 最终 mask 写成边集；manifest 会强制标记为 `gpu-raw`，防止与认证结果混淆。
- path matching coverage 已扩展到 `m=6`（3,840 outside、10,395 inside、4,989,600 bytes），固定生成器哈希为 `750842211d2a93e7`。

尚未完成的研究项（跨 target SoA continuation ready queue、generation cancellation、cuOpt 退化对偶稳定化和精确定价后边集导出）会显式安全回退，详见 [研究路线图](docs/research/05_Roadmap_and_Gates.md)。V3 的跨目标 leaf broker 在 d15112 32-target 上保持 `19,498 states/18 proofs`，五对 clean A/B 的单 GPU target execution 相对 CPU 为 `1.009x`，algorithm total 为 `1.001x`，process wall 为 `0.997x`，因此端到端只能判定为持平，`transposed` 继续保持 opt-in。这些结果不是论文 Table 7 的同协议对比，详见 [V3 跨目标 Leaf Broker](docs/research/64_V3_单GPU跨目标LeafBroker.md)。多 epoch 调度已可执行，但 `ht-epoch-limit` 只表示安全部分结果，不表示全图收敛。CPU 精确困难叶有 18 blocks 上限，CUDA 候选器上限为 13；任何超限或错误只返回 `unresolved`。HT 只提交完整 CPU 重放成功的 sidecar；旧 `cudaee lp-solve` 仍只输出数值结果。新的 `fgpu-elim --pdlp native` 只有在量化 multiplier 经完整 live-variable box bound 重算、强制目标边下界严格超过 incumbent，并写入 V4/V5 sidecar 后，才可授权 LP 删除。`run_fgpu_oneshot.sh` 默认拒绝 `*-limit/*-partial` 终止；只有显式 `CUDAEE_FGPU_ALLOW_PARTIAL=1` 才保留部分搜索作为调试产物。

## 快速开始

```bash
./tools/bootstrap.sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug

# 有空闲 GPU 时
cmake --preset cuda-release
cmake --build --preset cuda-release
./tools/run_pr299.sh

# M5 JV 中大型基准；可用 CUDAEE_BENCHMARK_TOUR 增加最优 tour 门禁
CUDAEE_BENCHMARK_GPU=1 tools/run_jv_benchmark.sh pcb3038 5

# M5 有界全图 HT CPU/CUDA pilot；同一切片、预算和独立 proof 重放
./tools/fetch_m5_opt_tours.sh
CUDAEE_BENCHMARK_GPU=1 \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/pcb3038.opt.tour \
tools/run_ht_scan_benchmark.sh pcb3038 8

# 两张空闲卡上的 HT target 静态切片 A/B；参数 0/1 是物理 GPU 索引
CUDAEE_CUDA_PRESET=cuda-sm86-release \
CUDAEE_HT_MAX_TARGETS=32 \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/d15112.opt.tour \
tools/run_ht_target_multigpu_ab.sh d15112 0 1 7

# V3 wavefront CPU/hybrid 七对 A/B；参数 2 是当前主机的物理 GPU 索引
CUDAEE_CUDA_PRESET=cuda-sm86-release \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/d15112.opt.tour \
tools/run_v3_single_gpu_ab.sh d15112 2 7

# V3 transposed 最佳 CPU vs 单 GPU leaf broker 五对 A/B
CUDAEE_CUDA_PRESET=cuda-sm86-release \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/d15112.opt.tour \
tools/run_v3_transposed_single_gpu_ab.sh d15112 2 5

# 构建受限 Concorde overlay，并验证 cuOpt→完整图精确定价握手
./tools/bootstrap_concorde.sh
./tools/run_concorde_cuopt_epoch.sh
./tools/run_concorde_cuopt_epoch.sh --tamper-model-hash

# cuOpt 冷启动 + 同会话稳定身份 warm start
./tools/run_cuopt_smoke.sh

# pcb442：完全图 -> 单 GPU JV -> 作者 KH-HS/JV 固定点（自动选择空闲 GPU）
./tools/run_complete_kh_jq_e2e.sh pcb442

# 新 FGPU one-shot：参数 2 是物理 GPU ordinal；省略时自动选择本机空闲卡
CUDAEE_FGPU_ENABLE_HT=0 tools/run_fgpu_oneshot.sh pcb442 2
CUDAEE_FGPU_ENABLE_HT=0 tools/run_fgpu_oneshot.sh pr1002 2

# 单卡全常驻主链与独立证书审计；默认 7 次，参数 1 是物理 GPU ordinal
CUDAEE_ALLOW_BUSY_GPU=1 tools/run_fgpu_resident.sh pcb442 1 7

# 全图 GPU raw 基准：无逐边 trace、CPU audit 或证书
CUDAEE_FGPU_RESIDENT_CPU_AUDIT=0 tools/run_fgpu_resident.sh pcb3038 1 7
```

CLI：

```bash
build/cuda-release/cudaee gpu-eliminate \
  --tsp third_party/ElimTSP/data/pr299.tsp \
  --edges third_party/ElimTSP/data/pr299.edg \
  --output artifacts/pr299.filtered.edg \
  --proof artifacts/pr299.proof --backend auto

build/cuda-release/cudaee verify \
  --tsp third_party/ElimTSP/data/pr299.tsp \
  --edges third_party/ElimTSP/data/pr299.edg \
  --proof artifacts/pr299.proof

build/cpu-release/cudaee tour-check \
  --tsp ../references/tensoraco/TSPLIB/pcb3038.tsp.gz \
  --edges artifacts/pcb3038.filtered.edg \
  --tour artifacts/pcb3038.opt.tour --expected-cost 137694

build/cuda-release/cudaee path-table \
  --paths 5 --backend auto \
  --output artifacts/path-table-m5.manifest

# 生成递归 HT 全局证书（只证明，不修改边集），随后独立重放
build/cuda-release/cudaee ht-prove \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg --u 2 --v 4 \
  --proof artifacts/recursive-point.ht-proof --backend auto \
  --leaf-backend auto \
  --scheduler wavefront --reply-backend auto --reply-frontier-batch-states 256 \
  --leaf-frontier-batch-states 256 \
  --cuda-min-cost-cells 128 \
  --path-append-backend auto \
  --propagation-backend auto --propagation-blocks 0 \
  --max-depth 1 --max-k 3 --max-deletion-sets 1
build/cuda-release/cudaee ht-verify \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --proof artifacts/recursive-point.ht-proof

# 把一个或多个同快照 sidecar 整批复核后提交，并用自包含 V2 再次独立重放
build/cuda-release/cudaee ht-commit \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --output artifacts/recursive-point.epoch.edg \
  --proof artifacts/recursive-point.epoch.proof \
  --ht-proof artifacts/recursive-point.ht-proof
build/cpu-release/cudaee verify \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --proof artifacts/recursive-point.epoch.proof

# JV 固定点与有界 HT sweep 交替；输出一个可由同一 verify 命令重放的联合证明
build/cpu-release/cudaee local-eliminate \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --output artifacts/recursive-point.local.edg \
  --proof artifacts/recursive-point.local.proof \
  --report artifacts/recursive-point.local.report \
  --backend cpu --max-jv-rounds 100 \
  --max-ht-epochs 1 --max-targets 1
build/cpu-release/cudaee verify \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --proof artifacts/recursive-point.local.proof

# 多 GPU 时改用 CUDA binary/backend，并为 local-eliminate 增加 --target-devices 0,1
```

所有命令会拒绝把输出写到仓库之外。
