# cudaEdgeElimination

面向对称 TSP 的可验证 GPU 局部边消元研究实现。首期把 GPU 作为候选生成器，并由 CPU 在不可变图快照上逐条复核；LP sidecar 使用 cuOpt 求近似解，但只有独立的精确定价/定点证书可以授权删除。

当前实现范围：

- TSPLIB `EUC_2D` / `CEIL_2D` 与 Concorde 稀疏边文件读取；
- Jonker–Volgenant（JV）快速消元的 CPU 基线与 CUDA 候选生成；
- JV 跨 epoch 精确静态键、动态 CSR edge-id 刷新与增长型 CUDA 驻留 workspace；
- epoch 快照、CPU 复核、最小度保护、确定性提交，以及兼容 JV V1 的自包含 HT V2 证明日志；
- `lp-epoch-v1` CSR 模型、cuOpt C API 动态 sidecar、残差与精确定点下界；
- Concorde 受限 overlay、列—边映射、`CCbigguy` 对偶注入与完整图精确定价证书；
- 路径系统规范化、`m<=5` CPU 生成/CUDA 查询兼容表与 `m=6,7` CPU 回退；
- proper 3/4/5-opt CPU 叶 witness、inside coverage 与 `path-kopt-proof-v1` 重放；
- 批量 CUDA k-opt 精确成本候选矩阵、逐 cell CPU 整数认证，以及仅对严格改善模板执行的完整 witness 重建；
- 收缩 forced outside matching 的 CPU 精确 Held–Karp 子集 DP 困难叶回退；
- 浅层 Hamilton–Tutte `c,d` AND–OR 根证明与 CPU 复核的 CUDA 候选筛选；
- CPU 递归 Hamilton–Tutte point/end moves、continuation arena 与全局 `recursive-ht-proof-v1`；
- 主机 BFS 工作图、cooperative multi-block CUDA continuation 传播、跨父状态 Hamilton/end reply count/write、point/end path-append、规范 child edge SoA、增量 k-opt leaf cost block 融合、GPU 驻留缓存与 128-cell CPU long-tail，并由 CPU 完整差分复核；
- `ht-prove` sidecar 的整批 CPU 重放、不可变快照绑定、规范度数门禁和 `ht-commit` 原子删边；
- `ht-scan` 的确定性有界目标切片、逐目标 wavefront、CPU 双重复核与 V2 原子提交；
- `local-eliminate` 的 JV 固定点、HT 无提交 sweep 推进、提交后目标重排，以及单一可重放 V2 证明组合；
- HT scan V2 阶段计时，以及用 `--leaf-backend` 将 CUDA leaf cost 与 CPU 候选器解耦的混合路径；
- CPU leaf CLI 默认开启、auto/CUDA 默认关闭且可显式 0/1 覆盖的 `--fuse-leaf-buckets` 调度；
- HT scan V13 leaf/setup/cost/reply/path/root/point-candidate 子阶段计时及 V16 五路 benchmark summary；
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
- leaf setup 的 proof/coverage/cursor 画像，以及同一批次只计算一次的不可变快照哈希；
- 8,192-cell 门槛以上按 task row 静态分片的有界 OpenMP CPU 精确成本矩阵；
- TSPLIB 最优 tour 的严格成本、节点置换、活动边完整性与规范哈希门禁；
- 带锁定来源 SHA-256 和本地精确复核的 pcb3038/rl5915/d15112 最优 tour 获取工具；
- CPU 单元测试、CUDA 差分测试入口和 pr299 集成脚本。

尚未完成的研究项（跨目标 HT 融合、M5 多 GPU、cuOpt 退化对偶稳定化和精确定价后边集导出）会显式安全回退，详见 [研究路线图](docs/research/05_Roadmap_and_Gates.md)。活动 edge-id 紧凑 launch 已在 d15112 上评测并因端到端回退而撤销。多 epoch 调度已可执行，但 `ht-epoch-limit` 只表示安全部分结果，不表示全图收敛。精确困难叶有 18 个 block 的硬上限，超限只返回 `unresolved`。HT 只提交完整 CPU 重放成功的 sidecar；`lp-solve` 本身也永不删除边，只有 Concorde 桥接路径经过完整图精确定价后才产生下界授权。

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

# 构建受限 Concorde overlay，并验证 cuOpt→完整图精确定价握手
./tools/bootstrap_concorde.sh
./tools/run_concorde_cuopt_epoch.sh
./tools/run_concorde_cuopt_epoch.sh --tamper-model-hash
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
```

所有命令会拒绝把输出写到仓库之外。
