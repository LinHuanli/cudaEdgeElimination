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
- 批量 CUDA k-opt 精确成本候选矩阵，以及 GPU 未命中后的 CPU completeness fallback；
- 收缩 forced outside matching 的 CPU 精确 Held–Karp 子集 DP 困难叶回退；
- 浅层 Hamilton–Tutte `c,d` AND–OR 根证明与 CPU 复核的 CUDA 候选筛选；
- CPU 递归 Hamilton–Tutte point/end moves、continuation arena 与全局 `recursive-ht-proof-v1`；
- 主机 BFS 工作图、cooperative multi-block CUDA continuation 传播、跨父状态 Hamilton/end reply count/write、point/end path-append、规范 child edge SoA、增量 k-opt leaf cost block 融合、GPU 驻留缓存与 128-cell CPU long-tail，并由 CPU 完整差分复核；
- `ht-prove` sidecar 的整批 CPU 重放、不可变快照绑定、规范度数门禁和 `ht-commit` 原子删边；
- `ht-scan` 的确定性有界目标切片、逐目标 wavefront、CPU 双重复核与 V2 原子提交；
- HT scan V2 阶段计时，以及用 `--leaf-backend` 将 CUDA leaf cost 与 CPU 候选器解耦的混合路径；
- 可显式启用、默认关闭的 `--fuse-leaf-buckets` 调度实验及 V3 leaf batch 计数；
- HT scan V4 leaf 子阶段计时及 V5 四路 benchmark summary；
- path-count matching catalog 与 3/4/5-opt reconnect templates 的线程安全不可变缓存；
- TSPLIB 最优 tour 的严格成本、节点置换、活动边完整性与规范哈希门禁；
- CPU 单元测试、CUDA 差分测试入口和 pr299 集成脚本。

尚未完成的研究项（跨目标 HT 融合和多 epoch 调度、M5 后续调优与多 GPU、cuOpt 退化对偶稳定化和精确定价后边集导出）会显式安全回退，详见 [研究路线图](docs/research/05_Roadmap_and_Gates.md)。精确困难叶有 18 个 block 的硬上限，超限只返回 `unresolved`。HT 只提交完整 CPU 重放成功的 sidecar；`lp-solve` 本身也永不删除边，只有 Concorde 桥接路径经过完整图精确定价后才产生下界授权。

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
CUDAEE_BENCHMARK_GPU=1 \
CUDAEE_BENCHMARK_TOUR=artifacts/lkh-tours/pcb3038.tour \
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
```

所有命令会拒绝把输出写到仓库之外。
