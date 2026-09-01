# cudaEdgeElimination

面向对称 TSP 的可验证 GPU 局部边消元研究实现。首期把 GPU 作为候选生成器，并由 CPU 在不可变图快照上逐条复核；LP sidecar 使用 cuOpt 求近似解，但只有独立的精确定价/定点证书可以授权删除。

当前实现范围：

- TSPLIB `EUC_2D` / `CEIL_2D` 与 Concorde 稀疏边文件读取；
- Jonker–Volgenant（JV）快速消元的 CPU 基线与 CUDA 候选生成；
- epoch 快照、CPU 复核、最小度保护、确定性提交和 `proof-v1` 证明日志；
- `lp-epoch-v1` CSR 模型、cuOpt C API 动态 sidecar、残差与精确定点下界；
- Concorde 受限 overlay、列—边映射、`CCbigguy` 对偶注入与完整图精确定价证书；
- 路径系统规范化、`m<=5` CPU 生成/CUDA 查询兼容表与 `m=6,7` CPU 回退；
- proper 3/4/5-opt CPU 叶 witness、inside coverage 与 `path-kopt-proof-v1` 重放；
- 批量 CUDA k-opt 精确成本候选矩阵，以及 GPU 未命中后的 CPU completeness fallback；
- 收缩 forced outside matching 的 CPU 精确 Held–Karp 子集 DP 困难叶回退；
- 浅层 Hamilton–Tutte `c,d` AND–OR 根证明与 CPU 复核的 CUDA 候选筛选；
- CPU 递归 Hamilton–Tutte point/end moves、continuation arena 与全局 `recursive-ht-proof-v1`；
- 主机 BFS 工作图、CUDA 反向层次 AND–OR 传播及逐状态 CPU 差分复核；
- CPU 单元测试、CUDA 差分测试入口和 pr299 集成脚本。

尚未完成的研究项（GPU 状态生成/批量叶/persistent continuation 与 epoch commit、cuOpt 退化对偶稳定化与精确定价后边集导出）会显式安全回退，详见 [研究路线图](docs/research/05_Roadmap_and_Gates.md)。精确困难叶有 18 个 block 的硬上限，超限只返回 `unresolved`。递归 HT proof 尚未接入删边；`lp-solve` 本身也永不删除边，只有 Concorde 桥接路径经过完整图精确定价后才产生下界授权。

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

build/cuda-release/cudaee path-table \
  --paths 5 --backend auto \
  --output artifacts/path-table-m5.manifest

# 生成递归 HT 全局证书（只证明，不修改边集），随后独立重放
build/cuda-release/cudaee ht-prove \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg --u 2 --v 4 \
  --proof artifacts/recursive-point.ht-proof --backend auto \
  --scheduler wavefront --propagation-backend auto \
  --max-depth 1 --max-k 3 --max-deletion-sets 1
build/cuda-release/cudaee ht-verify \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --proof artifacts/recursive-point.ht-proof
```

所有命令会拒绝把输出写到仓库之外。
