# cudaEdgeElimination

面向对称 TSP 的可验证 GPU 局部边消元研究实现。首期把 GPU 作为候选生成器，并由 CPU 在不可变图快照上逐条复核；LP sidecar 使用 cuOpt 求近似解，但只有独立的精确定价/定点证书可以授权删除。

当前实现范围：

- TSPLIB `EUC_2D` / `CEIL_2D` 与 Concorde 稀疏边文件读取；
- Jonker–Volgenant（JV）快速消元的 CPU 基线与 CUDA 候选生成；
- epoch 快照、CPU 复核、最小度保护、确定性提交和 `proof-v1` 证明日志；
- `lp-epoch-v1` CSR 模型、cuOpt C API 动态 sidecar、残差与精确定点下界；
- CPU 单元测试、CUDA 差分测试入口和 pr299 集成脚本。

尚未授权生产删除的研究项（HS path-system 全量 GPU 化、Concorde 完整图精确定价注入）会显式安全回退，详见 [研究路线图](docs/research/05_Roadmap_and_Gates.md)。

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
```

所有命令会拒绝把输出写到仓库之外。
