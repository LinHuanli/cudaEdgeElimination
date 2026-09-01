# 仓库与 Git 工作流

## 目录职责

```text
include/cuda_edge_elimination/  公共 C++ 接口
src/common/                     数据、距离、格式与哈希
src/cpu/                        独立 CPU 规范实现和验证器
src/cuda/                       只生成候选的 CUDA 实现
src/lp/                         LP epoch、cuOpt sidecar 与精确模型证书
src/cli/                        cudaee 命令行
integrations/concorde/          受限源码之外的补丁与导出模块
tests/                          unit/differential/integration/numeric
tools/                          环境、构建和实验入口
benchmarks/                     基准驱动，不放大结果
configs/                        可提交的参数配置
data/manifests/                 可提交的小型数据清单
third_party/ElimTSP/            固定 Git 子模块
```

## 分支与提交

`main` 始终可构建。功能分支命名 `feature/<topic>`、修复分支 `fix/<topic>`、实验分支 `experiment/<topic>`。提交按“文档/基础设施、CPU 规范实现、CUDA、LP、集成”拆分，避免把生成数据与源代码混合。

提交信息使用 Conventional Commits，例如 `feat(cuda): 添加 JV 候选 kernel`。代码评审重点依次为证明不变量、错误处理、测试覆盖和性能。

## CI 与本地 GPU 门禁

GitHub Actions 只运行 CPU 配置、格式检查和静态格式/schema 测试，不依赖私有数据或受限求解器。GPU/cuOpt/Concorde 测试在本机运行并把摘要附到提交或实验记录；大产物不推送 Git。

## 远端与发布

远端为 `https://github.com/LinHuanli/cudaEdgeElimination.git`。推送前检查 `git status`、子模块状态和大文件；未确定本项目许可证前不创建 release。受限依赖始终由脚本在使用者环境重建。
