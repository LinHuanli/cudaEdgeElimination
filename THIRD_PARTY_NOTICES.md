# Third-party notices

本仓库的自有源代码尚未声明统一许可证。下列依赖各自遵守原许可证：

- `third_party/ElimTSP`：MIT，固定为提交 `d7bacf0d79ddc1f6f9e77b027df16d446458e58c`。
- NVIDIA cuOpt：运行时通过项目内 `.venv` 安装，Apache-2.0 组件及 NVIDIA 包条款；不提交 wheel 或二进制。
- Concorde 03.12.19：仅学术研究用途；从仓库外 `references/concorde_code` 复制到忽略的 `.deps/` 构建，不提交源码或二进制。
- QSopt：仅学术研究用途；下载到忽略的 `.deps/`，不提交静态库或头文件。
- LKH-3.0.13：受限研究用途，仅作为外部只读参考，不进入本仓库。

实验发布前必须重新审计数据与二进制的分发许可。
