# M5 HT CPU cost 固定验证与重连计划

## 1. 根因

leaf exact cost matrix 的每个 task 只含 3–5 条删除边，但入口验证此前为每个 task 构造 `std::set<NodeEdge>`。pcb3038 的 8-target CPU-fused run 消费 727,635 个 cost rows，因此仅小集合查重就会触发数百万次堆节点分配。

随后每个矩阵 cell 还重复扫描 `EndpointMatching` 的 `2k` 个端口、重建规范 added edge，并重新检查该 port pair 是否等于删除边。距离本身已有 task 内惰性缓存，但端口配对、规范边和合法性仍按 51,309,996 个 cells 重复解释。

提交 `273ac9d` 做了三个同一层级的固定结构改写：

- task 验证改用 5 元固定数组，按原输入顺序线性查重；
- 每个 `k=3/4/5` 的规范 reconnect templates 一次性编译为恰好 k 个扁平端口对，并在线程安全静态初始化后只读复用；
- 每个 task 第一次访问 port pair 时缓存规范边、删除边冲突状态和精确整数距离，后续 templates 直接读取。

矩阵仍按 `[task][canonical template]` 排列，template 枚举、无效值 `INT64_MAX`、整数距离与溢出规则均不变。CUDA kernel 没有修改；CUDA 返回后仍逐 cell 与完整 CPU 矩阵比较。严格改善 cell 仍由通用路径重建完整 witness，并由独立 verifier 复核后才可能进入证明。

## 2. 正确性门禁

既有 k-opt 测试覆盖：

- proper 3/4/5-opt 的 `4/25/208` 个规范 templates 与固定 generator hash；
- CPU scalar、CPU matrix、auto/CUDA matrix 的 proof 字节差分；
- 随机图、多个路径布局、预算中断和 batch size；
- CPU/CUDA 全矩阵逐 cell 一致、候选重建与完整 proof replay；
- HT scan 即时复核、V2 原子提交和独立重放。

提交前 CPU Debug/Release 各 17/17、CUDA Release 20/20；k-opt 与 Hamilton–Tutte CUDA Debug 单元在 compute-sanitizer memcheck 下均为 0 errors。

## 3. clean-commit 正式协议

正式 runs 均绑定 clean commit `273ac9d`、物理 GPU 1、8 个 CPU cost threads 和锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T005535Z-2732365`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T005549Z-2732981`；
- d15112：`artifacts/d15112-ht-scan-20260902T005559Z-2733557`。

相对 clean baseline `8c19740`，三实例五路最终边集、工作签名、去 `metrics` 的 V2 proof、JV 固定点边集/proof 和受保护 tour 全部逐字节一致；每份 proof 均由独立进程重放，所有 tour-check 成功。共 54 项跨提交精确比较无差异。

## 4. 三实例性能结果

CPU-fused 单变量结果如下；加速均为 `8c19740 / 273ac9d`：

| 实例 | cost evaluate：基线 → 固定计划（ms） | cost 加速 | CPU certify 加速 | leaf 加速 | search 加速 | total 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pcb3038 | 631.746 → 418.325 | 1.510× | 1.388× | 1.284× | 1.274× | 1.271× | 1.263× |
| rl5915 | 39.753 → 25.160 | 1.580× | 1.450× | 1.169× | 1.131× | 1.110× | 1.083× |
| d15112 | 34.454 → 24.083 | 1.431× | 1.317× | 1.071× | 1.035× | 1.021× | 1.014× |

pcb3038 CPU-fused work graph 从 `1154.246 ms` 降至 `905.111 ms`，search 从 `1163.064 ms` 降至 `913.156 ms`，CLI wall 从 `1200.307 ms` 降至 `950.567 ms`。d15112 受 host residual、path/reply 和证明复核的固定成本影响更大，因此 30.10% 的 cost 阶段降幅只转化为 3.40% search 降幅。

## 5. 下一切片

pcb3038 CPU-fused leaf 中 cost evaluate 仍为 `418.325 ms`，其次是 cursor prepare `174.152 ms`、cursor consume `121.491 ms`、setup `56.654 ms` 和 scatter `41.128 ms`。下一实验先计量同一融合 cost batch 中实际 node pair 的重复率，再决定是否构建有上限的 batch-local distance cache；没有足够重复率时不扩大内存占用，改为优化 cursor prepare 的组合展开与 task 复制。

任何跨 task 缓存必须以完整整数坐标快照和规范 node pair 为键，容量超限时安全回退为当前 scorer；CPU/CUDA 逐 cell 认证、规范 proof 计数和独立 witness verifier 均不得改变。
