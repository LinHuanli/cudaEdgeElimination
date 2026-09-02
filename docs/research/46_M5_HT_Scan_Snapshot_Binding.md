# M5 HT scan 跨目标快照绑定复用

## 1. 根因与只读作用域

`ht-scan` 在选择目标后，会先在同一个不可变 `GraphSnapshot` 上完成整个 target 切片的搜索，最后才整批复核并原子提交。此前每个 target 都经公开 wavefront 入口重新执行两项与目标无关的工作：

- `KOptSnapshotBinding` 重新计算完整图内容哈希；
- `HtGraphValidationBinding` 重新验证坐标、CSR 边界、邻接严格排序和无向对称性。

提交 `649f3f4` 把这两个强类型 binding 上移到 `RunHtScanEpoch`，并通过内部 bound wavefront API 在同一同步 scan 的 targets 间复用。优化边界如下：

- 两个 binding 仍只可由实际 graph 对象经完整哈希/验证构造；
- bound wavefront 同时检查两个 binding 的对象身份，内容相同但对象不同的 graph copy 也会被拒绝；
- 公开 `ProveEdgeByWavefrontHt` 仍为每次独立调用自行构造 binding；
- scan 在每个 target 返回后仍重新执行 `ContentHash()`，逐目标检查图没有被搜索过程修改；
- 每个 target 仍独立生成工作图、证明和 canonical work counters，成功证明仍立即由公开 CPU verifier 重放；
- epoch commit、V2 verifier 和所有公开 proof verifier 完全不使用这些生成器 binding。

该改动只复用不可变输入的认证结果，不复用 target 相关候选、状态真值或删除结论；GPU 仍只产生候选，CPU 证明链仍是唯一授权边界。

## 2. 测试与提交门禁

新增单元测试比较公开/bound wavefront 的状态、序列化 proof、move 数和 Hamilton reply 数，并验证 equal graph copy 的对象错配会抛出异常。scan 集成测试与 V2 重放继续覆盖实际 bound 路径。

提交前完成：

- CPU Debug：17/17 CTest；
- CPU Release：17/17 CTest；
- CUDA Release：20/20 CTest，包含 CPU/GPU 差分；
- CUDA Debug Hamilton–Tutte 单元：compute-sanitizer memcheck 0 errors。

## 3. clean-commit 正式协议与等价性

正式 runs 均绑定 clean commit `649f3f4`、物理 GPU 1、8 个 CPU cost threads，以及上一基线中锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T003040Z-2716346`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T003054Z-2716980`；
- d15112：`artifacts/d15112-ht-scan-20260902T003101Z-2717527`。

相对 clean baseline `fb772f8`，对每个实例的 CPU、CPU-fused、CUDA、hybrid 和 fused 五路逐项比较：

- 最终 `.edg` 逐字节一致；
- `.work-signature` 逐字节一致；
- 去除运行时 `metrics` 行后的 V2 proof 逐字节一致；
- JV 固定点边集逐字节一致，去除 `metrics` 后的 JV proof 逐字节一致；
- 受保护最优 tour 文件一致，且每路运行的 tour-check 和独立 V2 replay 均成功。

三实例共 54 项跨提交精确比较全部通过。

## 4. 三实例性能结果

CPU-fused 的单变量结果如下；加速均为 `baseline / 649f3f4`：

| 实例 | candidate：基线 → scan binding（ms） | candidate 加速 | work graph 加速 | search 加速 | total 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 4.179 → 2.496 | 1.674× | 1.004× | 1.007× | 1.006× | 1.004× |
| rl5915 | 13.953 → 4.413 | 3.162× | 1.057× | 1.135× | 1.110× | 1.121× |
| d15112 | 76.568 → 10.410 | 7.355× | 1.001× | 1.246× | 1.154× | 1.123× |

candidate 是各 target wavefront 内的累计计时：基线包含每 target binding 构造，新版本不包含 scan 入口的一次性构造。因此 candidate 降幅只说明重复工作已被消除，不能单独当作端到端收益。`total_ms` 从 binding 构造之前开始，CLI wall 也完整包含它们；这两个指标分别从 `1239.994/241.968/624.270 ms` 降至 `1232.243/218.060/540.915 ms`，以及从 `1269.018/292.990/775.677 ms` 降至 `1263.935/261.416/690.825 ms`，没有隐藏一次性成本。

pcb3038 的计算量由 51,309,996 个 leaf cost cells 主导，跨目标绑定只占极小比例；rl5915 和 d15112 的图更大而本切片 leaf 工作较少，因此端到端收益更清晰。

## 5. 新画像与下一切片

d15112 CPU-fused search 已降至 `373.588 ms`。work graph 为 `328.486 ms`，其中 leaf `206.669 ms` 仍是最大阶段；leaf 内 cursor construct `69.004 ms`、cost evaluate `36.996 ms`、cursor consume `36.282 ms`、apply `24.720 ms` 和 proof verify `19.128 ms`。path append 为 `43.253 ms`，root child normalize 为 `28.309 ms`。

下一单变量实验先审计 leaf cursor 构造是否反复生成只依赖规范 path/forced-outside 的不可变索引，并为可复用部分建立内容完整、对象作用域明确的缓存。任何 fast path 都必须保持公开构造路径、完整 proof 字节、规范工作计数和三实例五路结果不变；若可复用比例不足，则保留实现并转向 CPU exact cost rows 的任务粒度与内存布局。
