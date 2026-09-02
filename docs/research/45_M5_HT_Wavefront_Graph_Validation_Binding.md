# M5 HT wavefront 图验证绑定复用

## 1. 根因与安全边界

Hamilton reply 和 end reply 的公开 batch API 都会先完整验证 HT graph：检查整数距离前提、CSR offsets、邻接严格排序、边界和无向对称性。该契约适合独立调用，但一个 wavefront 的所有 c,d、Hamilton 和 end batches 共享同一个只读 `GraphSnapshot`。d15112 的 15 个 Hamilton batches 因而累计花费 `121.818 ms` 重复扫描同一 CSR，end reply 又重复执行相同验证。

提交 `fb772f8` 引入内部强类型 `HtGraphValidationBinding`：

- 构造 binding 时完整执行一次 `ValidateHtGraph`，验证失败不能获得 binding；
- binding 保存 graph 对象身份，只能用于同一同步只读作用域；
- wavefront 的 c,d 候选、Hamilton reply 和 end reply 共用同一 binding；
- 每个 bound API 都检查对象身份，内容相同但对象不同的 graph copy 也会被拒绝；
- 公开 `EvaluateHtCdCandidates`、`EvaluateHtHamiltonReplies` 和 `EvaluateHtEndReplies` 仍在每次调用时完整验证 graph；
- CPU 仍完整生成 canonical replies，CUDA 路径仍逐数组与 CPU 结果比较，proof verifier 与 epoch 重放完全不使用 binding。

该优化只复用输入结构验证，不缓存候选、reply 或证明结论，也不改变任何删除授权边界。

## 2. 单元与提交门禁

新增单元测试逐项比较公开/bound API 的：

- c,d candidates、backend 和 CPU verified 标志；
- Hamilton offsets、完整 replies、unique centers 和 neighbor-pair 计数；
- end offsets 和完整 replies；
- graph-copy 对象错配拒绝。

提交前 CPU Debug/Release 各 17/17、CUDA Release 20/20；CPU/GPU 差分通过，完整 Hamilton–Tutte CUDA Debug 单元在 compute-sanitizer memcheck 下为 0 errors。

## 3. clean-commit 三实例结果

正式 runs 均绑定 clean commit `fb772f8`、物理 GPU 1、8 个 CPU cost threads 和锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260902T001937Z-2709222`；
- rl5915：`artifacts/rl5915-ht-scan-20260902T001947Z-2709782`；
- d15112：`artifacts/d15112-ht-scan-20260902T001951Z-2709212`。

相对 clean baseline `236022c`，CPU-fused 单变量结果如下：

| 实例 | Hamilton validation：基线 → binding（ms） | validation 加速 | Hamilton 加速 | end reply 加速 | work graph 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pcb3038 | 16.431 → 0.054 | 305.600× | 6.219× | 55.916× | 1.017× | 1.017× | 1.017× |
| rl5915 | 23.072 → 0.040 | 583.206× | 3.165× | 72.810× | 1.173× | 1.146× | 1.087× |
| d15112 | 121.818 → 0.036 | 3388.723× | 5.996× | 206.300× | 1.585× | 1.411× | 1.249× |

binding 构造已纳入 candidate 阶段；candidate 为 `4.116→4.179/13.678→13.953/75.435→76.568 ms`，三实例均处于小幅波动范围，没有把验证成本藏到未画像时间。三实例五路活动边、工作签名和去除 `metrics` 行后的 proof 与 baseline 逐字节一致；JV 固定点、V2 重放和三份受保护最优 tour 全部通过。

## 4. 新画像与下一切片

d15112 CPU-fused search 已降至 `465.462 ms`，work graph 为 `328.958 ms`；其中 leaf `206.440 ms` 重新成为最大子阶段。candidate 仍为 `76.568 ms`，主要包含每个 target 都重新构造一次相同 graph validation binding；wavefront 入口的 k-opt binding 也会为每个 target 重算相同 snapshot hash。

`ht-scan` 在提交前本来就对同一不可变 snapshot 顺序搜索全部 target。下一单变量实验应在 scan 入口构造一次 graph-validation 和 snapshot-hash binding，再以强类型内部 wavefront API 传给各 target；公开 wavefront API 继续自行构造。门禁必须包含对象错配拒绝、公开/bound wavefront proof 字节一致、三实例五路工作签名不变，以及提交前后的独立 V2 重放。
