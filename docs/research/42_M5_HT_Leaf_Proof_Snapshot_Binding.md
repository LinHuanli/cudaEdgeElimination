# M5 HT leaf proof 批内快照绑定复用

## 1. 根因与安全边界

`ProvePathSystemsByKOpt` 已在 batch 初始化时计算一次 `graph.ContentHash()`，并把该值写入所有 leaf proof；但生成后的每个成功 proof 又调用公开 `VerifyPathSystemKOptProof`，逐个重新扫描整图。该重复工作在大实例上主导 leaf verify。

提交 `0530ff4` 提取 `VerifyPathSystemKOptProofBoundToSnapshot`：

- 同一同步 batch 的内部复核复用 batch 入口哈希，但仍逐 proof 检查 path hash、兼容表、source 顺序、coverage 和完整 witness；
- `graph` 在 batch 生命周期内只读，proof 绑定值和实际 witness 计算使用同一对象；
- 公开 verifier 仍在每次调用时自行执行 `graph.ContentHash()`；HT 成功 proof 的最终 verifier、即时 scan verifier、epoch verifier 和独立重放均未改变。

因此该优化只删除生成器内部的重复快照扫描，不复用证明结论，也不降低任何删除授权边界。

## 2. clean-commit 三实例结果

正式 runs 均绑定 clean commit `0530ff4ab07f85e56edc15fd76e31c91c91323bf`、物理 GPU 1、8 个 CPU cost threads 和锁定的公开最优 tour：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T234134Z-2683155`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T234149Z-2683784`；
- d15112：`artifacts/d15112-ht-scan-20260901T234157Z-2684362`。

相对前一 clean commit `ea85ffa`，CPU fused 单变量结果如下：

| 实例 | leaf proof verify：基线 → 复用（ms） | verify 加速 | leaf 加速 | work graph 加速 | search 加速 | wall 加速 |
|---|---:|---:|---:|---:|---:|---:|
| pcb3038 | 42.958 → 4.836 | 8.882× | 1.041× | 1.037× | 1.037× | 1.036× |
| rl5915 | 125.568 → 7.138 | 17.591× | 1.744× | 1.391× | 1.355× | 1.282× |
| d15112 | 525.686 → 19.152 | 27.449× | 3.191× | 1.733× | 1.618× | 1.457× |

收益随 graph hash 成本和成功 leaf 数增长；pcb3038 的主要成本仍在 51,309,996 个精确 cost cells，因此端到端收益较小。

## 3. 门禁结果

- CPU Debug/Release 为 17/17，CUDA Release 为 20/20；
- CUDA Debug HT 单元在 compute-sanitizer 下为 0 error；
- 三实例五路规范工作计数、活动边、工作签名和去除计时行后的 proof 逐字节一致；
- JV 固定点、V2 重放、最终活动边与受保护最优 tour 全部通过。

## 4. 下一切片

`leaf_proof_initialize_ms` 仍为 `26.605/8.678/29.163 ms`，主要来自每个 leaf batch 再次计算同一 wavefront snapshot hash。下一单变量实验把 `ProveEdgeByWavefrontHt` 已计算且整个调用只读的 snapshot hash 显式传给内部 leaf batch API；公开 k-opt API 仍自行计算哈希，错误绑定最终也必须被独立 HT verifier 拒绝。
