# M5 HT 多实例融合门禁与 CPU 默认策略

## 1. 最优 tour 见证准备

[pcb3038 CPU leaf 融合实验](36_M5_HT_CPU_Leaf_Bucket_Fusion.md)只能证明单实例收益，不足以改变默认调度。本切片先补齐 rl5915 和 d15112 的安全见证，再运行完整 V12 五路门禁。

`configs/m5_opt_tours.tsv` 锁定 LKH 作者公开的 [TSPLIB tour 目录](http://akira.ruc.dk/~keld/research/LKH/LKH-1.2/TOURS/)、生成文件名和 SHA-256。`tools/fetch_m5_opt_tours.sh` 使用仓库内临时目录下载，哈希不匹配立即失败；通过哈希后仍要用本项目 `tour-check` 重新检查节点置换、TSPLIB 整数距离与 ElimTSP 稀疏图的全部 tour 边。下载物只进入已忽略的 `artifacts/lkh-tours/`，不修改外部数据树。

| 实例 | 锁定 SHA-256 | 精确成本 | 规范 tour hash | 原始稀疏图缺边 |
|---|---|---:|---|---:|
| pcb3038 | `74af9149b5bb904b441c127c5719e51dd51aec1fdda8b823037a61be0cf260a9` | 137,694 | `3d014f3fdfa4cd64` | 0 |
| rl5915 | `16047b1e1251a66e3fffebaf22efab33b16a7458094523adb171826b7c182592` | 565,530 | `bbd21e7d4a157b15` | 0 |
| d15112 | `cb189d1cd34a5d0d39bc8caebc0b064793047dcf8c2a71c1a599ac497c1e7ae8` | 1,573,084 | `4495654253f2318e` | 0 |

pcb3038 的公开 tour 与早先 LKH 独立找到的哈希 `ca0238497c090a3c` tour 是两个不同的同价最优解；已对正式 HT 输出额外检查公开 tour，结果同样为 0 缺边。

## 2. 三实例 clean-commit 五路结果

rl5915/d15112 先分别通过 1-target 冒烟：

- `artifacts/rl5915-ht-scan-20260901T223429Z-2633490`；
- `artifacts/d15112-ht-scan-20260901T223443Z-2634026`。

正式 8-target runs 为：

- pcb3038：`artifacts/pcb3038-ht-scan-20260901T222415Z-2628279`，clean commit `63133c7`；
- rl5915：`artifacts/rl5915-ht-scan-20260901T223506Z-2634576`，clean commit `6cb1145`；
- d15112：`artifacts/d15112-ht-scan-20260901T223520Z-2635062`，clean commit `6cb1145`。

三次均使用物理 GPU 1、8 个固定 CPU cost threads、相同搜索预算与受保护最优 tour。两个 commit 之间只有 V12 研究文档变化，执行代码和参数相同。

| 实例 | states | PROVEN / UNRESOLVED | 提交边 | CPU leaf（ms） | CPU 融合 leaf（ms） | leaf 加速 | CPU search（ms） | CPU 融合 search（ms） | search 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pcb3038 | 12,383 | 2 / 6 | 2 | 1,420.648 | 1,235.915 | 1.149× | 1,840.895 | 1,654.602 | 1.113× |
| rl5915 | 10,016 | 3 / 5 | 3 | 287.759 | 265.294 | 1.085× | 1,354.688 | 1,333.985 | 1.016× |
| d15112 | 10,003 | 2 / 6 | 2 | 882.344 | 747.463 | 1.180× | 7,617.406 | 7,496.056 | 1.016× |

对每个实例，非融合 CPU、融合 CPU、全 CUDA、hybrid 和融合 hybrid 都拥有相同 target 工作签名、规范 leaf/Hamilton 计数、独立 CPU proof 重放结果、最终边文件和受保护 tour。具体规范工作量为：

| 实例 | leaf calls | cursors | consumed rows | 改善候选 | CPU 认证 cells |
|---|---:|---:|---:|---:|---:|
| pcb3038 | 9,120 | 9,891 | 727,635 | 987 | 51,309,996 |
| rl5915 | 1,245 | 1,497 | 42,927 | 438 | 2,718,284 |
| d15112 | 1,114 | 1,371 | 37,925 | 414 | 2,286,204 |

三个实例的 CPU 融合路径都是五路中最快；对应的 all-CUDA/hybrid/fused-hybrid search 为 `2,482.606/2,260.692/2,093.695 ms`、`1,647.347/1,516.595/1,517.616 ms` 和 `8,071.956/7,776.223/7,611.014 ms`。CUDA leaf 融合在 rl5915 上略慢，因此不能把 CPU 结论无条件外推到 CUDA/auto 路径。

最终图与活动边 SHA-256 分别为：

- pcb3038：`fe11f98414b04c0e` / `0a4f3ae830a6dbbd8449d1d6c85a7944475b85294919eaace9f5471b0fb81810`；
- rl5915：`5149d251ecac204d` / `1087c7d5b98e650e8b2f11eadf8d5667112601365c9dec66b3c1e3fe2b4b4743`；
- d15112：`29c3b8fccaf1a3fc` / `39abcae8832b5eca8cee278237eafe52ed2f053fbe2abefd7f8593e746a189b4`。

## 3. 后端感知默认值

提交 `623f167` 将 CLI 策略改为：

- `--backend cpu` 继承出的 CPU leaf，或显式 `--leaf-backend cpu`，在未给出开关时默认 `--fuse-leaf-buckets 1`；
- `--leaf-backend auto|cuda` 仍默认为 0；
- 显式 `--fuse-leaf-buckets 0|1` 始终覆盖默认值；
- C++ `HtWavefrontOptions` API 的初始值仍为 false，避免静默改变已编译调用者的调度；
- V12 基准显式把第一路锁定为 0、第二路锁定为 1，所以默认值变化不会污染历史 A/B。

`ht-prove` stdout、`ht-scan` stdout 和 scan report 都显式记录实际融合值。CPU wavefront/scan 集成测试要求默认值为 1，CUDA hybrid 测试显式要求 0。

## 4. 回归门禁

默认策略变更后完成：

- CPU Debug 17/17；
- CPU Release 17/17；
- CUDA Release 20/20，含 CPU/CUDA 差分；
- GPU 1 `compute-sanitizer --tool memcheck`：0 errors；
- pcb3038 1-target V12 五路冒烟：规范工作量、proof、边文件和公开最优 tour 均通过。

GPU 仍只产生候选数组，CPU 逐 cell 整数矩阵认证、改善 witness 重建、proof 重放和 tour 门禁都未削弱。

## 5. 新瓶颈

融合后 pcb3038 的 leaf 仍占 CPU search 74.70%；但 rl5915/d15112 的 leaf 只占 19.89%/9.97%。两个更大实例的 `path_append_ms` 为 610.466/5,316.182 ms，分别占 search 45.76%/70.92%。这表明下一切片不应继续围绕 pcb3038 cost scorer 过拟合，而应先将 path-append 拆成批输入构造、候选评估、CPU 规范化/差分认证和 child 物化子阶段，然后用三实例决定优化目标。
