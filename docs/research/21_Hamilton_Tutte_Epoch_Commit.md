# Hamilton–Tutte 不可变 epoch 提交

## 范围与信任边界

M4.3b3b2b2b2c 把已经生成的 `CUDAEE_HT_RECURSIVE_PROOF_V1` sidecar 接入真实删边提交链。搜索与提交保持分离：`ht-prove` 可以用 CPU 或经 CPU 全量差分的 CUDA wavefront 生成候选证明；`ht-commit` 不信任生成后端，只在输入图的同一个不可变快照上调用 `VerifyHtRecursiveProof` 重放完整 continuation arena。

流程为：

`immutable snapshot -> read all sidecars -> CPU verify all -> canonicalize targets -> degree gate on copy -> publish epoch`

任何 sidecar 的快照哈希、目标边、AND replies、child 引用、leaf witness 或文件结构失败，整批抛错且调用方图不变。GPU 状态、搜索成功码和浮点量都不直接修改 `active` 位。

## 整批规范化与原子发布

`CommitHtProofEpoch` 接受同一快照上的多个递归 HT proofs：

1. 记录唯一的 `snapshot_hash` 和提交前活动边数；
2. 对每份 sidecar 完整 CPU 重放，并把规范目标端点映射回稳定 `edge_id`；
3. 以 `(u,v,serialized-proof)` 排序；同一目标有多份合法证明时保留字节序最小的一份；
4. 在规范边序上执行与 JV 相同的最小度门禁，任何顶点都不会因一批删除降到度数 2 以下；
5. 在 `GraphSnapshot` 副本上修改活动位并重建 CSR；
6. 证书复制、最终哈希计算全部成功后，才用一次 move 发布新图。

因此输入 sidecar 顺序不影响最终图。候选复核、度数门禁或内存分配的晚失败都不会暴露半提交 CSR。共用的内部提交器还会在发布前再次比较快照哈希，拒绝验证期间被替换的图。

## 消元证明 V2

JV-only 运行继续写 `CUDAEE_PROOF_V1`，原有工具和回归文件不变。只有实际提交 HT 删除时才写 `CUDAEE_PROOF_V2`：

- `record` 仍绑定 epoch、snapshot、稳定 edge id 与端点；方法为 `HT`，JV witness 位置固定为 `-1`；
- `certificate_index` 必须唯一引用一个内嵌 HT V1 sidecar；
- sidecar 使用显式字节长度和结束标记保存，不依赖外部路径；
- outer record 与 inner proof 的 snapshot/target 必须相同；
- 每份 sidecar 最多 256 MiB，聚合 payload 最多 256 MiB，主文件最多 512 MiB；record、metrics 和 sidecar 数量分别有一百万项上限；
- reader 拒绝重复字段、未知方法、非法索引、未引用/重复引用、截断 payload 和 `END` 后尾随字段。

`verify` 同时读取 V1/V2。V2 重放先在图副本上逐 epoch 验证全部 JV/HT 记录，再检查规范提交顺序、度数门禁和最终哈希；只有完整文件通过才更新调用方图。V2 因而是自包含删除证书，不依赖原始 `--ht-proof` 文件继续存在。

## CLI

一个 epoch 可以重复传入 `--ht-proof`：

```bash
build/cuda-release/cudaee ht-commit \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --output artifacts/recursive-point.epoch.edg \
  --proof artifacts/recursive-point.epoch.proof \
  --ht-proof artifacts/recursive-point.single-block.ht-proof \
  --ht-proof artifacts/recursive-point.two-block.ht-proof

build/cpu-release/cudaee verify \
  --tsp tests/data/recursive-point.tsp \
  --edges tests/data/recursive-point.edg \
  --proof artifacts/recursive-point.epoch.proof
```

输入 sidecar 必须全部绑定当前 `--edges` 快照。先提交一个 epoch 后，旧 sidecar 会因 snapshot hash 过期而被下一次提交拒绝；必须在新图上重新搜索。

## 回归证据

固定 recursive-point 实例把 single-block 与 cooperative two-block 产生的两份等价 sidecar 同时交给提交器：

| 指标 | 结果 |
|---|---:|
| snapshot hash | `d7bfbec67ffc9a66` |
| proposed / verified | 2 / 2 |
| unique committed | 1 |
| active edges | 28 -> 27 |
| final hash | `78ce8b9a9dc29473` |

目标 `2-4` 被删除，V2 经独立 CPU `verify` 重放到相同最终哈希。单元测试另外覆盖重复目标规范化、损坏 continuation arena、外层/内层绑定、一个合法加一个坏 sidecar 的整批原子失败，以及失败重放不修改调用方图。既有 n=6–12 最优边穷举、pr299 JV V1 重放与 CPU/CUDA 最终哈希回归保持通过。

## 后续衔接

M5 已在本提交器之上增加[有界全图 HT 目标扫描](25_M5_HT_Target_Scan.md)：同一快照的确定性目标切片先全部搜索，再调用本接口原子发布。`gpu-eliminate` 的自动候选器仍是 JV；跨目标融合、多 epoch 搜索策略和多 GPU 仍待完成。M3.1 的对偶稳定化与精确定价后边集导出也仍独立待完成。
