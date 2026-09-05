# 无标签完整图与全度数 metric（2026-09-06）

分支：`research/fgpu-hybrid-beat-hs2014`，基于 `4e323e2`。
目标与完整验收约定见 [执行计划](../design/FGPU_Hybrid_Beat_HS2014_Execution.md)。
本报告是已验证的实现检查点，不表示四实例目标或完整设计已完成。

## 已接入正式入口

- `solve --profile hybrid-e2e` 仅接收原始 TSPLIB 坐标；拒绝 tour、最优成本和预处理边集。
- GPU 生成完整边及三角 int64 距离，独立逐 bit 平方根重放；EUC/CEIL 均保持严格舍入。
  `vm1084` 原始数据含 `.5` 坐标，使用分母 2 的精确整数分子，不改写坐标或距离类型。
- 需要 LP 时，GPU 从最多 128 个确定性 NN 起点执行严格改进 2-opt、循环 Or-opt 1/2/3，
  到局部固定点后独立检查排列与成本。不保护普通 incumbent，不读取最优标签。
- `--lp-backend off` 保留 point nonpair、Direct Fix、nonpair fixing 和固定边传播；
  没有虚构的 LP 下界、迭代或 LP closure。
- Main-Edge 条件 pair 按 `(root, center, snapshot)` 缓存为 bitset，覆盖全部 CSR slots。
  显存不足以同时容纳所有根时分批完成全部根；不截断任务。条件 pair 不发布为全局 nonpair。
- 全度数 metric-excess 由 warp 并行覆盖 z 的邻边对，复用每个 z1 的 Opt24，再分摊 Opt34；
  该路径不再使用旧实现 degree≤4 的门槛。旧路径仍可显式消融。
- Main replay 只检查成功的一个/两个中心，独立重建条件 pair，不重做几何候选排序。
  修复 replay 的 witness 指针传参及纯 JV/Quick-HS 旧入口的收敛标志回归。

边界：CPU 仍解析文件、构造几何 KD 树和输出元数据；完整 Edge 数组目前有一次
GPU→host→resident 的接口往返，三角距离本身不回传。删边/pair/fixing 的候选、
精确授权与提交都在 GPU。不能把这个检查点称为“所有初始化和数据结构也不经过 CPU”。

## 测试

已验证构建冻结为 `.tmp/hybrid-main-cache-v1`，实际身份见每次运行 manifest。

- CPU **37/37**：`artifacts/hybrid-cpu-tests-v4.log`。
- CUDA **67/67**：`artifacts/hybrid-cuda-tests-main-cache.log`。
- 新入口 12 组小图 × 8 个 LP/距离缓存/条件缓存/metric 配置，共 96 次无标签完整求解。
  对独立全最优 tour oracle 检查边并集、固定边交集、最优邻边对；纯缓存消融终态一致。
- GPU bootstrap 24 例：整数/半整数 × EUC/CEIL × 6 个规模；独立成本与全部 2-opt/Or-opt
  邻域复核。memcheck 零错误，racecheck 零 hazard。
- 全度数 metric 240 例与串行精确谓词一致（49 open、191 closed），最大完整度数 23；
  racecheck 零 hazard。
- 带 Main 缓存的 7 节点完整 `solve`：全 kernel memcheck 零错误、racecheck 零 hazard。
  日志：`artifacts/hybrid-sanitizer/main-cache-{memcheck,racecheck}.log`。
  这些不是所有大实例、所有 kernel 形态的穷尽 sanitizer 覆盖。

## 当前实测，不混用版本

节点有其它训练作业，以下均为开发 pilot，不是独占节点三次中位数。

| 实例/冻结版本 | 输入边 | 剩余边 | 进程 wall |
| --- | ---: | ---: | ---: |
| pr1002，integer-bootstrap-v1，LP off、旧低度 metric | 501501 | 5619 | 2778.532 s |
| berlin52，replay-fixed，LP off、旧低度 metric | 1326 | 195 | 10.18 s |
| berlin52，main-cache-v1，LP off、全度数 metric | 1326 | 164 | 17.56 s |

`pr1002` 比 [2014 Table 1](https://arxiv.org/pdf/1402.7301) 的 4521 条多 **1098** 条；
9021/2778.532≈3.247 只是历史论文时间比，**不是等强度加速**。原始输出与锁定最优
tour 零冲突，GPU replay 拒绝为 0。Point 为 2231.39 s（约 80.31%），Main 仅
1.886 s：只优化 Main 原有重复计算不足以解决端到端热点。

`berlin52` 全度数版本少 31 条边（15.90%），但本次 wall 增加约 72.5%。独立
Concorde 从原始坐标求得最优成本 **7542**，两版本与该 tour 的 edge/fix/nonpair 均零冲突。
此例验证加强规则有效，不代表四实例达标。Concorde 只用于求解后的测试。

全度数 `pr1002` 的完整预跑已启动，目录 `artifacts/hybrid-pr1002-full-metric-pilot/`；
在结果落盘并通过 postcheck 前，不填写预计边数或加速比。

## 2014 单核参考与复现入口

作者源代码只复制到项目内 ignored `.deps/`，不提交受限源码。串行适配和真实
单 worker MPI 使用同一份完整 options；8 节点输出均为 8 条，`berlin52` 均为
214 条，完整边集一致。身份和补丁见 `build/hs2014-reference/identity.json`，
差分记录见 `artifacts/hs2014-reference-differential/summary.json`。

参考配置的 `S3_PT_SEARCH_DEPTH=11` 对应源码 `depth < option` 覆盖到 depth 10。
串行适配还修复了最后一批 worker 消息的 drain 时序；串行/MPI 两路径应用同一补丁。
这里是作者 2015-02-27 源码的锁定重运行，不把其运行时间冒充 Table 1 原机器时间。

```sh
# 所有命令在项目根执行；先设置项目内 TMPDIR/CUDA_CACHE_PATH。
python3 tools/prepare_hs2014_data.py
python3 tools/build_hs2014_reference.py
python3 tools/check_hs2014_reference.py
python3 tools/benchmark_hybrid.py --exe build/cuda-release/fgpu-elim \
  --gpu-uuid GPU_UUID --output artifacts/hybrid-formal --lp-backend off
python3 tools/benchmark_hs2014.py --cpu-core CPU_CORE --gpu-uuid GPU_UUID \
  --output artifacts/hs2014-formal
python3 tools/compare_hybrid.py --gpu-runs artifacts/hybrid-formal/runs.json \
  --cpu-runs artifacts/hs2014-formal/runs.json --output artifacts/hybrid-comparison.json
```

默认一轮预热、三次完整测量。正式测试应按协议交错/独占节点执行；两脚本分别提供
单侧采集，尚未自动编排 CPU/GPU 交错或判定全部 CPU 干扰。`--allow-busy` 的结果
明确不用于验收；不能因节点还有作业就把这些 pilot 标成 clean。

## 下一项与明确未完成项

下一项优先处理 Point 热点：保持完整回复覆盖，复用小叶子的组合枚举和距离，
优化成功见证的尝试次序；每一项做开关对照，不以 GPU 利用率替代 wall 收益。

通用动态 cut pool、primal mincut/奇边界 2-matching、LP 收敛门槛、统一 6-path/10-reveal
continuation、多输出 path-cover、跨根设备队列/取消和精确依赖唤醒仍未实现；manifest
如实声明当前通用局部域为 3 paths / extra depth 2。四实例的全量强度和双速度门槛
尚未通过，完整方案仍在继续实现。
