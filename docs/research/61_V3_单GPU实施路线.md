# V3 单 GPU 实施路线

## 当前进度（2026-09-03）

- A0 已完成：wavefront Trace、严格 reader、文件往返和窗口 replay 均有单元/CLI 门禁。
- A1 已完成：CPU compact value pass、成功后 traceback oracle、CUDA `k<=13` value 候选器和 candidate-only 集成均已落地。
- C1 完成 host-window 语义原型和第一版跨 target heterogeneous leaf broker：单 dispatcher 合并不同 required edge，CUDA 只回传候选位图，CPU 精确重建 witness。SoA continuation ready queue、generation token 与 CUDA Graph 调度仍待实现。
- C1.5 完成重复重放合并和 sidecar 并行验证；显式 move-only token 尚未实现，因此该阶段只标为部分完成。
- 单 GPU多 target workers 已完成，可让多个 worker 共享一个显式 ordinal；这不是多 GPU性能主线。

原型基线见 [V3 单 GPU原型实现与 Pilot](63_V3_单GPU原型实现与Pilot.md)；最新 broker 实现与正式性能门禁见 [V3 单 GPU 跨目标 Leaf Broker](64_V3_单GPU跨目标LeafBroker.md)。

## A0：短路 Trace 与 replay

- 定义版本化、可校验的 AND/OR Trace 格式。
- wavefront 和递归参考路径均记录规范子节点次序、真值、决定性 child、实际访问量与快照哈希。
- replay 以固定窗口模拟 speculation `1/2/4/8/无限`；一个窗口内已经发出的任务允许完成，但只按规范次序提交结论。
- Trace 不是证书，不能参与删边授权。

## A1：紧凑精确 DP

- 保留旧完整 DP 为测试 oracle。
- CPU value pass 只保存相邻 popcount 层；仅当存在严格改善时运行 traceback pass。
- CUDA 按 block 数分桶，`k<=13` 时每个 CTA 求解一个任务；设备能力、共享内存和整数上界在 launch 前验证。
- CUDA 返回最优值和状态，CPU 对成功项重建 witness 并调用既有 verifier。

## C1：转置短路调度

已落地的中间切片：

- 32 个 target workers 共享一个单 GPU dispatcher，以两请求机会式微批避免全 worker 栅栏。
- 合批 API 为每个 path system 保留独立 required edge 与 cursor，仅合并同 `k` cost tasks。
- CUDA 对 `4/25/208` 个 proper k-opt templates 生成每 task `1/1/4` 个 `uint64_t` candidate words，CPU 只对命中位重建 witness。
- `CUDAEE_HT_SCAN_REPORT_V19` 分开记录逻辑 leaf windows、物理 broker batches/states 和实际 speculation width。
- d15112 五对 clean A/B 中 target execution 约 `1.009x`，process wall 约 `0.997x`；端到端仍只能判定为持平。

剩余设计目标：

- continuation 使用 SoA：目标、父 continuation、规范 child ordinal、generation、program counter 和状态引用分离存储。
- 相同 program counter 的 ready work 跨目标合批，复用现有 leaf、reply、path-append 和 propagation kernel。
- OR 在第一个规范成功 child 后取消后继；AND 在第一个规范失败 child 后取消后继。
- speculation 任务使用 generation token 防止迟到结果污染已完成 continuation。
- 第一版使用设备常驻队列和 CUDA Graph 批次，每个调度轮次最多一次主机同步；只有画像确认 launch/空洞超过 15% 才引入持久 dispatcher。

## C1.5：验证热路径合并

- 规范 proof 在同一快照上只做一次完整 CPU 精确验证。
- 成功后生成 move-only 的验证 token；commit 只检查 epoch、快照指纹、目标边和 proof hash。
- 离线 `verify` 继续从序列化 proof 独立重放，token 本身不序列化。

## 提交策略

每个阶段独立提交并通过 CPU 测试；CUDA 阶段还必须通过差分测试与 compute-sanitizer。研究数据写入 `artifacts/`，可复现实验结论写入 `docs/research/`，不提交临时或受限参考代码。
