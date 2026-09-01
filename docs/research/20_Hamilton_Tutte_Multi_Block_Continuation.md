# Hamilton–Tutte cooperative multi-block continuation

## 范围

M4.3b3b2b2b2b2b2 把 continuation 真值传播从单 block 正确性基线扩展为可选 cooperative multi-block persistent kernel。主机工作图、leaf/reply/path-append 候选器、CPU 规范真值和 V1 proof 提取都不改变。

普通 CUDA block 之间没有安全的 grid barrier。若多个 block 独立观察 `queue_head == queue_tail`，可能在其他 block 尚未发布父状态时提前退出。因此本实现不采用自旋式“队列暂空即结束”，而使用 cooperative launch 保证所有驻留 block 能在每轮调用 `grid.sync()`。

## 冻结批次协议

设备全局控制字保存 `batch_begin`、`batch_end` 和 `stop`。cooperative kernel 执行：

1. grid thread 0 读取初始 `queue_tail`，冻结 `[0,batch_end)`；
2. 所有 grid threads 以全局 stride 消费该区间；
3. child 通过原子 `remaining_children/move_failed/remaining_moves/status` 完成父状态，并 append 到同一队列；
4. `grid.sync()` 等待本批全部原子更新和 queue 写入；
5. thread 0 把旧 `batch_end` 设为下一批起点，并冻结新的 `queue_tail`；
6. 新区间非空则继续；若为空，只有 `queue_tail == state_count` 才正常终止，否则报告停滞。

每个状态仍必须恰好入队一次。queue 超过 `state_count` 返回 overflow；合法有限 DAG 在全部状态完成前无新项返回 stalled。两者都使显式 CUDA 后端 unresolved，绝不使用部分状态。

## block 选择与回退

`HtWavefrontOptions::propagation_blocks` 和 CLI `--propagation-blocks N` 的语义为：

- `N=0`：自动；按 `ceil(state_count/256)` 计算有用 block 数，并截断到 cooperative kernel 的实际 residency；
- `N=1`：强制原 single-block persistent kernel；
- `N>1`：请求确切 cooperative block 数。

运行时同时检查 `cudaDevAttrCooperativeLaunch`、每 SM active blocks 和 SM 数。显式请求超过 residency 或设备不支持 cooperative launch 时闭门失败；`auto` 则安全保留 single-block 基线。返回指标 `propagation_blocks` 与 `propagation_cooperative` 记录实际 launch，不把请求值冒充执行值。

## 差分与并发回归

直接设备真值表覆盖：

- 固定 4-state AND/OR 图在 single block 与 2-block cooperative 下逐状态相同；
- 512 个同时完成的 child 汇入同一个 AND move，确保两个 block 都参与同一原子 `remaining_children`；
- 第 301 个 child 失败时根必须失败；改为成功后 513 个状态必须全部成功；
- 自动模式对 513 states 选择 3 blocks，并与显式 2-block 状态数组相同；
- 超过 cooperative residency 的显式 block 请求被拒绝。

固定 recursive-point 的完整运行显式请求 2 blocks，记录：

| 指标 | 值 |
|---|---:|
| states / moves / replies | 34 / 18 / 84 |
| propagation blocks | 2 |
| cooperative | 1 |
| CPU state verification | 通过 |

该运行生成的 4-node `recursive-ht-proof-v1` 与 single-block 文件逐字节相同，并通过独立 `ht-verify`。小图的自动模式仍选择一个 block；显式 2-block 只用于验证语义，不代表小图性能更好。

## 安全边界与后续

GPU 状态数组返回后仍与 `EvaluateWavefrontCpu` 的每个状态逐项比较；cooperative kernel 只加速真值传播，不直接授权删边。工作图仍由主机构建，成功子树仍从 CPU 认证状态提取并由全局 verifier 重放。

当前只支持单 GPU 同步 cooperative launch；不支持的设备保留原单 block 或 CPU fallback。验证成功的 HT 候选现已接入[不可变 snapshot/epoch 的确定性提交](21_Hamilton_Tutte_Epoch_Commit.md)；下一阶段补齐中大型实例的状态峰值、cooperative occupancy、全图目标调度与端到端收益记录。
