# Point 近点预热与剩余强度缺口（2026-09-06）

分支：`research/fgpu-hybrid-beat-hs2014`。前一个检查点为 `04eb81c`。
本文继续[冻结执行计划](../design/FGPU_Hybrid_Beat_HS2014_Execution.md)，不是完成声明。

## 本轮代码

新增显式实验开关 `--point-prime-near 1`，要求同时开启 `--point-near-first 1`
与 `--point-adaptive-start 1`。仅当真实 pair frontier 大于 edge frontier 时，
在首次延期服务里额外尝试一次近点前缀，给后续 edge 服务提前提供 nonpair 信息。
默认关闭，不按 instance 名称选择配置。

- 32 个近点仅限定这次额外预热，不限定最终 Point 域；未关闭的 pair 原样保留。
- 每个选中 point 的全部 Hamilton AND replies 必须关闭，并经过独立 GPU replay。
- 同一个不可变 `run_point_service` 标志控制 proposal 和 replay，不能因为预热完成
  标志已经改变而跳过 replay。
- 最终仍执行完整 Point/Direct-Fix 回扫，再重新唤醒 edge 服务直到联合固定点。
- manifest 单列预热次数、提案数和时间；`point_prime_ms` 已包含在 `lp_point_ms`，
  不能重复计入 wall。基准工具记录开关，混用配置不能通过四例验收。

新增中间快照测试暴露初始 CSR 问题：稳定边数组中已有 inactive 边时，host CSR
还含这些槽位，但 pair offsets 按活动度数计数，旧代码会被事务检查安全拒绝
（code 8），不写入 live 状态。现改为在首轮服务前由 GPU 压缩并验证 CSR。
正式 hybrid 入口仍只接收原始完整图；测试中间快照不放宽这一入口。

## pr299 同二进制单次对照

原始坐标完整图，无输入标签、tour 或预处理边；标签仅由计时外检查器使用。
两行使用同一张 RTX 4000 Ada、原生 primal-dual-sec、排列目录、近点优先、
adaptive-start；仅切换预热开关。节点存在其他作业，是开发 pilot，非 clean 三次中位数。

| 配置 | wall | 剩余边 | fixed | nonpair | Point | Quick | replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 预热关 | 174.619 s | 1246 | 26 | 2392 | 8.782 s | 42.460 s | 34.160 s |
| 预热开 | 170.085 s | 1246 | 26 | 2394 | 19.788 s | 33.552 s | 27.759 s |

预热一次耗时 11.012 s，产生 53,754 个完整关闭的提案。它节省后续 Quick/replay
时间，wall 减少约 2.60%，比值 1.027×；这不足以确认稳定收益，默认仍关闭。
两行 state hash 分别为 `6713bf43b73a784a`、`cac4154704984276`，不是同状态消融；
最优 tour edge/fix/nonpair 冲突均为零。累计提案数不是最终 surviving nonpair 数。

记录：`artifacts/hybrid-pr299-prime-ab-v3/`。
二进制 `.tmp/hybrid-point-prime-v3` SHA256：
`8f66fd5b67ece46b738c3976bf62bd99c4a72d70ae5c82283dd388bf8c4d3cdc`。
该冻结二进制含预热，但早于中间快照初始 CSR 修复；其原始完整图入口不受该修复影响。

## pr1002：速度已有收益，强度未过门槛

上一轮冻结 `.tmp/hybrid-point-frontier-v2` 的完整运行已结束：
501,501 → **5,002** 条边，24 fixed、10,297 nonpairs，**1,161.415 s**。
相比旧全度数版本的 3,176.233 s，为 **2.735×**；edge/fix/nonpair 三个输出文件
分别逐字节相同，最终 state hash 均为 `697baad650b13d0f`。
这是合并 LP/排列/排序/调度改动后的版本比较，不把比值归因于一个开关。
完整数据、时间分项和身份见[上一份报告](74_FGPU_Point_Reuse_and_Frontier_Scheduling.md)。

2014 为 4,521 条、9,021 s。因此历史 wall 比值是 **7.767×**，但本方法仍多
**481 条**，严格胜出还要再删除至少 **482 条**，不能宣布等强度加速或计划完成。
四例联合门槛、同机作者源码速度门槛尚未通过。

### 剩余缺口与后续实现顺序

1. **LP 强度及有效 UB**：GPU 自建 UB 为 266,046，最优标签 259,045 仅供事后解释，
   差 7,001（约 2.70%）。最终 LP 下界约 253,857，仍低于最优约 2.00%。
   当前适配器是有限 SEC；缺少动态通用整数 cut pool、原始解 mincut 与奇边界
   2-matching。PDHG 可行性 0.0034712、相对 gap 0.000140767 也未到 `1e-5`。
   不能把更多 PDHG 迭代或更好的调度直接等同于补齐这些约束。
2. **局部域完整度**：正式链仍最多 3 paths / extra-edge depth 2，未实现计划的
   6 paths / 10 reveal、共享端点规范化和统一 edge/nonpair/fixing continuation。
   这是真实方法域差距，近点预热不会扩大它；也不能在没有消融前把 481 条全归因于它。
3. **下一处速度重点**：新 pr1002 的 Quick 为 356.877 s、replay 为 300.238 s、
   Main 为 226.252 s，Point 已降至 146.708 s。继续只调 Point 的收益空间缩小。
   按计划实现共享多输出 path-cover、成功见证 replay、分形状队列与完整 AND 覆盖，
   然后以端到端和最终边集复测，不用 GPU 利用率本身当作加速证据。

远程 `cuda19` 的 L4 正在跑完整 vm1084；作者单核 pr1002 对照也尚未结束。
在它们结束前不填写最终结果，不把 L4 与 RTX 4000 Ada 的计时混作同硬件对照。
vm1084 在一次检查时已运行 **81 分钟**，仍在首轮 Quick-HS，GPU 利用率为 100%。
这已经暴露当前配置的密图长尾，不能用 pr1002 的结果代表四例都有加速；该运行继续
保留，无时间截断。初始 geometry 后仍有 48,295 条边，明显多于 pr1002 的 23,927，
但不能仅凭边数推断具体展开工作量或硬件归因，需要后续 root/reply/leaf 分项画像。
本机 RTX 4000 Ada 已启动 pcb1173，随后同卡串行运行 pcb3038，二者共用同一冻结
`.tmp/hybrid-point-prime-v4`、原生 LP/排列目录/近点优先/adaptive 配置，预热关闭。
记录目录：`artifacts/hybrid-pcb-native-frontier-v4/`。这是 pilot，不是提前通过门槛。
另在本机另一张空闲 RTX 4000 Ada 启动同卡串行 vm1084 预热开/关对照（先开后关），
共用冻结 v4 二进制：`artifacts/hybrid-vm1084-prime-ab-v4/`。各实例仍只使用一张 GPU，
不把跨实例并发算作多 GPU 加速；该对照未结束，不填写预热收益。

## 验证状态

近点预热的 18/19/24/40 点零成本完整图 12 次回归已通过：真实 frontier 等号边界、
延期、恰好一次预热、最终完整回扫、全部最优边和邻边对保留。
记录：`artifacts/hybrid-prime-v3-dense.log`。

中间快照 CSR 修复后：

- CPU **37/37**：`artifacts/hybrid-cpu-tests-prime-v4-coverage.log`。
- CUDA **68/68**，505.73 s：`artifacts/hybrid-cuda-tests-prime-v4.log`；含 234 次无标签
  完整求解和 12 次密图调度回归，非法预热依赖在创建输出前被拒绝。
- 第 7 项另加强并重跑：36 组 EUC/CEIL、重复坐标、多最优 tour，完全图及包含全部
  最优 tour 的中间快照，144 次 GPU 求解，实际 **293** 个预热提案。覆盖 2/4 CTA
  策略与排列目录开/关，逐一检查全部最优 edge、mandatory fix 和邻边对。
  这是测试专用快照，不作为无标签性能入口。记录：`artifacts/hybrid-prime-v4-coverage-details.log`。
- 加强前后的 positive-prime **memcheck 零错误、racecheck 零 hazard**，均已结束。
  专项关闭 LP、geometry 和 full-metric，
  隔离 Point/CSR/事务；这些其他路径由全套回归和先前专项另行覆盖。

加强后的 sanitizer 日志：`artifacts/hybrid-sanitizer/prime-v4-coverage-{memcheck,racecheck}.log`。
加强后的冻结测试二进制为 `.tmp/hybrid-point-prime-unit-v4-coverage`，SHA256：
`e63062e4300018d233b7922fb18e0ac0cba3296ff610fbfbb27431ce97ef4559`。
