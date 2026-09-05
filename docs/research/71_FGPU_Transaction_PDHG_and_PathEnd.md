# FGPU：快照事务、稀疏 PDHG 与完整端点 OR

本文记录 `research/fgpu-strength-upgrade-p0-p8` 上继 `563d78a` 后的实现。
它替代此前将静态窗口 SEC 次梯度称为“完整 cutting-plane”的表述。

## 实现与边界

| 项目 | 当前实现 | 不能据此声称的能力 |
|---|---|---|
| 快照事务 | fixed/non-pair/edge 的全部 replay 先完成，再检查组合终态并提交 | 不是一般 proof DAG 或通用 HT continuation |
| SEC replay | 根据封存的子集成员独立核对 cut validity、非负 dual、完整 incidence | 没有新增 local cut、comb 或 blossom |
| 稀疏 PDHG | 设备 CSR/CSC、整数系数、原始/对偶更新、对角预条件、平均迭代、warm start、64 次更新 CUDA Graph | 正式适配器目前仍只包含旧 SEC；不是完整 LP cutting-plane |
| LP 授权 | 新旧候选量化后比较精确 Signed128 下界；edge/fix/pair/path 共用独立 GPU replay | 不能将浮点 residual/gap 当作删边许可 |
| Point path-end | 对 3+3 reply 尝试全部四个端点 OR，各端点内部完整 AND；proposer 排序，replay 自然顺序 | 重叠形态仍保守开放；不是 m≤5、深度 6 的完整 HT |
| Opt34 消融 | 普通排列、预筛加排列、预筛加 subset DP 三种物理后端 | 预筛/DP 不保证更快，需要实测选择 |
| 可复现性 | build/source/executable/input SHA-256、GPU UUID、驱动、分阶段计时、交错进程 wall | 单次计时或不同终态不是等强度加速 |

正式入口仍是 `fgpu-elim solve --mode gpu-safe`，不调用 CPU solver、separator、
逐边审计或困难叶回退。默认不生成 certificate 文件。这里的“固定点”仅表示当前
实现的服务不再更新 edge/fixed/non-pair，不代表论文所有消除规则均已穷尽。

PDHG 的 `Iterate` 参数是执行批次长度；本次正式适配器每个 LP epoch 调用一批，
不能把 `termination=fixed-point` 解读为 `pdhg.converged`。manifest 同时记录
`pdhg_primal_violation`、`pdhg_relative_gap` 和实际迭代数。

## 为什么先修事务

旧流程在 point non-pair replay 前先写入了新 fixed bits。即使函数参数是
`const GraphView`，其指针仍指向被修改的 live 数组，不能保证同快照授权。

现在的顺序是：

```text
构造 pair view、封存 epoch
  → proposals
  → 全部独立 GPU replay（不修改 live 图）
  → prospective delete/fix/non-pair 组合校验
  → 发布 fixed/non-pair/edge
  → 下一 epoch / CSR 压缩
```

组合校验覆盖最低度、固定度≤2、delete/fix 冲突、至少一个包含所有固定边的
合法邻边对，以及 proper fixed subtour。覆盖全图的 Hamilton fixed cycle 允许保留。
任何失败均在 live 写入前抛出错误，不输出冒充成功的部分终态。

## 基准身份冻结

- 基线：在 `.tmp/baselines/563d78a` 创建 detached worktree 并独立构建；不把当前
  dirty 源码与旧可执行文件混用。
- 开始时未提交改动：保存在 `.tmp/strength-complete-recovery/pre-implementation.patch`。
- 原可执行文件 SHA-256：
  `a8938ea5c4c694b6bc1552352760a6004d8c51f725d339d5d8bac0f964b13843`。
- 新运行必须以自己的 JSON 内嵌身份为准，不能只写分支名或手工添加的版本标签。

## 最终验证版本的 pr299 同卡配对结果

本节绑定下文正确性门禁的 `37122d8a...` 二进制，已包含 3–6 节点完整环与
全最优解 oracle 的修复。GPU 为 `GPU-eeb32f5a-0239-4373-b0dc-cdb6f783c2ce`
（RTX 4000 Ada），每个变体预热一次、交错计时三次；全部以完整 1,208-edge
论文 LP 图为输入，未设置目标、reply、epoch 或时间截断。

| 实现 | 剩余边 / fixed / non-pairs | 进程 wall 中位数 | 三次范围 |
|---|---:|---:|---:|
| `563d78a` | 828 / 38 / 298 | 10.162 s | 10.149–10.213 s |
| 新版 2-CTA | 828 / 38 / 387 | 27.557 s | 27.521–27.635 s |
| 新版 4-CTA（默认） | 828 / 38 / 387 | 21.011 s | 21.009–21.012 s |

新版 4-CTA 相对同算法 2-CTA 为 **1.312x 同强度加速**，wall 减少 23.76%。
相对旧提交则仍为 **2.068 倍墙钟**：non-pair 增加 29.87%，但边数和 fixed
数量没有改善。因此本次交付没有达到“相对旧版整体加速”的目标。

新版六次正式计时终态 hash 均为 `afd5427ce4ea5135`，每次 1,353 次 GPU replay、
0 rejected、19 个验证通过的事务。当前编译的 2-CTA / 4-CTA 资源分别为
242 / 128 registers/thread、2 / 4 blocks/SM、1,312 / 1,552 local bytes/thread。
这些是静态资源指标，不是 achieved occupancy。4-CTA 首次正式计时中 Point
为 11.132 s，LP multiplier solver 为 1.334 s（其中 PDHG 为 0.385 s）。

原始记录：`artifacts/strength-complete/pr299-cycle-corrected-paired/`。运行边界
同卡无其他计算进程；同节点另一张卡同时进行 pcb3038 pilot，所以不是全节点独占
实验。后文旧消融记录保留以说明开发轨迹，不与本组跨卡混算严格配对加速。

### 最终验证版本的 pcb3038 全 LP 输入 pilot

同一冻结二进制，在 GPU `GPU-8ce2c8bb-d2fd-214d-c9ce-b22a66f38bbb` 上完成一次
完整 GPU-safe solve。输入是论文的 **6,883-edge LP 图**，不是 4,613,203-edge
完全图；没有预热或重复计时，以下仅作为单次 pilot。

| 指标 | 最新结果 |
|---|---:|
| 初始 → 剩余边 | 6,883 → 6,324 |
| fixed / non-pairs | 427 / 1,789 |
| non-pair 比例 | 7.86581% |
| 内部 E2E / 进程 wall | 585.561 s / 585.679 s |
| Point / LP multiplier solver | 543.360 s / 1.867 s |
| PDHG（包含在 LP solver 中） | 0.592 s，35,000 次迭代 |
| LP 下界 / 已知最优成本 | 136,231 / 137,694 |
| LP path closure | 0 |
| replay / rejected / 已验证事务 | 3,855 / 0 / 18 |
| 最终状态 hash | `ae40f6bb8a9aa459` |

Point 占内部 E2E **92.79%**，独立 GPU replay 仅 0.203 s。瓶颈不是 CPU 精确审计
（正式链没有它），也不是 PDHG 本身。当前下界与已知最优成本仍差 1,463，LP
path closure 为 0，说明应优先补强 cut 模型，再拆分／复用 Point 的重复叶工作；
这不是再增加外层轮数就能解决的差距。

相对旧提交 6,326 / 424 / 1,404，最新只多删 2 条边、增加 3 条 fixed 和 385 个
non-pair。相对已记录的 2023 论文结果 5,548 edges、934 fixed、49.4% non-pairs，
仍多 776 条边、少 507 条 fixed。旧提交内部 E2E 三次中位 153.220 s，本次单次
内部 E2E 约为其 3.822 倍；这不是同版本三次配对实验，不能当作精确的加速结论。

记录：`artifacts/strength-complete/pcb3038-cycle-corrected-pilot/`；逐次 manifest
在 `artifacts/paired-bench/pcb3038-cycle-corrected-pilot-0c91d21689/current/run-0/`。
运行边界同卡无其他计算进程，部分时段另一张卡在做 pr299 配对实验。论文原始
数据与硬件／协议区别沿用 [70 号报告](70_FGPU_Strength_Upgrade_P0_P8_Implementation.md)，
不能把不同强度或不同机器的时间比称为等强度 GPU 加速。

## 初次消融：确认性能回退

同一 RTX 4000 Ada、`pr299` 论文 LP 图（1,208 edges）、预热一次及三次计时，
全量 GPU-safe solve 的进程 wall 中位数：

| 实现 | 剩余边 | fixed | non-pairs | 进程 wall |
|---|---:|---:|---:|---:|
| `563d78a` | 828 | 38 | 298 | 10.493 s |
| 单端点、Opt34 预筛/DP、旧 SEC 求解器 | 828 | 38 | 356 | 22.321 s |
| 同上加稀疏 PDHG | 828 | 38 | 356 | 22.722 s |

原始记录位于 `artifacts/strength-complete/pr299-initial-paired/`。
主要回退来自 Point 服务，不是 PDHG：PDHG 的增量约 0.4 秒。
这组实验不能作为“方案加速成功”的证据。

### 四端点、Opt34 与寄存器消融

同一冻结二进制、相同四端点搜索，在 GPU `8ce2c8bb...` 上三次计时的中位数：

| Opt34 后端 | 剩余边 / fixed / non-pairs | 进程 wall |
|---|---:|---:|
| permutation | 828 / 38 / 387 | 27.712 s |
| prescreen-permutation | 828 / 38 / 387 | 47.870 s |
| prescreen-subset-dp | 828 / 38 / 387 | 49.502 s |

随后在 GPU `eeb32f5a...` 上进行同二进制的寄存器预算交错消融，保持 permutation
与完整四端点，三次中位 wall 从 26.918 s 降至 21.267 s（同强度 `1.266x`，减少
约 21.0%）。所有终态 hash 均为 `afd5427ce4ea5135`。
这是两组独立实验，不能混合两个 GPU 的中位数计算严格配对加速。

Point 原 kernel 使用 255 registers/thread，仅能驻留 2 个 128-thread CTA/SM。
新增 `--point-cta-blocks 2|4` 编译两份相同搜索域的 kernel，允许实测寄存器与
spill/occupancy 的权衡。默认选择 permutation 与 4-CTA 策略。它不是目标、回复数或
运行时间限制；manifest 记录请求策略与实际 kernel 资源。

### 6 节点修复后的同卡配对结果（早于 5 节点修复）

修复全图环边界后，在 GPU `8ce2c8bb...` 上重新交错运行基线与两个 CTA 后端，
每个变体预热一次、计时三次。没有为 solve 设置目标数、reply、epoch 或时间预算。

| 实现 | 剩余边 / fixed / non-pairs | 进程 wall 中位数 | 三次范围 |
|---|---:|---:|---:|
| `563d78a` | 828 / 38 / 298 | 10.492 s | 10.474–10.546 s |
| 当前 2-CTA | 828 / 38 / 387 | 28.132 s | 28.112–28.138 s |
| 当前 4-CTA（默认） | 828 / 38 / 387 | 22.145 s | 22.132–22.153 s |

4-CTA 相对当前 2-CTA 为 **1.270x 同强度加速**，进程 wall 减少 21.28%；
实际资源从 255 registers/thread、2 blocks/SM 变为 128 registers/thread、
4 blocks/SM，local bytes/thread 则从 1,184 增至 1,456。因此不是没有代价的
occupancy 提升，采用依据是完整 wall 的实测结果。

相对旧提交，当前默认 wall 是 **2.111 倍**，non-pair 增加 89（29.87%），但最终
边数和 fixed 数没有变化。这仍然是强度／时间权衡，**不是相对旧版的整体加速**。
新版本全部六次计时的状态 hash 为 `afd5427ce4ea5135`，均 1,353 次 GPU replay、
0 rejected、19 个验证通过的事务。计时卡边界无其他计算进程；另一张卡同时做
sanitizer，不能将本组称为全节点独占计时。

原始记录：`artifacts/strength-complete/pr299-final-paired/`。
本组实际二进制 SHA-256 为
`b64a7b15b0727e6f99946d434ba974289c5319056911e0a47aeac22e658a8d34`，
源码树 SHA-256 为
`3c085b2d704a67d4182f61048b906c86cff943dc10ce305c719345a95d8c0782`。
这是提交前构建，manifest 如实记录 `git_dirty=true`；后续提交或重新构建不能
反向改变这些测量的二进制身份。

随后更广的小图检查又发现下文记录的 5 节点快速过滤问题，因此本组是开发过程
消融，不能单独作为最终可靠性证明。`pr299` 的实测终态保留作对照；最终构建需
使用修复后的 oracle 重新通过全最优门禁。

### pcb3038 的全 LP 输入 pilot：未达到性能目标

采用四端点、permutation、PDHG、独立 SEC replay，尚未应用 4-CTA 策略的单次
全量 LP 输入结果为：

| 指标 | 结果 |
|---|---:|
| 初始 → 剩余边 | 6,883 → 6,324 |
| fixed / non-pairs | 427 / 1,789 |
| non-pair 比例 | 7.86581% |
| 下界 / 已知最优成本 | 136,231 / 137,694 |
| LP path closure | 0 |
| replay / rejected | 3,855 / 0 |
| 内部 E2E / 进程 wall | 896.006 s / 896.120 s |
| LP multiplier solver / Point | 1.889 s / 856.674 s |

原始记录在 `artifacts/strength-complete/pcb3038-four-endpoint-pilot/`，二进制 SHA
见每个运行 JSON。Point 占内部 E2E 约 95.6%。与 `563d78a` 的 6,326 条相比只少
2 条边，non-pair 多 385；相对论文 5,548 条仍多 776 条。
因此不能声称整体研究已加速，不能把提高 occupancy 当作补齐数学强度的替代。
此处只有单次 pilot，且不是最新版 4-CTA 计时，更不是完全图端到端结果。

4-CTA 在同一 GPU 的后续全量 pilot 为 **621.643 s 进程 wall / 621.528 s 内部
E2E**，终态仍为 6,324 / 427 / 1,789、hash `ae40f6bb8a9aa459`；Point 为
581.538 s。两次 pilot 的 wall 比为 1.442x，但不是三次配对中位数。该次也早于
5 节点修复，记录位于 `artifacts/strength-complete/pcb3038-final-cta4-pilot/`；目录
中的 `final` 是启动时的实验标签，不代表后续发现错误后仍可称为最终版。

## 新的全图环边界回归

新增的 6 节点全最优 tour 测试揭示：在 shared endpoint 与 fixed 连接同时存在时，
旧 `Opt` normalization 会将覆盖全部顶点的 Hamilton 环判为真子环。完整端点 OR
使该问题暴露，事务校验以 code=8（无合法邻边对）阻止了 live 更新。

修复针对被关闭分量精确检查去重后的顶点覆盖：仅真子环直接拒绝；完整环保守保留。
大实例先通过节点数上界 O(1) 排除，不为每次局部叶扫描整图。新增 CPU 定向回归，
以及全部最优 tour、3 个叶后端 × 2 个 CTA 策略的端到端门禁；后续补充后共 7 组小图。

## 5 节点快速过滤与测试 oracle 修复

额外小图 `(0,5),(9,3),(5,2),(5,11),(1,5)` 有两条成本同为 27 的最优 tour，
最优边并集为 7 条、必选边交集为 3 条。旧 Quick-HS 在 LP 的正确 7-edge 图上
会错误排除 `(0,3)`：`2+3+3` 的端点收缩形成三角环，但展开后恰好覆盖全部
5 个节点。GPU proposer 与 replay 共用了这个错误 fast filter，因此均会通过。

新增 `HasProperCycle233`，在端点闭环门禁中核对完整路径的顶点覆盖；同一闭环
在 5 节点图上保守保留，在 6 节点图上仍作为真子环关闭。CPU 定向测试同时覆盖
`ReplyAdmitsTour` 与 `CanEliminateWithWitness`，正式多后端测试新增该实例。

随后完整链发现 metric-excess 也无条件排除了 `z` 连接四节点路径两端的回复。
它在五节点图上同样可能是完整最优环。现已补齐 metric-excess 的五节点、strong
close-point 的四节点，以及 `Opt23` 的三节点完整环边界；并非关闭整个小图服务。
独立 C++ 性质测试枚举 36 个实例的所有最优 tour，在完全图、最优边并集、含额外
边的中间图三种快照上逐项检查 Main allowed-pair 与 Quick/Main edge 规则，108
种快照全部通过。其成本矩阵和最优解枚举不调用被测证明谓词。

同时发现旧 Python oracle 的 `union = edges; intersection = edges` 让两者共享
集合；`intersection_update` 会反过来缩小并集。这不是求解器功能，而是测试本身
的缺陷，意味着之前的“全部最优边并集”检查不充分。已改为独立集合，并新增
oracle 自测：上述两最优解图，以及 4 节点所有 tour 成本相等的图。

发现此问题后未提交已知失败的版本；中断了当时尚未结束的完整 overlap racecheck，
其日志只有启动头，**不计为通过**。后续检查必须绑定修复后的二进制。

## 正确性门禁

- 最新 CPU Release：35/35；修复后的 CUDA Release（sm_89）：62/62，全部通过，
  包含 `differential.pr299_cpu_cuda`。之前的 60/60 不包含新暴露的五节点边界，
  不用它代替本次最终回归。
- 新增小图门禁现包含 7 组 5–8 节点图；3 种叶后端 × 2 种 CTA 策略共 42 次
  完整 solve 已通过修复后的 oracle，检查最优边并集、必选边交集、最优邻边对
  并集及完整终态一致性，没有只检查给定的一条 tour。
- 额外开发检查：36 次 3–8 节点完整 GPU solve，包含 EUC_2D、CEIL_2D、重复
  坐标和并列最优解，全部通过。结果在
  `artifacts/strength-complete/additional-small-corrected/summary.json`。上述 108
  种 CPU 快照性质检查已加入正式 CTest；额外 36 次 GPU 运行是独立的开发记录。
- 事务单元：8 组 GPU 候选终态；SEC replay：11 组成员／incidence 篡改门禁。
- PDHG：8 组可独立求解的小 LP、warm start、非 64 整倍数迭代、非法 CSR/CSC
  与活动列集合、错误后复用恢复；最新构建的专用短测试 memcheck/racecheck 均为 0。
- 最新构建的核心单元、GPU 事务、SEC replay，均通过未过滤的 memcheck/racecheck。
  正式 overlap、6 节点 full-cycle、5 节点 tied-five 三个完整求解均通过未过滤的
  memcheck，以及仅覆盖 Point/Quick/Main/Fix/Nonpair/SEC replay/事务提交内核的
  racecheck（0 errors、0 warnings）。后者**不是全内核 racecheck**；PDHG 迭代
  内核的竞态检查由上述专用单元覆盖。求解范围没有因 sanitizer 而截断。
  日志为 `artifacts/strength-complete/sanitizer/cycle-corrected-*.log`，sanitizer 的
  wall 不参与性能统计；之前中断和失败的开发记录保留，不混入通过记录。

本节最终 GPU 构建及冻结副本 `.tmp/strength-complete-recovery/fgpu-cycle-corrected`
的 executable SHA-256 为
`37122d8aa8eb1029495bb1b533994bc6e0f00b7b7c1de1f686803a931c9f7778`，源码树
SHA-256 为 `004fce04d293e01773c040a90d68fc40e0ed6782326d9265a283e7257d64e882`。
构建环境为 GNU 16.1.1、CUDA 13.3.33、sm_89、Release，构建时 HEAD 为
`563d78aa1de25dc41985cbfc0e205c41ae1e7cba` 且 `git_dirty=true`。

这些测试支持当前已实现规则的回归可信性，不是任意规模、所有距离类型、所有
未实现 HT 形态的形式化正确性证明。本次未重新验证其他 CUDA 实码架构。

## 尚需完成的主计划

1. 将现有 SEC 适配器换为长期、可增长、通用整数系数 cut pool，并实现基于真实
   primal 的 mincut SEC、local chunk separation/lifting、简单 comb/blossom。
2. 完整 Direct/Close Point 与全度数 metric-excess，解决路径重叠的通用 normalization。
3. 通用 F/A state arena、lazy OR/windowed AND、generation cancellation/transposition，
   以及声明域内 m≤5、深度 6 的搜索。
4. multi-output inside matching/path-cover 与独立 GPU traceback；拆分 root/reply/leaf。
5. GPU 完整图流式构造、完整反向依赖、persistent ready queue、设备端收敛判定。
6. 在小图全最优解门禁通过后，完成 `pr299`、`pcb3038` 同输入/完全图的重复端到端
   对照，再报告论文强度差与墙钟比。

上述是尚未实现的能力，不是可以通过增加外层轮数或使用更多显存自动补齐的配置项。
