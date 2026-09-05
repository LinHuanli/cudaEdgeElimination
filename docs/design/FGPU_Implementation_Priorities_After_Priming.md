# 融合方案后续实施计划：先消除笛卡尔积浪费，再扩大有效证明域

日期：2026-09-06；代码基线 `a5b99c1`，继续使用分支
`research/fgpu-hybrid-beat-hs2014`。本计划细化[冻结验收协议](FGPU_Hybrid_Beat_HS2014_Execution.md)，
不降低四实例的强度或速度门槛，不把新计划写成已完成结果。

## 1. 已有证据与决策

| 完整图、无标签 pilot | 最终边数 | 2014 边数 | 差额 | 本方法 wall |
| --- | ---: | ---: | ---: | ---: |
| pr1002 | 5002 | 4521 | +481 | 1161.415 s |
| pcb1173 | 6493 | 6084 | +409 | 1902.547 s |

两例都通过已知最优 tour 的独立 edge/fix/nonpair 事后检查，GPU replay 拒绝为零；
不是独占节点三次中位数。pr1002 的 2.735× 是相对上个全度数版本，同终态，
不是相对论文的等强度加速。严格胜出还需分别至少删 482、410 条。
pcb1173 的 GPU UB 为 59,387，LP 下界约 56,158.7，标签 56,892 仅用于事后解释。

vm1084 的 L4 运行曾超过 81 分钟仍在首轮 Quick-HS；本机开启预热的一次 LP/Point
阶段又用了约 540.876 s，产生累计 1,431,322 个 nonpair 提案后仍进入昂贵 Quick。
两个硬件的时间不能直接比较。pcb3038 也已进入首轮 Quick，不能拿中间边数当终态。

源码定位：`RepliesClosedCta` 仍枚举
`C(deg(c),2) × C(deg(d),2)`，之后才执行已有 nonpair 和 root 兼容性过滤。
一项已在 c 侧被精确排除的 pair 因而仍对应整行 d 侧窗口。
`CountSurvivingPairsCta` 没有保存紧凑回复流，正式无 pair-trial 上限时连排序计数也不执行。
这是明确的数据展开问题；新一轮先解决它，为深搜及强 LP 提供可运行的基础。

## 2. 工作顺序与交付边界

### A. 完整 Quick/extra-edge 回复压缩（立即实施）

1. 独立 GPU kernel 生成每个 root 的原有候选中心，保持相同几何域和规范顺序。
2. 按 `(snapshot,root,center)` 统计并写出所有允许邻边对的稳定 ordinal；使用 GPU
   prefix scan 和实际总数分配工作区，不设每中心固定回复容量，不发布为全局 nonpair。
3. proposer 只枚举两个紧凑回复流的乘积。过滤只使用原谓词的必要条件，不新增未经
   验证的删边规则；空流是完整 AND 的空集，不是截断、异常或 allocation failure。
4. 单独 GPU 验证 kernel 检查每个条目的范围、严格递增、合法性及完整计数。
   验证计数使用自然邻接双循环与独立的 Opt22/Opt23 表达，不消费 proposer 的计数结论。
   当且仅当全部覆盖检查通过，replay 才可消费这份不可变流。
5. replay 只检查选中的 c,d，但仍覆盖其完整回复流；缓存、epoch、root 或中心身份
   不符即安全失败，不能以资源不足返回“全部关闭”。
6. 首个版本只改变物理展开，不改候选域、OR 顺序、局部叶判定、外层固定点规则。
   保留默认关闭的实验开关做同二进制对照；不因一例快就直接设成默认。

新增指标至少包括：实际压缩前/后 pair 条目、工作区峰值、构建和覆盖验证耗时，
Quick/extra-edge/replay 的 wall。计数区分容量、逻辑全集与短路后实际访问量；
不能把理论笛卡尔积缩减比写成端到端加速比。

正确性门禁：随机非度量/零成本/大整数及 EUC/CEIL 图，对独立 CPU 枚举；
已有 nonpair、fixed、inactive 稳定边 ID、零回复、重复 ordinal、漏条目和 snapshot
错配；n≤10 全最优 tour；开关最终 edge/fix/nonpair 文件对照。
先测试再做 pr299 全量 A/B，之后以同一配置跑四例，不引入目标数、时间或 epoch 截断。

### B. 动态通用 LP cuts 与收敛（下一主强度工作）

现有 `ResidentSecModelCuda` 只是有限 SEC 到 CSR/CSC 的适配器，不能直接称为完整 LP。
拆出 `GpuCutPool`，稳定 cut ID、规范集合键、整数 RHS/系数、动态 row offsets；
全部 active columns，不使用固定 incidence stride。删边后只更新活动列与模型版本，
本 dual 快照引用的 cuts 不退休、不改义，warm start 通过稳定 ID 保留。

实现顺序为：

1. 通用整数 cut row + GPU 独立结构验证 + Signed128 box bound/reduced-cost 重算；
   单元测试先覆盖负 RHS、负系数、量化范围、重复/缺失列和不合法 multiplier。
2. 在 PDHG **原始解**上分离阈值连通分量，再加 batched source–sink push-relabel
   mincut；任何候选集合重新计算边界，只发布合法非空真子集的 SEC。
   SEC 对应边界和至少为 2，参见 [Concorde 的 subtour 说明](https://www.math.uwaterloo.ca/tsp/methods/opt/subtour.htm)。
3. 奇边界 2-matching：保存 handle 与奇数条边界边 F，验证 F 为边界子集且不重复，
   使用整数行 `x(delta(S)\\F) - x(F) >= 1-|F|`；独立枚举小图全部 tour 验证。
   不把这种已验证的行称为完整 comb/local-cut 分离器。
4. PDHG 按批运行，最终模型可行性与相对 gap 均 ≤1e-5；批大小不是总迭代上限。
   增加 last/averaged iterate 诊断与可靠重启，避免靠无限延长含早期误差的平均值。
   LP 数值失败明确保留边/失败，不能把未收敛状态写为完成。
5. 在新增 cut 的固定点上发布独立 GPU 验证过的量化 dual；edge/fix/path 共用它。
   比较 LP off / SEC / SEC+2matching，含 GPU UB 与全部 LP/replay 成本。

UB 仍不得读最优标签。先记录当前 GPU tour 与 LP bound 的差距，再评估增加 GPU
局部搜索是否比增加 cuts 更省端到端时间；标签只解释实验，不决定运行中的停止或参数。

### C. 多输出 path-cover 与统一深层 HT

先实现供多个 outside orderings 共享的精确 path-cover：一次局部整数距离矩阵，
共享内部路径成本、inside matching 输出和完整 coverage。GPU traceback 给出改进
见证，replay 检查成功见证及 AND 覆盖，不重复 proposal 搜索。
CPU brute force 独立验证成本、严格改善、coverage、traceback；零成本必须保持开放。

随后统一 edge/nonpair/fixing root。规范化共享端点、固定连接、真子环与全图环；
声明域推进到 ≤6 paths / 10 reveal，所有可延伸端点及完整选中 Hamilton replies，
added-point 在统一空间邻域按 surviving replies 排名选 4 个，不按实例名特调。
没有 workspace 的状态进入可增长队列或明确失败，绝不据此宣称关闭。

### D. 队列和精确依赖

在 A/C 的 root、reply、leaf、replay 边界上拆分设备队列：形状分桶、lazy OR、
windowed AND、generation cancellation、精确 snapshot transposition。
把当前 dirty 半径升级为精确反向依赖；结束前仍完整 service sweep。
每项用逻辑工作量、物理工作量、排队时间、显存峰值及 E2E 归因，不以利用率替代性能。

## 3. 实验与发布纪律

- 所有文件、依赖、冻结二进制和日志留在本项目目录；外部 references/Datasets 只读。
- 当前长实验不删除、不修改其二进制、不把它们的中间结果填成最终数据；新旧记录分目录。
- 开发 pilot 可用空闲计算时段；GPU 已分配显存不等于可独占，保留整段干扰记录。
- 每个实现检查点须 CPU 全套、CUDA 差分、memcheck/racecheck。测试范围明确记载；
  性能改动先要求相同终态，扩展方法域则要求所有最优解保留且强度不回退。
- 四例正式验收共用一套配置、同一二进制；预热一次，CPU/GPU 交错三次 clean 测量。
  历史论文时间比与同机单核速度比分列。最终边数和双速度全部过线才宣布完成。

本轮首先交付 A 的实现与测试；B/C/D 保持显式未完成，不能用 A 的局部提速代替
“剩余边少于 2014”的最终目标。
