# 实现状态（2026-09-02）

| 工作包 | 状态 | 已验证证据 |
|---|---|---|
| M0 仓库与复现 | 完成 | 项目内依赖/构建目录、固定子模块、CPU CI、路径门禁 |
| M1 JV CPU/CUDA 闭环 | 完成（首期 JV 范围） | n=6–12 全最优边检查；pr299 CPU/GPU 哈希一致；proof 重放；compute-sanitizer 0 error |
| M2 cuOpt sidecar | 完成（模型内证书） | cuOpt C API 26.8.0；手算小 LP objective=1；残差为 0；精确下界=1 |
| M3 Concorde 导出 | 完成 | 内容寻址受限 overlay；Concorde graph 目标映射复核；随机 20 点和 pr299 CSR 往返 |
| M3 完整图 exact pricing | 完成（安全下界桥接） | `CCbigguy` 注入；完整图负 reduced-cost penalty；三方哈希；错配拒绝 |
| M3.1 对偶稳定化与边集导出 | 待实现 | pr299 PDLP 完整图界偏弱；尚未导出每边 exact RC/Concorde 消元后边集 |
| M4.1 path-system 组合层 | 完成 | 路径规范化；固定哈希表；368,047 单元 CPU/CUDA 全量差分；`m=6,7` CPU fallback |
| M4.2a CPU k-opt 叶证明 | 完成 | proper 3/4/5-opt `4/25/208` 模板；ElimTSP oracle 差分；`path-kopt-proof-v1` 独立重放 |
| M4.2b CUDA k-opt cost | 完成（候选器） | 批量精确成本矩阵；CPU/CUDA 单元一致；坏/漏候选 CPU 全模板兜底；memcheck 0 error |
| M4.3a 精确困难叶 | 完成（有界 CPU fallback） | 收缩 forced outside matching；Held–Karp 子集 DP；通用交换 witness 独立重放；block 超限为 unresolved |
| M4.3b1 浅层 HS AND–OR | 完成（研究 API） | `c,d` OR；完整邻边对 AND；嵌套 leaf 重放；CUDA flags 经 CPU 全量差分 |
| M4.3b2 递归 HT 语义与证书 | 完成（CPU 研究 API/CLI） | extra point/end；continuation arena；全局 proof V1；`ht-prove`/`ht-verify` 严格重放 |
| M4.3b3a 混合 GPU wavefront | 完成（研究 API/CLI） | 主机 BFS；CUDA 原子 continuation counters；单-block device-persistent queue；CPU 全状态差分 |
| M4.3b3b1 GPU path append | 完成（候选器） | point/end 状态内合批；稀疏分量/度数 kernel；CPU 规范子状态逐项认证 |
| M4.3b3b2 全设备 wavefront 与提交 | 待实现 | GPU reply/状态写出、批量叶、多 block/CPU long-tail、epoch commit |
| M5 中大型调优 | 待开始 | 首期不设最低加速比；pcb3038 尚未形成认证运行记录 |

## 当前基准结果

pr299 输入 1208 条边；JV 两个 epoch 后保留 1122 条，提交 86 条删除。CPU 与 CUDA 的最终内容哈希均为 `b9b67e9981518177`。这些数字是正确性回归结果，不构成论文性能结论。

cuOpt 手算 LP：状态 `OPTIMAL`，objective/dual objective 均为 `1`，primal violation 与 reduced-cost residual 均为 `0`，定点模型下界为 `16777216/16777216`。

Concorde 随机 20 点 epoch：25 行、43 列；QSopt 与 cuOpt 模型目标均为 `88`。cuOpt primal violation 为 `4.44e-15`，reduced-cost residual 为 `1.57e-14`；完整图 exact lower bound 为 `87.3932819641`，上界为 `88`。

pr299 Concorde epoch：454 行、888 列、8561 个非零元；cuOpt 状态 `OPTIMAL`，模型目标 `48187.777777780764`，primal violation `8.37e-11`，reduced-cost residual `4.87e-10`。完整图 exact lower bound 为 `43977.2693797`，合法但较弱；这说明后续需要对偶稳定化/迭代补列，而不是跳过负 reduced-cost penalty。

路径兼容表：`m=5` 为 45,360 字节，生成器哈希 `f6bccacc5c1fa84f`；362,880 个 `m=5` 单元和全部较小表均通过 CPU/CUDA 差分，CUDA memcheck 为 0 error。`m=6,7` 不建立完整表，按契约使用 CPU 直接判定。

CPU k-opt 叶证明：自动生成的 proper 3/4/5-opt 模板数为 `4/25/208`，每阶 2,000 个阈值定向随机矩阵与固定子模块 `swap.c` oracle 一致；成功 witness、全 outside coverage、预算 unresolved 和篡改拒绝均有回归。

CUDA k-opt cost：按删除集合与 proper template 形成精确成本矩阵，CPU/CUDA 逐单元一致；候选成功和 GPU 无命中后的 CPU completeness fallback 均通过回归，CUDA memcheck 为 0 error。当前尚未以真实 HS 任务报告加速比。

CPU 精确困难叶：将每条 forced outside edge 收缩为可双向访问的 block，其余节点为 singleton block；固定一个 block 的方向消除无向反转对称后，以 Held–Karp 子集 DP 穷举所有 block 次序和方向。比较时消去两条巡回共有的 outside 成本，并禁止候选重新使用 required path edge。成功结果转换为任意 `k>=2` 的交换 witness，再交给同一独立 verifier；默认关闭，启用时最多 18 个 block，内存不足或超限均返回 `unresolved`。60 组随机 7 点、每组两个 outside 的结果与直接 Hamilton 巡回枚举最优值一致，并覆盖 7-opt proof 往返。CPU Debug/ASan、CPU Release 与 CUDA Release 全量回归通过，GPU 2 上 k-opt memcheck 为 0 error。

浅层 HT：固定 `c,d` move 后，重新枚举两个中心所有通过严格 2/3-opt 快速筛选的邻边对，并要求其笛卡尔积中的每个 reply 都有 path-infeasibility 或完整 path-system leaf proof。固定 8 点实例覆盖 30 个非空 replies；删除 reply、篡改 move/snapshot/leaf 均被 verifier 拒绝。CUDA 在稀疏 32 点 EUC/CEIL 批次上输出与 CPU 完全一致的 `c,d` flags，memcheck 为 0 error。

递归 HT：CPU DFS 实现与后续 wavefront 相同的 `Leaf(F) OR ∨move ∧reply HT(F∪reply)` 真值。未解决状态可选择未出现在路径系统中的 point，或选择当前路径 endpoint；每个 move 必须记录其完整活动图 replies。成功子树保存为只向后引用的扁平 continuation arena，独立 verifier 从目标边重新规范化每个子状态并拒绝环、共享 child、遗漏 reply 和未引用节点。`CUDAEE_HT_RECURSIVE_PROOF_V1` 嵌入现有 path-k-opt V1 叶证明并严格拒绝非法计数、枚举值和尾随字段。固定 point/end 递归实例均由 8 点完整巡回穷举额外确认目标边不属于任何最优巡回；depth/budget fail-closed 与 proof 篡改已有回归。`ht-prove`/`ht-verify` 已覆盖固定实例的文件级端到端 CTest；未解决写入 `proven=0` 并返回退出码 3。DFS 调度本身仍在 CPU，GPU 版本使用下一段的独立 wavefront 路径。

混合 wavefront：主机 BFS 为每个状态先跑 leaf，再生成有界 point/end OR moves 及其完整 replies；所有 child 只指向下一层。CPU 从后向前得到规范状态真值；CUDA 从 leaf/vacuous/无 move 终态队列开始，每个完成 child 原子更新 move 的 `remaining_children/failed`，成功 move 立即完成父状态，失败 move 递减父状态的 `remaining_moves`，最后一个失败 move 才宣告 OR 失败。单-block persistent kernel 在设备端冻结并消费动态队列批次，要求每个状态恰好入队一次；队列溢出或提前停滞均失败。CUDA/CPU 任一状态不同即返回 `invalid`。成功时仅复制第一个成功 move 的子树到既有 continuation arena，再运行完整 proof verifier。

GPU path append：规范父路径展平为 `(node,component,degree)`，一个线程检查一个 point/end task。point 中心必须是新节点；两个连接点度数均小于 2 且不能来自同一分量。end 必须从现有端点出发，另一端只能是新节点或其他分量端点。同一父状态的全部 point 候选合为一个 batch、全部 end 候选合为另一个 batch；CPU 对每项仍执行 `NormalizePathSystem`，只有 flags 全等才使用 CPU 生成的规范 child。固定双父状态 11-task 表覆盖合并、成环、内部节点、重复边与新节点；实际 point CLI 运行生成 34 states、18 moves、84 replies，峰值 frontier 27，append 为 9 batches/84 tasks，最终压缩为 4 节点证明。这些是正确性样例，不是性能结论。

## 安全边界

`gpu-eliminate` 目前只实现 JV quick candidate search；path-system leaf 和递归 HT proof 尚未连接全设备 wavefront/epoch commit，因此也不授权删边；`lp-solve` 始终不修改图。Concorde 桥接已能产生完整图安全下界，但测试 wrapper 使用 `-B`，尚不输出消元边集。M4.3b3b2 与 M3.1 必须标为 pending；仍严禁从未完整验证的局部结果或 cuOpt 浮点 reduced cost 直接构造删除记录。
