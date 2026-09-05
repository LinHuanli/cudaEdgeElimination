# 实现状态（2026-09-06）

最新开发记录见 [Quick 精确回复压缩](76_FGPU_Exact_Quick_Reply_Compaction.md)，
后续顺序见[细化实施计划](../design/FGPU_Implementation_Priorities_After_Priming.md)。
新增默认关闭的完整 reply CSR，Quick/extra-edge proposer 和 replay 共用经过独立
GPU 完整覆盖检查的同快照流，不改变方法域、不截断回复。pr299 同 L4、同二进制
单次 pilot 为 **181.994→103.764 s（1.754×）**，edge/fix/nonpair 文件逐字节相同，
剩余边仍 1,246；CPU 37/37、CUDA 69/69，专项和 288 次 GPU 状态检查的
memcheck/racecheck 均零错误。四例试跑的 vm1084 在剩余约 8 GB 的 L4
安全报显存不足；转到剩余约 33 GB 的繁忙 L40S，成功分配约 10.7 GB 进程工作区，
继续完整运行，不算 clean 速度证据。

上一版 pcb1173 已完整结束：687,378→6,493 边、1,902.547 s；2014 为 6,084，
仍多 409 条。原生 LP cuts/收敛、6-path/10-reveal 与设备队列仍未完成，不能用
pr299 的执行加速冒充四例强度胜出。

此前开发记录见 [Point 近点预热与强度缺口](75_FGPU_Point_Priming_and_Strength_Gap.md)。
新增默认关闭的近点预热、完整收尾回扫及计数；修复中间快照首轮 inactive CSR 槽位。
CPU 37/37、CUDA 68/68（234 次无标签完整求解、12 次密图调度回归）通过；新增
36 组全最优 tour oracle 下的 144 次 GPU 状态检查，实际覆盖 293 个预热提案。
预热单次 `pr299` wall 为 174.62→170.09 s，剩余边仍 1,246，nonpair 终态不同。
本机 pcb1173 已启动，随后同卡运行 pcb3038；vm1084 在远程 L4 已观察到超过 81 分钟
仍停留首轮 Quick-HS 的长尾，不声称四例都有加速。

上一轮见 [Point 复用与真实 frontier 调度](74_FGPU_Point_Reuse_and_Frontier_Scheduling.md)。
新增 GPU 全排列目录、完整近点优先回退、真实邻边对 frontier 计数与延后调度；
修复大成本路径的强制连接代价界。CPU 37/37、CUDA 68/68（含 208 次小图完整求解）
通过。完整 `pr299` 的单次同二进制调度对照为 200.95→175.08 s，剩余边同为 1,246；
nonpair 终态不同，最优 tour 检查均通过，尚不是独占节点三次中位数。
完整 `pr1002` 已完成：501,501→5,002 条，3,176.233→1,161.415 s，终态一致；
仍比 2014 多 481 条。远程 L4 的 `vm1084` 仍在运行，硬件计时分开。四例验收未完成。

上个检查点见 [无标签完整图与全度数 metric](73_FGPU_Hybrid_Bootstrap_and_FullMetric.md)。
新增 GPU 完整图/距离/上界、半整数距离、LP-off pair/fixing、条件 pair 缓存和全度数
metric；CPU 37/37、CUDA 67/67。旧低度 metric 的完整 `pr1002` pilot 为
501501→5619 边、2778.532 s，仍比 2014 多 1098 条；全度数版本的新结果见上段。
`berlin52` 加强后 195→164 边，但该次 wall 从 10.18 增至 17.56 s。四实例目标、
通用动态 cuts 和 6-path/10-reveal continuation 尚未完成。以下均为较早检查点，
不得混用版本计时。

最新开发结果见 [快照事务、稀疏 PDHG 与完整端点 OR](71_FGPU_Transaction_PDHG_and_PathEnd.md)。
本次已接入原生稀疏 PDHG、SEC 独立成员验证、同快照事务和四端点 OR；完整
local-cut/comb LP、通用深层 HT 与 GPU 工作队列仍未完成。修复后的最终构建
`pr299` 同卡三次中位进程 wall：旧版 10.162 s，当前 4-CTA 21.011 s；non-pair
从 298 增至 387，剩余边同为 828。当前 4-CTA 相对当前 2-CTA 为 1.312x，但
相对旧版仍慢 2.068 倍。开发中额外小图暴露五节点全图环和测试 oracle 集合别名
问题，已修复；最新 CPU
35/35、CUDA 62/62 均通过，包含修复后的 42 次多后端全最优解求解；额外 36 次
小图完整 GPU 求解也通过。最新 sanitizer 的覆盖范围和零错误记录见新报告，
不能用之前 60/60 或中断的 racecheck 代替最终回归。
最新 `pcb3038` 完整 LP 输入单次 pilot：`6,883 -> 6,324`，427 fixed、1,789
non-pairs；进程 wall 585.679 s，内部 E2E 585.561 s，Point 占 92.79%。这不是
完全图输入基准，也未达到论文的 5,548-edge 强度门禁。
下文已有历史基准不应当成最新源码计时；构建／输入身份和原始日志按新报告核对。

| 工作包 | 状态 | 已验证证据 |
|---|---|---|
| M0 仓库与复现 | 完成 | 项目内依赖/构建目录、固定子模块、CPU CI、路径门禁 |
| M1 JV CPU/CUDA 闭环 | 完成（首期 JV 范围） | n=6–12 全最优边检查；pr299 CPU/GPU 哈希一致；proof 重放；compute-sanitizer 0 error |
| M2 cuOpt sidecar | 完成（模型内证书） | cuOpt C API 26.8.0；手算小 LP objective=1；残差为 0；精确下界=1 |
| M3 Concorde 导出 | 完成 | 内容寻址受限 overlay；Concorde graph 目标映射复核；随机 20 点和 pr299 CSR 往返 |
| M3 完整图 exact pricing | 完成（安全下界桥接） | `CCbigguy` 注入；完整图负 reduced-cost penalty；三方哈希；错配拒绝 |
| M3.1 对偶稳定化与边集导出 | 待实现 | pr299 PDLP 完整图界偏弱；尚未导出每边 exact RC/Concorde 消元后边集 |
| FGPU one-shot CLI | 完成（单 GPU、安全闭环） | geometry→LP-box→JV→可选 HT；五类输出；最终从初图重放；pcb442 单命令到 3,239 边固定点，pcb442/pr1002 最优 tour 零缺边 |
| FGPU native LP 删除 | 完成（degree-only） | CUDA multiplier；`2^24` 定点量化；完整 live-variable `__int128` box bound；V4 sidecar 与篡改拒绝 |
| FGPU fully-resident 终态 | 部分完成 | GPU 候选/成本/传播已接入；最终 exact replay、HT 控制与完全图物化仍含 CPU；subtour cuts/tile certificate 待实现 |
| FGPU resident raw 固定点 | 完成（degree-LP/local raw） | 默认无 CPU audit/证书/epoch 上限/节点上限；pcb3038 七次 clean 中位 56.33 s，4,613,203 → 23,720，七份 SHA-256 一致；强度仍不等价论文完整 LP |
| FGPU strength-upgrade `solve` | 部分完成（P0–P8 核心子集） | 单 GPU sparse resident；Signed128 LP；connectivity/local SEC；Main/metric；KH `-e2`；point non-pair；Direct Fix；联合固定点；安全修复后 pcb3038 论文 LP 图 `6883 -> 6326` |
| FGPU GPU-safe trust boundary | 当前实现回归通过 | 无 CPU 逐边 audit；同不可变快照 replay 后联合事务验证；独立 SEC 成员／incidence 检查；CPU 35/35、CUDA 62/62；sanitizer 的精确覆盖范围见新报告 |
| FGPU native sparse PDHG | 部分完成 | CSR/CSC、warm start、CUDA Graph 批次、量化精确 bound；当前适配器仍是有限 SEC，尚无真正 primal mincut 与通用 cut pool |
| FGPU Point 完整端点 OR | 完成（不相交 3+3 根形态） | 四端点全部尝试、端点内完整 AND、独立 replay；共享端点全图环回归修复；3 叶后端 × 2 CTA × 7 小图全最优检查；通用 overlap/HT 仍缺 |
| FGPU non-pair/fixing 固定点 | 部分完成 | LP/fixed-anchor/完整一层 point non-pair、non-pair-implied fixing、degree-2 propagation；深度 2–4 pair HT 与一般 fixing HT 尚缺 |
| M4.1 path-system 组合层 | 完成 | 路径规范化；固定哈希表；`m<=6` CPU/CUDA 差分；`m=6` 表为 4,989,600 bytes，`m=7` CPU fallback |
| M4.2a CPU k-opt 叶证明 | 完成 | proper 3/4/5-opt `4/25/208` 模板；ElimTSP oracle 差分；`path-kopt-proof-v1` 独立重放 |
| M4.2b CUDA k-opt cost | 完成（CPU 认证候选器） | public 完整矩阵逐 cell 差分；broker 使用 `1/1/4` words candidate mask；改善 witness CPU 完整重建；memcheck 0 error |
| M4.3a 精确困难叶 | 完成（有界 CPU fallback） | 收缩 forced outside matching；Held–Karp 子集 DP；通用交换 witness 独立重放；block 超限为 unresolved |
| V3 A0 短路 Trace/replay | 完成 | V1 严格文件格式；AND/OR 规范次序；1/2/4/8/all replay；部分/篡改 Trace 拒绝 |
| V3 A1 紧凑/CUDA exact DP | 完成（CUDA 候选器） | CPU 相邻 popcount 层；成功才 traceback；CUDA `k<=13`；逐值差分；阳性 CPU witness；阴性 unresolved |
| V3 C1 转置短路调度 | 部分完成（leaf broker 已落地） | heterogeneous required-edge 合批；单 GPU dispatcher；两请求微批；d15112 18 proofs；SoA continuation/generation 待实现 |
| V3 C1.5 验证热路径 | 部分完成 | scan 只在 commit 前完整重放一次；sidecar 并行只读验证；尚未引入显式 token 类型 |
| M4.3b1 浅层 HS AND–OR | 完成（研究 API） | `c,d` OR；完整邻边对 AND；嵌套 leaf 重放；CUDA flags 经 CPU 全量差分 |
| M4.3b2 递归 HT 语义与证书 | 完成（CPU 研究 API/CLI） | extra point/end；continuation arena；全局 proof V1；`ht-prove`/`ht-verify` 严格重放 |
| M4.3b3a 混合 GPU wavefront | 完成（研究 API/CLI） | 主机 BFS；CUDA 原子 continuation counters；single/cooperative persistent queue；CPU 全状态差分 |
| M4.3b3b1 GPU path append | 完成（候选器） | point/end 状态内合批；稀疏分量/度数 kernel；CPU 规范子状态逐项认证 |
| M4.3b3b2a1 GPU Hamilton replies | 完成（候选器） | 多中心 count/write；确定性 CSR 区间；CPU 完整列表逐元素认证 |
| M4.3b3b2a2 GPU end replies | 完成（候选器） | 多端点 count/write；空区间/重复 task；CPU 完整边列表认证 |
| M4.3b3b2b1 frontier reply batching | 完成（调度基线） | 可配置 chunk；跨父状态 spans；单状态/批量 V1 proof 逐字节一致 |
| M4.3b3b2b2a frontier path append | 完成（调度基线） | 多父状态稀疏 spans；point-first end 筛选；CPU 规范 child 全量认证 |
| M4.3b3b2b2b1 规范 child edge SoA | 完成（候选器） | CUDA count/write；不可行空 slice；CPU offsets/edges 全数组认证 |
| M4.3b3b2b2b2a frontier leaf batching | 完成（规则首行） | 确定性复杂度桶；跨 leaf cost rows；CPU witness/proof 复核 |
| M4.3b3b2b2b2b1 一般 leaf 游标 | 完成（增量融合） | 分段组合 cursor；3/4/5-opt；预算边界与随机 proof 字节差分 |
| M4.3b3b2b2b2b2a GPU leaf 驻留缓存 | 完成（线程/设备本地） | 精确坐标/模板键；增长型 workspace；命中与字节指标 |
| M4.3b3b2b2b2b2b1 CPU long-tail | 完成（128-cell 基线） | 缓存后交叉点；融合矩阵分流；CPU/CUDA proof 规范计数 |
| M4.3b3b2b2b2b2b2 multi-block continuation | 完成（cooperative 基线） | grid barrier；residency 门禁；512-way AND 跨 block 差分 |
| M4.3b3b2b2b2c HT epoch commit | 完成 | 整批 CPU 重放；V5 zlib/CRC32 内嵌 sidecar；8 GiB raw/448 MiB compressed/512 MiB file 有界门禁；旧 V1–V4 兼容 |
| M5 有界全图 HT scan | 完成（单快照 pilot） | 稳定目标切片；预算 unresolved；V2 阶段计时；三路工作签名；V2 原子提交；最优 tour 门禁 |
| M5 JV—HT 多 epoch 编排 | 完成（有界调度基线） | JV 固定点；无提交 sweep 推进；提交后重排；联合 V2 重放；pcb3038 CPU/CUDA/tour 门禁 |
| M5 JV 活动 edge-id 紧凑启动 | 已评测并排除 | d15112 启动行 `-2.925%`，kernel `+1.272%`，算法总时间反而回退 `1.052%`；原型已撤销 |
| M5 HT 跨目标根 `c,d` 候选融合 | 已评测并排除 | d15112 七对 A/B 的 candidate/search/total/wall 均回退；14 份 proof 重放、边集和 tour 门禁通过；原型已撤销 |
| M5 HT reply CUDA 驻留共享 | 完成（静态资源/工作区） | 完整图键；Hamilton/end 共用；d15112 七对 A/B；14 份 proof/tour 门禁；memcheck 0 error |
| M5 HT reply 精确任务去重 | 完成（batch-local） | 物理 tasks `28497 -> 418`；CPU 完整逻辑展开；V16/V19；d15112 七对 A/B 与 14 份 proof/tour 门禁 |
| M5 HT 跨 batch reply 结果缓存 | 已画像并排除 | 最多 `418 -> 282`，但 23/23 batches 仍含新 key、可消除调用为 0；observer 已移除 |
| M5 HT 目标级多 GPU | 完成（静态切片基线） | 固定设备 worker；顺序 CPU proof/原子提交；双 A4000 32-target `1.251×` target execution；V17/V20 |
| V3 单 GPU target workers | 完成 | 多 worker 共享单一显式 ordinal；原子动态领取与规范顺序消费；多设备保留确定性静态分片；worker 上限 32 |
| M5 中大型调优 | 进行中（JV 三轮 + HT host/device/multi-GPU fast paths） | point/path/reply/leaf/scan-binding fast paths；reply 驻留、任务去重与目标级静态多 GPU |

## 当前基准结果

FGPU strength-upgrade 同论文 LP 输入基准：修复 `-e2` 重叠路径的安全问题后，
`pcb3038` 从 6,883 条到 6,326 条，424 fixed、1,404 non-pairs（6.1687%），
2,836 次 device replay、0 rejected，最优 tour 零缺边。三次隔离 clean E2E 中位为
153.220 s；三次 graph/state hash 均为 `88098fbab9b930d3/a97ccf56515068ec`，
edge/fixed/non-pair 文件逐字节相同。2023 论文完整 bootstrap 为 5,548 edges、934
fixed、49.4% non-pairs、497 s；当前墙钟数字约为其 1/3.244，但终图多 778 条，
不能报告等强度加速。旧的 6,268-edge/62.635-s 结果存在错误 `-e2` 授权，已作废。详见
[FGPU 强度升级 P0–P8 实现与论文对齐](70_FGPU_Strength_Upgrade_P0_P8_Implementation.md)。

同一正式入口从 `pcb3038` 完全图运行到固定点：`4,613,203 -> 17,872`，19 fixed、
32,041/211,766 non-pairs，5,162,470 replay、0 rejected，E2E 7,495.652 s。按边数
比 2014 Step 2 的 17,940 条少 68 条，但比其累计约 673 s 慢 11.14 倍；相对 Step 3
的 14,869 条仍多 3,003 条，墙钟虽为论文总计 21,322 s 的 `1/2.845`，不能称等强度
加速。该长基准目前是单次计时；图/全状态 hash 为
`c0d80eb7a9b717ce/895b49f61309475c`。完整图上 LP + point、Quick-HS、GPU replay
分别占 E2E 47.98%、25.53%、23.48%。

FGPU resident raw 基准绑定 `e102216`：pcb3038 完全图在单张 RTX 4000 Ada 上
七次 clean 进程 wall 中位为 56.330 s，GPU solve 为 55.404 s；Geometry/PDLP/JV/
Quick-HS 分别为 4.484/0.083/0.268/50.230 s。七次都从 4,613,203 条删至
23,720 条，hash `824cfe92e7345428`，已知最优 tour 零缺边；常驻显存约 308.3 MiB。
该 raw 路径无 CPU 逐边审计和证书，不受 epoch 或节点人为上限截断。论文同实例
为 6,466 条，当前仍多 17,254 条（`3.668x`），所以 56.33 s 不能与论文
5,460.2 s LP 直接形成等强度加速比。

FGPU one-shot 快速主链在单张 RTX 4000 Ada 上：pcb442 `97,461 -> 8,015`，wall 4.872 秒，89,446 条删除 record；pr1002 `501,501 -> 23,288`，wall 18.439 秒，478,213 条删除 record。两者均包含最终 proof replay，官方最优 tour 分别为 50,778/259,045 且零缺边。pcb442 相对作者单轮 `KH -Jq` 的 12,914 条/94.89 秒同时更稀疏、更快；相对旧固定点 4,016 条/99.68 秒则仍弱，不能报告等强度 `20.46x`。pr1002 作者单轮为 21,651 条，FGPU 多 1,637 条；该作者运行与 HT 并发，只用于强度参考。详见 [FGPU One-Shot 实现与完整基准](66_FGPU_OneShot_实现与基准.md)。

pcb442 完整深 HT 现已用一条命令从 97,461 条完全图边到真实 `local-pdlp-fixed-point` 的 3,239 条：Geometry/LP/JV/HT 分别删 85,297/3,856/289/4,780 条，删除率 96.6766%，content hash `ba92119b724b2a1c`。单 GPU、16 target workers、8 replay threads 的 wall 为 3,201.08 秒，峰值 RSS 19,888,500 KiB；421,564,264-byte V5 证书含 6,812,894,031 raw / 414,979,169 compressed HT bytes，独立 verifier 在 1,351.77 秒内重放通过，官方 50,778 tour 零缺边。强度比作者旧固定点 4,016 条少 777 条（稀疏 19.35%），但是作者 99.68 秒的 `32.11x`；这是可靠的强度结果和明确的性能反例，不应在默认加速配置中开启。

另一条四段证书链到 3,231 边，比 one-shot 再少 8 边，但累计 wall 5,298.722 秒。one-shot 边集是该分段边集的严格超集；这个差异说明有界 HT 固定点存在 snapshot/分批路径依赖，不是唯一最小闭包。两条路径证书和 tour 门禁均通过；单命令 one-shot 是主结果，分段链作为交叉验证。

pr299 输入 1208 条边；JV 两个 epoch 后保留 1122 条，提交 86 条删除。CPU 与 CUDA 的最终内容哈希均为 `b9b67e9981518177`。这些数字是正确性回归结果，不构成论文性能结论。

M5 JV 正式基准绑定 `cac180f`：pcb3038、rl5915、d15112 分别从 `6883/29143/166499` 条边删除 `179/550/7312` 条，最终哈希为 `90d13888e351df17`、`0174cf46124ce870`、`76e196dd53d887d5`。进程内 CUDA 算法中位数为 `2.396/7.681/74.829 ms`，相对 CPU 为 `8.132×/25.679×/37.426×`；独立 CLI wall 加速为 `0.186×/0.940×/6.139×`。所有运行逐份 CPU 重放并比较输出；pcb3038 的 137,694 最优 tour 还通过 0 缺边门禁。另两实例尚缺本地最优 tour witness，不能标成 tour-checked。

M5 JV 驻留优化绑定 `25590af`：精确比较坐标、边端点和边权后复用静态 device arrays，每轮仍完整上传 active/CSR/witness。三实例 CUDA 算法中位数降为 `1.741/6.639/70.097 ms`，相对 CPU 为 `11.117×/29.788×/40.119×`；所有 timed epochs 均命中静态键和增长 workspace，峰值驻留为 `391148/1517168/8294196 bytes`。最终图哈希和输出 SHA-256 均未改变。

M5 JV 动态 edge-id 优化绑定 `41feceb`：CUDA CSR 不再重复上传 64 位权重，而用 32 位稳定 edge id 读取驻留权重；三实例算法中位数降为 `1.728/6.503/67.651 ms`，相对 CPU 为 `11.194×/30.517×/41.695×`，峰值驻留降为 `336084/1284024/6962204 bytes`。d15112 的同步 H2D/kernel/D2H 中位数为 `2.965/6.628/0.513 ms`；proof、最终图哈希与输出 SHA-256 均未改变。

M5 JV 活动 edge-id 紧凑 launch 排除实验基于 `5bf92d6`：d15112 三个 JV epochs 的 kernel 行数从 499,497 降至 484,885（`-2.925%`），候选、7,312 条删除、159,187 条最终活动边和哈希 `76e196dd53d887d5` 不变。最终原型的 kernel 中位数从 `6.614268` 降至 `6.530124 ms`，但 propose 从 `11.408986` 升至 `12.289997 ms`，算法总时间从 `63.425767` 升至 `64.093234 ms`。该方案端到端回退 `1.052%`，源码已撤销，只保留项目内实验 artifact。

M5 HT 跨目标根候选排除实验绑定 `a520591`：把 d15112 的 8 targets、2,400 条根 `c,d` screen tasks 合为一次 CUDA launch，仍逐 target 做 CPU flags、规范收尾、预算和 proof。七对交错 clean A/B 的 candidate/search/total/wall 中位数由 `149.537/777.077/940.211/1168.401 ms` 变为 `150.723/783.016/952.132/1197.778 ms`，分别回退 `0.793%/0.764%/1.268%/2.514%`；配对中位差也全部回退。14 份 proof 均独立重放，活动边文件一致，d15112 最优 tour 成本 1,573,084 且零缺边，最终哈希均为 `29c3b8fccaf1a3fc`。源码由 `cca0b55` 完整撤销，后续跨目标研究只保留 leaf/reply/work-graph 共享。

M5 HT reply CUDA 驻留共享绑定 `4396489`：线程/设备本地缓存对维度、距离类型、全部整数坐标和完整 CSR 做精确比较，Hamilton/end 共用设备图和 counts/offsets/error workspace，并分别增长输入/输出区；关闭开关在每批首尾释放，CPU 完整 offsets/replies 认证和 proof 不变。d15112 七对交错 A/B 中，Hamilton/end 中位数从 `238.188/73.257 ms` 降至 `227.889/70.369 ms`，search/total/wall 分别改善 `1.027%/0.753%/0.450%`；23 批中图命中 22 次、workspace 命中 15 次，驻留峰值只增加 805,596 bytes。14 份 proof 均独立重放，活动边和规范 proof SHA-256 一致，最优 tour 全部零缺边；报告/summary 升为 V15/V18，CUDA Debug Hamilton–Tutte memcheck 为 0 errors。

M5 HT reply 精确任务去重绑定 `4428aa4`：Hamilton 在固定 target 的 batch 内按 center 首次出现顺序折叠，end 使用无碰撞的有向 `(endpoint,internal_neighbor)` 整数 key；CPU 仍按原 task 顺序展开完整逻辑 offsets/replies，CUDA 只计算唯一 slices 并逐项与 CPU 比较。d15112 的 Hamilton/end 逻辑量 `23939/4558` 缩为 batch-local 唯一量 `242/176`，物理 CUDA tasks 从 28,497 降至 418（`-98.533%`），驻留峰值从 14,183,576 降至 1,653,672 bytes。七对交错 A/B 的 Hamilton/search/total/wall 中位数改善 `11.851%/3.327%/2.582%/2.158%`；14 份 proof、活动边、工作签名和最优 tour 全部一致，CPU Debug/Release、CUDA Release 与 memcheck 门禁通过。报告/summary 升为 V16/V19，默认开启并保留完整逻辑提交开关。

M5 HT 跨 batch reply 结果缓存排除画像基于 `589caf5`：临时 observer 记录完整逻辑 key，计时因同步 stderr 明确作废。Hamilton 按 `(target,center)` 从 242 个 batch-local keys 只能降至 160，end 按 scan 级有向 task 从 176 降至 122，合计再减少 136 个；但 15 个 Hamilton 与 8 个 end batches 全部仍含新 key，无法跳过任何同步 CUDA 调用。画像运行的边、规范 proof、工作签名和 d15112 tour 与正式结果相同，observer 已移除；当前不实现长生命周期结果缓存。

M5 HT 目标级多 GPU 静态切片绑定 `caed660`：显式 `target_devices` 为每个可见 ordinal 创建固定 worker，按相对目标序号轮转搜索；线程完成顺序不参与结果消费，成功 proof 仍按规范 target 顺序即时 CPU 重放并经原 V2 epoch 提交器整批重放。d15112 双空闲 RTX A4000 的 32-target 七对交错 A/B 中，target execution 从 `3208.626` 降至 `2564.352 ms`（`1.251×`），算法 total 从 `3739.995` 降至 `3081.042 ms`（`1.214×`），进程 wall 从 `4043.140` 降至 `3495.274 ms`（`1.157×`）。单/双卡始终为 40,044 states、52,917 replies、4,100 leaf calls 和 11 条提交边；17 次 CPU proof 重放、17 次最优 tour 检查、设备归属审计、CUDA Release 23/23 与 memcheck 0 errors 全部通过。报告/summary 升为 V17/V20，另以 `target_execution_ms` 区分并行墙钟和 `search_ms` 求和。

V3 单 GPU原型绑定 `a8faebb`：新增短路 Trace/replay、相邻 popcount 层 exact DP、`k<=13` CUDA value 候选器、保持 DFS 规范 proof 的 transposed host-window、单卡多 target workers，以及 commit 前唯一完整 proof 重放。d15112 A5000 32-target 中，transposed 把 wavefront 的 `40,044 states/11 proofs` 降至 `19,498 states/18 proofs`，但产生 19,504 个逐状态 leaf batches，总时间约 24 秒，暂不具备加速价值。clean-commit 七对 A/B 的 CPU/hybrid process wall 中位数为 `1,564.663/1,858.134 ms`，混合路径慢 `18.76%`；相对旧全 CUDA `2.845 s` 则改善约 `1.53×`。14 份计时 proof/tour 和两次预热全部通过，边、规范 proof 与工作签名哈希一致；默认保持 wavefront。该切片并未达到论文 Table 7 的约 73,850 条剩余边强度，不能作同协议论文速度比。

V3 单 GPU leaf broker 绑定 `e41c220/4de281c`：专用 dispatcher 将不同 target required edge 的 transposed leaf windows 合批，CUDA 对 proper 3/4/5-opt 只回传候选位图，CPU 重建命中 witness，最终由 `CommitHtProofEpoch` 在修改图前完整重放。两请求机会式微批取代全 active-worker 栅栏。d15112 五对 clean A/B 的 CPU/GPU target execution 为 `14,130.933/14,001.421 ms`（`1.009x`），algorithm total 为 `17,960.265/17,948.009 ms`（`1.001x`），process wall 为 `18,180.282/18,228.316 ms`（`0.997x`），端到端仍判定为持平。leaf cost 求和已达 `10.606x`，但 `s=4` 增加 `23.62%` 物理 leaf states 和 `75.25%` cost cells，同步 continuation 等待成为下一瓶颈。十次计时与两次预热的边集、规范 proof、工作签名和最优 tour 全部一致；CPU `23/23`、CUDA `26/26` 与两组 memcheck 均通过。详见 [V3 单 GPU 跨目标 Leaf Broker](64_V3_单GPU跨目标LeafBroker.md)。

M5 HT scan pilot 绑定 `cd5ec3e`：pcb3038 的 CPU JV 固定点有 6,704 条边和 6,476 个度数安全目标；最高权重 8-target 切片的 CPU/CUDA 工作签名均为 12,383 states、14,285 replies、9,120 leaf calls 和 5,085 moves，证明并提交相同 2 条边。CUDA/CPU search 为 `33.646/34.103 s`，仅 `1.014×`；最终 6,702 条边、哈希 `fe11f98414b04c0e`，两份 V2 均独立重放且 pcb3038 最优 tour 为 0 缺边。这是功能与资源 pilot，不是显著性能结论。

M5 HT profiling 绑定 `65f9488`：V2 报告把 work graph 拆成 leaf/path/reply 与 host residual，并用 `--leaf-backend` 建立第三条混合路径。相同 8-target clean run 中，CPU/全 CUDA/混合 search 为 `33.987/33.599/33.401 s`；混合仅 `1.018×`。CPU leaf 为 27.160 s（search 的 `79.914%`），Hamilton reply 为 6.245 s（`18.375%`），host residual 只有 0.228 s。三路 proof 均独立重放、边文件 SHA-256 一致且最优 tour 零缺边。

M5 HT bucket fusion 绑定 `b5fde24`：V3 报告加入 leaf frontier/bucket/cost batch 计数，显式融合将 8-target 的 frontier batches 从 500 降到 86、CUDA cost batches 从 1,835 降到 382，cost cells 与 proof 不变；但 clean run 的混合/fused search 为 `32.672/32.905 s`，交错重复也仅约 `1.002×`。因此默认保持关闭，并把下一门禁改为 leaf 内部 cost/cursor/certification 计时。

M5 HT leaf 子阶段画像绑定 `fe99f38`：V4 报告和 V5 summary 把 leaf 拆成 setup、cursor prepare、cost、scatter、consume、apply 与 verifier。相同 8-target clean run 中，混合 leaf 为 `26.489 s`，其中 setup `5.864 s`（`22.139%`）、GPU cost `0.584 s`（`2.204%`）、CPU cursor consume `19.695 s`（`74.352%`）；四路工作签名、最终边集、proof 重放与最优 tour 门禁均一致。下一步先缓存只由 path count/k 决定的不可变组合表，再细分 CPU completeness。

M5 HT 不可变表缓存绑定 `589bca0`：线程安全延迟缓存复用 path-count matching catalog 和 `k=3/4/5` reconnect templates，不缓存任何图相关状态。相对 `fe99f38`，CPU leaf/search 从 `27.231/34.095 s` 降至 `17.760/24.138 s`（`1.533×/1.413×`）；混合 setup 与 cursor consume 分别下降 `70.002%/23.278%`。四路工作签名、51,309,996 cost cells、最终边集、proof 重放和受保护 tour 不变；剩余混合 consume 为 `15.111 s`，是下一画像对象。

M5 HT completeness 画像绑定 `ee4f3aa`：V5 report/V6 summary 显示 727,635 个 CUDA cost rows 中 726,648 个（`99.864%`）进入 CPU 全模板 fallback，51,179,094/51,309,996 cells（`99.745%`）被通用重连器重新穷举。混合 consume 中 candidate/fallback 为 `0.057/15.107 s`，fallback 占 `99.033%`。三条 CUDA leaf 路径计数、proof、图和 tour 均一致；下一步以 CPU 独立精确矩阵替换昂贵的通用无改善认证，不降低 completeness。

M5 HT CPU 精确矩阵认证绑定 `48d68dc`：固定数组 scorer 对 CUDA cost matrix 的 51,309,996 cells 全部进行独立 CPU 整数认证，差异立即失败关闭；987 个严格改善模板仍由通用 CPU 路径重建完整 witness，旧 completeness fallback 降为 0。相同 8-target clean run 中，hybrid cursor consume 从 `15.255 s` 降至 `0.126 s`，leaf/search 为 `5.011/11.586 s`，相对 CPU scalar 达到 `3.571×/2.097×`；all-CUDA/fused search 分别为 `11.673/11.499 s`。四路工作签名、最终图哈希 `fe11f98414b04c0e`、proof 重放和 pcb3038 最优 tour 门禁完全一致。下一步让 CPU backend 复用相同矩阵 fast path，建立公平基线。

M5 HT CPU matrix 公平基线绑定 `9fa301d`：显式 CPU leaf 进入相同增量 cursor，并修复 cost block 尾部未消费 rows 被计入 proof 的既有规范计数问题。随机 direct/auto/CPU-matrix 三路现与 CPU scalar proof 逐字节一致。8-target clean run 中 CPU matrix leaf/search 为 `4.491/11.015 s`，优于 all-CUDA `4.848/11.730 s`、hybrid `5.034/11.430 s` 和 fused `4.974/11.594 s`；四路 51,309,996 cells、727,635 consumed rows、987 个候选、最终图和 tour 均一致。同步全量 CPU 认证下 GPU 没有净 leaf 加速，下一画像转向占 CPU search `55.19%` 的 Hamilton reply。

M5 HT Hamilton reply 主机优化绑定 `f12c181`：批 API 只验证一次图，同批重复 center 复用规范回复，并把 2-opt quick filter 提升为每邻边一次。8-target 的 72 batches/27,598 逻辑 centers 只实际枚举 1,395 个 batch-unique centers 和 11,515 个邻边对，仍输出相同 245,965 replies。CPU reply 从 `6.079 s` 降至 `0.019 s`（约 `321×`），CPU search 从 `11.015 s` 降至 `4.870 s`（约 `2.26×`）；all-CUDA/hybrid/fused search 为 `5.562/5.345/5.354 s`。四路工作图、51,309,996 leaf cells、最终图、proof 重放和 tour 门禁均一致；下一瓶颈是占 CPU search `91.38%` 的 leaf。

M5 HT leaf setup 画像与快照哈希复用绑定 `f71b472/c968b01`：V8 report/V10 summary 将 setup 拆为 proof 初始化、coverage 扫描和 cursor 构造，并要求四路 cursor 数一致。基线的 9,891 个 cursor 中，proof 初始化占 setup `92.57%`，根因是每个 leaf state 都重算同一不可变 graph 哈希。改为每 batch 计算一次后，CPU setup/leaf/search 从 `1.761/4.497/4.915 s` 降至 `0.232/2.981/3.398 s`，分别加速 `7.59×/1.51×/1.45×`。四路 51,309,996 cells、proof、最终图和 tour 均不变；下一瓶颈是占 CPU leaf `78.67%` 的精确 cost matrix。

M5 HT CPU cost row 并行与 leaf 桶融合绑定 `34cf918/63133c7/623f167`：8,192 cells 以上按 task row 静态分片，最多 8 线程；无 OpenMP、小 batch 和单线程保持串行。clean 1/8-thread A/B 使 CPU certify/leaf/search 从 `2.340/3.029/3.444 s` 降至 `0.701/1.442/1.867 s`。V12 再加入 CPU 融合为第五路；pcb3038 leaf/search 为 `1.149×/1.113×`，rl5915/d15112 leaf 为 `1.085×/1.180×`、search 均为 `1.016×`。三份锁定公开 tour 及三实例五路 proof、规范工作量和最终图全部通过。CLI 现对 CPU leaf 默认融合，auto/CUDA 默认不变，显式 0/1 可覆盖。大实例下一瓶颈是 path-append。

M5 HT path-append 画像与稀疏规范化绑定 `ba15919/b551a2e`：V10 report/V13 summary 将包含式总量拆为 parent prepare、child normalize、child edges、CUDA evaluate 和 compare。内部 fast path 只为实际路径节点构造有序邻接，通用 dense 实现继续独立 proof 重放；308-task 单元差分逐项比较完整规范结果。三实例 path-append 加速 `5.503×/9.538×/25.561×`，search 加速 `1.115×/1.679×/3.133×`；五路规范 proof、最终边和受保护 tour 全部不变。下一瓶颈是大实例 host-build residual。

M5 HT 根 child 规范化排除实验绑定 `6b2b8ad`：V11/V14 记录 72/151/647 次根 child dense 规范化，CPU-fused 仅为 0.773/2.174/28.642 ms，占三实例 host residual `0.406%/0.562%/2.605%`。未为低占比路径扩大共享 fast-path API；下一画像目标是逐 frontier state 的全维 point-candidate 扫描。

M5 HT point-candidate 画像与 Top-K 优化绑定 `5944476/4e4f8e3`：V12/V15 记录 state scans、checked/ranked/selected 节点、扫描与排序时间；严格全序的 `partial_sort(top 25)` 将三实例排序从 `137.639/298.303/858.648 ms` 降至 `11.901/14.314/28.784 ms`，CPU-fused search 加速 `1.086×/1.580×/1.552×`。三实例五路活动边、工作签名、规范 proof 和受保护 tour 全部不变。下一步验证同一 target 的静态评分顺序复用。

M5 HT point-candidate 静态次序缓存绑定 `ea85ffa`：每个 target 只计算一次中点评分和严格全序，逐 state 用 generation marks 过滤已有路径节点，同时保持完整维度 checked 与原 ranked 计数。相对 Top-K，三实例 scan 再加速 `188.364×/238.242×/293.326×`，CPU-fused search 加速 `1.045×/1.176×/1.131×`；相对全量排序画像的累计 search 加速为 `1.135×/1.858×/1.756×`。三实例五路规范计数、边、工作签名、proof 和 tour 全部不变。下一步转向跨目标 leaf 准备数据与调度共享。

M5 HT leaf proof 批内快照绑定复用绑定 `0530ff4`：同一同步 k-opt batch 内生成和复核使用一次入口 graph hash；公开 `VerifyPathSystemKOptProof`、HT 最终 verifier、scan 即时复核和 epoch 重放仍独立计算。三实例 leaf proof verify 加速 `8.882×/17.591×/27.449×`，CPU-fused search 加速 `1.037×/1.355×/1.618×`。三实例五路规范工作、边、proof 和 tour 全部不变。下一步复用同一 wavefront 的 snapshot binding。

M5 HT wavefront leaf 快照绑定复用绑定 `00c0156`：强类型 binding 只能由实际 graph 对象构造，并在同一只读 wavefront 的 leaf batches 间复用；对象错配拒绝，公开 API 和全部独立 verifier 仍自行哈希。rl5915/d15112 的 proof init 加速 `44.302×/99.936×`，CPU-fused search 加速 `1.063×/1.028×`；pcb3038 端到端无可测收益。三实例五路规范工作、边、proof 和 tour 全部不变。新画像把下一热点锁定为 d15112 的 `180.448 ms` path child normalize。

M5 HT path-append child 增量规范化绑定 `236022c`：已认证 parent 建立 node-location 索引，point/end child 直接合并至多两个规范链；通用 sparse parent 检查、独立 dense proof 重放和 CUDA 全数组认证均保留。992-task 差分逐项比较完整失败原因和规范 paths；三实例 child normalize 加速 `8.511×/9.752×/8.179×`，CPU-fused search 加速 `1.038×/1.220×/1.244×`。三实例五路规范工作、边、proof 和 tour 全部不变；下一热点是跨 reply batches 的重复 graph validation。

M5 HT wavefront 图验证绑定复用绑定 `fb772f8`：binding 构造时完整验证一次 CSR，c,d/Hamilton/end 内部 APIs 检查同一 graph 对象后复用；公开 APIs 和所有独立 verifier 仍完整验证。三实例 Hamilton validation 加速 `305.600×/583.206×/3388.723×`，CPU-fused search 加速 `1.017×/1.146×/1.411×`。三实例五路规范工作、边、proof 和 tour 全部不变；下一步在同一不可变 scan 的 targets 间复用 graph/hash bindings。

M5 HT scan 跨目标快照绑定复用绑定 `649f3f4`：同一只读 scan 只完整构造一次 graph/hash bindings，内部 wavefront 逐次检查对象身份；每 target 的内容哈希变更守卫、即时公开 verifier 和最终 V2 重放均保留。三实例 CPU-fused candidate 加速 `1.674×/3.162×/7.355×`，search 加速 `1.007×/1.135×/1.246×`，d15112 total/wall 加速 `1.154×/1.123×`。三实例五路规范工作、边、proof、JV 固定点和 tour 均经 54 项跨提交精确比较确认不变；下一热点是 d15112 leaf cursor construct 的 `69.004 ms`。

M5 HT leaf path 稀疏验证绑定复用绑定 `8c19740`：batch 内每个 path 以实际节点规模认证一次，后续 outside cursors 只检查 graph/path 对象身份；公开 scalar 与成功 proof 的 dense verifier 完全不变。2,000 组随机 sparse/dense 规范结果和失败原因一致，篡改 path 的 batch/scalar proof 字节一致。三实例 leaf setup 加速 `2.487×/4.641×/6.814×`，CPU-fused search 加速 `1.051×/1.105×/1.196×`；d15112 search/total/wall 降至 `312.481/478.235/627.747 ms`。三实例五路规范工作、边、proof、JV 固定点和 tour 均经 54 项跨提交精确比较确认不变。

M5 HT CPU cost 固定计划绑定 `273ac9d`：task 验证不再为 3–5 条边构造 `std::set`，每阶 canonical templates 编译为固定端口对，task 内规范边/冲突/距离惰性复用。CUDA kernel 不变且输出仍逐 cell 经完整 CPU 矩阵认证。三实例 cost evaluate 加速 `1.510×/1.580×/1.431×`，CPU-fused search 加速 `1.274×/1.131×/1.035×`；pcb3038 search/wall 降至 `913.156/950.567 ms`，d15112 search 为 `301.853 ms`。三实例五路规范工作、边、proof、JV 固定点和 tour 均经 54 项跨提交精确比较确认不变。

M5 HT CPU batch 距离表绑定 `0d506ab`：同步 cost batch 在 512 节点和约 6 MiB 硬上限内一次计算精确对称距离表，预计不足 2× 重用、图过大或小矩阵均回退 task-local scorer。pcb3038/d15112 画像的 batch 内 pair 重用上界为 `282.948×/131.206×`；三实例 cost evaluate 加速 `1.195×/1.057×/1.216×`。pcb3038 search/wall 降至 `843.266/881.674 ms`，d15112 search 降至 `298.238 ms`；rl5915 cost 改善但单次端到端回退 2.89%，不作为整体收益。CPU/CUDA 完整矩阵认证、sanitizer 和 54 项跨提交精确比较全部通过。

M5 HT leaf cost 零复制分发绑定 `45098d3`：每个 cursor 以生命周期受限的只读 span 消费融合矩阵连续 slice，不再构造和复制临时 `KOptCostBatchResult`。三实例 scatter 加速 `70.871×/30.817×/28.441×`；pcb3038 search/wall 降至 `820.532/861.406 ms`，rl5915 search 降至 `145.188 ms`。d15112 的 1.800 ms scatter 节省被 cost/host 波动覆盖，search 单次回退 1.39%，不作为端到端收益。CPU/CUDA 全套、memcheck 和 54 项跨提交精确比较全部通过。

M5 HT cursor prepare 路径边成本缓存绑定 `611c701`：删除位置改为固定 5 元数组，`TourContext` 按巡回位置一次缓存 selectable 路径边精确成本；严格改善候选仍由通用重连器重新计算。三实例 prepare 加速 `1.967×/1.503×/1.618×`；pcb3038 search/wall 降至 `728.263/763.032 ms`，d15112 search/wall 降至 `293.553/612.480 ms`。rl5915 leaf 改善 4.69% 而 search 回退 0.86%，不作为整体收益。CPU/CUDA 全套、memcheck 和 54 项跨提交精确比较全部通过。

M5 HT CPU cost 输出 workspace 绑定 `900f5d9`：临时画像否定同轮跨 k 距离表共享，并定位到 owning vector 的重复预清零；内部 CPU cursor 改以单一可增长 storage 的逻辑 span 全量覆写，公开 API 与 CUDA 完整认证不变。三实例 cost evaluate 加速 `1.407×/1.263×/1.306×`，CPU-fused search 加速 `1.148×/1.134×/1.032×`；pcb3038 search/wall 降至 `634.582/674.254 ms`，d15112 search 为 `284.330 ms`。Debug 全覆盖哨兵、CPU/CUDA 全套、memcheck 和 54 项跨提交精确比较全部通过。

M5 HT CPU cost 完全相同 task row 去重绑定 `a4afa29`：8,192-cell 与至少 25% row 缩减双门槛后，只评分完整 `port_nodes[10]+deleted_cost` key 的首次 row，cursor 通过已完整验证的只读映射零展开消费。公开 owning API、CUDA 每逻辑 row 的独立 CPU 矩阵认证和通用 witness verifier 不变；V13/V16 分开记录逻辑认证量与物理 scored/reused rows。三实例 CPU-fused 物理 row 复用为 `3.523×/2.143×/1.938×`，cost evaluate 加速 `1.344×/1.066×/1.102×`；pcb3038 search/wall 加速 `1.186×/1.181×`，d15112 为 `1.021×/1.013×`，rl5915 端到端处于负向噪声。CPU/CUDA 全套、memcheck 和选定三实例的 54 项跨提交精确比较全部通过。

M5 JV—HT 多 epoch 编排绑定 `e74b197`：调度器在工作副本上先达到 JV 固定点，无提交 HT 切片只推进当前 sweep offset，有提交则在新快照重跑 JV 并从 offset 0 重排；各阶段 records 与 HT sidecars 合为一个连续、可独立 CPU 重放的 V2 proof。8 点完整图以 7! 巡回穷举确认 7 条 JV 与 4 条 HT 删除都不属于任何最优巡回。pcb3038 干净提交三轮门禁中，CPU/CUDA 均提交 179 JV + 3 HT，活动边 `6883 -> 6701`、最终哈希 `dce8912b10c3736e`，边文件和规范 proof 完全相同；两份 proof 独立重放且 137,694 最优 tour 为 0 缺边。运行因显式三轮上限返回 `ht-epoch-limit`，不是全图收敛声明。

cuOpt 手算 LP：状态 `OPTIMAL`，objective/dual objective 均为 `1`，primal violation 与 reduced-cost residual 均为 `0`，定点模型下界为 `16777216/16777216`。

cuOpt 连续 epoch warm start：列由规范边端点稳定编号，行由方向/RHS/稳定列系数内容编号；重排按身份映射，重复身份、覆盖不足和未验收解均失败关闭。`CuOptSession` 在 presolve 关闭时调用 26.8.0 的 primal/dual 初值 C API；小 LP 第二轮实测 attempted/applied 均为 1、行列覆盖率均为 1。该路径仍只提供数值候选，不授权删边；完整图迭代负 reduced-cost 补列仍未完成。

Concorde 随机 20 点 epoch：25 行、43 列；QSopt 与 cuOpt 模型目标均为 `88`。cuOpt primal violation 为 `4.44e-15`，reduced-cost residual 为 `1.57e-14`；完整图 exact lower bound 为 `87.3932819641`，上界为 `88`。

pr299 Concorde epoch：454 行、888 列、8561 个非零元；cuOpt 状态 `OPTIMAL`，模型目标 `48187.777777780764`，primal violation `8.37e-11`，reduced-cost residual `4.87e-10`。完整图 exact lower bound 为 `43977.2693797`，合法但较弱；这说明后续需要对偶稳定化/迭代补列，而不是跳过负 reduced-cost penalty。

路径兼容表：`m=5` 为 45,360 字节，生成器哈希 `f6bccacc5c1fa84f`；362,880 个 `m=5` 单元和全部较小表均通过 CPU/CUDA 差分，CUDA memcheck 为 0 error。`m=6,7` 不建立完整表，按契约使用 CPU 直接判定。

CPU k-opt 叶证明：自动生成的 proper 3/4/5-opt 模板数为 `4/25/208`，每阶 2,000 个阈值定向随机矩阵与固定子模块 `swap.c` oracle 一致；成功 witness、全 outside coverage、预算 unresolved 和篡改拒绝均有回归。

CUDA k-opt cost：按删除集合与 proper template 形成精确成本矩阵，CUDA 全矩阵与独立 CPU 整数矩阵逐 cell 一致后才进入 consumer；非改善 completeness 由 CPU 矩阵认证，严格改善候选再由通用 CPU 路径重建并验证 witness。真实 HT 8-target 的 51,309,996 cells 全部认证，CUDA memcheck 为 0 error；相对 CPU scalar 的 hybrid search 为 `2.097×`，但仍需 CPU matrix 公平基线。

CPU 精确困难叶：将每条 forced outside edge 收缩为可双向访问的 block，其余节点为 singleton block；固定一个 block 的方向消除无向反转对称后，以 Held–Karp 子集 DP 穷举所有 block 次序和方向。比较时消去两条巡回共有的 outside 成本，并禁止候选重新使用 required path edge。成功结果转换为任意 `k>=2` 的交换 witness，再交给同一独立 verifier；默认关闭，启用时最多 18 个 block，内存不足或超限均返回 `unresolved`。60 组随机 7 点、每组两个 outside 的结果与直接 Hamilton 巡回枚举最优值一致，并覆盖 7-opt proof 往返。CPU Debug/ASan、CPU Release 与 CUDA Release 全量回归通过，GPU 2 上 k-opt memcheck 为 0 error。

浅层 HT：固定 `c,d` move 后，重新枚举两个中心所有通过严格 2/3-opt 快速筛选的邻边对，并要求其笛卡尔积中的每个 reply 都有 path-infeasibility 或完整 path-system leaf proof。固定 8 点实例覆盖 30 个非空 replies；删除 reply、篡改 move/snapshot/leaf 均被 verifier 拒绝。CUDA 在稀疏 32 点 EUC/CEIL 批次上输出与 CPU 完全一致的 `c,d` flags，memcheck 为 0 error。

递归 HT：CPU DFS 实现与后续 wavefront 相同的 `Leaf(F) OR ∨move ∧reply HT(F∪reply)` 真值。未解决状态可选择未出现在路径系统中的 point，或选择当前路径 endpoint；每个 move 必须记录其完整活动图 replies。成功子树保存为只向后引用的扁平 continuation arena，独立 verifier 从目标边重新规范化每个子状态并拒绝环、共享 child、遗漏 reply 和未引用节点。`CUDAEE_HT_RECURSIVE_PROOF_V1` 嵌入现有 path-k-opt V1 叶证明并严格拒绝非法计数、枚举值和尾随字段。固定 point/end 递归实例均由 8 点完整巡回穷举额外确认目标边不属于任何最优巡回；depth/budget fail-closed 与 proof 篡改已有回归。`ht-prove`/`ht-verify` 已覆盖固定实例的文件级端到端 CTest；未解决写入 `proven=0` 并返回退出码 3。DFS 调度本身仍在 CPU，GPU 版本使用下一段的独立 wavefront 路径。

混合 wavefront：主机 BFS 为每个状态先跑 leaf，再生成有界 point/end OR moves 及其完整 replies；所有 child 只指向下一层。CPU 从后向前得到规范状态真值；CUDA 从 leaf/vacuous/无 move 终态队列开始，每个完成 child 原子更新 move 的 `remaining_children/failed`，成功 move 立即完成父状态，失败 move 递减父状态的 `remaining_moves`，最后一个失败 move 才宣告 OR 失败。single-block 或 cooperative multi-block persistent kernel 在设备端冻结并消费动态队列批次，要求每个状态恰好入队一次；队列溢出或提前停滞均失败。CUDA/CPU 任一状态不同即返回 `invalid`。成功时仅复制第一个成功 move 的子树到既有 continuation arena，再运行完整 proof verifier。

GPU path append：规范父路径展平为 `(node,component,degree)` 与规范父边 spans，一个线程检查一个 point/end task。point 中心必须是新节点；两个连接点度数均小于 2 且不能来自同一分量。end 必须从现有端点出发，另一端只能是新节点或其他分量端点。CUDA count/write 为每个可行 task 输出父边加新增边的严格排序 CSR slice；CPU 对每项仍执行 `NormalizePathSystem`，独立重建 flags、offsets 和全部边，只有全数组相等才标记设备 child 已认证。固定双父状态 11-task 表得到 27 条 child edges；不可行 task 保持空 slice。工作图仍只使用 CPU 规范 child，这些是正确性样例，不是性能结论。

GPU Hamilton replies：CUDA 对每个中心先 count，再由主机建立 `uint64_t` 前缀区间，最后按排序 CSR 的确定顺序 write。整数平方根实现与 CPU 的 `EUC_2D`/`CEIL_2D` 精确边界一致；根 `c,d` 两中心合批，递归 point 的全部候选中心在单个父状态内合批。CPU 始终重新枚举完整 offsets 和 reply 列表并逐元素比较，只有 CPU 列表进入工作图。12 点完整图覆盖两种距离、重复中心和 CPU-only 回退；固定 point CLI 为 9 batches、16 centers、46 surviving pairs，证明规模和重放结果保持不变。这些仍是正确性指标，不代表端到端加速。

GPU end replies：同一父状态的全部路径 front/back 端点可合为一个 batch；每个线程从端点排序 CSR 排除内部邻点，经主机 `uint64_t` 前缀和后把规范活动边写入独占区间。CPU 重新扫描完整 offsets/edges，只有 CPU 列表进入候选排序和工作图。9 点完整图覆盖相反 endpoint 方向与重复 task，稀疏链覆盖 degree=1 的零长度区间；固定 recursive-end proof 同时要求 CPU 与全 CUDA wavefront 成功并由同一 V1 verifier 重放。

Frontier reply batching：当前层按最多 256 个父状态切成资源 chunk，point centers 与 end tasks 分别展平并一次生成，再按记录的 spans 恢复每个父状态候选。child 仍按原 state/candidate/reply 顺序物化，固定 point 实例的 `N=1` 与 `N=256` V1 proof 逐字节一致；Hamilton batches 从 9 降为 4。

Frontier path append：一个 chunk 的可尝试 point tasks 先共用一次多父状态 batch；根据 CPU 认证 flags 排除 vacuous-success states 后，才生成 end replies 和 end append batch。固定 point 实例的 path-append batches 从 9 降为 3，tasks 保持 84；end batches 从 2 降为 1，工作量在两种 chunk 大小下均为 8 tasks/48 edges。全局预算和 child 写入仍按原 state/candidate/reply 顺序执行，V1 proof 逐字节一致。

规范 child edge SoA：父状态新增严格排序的 edge spans。第一阶段输出可行 flags 与 child edge counts，主机建立 `uint64_t` offsets，第二阶段在 task 私有 slice 中复制父边、追加 reply 边并确定性排序；重复边、区间不一致或 CPU 全量差分失败均拒绝批次。固定 recursive-point 全 CUDA wavefront 为 3 batches/84 tasks/172 child edge records，34-state 工作图和 4 节点 V1 proof 保持不变并通过独立重放。

Frontier leaf batching：当前 reply chunk 先按 `(depth,path_count,node_count,max_k,incoming reply bucket)` 分桶。每轮把各 proof 首个未覆盖 outside 的同 k cost blocks 合并，CUDA matrix 先经 CPU 全矩阵逐 cell 认证，只有严格改善 cell 按原顺序执行 `TryReconnect`、inside coverage 与独立 proof verifier。固定 recursive-point 的 34 rows/136 cells 从 34 个 cost batches 降为 6 个，最大 batch 16；`N=1/256` 的 4 节点 V1 proof 逐字节一致。

一般 leaf 游标：每个 path/outside 保留当前 k、组合索引、统计计数和至多一个 pending block；主机只在 CPU 消费当前 block 后推进。固定双 7 点完整 3/4/5-opt 穷举把 50 tasks/2660 cells 合为 13 batches，预算 3 的 `2+1` block 停止点也与 scalar 一致；12 组随机图、三种路径布局和多种 k/预算/batch size 的 CPU/CUDA V1 proof 全部逐字节差分。

GPU leaf 驻留缓存：每个主机线程在首选设备上复用整数坐标、独立 3/4/5-opt 模板和增长型 task/cost buffers。距离快照逐坐标比较完整 kernel 依赖，模板同时比较生成器哈希与完整数组；owner-device buffer 可在 epoch 边界显式释放。固定 recursive-point 的 6 个 CUDA leaf batches 只有首批上传快照/模板，记录 5/5 次命中、4 次 workspace 命中和 1468-byte 驻留峰值，proof 保持不变。

CPU long-tail：项目内稳态基准在 RTX 4000 Ada 上定位 3-opt 64/256、4-opt 25/100 与 5-opt 208 cells 的 CPU/CUDA 交叉区间，默认用 128 cells 分流融合后的 `auto` 矩阵。31 个 3-opt leaf（124 cells）走 CPU、32 个（128 cells）走 CUDA；规范化的模板枚举计数使显式 CPU/CUDA、阈值两侧和不同 frontier batch size 都保持 V1 proof 字节一致。固定 recursive-point 的 auto leaf 六批全部进入 CPU long-tail，proof 与显式全 CUDA 相同。

Multi-block continuation：cooperative kernel 以 grid barrier 冻结并消费完成队列批次，只有 `queue_tail==state_count` 才正常终止；自动 block 数不超过 kernel 的实际 cooperative residency，显式越界闭门失败。固定 truth table 的 single/2-block 状态相同；512 个 child 跨两个 block 汇入同一 AND move 时，单失败和全成功真值均正确，513-state 自动模式选择 3 blocks。固定 recursive-point 显式 2 blocks 的 34-state 数组经 CPU 全量认证，V1 proof 与 single block 逐字节一致。

HT epoch commit：`ht-commit` 可重复接收同一不可变快照上的 recursive HT sidecar，先逐份 CPU 重放，再按 `(u,v,serialized-proof)` 规范化重复目标，并在图副本上执行共用最小度门禁与 CSR 重建。实际含 HT 删除时写 `CUDAEE_PROOF_V5`：outer record 唯一引用 inner HT V1，sidecar 逐份 zlib 压缩并绑定 raw size/CRC32；通用 `verify` 检查压缩上限、绑定、规范顺序、完整证明和最终哈希后才发布重放图。V1–V4 仍可读取；无 zlib 构建保持 V2–V4 原文路径。固定 recursive-point 的两份等价 sidecar 从 28 条边提交 1 条，哈希由 `d7bfbec67ffc9a66` 变为 `78ce8b9a9dc29473`；坏 sidecar 混入时整批零修改。JV-only 仍写原 V1。

## 安全边界

`gpu-eliminate` 的自动候选器仍只实现 JV；HT 可使用显式 `ht-prove -> ht-commit`、单快照 `ht-scan`，或由 `local-eliminate` 执行有界 JV—HT 多 epoch 调度。目标级静态多 GPU 已可显式启用，reply 的静态设备图和工作区可在各 worker 的 targets 间驻留，同一 batch 的精确重复任务也已折叠；跨 batch/target reply 结果缓存已经画像并因不能消除同步调用而排除，当前仍没有把单个 target 或共享 work graph 拆到多 GPU，也没有 leaf 语义结果缓存或 LP 删除授权。显式 `ht-epoch-limit` 结果只能视为安全部分消元，不能外推为完整 Local Elimination 固定点或性能加速。`lp-solve` 始终不修改图。Concorde 桥接已能产生完整图安全下界，但测试 wrapper 使用 `-B`，尚不输出消元边集；M3.1 仍为 pending。仍严禁从未完整验证的局部结果、过期 HT sidecar、cache hit、去重命中或 cuOpt 浮点 reduced cost 直接构造删除记录。

## pcb442 完全图端到端（2026-09-03）

新增 `complete-graph`、`--profile kh-jq`、`--complete-sweep/--target-batch-size`、初始/最终强度门禁，以及隔离构建的作者 KH 工具。`tools/run_complete_kh_jq_e2e.sh` 将作者 `-Jq` 分解为每轮 `CUDA JV -> KH -q/HS -> CUDA JV` 并重复至固定点。

锁定的 pcb442 完全图从 97,461 条边出发。clean commit `020287f` 首轮到 13,012 条，同环境作者 `-Jq` 为 12,914 条；时间 94.25/94.89 秒，总体持平且分解路径少删 98 条。10 轮固定点到 4,016 条，删除率 95.879377%，最优 tour 50,778 保留；从同一完全图输入为 99.68 秒，比作者单轮慢约 5.0%，但剩余边少 68.9%。强度不同，不能作为等强度加速比；尚未达到 1.25x 默认启用门槛。

所有 CUDA JV proof 均逐轮独立重放。作者实验 KH-HS 本身是 CPU 精确授权，但不导出可由独立 verifier 重放的 HT tree；V2 manifest 只提供输入/阶段哈希、重算配置、proof 哈希、tour 与固定点门禁，不得标记为 portable-proof。详见 `65_pcb442_完全图固定点端到端.md`。
