# Quick/extra-edge 精确回复压缩（2026-09-06）

分支 `research/fgpu-hybrid-beat-hs2014`，基于 `a5b99c1`。
本轮按照[细化实施计划](../design/FGPU_Implementation_Priorities_After_Priming.md)交付 A，
四例终态/速度目标仍以[冻结协议](../design/FGPU_Hybrid_Beat_HS2014_Execution.md)为准。

## 1. 实现边界

新增 `src/cuda/quick_reply_cache.{cuh,cu}` 和默认关闭的
`--quick-reply-cache 0|1`。Quick、extra-edge depth 1/2 的 proposer 与 GPU replay
均可消费压缩流；没有 CPU 逐边审计，没有截断根集合或回复集合。

原先每组 c,d 遍历 `C(deg(c),2) × C(deg(d),2)`，进入窗口后才判断单侧 pair
是否 active、已是 nonpair、是否违反 root 的 Opt22/Opt23。现在先在 GPU 生成
每个 root 原有的候选中心，再为 `(snapshot, root, center)` 建立稳定 ordinal CSR：

1. 相同邻接遍历、两跳范围与 `(cost,node)` 插入次序，不改变候选中心或 OR 顺序。
2. GPU count → CUB scan → 按实际总数扩容 → ballot 稳定写出。候选 stride 仍是
   原方法中心数，不是每中心回复容量；没有基于资源的回复丢弃或全局 nonpair 发布。
3. 独立 GPU verifier 用自然 CSR 双循环和独立 Opt22/Opt23 表达，对每个全集 pair
   检查“应存在”与“实际存在”等价，另查范围、严格递增、row offsets、root 与 epoch。
   比只比较数量更严格：重复一条不能补偿漏掉另一条。
4. 只有完整覆盖通过的行可由 proposer/replay 读取。replay 针对选中的 c,d，再检查
   整个压缩笛卡尔积；未覆盖、身份错配、溢出或分配失败不表示空 AND。
5. commit 后重新构造流，workspace 只复用容量，不复用旧 epoch 的语义。

共享端点必须使用 Quick 的 Opt22 特例，不能直接使用 Main 的纯距离 Compatible；
非度量/舍入距离也必须等价。Opt23 的 root 三角形、真子环/全图环、degree-2 邻点及
显式 fixed 都在独立检查中保留。完整 `ReplyPassesFastFilters` 及后续 LP/局部叶判定
未删减；压缩只移动其中不依赖另一个中心的必要条件。

## 2. 指标和计时口径

manifest 新增：

| 字段 | 含义 |
| --- | --- |
| `quick_reply_cache` | 本次是否开启，基准配置身份的一部分 |
| `quick_reply_raw_pairs` | 各次建流、有效中心原始三角 pair 数累计，非笛卡尔积访问量 |
| `quick_reply_compact_pairs` | 实际写出的完整单侧 pair 数累计 |
| `quick_reply_cache_bytes` | 可增长 workspace 的已分配容量高水位，不含其他模块 |
| `quick_reply_build_ms` | 候选、计数、scan、扩容、写出和同步的 host wall |
| `quick_reply_validation_ms` | 独立覆盖 kernel 与同步的 host wall |

build/validation 已包含在对应 Quick/extra-edge proposal 阶段，不重复加到 E2E。
两个 pair 计数都不是短路后的实际 leaf 工作量，其比值不能写成加速比。
`benchmark_fgpu.py` 支持同二进制开关对照；四例 collector/compare 记录配置，
混用该开关的运行不能通过正式验收。

## 3. 正确性门禁

专项 72 组图包含 EUC/CEIL、零成本、随机非度量、大整数，以及 stable CSR 的
inactive 槽位、显式 fixed、degree-2、nonpair 和打乱 root ID 顺序。
独立 CPU 自然枚举与 GPU 的所有 ordinal/顺序逐项比较：

- 37,307 个有效中心行：1,422 空流、35,885 非空流。
- 原始 pair 4,348,976，实际 compact pair 1,212,432。
- 468 次快照错配、重复、越界 ordinal、漏条目、错误 root/center、越界 offset 注入
  全部拒绝；失败后 consumer 的授权位也拒绝。
- CPU 37/37；专项 memcheck **0 errors**、racecheck **0 hazards**。
- CUDA **69/69**，416.76 s：`artifacts/hybrid-quick-reply-cuda-tests-v2.log`。
  含 14 组小图、286 次无标签完整求解（新增 52 次 cache-on 消融），12 次密图调度；
  EUC/CEIL、重复坐标、多最优 tour 的 36 组完全/中间快照，288 次 GPU 求解，
  cache-on/off 最终 edge/fix/nonpair 逐项一致，所有最优解均保留。
- 288 次完整 GPU 状态检查的 memcheck **0 errors**、racecheck **0 hazards**，
  实际覆盖 586 个近点预热提案，含 raw/compact 两次同终态验证。

原始日志：`artifacts/hybrid-quick-reply-cpu-tests-v1.log`、
`artifacts/hybrid-quick-reply-unit-standalone-v1.log`、
`artifacts/hybrid-sanitizer/quick-reply-v1-{memcheck,racecheck}.log`。
专项冻结二进制 `.tmp/hybrid-quick-reply-unit-standalone-v1` SHA256：
`7c99811caa4e926229f680fed1ebe9b4c424089d80dd06fe55f5ea11def60870`。

集成 sanitizer 日志 `artifacts/hybrid-sanitizer/quick-reply-optimal-v2-{memcheck,racecheck}.log`；
冻结 `.tmp/hybrid-quick-reply-optimal-unit-v2` SHA256：
`ce4b719e56cee22f4bbb849786f4b92414c5200d3ef383b5af9551306f7ca955`。
该专项覆盖 Quick/Point/事务，不开 LP、geometry、full metric 或 extra-edge；
extra-edge 的完整求解由无标签全套和 pr299 文件对照覆盖，不能宣称 sanitizer
已覆盖全部数学域。

## 4. 完整图时间对照

pr299 原始坐标完整图、原生 LP、排列目录、近点优先和 adaptive，prime 关闭。
同一 L4、同一 `.tmp/hybrid-quick-reply-v1`，先 compact 后 raw，warmup 0、各 1 次。
其他进程占用显存，其他卡也有测试/运行，属于 **pilot**，不是 clean 三次中位数。
仅事后 checker 读取 tour/cost；solver 不接收标签或预处理图。

| 配置 | wall | Quick | extra-edge | 全链 replay | 最终 edge / fixed / nonpair |
| --- | ---: | ---: | ---: | ---: | --- |
| raw | 181.994 s | 44.740 s | 39.809 s | 36.464 s | 1246 / 26 / 2392 |
| compact | 103.764 s | 4.544 s | 35.608 s | 2.508 s | 1246 / 26 / 2392 |

同二进制端到端 **1.754×**，wall 减少 **42.985%**。三份输出经 `cmp` 逐字节相同，
state hash 均为 `6713bf43b73a784a`，事后最优 tour 冲突和 replay 拒绝均为零。
Main 为 44.909/45.150 s，Point 为 11.464/11.402 s，未把波动归因于该优化。
compact 各 epoch 累计 raw pair 116,477,625、compact pair 44,344,540；缓存容量
160,183,807 bytes，build 120.745 ms、覆盖验证 189.585 ms。
该 pilot 支持继续评估，不构成四例正式验收，也不是与论文的速度比较。
记录：`artifacts/hybrid-pr299-quick-reply-ab-v1/`；二进制 SHA256：
`9a38589b13add6e82dd38e5fce99976177b7920d58bcae300740312060a46bf2`。
该冻结版本与本轮最终构建的算法相同，早于新增测试进入 CMake 和头文件格式化；
不同文件哈希必须保留各自构建身份，不能伪装成同一二进制。

最终构建冻结 `.tmp/hybrid-quick-reply-v2` SHA256：
`0530f46f74a8852fa1805370b379a1c5ea342fb4fe5c5dff205940b6923f3f6d`。
用相同配置启动四例时，cuda19 L4 仅剩约 8 GB 显存，vm1084 在首次完整建流分配
失败，9.061 s 安全退出，无最终输出，不能算作完成时间或删边强度。
失败记录 `artifacts/hybrid-four-l4-quick-reply-v2/` 原样保留。
改用 cuda20 的 L40S（启动时约 33 GB 剩余显存）运行相同四例序列，
`artifacts/hybrid-four-l40s-quick-reply-v2/`。该卡有高计算占用，先检查完整运行和
终态，不用这次时间证明 clean 加速。没有采用回复截断或人为目标数。
vm1084 已成功完成工作区分配，进程约占 10.7 GB，仍在首轮 Quick，尚无终态。
同时在 cuda19 的 L4 上以同一 v2 配置单独启动 pr1002 完整验证：
`artifacts/hybrid-pr1002-l4-quick-reply-v2/`。其时间不得直接除以旧 Ada 时间作为
同硬件加速，各次 GPU/host 身份均保留在原始记录中。

## 5. 更新的强度证据与下一步

上一版 v4 的 **pcb1173** 已完整结束：687,378 → **6,493** 条，23 fixed、14,690
nonpairs，wall **1,902.546619 s**；state hash `690c4653d30fd78f`，GPU replay 拒绝及
已知最优 tour 的事后 edge/fix/nonpair 冲突均为零。记录
`artifacts/hybrid-pcb-native-frontier-v4/runs.json`。

2014 为 6,084 条、11,451 s；我们还多 **409 条**，严格胜出至少还要删 410 条。
历史时间比约 6.019×，但不是同强度/同机器加速。GPU 自建 UB 59,387，标签 56,892
仅事后解释，LP 下界约 56,158.7；PDHG violation 0.00135254、relative gap 0.000144206。
Quick 556.994 s、Point 311.283 s、Main 432.873 s、extra-edge 121.278 s、replay 468.191 s。
本轮压缩性能尚不能回填到这次旧版本结果。

pr1002 已有 5,002 vs 2014 的 4,521，仍差 481。vm1084、pcb3038 和作者单核 pr1002
长运行继续保留；未结束之前不填写终态，也不拿中途 partial holder 的边数比较。

A 是相同方法域的执行改进，不能独自补强度。接下来按 B → C → D：动态通用整数
LP cut pool、原始解 SEC/奇边界 2-matching 和收敛；共享多输出 path-cover 后推进
6 paths / 10 reveal；最后设备队列、generation cancellation 与精确反向依赖。
这些仍明确未完成，四实例联合验收未通过。
