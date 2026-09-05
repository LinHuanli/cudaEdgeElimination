# Point 复用与真实 frontier 调度（2026-09-06）

分支：`research/fgpu-hybrid-beat-hs2014`；上个已提交检查点：`704554b`。
目标仍为[四实例强度及双速度门槛](../design/FGPU_Hybrid_Beat_HS2014_Execution.md)。
本记录不表示完整方案或四例验收已经完成。

## 实现与不变式

1. `--leaf-permutation-cache 1`：GPU 一次生成 1–8 个内部点的全部排列，按
   `(position,rank)` 保存，大小 **362,879 bytes**。独立 GPU replay 用输出排列重建
   Lehmer rank，检查范围、唯一性与完整行覆盖后才发布只读目录。叶子仍覆盖同一个
   全排列空间；独立 Point/Direct-Fix replay 仍使用自身的精确枚举。
2. `--point-near-first 1`：先尝试根中心的 32 个几何近点，再回到原来的完整度数桶
   扫描，跳过已尝试点。KD 选择失败则扫描全部点。32 仅是排序前缀，不是搜索上限；
   一个选定 point 的 Hamilton AND replies 仍须全部关闭。
3. `--point-adaptive-start 1`：先在 GPU 上计算真实 `sum_v C(deg(v),2)`，与
   `active_edges × quick_hs_candidates` 比较，再决定是否先收敛 edge services。
   到结束前必须执行完整 Point/Direct-Fix 服务，之后重新唤醒 edge services。
   新增 frontier、推迟轮数、实际 Point sweep 计数和阶段进度日志。
4. 两个 Main 缓存批循环按实际 `batch_count` 推进，避免最后一次固定步长加法超出
   `int32`。全部根仍完整覆盖。

三个性能开关目前均为显式实验选项，默认关闭；不能在没有完整测量前把它们全部当成
最佳默认配置。执行顺序变化可能影响有限局部域的最终强度，必须独立检查，不能只看 wall。

### 为什么旧的延后逻辑没有生效

首次 LP/pair epoch 之前没有分配 nonpair mask，`current_pair_count` 的初值为 0。
旧代码直接拿它判断 frontier，因而总是立即运行 Point。CSR 已完成压缩并不意味着
这个延迟创建的 pair 计数已经有效。

已有完整图运行中，geometry 后 `pr1002` 为 23,927 条边、`pr299` 为 5,905 条。
由度数和即可得出 pair frontier 分别至少约 111.9 万、22.7 万，而相应 edge frontier
为 382,832、94,480；旧的零值显然不能代表这两种工作量。修复只改变服务顺序，
没有目标数量、reply 数量、epoch 或时间截断。

### 强制外部连接的成本界

通用 path-order、两路径距离缓存和 GPU warp 叶子统一改为：若真实内部路径成本为
`L`，每条必须保留的外部连接赋值 `-(L+1)`。因为所有真实距离非负，遗漏任意一条
外部连接的排列都不可能严格优于原排列。入口的 `max(cost) ≤ INT64_MAX/64`
以及当前至多 10 点/3 路径域保证中间和安全。

旧的 `INT_MIN/(N-1)` 在大成本非度量路径单元测试中存在遗漏外部连接的假改进。
这里没有发现并声称当前四个 TSPLIB 实例误删；修复的是通用谓词的数值前提。
CPU 独立 oracle 直接强制外部连接，不复用上述负成本技巧。

## 已完成的完整图 pilot

以下三次使用同一冻结二进制 `.tmp/hybrid-point-catalog-v0`，原始坐标为
`../references/LocalElimination/data/pr299.tsp`；不输入 tour/cost/预处理边。
最优成本 48,191 只供独立事后检查。节点存在其他训练/参考作业，**均不是 clean 验收样本**。

| 配置 | 剩余边 | fixed | nonpair | 进程 wall | Point |
| --- | ---: | ---: | ---: | ---: | ---: |
| LP off，两个开关均关闭 | 1246 | 26 | 2388 | 245.613 s | 119.182 s |
| LP off，排列目录+近点优先 | 1246 | 26 | 2388 | 214.278 s | 93.438 s |
| GPU primal-dual-sec，排列目录+近点优先 | 1246 | 26 | 2398 | 189.397 s | 83.564 s |

前两行终态 hash 均为 `9b143dd240f33295`：wall **1.146×**、Point **1.276×**，
端到端减少约 12.76%。这是两个开关合并的单轮对照，尚未分别归因。
第三行 hash 为 `8486101e823cf762`；它产生不同 nonpair 状态，不可冒充同终态缓存消融。
其 GPU incumbent 为 49,014，LP 下界为 46,355.9；现有 LP 尚未满足计划中的收敛门槛，
也未在此例进一步降低剩余边数。

二进制 SHA256：`bc2e1302ffa0077cebcf99698e2a58e3726ad0d629a58c86f96c0894e2fdc8cf`。
源码树 SHA256：`6215ea89f6bc45c4ba69ee1e015c8f884f3ec781c1171fd9610cef46f36c6050`。
原始记录：`artifacts/hybrid-pr299-point-ab-v0/`、
`artifacts/hybrid-pr299-native-point-v0/`，每行均通过 edge/fix/nonpair 最优标签事后检查。
v0 尚不含强制连接界和 adaptive-start 修复，不把上述数据写成后续二进制的结果。

### 修正调度后的独立对照

同一个冻结二进制 `.tmp/hybrid-point-frontier-v2`、本机同一张 RTX 4000 Ada，
两行均开启 primal-dual-sec、排列目录与近点优先，只切换 adaptive-start。

| Point 首次执行策略 | 剩余边 | fixed | nonpair | 进程 wall | Point |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager，立即执行 | 1246 | 26 | 2398 | 200.948 s | 92.250 s |
| adaptive，按实际 frontier 延后 | 1246 | 26 | 2392 | 175.083 s | 8.739 s |

真实初始 pair frontier 为 248,285，而 edge frontier 为 94,480。adaptive 先完成
7 轮没有 Point 的服务轮次，随后实际执行 4 轮完整 Point sweep，最终进入联合固定点。
wall **1.148×**，减少 12.87%；Point **10.557×** 不能冒充端到端加速，因为
Main、Quick、extra-edge 和 replay 分别增加约 19.20、11.97、16.52、9.21 秒。
两行最优 tour 检查通过，但 nonpair 终态不同，不声称完整状态 hash 一致。
此对照同样是单次开发 pilot。记录：`artifacts/hybrid-pr299-frontier-ab-v2/`。

## 验证记录与复现

- CPU 37/37：`artifacts/hybrid-cpu-tests-point-v2.log`。
- GPU 排列目录逐行对照 CPU `next_permutation`；bootstrap 24 例，目录 memcheck/racecheck
  均为零错误/零 hazard：`artifacts/hybrid-sanitizer/catalog-{memcheck,racecheck}.log`。
- 强制连接界：56 组独立 CPU 路径检查含 10 点、零成本、大成本和安全上界；64 组
  CPU/GPU warp-DP 差分含大成本非度量输入。冻结 v1 的 unit.core 已通过。
- 冻结 v1 的 8 节点完整 solve（两个开关开启）memcheck/racecheck 均通过：
  `artifacts/hybrid-sanitizer/point-v1-{memcheck,racecheck}.log`。
- 6 节点十亿量级坐标完整求解通过全最优 tour 检查，最优成本 6,477,669,374；
  输出 `artifacts/hybrid-large-after/`。
- 包含 adaptive-start 的 CUDA 全套 **68/68**，386.52 s：
  `artifacts/hybrid-cuda-tests-point-v2.log`。14 组小图共 208 次无标签完整求解，
  包含两组大坐标 LP-off 样例；18/19/24 点零成本完全图验证等号边界、真实计数、
  推迟和最终完整回扫（全部最优边/邻边对必须保留，不穷举阶乘数量的 tour）。
- v2 的 19 节点密图 adaptive-start memcheck 零错误、racecheck 零 hazard；
  此专项关闭 full-metric 以隔离调度，全度数 metric 已由完整测试及先前专项覆盖。
  日志：`artifacts/hybrid-sanitizer/frontier-v2-{memcheck,racecheck}.log`。

可用 `tools/benchmark_fgpu.py --profile hybrid-e2e` 做单因素交错实验。此模式的
`--tour`、`--expected-cost` 仅由外部事后检查器读取，不会传给计时求解进程；
单元测试覆盖这一边界。所有原始记录、编译缓存和二进制均留在项目内。
采集器新增节点/CPU 身份检查；同机验收拒绝跨节点、混合 GPU 型号或缺少主机身份的记录。

本机 CUDA 13.3 构建采用 `--split-compile=4 -Xptxas=--split-compile=4` 缩短编译等待，
没有启用 fast-math。后续 A/B 应使用同一编译身份，避免把编译差异混入开关收益。

## 尚待完成

`pr1002` 全度数版本 `.tmp/hybrid-main-cache-v1` 已完成：501,501→**5,002** 条边，
24 fixed、10,297 nonpairs，进程 wall **3,176.233 s**。GPU replay 拒绝为 0，
最优 tour edge/fix/nonpair 冲突为 0。相比旧低度 metric 的 5,619 条少 617 条，
但仍比 2014 多 481 条；历史论文时间比 2.840× 不是等强度加速。
Point 为 2,399.4 s（75.54%），Main 为 180.798 s，Quick 为 245.084 s，
extra-edge 为 125.338 s，replay 为 223.317 s。原始数据：
`artifacts/hybrid-pr1002-full-metric-pilot/`。该版本不含本报告的 Point 性能开关。

新配置完整 `pr1002` 已完成于本机同一张 RTX 4000 Ada：**1,161.415 s**，
仍为 **5,002 edges / 24 fixed / 10,297 nonpairs**。与上个全度数版本的最终
state hash 同为 `697baad650b13d0f`，GPU replay 拒绝及最优标签冲突均为零。
单次 pilot 的 wall 比为 **2.735×**；同时切换了原生 LP、排列目录、近点优先和
自适应调度，因此这不是一个开关的单因素收益。Point 从 2,399.4 s 降至 146.708 s，
但 Quick/Main/replay 分别增至 356.877/226.252/300.238 s。
GPU 自建 incumbent 为 266,046；原生 LP 最终下界约 253,857，原始可行性偏差
0.0034712、相对 gap 0.000140767，均未达计划的 `1e-5`。
历史论文时间比为 **7.767×**，但仍多 481 条边，不能称为等强度加速。
记录：`artifacts/hybrid-pr1002-native-frontier-v2/`，节点有其他作业，`clean=false`。

作者单核全量对照仍在运行，未结束前不填写最终边数或速度比。
`cuda19` 空闲 L4 的 GPU smoke 已通过
全最优解检查，并启动同配置 `vm1084`：`artifacts/hybrid-vm1084-l4-point-v2/`。
L4 与本机计时分开，不混作同一硬件的正式四例加速比。
四例门槛尚未通过。动态通用 cut pool、primal mincut/奇边界 2-matching、统一
6-path/10-reveal continuation、多输出 path-cover、设备队列与精确依赖唤醒等，
仍按执行计划继续，不因本次局部优化有收益就标为完成。
