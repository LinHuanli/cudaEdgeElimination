# 单 GPU 融合方案：超过 2014 的强度与端到端速度

基线：`4e323e2`；工作分支：`research/fgpu-hybrid-beat-hs2014`。
本文冻结用户确认的目标；实现状态另记于 research 文档，不能把计划写成完成结果。

## 验收协议

2014 是对照目标，不是算法边界；允许融合 2014 几何/Direct/Close/metric-excess
与 2023 LP/nonpair/fixing/Hamilton–Tutte。所有状态提交经过同一快照的独立 GPU replay。

| 实例 | 2014 剩余边 | 本方法必须达到 | 2014 总秒数 |
| --- | ---: | ---: | ---: |
| pr1002 | 4521 | ≤4520 | 9021 |
| vm1084 | 4610 | ≤4609 | 4865 |
| pcb1173 | 6084 | ≤6083 | 11451 |
| pcb3038 | 14869 | ≤14868 | 21322 |

来源：[2014 Table 1](https://arxiv.org/pdf/1402.7301)。四例全部通过；另须快于
同机作者源码单核对照，且剩余边不多于该对照。工程期望速度留出至少 1.25 倍余量。
原文历史机器与同机基线的加速比必须分列，不混用。

正式入口从原始 TSPLIB 坐标/完整图开始，不输入 optimum tour/cost/预处理边集。
若 LP 需要 UB，在 GPU 上生成有效 tour，其时间计入端到端。标签仅由独立事后检查器读取。
CPU 可解析和写文件，不参与求解或逐候选精确审计。普通 incumbent 不受保护，
即使其边或邻边对被合法删除，也不得报错。无目标数、回复数、时间、外层 epoch 截断，
不以达到论文边数为提前停止条件；OOM/设备异常明确失败。

端到端主指标为外部进程 wall，包含初始化、图构造、上界、所有求解/replay、输出；
不含独立事后检查。预热 1 次、完整测量 3 次，取中位数，保留所有原始结果。
CPU/GPU 交错执行，不同重实验不共用节点；记录整段 GPU 进程、占用、时钟、温度和 CPU 干扰。
污染样本保留但不计入正式汇总；终态 hash 重复一致、replay 拒绝数与标签冲突数均为零。

## 算法与实现依赖

1. 锁定四例原始坐标、最优标签及 SHA256；准备隔离的作者单核源码对照。
2. GPU 完整图和精确三角距离缓存；128 个确定性 NN 起点，严格改进 2-opt、
   Or-opt 到各自局部不动点；独立 GPU 检查排列和整数成本后发布 UB。
3. LP 与 pair/fixing 解耦。Direct/Close、全度数 metric-excess 用共享整数归约；
   高度数分段 bitset，不直接移除旧昂贵 Opt 检查的度数门槛。
   `(root edge,center,snapshot)` 条件 pair 只算一次，不得发布为全局 nonpair。
4. 动态通用整数系数 LP cut pool，所有活动列，无固定 incidence stride。
   PDHG 原始解上的连通/阈值分量与 GPU push-relabel mincut 分离 SEC；
   添加可独立验证的奇边界 2-matching cuts。分批求解至可行性、相对 gap 均 ≤1e-5。
   浮点结果只引导；整数 cut/dual/edge/fix/path replay 才授权。
5. 统一 edge/nonpair/fixing continuation，声明搜索域 ≤6 paths、10 次 reveal；
   所有可延伸端点加空间邻域中按 surviving replies 排名的 4 个 added-point 候选；
   lazy OR、被选择分支的完整 Hamilton AND。共享端点/fixed/proper subtour/全图环规范化。
6. 叶子低阶→3/4/5-opt coverage→精确多输出 path-cover DP，复用距离与内部路径成本；
   replay 验证成功见证和完整 AND 覆盖，不重跑 proposal 搜索。无改进即开放。
7. 分离 root/reply/leaf/replay kernel，按形状分桶、设备 continuation 队列、
   windowed AND、generation cancellation、快照 transposition；窗口只分批不丢任务。
   精确反向依赖唤醒，结束前完整 service sweep；workspace 可增长复用。

## 接口与配置选择

新入口 `fgpu-elim solve --profile hybrid-e2e --instance INSTANCE.tsp --mode gpu-safe ...`。
保持 legacy 接口和输出兼容，新增上界来源、声明域、LP 收敛与队列/cache 指标。
比较 off/SEC/SEC+2matching 三种 LP 模式及距离缓存、共享叶子的单因素消融；
四例共用一套配置。在全部通过双速度及强度门槛的配置中选择 wall 几何均值最小者，
不允许 instance-name 特调。尚未实现的模式显式拒绝，不伪装成其他后端。

## 作者参考与正确性测试

现有作者源码 README 日期为 2015-02-27，样例 options 不是 Table 1 设置。
对照配置：S1 中点 1/候选 10；S2 23 位置/11 候选；S3 depth10/6 paths/
450 邻域/4 候选，全部目标、迭代至固定点。关闭 tour/nonpair/hyperplane/debug。
只允许构建路径与串行通信适配；与原始单 worker 差分。固定一个物理 CPU 核，
保留源码、补丁、配置和可执行文件 hash。受限源码只在项目内 ignored 目录构建，不提交。

保留已有 CPU 35、CUDA 62 项；增加 n=3–10 全最优 tour 检查（ties、重复坐标、
EUC/CEIL、全图环/真子环），条件 pair 泄漏、普通 incumbent 被合法排除、cut 篡改、
负系数和溢出测试，GPU path-cover 对独立 CPU oracle 的成本/coverage/traceback 差分，
队列扩容/取消/重复完成/空回复/epoch 切换/最终全扫测试。
运行 memcheck/racecheck，并准确记录覆盖范围。

## 完成定义

最终交付一条可复现的完整执行链与四例实际通过的数据，不以局部提速、LP 输入图成功、
或 P0–P8 模块数量宣称完成。多 GPU、100k 扩展、复杂 comb/domino/local-cut lifting
不属于本轮硬范围。所有代码、下载、缓存、构建与输出严格留在本项目目录内。
