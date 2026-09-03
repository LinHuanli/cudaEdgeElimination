# FGPU 证书、数值与失败关闭契约

## 1. 信任模型

FGPU 当前采用两层模型：

1. CUDA 负责大规模候选生成、exact integer cost kernel、PDLP multiplier 和 AND–OR 状态传播；
2. CPU verifier 在同一不可变快照上重算最终数学谓词，并且是删除授权的唯一边界。

因此：

- GPU 阴性结果只表示 `UNRESOLVED/EXHAUSTED`，目标边保留；
- GPU 阳性必须携带完整见证；
- 任一候选失败会停止整个 epoch，不产生部分提交；
- 全部输出前从初始图再次重放完整证书链；
- `aggressive-fp32` 也不能绕过 exact verifier。

## 2. V4 外层证书

当证书包含 LP 删除时，格式头为 `CUDAEE_PROOF_V4`。主要对象是：

```text
record <epoch> <snapshot_hash> <edge_id> <u> <v> <method>
       <witness> <certificate_index> <second_witness>

lp_box_proof <index> <snapshot_hash> <fractional_bits> <incumbent_cost>
lp_box_dual <dimension> <q_0> ... <q_(n-1)>
end_lp_box_proof
```

方法与 payload：

| method | payload | verifier |
|---|---|---|
| `GEOM_MAIN` | 两个 distinct potential points | exact integer compatibility + MPFR interval Main-Edge |
| `JV` | 一个中心见证 | 独立 CPU JV 谓词 |
| `HT` | 唯一 inner continuation sidecar index | 完整 Hamilton replies、leaf witness、AND–OR 重放 |
| `LP_BOX` | 可共享 LP sidecar index | `__int128` 完整 box bound 与 forced-one 严格比较 |

LP sidecar 可由同一 snapshot 的多条边共享；HT sidecar 必须唯一引用。孤立 sidecar、重复 HT 引用、方法与 payload 不匹配、epoch 不连续、哈希断链、边端点错配和尾随字段均被拒绝。

## 3. 数值边界

### 3.1 距离

- EUC_2D 最终距离使用整数坐标和修正后的整数平方根；
- CEIL_2D 的 local/HT/JV 路径继续使用 exact integer distance；
- strongly-potential 几何当前仅在安全整数坐标 EUC_2D 上启用；其他类型自动跳过该阶段；
- local move 的最终判断始终是 `int64/__int128` 的严格成本比较。

### 3.2 几何

GPU filter 可使用 FP32/FP64，但 proof replay 以 256-bit MPFR outward intervals 重建角度代数式。区间无法证明严格大于零时保留边。

### 3.3 LP

PDLP/subgradient 的 double multiplier 本身不是证明。它先量化为公共分母 `2^fractional_bits`；默认 24 bits，最大 40 bits。验证器检查：

- multiplier 有限且维度等于顶点数；
- snapshot hash 与 record epoch 一致；
- incumbent 非负；
- 每条活动变量都进入 reduced-cost box contribution；
- 所有乘加无 `__int128` overflow；
- forced-one bound 严格大于 `U*D`，等号不能删除。

不活动变量不属于当前 sparse relaxation；该 sidecar 只能用于其绑定的快照。

## 4. 原子 epoch 重放

同一 epoch 的验证流程为：

```text
计算一次 snapshot hash
  -> 构造一次 geometry nearest-neighbor / LP reduced-cost 共享数据
  -> 并行验证每个 record
  -> 按原 record 顺序选择首个错误
  -> 规范排序与去重
  -> 在 degree 副本上执行最小度 2 门禁
  -> 重建 CSR
  -> 原子替换工作图
```

并行只改变纯只读谓词的执行时序，不改变 proof 顺序、提交边集或最终哈希。

## 5. 输出绑定

- `.edg` 必须与 proof replay 的活动边集逐边一致；
- `.fix` 当前等于最终活动图中至少一个端点 degree=2 的所有边，verifier 会重新推导并逐边比较；
- `.nonpairs` 当前必须是 ElimTSP 格式的规范空集合，verifier 会检查 header、每个节点行和尾随字段；
- `.manifest` 记录阶段数、后端、GPU ordinal、LP bound、各类提交数、tour hash 和 wall time；
- 外层运行脚本另记录 Git/GPU UUID/输入与输出 SHA-256。
- CLI 在创建目录前解析完整输出路径的符号链接，库层再拒绝五个输出互相重名或覆盖 instance/input-edges/tour。

## 6. 负向测试

CTest 已覆盖：

- 把几何 potential point 改成目标边端点，必须报 `GEOM_MAIN` 复核失败；
- 把一个 LP multiplier 改成极端但语法合法的 `int64`，重算下界后必须拒绝；
- 从 `.fix` 删除一条记录并同步修改 header，必须因与证明图推导不一致而拒绝；
- 文件级符号链接逃逸、输出重名与输出覆盖输入，必须在写文件前拒绝；
- 已知最优 tour 的成本和每条活动边完整性；
- CPU/CUDA path matching、k-opt、JV、HT 的差分；
- 新几何/PDLP kernel 的 compute-sanitizer memcheck，当前 smoke 为 0 errors。

## 7. 不得使用的表述

在 GPU-only verifier、tile certificate、subtour cuts 和完整 non-pair/fixing proof 落地前，不得声称：

- “fully GPU-certified”；
- “100k 实例已端到端完成”；
- “与 4,016-edge pcb442 固定点等强度 20x”；
- “PDLP 浮点 reduced cost 直接删边”；
- “`ht-epoch-limit` 等于固定点”。

可以声称的是：当前所有已提交删除均有独立 exact replay；失败和预算耗尽均保留边；pcb442/pr1002 的已知最优 tour 门禁为零缺边。
