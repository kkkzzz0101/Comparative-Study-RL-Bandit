# 环境使用手册

[English version](environments.md)

本文档是当前 benchmark 中所有可调用环境的代码对应手册。Python 实现位于 `src/bandit_benchmark/environments/`，配置文件中的调用名称定义在 `src/bandit_benchmark/registry.py`。

## 1. 数据模型

每个 full-table 环境都实现以下接口：

```python
generate(horizon: int, scenario_seed: int, reward_seed: int) -> Scenario
```

一个 `Scenario` 包含：

| 字段 | 形状/类型 | 含义 |
| --- | --- | --- |
| `expected_rewards` | `(T, K)` 浮点数组 | 隐藏的期望奖励 `mu[t, arm]`；评估程序知道，算法不知道 |
| `reward_table` | `(T, K)` 浮点数组 | 每一轮、每个 arm 预先生成的潜在奖励 |
| `kind` | 字符串 | `stochastic`、`nonstationary` 或 `adversarial` |
| `metadata` | 字典 | 随机种子、变化点、variation、gap 和生成器专属信息 |

在第 `t` 轮，算法选择一个 arm，只能得到：

```python
scenario.reward_table[t, selected_arm]
```

算法永远看不到完整的一行，也看不到 `expected_rewards`。预先生成整张表可以保证相同随机种子下，不同算法面对完全相同的潜在结果。

三个随机源彼此分离：

- `scenario_seed`：在适用时控制隐藏均值轨迹、变化点、leader 或相位；
- `reward_seed`：控制 Bernoulli 潜在结果；
- `policy_seed`：控制算法自身随机性，由 runner 分配。

## 2. 快速调用

用 random sanity policy 运行全部核心环境：

```bash
bandit-benchmark run --config configs/smoke.toml
```

当前 smoke 配置会运行 stationary、canonical abrupt、random abrupt、gradual drift、bounded random walk、两种 hazard mode 和 rapid switching。

一个 TOML 环境配置项由 registry `name`、可选实验 `id` 和构造参数组成：

```toml
[[environments]]
name = "hazard_switching"
id = "hazard_per_arm"

[environments.params]
n_arms = 5
hazard_rate = 0.01
mode = "per_arm"
```

同一种生成器以不同参数多次出现时，必须提供唯一 `id`。结果中的 run ID 和 summary 使用该 ID 分组，`environment_name` 则保存底层 registry 名称。

直接通过 Python 调用：

```python
from bandit_benchmark.registry import build_environment

environment = build_environment(
    "random_abrupt",
    {
        "n_arms": 5,
        "n_changes": 3,
        "min_segment_length": 100,
        "min_jump": 0.05,
        "max_jump": 0.30,
    },
)
scenario = environment.generate(
    horizon=1000,
    scenario_seed=7,
    reward_seed=10_007,
)

print(scenario.expected_rewards.shape)       # (1000, 5)
print(scenario.metadata["change_points"])
print(scenario.metadata["total_variation"])
```

## 3. 当前可用环境

| Registry 名称 | Python 类 | Kind | 变化机制 |
| --- | --- | --- | --- |
| `stationary` | `StationaryBernoulli` | stochastic | 固定 Bernoulli 均值，可控制 top-arm gap |
| `canonical_abrupt` | `CanonicalAbruptBernoulli` | nonstationary | 已发表的三臂、两个变化点回归案例 |
| `random_abrupt` | `RandomAbruptBernoulli` | nonstationary | 不规则变化点、部分 arm 变化、有界跳变 |
| `gradual_drift` | `GradualDriftBernoulli` | nonstationary | 带相位差的正弦均值轨迹 |
| `bounded_random_walk` | `BoundedRandomWalkBernoulli` | nonstationary | 独立 Gaussian 均值随机游走并做反射 |
| `hazard_switching` | `HazardSwitchingBernoulli` | nonstationary | 随机 global 或 per-arm 重抽事件 |
| `rapid_switching` | `RapidSwitchingBernoulli` | nonstationary | 快速 block leader 切换并加入结果翻转 |
| `csv_mean_path` | `CSVMeanPathEnvironment` | nonstationary | 从 CSV 读取 CTR/mean 列，再采样 Bernoulli |

兼容名称：

| Registry 名称 | 含义 |
| --- | --- |
| `abrupt_change` | 旧版等间隔、全局重抽生成器；为旧配置保留 |
| `oblivious_adversarial` | `rapid_switching` 的弃用名称；现在正确地按 nonstationary 评估 |

`abrupt_change` 接受必填 `n_arms` 和可选 `n_changes=3`。兼容生成器仅用于旧配置，不属于正式实验集合；新实验应使用文档中的正式名称。

当前正式环境集合中，没有任何生成器被描述为 canonical strategic adversary。真正的 deterministic/adaptive adversarial suite 需要单独声明 payoff 模型和 comparator。

## 4. 各环境说明

### 4.1 `stationary`

所有 arm 的均值从头到尾保持不变。`scenario_seed` 决定这些均值如何随机分配到 arm index。

| 参数 | 默认值 | 限制/作用 |
| --- | ---: | --- |
| `n_arms` | 必填 | 至少为 2 |
| `low` | `0.15` | 候选均值下界；两个 arm 且显式设置 `gap` 时，实际较低均值为 `high-gap` |
| `high` | `0.85` | 最优均值；满足 `0 <= low < high <= 1` |
| `gap` | 不填写 | 如果提供，则固定最优和第二优 arm 的均值差 |

可以用多组 `gap` 构造简单和困难的识别问题。

### 4.2 `canonical_abrupt`

这是 Garivier–Moulines 的三臂 Bernoulli 案例。`T=10,000` 时：

```text
t < 3000 或 t >= 5000: [0.5, 0.3, 0.4]
3000 <= t < 5000:      [0.5, 0.3, 0.9]
最优 arm:                 0 -> 2 -> 0
```

其他 horizon 会把变化点缩放到 30% 和 50%。`n_arms` 默认为并且必须等于 `3`，`horizon` 至少为 `3`。均值轨迹是确定的，但为了统一接口仍然会记录 `scenario_seed`。

### 4.3 `random_abrupt`

这是主要的不规则 piecewise-stationary 生成器。它在满足最短 segment 长度的前提下随机抽取变化点。每个变化点上，各个 arm 按给定概率独立决定是否改变，并保证至少有一个 arm 改变；不会强制最优 arm 每次都发生变化。

| 参数 | 默认值 | 限制/作用 |
| --- | ---: | --- |
| `n_arms` | 必填 | 至少为 2 |
| `n_changes` | `5` | 正整数变化点数量 |
| `min_segment_length` | `100` | 相邻边界之间的最少轮数 |
| `arm_change_probability` | `0.5` | 每个 arm 的变化概率，位于 `(0,1]` |
| `min_jump` | `0.05` | 最小均值跳变幅度 |
| `max_jump` | `0.30` | 最大跳变幅度，不超过 `0.5` |
| `initial_low` | `0.10` | 初始均值下界 |
| `initial_high` | `0.90` | 初始均值上界 |

horizon 必须满足：

```text
horizon >= (n_changes + 1) * min_segment_length
```

### 4.4 `gradual_drift`

每个 arm 都遵循带相位差的正弦轨迹：

```text
mu[t, a] = 0.5 + amplitude * sin(time_phase[t] + arm_phase[a] + phase_offset)
```

| 参数 | 默认值 | 限制/作用 |
| --- | ---: | --- |
| `n_arms` | 必填 | 至少为 2 |
| `cycles` | `1.5` | horizon 内的正周期数量，必须为正 |
| `amplitude` | `0.3` | 位于 `(0,0.45]` |

它适合控制 smooth drift，但不应该只用周期轨迹评价一个算法。

### 4.5 `bounded_random_walk`

每个 arm 独立执行 Gaussian 随机步：

```text
mu[t, a] = reflect(mu[t-1, a] + Normal(0, step_std))
```

这里使用 reflection 而不是 clipping，确保数值位于 `[0,1]`，同时避免大量概率被人为堆积在恰好 0 或 1 的边界上。

| 参数 | 默认值 | 限制/作用 |
| --- | ---: | --- |
| `n_arms` | 必填 | 至少为 2 |
| `step_std` | `0.03` | 非负的漂移尺度 |
| `initial_low` | `0.20` | 初始均值下界 |
| `initial_high` | `0.80` | 初始均值上界 |

### 4.6 `hazard_switching`

首先均匀抽取初始均值。之后每一轮都可能触发 hazard event，重新抽取所有 arm 或某些单独 arm 的均值。

| 参数 | 默认值 | 限制/作用 |
| --- | ---: | --- |
| `n_arms` | 必填 | 至少为 2 |
| `hazard_rate` | `0.001` | 位于 `[0,1]` 的事件概率 |
| `mode` | `"global"` | `"global"` 或 `"per_arm"` |
| `redraw_low` | `0.0` | 重抽下界 |
| `redraw_high` | `1.0` | 重抽上界 |

global 模式中，一次事件会重抽全部 arm；per-arm 模式中，每个 arm 在每一轮都有独立的事件概率。

### 4.7 `rapid_switching`

每个 block 有一个高均值 leader，其余 arm 为低均值。默认情况下相邻 block 的 leader 不同。完成 Bernoulli 采样后，每个结果独立地以 `corruption` 概率翻转。

用于评估的有效均值为：

```text
p_effective = corruption + (1 - 2 * corruption) * p
```

| 参数 | 默认值 | 限制/作用 |
| --- | ---: | --- |
| `n_arms` | 必填 | 至少为 2 |
| `block_size` | `50` | 每个 leader block 的正整数轮数 |
| `base_mean` | `0.15` | 非 leader 的 Bernoulli 均值 |
| `leader_mean` | `0.85` | leader 均值，必须大于 `base_mean` |
| `corruption` | `0.10` | 位于 `[0,0.5)` 的独立翻转概率 |
| `force_leader_change` | `true` | 防止相邻 block 使用相同 leader |

它是高变化随机压力测试，不代表已经证明可以应对战略性 adversary。当前 runner 对它使用 dynamic pseudo-regret。

## 5. Metadata 和结果记录

每个生成的 scenario 至少记录以下可 JSON 序列化的 metadata：

- `scenario_seed`、`reward_seed` 和 `reward_model`；
- `change_family`、`change_points` 和 `changed_arms`；
- `total_variation` 和 `max_step_change`；
- `minimum_gap`；
- `optimal_switch_points` 和 `n_optimal_switches`；
- 审计 scenario 所需的生成器专属信息和实际事件。

`change_points` 表示生成器声明的离散事件，例如 breakpoint 或 redraw；smooth drift、random walk 和 CSV mean path 的该字段为空。`optimal_switch_points` 根据 mean table 计算 argmax 的变化，因此即使 `change_points` 为空，它也可能非空。

runner 会把整个字典写入 `runs.csv` 的 `scenario_metadata` 列，因此之后可以重建并审计随机变化时间表。

## 6. 真实数据接口

### 6.1 已实现：完整 mean-path/CTR 桥接接口

`MeanPathDataEnvironment` 是 semi-synthetic 数据源的抽象接口，要求数据源能够提供完整 `(T,K)` 期望奖励表。其公共 `generate()` 会验证数据并生成可公平比较的 Bernoulli 潜在结果。

`CSVMeanPathEnvironment` 是第一个可调用适配器。CSV 必须包含表头以及至少 `horizon` 行：

```csv
time,item_a,item_b,item_c
0,0.10,0.20,0.30
1,0.12,0.18,0.32
2,0.16,0.17,0.29
```

```toml
[[environments]]
name = "csv_mean_path"

[environments.params]
path = "data/ctr_mean_path.csv"
reward_columns = ["item_a", "item_b", "item_c"]
delimiter = ","
```

| 参数 | 默认值 | 限制/作用 |
| --- | ---: | --- |
| `path` | 必填 | CSV 数据源路径 |
| `reward_columns` | 必填 | 至少两个不重复的表头名称，并按 arm 顺序排列 |
| `delimiter` | `","` | 必须恰好为一个字符 |

被选择的列必须是 finite 且位于 `[0,1]`。相对 `path` 根据进程当前工作目录解析；使用本文命令时通常是仓库根目录，并不是相对于 TOML 文件解析。数据源 mean path 是确定的，因此会记录 `scenario_seed`，但它不会改变轨迹。这个适配器适用于按时间分箱的 CTR 曲线，不适用于原始 logged interaction。

### 6.2 已预留：logged replay/OPE

`LoggedBanditEvent` 和 `LoggedBanditDataset` 预留了未来事件接口，包含：

```text
timestamp, context, available_actions, logged_action, reward, propensity
```

logged data 不包含未选择 arm 的 counterfactual reward，因此不能真实地产生当前的 full `Scenario`。专门的 chronological replay/OPE runner 目前有意留待以后实现。

## 7. 新增环境的方法

1. 继承 `BanditEnvironment` 或 `MeanPathDataEnvironment`。
2. 实现 `generate(...)`；数据 mean path 则实现 `load_expected_rewards(...)`。
3. 返回通过验证的 `Scenario`，并保证 metadata 可以 JSON 序列化。
4. 从 `environments/__init__.py` 导出类。
5. 在 `registry.py` 注册配置调用名称。
6. 添加 reproducibility、数值边界、metadata 和机制专属测试。
7. 在中英文两份环境手册中同步记录准确的构造参数。

对 full-table/oblivious scenario，环境生成代码不得依赖算法做出的 action。
