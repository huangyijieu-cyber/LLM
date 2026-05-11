# 强化学习算法 (RL_Algorithm)

## 1. 基本概念

### 1.1 基本要素

| 要素 | 符号 | 说明 |
|------|------|------|
| 智能体 (Agent) | — | 学习者和决策者 |
| 环境 (Environment) | — | 智能体之外的一切，与智能体进行交互 |
| 状态 (State) | `s` / `S` | 对当前情形的描述，来自环境 |
| 动作 (Action) | `a` / `A` | 智能体在某一状态下采取的行为 |
| 奖励 (Reward) | `r` / `R` | 环境对智能体动作的即时反馈: `p(r = 1 \| s1, a1)` |
| 策略 (Policy) | `π` | 从状态到动作的映射: `π(a\|s)` |
| 轨迹 (Trajectory) | `τ` | 一系列交互序列: `s₀, a₀, r₁, s₁, a₁, r₂, …` |
| 回报 (Return) | `G_t` | 从时刻 `t` 开始的累积奖励总和 (可能带折扣) |
| 价值函数 (Value Function) | `V(s)` 或 `V^π(s)` | 在状态 `s` 下，遵循策略 `π` 的期望回报 |
| 动作价值函数 (Q函数) | `Q(s,a)` 或 `Q^π(s,a)` | 在状态 `s` 采取动作 `a` 后，遵循策略 `π` 的期望回报 |

### 1.2 马尔可夫决策过程 (Markov Decision Process, MDP)

#### 1.2.1 集合 (Sets)

- 状态 (State): $\mathcal{S}$ 
- 动作 (Action): $\mathcal{A}$ 
- 奖励 (Reward): $\mathcal{R}$ 

#### 1.2.2 概率分布 (Probabilistic Distribution)

- 状态转移概率: 在当前状态 $s$ 执行动作 $a$ 后, 环境转移到下一状态 $s'$ 的概率: $P(s' \mid s, a)$,
- 奖励概率: 在状态 $s$ 采取动作 $a$ 后获得的即时奖励: $P(r \mid s, a)$.

#### 1.2.3 策略 (Policy)

在状态 $s$, 选择动作 $a$ 的可能性为 $\pi(a \mid s)$.

#### 1.2.4 马尔可夫性质 (Markov Property)

下一状态仅依赖于当前状态和动作, 与历史无关.
$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots)$

## 2. 贝尔曼公式 (Bellman Equation)

详见 **[贝尔曼公式](./Bellman_Equation.md)**.

## 3. 贝尔曼最优公式 (Bellman Optimality Equation, BOE)

详见 **[贝尔曼最优公式 (BOE)](./BOE.md)**.