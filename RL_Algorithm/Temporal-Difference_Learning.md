# 时序差分学习 (Temporal-Difference Learning)

TD 学习是继蒙特卡洛 (MC) 学习之后的第二个**无模型 (model-free)** 方法, 且相较于 MC 具有若干优势

## 1. TD 学习: 状态值估计

### 1.1 问题与算法
目标: 给定策略 $\pi$, 估计**状态价值 $v_{\pi}(s)$** ; 经验样本来自轨迹 $(s_t,r_{t+1},s_{t+1})$.
算法 (TD learning of state values):
  $$
  v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t)\big[v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1}))\big],
  \qquad
  v_{t+1}(s)=v_t(s),\ \forall s\neq s_t
  $$
- **TD 目标** (TD target): $\bar{v}_t \doteq r_{t+1} + \gamma v_t(s_{t+1})$
- **TD 误差** (TD error): $\delta_t \doteq v_t(s_t) - \bar{v}_t = v_t(s_t) - [r_{t+1} + \gamma v_t(s_{t+1})]$

### 1.2 性质与解释

算法驱动 $v(s_t)$ 靠近 $\bar{v}_t$: 由 $|v_{t+1}(s_t)-\bar{v}_t| = |1-\alpha_t(s_t)|\,|v_t(s_t)-\bar{v}_t|$, 且 $0<1-\alpha_t(s_t)<1$ 可得距离在缩小.
TD 误差的解释:
  - 反映相邻时间步的价值差异.
  - 若 $v_t = v_{\pi}$, 则定义 $\delta_{\pi,t}=v_{\pi}(s_t)-[r_{t+1}+\gamma v_{\pi}(s_{t+1})]$ 的期望为零.

该算法仅估计给定策略的**状态价值**, 不估计动作价值, 也不搜索最优策略.

### 1.3 数学本质与收敛性
数学上: 它是求解**贝尔曼期望方程**的无模型算法:
  $$v_{\pi}(s) = \mathbb{E}[R + \gamma v_{\pi}(S') \mid S=s]$$
重写为 $g(v(s)) = v(s) - \mathbb{E}[R+\gamma v_{\pi}(S')|s] = 0$ , 用 $r,s'$ 构造噪声观测, 代入 RM 算法即得 TD 更新.
**收敛定理**: 若 $\sum_t \alpha_t(s)=\infty$ 且 $\sum_t \alpha_t^2(s)<\infty$ (对所有 $s$ ), 则 $v_t(s)$ 以概率 1 收敛到 $v_{\pi}(s)$.  
  实际中常取常数 $\alpha$ , 此时算法在期望意义上收敛.

### 1.4 TD 学习与蒙特卡洛学习的对比
|  | TD / Sarsa | MC |
|--|-------------|-----|
| 更新方式 | 在线 (online) : 每次获得奖励立即更新 | 离线 (offline) : 必须等到整个回合结束 |
| 任务类型 | 可处理持续任务 (continuing tasks) | 只能处理有终止状态的回合任务 (episodic tasks) |
| 自举 (bootstrapping) | 自举: 更新依赖先前估计, 需要初始猜测 | 无自举: 直接基于完整回报估计 |
| 估计方差 | 方差较低 (涉及的随机变量较少) | 方差高 (回报累积多个随机奖励) |

## 2. Sarsa: 动作值估计与策略改进
### 2.1 算法与收敛性
目标: 估计给定策略 $\pi$ 的动作价值 $q_{\pi}(s,a)$.

经验样本: $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ (State-Action-Reward-State-Action, 故称 Sarsa).

更新公式:
  $$
  q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - \alpha_t(s_t,a_t)\big[q_t(s_t,a_t) - (r_{t+1} + \gamma q_t(s_{t+1},a_{t+1}))\big]
  $$

数学本质: 求解贝尔曼方程的动作值形式:
  $$q_{\pi}(s,a) = \mathbb{E}[R + \gamma q_{\pi}(S',A') \mid s,a]$$

收敛条件与状态值 TD 类似: $\sum \alpha_t =\infty,\ \sum \alpha_t^2 < \infty$.

### 2.2 与 Policy Imporve 结合 (寻找最优策略)

将 Sarsa 与策略改进步骤组合, 整体仍称为 Sarsa. 

对每回合, 从初始状态-动作开始, 收集 $(r_{t+1},s_{t+1},a_{t+1})$, 更新 $q(s_t,a_t)$, 然后用 $\epsilon$-greedy 立刻更新策略.

策略使用 $\epsilon$-greedy 以平衡探索与利用.

## 3. n-step Sarsa: 统一 Sarsa 与 MC
- 定义 n 步回报:
  $$
  G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_{\pi}(S_{t+n},A_{t+n})
  $$
- n-step Sarsa 更新 (在 $t+n$ 时刻执行):
  $$
  q_{t+n}(s_t,a_t) = q_{t+n-1}(s_t,a_t) - \alpha\big[q_{t+n-1}(s_t,a_t) - (r_{t+1}+\gamma r_{t+2}+\dots+\gamma^n q_{t+n-1}(s_{t+n},a_{t+n}))\big]
  $$
- 当 $n=1$ 即为 (单步) Sarsa; $n\to\infty$ 即退化为 MC 学习.  
  n 较大 → 接近 MC (高方差, 低偏差); n 较小 → 接近 Sarsa (低方差, 初始猜测导致的偏差较大).


## 4. Q-learning: 直接学习最优动作价值

### 4.1 算法与数学解释
- Q-learning 直接估计最优动作价值 $q^*(s,a)$.
- 数学上求解**贝尔曼最优方程**:
  $$
  q(s,a) = \mathbb{E}\big[R_{t+1} + \gamma \max_a q(S_{t+1},a) \mid S_t=s, A_t=a\big]
  $$
- Q-learning 更新公式:
  $$
  q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - \alpha\big[q_t(s_t,a_t) - (r_{t+1} + \gamma \max_a q_t(s_{t+1},a))\big]
  $$
- 由学习到的 q 值可直接得到贪心策略, 从而获得最优策略.

### 4.2 On-policy 与 Off-policy

- **行为策略** (behavior policy): 生成经验样本的策略; **目标策略** (target policy): 被优化/评估的策略.
- Sarsa 和 MC 均属 **on-policy**: 目标策略 $\pi$ 也是行为策略, 经验由 $\pi$ 生成并用于评估 $\pi$.
- Q-learning 是 **off-policy**: 目标最优策略与行为策略可以不同, 因此可利用任意行为策略生成的样本学习最优策略.

### 4.3 统一视角

所有 TD 算法可写为统一更新格式:
$$
q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - \alpha_t(s_t,a_t)\big[q_t(s_t,a_t) - \bar{q}_t\big]
$$
不同算法的 TD 目标 $\bar{q}_t$ 如下:

| 算法 | TD 目标 $\bar{q}_t$ |
|------|---------------------|
| Sarsa | $r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1})$ |
| n-step Sarsa | $r_{t+1} + \gamma r_{t+2} + \dots + \gamma^n q_t(s_{t+n}, a_{t+n})$ |
| Q-learning | $r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)$ |
| Monte Carlo | $r_{t+1} + \gamma r_{t+2} + \dots$ (此时 $\alpha_t = 1$， 即 $q_{t+1} = \bar{q}_t$) |

所有 TD 算法都可视为求解贝尔曼方程或贝尔曼最优方程的随机近似算法，具体对应如下:

| 算法 | 求解的方程 |
|------|-----------|
| Sarsa | BE: $q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]$ |
| n-step Sarsa | BE: $q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots + \gamma^n q_\pi(S_{t+n}, A_{t+n}) \mid S_t = s, A_t = a]$ |
| Q-learning | BOE: $q(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_a q(S_{t+1}, a) \mid S_t = s, A_t = a]$ |
| Monte Carlo | BE: $q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \dots \mid S_t = s, A_t = a]$ |

其中 BE 是贝尔曼公式, BOE 是贝尔曼最优公式.