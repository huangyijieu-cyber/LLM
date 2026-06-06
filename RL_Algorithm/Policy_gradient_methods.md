# 策略梯度方法 (Policy gradient methods)

## 1. 核心思想: 从基于值到基于策略
**之前 (表格型方法)**: 策略 $\pi(a|s)$ 存储在表格中, 直接查表获取动作概率, 通过直接修改表格条目来更新策略.

**现在 (函数型方法)**: 策略用一个参数化的函数表示 $\pi(a|s,\theta)$ (例如神经网络), 其中 $\theta \in \mathbb{R}^{m}$ 是参数向量.
  - **优势**: 当状态空间很大时, 表格方法在存储和泛化方面效率低下, 而函数表示法更具优势.
- **关键转变**: 在表格情况下, 如果策略 $\pi$ 能使每个状态值最大化, 那么它就是最优的.
- **更新方式**: 策略不能通过改表格条目来更新, 只能通过改变参数 $\theta$ 来更新.

## 2. 定义最优策略的度量 (Metrics)
为了用梯度方法优化策略, 我们需要定义可以最大化的标量目标函数 $J(\theta)$. 根据不同的任务设置, 主要有两种度量方式.

### 2.1 度量一: 平均值 (Average Value)
- **表达式 1**: $\bar{v}_{\pi} \doteq \sum_{s\in \mathcal{S}} d(s) v_{\pi}(s)$

- **表达式 2**: $\bar{v}_{\pi} = \mathbb{E}_{S \sim d}[v_{\pi}(S)]$

- **表达式 3**: $\bar{v}_{\pi} = \lim_{n\to\infty} \mathbb{E} \left[ \sum_{t=0}^{n} \gamma^{t} R_{t+1} \right]$

- **权重分布 $d$ 的选择**:
  - **情况 1 (与策略无关, $d_0$)**: 梯度容易计算, $\nabla_{\theta} \bar{v}_{\pi} = d^{T} \nabla_{\theta} v_{\pi}$.
    - 所有状态同等重要: $d_0(s) = 1/|\mathcal{S}|$.
    - 仅关心特定状态 $s_0$ (如所有情节的起始状态): $d_0(s_0)=1$, 此时 $\bar{v}_{\pi} = v_{\pi}(s_0)$.
  - **情况 2 (依赖于策略, $d_{\pi}$)**: 选择 $d_{\pi}(s)$ 作为策略 $\pi$ 下的稳态分布.
   > $d_{\pi}$ 反映了在给定策略 $\pi$ 下马尔可夫决策过程的长期行为. 如果一个状态在长期运行中被频繁访问, 它就更重要, 应该获得更大的权重.

### 2.2 度量二: 平均单步奖励 (Average Reward)
- **表达式 1**: $\bar{r}_{\pi} \doteq \sum_{s\in \mathcal{S}} d_{\pi}(s) r_{\pi}(s) = \mathbb{E}[r_{\pi}(S)]$,  其中 $S \sim d_{\pi}$.
- **表达式 2**: 沿着一条轨迹的平均单步奖励:
  $$
  \bar{r}_{\pi} = \lim_{n\to\infty} \frac{1}{n} \mathbb{E}\left[ \sum_{t=0}^{n-1} R_{t+1} | S_0 = s_0 \right]
  $$
- **备注**:
  - $\bar{r}_{\pi}$ 是即时奖励的加权平均.
  - $r_{\pi}(s)$ 是在状态 $s$ 下能获得的平均即时奖励.

## 3. 度量函数的梯度

梯度计算是策略梯度方法中最复杂的部分之一.

- **统一表达式**:

$$
\nabla_{\theta} J(\theta) = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_{\theta} \pi(a|s, \theta) q_{\pi}(s, a)
$$

其中 $J(\theta)$ 可以是 $\bar{v}_{\pi}$ 或 $\bar{r}_{\pi}$ 等目标, $\eta$ 是状态的某种分布或权重.

- **对数似然形式**:

利用

$$
\nabla_{\theta} \ln \pi(a|s,\theta) = \frac{\nabla_{\theta} \pi(a|s,\theta)}{\pi(a|s,\theta)}
$$

可将梯度写成:

$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{S \sim \eta, A \sim \pi} \left[ \nabla_{\theta} \ln \pi(A|S, \theta) q_{\pi}(S, A) \right]
$$

这个形式非常重要, 因为它可以用采样来近似梯度:

$$
\nabla_{\theta} J \approx \nabla_{\theta} \ln \pi(a|s, \theta) q_{\pi}(s, a)
$$

这就是随机梯度上升 (Stochastic Gradient Ascent) 的基本思想.

- **实现要求**: 需要 $\pi(a|s,\theta) > 0$ 对所有 $s, a, \theta$ 成立. 通常可以通过 Softmax 策略实现:

$$
\pi(a|s, \theta) = \frac{e^{h(s, a, \theta)}}{\sum_{a' \in \mathcal{A}} e^{h(s, a', \theta)}}
$$

这种策略是随机的, 因而天然具有探索性.

## 4. 梯度上升算法: REINFORCE

REINFORCE 是基于蒙特卡洛回报估计的经典策略梯度算法.

### 4.1 算法流程

- 初始化参数 $\theta$, 折扣因子 $\gamma \in (0,1)$, 学习率 $\alpha > 0$.
- 对每个情节执行:
  1. 按照当前策略 $\pi(\theta)$ 生成一条轨迹 $\{s_0, a_0, r_1, \dots, s_{T-1}, a_{T-1}, r_T\}$.
  2. 对 $t = 0, 1, \dots, T-1$:
     - 计算蒙特卡洛回报:

$$
q_t(s_t, a_t) = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k
$$

     - 做策略梯度上升:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \ln \pi(a_t | s_t, \theta) q_t(s_t, a_t)
$$

### 4.2 算法解释

将更新公式改写为:

$$
\theta_{t+1} = \theta_t + \alpha \underbrace{\left( \frac{q_t(s_t, a_t)}{\pi(a_t|s_t, \theta_t)} \right)}_{\beta_t} \nabla_{\theta} \pi(a_t|s_t, \theta_t)
$$

其中 $\beta_t$ 决定动作概率的增量方向和幅度.

- 如果 $\beta_t > 0$, 则在状态 $s_t$ 选择 $a_t$ 的概率会增加.
- 如果 $\beta_t < 0$, 则在状态 $s_t$ 选择 $a_t$ 的概率会降低.

从探索与利用的角度看:

- **利用 (Exploitation)**: $\beta_t$ 与 $q_t(s_t, a_t)$ 成正比. 动作价值越大, 该动作下次被选中的概率就越大.
- **探索 (Exploration)**: $\beta_t$ 与 $\pi(a_t|s_t, \theta_t)$ 成反比. 当前概率越低, 一旦尝试后得到较高回报, 概率增幅就越大.

### 4.3 采样方式 (On-Policy)

动作 $A$ 是从当前策略 $\pi(\theta)$ 中采样得到的, 因此 REINFORCE 是 On-Policy 方法.
