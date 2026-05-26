# 值函数近似 (Value function methods)

## 1. 从表格到函数：动机

**表格方法** ：状态值 $v_\pi(s)$ 或动作值 $q_\pi(s,a)$ 直接存储在表格中.
  - 优点: 直观, 易于分析.
  - 缺点: 难以处理大规模或连续的状态/动作空间. 具体体现在两方面:
    1.  **存储**: 需要存储 $|\mathcal{S}|$ 个值, 可能过大;
    2.  **泛化能力**: 无法对未访问状态的值进行合理推断.

**函数近似方法** ：使用参数化函数 $\hat{v}(s, w) \approx v_\pi(s)$ 来近似状态值, 其中 $w \in \mathbb{R}^m$ 是参数向量.
  - **两个根本区别**:
    1.  **如何获取状态值**: 表格是直接查表; 函数需要将状态 $s$ 输入函数计算, 即 $s \to \phi(s) \to \phi^T(s)w = \hat{v}(s,w)$ .
    2.  **如何更新状态值**: 表格直接修改表格中的数值; 函数需要通过更新参数 $w$ 间接改变所有状态的值.
  - **核心优势**:
    1.  **存储**: 参数 $w$ 的维度远小于状态总数 $|\mathcal{S}|$.
    2.  **泛化**: 更新 $w$ 会同时改变"邻近"状态的估计值, 即使它们未被访问.

## 2. 状态值估计算法

**目标**: 对于给定策略 $\pi$ , **找到最优参数 $w$ 使得 $\hat{v}(s,w)$ 尽可能逼近真实状态值 $v_\pi(s)$.** 这是一个策略评估问题.

### 2.1 目标函数

目标函数定义为均方误差:

```math
J(w) = \mathbb{E}[(v_\pi(S) - \hat{v}(S,w))^2]
```

期望是关于状态随机变量 $S$ 的. $S$ 的分布有两种常见选择:

1.  **均匀分布**: 所有状态同等重要, $J(w) = \frac{1}{|\mathcal{S}|}\sum_{s} (v_\pi(s)-\hat{v}(s,w))^2$.
    - **缺点**: 未考虑马尔可夫过程在给定策略下的实际动态, 某些状态很少被访问却被同等看待.
2.  **稳态分布**: 描述马尔可夫过程在长期运行下状态的访问概率, 记作 $d_\pi(s)$ , 满足 $d_\pi(s) \ge 0$ 且 $\sum_s d_\pi(s)=1$.
    - $J(w) = \sum_{s} d_\pi(s) (v_\pi(s) - \hat{v}(s,w))^2$ , 是一个加权平方误差.
    - 更频繁访问的状态具有更高的权重, 反映其实重要性.

### 2.2 优化算法

使用梯度下降最小化 $J(w)$ :

```math
w_{k+1} = w_k - \alpha_k \nabla_w J(w_k)
```

真实梯度为:

```math
\nabla_w J(w) = -2 \mathbb{E}[(v_\pi(S) - \hat{v}(S,w)) \nabla_w \hat{v}(S,w)]
```

用 **随机梯度** 代替期望, 得到参数更新公式:

```math
w_{t+1} = w_t + \alpha_t (v_\pi(s_t) - \hat{v}(s_t, w_t)) \nabla_w \hat{v}(s_t, w_t)
```

由于 $v_\pi(s_t)$ 是未知的待估计量, 必须用近似值替代, 从而产生可实现算法:

- **蒙特卡洛 (MC)**: 用折扣回报 $g_t$ 代替 $v_\pi(s_t)$:

```math
w_{t+1} = w_t + \alpha_t (g_t - \hat{v}(s_t, w_t)) \nabla_w \hat{v}(s_t, w_t)
```

- **时序差分 (TD)**: 用 TD 目标 $r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t)$ 代替 $v_\pi(s_t)$:

```math
w_{t+1} = w_t + \alpha_t [r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t) - \hat{v}(s_t, w_t)] \nabla_w \hat{v}(s_t, w_t)
```

### 2.3 函数近似器的选择

1.  **线性函数近似** (传统广泛使用):

```math
\hat{v}(s,w) = \phi^T(s) w
```

    其中 $\phi(s)$ 是特征向量, 可为多项式基, 傅里叶基等. 此时 $\nabla_w \hat{v}(s,w) = \phi(s)$ , TD 更新变为 **TD-Linear** 算法:

```math
w_{t+1} = w_t + \alpha_t [r_{t+1} + \gamma \phi^T(s_{t+1}) w_t - \phi^T(s_t) w_t] \phi(s_t)
```

    - **缺点**: 难以选取合适的特征向量.
    - **优点**: 理论性质更好理解; **表格表示是线性函数表示的一种特例** (取 $\phi(s) = e_s$ 为单位向量, 则 $\hat{v}(s,w) = w(s)$ ).
2.  **神经网络** (当前广泛使用): 作为非线性函数近似器, 输入为 $s$ , 输出为 $\hat{v}(s,w)$ , 参数为 $w$ (所有权重和偏置).

## 3. Sarsa 与函数近似

扩展到动作值函数 $\hat{q}(s,a,w)$, Sarsa 更新为:

```math
w_{t+1} = w_t + \alpha_t [r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t)] \nabla_w \hat{q}(s_t, a_t, w_t)
```

结合策略评估与策略改进 (如 $\epsilon$ -greedy), 即可搜索最优策略.

## 4. Q-learning 与函数近似

Q-learning 更新为:

```math
w_{t+1} = w_t + \alpha_t [r_{t+1} + \gamma \max_{a \in \mathcal{A}(s_{t+1})} \hat{q}(s_{t+1}, a, w_t) - \hat{q}(s_t, a_t, w_t)] \nabla_w \hat{q}(s_t, a_t, w_t)
```

与 Sarsa 的区别在于用 $\max_a \hat{q}(s_{t+1}, a, w_t)$ 代替 $\hat{q}(s_{t+1}, a_{t+1}, w_t)$.

## 5. Deep Q-learning (DQN)

将深度神经网络引入强化学习的里程碑式算法, 核心是使用非线性网络近似最优动作值函数.

### 5.1 目标函数与梯度

目标函数 (损失函数) 为:

```math
J(w) = \mathbb{E}\left[\left(R + \gamma \max_{a \in \mathcal{A}(S')} \hat{q}(S', a, w) - \hat{q}(S, A, w)\right)^2\right]
```

- **梯度计算难题**: 参数 $w$ 同时出现在 $\hat{q}(S,A,w)$ 和目标值 $y \doteq R + \gamma \max_a \hat{q}(S',a,w)$ 中, 且由于 max 操作, $\nabla_w y \neq \gamma \max_a \nabla_w \hat{q}(S',a,w)$.
- **解决方案**: 引入 **两个网络**:
  - **主网络** $\hat{q}(s,a,w)$ : 负责更新.
  - **目标网络** $\hat{q}(s,a,w_T)$ : 计算目标值, 其参数 $w_T$ 暂时固定.

此时目标函数退化为:

```math
J = \mathbb{E}\left[\left(R + \gamma \max_{a \in \mathcal{A}(S')} \hat{q}(S', a, w_T) - \hat{q}(S, A, w)\right)^2\right]
```

梯度变为易于计算的形式:

```math
\nabla_w J = \mathbb{E}\left[ \left( R + \gamma \max_{a} \hat{q}(S', a, w_T) - \hat{q}(S, A, w) \right) \nabla_w \hat{q}(S, A, w) \right]
```

### 5.2 关键技术

1.  **双网络机制**:
    - 初始化时 $w$ 和 $w_T$ 相同.
    - 每轮从经验回放池中采样 mini-batch $\{(s, a, r, s')\}$.
    - 对每个样本计算目标值 $y_T = r + \gamma \max_{a \in \mathcal{A}(s')} \hat{q}(s', a, w_T)$.
    - 以 $(y_T - \hat{q}(s, a, w))^2$ 为损失训练主网络.
    - 每隔 $C$ 步将主网络参数复制给目标网络.

2.  **经验回放**:
    - 收集的经验样本不按顺序使用, 而是存入 **回放池** $\mathcal{B} \doteq \{(s, a, r, s')\}$.
    - 训练时从池中 **均匀随机** 采样 mini-batch.
    - **必要性**: 目标函数 $J(w)$ 中假设 $(S,A)$ 服从均匀分布 (因无先验知识, 不能使用取决于策略的稳态分布), 但智能体收集的样本是时序相关的. 通过均匀随机采样打破相关性, 满足均匀分布假设.
    - **表格 Q-learning 为何不需要?** 因为表格方法旨在对所有 $(s,a)$ 求解贝尔曼最优方程, 不涉及定义标量目标函数所需的分布.