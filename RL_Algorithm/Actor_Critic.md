# 演员-评论家 (Actor Critic)

## 1. 核心思想

**Actor-Critic 方法仍然属于策略梯度方法** (Policy Gradient Methods), 但它把策略梯度和价值函数方法结合起来.

- **Actor**: 策略更新部分, 负责决定如何行动.
- **Critic**: 价值估计部分, 负责评价 Actor 当前动作的好坏.

回顾策略梯度更新:

$$
\theta_{t+1} = \theta_t + \alpha \nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
$$

从这个式子可以看出:

- $\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)$ 是 **Actor**, 用来更新策略.
- $q_t(s_t,a_t)$ 是 **Critic**, 用来提供评价信号.

## 2. 最简单的 Actor-Critic: QAC

策略梯度方法的关键问题是: **如何得到 $q_t(s_t,a_t)$ ?**

两种方法:

1. **MC 估计**: 用完整回报估计动作值, 对应 REINFORCE.
2. **TD 估计**: 用一步 TD 目标估计动作值, 对应 Actor-Critic.

QAC 使用动作值函数 $q(s,a,w)$ 作为 Critic.

Actor 更新:

$$
\theta_{t+1} = \theta_t + \alpha_{\theta}\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)q(s_t,a_t,w_t)
$$

Critic 更新:

$$
w_{t+1} = w_t + \alpha_w[r_{t+1}+\gamma q(s_{t+1},a_{t+1},w_t)-q(s_t,a_t,w_t)]\nabla_w q(s_t,a_t,w_t)
$$

理解:

- Critic 本质上是 **Sarsa + 函数近似**.
- Actor 根据 Critic 给出的动作值更新策略.
- QAC 是最基础的 Actor-Critic 形式.

## 3. Advantage Actor-Critic (A2C)

### 3.1 Baseline

策略梯度中可以减去一个只依赖状态的 baseline $b(S)$:

$$
\nabla_{\theta}J(\theta)=\mathbb{E}[\nabla_{\theta}\ln\pi(A|S,\theta)(q_{\pi}(S,A)-b(S))]
$$

这样不会改变梯度期望, 因为:

$$
\mathbb{E}[\nabla_{\theta}\ln\pi(A|S,\theta)b(S)] = 0
$$

常用选择:

$$
b(s)=v_{\pi}(s)
$$

于是得到优势函数:

$$
A_{\pi}(s,a)=q_{\pi}(s,a)-v_{\pi}(s)
$$

含义: 一个动作的价值不是看绝对值, 而是看它比当前状态平均水平好多少.

### 3.2 TD Error 近似 Advantage

A2C 中不直接估计 $q_{\pi}(s,a)$, 而用 TD error 近似 advantage:

$$
\delta_t = r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w_t)
$$

因此只需要一个状态值函数 $v(s,w)$.

### 3.3 A2C 算法

Actor 更新:

$$
\theta_{t+1}=\theta_t+\alpha_{\theta}\delta_t\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
$$

Critic 更新:

$$
w_{t+1}=w_t+\alpha_w\delta_t\nabla_w v(s_t,w_t)
$$

备注:

- A2C 是 on-policy 方法.
- $\delta_t>0$ 表示动作比平均水平好, 增大该动作概率.
- $\delta_t<0$ 表示动作比平均水平差, 降低该动作概率.
- 随机策略本身具有探索性, 通常不需要额外使用 $\epsilon$-greedy.

## 4. Off-Policy Actor-Critic

普通策略梯度是 on-policy 的, 因为样本动作来自目标策略 $\pi$.

若样本来自行为策略 $\beta$, 但我们想更新目标策略 $\pi$, 需要使用重要性采样 (Importance Sampling).

基本思想:

$$
\mathbb{E}_{X\sim p_0}[X]=\mathbb{E}_{X\sim p_1}\left[\frac{p_0(X)}{p_1(X)}X\right]
$$

Off-policy 策略梯度:

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{S\sim\rho,A\sim\beta}\left[\frac{\pi(A|S,\theta)}{\beta(A|S)}\nabla_{\theta}\ln\pi(A|S,\theta)q_{\pi}(S,A)\right]
$$

用 TD error 近似 advantage:

$$
\delta_t=r_{t+1}+\gamma v(s_{t+1},w_t)-v(s_t,w_t)
$$

Actor 更新:

$$
\theta_{t+1}=\theta_t+\alpha_{\theta}\frac{\pi(a_t|s_t,\theta_t)}{\beta(a_t|s_t)}\delta_t\nabla_{\theta}\ln\pi(a_t|s_t,\theta_t)
$$

Critic 更新:

$$
w_{t+1}=w_t+\alpha_w\frac{\pi(a_t|s_t,\theta_t)}{\beta(a_t|s_t)}\delta_t\nabla_w v(s_t,w_t)
$$

## 5. Deterministic Actor-Critic (DPG)

之前的策略是随机策略 $\pi(a|s,\theta)$, 而 DPG 使用确定性策略:

$$
a=\mu(s,\theta)
$$

确定性策略的优势:

- 适合连续动作空间.
- $\mu(s,\theta)$ 可以用神经网络表示, 输入状态, 输出动作.

确定性策略梯度:

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{S\sim\rho_{\mu}}[\nabla_{\theta}\mu(S,\theta)\nabla_a q_{\mu}(S,a)|_{a=\mu(S,\theta)}]
$$

DPG 的 TD error:

$$
\delta_t=r_{t+1}+\gamma q(s_{t+1},\mu(s_{t+1},\theta_t),w_t)-q(s_t,a_t,w_t)
$$

Actor 更新:

$$
\theta_{t+1}=\theta_t+\alpha_{\theta}\nabla_{\theta}\mu(s_t,\theta_t)\nabla_a q(s_t,a,w_t)|_{a=\mu(s_t,\theta_t)}
$$

Critic 更新:

$$
w_{t+1}=w_t+\alpha_w\delta_t\nabla_w q(s_t,a_t,w_t)
$$

备注:

- DPG 是 off-policy 方法.
- 行为策略常写成 $\mu(s,\theta)+\text{noise}$, 用噪声完成探索.
- 若 Critic 用神经网络表示, 就得到 DDPG.

## 6. 总结

| 方法 | Critic | 特点 |
|---|---|---|
| QAC | $q(s,a,w)$ | 最基础的 Actor-Critic |
| A2C | $v(s,w)$ | 用 TD error 估计 advantage |
| Off-policy AC | $v(s,w)$ | 用 importance sampling 修正分布差异 |
| DPG | $q(s,a,w)$ | 确定性策略, 适合连续动作 |
