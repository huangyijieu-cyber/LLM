# 蒙特卡洛方法 (Monte Carlo methods)

## 1. 基本思想

- **蒙特卡洛估计** 是一类依靠 **重复随机采样** 来解决近似问题的技术.
- **核心优势** ：不需要模型即可进行估计.

- **大数定律** ：对于随机变量 $X$, 设 $\{x_j\}_{j=1}^N$ 为独立同分布样本, $\bar{x} = \frac{1}{N}\sum_{j=1}^N x_j$ 为样本均值, 则

```math
  \mathbb{E}[\bar{x}] = \mathbb{E}[X], \quad \mathrm{Var}[\bar{x}] = \frac{1}{N}\mathrm{Var}[X].
```

  $\bar{x}$ 是 $\mathbb{E}[X]$ 的无偏估计, 方差随 $N$ 增大趋近于零.

## 2. Model-free

策略迭代的两个步骤:
- 策略评估: $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$
- 策略改进: $\pi_{k+1} = \arg \max_{\pi}(r_{\pi} + \gamma P_{\pi} v_{\pi_k})$

动作值的两种表达式:
- **需要模型**:

```math
  q_{\pi_k}(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_{\pi_k}(s')
```

- **不需要模型**:

```math
  q_{\pi_k}(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]
```

基于数据估计 $q_{\pi_k}(s,a)$ 的方法: 从 $(s,a)$ 出发, 遵循策略 $\pi_k$ 生成一条 episode, 该 episode 的回报 $g(s,a)$ 即为 $G_t$ 的一个样本. 使用多个样本的平均值估计:

```math
q_{\pi_k}(s,a) \approx \frac{1}{N}\sum_{i=1}^{N} g^{(i)}(s,a).
```

## 3. MC-basic 算法

### 3.1 算法原理

MC Basic 是策略迭代的无模型变体. 在第 $k$ 次迭代中:
- **策略评估**: 对每个 $(s,a)$ 收集足够多的 episode, 用平均 episode 的 $q_k(s,a)$ 近似 $q_{\pi_k}(s,a)$.
- **策略改进**: 求解 $\pi_{k+1}(s) = \arg \max_{\pi} \sum_a \pi(a|s) q_k(s,a)$, 贪心策略为 $\pi_{k+1}(a_k^*|s) = 1$, 其中 $a_k^* = \arg \max_a q_k(s,a)$.

### 3.2 具体实现

1. 初始化: 初始策略 $\pi_0$.
2. 目标: 搜索最优策略.
3. 对于第 $k$ 次迭代 ($k = 0, 1, 2, \dots$), 执行:
   - 对于每一个状态 $s \in \mathcal{S}$, 执行:
     - 对于每一个动作 $a \in \mathcal{A}(s)$, 执行:
       - 回合收集: 收集从 $(s, a)$ 出发, 遵循 $\pi_k$ 的足够多的回合.
       - 策略评估: $q_k(s, a) = $ 所有回合的平均回报.
     - 策略改进:
       - 寻找最大动作值: $a_k^*(s) = \arg\max_a q_k(s, a)$.
       - 策略更新: 如果 $a = a_k^*$, 则 $\pi_{k+1}(a|s) = 1$, 否则 $\pi_{k+1}(a|s) = 0$.

## 4. MC Exploring Starts 算法

### 4.3.1 数据高效利用
- **访问 (Visit)**: 一个状态-动作对在 episode 中出现一次, 称为对该对的一次 Visit.
- **首次访问方法 (Initial-visit method)**: 只使用状态-动作对首次出现后的 episode 来估计其值. 这正是 MC Basic 的做法.
- **更高效的方法**: 利用 episode 中每次访问的回报, 更新多个状态-动作对的值.

### 4.3.2 逐回合更新

两种更新策略的方式:
1. 等待所有与某个状态-动作对相关的回合都收集完毕后, 再用平均回报更新该对的值 (MC Basic 采用).
2. 使用 **单个 episode** 的回报来近似动作值, 可以逐 episode 改进策略.

### 4.3.3 探索开始 (Exploring Starts)

**探索开始条件**: 需要确保从每个状态-动作对 $(s_0, a_0)$ 都有可能被选为 **起始对** 来生成 episode.
- 原因: 只有探索了每个动作值, 才能正确选择最优动作. 如果某个动作未被探索, 它可能恰好是最优的, 从而被遗漏.
- 在实践中, 探索开始很难实现, 特别是在涉及与物理环境交互的应用中, 难以从每个状态-动作对开始收集回合.

### 4.3.4 具体实现

1. 初始化: 初始策略 $\pi_0$ 和初始值 $q(s, a)$; $Returns(s, a) = 0$, $Num(s, a) = 0$.
2. 目标: 搜索最优策略.
3. 对于每一个回合, 执行:
   - 回合生成: 选择起始状态-动作对 $(s_0, a_0)$ (满足探索开始条件), 遵循当前策略生成长度为 $T$ 的回合.
   - 初始化回报: $g \leftarrow 0$.
   - 对于 $t = T-1, T-2, \dots, 0$, 执行:
     - 计算累积回报: $g \leftarrow \gamma g + r_{t+1}$.
     - 更新回报总和: $Returns(s_t, a_t) \leftarrow Returns(s_t, a_t) + g$.
     - 更新访问次数: $Num(s_t, a_t) \leftarrow Num(s_t, a_t) + 1$.
     - 策略评估: $q(s_t, a_t) \leftarrow Returns(s_t, a_t) / Num(s_t, a_t)$.
     - 策略改进: 如果 $a = \arg\max_a q(s_t, a)$, 则 $\pi(a|s_t) = 1$, 否则 $\pi(a|s_t) = 0$.

## 4.4 MC ε-Greedy 算法

### 4.4.1 软策略与 ε-贪心策略

**软策略 (Soft policy)**: 采取任何动作的概率均为正.
**ε-贪心策略**:

```math
\pi(a|s) =
\begin{cases}
1 - \frac{\epsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1), & \text{对于贪心动作}, \\
\frac{\epsilon}{|\mathcal{A}(s)|}, & \text{对于其他 } |\mathcal{A}(s)| - 1 \text{ 个动作}.
\end{cases}
```

其中 $\epsilon \in [0,1]$, $|\mathcal{A}(s)|$ 是状态 $s$ 下可用的动作数.

- 当 $\epsilon \to 0$, 趋近贪心策略 (利用多, 探索少).
- 当 $\epsilon \to 1$, 趋近均匀分布 (探索多, 利用少).

使用软策略后, 少数足够长的回合就可以访问每个状态-动作对. 这样我们就不需要大量从每个状态-动作对开始的回合. 因此, 探索开始的要求可以被移除.

### 4.4.2 嵌入 ε-贪心策略
策略改进步骤变为在 $\epsilon$ -贪心策略集合 $\Pi_{\epsilon}$ 中求解:

```math
\pi_{k+1}(s) = \arg \max_{\pi \in \Pi_{\epsilon}} \sum_a \pi(a|s) q_{\pi_k}(s,a).
```

最优解即为 $\epsilon$ -贪心策略, 其贪心动作为 $a_k^* = \arg \max_a q_{\pi_k}(s,a)$.

### 4.4.3 具体实现

1. 初始化: 初始策略 $\pi_0$, 初始值 $q(s, a)$; $Returns(s, a) = 0$, $Num(s, a) = 0$; 探索参数 $\epsilon \in (0, 1]$.
2. 目标: 搜索最优策略.
3. 对于每一个回合, 执行:
   - 回合生成: 选择起始状态-动作对 $(s_0, a_0)$ (不需要探索开始条件), 遵循当前策略生成长度为 $T$ 的回合.
   - 初始化回报: $g \leftarrow 0$.
   - 对于 $t = T-1, T-2, \dots, 0$, 执行:
     - 计算累积回报: $g \leftarrow \gamma g + r_{t+1}$.
     - 更新回报总和: $Returns(s_t, a_t) \leftarrow Returns(s_t, a_t) + g$.
     - 更新访问次数: $Num(s_t, a_t) \leftarrow Num(s_t, a_t) + 1$.
     - 策略评估: $q(s_t, a_t) \leftarrow Returns(s_t, a_t) / Num(s_t, a_t)$.
     - 策略改进:
       - 寻找最大动作值: $a^* = \arg\max_a q(s_t, a)$.
       - 策略更新:
         - 如果 $a = a^*$, 则 $\pi(a|s_t) = 1 - \frac{|\mathcal{A}(s_t)| - 1}{|\mathcal{A}(s_t)|} \epsilon$.
         - 如果 $a \neq a^*$, 则 $\pi(a|s_t) = \frac{\epsilon}{|\mathcal{A}(s_t)|}$.