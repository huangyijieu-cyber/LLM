## 1. 值迭代算法 (Value_Iteration)

### 1.1 矩阵-向量形式与分解
值迭代的更新公式为

$$
v_{k+1} = f(v_k) = \max_{\pi}(r_{\pi} + \gamma P_{\pi} v_k), \quad k=1,2,3,\dots
$$

该过程可分解为两步:

- **策略更新 (Policy Update):** 求解

$$
  \pi_{k+1} = \arg \max_{\pi}(r_{\pi} + \gamma P_{\pi} v_k)
$$

- **值更新 (Value Update):**

$$
  v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k
$$

### 1.2 元素形式
对于每个状态 $s$ , 策略更新的元素形式为

$$
\pi_{k+1}(s) = \arg \max_{\pi} \sum_a \pi(a|s) \left( \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right)
$$

定义动作值 $q_k(s,a)$ 为

$$
q_k(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_k(s')
$$

最优动作 $a_k^*(s) = \arg \max_a q_k(s,a)$ , 由此得到的策略 $\pi_{k+1}$ 称为 **贪心策略 (greedy policy)** , 因为它直接选择最大的 q 值.

值更新等价于

$$
v_{k+1}(s) = \max_a q_k(s,a)
$$

### 1.3. 具体实现

1. 初始化: 已知概率模型 $p(r|s, a)$ 和 $p(s'|s, a)$; 初始猜测值 $v_0$.
2. 当 $v_k$ 未收敛 (即 $||v_k - v_{k-1}||$ 大于预设的极小阈值) 时, 对于第 $k$ 次迭代, 执行:
   - 对于每一个状态 $s \in \mathcal{S}$, 执行:
     - 对于每一个动作 $a \in \mathcal{A}(s)$, 执行:
       - 计算 q-value: $q_k(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_k(s')$.
     - 寻找最大动作值: $a_k^*(s) = \arg\max_a q_k(a, s)$.
     - 策略更新: 如果 $a = a_k^*$, 则 $\pi_{k+1}(a|s) = 1$, 否则 $\pi_{k+1}(a|s) = 0$.
     - 价值更新: $v_{k+1}(s) = \max_a q_k(a, s)$.

## 2. 策略迭代 (Policy_Iteration)

### 2.1 算法框架
从一个初始策略 $\pi_0$ 开始, 重复以下两步:
- **策略评估 (Policy Evaluation, PE):** 计算当前策略 $\pi_k$ 的状态值 $v_{\pi_k}$ , 即求解贝尔曼方程

$$
  v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}
$$

  注意这里的 $v_{\pi_k}$ 是真实的状态值函数.
- **策略改进 (Policy Improvement, PI):**

$$
  \pi_{k+1} = \arg \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_{\pi_k})
$$

  该最大化是按分量进行的.

迭代序列:

$$
\pi_0 \xrightarrow{PE} v_{\pi_0} \xrightarrow{PI} \pi_1 \xrightarrow{PE} v_{\pi_1} \xrightarrow{PI} \pi_2 \xrightarrow{PE} \dots
$$

#### Q1: 如何计算 $v_{\pi_k}$ ?
  闭式解: $v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}$
  迭代解:

$$
  v_{\pi_k}^{(j+1)} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}^{(j)}, \quad j = 0,1,2,\dots
$$

  策略迭代本身是一个外层迭代, 其中内嵌了另一个迭代过程用于策略评估.

### 2.3 具体实现

1. 初始化: 已知概率模型 $p(r|s, a)$ 和 $p(s'|s, a)$; 初始猜测策略 $\pi_0$.
2. 当 $v_{\pi_k}$ 未收敛时, 对于第 $k$ 次迭代, 执行:
   - 策略评估 —— 无限次内部迭代:
     - 初始化: 给定任意初始猜测值 $v_{\pi_k}^{(0)}$.
     - 当 $v_{\pi_k}^{(j)}$ 未收敛时 (即内部迭代 $j \to \infty$ 直至完全收敛), 对于第 $j$ 次迭代, 执行:
       - 对于每一个状态 $s \in \mathcal{S}$, 执行:
         - $v_{\pi_k}^{(j+1)}(s) = \sum_a \pi_k(a|s) \left[ \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}^{(j)}(s') \right]$.
   - 策略改进:
     - 对于每一个状态 $s \in \mathcal{S}$, 执行:
       - 对于每一个动作 $a \in \mathcal{A}(s)$, 执行:
         - 基于收敛的价值计算: $q_{\pi_k}(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_{\pi_k}(s')$.
       - 寻找最大动作值: $a_k^*(s) = \arg\max_a q_{\pi_k}(s, a)$.
       - 策略更新: 如果 $a = a_k^*$, 则 $\pi_{k+1}(a|s) = 1$, 否则 $\pi_{k+1}(a|s) = 0$.

## 3. 截断策略迭代 (Truncated Policy Iteration)

### 3.1 从值迭代与策略迭代的比较到截断
- 策略迭代在 PE 步中需要求解 $v_{\pi_k} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}$ , 这等价于无穷多次迭代.
- 值迭代只做一步: $v_{k+1} = r_{\pi_{k+1}} + \gamma P_{\pi_{k+1}} v_k$ .

具体地, 考虑从 $v_0$ 开始计算 $v_{\pi_1}$ 的过程:

$$
v_{\pi_1}^{(0)} = v_0 \;\rightarrow\; v_{\pi_1}^{(1)} = r_{\pi_1} + \gamma P_{\pi_1} v_0 \;\rightarrow\; v_{\pi_1}^{(2)} \dots \;\rightarrow\; v_{\pi_1}^{(\infty)}
$$

**值迭代只给出 $v_1 = v_{\pi_1}^{(1)}$ , 而策略迭代需要 $v_{\pi_1}^{(\infty)}$ .** 截断策略迭代则只计算有限步 $j_{\text{truncate}}$ , 即用 $v_{\pi_1}^{(j)}$ 作为 $v_{\pi_1}$ 的近似.

### 3.2 值改进性质

若将策略评估的初始值设为 $v_{\pi_k}^{(0)} = v_{\pi_{k-1}}$ , 则对每个 $j$ 都有

$$
v_{\pi_k}^{(j+1)} \geq v_{\pi_k}^{(j)}
$$

因此截断不会破坏收敛性, 仍能保证单调改进.

### 3.3 具体实现

1. 初始化: 已知概率模型 $p(r|s, a)$ 和 $p(s'|s, a)$; 初始猜测策略 $\pi_0$.
2. 当 $v_k$ 未收敛时, 对于第 $k$ 次迭代, 执行:
   - 策略评估 —— 有限次内部迭代:
     - 初始化: 将上次的结果作为初始猜测值 $v_k^{(0)} = v_{k-1}$. 设定最大内部迭代次数为 $j_{truncate}$.
     - 当 $j < j_{truncate}$ 时, 执行:
       - 对于每一个状态 $s \in \mathcal{S}$, 执行:
         - $v_k^{(j+1)}(s) = \sum_a \pi_k(a|s) \left[ \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_k^{(j)}(s') \right]$.
     - 直接截断并赋值: 设定 $v_k = v_k^{(j_{truncate})}$.
   - 策略改进:
     - 对于每一个状态 $s \in \mathcal{S}$, 执行:
       - 对于每一个动作 $a \in \mathcal{A}(s)$, 执行:
         - 基于截断的近似价值计算: $q_k(s, a) = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_k(s')$.
       - 寻找最大动作值: $a_k^*(s) = \arg\max_a q_k(s, a)$.
       - 策略更新: 如果 $a = a_k^*$, 则 $\pi_{k+1}(a|s) = 1$, 否则 $\pi_{k+1}(a|s) = 0$.