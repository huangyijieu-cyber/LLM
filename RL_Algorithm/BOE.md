# Bellman Optimality Equation (BOE)

## 1. 最优策略的定义

- 策略比较: 若对所有状态 $s \in \mathcal{S}$, 有 $v_{\pi_1}(s) \ge v_{\pi_2}(s)$,则称策略 $\pi_1$ **优于** 策略 $\pi_2$.
- **最优策略** $\pi^*$: 若对任意其他策略 $\pi$ 和所有状态 $s$, 均有 $v_{\pi^*}(s) \ge v_{\pi}(s)$, 则称 $\pi^*$ 为最优策略.

## 2. 贝尔曼最优方程 (BOE) 的形式

- **元素形式 (elementwise form)**:

```math
  v(s) = \max_{\pi} \sum_{a} \pi(a|s) \left( \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s') \right), \quad \forall s \in \mathcal{S}
```

  或写为

```math
  v(s) = \max_{\pi} \sum_{a} \pi(a|s) q(s,a), \quad \forall s \in \mathcal{S}
```

  其中 $q(s,a) = \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s')$ 为动作值.

- **矩阵-向量形式 (matrix-vector form)**:

```math
  v = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v)
```

  其中 $[r_{\pi}]_s \triangleq \sum_a \pi(a|s) \sum_r p(r|s,a) r$, $[P_{\pi}]_{s,s'} = p(s'|s) \triangleq \sum_a \pi(a|s) p(s'|s,a)$, 最大化是按元素进行的.

## 3. 求解 BOE

在给定当前动作值 $q(s,a)$ 时, 求解 $\max_{\pi} \sum_a \pi(a|s) q(s,a)$ **等价于将全部概率赋予最大的动作值对应的动作**.

对任意满足 $\sum_a c_a = 1, c_a \ge 0$ 的系数, 有 $\sum_a c_a q(s,a) \le \max_a q(s,a)$, 等号在 $c_{a^*}=1, a^*=\arg\max_a q(s,a)$ 时取得.

因此, 贝尔曼最优方程化简为:

```math
  v(s) = \max_{a \in \mathcal{A}(s)} q(s,a)
```

  且取得最优时的策略为 **确定性贪心策略**:

```math
  \pi(a|s) =
  \begin{cases}
  1, & a = a^*, \quad a^* = \arg\max_a q(s,a) \\
  0, & a \neq a^*
  \end{cases}
```

由压缩映射定理, 贝尔曼最优方程 **存在唯一的解** $v^*$. 可以通过 **[值迭代 (value iteration)](./Value_Iteration.md)** 算法求解:

```math
  v_{k+1} = f(v_k) = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_k)
```

  对任意初始 $v_0$, 序列 $\{v_k\}$ 指数级快速收敛到唯一解 $v^*$, 收敛速度由 $\gamma$ 决定.

## 4. BOE 的最优性

设 $v^*$ 是 BOE 的唯一解, $\pi^*$ 是达到右侧最大化的策略, 则有:

```math
  v^* = r_{\pi^*} + \gamma P_{\pi^*} v^*
```

  即 $v^* = v_{\pi^*}$, 是策略 $\pi^*$ 对应的状态值.
- **策略最优性定理:** 对于任何策略 $\pi$, 均有 $v^* \ge v_{\pi}$. 因此 $v^*$ 是 **最优状态值**, $\pi^*$ 是 **最优策略**.
- **贪心最优策略定理:** 对每个状态 $s$, 确定性贪心策略

```math
  \pi^*(a|s) =
  \begin{cases}
  1, & a = a^*(s) = \arg\max_{a} q^*(s,a) \\
  0, & \text{否则}
  \end{cases}
```

  其中 $q^*(s,a) = \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v^*(s')$, 总是最优的 (即能解决 BOE).