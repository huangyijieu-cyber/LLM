#贝尔曼公式 (Bellman Equation)

## 1. 状态值 (State Value)

对于一般的随机策略, 单步过程为:
$$
S_t \xrightarrow{A_t} R_{t+1}, S_{t+1}
$$
其中:
- $S_t$: $t$ 时刻状态
- $A_t$: 在 $S_t$ 采取的动作
- $R_{t+1}$: 采取动作后获得的奖励
- $S_{t+1}$: 转移到的下一状态
所有变量均为随机变量.

概率分布:
- 策略: $\pi(a|s) = \Pr(A_t = a | S_t = s)$
- 奖励分布: $p(r|s,a) = \Pr(R_{t+1} = r | S_t = s, A_t = a)$
- 状态转移分布: $p(s'|s,a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)$

**状态值函数 (state value function)** 定义为给定策略 $\pi$ 下, 从状态 $s$ 出发的期望回报:
$$
v_{\pi}(s) = \mathbb{E}[G_t | S_t = s]
$$

## 2. 贝尔曼方程定义

经过推导可得 **贝尔曼方程 (Bellman equation)** 的元素形式:

$$
v_{\pi}(s) = \sum_{a} \pi(a|s) \left[ \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_{\pi}(s') \right].
$$

## 3. 贝尔曼方程的矩阵-向量形式

定义策略 $\pi$ 下的期望即时奖励和状态转移概率:
$$
r_{\pi}(s) \triangleq \sum_{a} \pi(a|s) \sum_{r} p(r|s,a) r, \qquad
p_{\pi}(s'|s) \triangleq \sum_{a} \pi(a|s) p(s'|s,a).
$$

则贝尔曼方程可简写为:
$$
v_{\pi}(s) = r_{\pi}(s) + \gamma \sum_{s'} p_{\pi}(s'|s) v_{\pi}(s').
$$

将所有状态的值写成向量 $\mathbf{v}_{\pi}$, 得到 **矩阵-向量形式**:
$$
\mathbf{v}_{\pi} = \mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v}_{\pi},
$$
其中 $\mathbf{P}_{\pi}$ 为状态转移概率矩阵, $[\mathbf{P}_{\pi}]_{s,s'} = p_{\pi}(s'|s)$.

## 4. 求解状态值

求解贝尔曼方程即 **策略评估 (policy evaluation)**, 是寻找更好策略的基础.

- **闭式解 (Closed-form solution)**:
  $$
  \mathbf{v}_{\pi} = (I - \gamma \mathbf{P}_{\pi})^{-1} \mathbf{r}_{\pi}.
  $$
  矩阵 $I - \gamma \mathbf{P}_{\pi}$ 可逆.

- **迭代解法 (Iterative solution)**:
  $$
  \mathbf{v}_{k+1} = \mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v}_{k}.
  $$
  初始时随机选择一个值作为 $\mathbf{v}_{k}$ 进行迭代, 可以证明当 $k \to \infty$ 时, $\mathbf{v}_k$ 最终会收敛到 $\mathbf{v}_{\pi}$ (因为 $\gamma < 1$, 误差按 $\gamma^k$ 衰减).

  实际计算中往往使用迭代法. 通过求解策略的状态值, 可以**评估这个策略的好坏**.

## 5. 动作值 (Action Value)

**动作值函数 (action value function)** 定义为在状态 $s$ 采取动作 $a$ 后的期望回报:
$$
q_{\pi}(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a].
$$

**状态值与动作值的关系:**
$$
v_{\pi}(s) = \sum_{a} \pi(a|s) q_{\pi}(s,a). \tag{1}
$$
$$
q_{\pi}(s,a) = \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_{\pi}(s'). \tag{2}
$$