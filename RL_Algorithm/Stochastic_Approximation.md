# 随机近似理论 (Stochastic Approximation)


## 1. 均值估计的增量算法

**均值估计的增量算法**: 对于随机变量 $X$ 的样本 $\{x_i\}$, 我们想估计 $\mathbb{E}[X]$. 传统的批量平均需要等待所有样本, 而增量更新避免了这一缺点. 推导如下:

设 $w_k = \frac{1}{k-1}\sum_{i=1}^{k-1} x_i$, 则
$$
w_{k+1} = \frac{1}{k}\sum_{i=1}^{k} x_i = w_k - \frac{1}{k}(w_k - x_k)
$$
更一般地, 用系数 $\alpha_k > 0$ 替换 $1/k$, 得到算法:
$$
w_{k+1} = w_k - \alpha_k (w_k - x_k)
$$
该算法在 $\alpha_k$ 满足温和条件时收敛到 $\mathbb{E}[X]$. 它是一个特殊的 SA 算法和特殊的随机梯度下降 (SGD) 算法.

## 2. 随机逼近算法 (Robbins-Monro, RM) 

**RM 算法** 是一类求解求根或优化问题的随机迭代算法. 它**不需要知道目标函数或其导数的表达式**, 仅依靠带噪声的观测数据.

**问题形式**: 求方程 $g(w) = 0$ 的根, 其中 $w \in \mathbb{R}$, 函数 $g$ 未知 (黑箱), 但可以获得带有噪声的观测:
  $$
  \tilde{g}(w_k, \eta_k) = g(w_k) + \eta_k
  $$
  其中 $\eta_k$ 是观测误差. RM 算法的更新式为:
  $$
  w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)
  $$

>**收敛定理 (Robbins-Monro 定理)**: 若满足以下三个条件, 则 $w_k$ 以概率1收敛到真实根 $w^*$:
>  1. $0 < c_1 \leq \nabla_w g(w) \leq c_2$ 对所有 $w$ 成立 (函数单调递增, 保证根存在且唯一; 用于优化时对应函数凸性).
>  2. $\sum_{k=1}^{\infty} a_k = \infty$ 且 $\sum_{k=1}^{\infty} a_k^2 < \infty$ (步长序列趋于零但不能太快; 典型序列是 $a_k = 1/k$).
>  3. $\mathbb{E}[\eta_k | \mathcal{H}_k] = 0$ 且 $\mathbb{E}[\eta_k^2 | \mathcal{H}_k] < \infty$ (观测噪声零均值, 方差有限, 不要求高斯分布).

**RM 算法应用于均值估计**:

设函数 $g(w) = w - \mathbb{E}[X]$, 需要求解 $g(w)=0$.

观测值为 $\tilde{g}(w, x) = w - x$, 可分解为 $g(w) + (\mathbb{E}[X] - x) = g(w) + \eta$.

则 RM 更新变为:
$$
w_{k+1} = w_k - \alpha_k (w_k - x_k)
$$
这正是前面的均值估计算法, 其收敛性由 RM 定理保证.

## 3. 随机梯度下降 (Stochastic Gradient Descent, SGD)

### 3.1 梯度下降算法

**优化问题**: 最小化函数 $\min_w J(w) = \mathbb{E}[f(w, X)]$, 其中 $X$ 是随机变量.

- **梯度下降 (GD)**: $w_{k+1} = w_k - \alpha_k \mathbb{E}[\nabla_w f(w_k, X)]$. 缺点是需知分布.
- **批量梯度下降 (BGD)**: $w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i)$. 缺点是需要大量样本.
- **随机梯度下降 (SGD)**: $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$. 每次仅用一个样本 $x_k$.

### 3.2 SGD 算法

**SGD 是特殊的 RM 算法**, 以下是证明过程:
令 $g(w) = \nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)]$, 求根 $g(w)=0$.
观测梯度 $\nabla_w f(w_k, x_k)$ 可写为:
$$
\nabla_w f(w_k, x_k) = \mathbb{E}[\nabla_w f(w, X)] + \eta_k = g(w) + \eta_k
$$ 
RM 算法代入即得 SGD: $w_{k+1} = w_k - a_k \nabla_w f(w_k, x_k)$. 收敛性由 RM 定理保证.

### 3.3 SGD 的收敛条件与模式

**SGD 收敛条件** (对应于 RM 三条件):
1. $0 < c_1 \leq \nabla_w^2 f(w, X) \leq c_2$ (严格凸性).
2. $\sum a_k = \infty$, $\sum a_k^2 < \infty$.
3. 样本 $\{x_k\}$ 是独立同分布的.

**SGD 的收敛模式 (相对误差分析)**:

定义相对误差 $\delta_k = \frac{|\nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)]|}{|\mathbb{E}[\nabla_w f(w_k, X)]|}$.
利用均值定理和严格凸性可得:
$$
\delta_k \leq \frac{|\nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)]|}{c |w_k - w^*|}
$$
因此:
- 当距离最优解 $|w_k - w^*|$ 较大时, 相对误差上界小, **SGD 表现类似于 GD**.
- 当接近 $w^*$ 时, 相对误差可能较大, **SGD 在最优解附近表现出更多的随机性**.

### 3.4 BGD, MBGD 与 SGD 的比较

基于给定样本集 $\{x_i\}_{i=1}^n$ 最小化 $J(w) = \mathbb{E}[f(w, X)]$, 三种算法的更新为:

- **BGD**: $w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i)$, 每次用全部 $n$ 个样本.
- **MBGD**: $w_{k+1} = w_k - \alpha_k \frac{1}{m} \sum_{j \in \mathcal{I}_k} \nabla_w f(w_k, x_j)$, $\mathcal{I}_k$ 是从 $\{1,\dots,n\}$ 中随机采样的 $m$ 个索引的集合.
- **SGD**: $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$, $x_k$ 在时刻 $k$ 随机抽取.

特点:
- MBGD 比 SGD 随机性更小, 因为使用了更多样本.
- MBGD 比 BGD 更灵活高效, 无需每次使用全部样本.
- $m=1$ 时 MBGD 退化为 SGD; $m=n$ 时, MBGD 是随机有放回采样 $n$ 次, 而 BGD 使用了所有 $n$ 个样本一次, 二者有细微差别.

**均值估计实例**:
问题 $\min_w J(w) = \frac{1}{2n} \sum_{i=1}^n \| w - x_i \|^2$ 对应三个算法的更新:
- BGD: $w_{k+1} = w_k - \alpha_k (w_k - \bar{x})$
- MBGD: $w_{k+1} = w_k - \alpha_k (w_k - \bar{x}_k^{(m)})$, $\bar{x}_k^{(m)}$ 为小批量均值
- SGD: $w_{k+1} = w_k - \alpha_k (w_k - x_k)$