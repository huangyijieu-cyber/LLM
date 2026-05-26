# PPO

## 1. 核心概念

PPO 是一种策略梯度(Policy Gradient)方法, 用于解决强化学习中"如何稳定地更新策略"的问题. 它的目标是在提升策略性能的同时, 不让策略变化过大. 它的核心思想是: 在优化策略时, 对每个动作的概率变化设置一个"安全区间".

在 RLHF 中, PPO 是用来 优化语言模型, 使其输出更符合人类偏好 的算法. 用于最大化奖励模型分数, 保持语言流畅性和多样性.

在 PPO 训练中, 我们需要在显存中同时维护四个模型:

1. **Actor (策略模型 $\pi_\theta$)**: 我们要训练的 LLM. 初始为 SFT 训练完的模型. 它负责根据 Prompt 生成 Token. **它是可训练的.**
2. **Critic (价值模型 $V_\omega$)**: 与 Actor 模型同步更新的模型, 用于预测按照当前模型回答, 未来可能得到的总回报. **它是可训练的.**
3. **Reference Model (参考模型 $\pi_{ref}$)**: SFT 阶段后的模型副本, **参数冻结**. 作为参考模型, 用于计算 KL 散度, 防止 Actor 为了刷高分而学会说怪话.
4. **Reward Model (奖励模型 $R_\phi$)**: 用来为 Actor 生成的回答评分的模型, 基于人类偏好训练而来, **参数冻结**.

## 2. 基本公式

## 2.1 即时奖励

我们必须先算出生成每一个 Token $a_t$ 到底拿了多少分. 在 RLHF 中, 奖励函数的设计如下:

$$
r_t = \begin{cases}
- \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}, & \text{当 } t < T \text{ (句子未结束)} \\
R_\phi(x, y) - \beta \log \frac{\pi_T(a_T|s_T)}{\pi_{ref}(a_T|s_T)}, & \text{当 } t = T \text{ (生成最后一个词)}
\end{cases}
$$

- **$\pi_\theta$ (Actor 模型)**: 当前正在训练的模型的生成概率.
-  **$\pi_{ref}$ (Reference 模型)**: 冻结的初始模型的生成概率.
-  **$\log \frac{\pi_\theta}{\pi_{ref}}$**: 这就是 **KL 散度惩罚项**. 如果 Actor 生成的词偏离 Reference 太远, 这个值会变大. 乘以系数 $-\beta$ 后, 就会变成负分(扣分).
-  **$R_\phi$ (Reward 模型)**: 只有在生成完最后一个词($t=T$)时, 也就是得到完整句子 $y$ (根据 Prompt $x$ 生成)后, Reward 模型才给出最终得分.

### 2.2 优势函数

得到每一步的奖励 $r_t$ 后, 我们利用 **Critic(评论家模型)** 来计算优势. PPO 通常使用 **GAE (广义优势估计, Generalized Advantage Estimation)** 来降低方差.

**1. 首先计算 TD Error $\delta_t$:**

$$
\delta_t = r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)
$$

*   **$V_\omega$ (Critic 模型)**: 参数为 $\omega$ 的价值模型. 它预测在状态 $s_t$ 下, 未来能拿到的总奖励.
*   $r_t$: 刚才算出的这一步的实际奖励.
*   $\gamma$: 折扣因子.
*   **$\delta_t$ 的物理意义**: 这一步的实际收益($r_t +$ 下一步的预测), 减去这一步原本的预测($V_\omega(s_t)$). 意思是 $s_t$ 这步走了以后, 总的回报比没走这步上升还是下降了.

**2. 计算 GAE 优势函数 $\hat{A}_t$:**

$$
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}
$$

*   $\lambda$: GAE 的平滑参数(通常设为 0.95).
*   这个公式的意义在于为了让计算更稳, 我们不仅看当前这一步的 $\delta_t$, 还把未来几步的 $\delta$ 也按比例($\gamma \lambda$)加进来, 得到一个更宏观的 **"动作优势 $\hat{A}_t$ "**.

### 2.3 Actor 目标函数

现在我们要去更新 **Actor 模型** 的参数 $\theta$ 了. PPO 的核心是 **截断(Clipping)**, 防止模型更新过猛.

**1. 定义概率比值比 $ratio$:**

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

*   $\pi_\theta$ 是当前参数下生成该词的概率.
*   $\pi_{\theta_{old}}$ 是本次参数更新 **前** 生成该词的概率.

**2. Actor 的截断目标函数(最大化该函数):**

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\Big(\rho_t(\theta) \hat{A}_t, \; \text{clip}\big(\rho_t(\theta), 1-\epsilon, 1+\epsilon\big) \hat{A}_t\Big) \right]
$$

*   **$\hat{A}_t$**: 就是我们在第二步用 **Critic** 算出来的优势.
*   **$\epsilon$**: 截断超参数(通常为 0.2).
*   **$\text{clip}(\cdot, 0.8, 1.2)$**: 强行把概率比值 $\rho_t$ 限制在 $[0.8, 1.2]$ 的区间内.
*   **$\min(\dots)$**: 取(未截断的项)和(截断的项)中的最小值. 这是一种悲观下界策略, 确保如果优势是正的, 概率增加最多不超过 20%; 如果优势是负的, 概率减少最多也不超过 20%.

### 2.4 Critic 目标函数

**Critic 模型** 的参数 $\omega$ 也在不断更新, 它的目标是让自己预测的 $V_\omega(s_t)$ 越来越接近实际获得的回报. 这是一个标准的回归问题(均方误差 MSE).

**Critic 的目标函数(最小化该函数):**

$$
L^{VF}(\omega) = \hat{\mathbb{E}}_t \left[ \Big( V_\omega(s_t) - V_t^{target} \Big)^2 \right]
$$

*   **$V_\omega(s_t)$ (Critic 模型)**: Critic 当前给出的预测值.
*   **$V_t^{target}$**: 实际目标值. 通常通过 $V_t^{target} = \hat{A}_t + V_{\omega_{old}}(s_t)$ 来计算(即实际获得的总回报近似值).

## 3. 训练流程

### 3.1 初始化(Initialization)

在训练开始阶段, 系统加载并初始化四个核心模型:

- **策略模型(Actor) $\pi_\theta$**: 从 SFT 模型加载权重, 作为训练的主体.

- **参考模型(Reference) $\pi_\mathrm{ref}$**: 复制 SFT 权重并 **冻结参数(不参与梯度更新)**, 用于约束策略更新的 KL 散度.

- **奖励模型(Reward) $R_\phi$**: 已训练好的评分模型, **冻结参数**, 用于为生成文本提供即时奖励.

- **价值模型(Critic) $V_\omega$**: 通常以奖励模型初始化, 输出层修改为预测分数的形式, 参与训练以估计状态值函数.

### 3.2 样本生成(Rollout Phase)

1. 从数据集中采样一批 Prompt(例如 $N$ 个问题).

2. **策略模型 $\pi_\theta$** 基于当前参数生成 **回答序列**, 同时记录每一步的状态 $s_t$ 与动作 $a_t$.

3. 每个生成动作对应的概率 $\pi_{\theta_\mathrm{old}}(a_t|s_t)$ 被存储, 用于后续重要性采样.

### 3.3 奖励计算(Reward Computation Phase)

1. **奖励模型 $R_\phi$** 对生成序列进行评分, 得到总分 $R(y)$.

2. **参考模型 $\pi_\mathrm{ref}$** 对每个生成动作提供概率分布, 用于计算 KL 散度惩罚:

$$
    r_t = R(y) - \beta \cdot \log \frac{\pi_\theta(a_t|s_t)}{\pi_\mathrm{ref}(a_t|s_t)}
$$

3. 结合奖励和 KL 惩罚, 得到每一步的即时奖励 $r_t$, 形成完整的回报序列.

### 3.4 优势估计(Advantage Estimation Phase)

1. **价值模型 $V_\omega$** 对每个状态 $s_t$ 预测期望回报 $\hat{V}_t$.

2. 计算 TD error:

$$
    \delta_t = r_t + \gamma V_\omega(s_{t+1}) - V_\omega(s_t)
$$

3. 使用 GAE 整合 $\delta_t$, 得到每一步的优势函数:

$$
    \hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
$$

4. 计算 **Critic 的训练目标**:

$$
    V_\mathrm{target} = \hat{A}_t + V_\omega(s_t)
$$

### 3.5 策略与价值更新(PPO Update Phase)

将当前经验池中收集到的整批数据打乱，切分成多个 Mini-batch（微批次）。对这批数据进行若干次内部循环（PPO Epochs，如循环 4 遍）。对每个 Mini-batch 执行以下更新：

1. **策略模型更新(Actor Update)**
    - 计算重要性采样比率:  $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\mathrm{old}}(a_t|s_t)}$
    - 构建 PPO 截断目标函数:

$$
        L^\mathrm{CLIP}(\theta) = \mathbb{E}_t\big[ \min(\rho_t \hat{A}_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) \big]
$$

    - 最大化 $L^\mathrm{CLIP}(\theta)$, 保证策略更新幅度受限, 防止过度偏离参考模型.

2. **价值模型更新(Critic Update)**

    - 通过均方误差(MSE Loss)优化:

$$
        L^\mathrm{V}(\omega) = \mathbb{E}_t \big[ (V_\omega(s_t) - V_\mathrm{target})^2 \big]
$$

- **反向传播** 使用优化器(如 AdamW)同时更新策略模型 $\theta$ 与价值模型 $\omega$.

### 3.6 数据清理与下一轮迭代

1. 当前回合的生成数据被丢弃, 经验池清空.
2. 策略和价值模型参数保留更新结果, 作为下一批 Prompt 的初始状态.
3. 循环进行, 直至策略收敛或达到预设训练轮次.

## 4. 局限性

1. 显存墙 (Memory Wall): 每次训练都需同时加载四个模型, 极大地增加了显存负担.
2. 生成瓶颈: PPO 是 on-policy 算法, 每训练一步必须生成新的文本, 因此极大地拖慢了训练节奏.
