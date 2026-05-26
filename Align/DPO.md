# DPO

## 1. 核心概念

DPO (Direct Preference Optimization) 是一种用于训练语言模型以匹配人类偏好的优化方法. 与 PPO 不同, DPO **直接使用偏好比较数据进行优化**, 而不依赖于价值函数或策略截断. 其核心思想是: 给定人类标注的"好回答"和"差回答", 模型学习调整概率, 使"好回答"被模型生成的概率高于"差回答". 它**将复杂的强化学习问题, 等价转化为一个简单的二分类问题**. 不需要训练奖励模型, 也不需要在训练期间进行文本生成, 而是直接在人类偏好数据集上优化语言模型.

在 DPO 训练过程中, 我们只需维护两个模型:

1. **Policy Model (策略模型** $\pi_\theta$**)**: 我们要训练的 LLM. 初始为 SFT 训练完的模型. 它负责在给定 Prompt 时评估输出序列的概率. **它是可训练的.**
2. **Reference Model (参考模型** $\pi_{ref}$**)**: SFT 阶段后的模型副本, **参数冻结**. 作为参考模型, 用于提供基准概率, 防止策略模型在优化过程中过度偏离原有的语言能力.

DPO 不需要单独的 Reward 模型和 Critic 模型, 这是因为在数学原理上, 策略模型本身兼任了奖励模型的角色.

## 2. 基本公式

### 2.1 隐式奖励值 (从奖励到策略的等价转换)

在 PPO 中, 我们需要一个显式的奖励模型 $R_\phi(x,y)$. 而 DPO 的作者通过数学推导发现, 在 KL 散度约束下, RLHF 的最优策略有一个闭式解(Closed-form solution). 通过将这个闭式解反转, 我们可以用**策略模型的概率**直接表达出**隐式的奖励值**:

$$
r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

- **$r(x, y)$**: 隐式奖励值.
- **$\pi_\theta$ (策略模型)** 与 **$\pi_{ref}$ (参考模型)**: 模型对回答 $y$ 的生成概率.
- **$\beta$**: 控制偏离参考模型程度的超参数(类似于 PPO 中的 KL 惩罚系数).
- **$Z(x)$**: 配分函数(Partition function), 在后续的偏好计算中会被直接抵消掉, **因此无需计算**.

### 2.2 偏好概率模型 (Bradley-Terry 模型)

对于一个 Prompt $x$, 给定一对回答: 被人类偏好的回答 $y_w$ (win) 和被拒绝的回答 $y_l$ (lose). 人类偏好 $y_w$ 胜过 $y_l$ 的概率通常用 Bradley-Terry (BT) 模型表示为:

$$
P(y_w \succ y_l | x) = \sigma \big( r(x, y_w) - r(x, y_l) \big)
$$

- **$\sigma$**: Sigmoid 函数.

其物理意义是: 两者的奖励分差越大, $y_w$ 击败 $y_l$ 的概率越接近 100%.

### 2.3 DPO 目标函数

将 2.1 中的隐式奖励公式代入 2.2 的 BT 模型中, 由于 $Z(x)$ 是只与 prompt $x$ 有关的常数, 刚好在减法中被消去. 最终我们得到了极其简洁的 **DPO 目标函数(最小化该负对数似然损失)**:

$$
L_{DPO}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

- **$\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$**: 模型给偏好回答 $y_w$ 的隐式奖励.
- **$\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$**: 模型给拒绝回答 $y_l$ 的隐式奖励.

**目标函数的物理意义**: 增加 $y_w$ 的概率, 同时降低 $y_l$ 的概率. 当两者之间的隐式奖励差值越大, Loss 就越小. 由于引入了 $\pi_{ref}$ 作为分母, 模型在拉开概率差距的同时, 被隐式地限制了不能偏离初始 SFT 模型太远.

### 3. 训练流程

与 PPO 边生成边训练的 On-policy 方式不同, DPO 是一种 **Offline(离线)** 算法, 它的训练流程非常类似于标准的监督微调(SFT).

### 3.1 初始化 (Initialization)

在训练开始阶段, 系统加载两个核心模型并准备数据:

- **策略模型 $\pi_\theta$**: 从 SFT 模型加载权重, 作为训练的主体, 开启梯度记录.
- **参考模型 $\pi_\mathrm{ref}$**: 复制 SFT 权重并**完全冻结参数**.
- **准备偏好数据集 $D$**: 数据集中的每条数据均为三元组 $(x, y_w, y_l)$, 即 Prompt, 获胜回答, 失败回答.

### 3.2 数据采样与前向传播 (Forward Pass Phase)

1. 从偏好数据集中采样一个 Batch 的三元组 $(x, y_w, y_l)$. **(注意: 此时不需要模型生成任何新文本)**.
2. 将获胜回答 $y_w$ 和失败回答 $y_l$ 分别输入到 **策略模型 $\pi_\theta$** 中, 计算对应 Token 的 Log 概率, 记为 $\log \pi_\theta(y_w|x)$ 和 $\log \pi_\theta(y_l|x)$.
3. 将相同的 $y_w$ 和 $y_l$ 输入到 **参考模型 $\pi_\mathrm{ref}$** 中, 获取基准 Log 概率, 记为 $\log \pi_{ref}(y_w|x)$ 和 $\log \pi_{ref}(y_l|x)$.

### 3.3 隐式奖励计算 (Implicit Reward Computation Phase)

根据前向传播得到的概率, 计算两个回答的隐式奖励值:
1. 获胜回答的隐式奖励: $\hat{r}_w = \beta \cdot (\log \pi_\theta(y_w|x) - \log \pi_{ref}(y_w|x))$
2. 失败回答的隐式奖励: $\hat{r}_l = \beta \cdot (\log \pi_\theta(y_l|x) - \log \pi_{ref}(y_l|x))$

### 3.4 损失计算与策略更新 (Loss & Update Phase)

1. 计算两者隐式奖励的差值(Margin): $\Delta \hat{r} = \hat{r}_w - \hat{r}_l$
2. 将差值通过 Sigmoid 函数并取对数, 得到 DPO Loss:

$$
Loss = - \log \sigma(\Delta \hat{r})
$$

3. **反向传播**使用优化器(如 AdamW)更新策略模型 $\pi_\theta$ 的参数, 使得 $\Delta \hat{r}$ 尽可能大.

### 3.5 下一轮迭代

清空梯度, 从数据集中抽取下一个 Batch 的 $(x, y_w, y_l)$, 重复上述过程, 直到跑完所有 Epoch.

## 4. 局限性

1. **分布外限制 (OOD, Out-of-Distribution)**: DPO 完全依赖于预先收集的静态离线数据集. 如果在推理时遇到了完全偏离训练集分布的 Prompt, 模型表现不如能够在训练时自己生成数据并试错的 PPO.
2. **缺乏探索能力 (No Exploration)**: 因为 DPO 在训练时不生成任何新文本, 它只能学会"在 $y_w$ 和 $y_l$ 中选更好的", 而无法像 PPO 那样通过探索发现超越数据集中 $y_w$ 的"神级回答".
3. **对数据质量极度敏感**: 如果偏好数据集中的 $y_w$ 和 $y_l$ 标注存在大量噪音(标反了), 或者 $y_w$ 仅仅是字数比 $y_l$ 长(Length Bias), DPO 会非常容易过拟合并放大这些噪音/偏见.