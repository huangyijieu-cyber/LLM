# GRPO

## 1. 核心概念

GRPO 是一种专为大规模语言模型设计的策略梯度算法, 用于解决 PPO 中"Critic 模型占用显存过大"以及"优势计算复杂"的问题. 它的核心思想是: **彻底抛弃 Critic(价值模型), 通过让模型对同一个 Prompt 生成多个不同的回答(组成一个 Group), 在组内进行相对打分来计算优势.**

在 GRPO 训练过程中, 我们只需要在显存中维护三个(甚至两个)模型:

1. **Policy Model (策略模型 $\pi_\theta$)**: 我们要训练的 LLM. 负责根据 Prompt 生成 Token. **它是可训练的.**
2. **Reference Model (参考模型 $\pi_{ref}$)**: SFT 阶段后的模型副本, **参数冻结**. 用于计算 KL 散度, 防止策略模型在优化时"过度拟合奖励模型而说怪话".
3. **Reward Model / Rule Verifier (奖励模型 $R_\phi$ 或 规则校验器)**: 用来为生成的回答评分, **参数冻结**.(注: 在 DeepSeek-R1 的强化学习中, 直接使用了基于数学/代码规则的校验器来给出 0 或 1 的硬性奖励, 此时连奖励模型的显存都省了).

## 2. 基本公式

在 PPO 中, 我们依赖 Critic 来预测 Baseline. 但在 GRPO 中, Baseline 是通过 **组内比较** 得出的.
对于每一个输入提示(Prompt) $x$, 我们让当前的策略模型 $\pi_\theta$ 独立生成 $G$ 个不同的回答:

```math
\{y_1, y_2, \dots, y_G\} \sim \pi_\theta(y|x)
```

然后, 使用奖励模型(或规则)对这 $G$ 个回答分别打分, 得到一组原始奖励:

```math
\{r_1, r_2, \dots, r_G\}
```

### 2.2 组相对优势计算 (Group Relative Advantage)

得到了这 $G$ 个回答的得分后, GRPO 不使用复杂的 GAE 计算, 而是直接对这组得分进行 **标准化处理(Z-Score)**, 得到每个回答的优势 $\hat{A}_i$:

```math
\hat{A}_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}
```

- **$\text{mean}(r_{1..G})$**: 这 $G$ 个回答得分的平均值(这就是 GRPO 的 Baseline).
- **$\text{std}(r_{1..G})$**: 这 $G$ 个得分的标准差.
- **物理意义**: 如果 $y_i$ 的得分高于这批回答的平均水平, 它的优势 $\hat{A}_i$ 就是正的(鼓励这种生成方式); 如果低于平均水平, 优势就是负的(抑制这种生成方式).

### 2.3 KL 散度惩罚

与 PPO 将 KL 散度作为每一步的"扣分项"叠加在奖励上不同, GRPO 直接将 KL 散度作为正则化项加在了最后的 **损失函数** 中. 为了保证梯度的无偏估计, GRPO 使用了如下精确的 KL 散度估计公式:

```math
D_{KL}(\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(a_t|s_t)}{\pi_\theta(a_t|s_t)} - \log \frac{\pi_{ref}(a_t|s_t)}{\pi_\theta(a_t|s_t)} - 1
```

- 这个估计算法相比直接计算 $\log(\pi_\theta / \pi_{ref})$ 更加稳定, 能防止 KL 散度在训练过程中变成负数.

### 2.4 GRPO 目标函数

合并上面的优势和 KL 惩罚, 我们得到了 GRPO 最终的截断目标函数(最大化该函数):

```math
L_{GRPO}(\theta) = \mathbb{E} \left[\min\Big(\rho_{i,t} \hat{A}_i, \; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i\Big) - \beta D_{KL} \right]
```

- **$\rho_{i,t}$**: 重要性采样比率 $\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}$ (同 PPO).
- **$\hat{A}_i$**: 刚刚算出的组内相对优势. 注意, 同一个句子 $y_i$ 中的所有 token 共享这个句子级别的优势.
- **$\beta D_{KL}$**: 对偏离参考模型的惩罚项.

## 3. 训练流程

### 3.1 初始化 (Initialization)

在训练开始阶段, 系统加载三个核心模型(或两个模型+一套评判规则):

- **策略模型(Actor) $\pi_\theta$**: 加载初始权重, 作为训练的主体.
- **参考模型(Reference) $\pi_\mathrm{ref}$**: 复制初始权重并 **冻结参数**, 用于约束策略更新.
- **奖励模型(Reward) $R_\phi$ / 规则校验器**: **冻结**, 用于为回答打分. **无需初始化 Critic 模型.**

### 3.2 组内样本生成 (Group Rollout Phase)

1. 从数据集中采样一个 Prompt $x$ (或者一个 Batch 的 Prompts).
2. **策略模型 $\pi_\theta$** 针对该 Prompt $x$, 并行生成 $G$ 个不同的回答序列 $\{y_1, y_2, \dots, y_G\}$ (例如 $G=4$ 或 $8$).
3. 记录生成这 $G$ 个序列时的旧策略概率 $\pi_{\theta_{old}}$.

### 3.3 奖励打分与优势计算 (Reward & Advantage Phase)

1. **奖励模型 / 规则系统** 对这 $G$ 个回答进行评估, 输出标量奖励 $\{r_1, r_2, \dots, r_G\}$.
2. 计算这组奖励的均值和标准差.
3. 利用标准化公式 $\hat{A}_i = \frac{r_i - \text{mean}}{\text{std}}$ 为每个回答赋上 **优势值**.

### 3.4 策略更新 (GRPO Update Phase)

将当前经验池中收集到的整批数据打乱, 切分成多个 Mini-batch(微批次). 对这批数据进行若干次内部循环(PPO Epochs, 如循环 4 遍). 对每个 Mini-batch 执行以下更新:

1. 将这 $G$ 个回答重新输入到 **策略模型 $\pi_\theta$** 和 **参考模型 $\pi_\mathrm{ref}$** 中, 计算最新的生成概率和基准概率.
2. 对于每个回答的每个 Token, 计算重要性采样比率 $\rho_{i,t}$.
3. 计算准确的 KL 散度惩罚项 $D_{KL}$.
4. 构建并计算 GRPO 的目标函数 $L_{GRPO}(\theta)$.
5. **反向传播** 使用优化器更新策略模型 $\pi_\theta$ 的参数.

### 3.5 数据清理与下一轮迭代

1. 丢弃当前 Prompt 生成的这组数据.
2. 抽取下一个 Prompt 继续生成 $G$ 个回答, 循环进行, 直至收敛.

## 4. 局限性

1. **极高的生成开销**: 因为针对每一个 Prompt 必须生成 $G$ 个回答(通常 $G=4 \sim 16$), 在 Rollout 阶段会消耗极其庞大的算力和时间.
2. **缺乏细粒度的信度分配 (Credit Assignment)**: GRPO 算出的优势 $\hat{A}_i$ 是一个 **句子级别** 的全局优势, 句子里的每一个 Token 都共享同一个优势值. 它无法像 PPO 的 Critic 那样, 精确评估"到底是哪一步生成的词导致了得分变高/变低".
3. **依赖 Group Size 的大小**: 如果 $G$ 设置得太小(比如 $G=2$), 均值和标准差的估算会非常不稳定, 导致训练崩溃; 如果 $G$ 设置过大, 又会面临 OOM(显存溢出)和生成时间过长的问题.