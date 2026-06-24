# DAPO

## 1. 核心概念

DAPO 全称是 **Decoupled Clip and Dynamic sAmpling Policy Optimization**，是一种在 GRPO 基础上改进的大模型强化学习算法。它主要用于解决 GRPO 在长推理链 Long-CoT 训练中出现的几个问题：**熵坍缩、有效样本不足、长回答训练信号被稀释、超长输出带来奖励噪声**。

它的核心思想是：

**在保留 GRPO“同一个 Prompt 生成多个回答，并在组内进行相对打分”的基础上，通过非对称裁剪、动态采样、Token 级别损失和超长惩罚，让大模型强化学习训练更加稳定。**

在 DAPO 训练过程中，通常只需要维护两个主要模型和一个奖励系统：

1. **Policy Model（策略模型）** $\pi_\theta$：我们要训练的 LLM。负责根据 Prompt 生成 Token。**它是可训练的。**
2. **Old Policy Model（旧策略模型）** $\pi_{\theta_{old}}$：采样生成回答时的策略模型快照。用于计算重要性采样比率 $\rho_{i,t}$，**参数在当前更新阶段固定**。
3. **Reward Model / Rule Verifier（奖励模型或规则校验器）** $R_\phi$：用来为生成的回答评分，**参数冻结**。在数学、代码等任务中，通常使用规则校验器判断答案是否正确。

和 GRPO 一样，DAPO **不需要 Critic 模型**。但和常见 GRPO 不同的是，DAPO 论文版目标函数中 **移除了 KL 惩罚项**，不再强制策略模型一直贴近 Reference Model。DAPO 论文认为，在 Long-CoT 推理训练中，过强的 KL 约束可能限制模型探索和推理能力提升。

---

## 2. 基本公式

在 GRPO 中，我们通过同一个 Prompt 的多个回答计算组内相对优势。DAPO 仍然保留这一点。

对于每一个输入提示 Prompt $x$，我们让当前旧策略模型 $\pi_{\theta_{old}}$ 独立生成 $G$ 个不同的回答：

$$
{y_1, y_2, \dots, y_G} \sim \pi_{\theta_{old}}(y|x)
$$

然后，使用奖励模型或规则校验器对这 $G$ 个回答分别打分，得到一组原始奖励：

$$
{r_1, r_2, \dots, r_G}
$$

例如在数学任务中，答案正确时：

$$
r_i = 1
$$

答案错误时：

$$
r_i = 0
$$

### 2.2 组相对优势计算（Group Relative Advantage）

DAPO 的优势计算方式和 GRPO 基本一致，仍然使用组内标准化奖励：

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}
$$

* $\text{mean}(r_{1..G})$：这 $G$ 个回答得分的平均值，也就是组内 Baseline。
* $\text{std}(r_{1..G})$：这 $G$ 个得分的标准差。
* **物理意义**：如果 $y_i$ 的得分高于这批回答的平均水平，它的优势 $\hat{A}_i$ 就是正的，训练会鼓励这种回答；如果低于平均水平，优势就是负的，训练会抑制这种回答。

但是 DAPO 进一步注意到一个问题：

如果同一个 Prompt 生成的 $G$ 个回答全部正确：

$$
{1,1,1,1}
$$

或者全部错误：

$$
{0,0,0,0}
$$

那么组内没有相对差异，优势值几乎没有有效训练信号。因此 DAPO 引入了 **Dynamic Sampling（动态采样）** 来过滤这种无效 group。

---

### 2.3 DAPO 的关键改动

与 GRPO 相比，DAPO 主要引入了四个关键改动。

#### 1. Clip-Higher / Decoupled Clip

在 GRPO 或 PPO 中，重要性采样比率通常使用对称裁剪：

$$
\text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)
$$

其中：

$$
\rho_{i,t} = \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}
$$

DAPO 将上下裁剪范围拆开：

$$
\text{clip}(\rho_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})
$$

其中通常有：

$$
\epsilon_{\text{high}} > \epsilon_{\text{low}}
$$

* $\epsilon_{\text{low}}$：限制策略模型降低某些 token 概率的幅度。
* $\epsilon_{\text{high}}$：限制策略模型提高某些 token 概率的幅度。
* **物理意义**：DAPO 提高上裁剪范围，让模型可以更大胆地提高低概率但高奖励 token 的概率，从而增强探索能力，缓解熵坍缩。verl 文档中也将 `clip_ratio_low` 和 `clip_ratio_high` 对应到 DAPO 的上下裁剪范围。

---

#### 2. Dynamic Sampling

DAPO 会过滤掉没有有效梯度信号的 Prompt group。

对于每个 Prompt，如果生成 $G$ 个回答后的奖励满足：

$$
\sum_{i=1}^{G} r_i = 0
$$

说明全部回答错误；如果满足：

$$
\sum_{i=1}^{G} r_i = G
$$

说明全部回答正确。

这两种情况都会被过滤掉。DAPO 只保留：

$$
0 < \sum_{i=1}^{G} r_i < G
$$

也就是只保留“有的回答对，有的回答错”的 group。

* **全部正确**：模型已经会了，继续训练信号弱。
* **全部错误**：组内分不出哪个回答更好，优势信号弱。
* **部分正确、部分错误**：最适合学习，因为模型可以知道哪些回答相对更好。

verl 官方文档也说明，DAPO 会过滤 accuracy 全为 0 或全为 1 的 groups，如果有效样本不够，会继续采样直到填满 batch。

---

#### 3. Token-Level Policy Gradient Loss

GRPO 常见做法是 sample-level loss，也就是先对每条回答内部的 token 求平均，再对不同回答求平均：

$$
L_{\text{sample}} =
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|y_i|}
\sum_{t=1}^{|y_i|}
L_{i,t}
$$

DAPO 改成 token-level loss，也就是直接在所有 token 上求平均：

$$
L_{\text{token}} =
\frac{1}{\sum_{i=1}^{G}|y_i|}
\sum_{i=1}^{G}
\sum_{t=1}^{|y_i|}
L_{i,t}
$$

* $|y_i|$：第 $i$ 个回答的长度。
* **物理意义**：在长 CoT 推理中，不同回答长度差异很大。如果按 sample-level 平均，长回答中的 token 梯度会被稀释。DAPO 改成 token-level 聚合后，每个 token 都能更直接地参与训练。

注意：

**DAPO 的 reward / advantage 仍然是回答级别的，但 loss 聚合方式变成了 token 级别。**

也就是说，同一个回答里的 token 仍然共享同一个 $\hat{A}_i$，只是最终 loss 不再先按每条回答平均。

---

#### 4. Overlong Reward Shaping

Long-CoT 训练中，模型容易生成特别长的回答。如果回答超过最大长度被截断，直接给 0 分或强惩罚，会产生奖励噪声。

DAPO 使用 **Overlong Reward Shaping**，也就是对过长回答进行软惩罚：

$$
r'_i = r_i + p_{\text{length}}(y_i)
$$

其中：

$$
p_{\text{length}}(y_i) \leq 0
$$

* 如果回答没有超长，则：

$$
p_{\text{length}}(y_i) = 0
$$

* 如果回答超过预设长度，则长度越长，惩罚越大。

* **物理意义**：这样既能抑制模型无限生成很长的回答，又不会因为回答稍微超长就直接把奖励变成强噪声。DAPO 论文将其称为 Soft Overlong Punishment，用于对被截断样本进行长度感知的奖励塑形。

---

### 2.4 DAPO 目标函数

合并上面的组内优势、非对称裁剪和 token-level loss，可以得到 DAPO 的目标函数：

$$
L_{\text{DAPO}}(\theta)
=
\mathbb{E}
\left[
\frac{1}{\sum_i |y_i|}
\sum_{i=1}^{G}
\sum_{t=1}^{|y_i|}
\min
\left(
\rho_{i,t}\hat{A}_i,
\text{clip}
\left(
\rho_{i,t},
1-\epsilon_{\text{low}},
1+\epsilon_{\text{high}}
\right)
\hat{A}_i
\right)
\right]
$$

* $\rho_{i,t}$：重要性采样比率：

$$
\rho_{i,t} =
\frac{\pi_\theta(a_{i,t}|s_{i,t})}
{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}
$$

* $\hat{A}_i$：第 $i$ 个回答的组内相对优势。
* $\epsilon_{\text{low}}$：下裁剪范围。
* $\epsilon_{\text{high}}$：上裁剪范围。
* $\sum_i |y_i|$：当前 batch 或 group 中所有回答的 token 总数。
* **没有 $\beta D_{KL}$ 项**：DAPO 论文版目标函数移除了 KL penalty。

与 GRPO 公式相比，DAPO 的变化可以理解为：

$$
\text{GRPO} = \text{Group Advantage} + \text{Symmetric Clip} + \text{KL Penalty}
$$

而：

$$
\text{DAPO} = \text{Group Advantage} + \text{Decoupled Clip} + \text{Dynamic Sampling} + \text{Token-Level Loss} + \text{Overlong Reward Shaping}
$$

---

## 3. 训练流程

### 3.1 初始化（Initialization）

在训练开始阶段，系统加载几个核心组件：

* **策略模型（Actor）** $\pi_\theta$：加载初始权重，作为训练主体。
* **旧策略模型（Old Policy）** $\pi_{\theta_{old}}$：用于生成 rollout，并记录生成 token 时的旧概率。
* **奖励模型 / 规则校验器** $R_\phi$：冻结，用于为回答打分。
* **无需初始化 Critic 模型**。
* **通常无需 Reference Model 计算 KL 惩罚**，因为 DAPO 论文版目标中移除了 KL penalty。

---

### 3.2 组内样本生成（Group Rollout Phase）

1. 从数据集中采样一个 Prompt $x$，或者一个 batch 的 Prompts。
2. 使用策略模型 $\pi_{\theta_{old}}$ 针对该 Prompt 生成 $G$ 个不同回答：

$$
{y_1, y_2, \dots, y_G}
$$

3. 记录生成每个回答时，每个 token 的旧策略概率：

$$
\pi_{\theta_{old}}(a_{i,t}|s_{i,t})
$$

这一步和 GRPO 很像，都是对同一个 Prompt 生成多个回答。

---

### 3.3 奖励打分与动态采样（Reward & Dynamic Sampling Phase）

1. 奖励模型或规则系统对这 $G$ 个回答进行评分：

$$
{r_1, r_2, \dots, r_G}
$$

2. 判断这个 group 是否有有效训练信号。

如果：

$$
\sum_{i=1}^{G} r_i = 0
$$

说明全部回答错误，丢弃。

如果：

$$
\sum_{i=1}^{G} r_i = G
$$

说明全部回答正确，丢弃。

如果：

$$
0 < \sum_{i=1}^{G} r_i < G
$$

说明部分正确、部分错误，保留。

3. 如果有效 group 数量不足，则继续采样新的 Prompt 或新的回答，直到填满训练 batch。

4. 对保留下来的 group 计算均值和标准差：

$$
\mu = \text{mean}(r_{1..G})
$$

$$
\sigma = \text{std}(r_{1..G})
$$

5. 计算每个回答的优势：

$$
\hat{A}_i = \frac{r_i-\mu}{\sigma}
$$

---

### 3.4 策略更新（DAPO Update Phase）

将当前经验池中收集到的有效数据打乱，切分成多个 Mini-batch。对这批数据进行若干次内部循环。对每个 Mini-batch 执行以下更新：

1. 将生成的回答重新输入到**当前策略模型** $\pi_\theta$ 中，计算当前 token 概率。
2. 对于每个回答的每个 token，计算重要性采样比率：

$$
\rho_{i,t} =
\frac{\pi_\theta(a_{i,t}|s_{i,t})}
{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}
$$

3. 使用 DAPO 的非对称裁剪：

$$
\text{clip}
\left(
\rho_{i,t},
1-\epsilon_{\text{low}},
1+\epsilon_{\text{high}}
\right)
$$

4. 构建 DAPO 的 token-level 目标函数：

$$
L_{\text{DAPO}}(\theta)
=
\mathbb{E}
\left[
\frac{1}{\sum_i |y_i|}
\sum_i
\sum_t
\min
\left(
\rho_{i,t}\hat{A}_i,
\text{clip}
\left(
\rho_{i,t},
1-\epsilon_{\text{low}},
1+\epsilon_{\text{high}}
\right)
\hat{A}_i
\right)
\right]
$$

5. 如果回答过长，则通过 Overlong Reward Shaping 对奖励进行长度惩罚。
6. 反向传播，使用优化器更新策略模型 $\pi_\theta$ 的参数。

---

### 3.5 数据清理与下一轮迭代

1. 丢弃当前已经用于更新的 rollout 数据。
2. 更新旧策略模型 $\pi_{\theta_{old}}$ 或同步当前策略权重。
3. 抽取下一个 batch 的 Prompts。
4. 继续生成 $G$ 个回答，进行奖励打分、动态采样、优势计算和策略更新。
5. 循环进行，直至模型收敛。

---

## 4. 局限性

1. **生成开销更高**：DAPO 使用 Dynamic Sampling，会过滤掉全对或全错的 group。如果有效 group 不够，就需要继续采样，因此 rollout 阶段的生成成本可能比普通 GRPO 更高。

2. **仍然缺乏细粒度的信度分配（Credit Assignment）**：DAPO 虽然使用 token-level loss，但优势 $\hat{A}_i$ 仍然是回答级别的。也就是说，同一个回答里的所有 token 仍然共享同一个优势值，它并不能精确判断到底是哪一步推理或哪个 token 导致最终奖励变高或变低。

3. **依赖可验证奖励**：DAPO 特别适合数学、代码这类可以用规则验证器判断对错的任务。但对于开放式对话、写作、偏好对齐等任务，很难设计稳定可靠的规则奖励。

4. **Dynamic Sampling 可能改变数据分布**：DAPO 会过滤掉全对和全错的样本，因此训练会更集中在“部分正确、部分错误”的中等难度样本上。这可以提高有效梯度比例，但也可能让模型较少接触特别简单或特别困难的问题。

5. **对超参数比较敏感**：DAPO 额外引入了 $\epsilon_{\text{low}}$、$\epsilon_{\text{high}}$、group size、动态采样次数、最大输出长度、超长惩罚系数等超参数。如果设置不合适，可能会导致训练不稳定、生成过长、探索不足或过度更新。

## 5. DAPO 和 GRPO 的区别

|对比项|GRPO|DAPO|
|---|---|---|
|是否需要 Critic|不需要|不需要|
|优势计算|组内 reward 标准化|组内 reward 标准化|
|Clip 方式|通常对称 clip|上下界分离，Clip-Higher|
|KL 惩罚|通常有 KL penalty|论文版移除 KL penalty|
|样本生成|每个 prompt 生成 G 个回答|生成 G 个回答后过滤无效 group|
|样本利用|全对/全错 group 也可能进入训练|过滤全对/全错 group|
|Loss 聚合|sample-level|token-level|
|长回答处理|容易稀释 token 梯度|所有 token 统一聚合|
|超长输出|容易产生奖励噪声|overlong reward shaping|
|适用场景|通用 RLHF / reasoning RL|更偏大规模 Long-CoT reasoning RL|

一句话总结：

**GRPO 是 PPO 去掉 Critic；DAPO 是在 GRPO 基础上专门为 Long-CoT 推理训练做系统级增强。**
