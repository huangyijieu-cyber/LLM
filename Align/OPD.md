# OPD

## 1. 核心概念

OPD 全称是 **On-Policy Distillation**, 即在线策略蒸馏. 它是一种用于大模型后训练和模型压缩的蒸馏方法, 主要用于解决传统离线蒸馏中的 **Exposure Bias(暴露偏差)** 问题.

传统 Knowledge Distillation(KD) 通常让 Student Model 在一批固定的 Teacher 生成数据或人工数据上学习. 这类方法的问题是: 训练时 Student 看到的是 Teacher 写好的正确前缀, 但推理时 Student 必须基于自己已经生成的前缀继续往下写. 一旦 Student 早期生成出错, 后续上下文就会偏离训练分布, 错误会在自回归生成中不断放大.

OPD 的核心思想是:

**让 Student Model 自己生成回答, 然后让 Teacher Model 在 Student 自己生成的前缀上给出监督信号, 使 Student 学会在自己真实会遇到的状态中纠错和恢复.**

在 OPD 训练过程中, 通常需要维护两个主要模型:

1. **Student Model(学生模型)** $p_\theta$: 我们要训练的 LLM. 它负责根据 Prompt 生成 Token, 并在训练中更新参数. **它是可训练的.**
2. **Teacher Model(教师模型)** $p_T$: 能力更强的大模型, 用来对 Student 生成的前缀给出 token-level 分布, logit, reward 或 preference 信号. **参数冻结.**

有时还会额外使用:

3. **Dataset(提示词数据集)** $D$: 只提供输入 Prompt $x$, 不一定需要标准答案.
4. **Reward Model / Rule Verifier(奖励模型或规则校验器)** $R_\phi$: 当 OPD 和 RL 结合时, 用于提供 outcome-level 奖励.

和普通离线蒸馏相比, OPD 的区别在于:

$$
\text{Offline KD}: y \sim p_T(y|x) \quad \text{or} \quad y \sim D
$$

而:

$$
\text{OPD}: y \sim p_\theta(y|x)
$$

也就是说, OPD 的训练样本来自 **Student 当前策略自己的生成分布**, 而不是固定的 Teacher 数据集.

---

## 2. 基本公式

### 2.1 自回归生成建模

对于一个 Prompt $x$, LLM 生成回答 $y = \{y_1, y_2, \dots, y_T\}$ 时, 可以写成逐 token 的条件概率乘积:
$$
p_\theta(y|x)=\prod_{t=1}^{T}p_\theta(y_t|x,y_{<t})
$$
其中:
- $p_\theta$: Student Model 的策略分布.
- $p_T$: Teacher Model 的策略分布.
- $y_{<t}$: 当前 token 之前已经生成的前缀.
- $p_\theta(\cdot|x,y_{<t})$: Student 在当前前缀下对下一个 token 的概率分布.
- $p_T(\cdot|x,y_{<t})$: Teacher 在同一个前缀下对下一个 token 的概率分布.

OPD 的关键点是: 这里的 $y_{<t}$ 不是 Teacher 写出来的标准前缀, 而是 Student 自己生成出来的前缀.

---

### 2.2 On-Policy Rollout

对于训练集中的 Prompt $x$, OPD 先让当前 Student Model 生成回答:

$$
y \sim p_{\theta_{old}}(y|x)
$$

这里通常使用 $p_{\theta_{old}}$ 表示生成 rollout 时的 Student 快照. 在当前更新阶段, 这批生成数据被视作固定样本, **不对采样过程本身反向传播**.

得到回答 $y$ 后, 对每一个前缀 $(x,y_{<t})$, Teacher 给出下一个 token 的监督分布:

$$
p_T(\cdot|x,y_{<t})
$$

Student 则给出:

$$
p_\theta(\cdot|x,y_{<t})
$$

OPD 的训练目标就是让 Student 在这些自己生成的状态上尽量接近 Teacher.

---

### 2.3 Token-Level Distillation Loss

对一条 Student 生成的回答 $y$, 可以定义 Teacher 和 Student 在每个 token 位置上的分布差异:

$$
D(p_T,p_\theta)(y|x)
=
\frac{1}{|y|}
\sum_{t=1}^{|y|}
D
\left(
p_T(\cdot|x,y_{<t})
||
p_\theta(\cdot|x,y_{<t})
\right)
$$

其中:

- $D(\cdot||\cdot)$: 某种分布距离, 常见选择包括 Forward KL, Reverse KL, JS Divergence 等.
- $|y|$: 当前回答的 token 数.
- **物理意义**: Teacher 在 Student 自己写出的每一个前缀上告诉它: "在这个位置, 更合理的下一个 token 分布应该是什么".

OPD 的整体目标函数可以写成:

$$
L_{\text{OPD}}(\theta)
=
\mathbb{E}_{x \sim D}
\left[
\mathbb{E}_{y \sim p_{\theta_{old}}(\cdot|x)}
\left[
\sum_{t=1}^{|y|}
D_f
\left(
p_T(\cdot|x,y_{<t})
||
p_\theta(\cdot|x,y_{<t})
\right)
\right]
\right]
$$

注意:

**OPD 是 on-policy 的, 因为训练数据 $y$ 来自 Student 自己的当前生成分布. 但在一次参数更新中, 通常不对采样动作本身求梯度, 只对 Teacher-Student 的分布差异求梯度.**

---

### 2.4 Forward KL

Forward KL 的形式是:

$$
D_{KL}(p_T||p_\theta)
=
\sum_{v \in V}
p_T(v|x,y_{<t})
\log
\frac{p_T(v|x,y_{<t})}
{p_\theta(v|x,y_{<t})}
$$

其中 $V$ 是词表.

Forward KL 的特点是 **mode-covering**, 也就是 Student 会尽量覆盖 Teacher 认为可能的所有 token.

* **优点**: 保留 Teacher 分布中的多样性, 适合开放式生成, 对话, 写作等多解任务.
* **缺点**: 如果 Student 容量较小, 它可能无法覆盖 Teacher 的所有模式, 容易学到一个比较平均的分布, 生成结果可能不够锐利.

---

### 2.5 Reverse KL

Reverse KL 的形式是:

$$
D_{KL}(p_\theta||p_T)
=
\sum_{v \in V}
p_\theta(v|x,y_{<t})
\log
\frac{p_\theta(v|x,y_{<t})}
{p_T(v|x,y_{<t})}
$$

Reverse KL 的特点是 **mode-seeking**, 也就是 Student 更倾向于集中到 Teacher 概率最高的模式上.

* **优点**: 对数学, 代码, 推理这类通常存在较明确正确路径的任务更合适, 可以让 Student 更坚定地学习高质量路径.
* **缺点**: 容易牺牲多样性, 如果 Teacher 在 Student 错误前缀上的判断不可靠, Reverse KL 可能会放大错误监督.

因此在很多 OPD 实践中, Reverse KL 或 JS Divergence 会比单纯 Forward KL 更常见.

---

### 2.6 Mixed On-Policy Distillation

纯 OPD 完全依赖 Student 自己生成的数据. 如果训练早期 Student 太弱, 生成的前缀质量很差, Teacher 在这些前缀上的监督也可能不稳定. 因此常见做法是混合离线样本和在线样本:

$$
y \sim
\begin{cases}
p_{\theta_{old}}(y|x), & \text{with probability } \lambda \\
D_{\text{offline}}, & \text{with probability } 1-\lambda
\end{cases}
$$

其中:

- $\lambda$: Student on-policy 数据比例.
- $D_{\text{offline}}$: 固定离线数据, 可以是人工答案, Teacher 生成答案, 或历史高质量轨迹.

当:

$$
\lambda = 0
$$

退化为普通离线 KD.

当:

$$
\lambda = 1
$$

就是纯 OPD.

* **物理意义**: 训练早期可以多用离线高质量样本稳定学习, 随着 Student 变强, 逐渐提高 on-policy 样本比例, 让训练分布更接近推理分布.

---

### 2.7 OPD 和 RL 的结合

OPD 也可以和 GRPO, PPO, DAPO 这类 RL 算法结合. RL 提供 outcome-level 奖励, OPD 提供 token-level 密集监督.

一个简化的混合目标可以写成:

$$
L_{\text{total}}(\theta)
=
L_{\text{RL}}(\theta)
+
\alpha L_{\text{OPD}}(\theta)
$$

其中:

- $L_{\text{RL}}$: 来自 PPO / GRPO / DAPO 的强化学习目标.
- $L_{\text{OPD}}$: Teacher 对 Student 自生成轨迹的蒸馏损失.
- $\alpha$: 控制蒸馏信号强度的超参数.

如果使用最大化形式, 也可以理解为:

$$
J_{\text{total}}(\theta)
=
J_{\text{RL}}(\theta)
-
\alpha L_{\text{OPD}}(\theta)
$$

* **物理意义**: RL 告诉模型 "最终答案好不好", OPD 告诉模型 "每一步更像强 Teacher 会怎么写". 这样可以缓解 RL 奖励稀疏的问题.

---

## 3. 训练流程

### 3.1 初始化(Initialization)

在训练开始阶段, 系统加载两个核心模型:

- **Student Model** $p_\theta$: 从 SFT 模型或较小的 base model 加载权重, 作为训练主体.
- **Teacher Model** $p_T$: 加载更强的大模型, **冻结参数**, 只用于提供监督信号.
- **Prompt Dataset** $D$: 提供输入 Prompt $x$.
- **可选 Reward / Verifier** $R_\phi$: 如果和 RL 结合, 用于给完整回答打分.

OPD 通常 **不需要 Critic 模型**. 如果只是纯蒸馏, 也不需要 Reward Model.

---

### 3.2 Student Rollout Phase

1. 从数据集中采样一批 Prompt:

$$
x \sim D
$$

2. 使用 Student Model 生成回答:

$$
y \sim p_{\theta_{old}}(y|x)
$$

3. 保存每条回答的完整 token 序列:

$$
y = \{y_1,y_2,\dots,y_T\}
$$

4. 对每个位置构造前缀:

$$
(x,y_{<t})
$$

这一步是 OPD 和普通离线 KD 的根本区别. 普通 KD 学的是 Teacher 前缀, OPD 学的是 Student 自己真实会走到的前缀.

---

### 3.3 Teacher Feedback Phase

对于 Student 生成出的每一个前缀 $(x,y_{<t})$, 将它输入 Teacher Model, 得到 Teacher 的下一 token 分布:

$$
p_T(\cdot|x,y_{<t})
$$

如果 Teacher 是 white-box model, 可以直接拿到 logits 或概率分布.

如果 Teacher 是 black-box API, 可能只能拿到:

- 采样结果.
- 排序结果.
- 标量分数.
- pairwise preference.

此时 OPD 需要把这些弱监督信号转成 token-level 或 sequence-level 的训练目标.

---

### 3.4 Distillation Loss Phase

将同样的前缀输入 Student Model, 得到:

$$
p_\theta(\cdot|x,y_{<t})
$$

然后计算 Teacher 和 Student 的分布差异:

$$
D_f
\left(
p_T(\cdot|x,y_{<t})
||
p_\theta(\cdot|x,y_{<t})
\right)
$$

对整条回答的所有 token 求和或求平均:

$$
L_{\text{OPD}}
=
\frac{1}{|y|}
\sum_{t=1}^{|y|}
D_f
\left(
p_T(\cdot|x,y_{<t})
||
p_\theta(\cdot|x,y_{<t})
\right)
$$

常见选择:

1. **Forward KL**: 更偏多样性, 但可能学得平均.
2. **Reverse KL**: 更偏高置信路径, 适合推理和代码.
3. **JS Divergence**: 在 Forward KL 和 Reverse KL 之间折中.

---

### 3.5 Student Update Phase

1. 对 $L_{\text{OPD}}$ 反向传播.
2. 使用优化器更新 Student Model 参数 $\theta$.
3. Teacher Model 保持冻结.
4. 当前 rollout 数据用完后丢弃.
5. 用更新后的 Student 继续生成下一批 on-policy 数据.

这形成一个不断循环的过程:

$$
\text{Student Generate}
\rightarrow
\text{Teacher Feedback}
\rightarrow
\text{Distill Update}
\rightarrow
\text{Student Generate Again}
$$

随着 Student 变强, 它生成的数据分布也会不断变化, 因此 OPD 的训练分布是非静态的.

---

### 3.6 可选: 和 RL 一起训练

如果 OPD 和 GRPO / DAPO 一起使用, 一轮训练可以变成:

1. Student 对同一个 Prompt 生成一个或多个回答.
2. Reward Model / Verifier 对完整回答给出 outcome reward.
3. Teacher Model 对 Student 的中间前缀给出 token-level 分布.
4. 使用 GRPO / DAPO 计算策略梯度目标.
5. 使用 OPD 计算蒸馏损失.
6. 合并两个 loss 后更新 Student.

此时可以把 OPD 理解成 RL 的 **dense guidance**, 它给稀疏奖励补上更密集的逐 token 训练信号.

---

## 4. 局限性

1. **Teacher 推理成本高**: OPD 需要 Teacher 在 Student 自己生成的前缀上反复给出反馈. 如果 Teacher 很大, 每轮训练都要额外跑 Teacher forward, 成本会非常高.

2. **训练分布非静态**: Student 更新后, 下一轮生成的数据分布也会改变. 这使得 OPD 的训练过程比普通离线 KD 更复杂, 也更容易受到 rollout 质量影响.

3. **Teacher 在错误前缀上可能不可靠**: OPD 虽然让 Teacher 看到 Student 的错误前缀, 但如果 Student 的前缀已经严重偏离正常分布, Teacher 的下一步建议也可能变得不稳定. 这有时被称为反向的 Exposure Bias.

4. **容易受 Teacher-Student 差距影响**: 如果 Student 太弱, 早期生成的轨迹质量很低, Teacher 的监督信号可能很噪. 如果 Teacher 太强且分布太尖锐, Student 又可能学不动或发生模式坍缩.

5. **可能损失多样性**: 使用 Reverse KL 时, Student 往往会集中到 Teacher 的高概率模式上. 对数学和代码可能是优点, 但对开放式对话和写作可能会降低多样性.

6. **需要选择合适的 divergence**: Forward KL, Reverse KL, JS Divergence 的训练行为不同. 如果任务需要唯一正确答案, 通常更偏向 Reverse KL. 如果任务需要多种合理表达, 通常要保留 Forward KL 或 JS Divergence.

7. **依赖 logit 访问和 tokenizer 兼容**: 标准 token-level OPD 通常需要拿到 Teacher 的 next-token logits, 并且默认 Teacher 和 Student 的 token 空间可以对齐. 如果 Teacher 是 black-box API, 或者 Teacher 与 Student 使用不同 tokenizer, 就需要额外的 token mapping, top-k approximation 或 sequence-level preference 蒸馏.

8. **纯 OPD 容易受 Teacher 上限限制**: 如果没有额外 reward 或 verifier, Student 本质上是在模仿 Teacher. 它通常很难系统性超过 Teacher, 只能在成本, 部署和稳定性上受益.

---

## 5. OPD 和 Offline KD 的区别

|对比项|Offline KD|OPD|
|---|---|---|
|训练样本来源|固定数据集或 Teacher 生成数据|Student 当前策略自己生成的数据|
|是否 on-policy|不是|是|
|是否缓解 Exposure Bias|较弱|较强|
|Teacher 监督位置|通常在标准前缀上监督|在 Student 自己生成的前缀上监督|
|训练分布|静态|随 Student 更新而变化|
|计算成本|较低, 可预先生成数据|较高, 需要反复 rollout 和 Teacher feedback|
|常见 loss|Forward KL / CE|Forward KL / Reverse KL / JS Divergence|
|适合场景|普通模型压缩, 数据蒸馏|长推理, 代码, 数学, agent, 多步生成|
|主要风险|训练-推理分布不一致|Teacher 在坏前缀上监督不可靠|

一句话总结:

**Offline KD 是让 Student 学 Teacher 写好的答案; OPD 是让 Student 先自己写, 再让 Teacher 在 Student 自己写到的位置上纠正它.**

---

## 6. OPD 和 GRPO / DAPO 的关系

|对比项|GRPO / DAPO|OPD|
|---|---|---|
|核心信号|Reward / Verifier 给出的 outcome reward|Teacher 给出的 token-level 分布或偏好|
|信号粒度|通常是回答级别, 再广播到 token|通常是 token 级别|
|是否需要 Teacher|不一定, 可以只用规则奖励|通常需要 Teacher|
|是否需要 Reward|需要奖励或规则打分|纯 OPD 不需要|
|训练目标|最大化奖励并限制策略更新|让 Student 在自生成轨迹上接近 Teacher|
|优势|可以直接优化任务目标|监督更密集, 稳定性更好|
|局限|奖励稀疏, credit assignment 难|Teacher 成本高, 可能受 Teacher 上限限制|

组合起来可以理解为:

$$
\text{GRPO / DAPO}: \text{告诉模型最终哪条回答更好}
$$

$$
\text{OPD}: \text{告诉模型每一步更像强 Teacher 会怎么走}
$$

因此, 在长 CoT 推理训练中, OPD 经常可以作为 GRPO / DAPO 的辅助项, 用来提供更密集的中间监督信号.
