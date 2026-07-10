# verl

## 1. 核心概念

verl 是一个面向大模型强化学习后训练的框架. 它主要用于 [RLHF](../Align/RLHF.md), reasoning RL, verifiable reward 训练等场景.

与 [TRL](./TRL.md) 更偏单机实验和算法 Trainer 不同, verl 更强调:

**把 rollout, reward, advantage, policy update 等 RL 训练环节拆成可组合的分布式模块, 并支持大规模训练.**

verl 的核心思想可以理解为:

**用一个灵活的 RL 数据流系统, 把训练模型, 推理引擎, 奖励函数, 参考模型和分布式资源管理起来.**

它常用于:

- [PPO](../Align/PPO.md) 训练.
- [GRPO](../Align/GRPO.md) 训练.
- [DAPO](../Align/DAPO.md) 训练.
- 数学和代码类 verifiable reward 训练.
- 长 CoT reasoning RL.
- 大规模 on-policy rollout.

---

## 2. 核心组件

### 2.1 Actor

Actor 是当前正在训练的策略模型.

它负责:

1. 根据 Prompt 生成回答.
2. 在训练阶段计算当前策略概率.
3. 接收 RL loss 或 SFT loss 的梯度更新.

在 [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md), [DAPO](../Align/DAPO.md) 中, Actor 都是训练主体.

---

### 2.2 Rollout Engine

Rollout Engine 负责在线生成样本.

verl 通常可以结合高性能推理引擎, 例如 [vLLM](./vllm.md) 或 [SGLang](./SGLang.md), 来提高 rollout 吞吐.

Rollout 阶段需要保存:

- Prompt.
- Generated response.
- Token ids.
- Log probabilities.
- Attention mask.
- Response length.

对于 [GRPO](../Align/GRPO.md) / [DAPO](../Align/DAPO.md), 同一个 Prompt 往往需要生成多个 response:

$$
\{y_1,y_2,\dots,y_G\}
$$

这样后续才能计算组内相对优势.

---

### 2.3 Reference Model

Reference Model 是冻结的参考策略, 通常来自 SFT 模型. 它在 [DPO](../Align/DPO.md), [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md) 等算法中都很常见.

它的作用是:

- 计算 KL penalty.
- 限制 Actor 不要偏离初始模型太远.
- 为 [PPO](../Align/PPO.md) / [GRPO](../Align/GRPO.md) 提供 reference log probabilities.

在某些算法中, 例如 [DAPO](../Align/DAPO.md) 论文版目标, KL penalty 可能被移除, 此时 Reference Model 可以不参与核心 loss.

---

### 2.4 Critic

Critic 用于 [PPO](../Align/PPO.md), 负责预测状态价值.

它的作用是:

- 估计 value.
- 计算 advantage.
- 降低策略梯度方差.

但在 [GRPO](../Align/GRPO.md) / [DAPO](../Align/DAPO.md) 中, 通常不需要 Critic, 因为 advantage 来自同一个 Prompt 的组内 reward 标准化.

---

### 2.5 Reward Model / Verifier

Reward 模块负责给生成结果打分.

它可以是:

1. **Reward Model**: 基于偏好数据训练出的打分模型.
2. **Rule Verifier**: 数学, 代码, 格式等规则校验器.
3. **External Tool**: 单元测试, 编译器, 检索系统, judge 模型.
4. **Hybrid Reward**: 多个 reward 加权组合.

在 reasoning RL 中, verl 经常和 verifiable reward 结合, 例如答案正确给 1, 错误给 0.

---

### 2.6 Advantage Estimator

不同算法使用不同的 advantage 计算方式.

1. **[PPO](../Align/PPO.md)**: 通常使用 GAE, 需要 Critic.
2. **[GRPO](../Align/GRPO.md)**: 使用 group relative advantage, 不需要 Critic.
3. **[DAPO](../Align/DAPO.md)**: 在 GRPO 基础上加入 dynamic sampling, token-level loss, overlong reward shaping 等改动.

verl 的价值在于把这些步骤模块化, 让不同算法可以复用同一套 rollout 和 update 基础设施.

---

## 3. 训练流程

### 3.1 初始化

训练开始时通常需要准备:

- Actor model.
- Reference model.
- Critic model, 如果使用 [PPO](../Align/PPO.md).
- Reward model 或 verifier.
- Tokenizer.
- Prompt dataset.
- Rollout engine.
- 分布式资源配置.

---

### 3.2 Rollout Phase

1. 从 prompt dataset 中采样一批 Prompt.
2. Actor 使用 rollout engine 生成回答.
3. 保存生成 token 的 old log probabilities.
4. 如果是 [GRPO](../Align/GRPO.md) / [DAPO](../Align/DAPO.md), 对每个 Prompt 生成多个回答.

这一步通常是 RLHF 中最耗时的部分, 因为它需要实际生成大量 token.

---

### 3.3 Reward Phase

1. 将生成的回答发送给 reward model 或 verifier.
2. 得到每条回答的 reward.
3. 对格式错误, 超长输出, 空回答等特殊情况做 reward shaping.
4. 如果是 [DAPO](../Align/DAPO.md), 还可能过滤全对或全错的 group.

---

### 3.4 Advantage Phase

根据算法不同, 计算 advantage:

- [PPO](../Align/PPO.md) 使用 Critic 和 GAE.
- [GRPO](../Align/GRPO.md) 使用组内 reward 标准化.
- [DAPO](../Align/DAPO.md) 使用组内 reward 标准化 + dynamic sampling.

例如 [GRPO](../Align/GRPO.md) 中:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}
$$

---

### 3.5 Update Phase

1. 将 rollout 数据切分为 mini-batch.
2. 重新前向 Actor, 计算当前 log probabilities.
3. 计算 ratio, KL, policy loss 等.
4. 如果有 Critic, 同时计算 value loss.
5. 反向传播并更新 Actor.
6. 丢弃旧 rollout, 进入下一轮.

---

## 4. 特点

1. **面向 on-policy RL**: verl 专门处理生成, 打分, 更新循环, 比普通 SFT Trainer 更适合 RLHF.

2. **组件解耦**: Actor, Critic, Reference, Reward, Rollout 可以拆开部署和调度. 这和 [RLHF](../Align/RLHF.md) 的多模型流程天然对应.

3. **适合 reasoning RL**: 数学, 代码, verifier reward, long CoT 是 verl 的典型应用场景.

4. **支持多种算法**: [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md), [DAPO](../Align/DAPO.md) 等都可以在同一套框架中表达.

5. **可结合推理引擎**: rollout 可以接入 [vLLM](./vllm.md) 或 [SGLang](./SGLang.md), 提高生成吞吐.

6. **适合大规模分布式训练**: 相比单机 Trainer, verl 更关注多模型, 多角色, 多资源的 RL 训练系统.

---

## 5. 局限性

1. **系统复杂度高**: RLHF 本身就比 SFT 复杂, verl 又涉及分布式 rollout, reward, update, 调试成本较高.

2. **资源需求大**: Actor, Reference, Critic, Reward, Rollout engine 可能同时占用大量 GPU.

3. **Reward 设计仍然是核心难点**: 框架能执行 RL, 但不能自动保证 reward 合理.

4. **吞吐依赖配置**: rollout batch size, prompt length, response length, GPU 分配, [vLLM](./vllm.md) 配置都会影响训练效率.

5. **不适合简单微调任务**: 如果只是 SFT 或 [DPO](../Align/DPO.md) 小实验, [TRL](./TRL.md) 更轻量.

6. **版本和生态变化快**: 大模型 RL 框架仍在快速发展, 具体 API 和最佳实践需要参考当前版本.

---

## 6. verl 和 TRL 的区别

|对比项|[TRL](./TRL.md)|verl|
|---|---|---|
|核心定位|后训练算法 Trainer|大规模 RLHF / reasoning RL 系统|
|适合规模|小到中等规模实验|中到大规模分布式 RL|
|典型算法|SFT, [DPO](../Align/DPO.md), [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md)|[PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md), [DAPO](../Align/DAPO.md), verifier RL|
|Rollout|Trainer 内部生成|可接入高性能 rollout engine|
|系统复杂度|较低|较高|
|适合用户|研究实验, 快速复现|大规模训练, 工程化 RLHF|

一句话总结:

**verl 是面向大模型强化学习后训练的系统框架, 重点不是单个 loss 怎么写, 而是如何把 rollout, reward 和 policy update 高效组织起来.**
