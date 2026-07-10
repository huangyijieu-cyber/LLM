# TRL

## 1. 核心概念

TRL 全称是 **Transformer Reinforcement Learning**, 是 Hugging Face 生态中用于大模型后训练的框架. 它主要服务于 **SFT, Reward Modeling, Preference Optimization, [RLHF](../Align/RLHF.md)** 等流程.

TRL 本身不是一个底层分布式训练引擎, 而是一个 **算法训练框架**. 它把 Transformers, Datasets, [PEFT](../Finetune/PEFT.md), Accelerate, [DeepSpeed](./DeepSpeed.md), [FSDP](../Distributed_Training/ZeRO.md) 等组件封装起来, 让用户可以用相对统一的 Trainer 接口完成大模型对齐训练.

它的核心思想是:

**把复杂的后训练算法封装成 Trainer, 让研究者和工程师可以更快地实验 SFT, [DPO](../Align/DPO.md), [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md) 等对齐方法.**

在 TRL 中, 通常会涉及几个核心对象:

1. **Model**: 要训练的语言模型, 通常来自 Hugging Face Transformers.
2. **Tokenizer / Processor**: 负责文本或多模态输入的编码.
3. **Dataset**: 训练数据, 可以是指令数据, 偏好数据, prompt 数据或 reward 数据.
4. **Trainer**: TRL 对不同训练算法的封装, 例如 `SFTTrainer`, `DPOTrainer`, `PPOTrainer`, `GRPOTrainer`.
5. **Training Config**: 训练超参数, 包括 batch size, learning rate, max length, beta, reward function 等.

---

## 2. 核心模块

### 2.1 SFTTrainer

`SFTTrainer` 用于监督微调, 对应传统的 instruction tuning. 它通常是 [RLHF](../Align/RLHF.md), [DPO](../Align/DPO.md), [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md) 等后训练方法的起点.

它的输入通常是:

- Prompt.
- Response.
- Chat template.
- 可选 system prompt.

训练目标是让模型在给定 Prompt 时最大化标准答案的 token 概率.

SFTTrainer 的特点:

1. **上手简单**: 接近普通 Transformers Trainer.
2. **支持 chat template**: 适合指令模型和对话模型.
3. **支持 [PEFT](../Finetune/PEFT.md)**: 可以很方便地结合 LoRA, QLoRA 等参数高效微调方法.
4. **适合作为对齐第一步**: [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md), [DPO](../Align/DPO.md) 等方法通常都以 SFT 模型为初始策略.

---

### 2.2 RewardTrainer

`RewardTrainer` 用于训练奖励模型.

在 [RLHF](../Align/RLHF.md) 中, 奖励模型通常通过 pairwise preference 数据训练. 每条数据包含:

- Prompt $x$.
- 被偏好的回答 $y_w$.
- 被拒绝的回答 $y_l$.

训练目标是让奖励模型满足:

$$
R_\phi(x,y_w) > R_\phi(x,y_l)
$$

RewardTrainer 的特点:

1. **适合经典 RLHF 流程**: SFT -> Reward Model -> PPO.
2. **将偏好数据转成标量奖励**: 后续 PPO 可以直接使用 reward score.
3. **依赖数据质量**: 如果偏好数据存在噪声, 奖励模型会学到错误偏好.

---

### 2.3 Preference Optimization Trainers

TRL 提供了多种偏好优化 Trainer, 用于不显式训练奖励模型的对齐方法.

常见包括:

1. **DPOTrainer**: 实现 [Direct Preference Optimization](../Align/DPO.md). 直接使用 $(x,y_w,y_l)$ 偏好数据优化策略模型.
2. **KTOTrainer**: 使用正负反馈信号进行优化, 不一定需要成对比较.
3. **ORPOTrainer**: 将 SFT 和 preference optimization 合并到一个目标中.
4. **CPO / IPO 等 Trainer**: 用不同的偏好建模方式约束策略更新.

这类方法的共同特点是:

**不需要在线生成 rollout, 直接在离线偏好数据上训练.**

因此它们通常比 PPO 更简单, 成本更低, 也更容易复现.

---

### 2.4 PPOTrainer

`PPOTrainer` 用于经典 [RLHF](../Align/RLHF.md) 中的 [PPO](../Align/PPO.md) 阶段.

PPO 训练通常需要:

1. **Policy Model**: 当前要训练的模型.
2. **Reference Model**: 冻结的初始模型, 用于计算 KL penalty.
3. **Reward Model**: 对生成回答打分.
4. **Value Head / Critic**: 估计状态价值, 用于计算 advantage.

PPOTrainer 的特点:

- **on-policy**: 每轮训练都要让模型生成新回答.
- **训练更复杂**: 需要处理 rollout, reward, KL, advantage, value loss 等多个部分.
- **探索能力更强**: 相比 DPO, PPO 可以通过在线采样发现离线数据中没有的新回答.

---

### 2.5 GRPOTrainer

`GRPOTrainer` 用于基于组内相对优势的 [GRPO](../Align/GRPO.md) 强化学习训练.

GRPO 的关键思想是:

**对同一个 Prompt 生成多个回答, 根据组内 reward 相对高低计算 advantage, 从而避免单独训练 Critic.**

相比 PPO, GRPO 的特点是:

1. **不需要 Critic**: 显存占用更低.
2. **适合规则奖励任务**: 例如数学, 代码, 格式校验.
3. **需要多次采样**: 每个 Prompt 要生成 $G$ 个回答, rollout 成本较高.
4. **更适合 reasoning RL**: 尤其是可验证答案的长推理任务.

---

### 2.6 PEFT 和分布式支持

TRL 和 Hugging Face 生态结合很紧密, 因此可以自然使用:

- **[PEFT](../Finetune/PEFT.md)**: LoRA, QLoRA 等参数高效微调.
- **Accelerate**: 多 GPU 启动和设备管理.
- **[DeepSpeed](./DeepSpeed.md)**: [ZeRO](../Distributed_Training/ZeRO.md), offload, 大模型训练优化.
- **[FSDP](../Distributed_Training/ZeRO.md)**: PyTorch 原生参数分片训练.

这使得 TRL 很适合做中小规模后训练实验. 如果训练规模特别大, 通常还需要结合更底层的训练框架或专门的 RL 系统.

---

## 3. 使用流程

### 3.1 数据准备

根据训练方法不同, 数据格式也不同.

1. **SFT**: 需要 prompt-response 数据.
2. **DPO / IPO / ORPO**: 需要 preference pair, 即 $(x,y_w,y_l)$.
3. **Reward Model**: 需要 chosen / rejected 数据.
4. **PPO / GRPO**: 通常只需要 prompt, 回答由模型在线生成.

---

### 3.2 选择 Trainer

常见选择方式如下:

|任务|推荐 Trainer|
|---|---|
|指令微调|`SFTTrainer`|
|训练奖励模型|`RewardTrainer`|
|离线偏好对齐|[DPO](../Align/DPO.md) 的 `DPOTrainer`|
|经典 RLHF|[PPO](../Align/PPO.md) 的 `PPOTrainer`|
|数学 / 代码推理 RL|[GRPO](../Align/GRPO.md) 的 `GRPOTrainer`|

---

### 3.3 配置模型和训练参数

通常需要配置:

- 模型路径.
- tokenizer 或 processor.
- 最大输入长度.
- 最大生成长度.
- batch size.
- learning rate.
- PEFT 配置.
- DeepSpeed 或 FSDP 配置.
- 对齐算法特有参数, 例如 DPO 的 $\beta$, GRPO 的 group size.

---

### 3.4 训练和保存

训练流程通常是:

1. 加载模型和 tokenizer.
2. 加载数据集.
3. 构建 Trainer.
4. 调用 `trainer.train()`.
5. 保存模型或 LoRA adapter.
6. 使用 Transformers 或 vLLM / SGLang 进行推理部署.

---

## 4. 特点

1. **算法覆盖广**: TRL 覆盖了 SFT, reward modeling, DPO, PPO, GRPO 等主流对齐流程.

2. **Hugging Face 生态兼容性强**: 可以直接使用 Transformers 模型, Datasets 数据集, PEFT adapter, Accelerate 启动脚本.

3. **适合快速实验**: 如果目的是复现论文, 跑小规模实验, 比较不同对齐算法, TRL 的开发效率很高.

4. **配置成本低**: 相比从零实现 PPO 或 GRPO, TRL 已经封装了训练循环, loss, logging, checkpoint 等通用逻辑.

5. **适合教学和研究**: 代码接口较直观, 很适合理解各种 post-training 算法的差异.

---

## 5. 局限性

1. **不是极致性能框架**: TRL 更关注算法易用性, 不是专门为超大规模 RL 训练吞吐量设计的底层系统.

2. **大规模 on-policy RL 成本高**: PPO / GRPO 需要在线生成数据, 如果 rollout 很大, 仅靠普通 Trainer 架构可能不如 verl 这类专门 RL 系统灵活.

3. **分布式能力依赖外部组件**: TRL 自身不实现 ZeRO, FSDP, TP, PP, 需要依赖 Accelerate, DeepSpeed, PyTorch FSDP 等工具.

4. **复杂 RLHF 流程需要自己拼接**: 多 reward, 多模型部署, 异步 rollout, 大规模 verifier 等复杂系统能力需要用户额外开发.

5. **版本变化较快**: Trainer 名称, 参数和数据格式可能随 TRL 版本变化, 使用时需要对照当前官方文档.

---

## 6. TRL 和其他框架的区别

|框架|核心定位|更适合做什么|
|---|---|---|
|TRL|对齐算法 Trainer|SFT, [DPO](../Align/DPO.md), [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md) 实验|
|[verl](./Verl.md)|大规模 [RLHF](../Align/RLHF.md) / reasoning RL 系统|分布式 rollout + reward + update|
|[DeepSpeed](./DeepSpeed.md)|训练加速和显存优化|[ZeRO](../Distributed_Training/ZeRO.md), offload, 大模型训练|
|[Megatron](./Megatron.md)|超大模型预训练并行框架|[TP, PP, CP, EP](../Distributed_Training/Main.md) 混合并行|
|[vLLM](./vllm.md)|推理服务引擎|高吞吐 serving 和 rollout|
|[SGLang](./SGLang.md)|推理服务和结构化生成框架|Agent, structured output, prefix cache|

一句话总结:

**TRL 是 Hugging Face 生态中的大模型后训练工具箱, 适合快速实现和比较各种对齐算法.**
