# Framework

本目录主要记录大模型训练, 后训练和推理部署中常见的工程框架. 这些框架并不是同一类东西, 有的负责 **训练加速**, 有的负责 **对齐算法**, 有的负责 **推理服务**, 有的负责 **大规模 RLHF 系统**.

可以先按下面这条主线理解:

$$
\text{Pretrain / SFT}
\rightarrow
\text{Preference Alignment / RLHF}
\rightarrow
\text{Inference Serving}
$$

不同框架大致对应不同阶段:

|阶段|常见框架|主要作用|
|---|---|---|
|大模型预训练|[Megatron](./Megatron.md), [DeepSpeed](./DeepSpeed.md)|解决大模型训练时的并行和显存问题|
|监督微调 / 偏好对齐|[TRL](./TRL.md)|快速实现 SFT, [DPO](../Align/DPO.md), [PPO](../Align/PPO.md), [GRPO](../Align/GRPO.md) 等算法|
|大规模 RLHF / Reasoning RL|[verl](./Verl.md)|组织 rollout, reward, advantage, policy update 等 RL 训练流程|
|推理服务 / Rollout|[vLLM](./vllm.md), [SGLang](./SGLang.md)|高吞吐生成, KV cache 管理, serving, structured output|

---

## 1. 训练类框架

### 1.1 Megatron

[Megatron](./Megatron.md) 主要用于 **超大规模 Transformer 预训练**.

它的核心能力是把模型本身切开, 使用:

- [Tensor Parallelism](../Distributed_Training/Main.md).
- [Pipeline Parallelism](../Distributed_Training/Main.md).
- [Data Parallelism](../Distributed_Training/Main.md).
- [Sequence Parallelism](../Distributed_Training/Main.md).
- Expert Parallelism.

简单理解:

**模型太大, 单张 GPU 放不下或训练不动时, Megatron 负责把模型高效切到很多 GPU 上.**

适合场景:

1. 从头预训练大模型.
2. 训练百亿, 千亿级别模型.
3. 训练 [MoE](../basic/MOE.md) 模型.
4. 需要 TP + PP + DP 等多维并行.

不太适合:

1. 简单 LoRA 微调.
2. 单机小规模实验.
3. 快速跑 [DPO](../Align/DPO.md) 或 [SFT](../Finetune/Main.md) baseline.

---

### 1.2 DeepSpeed

[DeepSpeed](./DeepSpeed.md) 主要用于 **训练加速和显存优化**.

它最重要的能力是 [ZeRO](../Distributed_Training/ZeRO.md):

- ZeRO-1: 切分 optimizer state.
- ZeRO-2: 切分 optimizer state 和 gradients.
- ZeRO-3: 切分 optimizer state, gradients 和 parameters.

简单理解:

**模型训练显存不够时, DeepSpeed 负责减少数据并行中的冗余存储.**

适合场景:

1. 大模型全参数微调.
2. 大模型 [PEFT](../Finetune/PEFT.md) 微调.
3. Hugging Face Trainer / [TRL](./TRL.md) 训练时省显存.
4. 和 [Megatron](./Megatron.md) 组合做大规模训练.

不太适合:

1. 单纯推理服务.
2. 复杂 Agent 编排.
3. 只想做最简单的小模型实验.

---

## 2. 后训练和对齐框架

### 2.1 TRL

[TRL](./TRL.md) 是 Hugging Face 生态中的 **后训练算法工具箱**.

它把很多对齐算法封装成 Trainer, 例如:

- `SFTTrainer`: 指令微调.
- `RewardTrainer`: 奖励模型训练.
- `DPOTrainer`: [DPO](../Align/DPO.md).
- `PPOTrainer`: [PPO](../Align/PPO.md).
- `GRPOTrainer`: [GRPO](../Align/GRPO.md).

简单理解:

**想快速跑 SFT, DPO, PPO, GRPO 这类对齐实验时, 优先看 TRL.**

适合场景:

1. 快速复现对齐算法.
2. 小到中等规模 SFT / DPO / PPO / GRPO 实验.
3. 结合 [PEFT](../Finetune/PEFT.md) 做 LoRA 或 QLoRA 微调.
4. 教学和研究中比较不同 post-training 方法.

不太适合:

1. 极大规模 on-policy RL.
2. 复杂分布式 rollout.
3. 多 reward, 多模型异步调度.

---

### 2.2 verl

[verl](./Verl.md) 是面向 **大规模 RLHF / reasoning RL** 的训练系统.

它重点处理的是:

- Actor rollout.
- Reference logprob.
- Reward / verifier.
- Advantage estimation.
- Policy update.
- 多模型分布式调度.

简单理解:

**TRL 更像算法 Trainer, verl 更像完整 RLHF 训练系统.**

适合场景:

1. 大规模 [PPO](../Align/PPO.md) 训练.
2. 大规模 [GRPO](../Align/GRPO.md) / [DAPO](../Align/DAPO.md) 训练.
3. 数学, 代码等可验证奖励任务.
4. 长 [CoT](../Inference/CoT.md) reasoning RL.
5. 需要结合 [vLLM](./vllm.md) 或 [SGLang](./SGLang.md) 做高吞吐 rollout.

不太适合:

1. 简单 SFT.
2. 单机小规模 [DPO](../Align/DPO.md).
3. 不需要在线生成的普通监督训练.

---

## 3. 推理类框架

### 3.1 vLLM

[vLLM](./vllm.md) 是一个 **高吞吐 LLM 推理服务引擎**.

它的核心能力是:

- PagedAttention.
- [KV Cache](../Inference/KV_Cache.md) 管理.
- Continuous batching.
- OpenAI-compatible API.
- Prefix caching.
- LoRA serving.

简单理解:

**模型训练完以后, 想高并发部署或大批量生成, vLLM 是通用推理引擎.**

适合场景:

1. 在线 Chat Completion 服务.
2. 离线批量推理.
3. [RLHF](../Align/RLHF.md) / [GRPO](../Align/GRPO.md) 中的大规模 rollout.
4. 多 LoRA adapter serving.
5. 高吞吐 OpenAI-compatible API.

不太适合:

1. 训练模型.
2. 复杂多步 Agent workflow 编排.
3. 需要强结构化输出控制的任务.

---

### 3.2 SGLang

[SGLang](./SGLang.md) 是一个 **推理服务 + LLM 编程框架**.

它的核心能力是:

- RadixAttention.
- Prefix [KV Cache](../Inference/KV_Cache.md) reuse.
- Structured output.
- 多步 language program.
- Agent workflow.
- OpenAI-compatible serving.

简单理解:

**vLLM 更偏通用高吞吐 serving, SGLang 更偏复杂 LLM 应用编排和结构化生成.**

适合场景:

1. Agent 多轮工具调用.
2. JSON / regex / schema 结构化输出.
3. 多个请求共享长前缀.
4. RAG 长上下文复用.
5. 多分支生成和复杂 reasoning workflow.

不太适合:

1. 模型训练.
2. 只需要最简单批量推理的任务.
3. 不需要结构化输出或复杂 workflow 的服务.

---

## 4. 如何选择

如果目标是 **从头训练大模型**:

1. 优先看 [Megatron](./Megatron.md).
2. 显存优化和 ZeRO 看 [DeepSpeed](./DeepSpeed.md).
3. 相关并行知识看 [Distributed Training](../Distributed_Training/Main.md).

如果目标是 **微调或偏好对齐**:

1. 简单 SFT / DPO / PPO / GRPO 实验看 [TRL](./TRL.md).
2. 参数高效微调看 [PEFT](../Finetune/PEFT.md).
3. 大规模 RLHF / reasoning RL 看 [verl](./Verl.md).

如果目标是 **部署推理服务**:

1. 通用高吞吐 serving 看 [vLLM](./vllm.md).
2. 复杂 Agent, structured output, prefix reuse 看 [SGLang](./SGLang.md).
3. 推理显存和 [KV Cache](../Inference/KV_Cache.md) 原理看 [Inference](../Inference/Main.md).

---

## 5. 总结

|框架|一句话定位|最常用场景|
|---|---|---|
|[Megatron](./Megatron.md)|超大 Transformer 预训练并行框架|从头训练大模型|
|[DeepSpeed](./DeepSpeed.md)|训练显存优化和加速框架|ZeRO, offload, 大模型微调|
|[TRL](./TRL.md)|Hugging Face 后训练算法工具箱|SFT, DPO, PPO, GRPO 实验|
|[verl](./Verl.md)|大规模 RLHF / reasoning RL 系统|rollout + reward + update|
|[vLLM](./vllm.md)|高吞吐推理服务引擎|在线 serving, 批量生成, rollout|
|[SGLang](./SGLang.md)|LLM 推理编程和结构化生成框架|Agent, structured output, prefix cache|

一句话总结:

**Megatron 和 DeepSpeed 解决训练时怎么放得下, 训得快; TRL 和 verl 解决后训练怎么对齐; vLLM 和 SGLang 解决推理时怎么跑得快, 服务得稳.**
