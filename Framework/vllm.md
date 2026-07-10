# vLLM

## 1. 核心概念

vLLM 是一个面向大模型推理和服务部署的高吞吐推理引擎. 它和 [SGLang](./SGLang.md) 都属于推理框架, 不负责模型训练. 它的核心目标是:

**让 LLM serving 在高并发场景下更快, 更省显存, 更容易部署.**

vLLM 不是训练框架, 它主要负责模型推理. 在大模型系统中, vLLM 常用于:

- 在线 Chat Completion 服务.
- 离线批量推理.
- [RLHF](../Align/RLHF.md) / [GRPO](../Align/GRPO.md) 中的 rollout engine.
- OpenAI-compatible API server.
- 多 LoRA adapter 推理.

vLLM 最经典的技术是 **PagedAttention**, 用类似操作系统分页的方式管理 [KV cache](../Inference/KV_Cache.md), 解决长文本和高并发推理中的显存碎片问题.

---

## 2. 核心模块

### 2.1 PagedAttention

LLM 自回归生成时, 每生成一个 token, 都需要保存当前层的 Key 和 Value, 这部分缓存叫 **[KV cache](../Inference/KV_Cache.md)**.

普通推理系统中, KV cache 往往按连续显存块分配. 但不同请求的长度不同, 会导致:

- 显存碎片.
- 预留空间浪费.
- 并发请求数量受限.

PagedAttention 的思路是:

**把 [KV cache](../Inference/KV_Cache.md) 切成固定大小的 block, 像操作系统虚拟内存一样按需分配和映射.**

这样做的好处是:

1. 减少 KV cache 显存浪费.
2. 支持更多并发请求.
3. 更好地处理不同长度请求.
4. 方便实现 prefix sharing 和 beam search 等功能.

---

### 2.2 Continuous Batching

传统 batch 推理通常要等一个 batch 内所有请求都完成后, 才能处理下一批请求.

Continuous Batching 的思路是:

**每个 decoding step 动态调度请求, 已完成的请求退出, 新请求可以插入.**

这对在线服务非常重要, 因为真实请求往往:

- 到达时间不同.
- prompt 长度不同.
- 输出长度不同.

Continuous Batching 可以提高 GPU 利用率, 降低排队浪费.

---

### 2.3 OpenAI-Compatible Server

vLLM 提供 OpenAI-compatible API server.

这意味着很多应用可以用类似 OpenAI Chat Completion 的方式调用本地或私有模型.

常见接口包括:

- Chat completion.
- Completion.
- Embedding.
- Streaming response.

这种兼容性让 vLLM 很适合作为生产服务的后端.

---

### 2.4 Prefix Caching

Prefix Caching 用于复用相同前缀的 [KV cache](../Inference/KV_Cache.md).

如果多个请求共享同一段长 prompt, 例如:

- 相同 system prompt.
- 相同工具说明.
- 相同 RAG 文档.
- 同一个问题采样多个回答.

那么 prefix 部分可以只 prefill 一次, 后续请求复用缓存.

这对以下场景很有用:

1. Agent tool schema 很长.
2. RAG context 很长.
3. Best-of-N / self-consistency 需要多次采样.
4. RL rollout 中同一个 prompt 生成多个 response.

---

### 2.5 Speculative Decoding

Speculative Decoding 是一种加速解码的方法.

基本思路是:

1. 用一个小模型或 draft model 快速提出多个候选 token.
2. 用大模型一次性验证这些 token.
3. 如果候选被接受, 就能减少大模型逐 token 调用次数.

它适合解码瓶颈明显的场景, 但收益取决于 draft model 质量和接受率.

---

### 2.6 Quantization 和 LoRA

vLLM 支持多种推理优化能力, 常见包括:

- Weight quantization.
- KV cache quantization.
- [LoRA](../Finetune/PEFT.md) adapter serving.
- Multi-[LoRA](../Finetune/PEFT.md) serving.

这些能力可以降低部署成本, 或让一个 base model 同时服务多个微调任务.

---

### 2.7 并行部署

vLLM 支持在多 GPU 上部署模型.

常见并行方式包括:

- Tensor Parallelism.
- Pipeline Parallelism.
- Data Parallel style replica.

对于大模型服务, 通常要根据模型大小和请求量选择并行策略:

- 模型单卡放得下: 多副本提高并发.
- 模型单卡放不下: 使用 TP 或 PP 切分模型.
- 请求量很大: 多实例部署并配合负载均衡.

---

## 3. 使用流程

### 3.1 离线推理

离线推理通常用于批量生成:

1. 加载模型.
2. 构造 prompts.
3. 设置 sampling parameters.
4. 调用 vLLM engine 生成.
5. 保存输出结果.

适合数据生成, 评测, 蒸馏数据构造等任务.

---

### 3.2 在线服务

在线服务通常流程是:

1. 启动 vLLM server.
2. 指定模型路径, tensor parallel size, dtype, max context length 等.
3. 应用侧通过 OpenAI-compatible API 发送请求.
4. vLLM runtime 自动完成 batching, scheduling 和 KV cache 管理.

---

### 3.3 作为 RL Rollout Engine

在 [RLHF](../Align/RLHF.md), [GRPO](../Align/GRPO.md), [DAPO](../Align/DAPO.md) 中, 训练系统需要大量生成回答.

vLLM 可以作为 rollout engine:

1. 接收 prompt batch.
2. 为每个 prompt 生成一个或多个 response.
3. 返回 token ids, text, log probabilities.
4. 交给 reward / verifier 打分.
5. 再由训练框架更新 policy model.

这也是 [verl](./Verl.md) 等 RL 框架经常接入推理引擎的原因.

---

## 4. 特点

1. **高吞吐 serving**: vLLM 的重点是提高在线推理吞吐和 GPU 利用率.

2. **[KV cache](../Inference/KV_Cache.md) 管理强**: PagedAttention 是 vLLM 的核心优势, 对高并发和长输出很重要.

3. **部署接口友好**: OpenAI-compatible API 让上层应用迁移成本低.

4. **适合 rollout**: 在强化学习后训练中, 大量生成样本可以交给 vLLM 提速.

5. **支持多种推理优化**: 包括 prefix caching, speculative decoding, quantization, LoRA 等.

---

## 5. 局限性

1. **不是训练框架**: vLLM 不负责反向传播, optimizer, SFT, DPO 或 PPO 更新.

2. **模型支持依赖版本**: 新模型结构, 新 tokenizer, 新多模态处理逻辑都需要 runtime 支持.

3. **长上下文仍然消耗大量 [KV cache](../Inference/KV_Cache.md)**: PagedAttention 能减少浪费, 但不能让 KV cache 消失.

4. **性能依赖 workload**: prompt 长度, 输出长度, 并发量, batch 调度都会影响吞吐.

5. **复杂 Agent 编排不是核心能力**: vLLM 更偏 serving engine, 如果需要复杂多步 LLM program, [SGLang](./SGLang.md) 可能更贴合.

---

## 6. vLLM 和 SGLang 的区别

|对比项|vLLM|[SGLang](./SGLang.md)|
|---|---|---|
|核心定位|高吞吐推理引擎|推理 runtime + LLM 编程框架|
|代表技术|PagedAttention|RadixAttention|
|重点能力|KV cache 分页, continuous batching, serving API|prefix reuse, structured output, language program|
|适合场景|通用 serving, batch inference, RL rollout|Agent, 多步生成, 结构化输出|
|训练能力|不负责训练|不负责训练|
|API 风格|OpenAI-compatible serving 更突出|程序化 workflow 更突出|

一句话总结:

**vLLM 是大模型推理服务的高吞吐引擎, 核心价值在于更高效地调度请求和管理 KV cache.**
