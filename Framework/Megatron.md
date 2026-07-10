# Megatron

## 1. 核心概念

Megatron 通常指 **Megatron-LM / Megatron-Core**, 是 NVIDIA 生态中用于训练超大规模 [Transformer](../basic/Transformer%20架构.md) 模型的高性能分布式训练框架.

它的核心目标不是让单卡训练更方便, 而是解决:

**当模型参数, 激活值和训练数据都大到单机单卡无法承载时, 如何把模型高效切分到大量 GPU 上训练.**

Megatron 的核心思想是:

**围绕 [Transformer](../basic/Transformer%20架构.md) 结构设计多维并行策略, 将 [Tensor Parallelism](../Distributed_Training/Main.md), [Pipeline Parallelism](../Distributed_Training/Main.md), [Data Parallelism](../Distributed_Training/Main.md), [Sequence Parallelism](../Distributed_Training/Main.md), Context Parallelism, Expert Parallelism 等组合起来.**

它更偏底层和系统工程, 常用于:

- 大模型预训练.
- 超大规模语言模型训练.
- MoE 模型训练.
- 长上下文模型训练.
- NVIDIA GPU 集群上的高性能训练.

---

## 2. 核心并行能力

### 2.1 Tensor Parallelism

[Tensor Parallelism](../Distributed_Training/Main.md)(TP) 是 Megatron 最经典的能力之一.

它的思路是:

**把 Transformer 层内部的大矩阵乘法切分到多个 GPU 上并行计算.**

例如在 MLP 中:

- 第一层线性层可以按列切分.
- 第二层线性层可以按行切分.
- 多个 GPU 各自计算一部分中间结果.
- 最后通过通信聚合得到完整输出.

在 [Attention](../basic/Attention.md) 中:

- 可以把不同 attention head 分到不同 GPU.
- 每个 GPU 只计算一部分 Q, K, V 和 attention output.

TP 的特点:

1. **不依赖 batch size**: 即使 batch 很小, 也能通过切分模型宽度来并行.
2. **通信频繁**: 每个 Transformer block 内部都需要通信.
3. **适合节点内部**: 通常要求 NVLink / NVSwitch 等高带宽连接.

---

### 2.2 Pipeline Parallelism

[Pipeline Parallelism](../Distributed_Training/Main.md)(PP) 按模型深度切分, 即把不同 [Transformer](../basic/Transformer%20架构.md) 层放到不同 GPU 或节点上.

其核心思想是:

**第 1 组 GPU 负责前几层, 第 2 组 GPU 负责中间层, 第 3 组 GPU 负责后几层.**

为了减少流水线气泡, 通常会把一个 batch 切分为多个 micro-batch, 让不同 pipeline stage 同时处理不同 micro-batch.

PP 的特点:

1. **适合模型层数很多的情况**.
2. **可以跨节点使用**.
3. **存在 pipeline bubble**.
4. **micro-batch 数量会影响利用率**.

---

### 2.3 Data Parallelism

[Data Parallelism](../Distributed_Training/Main.md)(DP) 是最直观的并行方式.

它的思路是:

- 每个数据并行组持有一份模型副本.
- 不同 GPU 处理不同数据.
- 反向传播后同步梯度.

在 Megatron 中, DP 通常和 TP, PP 组合使用.

例如:

$$
\text{Total GPUs} = TP \times PP \times DP
$$

如果总共有 64 张 GPU, 可以设置:

$$
TP = 8,\quad PP = 4,\quad DP = 2
$$

这表示每个模型副本需要 $8 \times 4 = 32$ 张 GPU, 一共有 2 个数据并行副本.

---

### 2.4 Sequence Parallelism

[Sequence Parallelism](../Distributed_Training/Main.md)(SP) 主要用于进一步降低激活值显存.

在 TP 中, 有些操作不能按 hidden dimension 被完全切分, 例如 [LayerNorm](../basic/Normalization.md), Dropout 等逐 token 操作. 这些操作会导致每个 TP rank 保留一份完整 sequence activation.

SP 的思路是:

**把 sequence dimension 也切开, 让不同 GPU 只保存一部分 token 的激活值.**

它通常和 TP 配合使用, 用来减少训练长序列时的 activation memory.

---

### 2.5 Context Parallelism

Context Parallelism(CP) 面向长上下文训练.

当 sequence length 很长时, 单个 GPU 保存完整上下文的 attention 相关激活会非常昂贵. CP 的思路是:

**沿 context / sequence 维度切分长上下文, 让不同 GPU 负责不同片段.**

CP 更适合:

- 长上下文预训练.
- 长文档建模.
- 大 batch + long sequence 的训练场景.

它和 SP 的区别可以简单理解为:

- **SP** 更偏减少 Transformer 层中部分 activation 的冗余.
- **CP** 更偏完整支持长上下文维度的并行计算.

---

### 2.6 Expert Parallelism

Expert Parallelism(EP) 主要用于 [MoE](../basic/MOE.md) 模型.

[MoE](../basic/MOE.md) 中有多个 expert, 每个 token 只路由到其中一部分 expert. EP 的思路是:

**把不同 expert 放到不同 GPU 上, token 根据 router 分发到对应 expert 计算.**

EP 的关键问题包括:

- token dispatch.
- load balance.
- expert capacity.
- all-to-all 通信.
- router loss.

EP 通常会和 TP, PP, DP 一起组成更复杂的并行策略.

---

## 3. 核心功能

### 3.1 Transformer Engine 和混合精度

Megatron 通常和 NVIDIA Transformer Engine 配合使用, 支持 BF16, FP16, FP8 等混合精度训练.

混合精度的目的:

1. 降低显存.
2. 提高 Tensor Core 利用率.
3. 提升训练吞吐.

在超大规模训练中, 混合精度基本是默认配置.

---

### 3.2 Distributed Optimizer

Megatron 支持分布式优化器, 用于减少 [optimizer](../basic/Optimizer.md) state 的冗余显存.

普通数据并行中, 每个 DP rank 都保存完整 optimizer state. 对 Adam 来说, optimizer state 往往比模型参数还大.

分布式优化器的思路和 [ZeRO](../Distributed_Training/ZeRO.md) 类似:

**在数据并行维度上切分 optimizer state 和梯度, 减少每张 GPU 的显存压力.**

它和 [ZeRO](../Distributed_Training/ZeRO.md) 的思路相近, 都是在数据并行维度上减少冗余.

---

### 3.3 Checkpoint

Megatron 的 checkpoint 通常和并行策略强相关.

如果训练使用了:

- TP.
- PP.
- EP.
- Distributed optimizer.

那么 checkpoint 也会被切成多个 shard.

因此在 Megatron 中, checkpoint 管理非常重要:

1. 保存时要记录并行配置.
2. 加载时要匹配或转换并行配置.
3. 推理或微调时可能需要 merge 或 reshard.

---

### 3.4 数据处理

Megatron 也提供大规模预训练数据处理能力, 包括:

- 文本 tokenization.
- indexed dataset.
- shuffle index.
- 多进程数据读取.
- 训练时高吞吐 data loader.

对于预训练, 数据管道和模型并行同样重要. 如果数据读取跟不上, GPU 即使并行策略正确也会空等.

---

## 4. 使用流程

典型 Megatron 训练流程是:

1. 准备 tokenizer 和预训练数据.
2. 将数据处理成 Megatron 支持的 indexed dataset.
3. 选择模型结构, 例如 GPT, T5, BERT, MoE.
4. 配置并行策略, 例如 TP, PP, DP, SP, CP.
5. 配置混合精度和优化器.
6. 启动分布式训练.
7. 定期保存分片 checkpoint.
8. 根据需要转换 checkpoint 用于推理或继续训练.

---

## 5. 特点

1. **极强的分布式训练能力**: Megatron 的核心优势是大规模 Transformer 并行训练.

2. **多维并行体系完整**: TP, PP, DP, SP, CP, EP 都可以组合使用.

3. **适合 NVIDIA GPU 集群**: 充分利用 NVLink, NVSwitch, InfiniBand, Tensor Core 等硬件能力.

4. **适合预训练**: 相比普通微调框架, Megatron 更适合从头训练大模型.

5. **性能优先**: 设计目标是吞吐和可扩展性, 而不是最简单的 API.

---

## 6. 局限性

1. **上手难度高**: 并行配置, 数据格式, checkpoint 管理都比较复杂.

2. **硬件依赖强**: Megatron 最适合 NVIDIA 高速互联 GPU 集群, 在普通单机环境中优势不明显.

3. **不适合快速小实验**: 如果只是 LoRA 微调或 DPO 小实验, TRL / Transformers / PEFT 更方便.

4. **checkpoint 转换复杂**: TP, PP, EP 不同设置下的权重切分方式不同, 跨框架转换需要额外工具.

5. **工程调参成本高**: micro-batch, global batch, TP size, PP size, sequence length, activation checkpoint 都会影响吞吐和稳定性.

---

## 7. Megatron 和 DeepSpeed 的区别

|对比项|Megatron|[DeepSpeed](./DeepSpeed.md)|
|---|---|---|
|核心定位|超大 Transformer 并行训练框架|训练优化和显存优化框架|
|最强能力|[TP, PP](../Distributed_Training/Main.md), CP, EP 等模型并行|[ZeRO](../Distributed_Training/ZeRO.md), offload, optimizer state sharding|
|典型场景|大模型预训练|大模型训练加速, 微调, ZeRO 优化|
|硬件倾向|NVIDIA GPU 集群|更通用, 可结合多种训练代码|
|API 风格|偏底层和训练脚本|配置式集成更强|
|常见组合|Megatron + DeepSpeed|DeepSpeed + HF Trainer / Megatron|

一句话总结:

**Megatron 是面向超大模型预训练的高性能并行框架, 核心价值在于把 Transformer 模型本身高效切分到大量 GPU 上.**
