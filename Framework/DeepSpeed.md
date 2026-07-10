# DeepSpeed

## 1. 核心概念

DeepSpeed 是 Microsoft 开源的大规模深度学习训练优化框架. 它常和 [TRL](./TRL.md), [Megatron](./Megatron.md), Hugging Face Trainer 等训练框架结合使用. 它的核心目标是:

**让大模型训练更省显存, 更高吞吐, 更容易扩展到多 GPU 和多节点.**

DeepSpeed 不是一个特定模型结构, 也不是一个特定算法. 它更像是一个训练系统加速层, 可以和 PyTorch, Hugging Face Trainer, Accelerate, Megatron 等框架结合使用.

DeepSpeed 最核心的技术是 **[ZeRO](../Distributed_Training/ZeRO.md)(Zero Redundancy Optimizer)**. ZeRO 通过切分 [optimizer](../basic/Optimizer.md) state, gradient 和 parameter, 减少 [数据并行](../Distributed_Training/Main.md) 中的显存冗余.

---

## 2. 核心模块

### 2.1 [ZeRO](../Distributed_Training/ZeRO.md)

在普通数据并行中, 每张 GPU 都保存完整的:

- Model parameters.
- Gradients.
- Optimizer states.

这会造成大量冗余. DeepSpeed ZeRO 的思路是:

**在数据并行 rank 之间切分这些状态, 让每张 GPU 只保存其中一部分.**

ZeRO 分为三个阶段:

|阶段|切分内容|显存节省|
|---|---|---|
|ZeRO-1|Optimizer states|较低|
|ZeRO-2|Optimizer states + gradients|中等|
|ZeRO-3|Optimizer states + gradients + parameters|最高|

简单理解:

- **ZeRO-1**: 只切 Adam 的一阶矩和二阶矩等优化器状态.
- **ZeRO-2**: 进一步切分梯度.
- **ZeRO-3**: 连模型参数也切分, 计算时再临时 all-gather 当前层参数.

ZeRO-3 和 [FSDP](../Distributed_Training/ZeRO.md) 的思想很接近, 都是参数分片训练.

---

### 2.2 ZeRO-Offload

ZeRO-Offload 的目标是进一步节省 GPU 显存.

它会把部分状态放到 CPU 内存中, 例如:

- Optimizer states.
- Gradients.
- Parameters.

这样可以训练更大的模型, 但代价是:

1. CPU-GPU 数据传输增加.
2. 训练吞吐可能下降.
3. 对 PCIe / NVLink 带宽更敏感.

因此 Offload 更适合 **显存不足但可以接受速度变慢** 的场景.

---

### 2.3 ZeRO-Infinity

ZeRO-Infinity 是 Offload 思路的进一步扩展.

它可以把模型状态扩展到:

- GPU memory.
- CPU memory.
- NVMe storage.

目标是训练远超单机 GPU 显存容量的大模型.

但需要注意:

**NVMe 访问速度远低于 GPU 显存, 因此 ZeRO-Infinity 更偏向解决能不能训练的问题, 不一定是最高吞吐方案.**

---

### 2.4 Activation Checkpointing

Activation Checkpointing 也叫 gradient checkpointing, 在 [分布式训练](../Distributed_Training/Main.md) 中常用于降低 activation memory.

普通训练中, 前向传播会保存大量中间激活值用于反向传播. Activation Checkpointing 的思路是:

**前向时只保存少量关键节点, 反向时重新计算中间激活值.**

它用额外计算换显存, 特别适合深层 Transformer.

优点:

- 显著降低 activation memory.
- 可以支持更大 batch size 或更长 sequence length.

缺点:

- 反向传播变慢.
- 计算量增加.

---

### 2.5 Mixed Precision 和优化器

DeepSpeed 支持混合精度训练, 例如:

- FP16.
- BF16.
- FP32 master weights.

它也提供一些高性能优化器和 fused kernel, 用于减少 Python overhead 和提升 GPU 利用率.

常见优化器包括:

- FusedAdam.
- CPUAdam.
- OneBitAdam.

---

### 2.6 DeepSpeed-Inference

DeepSpeed 也提供推理优化能力. 但如果重点是 LLM serving, 通常也会和 [vLLM](./vllm.md), [SGLang](./SGLang.md) 这类专门推理框架对比.

DeepSpeed-Inference 关注:

- Tensor parallel inference.
- Kernel injection.
- Quantization.
- 大模型推理显存优化.

不过在 LLM serving 场景中, 现在也经常会使用 [vLLM](./vllm.md) 或 [SGLang](./SGLang.md) 作为专门的推理引擎.

---

## 3. 配置方式

DeepSpeed 通常通过 JSON 配置文件使用.

一个典型配置会包含:

- train batch size.
- micro batch size.
- gradient accumulation steps.
- fp16 / bf16.
- ZeRO stage.
- offload 选项.
- optimizer.
- scheduler.
- activation checkpointing.

DeepSpeed 的重要特点是:

**训练代码不一定要大改, 很多能力通过配置文件开启.**

例如在 Hugging Face Trainer 中, 只需要指定:

```text
--deepspeed ds_config.json
```

就可以使用 DeepSpeed 配置.

---

## 4. 使用流程

典型流程如下:

1. 编写 PyTorch 或 Transformers 训练代码.
2. 准备 DeepSpeed JSON 配置.
3. 设置 ZeRO stage 和 batch 相关参数.
4. 选择 FP16 或 BF16.
5. 根据显存情况决定是否开启 offload.
6. 使用 `deepspeed` 或 Accelerate 启动训练.
7. 保存 checkpoint.
8. 根据需要进行 checkpoint merge 或转换.

---

## 5. 特点

1. **ZeRO 是核心优势**: DeepSpeed 最重要的能力是通过 ZeRO 减少数据并行冗余.

2. **集成方便**: 可以和 Hugging Face Trainer, Accelerate, Megatron 等生态结合.

3. **适合显存优化**: 当模型放不下或 batch 太小时, ZeRO, offload, checkpointing 都很有用.

4. **配置式使用**: 很多功能可以通过 JSON 配置打开, 不需要完全重写训练逻辑.

5. **训练和推理都有覆盖**: 主要强项是训练, 同时也提供推理优化模块.

---

## 6. 局限性

1. **配置复杂**: ZeRO stage, offload, batch size, gradient accumulation 之间会相互影响.

2. **调试难度高**: 分布式训练报错可能来自通信, 显存, 数据, optimizer, checkpoint 等多个层面.

3. **Offload 会降低速度**: 把状态放到 CPU 或 NVMe 能省显存, 但会带来传输瓶颈.

4. **ZeRO-3 通信开销高**: 参数被切分后, 每层计算前需要 all-gather 参数, 通信压力比普通 DDP 更大.

5. **checkpoint 管理复杂**: ZeRO checkpoint 通常是分片保存, 推理或转换时需要额外处理.

---

## 7. DeepSpeed 和 FSDP / Megatron 的关系

|对比项|DeepSpeed|[FSDP](../Distributed_Training/ZeRO.md)|[Megatron](./Megatron.md)|
|---|---|---|---|
|核心能力|[ZeRO](../Distributed_Training/ZeRO.md), offload, 训练优化|PyTorch 原生参数分片|[Transformer](../basic/Transformer%20架构.md) 模型并行训练|
|主要解决|显存冗余和训练扩展|参数, 梯度, optimizer state 分片|TP, PP, CP, EP 多维并行|
|使用方式|JSON 配置 + engine|PyTorch API|训练脚本和并行配置|
|适合场景|大模型微调, 训练加速|PyTorch 原生分片训练|超大规模预训练|
|学习成本|中等偏高|中等|高|

一句话总结:

**DeepSpeed 是大模型训练中的显存优化和训练加速工具箱, 其中 ZeRO 是最核心的能力.**
