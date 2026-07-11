# Projector

Projector 也叫 Connector, Adapter 或 Multimodal Projector. 它是 VLM 中连接 [Vision Encoder](./Vision_Encoder.md) 和 LLM 的桥.

它的核心任务是:

**把 Vision Encoder 输出的视觉特征映射到 LLM 能理解的 embedding space.**

形式上:

$$
X_v \in \mathbb{R}^{N_v \times d_v}
$$

经过 Projector:

$$
H_v = f_{\text{proj}}(X_v) \in \mathbb{R}^{N'_v \times d_{\text{LLM}}}
$$

其中:

- $N_v$: 原始 visual token 数量.
- $d_v$: Vision Encoder hidden size.
- $N'_v$: 投影后的 visual token 数量.
- $d_{\text{LLM}}$: LLM hidden size.

Projector 不只是维度转换模块. 它还会影响 visual token 数量, 视觉信息压缩程度, 训练稳定性和推理成本.

---

## 1. 为什么需要 Projector

Vision Encoder 和 LLM 的 embedding 空间通常不同.

例如:

- CLIP vision hidden size 可能是 1024.
- LLaMA hidden size 可能是 4096.

直接把视觉特征送入 LLM 会有两个问题:

1. 维度不匹配.
2. 语义空间不匹配.

Projector 需要解决:

1. **维度对齐**: $d_v \rightarrow d_{\text{LLM}}$.
2. **语义对齐**: 把视觉特征翻译成语言模型可理解的表示.
3. **token 压缩**: 减少 visual token 数量.
4. **训练稳定性**: 减少直接微调 LLM 的难度.
5. **推理成本控制**: 降低 high-resolution, multi-image, video 场景下的 token 压力.

---

## 2. Linear Projector

Linear Projector 是最简单的方式:

$$
H_v = X_v W
$$

其中:

$$
W \in \mathbb{R}^{d_v \times d_{\text{LLM}}}
$$

### 2.1 特点

1. **实现简单**.
2. **参数少**.
3. **训练稳定**.
4. **表达能力有限**.
5. **不主动压缩 token 数量**.

Linear Projector 适合早期对齐或小规模实验, 但现代 VLM 更常使用 MLP Projector 或更复杂的 connector.

---

## 3. MLP Projector

MLP Projector 是 LLaVA 路线常用方案.

常见形式:

$$
H_v = W_2 \cdot \sigma(W_1X_v)
$$

其中 $\sigma$ 可以是 GELU 或其他激活函数.

### 3.1 特点

1. **比 Linear 表达能力更强**.
2. **实现仍然简单**.
3. **不改变 LLM 主体结构**.
4. **工程上非常常见**.
5. **通常不主动压缩 token 数量**.

### 3.2 适合场景

1. LLaVA 类架构.
2. 通用图像问答.
3. 多模态 instruction tuning.
4. 作为 VLM baseline.

### 3.3 局限

MLP Projector 会把 patch features 基本原样映射给 LLM. 当输入是高分辨率, 多图或视频时, visual token 数量可能很大.

因此 LLaVA-NeXT / OneVision 这类模型需要配合 AnyRes, tile 策略和推理优化来控制成本, 详见 [Architecture](./Architecture.md).

---

## 4. Q-Former

Q-Former 来自 [BLIP-2](https://arxiv.org/abs/2301.12597).

它使用一组 learnable query tokens 从视觉特征中读取信息:

$$
Q = \{q_1,q_2,\dots,q_M\}
$$

通过 cross-attention:

$$
Q' = \text{CrossAttention}(Q,X_v)
$$

最后将 $Q'$ 映射给 LLM.

---

### 4.1 具体作用

假设原始图像有很多 patch tokens:

$$
X_v \in \mathbb{R}^{N_v \times d_v}
$$

Q-Former 输出固定数量的 query features:

$$
Q' \in \mathbb{R}^{M \times d_q}
$$

通常:

$$
M \ll N_v
$$

因此 Q-Former 本质上是一种 **query-based visual token compression**.

### 4.2 特点

1. **可以压缩视觉 token**.
2. **参数效率高**.
3. **适合冻结 Vision Encoder 和 LLM**.
4. **结构比 MLP 复杂**.
5. **可能损失 OCR 和 dense perception 细节**.

### 4.3 和 MLP Projector 的区别

|对比项|MLP Projector|Q-Former|
|---|---|---|
|代表模型|LLaVA|BLIP-2|
|核心方式|逐 patch 映射|query tokens 读取视觉信息|
|是否压缩 token|通常不压缩|明显压缩|
|优点|简单, 易复现|token 少, 参数效率高|
|局限|token 成本高|可能丢失细节|

---

## 5. Perceiver Resampler

Perceiver Resampler 在 [Flamingo](https://arxiv.org/abs/2204.14198) 中使用.

它使用一组 latent tokens 从视觉特征中提取固定长度表示:

$$
Z = \text{Resampler}(X_v)
$$

其中 $Z$ 是固定数量的 visual latents.

### 5.1 特点

1. **适合多图输入**.
2. **可以控制视觉 token 数量**.
3. **适合 interleaved image-text 输入**.
4. **训练和实现成本较高**.
5. **通常和 cross-attention 注入配合使用**.

### 5.2 适用场景

1. 多图上下文.
2. 图文交错输入.
3. multimodal few-shot learning.
4. 视频帧或多页文档压缩.

---

## 6. Patch Merger

Patch Merger 常见于 Qwen-VL 系列等高分辨率 VLM.

Dynamic Resolution 会产生不同数量的 visual tokens:

$$
N_v = f(H,W)
$$

如果直接把所有 tokens 送入 LLM, prefill 和 [KV Cache](../Inference/KV_Cache.md) 成本会很高.

Patch Merger 的目标是:

$$
X_v \in \mathbb{R}^{N_v \times d_v}
\rightarrow
H_v \in \mathbb{R}^{N'_v \times d_{\text{LLM}}}
$$

其中:

$$
N'_v < N_v
$$

### 6.1 特点

1. **控制高分辨率输入的 token 数**.
2. **保留局部空间信息**.
3. **适合 OCR, document, chart 等场景**.
4. **需要在细节保留和压缩率之间权衡**.

Patch Merger 和 Dynamic Resolution 通常一起出现, 详见 [Architecture](./Architecture.md).

---

## 7. Cross-Attention Connector

Cross-Attention Connector 不一定把视觉 token 直接拼到文本 token 前面, 而是在 LLM 的某些层中让文本 hidden states 通过 cross-attention 读取视觉特征.

形式:

$$
H_t' = \text{CrossAttention}(H_t,H_v)
$$

### 7.1 特点

1. **融合能力强**.
2. **可以在多层注入视觉信息**.
3. **适合复杂多模态融合**.
4. **会改变 LLM 结构**.
5. **工程复杂度更高**.

Flamingo 的 gated cross-attention 就是这一类思想.

---

## 8. Visual Token 的插入方式

### 8.1 Prefix 插入

最常见方式:

$$
[\text{visual tokens}; \text{text tokens}]
$$

优点:

1. 简单.
2. 不改 LLM 结构.
3. 适合 causal LLM.

缺点:

1. visual tokens 占用上下文长度.
2. visual tokens 越多, [KV Cache](../Inference/KV_Cache.md) 越大.

---

### 8.2 Placeholder 替换

Prompt 中放一个 `<image>` token:

```text
USER: <image>
What is in this image?
```

实际输入时, `<image>` 会被展开成一串 visual tokens.

这是 LLaVA, Qwen-VL 等模型常见做法.

---

### 8.3 Interleaved 插入

多图多文本场景中, 输入可能是:

```text
<image_1> text_1 <image_2> text_2 question
```

这种方式适合:

1. 多图对比.
2. 教程截图理解.
3. Agent 操作记录.
4. 视频帧序列.
5. 多页文档理解.

---

## 9. Projector 训练策略

### 9.1 只训练 Projector

早期对齐阶段常用:

1. 冻结 Vision Encoder.
2. 冻结 LLM.
3. 只训练 Projector.

优点:

1. 稳定.
2. 显存小.
3. 不破坏 LLM 语言能力.

缺点:

1. 上限有限.
2. 无法充分适配复杂任务.

---

### 9.2 训练 Projector + LLM LoRA

常见于多模态 instruction tuning.

做法:

1. Vision Encoder 冻结.
2. Projector 全量训练.
3. LLM 使用 [LoRA](../Finetune/PEFT.md) 训练.

优点:

1. 成本较低.
2. 可以适配多模态指令.
3. 不需要全参数训练.

---

### 9.3 全参数训练

高性能 VLM 可能会训练:

1. Vision Encoder.
2. Projector.
3. LLM.

优点是上限高. 缺点是成本大, 容易破坏已有能力, 需要更谨慎的数据和超参数.

大规模训练通常需要 [DeepSpeed](../Framework/DeepSpeed.md), [Megatron](../Framework/Megatron.md) 等框架支持.

---

## 10. Projector 对比

|Projector|代表模型|是否压缩 token|优点|局限|
|---|---|---|---|---|
|Linear|早期 VLM|否|简单, 稳定|表达能力弱|
|MLP|LLaVA|通常否|主流, 易实现|visual token 多|
|Q-Former|BLIP-2|是|参数效率高, 可压缩|结构复杂, 可能丢细节|
|Perceiver Resampler|Flamingo|是|适合多图和交错输入|训练复杂|
|Patch Merger|Qwen-VL|是|适合高分辨率输入|压缩率和细节需要权衡|
|Cross-Attention|Flamingo 类|不一定|融合能力强|改动 LLM 结构|

---

## 11. 总结

Projector 是 VLM 的视觉到语言接口.

1. MLP Projector 简单, 是 LLaVA 类模型的主流选择.
2. Q-Former 通过 query tokens 压缩视觉信息, 适合冻结模型连接.
3. Perceiver Resampler 适合多图和 interleaved image-text.
4. Patch Merger 适合 dynamic resolution 和高分辨率输入.
5. Cross-Attention Connector 融合能力强, 但工程复杂度更高.

Projector 的选择需要同时考虑模型效果, visual token 数量, 推理成本和训练稳定性.
