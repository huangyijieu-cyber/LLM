# Projector

Projector 也叫 Connector, Adapter 或 Multimodal Projector. 它是 VLM 中连接视觉编码器和 LLM 的桥.

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
- $d_v$: vision encoder hidden size.
- $N'_v$: 投影后的 visual token 数量.
- $d_{\text{LLM}}$: LLM hidden size.

---

## 1. 为什么需要 Projector

Vision Encoder 和 LLM 的 embedding 空间通常不同.

例如:

- CLIP vision hidden size 可能是 1024.
- LLaMA hidden size 可能是 4096.

直接把视觉特征送入 LLM 会维度不匹配, 语义空间也不匹配.

Projector 需要解决:

1. **维度对齐**: $d_v \rightarrow d_{\text{LLM}}$.
2. **语义对齐**: 把视觉特征翻译成语言模型可理解的表示.
3. **token 压缩**: 减少 visual token 数量.
4. **训练稳定性**: 减少直接微调 LLM 的难度.

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

适合早期对齐或小规模实验.

---

## 3. MLP Projector

MLP Projector 是 LLaVA 路线常用方案.

常见形式:

$$
H_v = W_2 \cdot \sigma(W_1 X_v)
$$

其中 $\sigma$ 可以是 GELU 或其他激活函数.

### 3.1 特点

1. **比 Linear 表达能力更强**.
2. **实现仍然简单**.
3. **工程上非常常见**.
4. **不主动压缩 token 数量**.

### 3.2 适合场景

- LLaVA 类架构.
- 图像问答.
- 通用 VLM instruction tuning.

---

## 4. Q-Former

Q-Former 来自 [BLIP-2](https://arxiv.org/abs/2301.12597).

它使用一组可学习 query tokens 从视觉特征中读取信息:

$$
Q = \{q_1,q_2,\dots,q_M\}
$$

通过 cross-attention:

$$
Q' = \text{CrossAttention}(Q, X_v)
$$

最后将 $Q'$ 映射给 LLM.

### 4.1 特点

1. **可以压缩视觉 token**: 输出 query token 数 $M$ 通常远小于 patch token 数.
2. **参数效率高**.
3. **适合冻结 vision encoder 和 LLM**.
4. **结构比 MLP 复杂**.

### 4.2 直观理解

MLP Projector 是把所有视觉 patch 翻译给 LLM.

Q-Former 是让一组 query 主动问视觉编码器:

**哪些视觉信息对语言任务有用?**

---

## 5. Perceiver Resampler

Perceiver Resampler 在 Flamingo 中使用.

它也使用一组 latent tokens 从视觉特征中提取固定长度表示.

形式上:

$$
Z = \text{Resampler}(X_v)
$$

其中 $Z$ 是固定数量的 visual latents.

### 5.1 特点

1. **适合多图输入**.
2. **可以控制视觉 token 数量**.
3. **适合 interleaved image-text 输入**.
4. **训练和实现成本较高**.

---

## 6. Cross-Attention Connector

Cross-Attention Connector 不一定把视觉 token 直接拼到文本 token 前面, 而是在 LLM 的某些层中让文本 hidden states 通过 cross-attention 读取视觉特征.

类似:

$$
H_t' = \text{CrossAttention}(H_t, H_v)
$$

### 6.1 特点

1. **融合能力强**.
2. **可以在多层注入视觉信息**.
3. **适合复杂多模态融合**.
4. **会改变 LLM 结构, 工程复杂度更高**.

Flamingo 的 gated cross-attention 就是这一类思想.

---

## 7. Visual Token 的插入方式

### 7.1 Prefix 插入

最常见方式:

$$
[\text{visual tokens}; \text{text tokens}]
$$

优点:

- 简单.
- 不改 LLM 结构.
- 适合 causal LLM.

缺点:

- visual tokens 占用上下文长度.
- visual tokens 越多, [KV Cache](../Inference/KV_Cache.md) 越大.

---

### 7.2 Placeholder 替换

Prompt 中放一个 `<image>` token:

```text
USER: <image>
What is in this image?
```

实际输入时, `<image>` 会被展开成一串 visual tokens.

这是 LLaVA, Qwen-VL 等模型常见做法.

---

### 7.3 Interleaved 插入

多图多文本场景中, 输入可能是:

```text
Image 1 + text 1 + Image 2 + text 2 + question
```

这种方式适合:

- 多图对比.
- 教程截图理解.
- Agent 操作记录.
- 视频帧序列.

---

## 8. Projector 训练策略

### 8.1 只训练 Projector

早期对齐阶段常用:

- 冻结 Vision Encoder.
- 冻结 LLM.
- 只训练 Projector.

优点:

1. 稳定.
2. 显存小.
3. 不破坏 LLM 语言能力.

缺点:

1. 上限有限.
2. 无法充分适配复杂任务.

---

### 8.2 训练 Projector + LLM LoRA

常见于多模态 instruction tuning.

做法:

- Vision Encoder 冻结.
- Projector 全量训练.
- LLM 用 [LoRA](../Finetune/PEFT.md) 训练.

优点:

1. 成本较低.
2. 可以适配多模态指令.
3. 不需要全参数训练.

---

### 8.3 全参数训练

高性能 VLM 可能会训练:

- Vision Encoder.
- Projector.
- LLM.

优点是上限高, 缺点是成本大, 容易破坏已有能力, 需要更谨慎的数据和超参数.

---

## 9. Projector 对比

|Projector|代表模型|是否压缩 token|优点|局限|
|---|---|---|---|---|
|Linear|早期 VLM|否|简单, 稳定|表达能力弱|
|MLP|LLaVA|否|主流, 易实现|visual token 多|
|Q-Former|BLIP-2|是|参数效率高, 可压缩|结构复杂|
|Perceiver Resampler|Flamingo|是|适合多图和交错输入|训练复杂|
|Cross-Attention|Flamingo 类|不一定|融合能力强|改动 LLM 结构|

一句话总结:

**Projector 是 VLM 的翻译器, 它把视觉特征翻译成 LLM 能读懂的 visual tokens. 当前工程最常见的是 MLP Projector, 更复杂的模型会使用 Q-Former, Perceiver 或 Cross-Attention 做 token 压缩和深度融合.**
