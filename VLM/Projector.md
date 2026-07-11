# Projector

Projector 也叫 Connector, Adapter 或 Multimodal Projector, 位于 [Vision Encoder](./Vision_Encoder.md) 和 LLM 之间. Vision Encoder 输出的是视觉特征空间, LLM 接收的是语言 hidden space, 因而两者不能简单地直接连接:

$$
X_v \in \mathbb{R}^{N_v \times d_v}
\xrightarrow{f_{\mathrm{proj}}}
H_v \in \mathbb{R}^{N'_v \times d_{\mathrm{LLM}}}
$$

**Projector 不只负责 hidden size 映射, 还可能决定 visual token 的压缩程度.** 因此判断一个 Connector 时需要同时看两个量: hidden size 是否完成映射, token length 是否发生变化.

---

## 1. MLP Projector

Vision Encoder 和 LLM 的 hidden size 通常不同, 例如视觉特征可能是 1024 维, LLM hidden size 可能是 4096 维. Linear Projector 是最简单的连接方式:

$$
H_v = X_vW, \qquad W \in \mathbb{R}^{d_v \times d_{\mathrm{LLM}}}
$$

它参数少且训练稳定, 但只能做线性空间变换, 也不主动压缩 token. 更常见的 MLP Projector 增加一层非线性:

$$
H_v = W_2\sigma(W_1X_v)
$$

其中 $\sigma$ 通常为 GELU. **MLP Projector 的优势是表达能力高于 Linear, 同时不改变 LLM 主体结构.** LLaVA 路线可以把每个 patch feature 直接映射成 visual token, 对通用 VQA 和多模态指令微调通常已经是很强的 baseline.

MLP 的关键局限是通常保持 $N'_v = N_v$. 它完成了 feature projection, 却没有解决 token compression. 固定低分辨率图像中问题不大, 但 AnyRes tile, 多图或视频会产生大量 patch tokens, 这些 token 会完整进入 LLM, 增加 prefill 和 [KV Cache](../Inference/KV_Cache.md). 因此 MLP 路线常需要在图像预处理阶段限制 tile 数, 或额外加入 pooling / merging, 具体见 [Architecture](./Architecture.md).

---

## 2. Query-Based Connector

### 2.1 Q-Former

[BLIP-2](https://arxiv.org/abs/2301.12597) 使用 Q-Former 连接冻结 Vision Encoder 和冻结 LLM. 它维护 $M$ 个 learnable query tokens:

$$
Q = \{q_1,q_2,\dots,q_M\}
$$

这些 query 通过 cross-attention 从全部视觉特征中读取信息:

$$
Q' = \mathrm{CrossAttention}(Q,X_v)
$$

无论原图产生多少 patch, 输出长度都固定为 $M$, 通常满足 $M \ll N_v$. **Q-Former 不是逐 patch 映射, 而是用少量 learnable queries 主动读取并压缩视觉信息.** 它适合冻结两侧大模型并低成本训练中间接口.

固定 query 数同时构成信息瓶颈. 对场景语义和 Caption, 少量视觉摘要可能足够; 对 OCR, 文档和 dense grounding, 每个局部区域都可能包含答案, 过度压缩容易丢失小字和空间细节.

### 2.2 Perceiver Resampler

[Flamingo](https://arxiv.org/abs/2204.14198) 的 Perceiver Resampler 同样使用固定数量 latent tokens 读取可变长度视觉特征:

$$
X_v \rightarrow Z, \qquad Z \in \mathbb{R}^{M \times d}
$$

它更强调把不同图片或视频帧统一成固定长度 visual latents, 便于处理 interleaved image-text 和 multimodal few-shot context. Resampler 通常与 LLM 中间层的 gated cross-attention 配合, 而不是简单地把所有 visual latents 放到文本前缀.

**Q-Former 更强调连接冻结模型, Perceiver Resampler 更强调多图和 interleaved context.** 两者都属于 learnable query / latent compression, 用可控 token 数换取推理效率, 也都需要接受压缩造成的信息损失.

---

## 3. Spatial Token Merger

Patch Merger 常用于 Qwen-VL 等 Dynamic Resolution 架构. Dynamic Resolution 让不同尺寸图片产生不同数量的视觉 patch:

$$
N_v = f(H,W)
$$

Patch Merger 将相邻 patch 的特征组合并投影, 使输出满足 $N'_v < N_v$. **与固定 query 的全局摘要相比, Patch Merger 按局部邻域压缩, 更容易保留二维结构.** 这对 OCR, Document QA 和 Chart QA 很重要, 因为文字内容与所在区域都需要进入 LLM.

压缩率越高, LLM 成本越低, 但小文字和局部目标越容易被合并掉. 因此 Patch Merger 需要和 Vision Encoder 分辨率一起设计: Dynamic Resolution 负责获取细节, Merger 负责把这些细节压缩到可接受的 token budget. 只提高分辨率而没有 token 控制, 会把成本直接转移给 LLM; 只提高压缩率, 又会抵消高分辨率带来的收益.

---

## 4. 视觉信息如何进入 LLM

### 4.1 Input Token Injection

LLaVA 类模型通常在 prompt 中使用 `<image>` placeholder. 实际 forward 时, 该位置会被替换成一组 visual embeddings:

```text
USER: <image>
What is shown in this image?
```

对应的输入可以抽象为:

$$
[\text{visual tokens};\text{text tokens}]
$$

**Input Token Injection 的优点是不修改 LLM block, 代价是 visual tokens 与普通文本一样占用上下文和 KV Cache.** 多图场景可以按照 `<image_1> text_1 <image_2> text_2` 交错插入, 使图像与相关文字保持局部对应.

### 4.2 Cross-Attention Injection

Flamingo 类模型在 LLM 的部分中间层加入 cross-attention:

$$
H_t' = \mathrm{CrossAttention}(H_t,H_v)
$$

文本 hidden states 在生成过程中读取独立保存的视觉特征. 这种结构可以多层注入视觉信息, 对多图和复杂融合更灵活, 但需要修改 LLM block, checkpoint 和 serving 实现. Gated Cross-Attention 还会使用可学习 gate 控制视觉残差强度, 避免训练初期突然破坏预训练语言表示.

|注入方式|代表路线|优点|代价|
|---|---|---|---|
|Visual Token Prefix / Placeholder|LLaVA, Qwen-VL, InternVL|结构简单, 复用原生 LLM|占用上下文和 KV Cache|
|Interleaved Tokens|OneVision 等多图模型|图像和文字对应关系清晰|序列更长, 顺序管理复杂|
|Cross-Attention|Flamingo 类|多层融合, 视觉 memory 独立|需要修改 LLM 和 serving|

---

## 5. Projector 训练

Projector Alignment 阶段通常冻结 Vision Encoder 和 LLM, 只用 image-caption 或简单 VQA 数据训练 Connector. **先对齐 Projector 的目的, 是避免陌生 visual embeddings 在训练初期直接扰动 LLM.** 该阶段成本低, 但只靠简单 Caption 学到的接口通常不足以支持复杂问答.

Multimodal Instruction Tuning 时, 常见做法是继续全量训练 Projector, 同时对 LLM 使用 [LoRA](../Finetune/PEFT.md) 或全参数微调. Vision Encoder 可以保持冻结, 也可以解冻后部层以适配 OCR 和医疗等领域. 高性能模型可能联合训练三个模块, 上限更高, 但需要更谨慎的 learning rate, 数据配比和分布式训练, 可参考 [DeepSpeed](../Framework/DeepSpeed.md) 与 [Megatron](../Framework/Megatron.md).

训练策略的关键不是简单地决定哪些参数可训练, 而是避免各模块学习速度失衡. Projector 初始化时离 LLM embedding space 较远, 通常需要更快完成基础对齐; 已经预训练好的 Vision Encoder 和 LLM 则使用较小 learning rate, 防止通用能力被快速破坏.

---

## 6. Connector 对比

|Connector|代表模型|Token 长度|核心优势|主要局限|
|---|---|---|---|---|
|Linear|早期 VLM|基本不变|参数少, 对齐稳定|表达能力弱|
|MLP|LLaVA|基本不变|简单, 主流, 易复现|高分辨率 token 成本高|
|Q-Former|BLIP-2|压成固定 query 数|冻结模型连接, 参数效率高|dense detail 容易损失|
|Perceiver Resampler|Flamingo|压成固定 latent 数|适合多图和交错上下文|通常配合复杂 cross-attention|
|Patch Merger|Qwen-VL|按空间邻域缩短|兼顾高分辨率和局部结构|压缩率需要精细权衡|
|Cross-Attention Connector|Flamingo 类|视觉 memory 可独立|融合能力强|修改 LLM, 工程复杂|

MLP, Q-Former 和 Patch Merger 的区别可以归结为: MLP 主要改变特征维度, Q-Former 用固定 query 做全局摘要, Patch Merger 按空间邻域压缩 patch. 选择时应先判断任务是否依赖 dense visual detail, 再确定能够接受的 visual token 数量.

---

## 7. 总结

Projector 是视觉表示进入语言模型的接口, 也是 VLM 中最直接的 token 信息瓶颈. MLP Projector 结构简单且仍是主流基线; Q-Former 和 Perceiver Resampler 用固定长度表示换取参数与推理效率; Patch Merger 更适合 Dynamic Resolution 下的局部压缩; Cross-Attention 提供更深的融合但增加工程复杂度. Connector 的效果必须和 [Vision_Encoder](./Vision_Encoder.md) 的输出以及 [Inference](./Inference.md) 的 token budget 一起判断.
