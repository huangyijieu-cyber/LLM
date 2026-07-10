# VLM Training

VLM 训练的核心目标是:

**让模型学会视觉和语言之间的对齐, 并能根据视觉输入完成指令任务.**

一个常见训练流程是:

$$
\text{Image-Text Pretraining}
\rightarrow
\text{Projector Alignment}
\rightarrow
\text{Multimodal Instruction Tuning}
\rightarrow
\text{Preference Alignment}
\rightarrow
\text{Domain Adaptation}
$$

---

## 1. 图文对齐预训练

图文对齐预训练让模型知道图片和文字之间的语义对应关系.

### 1.1 Contrastive Learning

代表模型是 [CLIP](https://arxiv.org/abs/2103.00020).

输入是一批图文对:

$$
\{(I_i,T_i)\}_{i=1}^{B}
$$

目标是让匹配的 $(I_i,T_i)$ 相似度更高, 不匹配的 $(I_i,T_j)$ 相似度更低.

直观理解:

**图片 "一只猫" 应该靠近文本 "a cat", 远离文本 "a car".**

### 1.2 Sigmoid Contrastive Learning

[SigLIP](https://arxiv.org/abs/2303.15343) 使用 sigmoid loss, 将每个 image-text pair 当作二分类问题.

相比 CLIP 的 softmax 对比学习, SigLIP 对分布式大规模训练更友好.

### 1.3 作用

图文对齐预训练通常用于训练 vision encoder, 让视觉特征天然具备语言语义.

后续 VLM 可以直接复用这些视觉塔.

---

## 2. Caption / Generative Pretraining

Caption 预训练让模型根据图片生成描述.

形式:

$$
P(y|I)
$$

其中 $I$ 是图像, $y$ 是 caption.

训练目标和普通语言模型类似:

$$
L = -\sum_{t=1}^{T} \log P(y_t \mid I,y_{1:t-1})
$$

### 2.1 特点

1. **直接训练生成能力**.
2. **适合 image captioning 和 VQA**.
3. **比纯对比学习更接近对话式 VLM**.

### 2.2 局限

caption 数据通常描述图片显著内容, 不一定覆盖:

- 细粒度推理.
- OCR.
- 医学诊断.
- 多轮对话.
- 指令遵循.

因此后续还需要 instruction tuning.

---

## 3. Projector Alignment

在 LLaVA 类模型中, 常见第一阶段是 projector alignment.

训练设置:

- 冻结 Vision Encoder.
- 冻结 LLM.
- 只训练 [Projector](./Projector.md).

目标是让视觉特征能进入 LLM embedding space.

常用数据:

- image-caption pair.
- image-text pair.
- 简单 VQA pair.

### 3.1 为什么先对齐 Projector

如果一开始直接训练整个 VLM, LLM 看到的是完全陌生的视觉 embedding, 训练会不稳定.

先训练 projector 等于先教模型:

**这些 visual tokens 大概对应什么语言语义.**

---

## 4. Multimodal Instruction Tuning

Multimodal Instruction Tuning 是现代 VLM 的关键阶段.

它类似 [SFT](../Finetune/Main.md), 但输入中包含图像:

```text
USER: <image>
What is abnormal in this X-ray?

ASSISTANT: ...
```

训练目标是最大化标准回答的 token 概率:

$$
L_{\mathrm{SFT}} = -\sum_{t=1}^{T} \log P(y_t \mid I,x,y_{1:t-1})
$$

### 4.1 数据类型

1. **General VQA**: 通用图片问答.
2. **OCR QA**: 图片中文字理解.
3. **Document QA**: 文档截图理解.
4. **Chart QA**: 图表问答.
5. **Grounding**: 区域定位和坐标输出.
6. **Medical QA**: 医学影像和报告问答.

### 4.2 训练参数选择

常见策略:

1. 冻结 vision encoder.
2. 训练 projector.
3. LLM 使用 [LoRA](../Finetune/PEFT.md) 或全参数微调.

大模型训练中, 也可能使用 [DeepSpeed](../Framework/DeepSpeed.md) 或 [Megatron](../Framework/Megatron.md) 来做分布式训练.

---

## 5. Preference Alignment

VLM 也可以做偏好对齐.

### 5.1 Multimodal DPO

和 [DPO](../Align/DPO.md) 类似, 数据变成:

$$
(I,x,y_w,y_l)
$$

其中:

- $I$: 图像.
- $x$: 文本 prompt.
- $y_w$: 更好回答.
- $y_l$: 更差回答.

目标是让模型更偏好 $y_w$.

### 5.2 Multimodal RLHF

和 [RLHF](../Align/RLHF.md) 类似, VLM 可以通过 reward model 或规则 verifier 优化.

例如:

- OCR 是否正确.
- 医学诊断是否安全.
- 坐标输出是否匹配目标框.
- JSON 格式是否正确.

### 5.3 Multimodal GRPO / DAPO

对同一张图片和 prompt 生成多个回答, 然后使用 reward 或 verifier 打分:

$$
\{y_1,y_2,\dots,y_G\}
$$

再使用 [GRPO](../Align/GRPO.md) 或 [DAPO](../Align/DAPO.md) 做组内相对优化.

这种方式适合可验证任务:

- OCR exact match.
- Chart QA.
- Medical multiple choice.
- Grounding box IoU.

---

## 6. OCR / Document 专项训练

现代强 VLM 很重视 OCR 和文档能力.

专项训练通常包括:

1. TextVQA 类数据.
2. 文档截图问答.
3. 表格理解.
4. 图表问答.
5. 屏幕截图理解.

训练重点:

- 高分辨率输入.
- 保留细粒度 visual tokens.
- 构造文字位置和内容相关问题.
- 加入结构化输出.

这类能力对医疗场景也很重要, 因为病历, 检查单, 报告都是文档型数据.

---

## 7. Grounding 训练

Grounding 是让 VLM 能把语言和图像区域对应起来.

常见任务:

- 根据文字找区域.
- 根据区域回答问题.
- 输出 bounding box.
- 输出 point 或 mask.

训练数据通常包含:

$$
(\text{image}, \text{text}, \text{box})
$$

### 7.1 价值

Grounding 可以缓解 VLM 幻觉, 因为模型不仅要回答, 还要说明视觉证据在哪里.

在医疗场景中, grounding 很重要:

- 指出病灶区域.
- 标出异常影像位置.
- 辅助医生检查模型依据.

---

## 8. 医疗领域继续训练

医疗 VLM 通常需要领域继续训练.

原因:

1. 通用图片和医学影像分布差异大.
2. 医学术语密集.
3. 专业标注稀缺.
4. 安全要求高.

常见训练数据:

- X-ray image-report pair.
- CT / MRI 影像描述.
- 病理图像 patch.
- 医学 VQA.
- 医学教材和指南.

详见 [Medical_VLM](./Medical_VLM.md).

---

## 9. 训练阶段对比

|阶段|训练对象|主要数据|目标|
|---|---|---|---|
|图文对齐预训练|Vision Encoder|image-text pair|学习视觉和文本语义对齐|
|Caption 预训练|VLM|image-caption|学习根据图像生成文本|
|Projector Alignment|Projector|image-caption / VQA|把视觉特征映射到 LLM 空间|
|Instruction Tuning|Projector + LLM|多模态指令数据|学会根据图片回答问题|
|Preference Alignment|VLM|chosen / rejected|优化回答偏好和安全性|
|Domain Adaptation|VLM|医疗等领域数据|适配垂直场景|

一句话总结:

**VLM 训练不是单一步骤, 而是先图文对齐, 再把视觉特征接入 LLM, 再用多模态指令和偏好数据训练模型真正会用图像回答问题.**
