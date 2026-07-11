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

其中不同阶段解决的问题不同:

1. 图文预训练解决视觉和语言语义对齐.
2. Projector Alignment 解决 visual tokens 如何进入 LLM.
3. Instruction Tuning 解决模型是否会按指令看图回答.
4. Preference Alignment 解决回答偏好, 幻觉和安全性.
5. Domain Adaptation 解决医疗, 文档, 图表等垂直场景.

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

这类训练常用于训练 [Vision Encoder](./Vision_Encoder.md), 使视觉特征天然具备语言语义.

### 1.2 Sigmoid Contrastive Learning

[SigLIP](https://arxiv.org/abs/2303.15343) 使用 sigmoid loss, 将每个 image-text pair 当作二分类问题.

相比 CLIP 的 softmax 对比学习, SigLIP 对分布式大规模训练更友好.

### 1.3 作用

图文对齐预训练的价值:

1. 提供强视觉语义表示.
2. 提供可复用的 vision tower.
3. 提升 zero-shot 和 retrieval 能力.
4. 为后续 VLM 图文对齐降低难度.

局限:

1. 不能直接生成长回答.
2. 对 OCR, document, medical image 等细节任务不一定足够.
3. 需要后续 instruction tuning 才能成为对话式 VLM.

---

## 2. Caption / Generative Pretraining

Caption 预训练让模型根据图片生成描述.

形式:

$$
P(y \mid I)
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
4. **可作为 instruction tuning 之前的生成式预训练**.

### 2.2 局限

Caption 数据通常描述图片显著内容, 不一定覆盖:

1. 细粒度推理.
2. OCR.
3. 医学诊断.
4. 多轮对话.
5. 指令遵循.
6. grounding 和坐标输出.

因此后续还需要 multimodal instruction tuning.

---

## 3. Projector Alignment

在 LLaVA 类模型中, 常见第一阶段是 projector alignment.

训练设置:

1. 冻结 [Vision Encoder](./Vision_Encoder.md).
2. 冻结 LLM.
3. 只训练 [Projector](./Projector.md).

目标是让视觉特征能进入 LLM embedding space.

常用数据:

1. image-caption pair.
2. image-text pair.
3. 简单 VQA pair.

---

### 3.1 为什么先对齐 Projector

如果一开始直接训练整个 VLM, LLM 看到的是完全陌生的 visual embeddings, 训练会不稳定.

先训练 Projector 的作用是:

1. 对齐 Vision Encoder hidden space 和 LLM hidden space.
2. 降低后续 instruction tuning 难度.
3. 避免过早破坏 LLM 语言能力.
4. 降低训练成本.

---

### 3.2 常见问题

Projector Alignment 数据如果太简单, 模型可能只学会图像描述, 不会复杂问答.

如果 alignment 数据质量差, 可能导致:

1. visual tokens 语义不稳定.
2. 后续 instruction tuning 收敛慢.
3. 图像细节无法有效进入 LLM.

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

其中:

- $I$: 图像.
- $x$: 用户指令.
- $y$: 标准回答.

---

### 4.1 数据类型

常见数据类型:

1. **General VQA**: 通用图片问答.
2. **OCR QA**: 图片中文字理解.
3. **Document QA**: 文档截图理解.
4. **Chart QA**: 图表问答.
5. **Grounding**: 区域定位和坐标输出.
6. **Multi-Image QA**: 多图比较.
7. **Video QA**: 视频帧理解.
8. **Medical QA**: 医学影像和报告问答.

不同数据对应不同能力. 如果 OCR 数据不足, 模型读字能力通常较弱. 如果 grounding 数据不足, 模型很难输出可靠位置.

---

### 4.2 训练参数选择

常见策略:

1. 冻结 Vision Encoder.
2. 训练 Projector.
3. LLM 使用 [LoRA](../Finetune/PEFT.md) 或全参数微调.

高性能 VLM 也可能训练全部模块:

1. Vision Encoder.
2. Projector.
3. LLM.

大模型训练中, 可能使用 [DeepSpeed](../Framework/DeepSpeed.md) 或 [Megatron](../Framework/Megatron.md) 做分布式训练.

---

### 4.3 Instruction Tuning 的风险

Multimodal Instruction Tuning 常见问题:

1. 合成数据风格过强, 模型回答模式单一.
2. QA 数据太短, 模型缺少解释能力.
3. Caption 数据太多, 模型倾向于描述而不是回答.
4. OCR / grounding 数据不足, 细节能力弱.
5. 医疗等高风险领域缺少安全边界.

因此数据配比非常重要, 详见 [Data_Process](./Data_Process.md).

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

常见偏好维度:

1. 是否符合图像证据.
2. 是否减少幻觉.
3. 是否遵循输出格式.
4. 是否表达不确定性.
5. 医疗回答是否安全.

---

### 5.2 Multimodal RLHF

和 [RLHF](../Align/RLHF.md) 类似, VLM 可以通过 reward model 或规则 verifier 优化.

例如:

1. OCR 是否正确.
2. 医学诊断是否安全.
3. 坐标输出是否匹配目标框.
4. JSON 格式是否正确.
5. 图表计算是否正确.

---

### 5.3 Multimodal GRPO / DAPO

对同一张图片和 prompt 生成多个回答, 然后使用 reward 或 verifier 打分:

$$
\{y_1,y_2,\dots,y_G\}
$$

再使用 [GRPO](../Align/GRPO.md) 或 [DAPO](../Align/DAPO.md) 做组内相对优化.

这种方式适合可验证任务:

1. OCR exact match.
2. Chart QA.
3. Medical multiple choice.
4. Grounding box IoU.
5. 数学图表推理.

大规模 rollout 可以结合 [vLLM](../Framework/vllm.md), [SGLang](../Framework/SGLang.md), [verl](../Framework/Verl.md).

---

## 6. OCR / Document 专项训练

现代强 VLM 很重视 OCR 和文档能力.

专项训练通常包括:

1. TextVQA 类数据.
2. 文档截图问答.
3. 表格理解.
4. 图表问答.
5. 屏幕截图理解.
6. 多页文档理解.

训练重点:

1. 高分辨率输入.
2. 保留细粒度 visual tokens.
3. 构造文字位置和内容相关问题.
4. 加入结构化输出.
5. 保留 layout 信息.

这类能力对医疗场景也很重要, 因为病历, 检查单, 报告都是文档型数据.

---

## 7. Grounding 训练

Grounding 是让 VLM 能把语言和图像区域对应起来.

常见任务:

1. 根据文字找区域.
2. 根据区域回答问题.
3. 输出 bounding box.
4. 输出 point 或 mask.
5. 解释答案来自哪个区域.

训练数据通常包含:

$$
(\text{image}, \text{text}, \text{box})
$$

或:

$$
(\text{image}, \text{region}, \text{answer})
$$

### 7.1 价值

Grounding 可以缓解 VLM 幻觉, 因为模型不仅要回答, 还要说明视觉证据在哪里.

在医疗场景中, grounding 很重要:

1. 指出病灶区域.
2. 标出异常影像位置.
3. 辅助医生检查模型依据.

---

## 8. Video 训练

Video VLM 训练通常将视频转为帧序列:

$$
V = \{I_1,I_2,\dots,I_T\}
$$

训练任务包括:

1. Video caption.
2. Video QA.
3. Action recognition.
4. Temporal reasoning.
5. 多帧事件描述.

关键问题:

1. 帧数多, visual tokens 多.
2. 需要时间位置编码.
3. 需要处理长视频压缩.
4. 需要构造时间顺序相关问题.

详见 [Inference](./Inference.md).

---

## 9. 医疗领域继续训练

医疗 VLM 通常需要领域继续训练.

原因:

1. 通用图片和医学影像分布差异大.
2. 医学术语密集.
3. 专业标注稀缺.
4. 安全要求高.
5. 需要表达不确定性和建议复核.

常见训练数据:

1. X-ray image-report pair.
2. CT / MRI 影像描述.
3. 病理图像 patch.
4. 医学 VQA.
5. 医学教材和指南.
6. 医疗文档 OCR.

详见 [Medical_VLM](./Medical_VLM.md).

---

## 10. 训练阶段对比

|阶段|训练对象|主要数据|目标|
|---|---|---|---|
|Image-Text Pretraining|Vision Encoder|image-text pair|学习视觉和文本语义对齐|
|Caption Pretraining|VLM|image-caption|学习根据图像生成文本|
|Projector Alignment|Projector|image-caption / VQA|把视觉特征映射到 LLM 空间|
|Instruction Tuning|Projector + LLM|多模态指令数据|学会根据图片回答问题|
|Preference Alignment|VLM|chosen / rejected|优化回答偏好和安全性|
|Domain Adaptation|VLM|医疗等领域数据|适配垂直场景|

---

## 11. 总结

VLM 训练不是单一步骤.

1. 图文对齐预训练让视觉特征具备语言语义.
2. Projector Alignment 让视觉特征能进入 LLM 空间.
3. Multimodal Instruction Tuning 让模型学会按指令看图回答.
4. Preference Alignment 用于降低幻觉, 优化格式和安全性.
5. OCR, grounding, video, medical 等能力需要专项数据和专项训练.
