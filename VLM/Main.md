# VLM

VLM 全称是 **Vision-Language Model**, 即视觉语言模型. 它的目标是让模型同时理解视觉输入和文本输入, 并用自然语言或结构化格式输出结果.

VLM 可以看作是在 [LLM](../basic/Transformer%20架构.md) 的基础上接入视觉信息:

$$
\text{Image / Video / Document}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Projector / Connector}
\rightarrow
\text{LLM}
\rightarrow
\text{Text Answer}
$$

其核心思想是:

**先把图像编码成视觉特征, 再把视觉特征映射到语言模型能理解的 embedding 空间, 最后让 LLM 像处理文本 token 一样处理 visual tokens.**

---

## 1. VLM 主要解决什么问题

VLM 的典型任务包括:

1. **Image Captioning**: 给图片生成描述.
2. **VQA(Visual Question Answering)**: 根据图片回答问题.
3. **OCR / Document QA**: 理解图片中的文字, 表格和文档布局.
4. **Visual Grounding**: 根据文本定位图像区域, 或根据区域生成描述.
5. **Chart / Table Understanding**: 理解图表, 表格和坐标信息.
6. **Video Understanding**: 理解多帧视频内容.
7. **Medical VLM**: 理解 X-ray, CT, MRI, 病理图像和医学报告.

其中医疗大模型场景尤其关注:

- 医学影像问答.
- 医学报告生成.
- 病理图像理解.
- 多模态病历理解.
- 医疗安全和幻觉控制.

详见 [Medical_VLM](./Medical_VLM.md).

---

## 2. VLM 的基本组成

### 2.1 Vision Encoder

Vision Encoder 负责把图像转成视觉特征.

常见选择:

- ViT.
- CLIP Vision Encoder.
- SigLIP.
- DINOv2.
- EVA-CLIP.

详见 [Vision_Encoder](./Vision_Encoder.md).

---

### 2.2 Projector / Connector

Projector 负责把视觉特征映射到 LLM hidden size.

常见选择:

- Linear Projector.
- MLP Projector.
- Q-Former.
- Perceiver Resampler.
- Cross-Attention Connector.

详见 [Projector](./Projector.md).

---

### 2.3 LLM

LLM 负责理解融合后的文本 token 和 visual token, 并生成回答.

常见基座:

- LLaMA / Vicuna.
- Qwen.
- InternLM.
- Mistral.
- DeepSeek.

LLM 的基础结构见 [Transformer](../basic/Transformer%20架构.md), 注意力机制见 [Attention](../basic/Attention.md).

---

## 3. 主流 VLM 技术路线

### 3.1 CLIP 路线

[CLIP](https://arxiv.org/abs/2103.00020) 使用图像编码器和文本编码器做对比学习.

它的核心目标是:

$$
\text{matched image-text pair similarity} \uparrow
$$

$$
\text{unmatched image-text pair similarity} \downarrow
$$

CLIP 的价值在于学到强大的图文对齐表示, 但它本身不是一个强生成式聊天模型. 后续很多 VLM 都使用 CLIP 或类似视觉编码器作为视觉塔.

---

### 3.2 BLIP / BLIP-2 路线

BLIP 系列强调图文预训练和生成能力.

[BLIP-2](https://arxiv.org/abs/2301.12597) 的核心是 **Q-Former**:

$$
\text{Frozen Vision Encoder}
\rightarrow
\text{Q-Former}
\rightarrow
\text{Frozen LLM}
$$

它通过少量可训练 query tokens 从视觉编码器中提取和语言相关的信息, 再对接冻结 LLM.

BLIP-2 的特点:

1. 视觉编码器和 LLM 可以冻结.
2. 主要训练中间的 Q-Former.
3. 参数效率较高.
4. 适合作为早期 VLM 架构理解样板.

---

### 3.3 Flamingo 路线

[Flamingo](https://arxiv.org/abs/2204.14198) 使用 Perceiver Resampler 和 gated cross-attention 把视觉信息注入语言模型.

它的特点是:

1. 支持 interleaved image-text 输入.
2. 可以处理多图上下文.
3. 强调 few-shot multimodal learning.
4. 架构比简单 projector 更复杂.

Flamingo 对后续多图多轮 VLM 有较大影响, 但在开源实践中, LLaVA 类简单 projector 路线更常见.

---

### 3.4 LLaVA 路线

[LLaVA](https://arxiv.org/abs/2304.08485) 是当前最经典的开源 VLM 路线之一.

它的结构非常直接:

$$
\text{CLIP Vision Encoder}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{LLM}
$$

LLaVA 的关键不是复杂架构, 而是 **Visual Instruction Tuning**.

训练通常分两步:

1. **Feature Alignment**: 冻结 vision encoder 和 LLM, 训练 projector.
2. **Visual Instruction Tuning**: 用多模态指令数据训练 projector 和 LLM.

LLaVA 路线的优点是简单, 易复现, 工程友好, 因此成为很多后续开源 VLM 的基础范式.

---

### 3.5 Qwen-VL / InternVL 路线

Qwen-VL, Qwen2-VL, Qwen2.5-VL 和 InternVL 系列代表了当前更强的开源 VLM 方向.

这类模型通常不只是简单图片问答, 还强调:

- 高分辨率图像理解.
- OCR 和文档理解.
- 多图输入.
- 视频理解.
- grounding 和坐标输出.
- 更强的中文能力.

它们的共同趋势是:

1. 更强视觉编码器.
2. 更好的 visual token 压缩策略.
3. 更大规模多模态指令数据.
4. 更强 OCR, chart, document 能力.
5. 更接近通用多模态助手.

---

## 4. VLM 训练流程

VLM 训练通常分为几个阶段:

1. **图文预训练**: 学习图像和文本的基础对齐.
2. **Projector 预训练**: 让视觉特征能进入 LLM embedding 空间.
3. **多模态指令微调**: 训练模型根据图片回答问题.
4. **偏好对齐**: 使用 [DPO](../Align/DPO.md), [RLHF](../Align/RLHF.md) 等方法优化回答风格和安全性.
5. **领域微调**: 在医疗, 文档, 图表等垂直场景中继续训练.

详见 [Training](./Training.md).

---

## 5. VLM 数据和评测

VLM 的能力高度依赖数据.

重要数据类型包括:

- Image-text pair.
- Caption 数据.
- OCR 数据.
- VQA 数据.
- Grounding 数据.
- Document QA.
- Chart QA.
- Medical image-report pair.

详见 [Data_Process](./Data_Process.md).

评测方面, 需要同时考察:

- 通用视觉理解.
- OCR.
- 文档理解.
- 数学图表推理.
- 幻觉.
- 医疗安全性.

详见 [Evaluation](./Evaluation.md).

---

## 6. VLM 推理特点

VLM 推理相比纯 LLM 多了视觉输入处理:

1. 图片预处理.
2. 视觉编码器 forward.
3. visual token 插入上下文.
4. LLM prefill.
5. LLM decode.

视觉 token 会占用上下文长度, 从而影响 [KV Cache](../Inference/KV_Cache.md), batch size 和推理速度.

详见 [Inference](./Inference.md).

---

## 7. 学习顺序

建议按以下顺序学习:

1. [Main](./Main.md): 了解 VLM 整体结构.
2. [Architecture](./Architecture.md): 理解主流模型路线.
3. [Vision_Encoder](./Vision_Encoder.md): 理解视觉塔.
4. [Projector](./Projector.md): 理解视觉特征如何进入 LLM.
5. [Training](./Training.md): 理解训练流程.
6. [Data_Process](./Data_Process.md): 理解数据构造.
7. [Evaluation](./Evaluation.md): 理解评测体系.
8. [Inference](./Inference.md): 理解部署和推理成本.
9. [Medical_VLM](./Medical_VLM.md): 理解医疗场景.

一句话总结:

**VLM 的核心不是让 LLM 直接看图片, 而是把图片变成 LLM 能理解的 visual tokens, 再通过多模态数据训练模型学会图文联合推理.**
