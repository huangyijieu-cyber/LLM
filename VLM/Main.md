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
\text{Text / Box / Structure}
$$

其核心思想是:

**先把图像编码成视觉特征, 再把视觉特征映射到语言模型能理解的 embedding space, 最后让 LLM 像处理文本 token 一样处理 visual tokens.**

---

## 1. VLM 主要解决什么问题

VLM 的典型任务包括:

1. **Image Captioning**: 根据图片生成描述.
2. **VQA (Visual Question Answering)**: 根据图片回答问题.
3. **OCR / Document QA**: 理解图片中的文字, 表格和文档布局.
4. **Chart / Table Understanding**: 理解图表, 表格和坐标信息.
5. **Visual Grounding**: 根据文本定位图像区域, 或根据区域生成描述.
6. **Multi-Image Understanding**: 多图对比, 多页文档, 前后检查对比.
7. **Video Understanding**: 理解多帧视频内容和时间变化.
8. **Medical VLM**: 理解 X-ray, CT, MRI, 病理图像和医学报告.

其中医疗场景尤其关注:

- 医学影像问答.
- 医学报告生成.
- 病理图像理解.
- 多模态病历理解.
- 病灶定位.
- 医疗安全和幻觉控制.

详见 [Medical_VLM](./Medical_VLM.md).

---

## 2. VLM 的基本组成

### 2.1 Vision Encoder

[Vision Encoder](./Vision_Encoder.md) 负责把图像转成视觉特征.

常见选择:

1. ViT.
2. CLIP Vision Encoder.
3. SigLIP.
4. DINOv2.
5. EVA-CLIP.
6. InternViT / strong vision foundation model.

Vision Encoder 决定模型能否看清图像细节. 如果视觉塔已经丢失小字, 病灶或局部目标, 后续 LLM 很难通过语言推理恢复这些信息.

---

### 2.2 Projector / Connector

[Projector](./Projector.md) 负责把视觉特征映射到 LLM hidden size.

常见选择:

1. Linear Projector.
2. MLP Projector.
3. Q-Former.
4. Perceiver Resampler.
5. Patch Merger.
6. Cross-Attention Connector.

Projector 不只是做维度对齐, 还会影响:

1. visual token 数量.
2. 视觉信息压缩程度.
3. 训练稳定性.
4. 高分辨率输入成本.

---

### 2.3 LLM

LLM 负责理解融合后的 text tokens 和 visual tokens, 并生成回答.

常见基座:

1. LLaMA / Vicuna.
2. Qwen.
3. InternLM.
4. Mistral.
5. DeepSeek.

LLM 的基础结构见 [Transformer](../basic/Transformer%20架构.md), 注意力机制见 [Attention](../basic/Attention.md).

---

## 3. 主流 VLM 技术路线

主流架构路线详见 [Architecture](./Architecture.md). 可以先按下面几类理解.

### 3.1 CLIP / SigLIP 路线

CLIP / SigLIP 主要解决图文表征对齐:

$$
\text{Image Encoder}(I) \rightarrow z_I
$$

$$
\text{Text Encoder}(T) \rightarrow z_T
$$

目标是让匹配图文对相似度更高, 不匹配图文对相似度更低.

这类模型本身通常不负责生成长回答, 但常作为 VLM 的视觉塔.

---

### 3.2 BLIP / BLIP-2 路线

BLIP 系列强调图文生成预训练.

[BLIP-2](https://arxiv.org/abs/2301.12597) 的核心是 Q-Former:

$$
\text{Frozen Vision Encoder}
\rightarrow
\text{Q-Former}
\rightarrow
\text{Frozen LLM}
$$

它使用少量 query tokens 从视觉特征中提取语言相关信息, 再对接冻结 LLM.

---

### 3.3 Flamingo 路线

[Flamingo](https://arxiv.org/abs/2204.14198) 使用 Perceiver Resampler 和 gated cross-attention 把视觉信息注入语言模型.

它的重要特点:

1. 支持 interleaved image-text 输入.
2. 适合多图上下文.
3. 强调 multimodal few-shot learning.
4. 需要修改 LLM 中间层, 工程复杂度较高.

---

### 3.4 LLaVA 路线

[LLaVA](https://arxiv.org/abs/2304.08485) 是经典开源 VLM 路线.

结构:

$$
\text{CLIP Vision Encoder}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{LLM}
$$

LLaVA 的关键不是复杂架构, 而是:

1. 使用简单 Projector 对齐 visual tokens.
2. 使用多模态指令数据进行 Visual Instruction Tuning.
3. 形成容易复现的开源 VLM 基础范式.

---

### 3.5 Qwen-VL / InternVL 路线

Qwen-VL 和 InternVL 代表更强的开源 VLM 方向.

共同趋势:

1. 更高分辨率输入.
2. 更强 Vision Encoder.
3. 更好的 visual token 压缩.
4. 更强 OCR, document, chart 能力.
5. 支持 grounding 和坐标输出.
6. 支持多图和视频.
7. 更系统的数据配方和训练流程.

区别:

- Qwen-VL 更偏真实世界多模态助手, 强调 OCR, document, grounding, video, GUI.
- InternVL 更偏强视觉塔和系统训练配方, 强调 dynamic high resolution, V2PE, 原生多模态预训练和偏好优化.

---

## 4. VLM 训练流程

VLM 训练通常分为几个阶段:

1. **Image-Text Pretraining**: 学习图像和文本的基础对齐.
2. **Caption / Generative Pretraining**: 学习根据图像生成文本.
3. **Projector Alignment**: 让视觉特征进入 LLM embedding space.
4. **Multimodal Instruction Tuning**: 训练模型根据图片回答问题.
5. **Preference Alignment**: 使用 [DPO](../Align/DPO.md), [RLHF](../Align/RLHF.md), [GRPO](../Align/GRPO.md) 等方法优化回答偏好和安全性.
6. **Domain Adaptation**: 在医疗, 文档, 图表等垂直场景中继续训练.

详见 [Training](./Training.md).

---

## 5. VLM 数据和评测

VLM 的能力高度依赖数据.

重要数据类型:

1. Image-text pair.
2. Caption 数据.
3. Multimodal instruction data.
4. OCR / Document 数据.
5. Grounding 数据.
6. Video 数据.
7. Medical image-report pair.

详见 [Data_Process](./Data_Process.md).

VLM 评测需要拆成多个维度:

1. 通用视觉理解.
2. OCR.
3. 文档和图表.
4. 视觉推理.
5. grounding.
6. hallucination.
7. 医疗安全性.

详见 [Evaluation](./Evaluation.md).

---

## 6. VLM 推理特点

VLM 推理相比纯 LLM 多了视觉输入处理:

1. 图片预处理.
2. Vision Encoder forward.
3. Projector forward.
4. visual tokens 插入上下文.
5. LLM prefill.
6. LLM decode.

Visual tokens 会占用上下文长度, 从而影响 [KV Cache](../Inference/KV_Cache.md), batch size 和推理速度.

详见 [Inference](./Inference.md).

---

## 7. 学习顺序

建议按以下顺序学习:

1. [Main](./Main.md): 了解 VLM 整体结构.
2. [Architecture](./Architecture.md): 理解主流模型路线.
3. [Vision_Encoder](./Vision_Encoder.md): 理解视觉塔.
4. [Projector](./Projector.md): 理解视觉特征如何进入 LLM.
5. [Training](./Training.md): 理解训练流程.
6. [Data_Process](./Data_Process.md): 理解数据构造和清洗.
7. [Evaluation](./Evaluation.md): 理解评测体系.
8. [Inference](./Inference.md): 理解部署和推理成本.
9. [Medical_VLM](./Medical_VLM.md): 理解医疗场景.

---

## 8. 总结

VLM 的核心不是让 LLM 直接处理像素, 而是:

1. 用 Vision Encoder 提取视觉特征.
2. 用 Projector 将视觉特征映射成 visual tokens.
3. 用 LLM 结合 visual tokens 和 text tokens 做生成.
4. 用多模态数据训练模型学会图文联合推理.

当前强 VLM 的关键在于高分辨率细节保留, visual token 成本控制, 空间/时间位置建模, 数据配比和安全评测.
