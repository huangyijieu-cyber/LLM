# VLM

VLM 全称是 **Vision-Language Model**, 即视觉语言模型. 它把图像, 视频或文档转换成语言模型能够处理的表示, 使模型可以围绕视觉内容进行问答, 推理和生成.

现代 VLM 通常采用下面的结构:

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

**VLM 的重点不是让 LLM 直接读取像素, 而是把视觉信息转换为 visual tokens, 再让 LLM 联合处理视觉与文本上下文.** 其中 LLM 的基础结构见 [Transformer](../basic/Transformer%20架构.md), 融合过程依赖 [Attention](../basic/Attention.md).

---

## 1. 基本结构

[Vision Encoder](./Vision_Encoder.md) 负责从像素中提取语义和空间特征. [Projector](./Projector.md) 负责将视觉特征映射到 LLM hidden space, 必要时还会压缩 token. LLM 则基于视觉上下文理解问题并生成回答. 三个模块的能力不是相互替代的: Vision Encoder 没有保留的小字和病灶细节, Projector 与 LLM 通常无法恢复; Projector 压缩过强时, 强视觉塔也会丢失局部信息; LLM 能力不足时, 模型即使看清图像也无法完成复杂推理.

VLM 架构的核心矛盾是 **视觉信息量和计算成本之间的权衡**. 对 patch size 为 $P \times P$ 的图像, visual token 数量近似为:

$$
N_v \approx \frac{H \times W}{P^2}
$$

提高分辨率可以改善 OCR, 文档, 图表和小目标识别, 但也会增加 LLM prefill, [KV Cache](../Inference/KV_Cache.md) 和 serving 调度成本. **因此主流架构都在解决两个问题: 如何保留足够的视觉细节, 以及如何控制进入 LLM 的 visual token 数量.**

---

## 2. 能力范围

VLM 的能力可以按视觉证据的形式划分. 通用 VQA 和 Caption 主要依赖场景语义; OCR, Document QA 和 Chart QA 需要读取高分辨率文字并理解二维布局; Grounding 要求把文本与具体区域对应; Visual Reasoning 还要在感知结果上进行数学, 空间或常识推理. 多图和视频任务进一步引入跨图关联与时间顺序, 医疗任务则增加专业知识, 不确定性和安全边界.

|能力|典型任务|主要依赖|
|---|---|---|
|通用视觉理解|Caption, VQA, counting|图文对齐和场景语义|
|文字与文档|OCR, DocVQA, ChartQA|高分辨率, layout, 结构化输出|
|区域定位|Grounding, GUI click, lesion localization|局部特征, 空间位置, 坐标格式|
|视觉推理|MathVista, science QA, multi-image comparison|视觉证据和 LLM 推理能力|
|多图与视频|多页文档, 视频问答, temporal reasoning|token 压缩, 跨图关联, 时间位置|
|医疗多模态|Medical VQA, report generation, lesion grounding|专业域适配, 细粒度感知, 安全性|

医疗场景的任务, 数据和风险单独见 [Medical_VLM](./Medical_VLM.md).

---

## 3. 主流技术路线

### 3.1 图文表征路线

[CLIP](https://arxiv.org/abs/2103.00020) 和 [SigLIP](https://arxiv.org/abs/2303.15343) 使用大规模图文对训练 image encoder 和 text encoder, 使匹配图文的表示更接近. 它们本身通常不负责长文本生成, 但提供了适合接入 LLM 的视觉语义特征, 因而常被用作 VLM 的 Vision Encoder.

### 3.2 冻结模型连接路线

[BLIP-2](https://arxiv.org/abs/2301.12597) 使用 Q-Former 从视觉特征中提取固定数量的 query features, 以较低训练成本连接冻结 Vision Encoder 和冻结 LLM. [Flamingo](https://arxiv.org/abs/2204.14198) 使用 Perceiver Resampler 压缩视觉特征, 再通过 gated cross-attention 在 LLM 中间层注入视觉信息. 两者都重视 token 压缩和参数效率, 但结构比简单 Projector 更复杂.

### 3.3 Decoder-Only VLM 路线

当前开源 VLM 的主流做法是将 visual tokens 直接放入 decoder-only LLM 的上下文. 这条路线保留了 LLM 原有的生成接口, 便于复用 [SFT](../Finetune/Main.md), [DPO](../Align/DPO.md), [RLHF](../Align/RLHF.md) 和 [GRPO](../Align/GRPO.md) 等后训练方法. 常见语言基座包括 LLaMA / Vicuna, Qwen, InternLM, Mistral 和 DeepSeek 等, 但基座名称不是架构分类的关键, 关键仍是视觉 token 如何进入并参与 LLM 计算.

|路线|核心设计|主要特点|
|---|---|---|
|LLaVA|CLIP Vision Encoder + MLP Projector + LLM|结构简单, Visual Instruction Tuning 范式清晰|
|LLaVA-NeXT / OneVision|AnyRes, global view + local tiles|加强高分辨率, 多图和视频能力|
|Qwen-VL|Dynamic Resolution, Patch Merger, M-RoPE|OCR, document, grounding, video 和 GUI 能力完整|
|InternVL|强 Vision Encoder, Dynamic High Resolution, V2PE|强调视觉底座, 高分辨率处理和系统训练配方|

各路线的结构与关键技巧见 [Architecture](./Architecture.md).

---

## 4. 训练和数据

VLM 通常不是一次训练完成, 而是逐步建立图文对齐, 指令遵循和领域能力:

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

**图文预训练建立语义, Projector Alignment 对齐接口, Instruction Tuning 学习任务, Preference Alignment 约束回答行为.** 医疗, 文档, 视频等能力还依赖专门的数据与继续训练. 具体流程见 [Training](./Training.md).

数据和能力需要一一对应. Image-text pair 适合学习基础对齐, Caption 数据训练描述能力, Instruction data 训练问答和指令遵循, OCR / Grounding / Video / Medical 数据负责补足专项能力. 数据构造, 清洗和配比见 [Data_Process](./Data_Process.md).

---

## 5. 推理和评测

VLM 推理在普通 LLM 之前增加了图像预处理, Vision Encoder forward 和 Projector forward. **真正影响吞吐的关键通常不是 Projector 参数量, 而是 visual tokens 增加了输入长度, 从而扩大 prefill 和 KV Cache.** 多图, 高分辨率和视频场景都需要显式管理 visual token budget, 详见 [Inference](./Inference.md).

VLM 评测也不能只看一个综合分数. 模型回答错误时, 需要区分是视觉感知失败, OCR 失败, 推理失败还是视觉幻觉. 因此评测通常组合通用 VQA, OCR / Document, Reasoning, Grounding, Hallucination, Video 和 Medical 等维度, 详见 [Evaluation](./Evaluation.md).

---

## 6. 笔记关系

|笔记|核心问题|
|---|---|
|[Architecture](./Architecture.md)|主流 VLM 如何组织 Vision Encoder, Connector 和 LLM|
|[Vision_Encoder](./Vision_Encoder.md)|图像如何转成视觉特征|
|[Projector](./Projector.md)|视觉特征如何对齐和压缩后进入 LLM|
|[Training](./Training.md)|VLM 各训练阶段分别解决什么问题|
|[Data_Process](./Data_Process.md)|不同能力需要什么数据, 如何清洗和配比|
|[Evaluation](./Evaluation.md)|如何定位模型能力和失败来源|
|[Inference](./Inference.md)|visual tokens 如何影响延迟, 显存和吞吐|
|[Medical_VLM](./Medical_VLM.md)|医疗 VLM 的任务, 训练, 风险和评测|

---

## 7. 总结

VLM 的基本链路是 **视觉编码 -> 特征对齐与压缩 -> 语言建模**. 当前主流模型之间的差异, 主要来自分辨率策略, visual token 压缩方式, 空间与时间位置编码, 训练数据配方以及偏好对齐. 理解 VLM 时需要始终把效果和成本放在一起: 视觉细节决定模型能看到什么, visual token budget 决定这些信息能否以可接受的代价进入 LLM.
