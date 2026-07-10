# VLM Architecture

VLM Architecture 主要解决一个问题:

**视觉信息如何进入语言模型, 并和文本信息一起参与推理和生成.**

现代 VLM 通常可以抽象为:

$$
\text{Image / Video / Document}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Projector / Connector}
\rightarrow
\text{LLM}
\rightarrow
\text{Answer}
$$

其中:

- [Vision Encoder](./Vision_Encoder.md): 将图像转成视觉特征.
- [Projector](./Projector.md): 将视觉特征映射到 LLM hidden space.
- LLM: 负责结合 visual tokens 和 text tokens 生成回答.

不同 VLM 架构的核心差异主要体现在:

1. **视觉 token 如何产生**: 固定分辨率 patch, dynamic resolution, tile, video frame.
2. **视觉 token 如何压缩**: MLP Projector, Q-Former, Perceiver Resampler, Patch Merger.
3. **视觉信息如何注入 LLM**: prefix visual tokens, interleaved image-text tokens, cross-attention.
4. **位置编码如何处理**: 1D text position, 2D image position, 3D video position.
5. **训练数据覆盖哪些能力**: OCR, document, chart, grounding, video, medical image.

---

## 1. 基本结构

### 1.1 Visual Tokens

VLM 不能直接把像素送入 LLM. 原始图像一般表示为:

$$
I \in \mathbb{R}^{H \times W \times C}
$$

LLM 期望的输入是 token embedding:

$$
X \in \mathbb{R}^{N \times d_{\text{model}}}
$$

因此需要先通过 [Vision Encoder](./Vision_Encoder.md) 得到视觉特征:

$$
X_v = \text{VisionEncoder}(I)
$$

其中:

$$
X_v = \{v_1,v_2,\dots,v_N\}
$$

这些 $v_i$ 就是后续进入 LLM 的视觉信息来源.

---

### 1.2 Projector

Vision Encoder 输出的 hidden size 通常和 LLM 不一致, 需要 [Projector](./Projector.md) 做映射:

$$
H_v = f_{\text{proj}}(X_v)
$$

其中:

$$
H_v \in \mathbb{R}^{N'_v \times d_{\text{LLM}}}
$$

Projector 的作用包括:

1. **维度对齐**: $d_v \rightarrow d_{\text{LLM}}$.
2. **语义对齐**: 将视觉特征映射到语言模型可理解的空间.
3. **token 压缩**: 在高分辨率或多图场景下降低 token 数.
4. **训练稳定性**: 降低视觉特征直接进入 LLM 的训练难度.

常见 Projector:

|Projector|代表模型|特点|
|---|---|---|
|Linear Projector|早期 VLM|简单, 表达能力有限|
|MLP Projector|LLaVA 系列|实现简单, 工程常用|
|Q-Former|BLIP-2|使用 query tokens 压缩视觉信息|
|Perceiver Resampler|Flamingo|将任意长度视觉特征压成固定长度 latent|
|Patch Merger|Qwen-VL 系列|在高分辨率输入下控制 visual token 数|

---

### 1.3 Visual Token Budget

VLM 中一个重要问题是 **visual token 数量**.

假设图像分辨率为 $H \times W$, patch size 为 $P \times P$, 那么 patch token 数近似为:

$$
N_v \approx \frac{H \times W}{P^2}
$$

分辨率越高, 细节保留越好, 但 visual token 数也会增加. 这会导致:

1. Prefill 计算成本增加.
2. [KV Cache](../Inference/KV_Cache.md) 增加.
3. batch size 下降.
4. 多图和视频输入成本更高.

因此主流 VLM 都需要在 **细节保留** 和 **token 成本** 之间做权衡.

---

## 2. 主流架构路线总览

|路线|代表模型|核心特点|重要程度|
|---|---|---|---|
|LLaVA 系列|[LLaVA](https://arxiv.org/abs/2304.08485), [LLaVA-1.5](https://arxiv.org/abs/2310.03744), [LLaVA-OneVision](https://arxiv.org/abs/2408.03326)|CLIP Vision Encoder + MLP Projector + LLM, 结构简单, 指令微调效果好|重点|
|Qwen-VL 系列|[Qwen-VL](https://arxiv.org/abs/2308.12966), [Qwen2-VL](https://arxiv.org/abs/2409.12191), [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), [Qwen3-VL](https://arxiv.org/abs/2511.21631)|dynamic resolution, M-RoPE, OCR, document, grounding, video 能力强|重点|
|InternVL 系列|[InternVL](https://arxiv.org/abs/2312.14238), [InternVL1.5](https://arxiv.org/abs/2404.16821), [InternVL2.5](https://arxiv.org/abs/2412.05271), [InternVL3](https://arxiv.org/abs/2504.10479)|强视觉塔, dynamic high resolution, V2PE, 多阶段训练|重点|
|BLIP-2|[BLIP-2](https://arxiv.org/abs/2301.12597)|Q-Former 连接冻结 Vision Encoder 和冻结 LLM|中等|
|Flamingo|[Flamingo](https://arxiv.org/abs/2204.14198)|Perceiver Resampler + gated cross-attention, 支持 interleaved image-text|中等|
|CLIP / SigLIP|[CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343)|图文表征对齐, 常作为 Vision Encoder|基础|
|BLIP|BLIP|Encoder-Decoder 生成式图文预训练|基础|

---

## 3. LLaVA 系列

### 3.1 核心思想

LLaVA 是当前最经典的开源 VLM 架构之一. 它的核心思想是:

**使用一个简单的 MLP Projector 将 CLIP 视觉特征映射到 LLM hidden space, 再通过 Visual Instruction Tuning 让 LLM 学会看图回答问题.**

它的重点不在复杂连接器, 而在:

1. 复用强 Vision Encoder.
2. 使用简单 Projector 对齐视觉和语言空间.
3. 构造多模态指令数据.
4. 让 LLM 通过 SFT 学会根据图像回答.

---

### 3.2 技术架构

LLaVA 的基本结构为:

$$
\text{Image}
\rightarrow
\text{CLIP Vision Encoder}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{LLM}
$$

视觉特征经过 Projector 后, 被当作 visual tokens 拼接到文本 token 中:

$$
[\text{visual tokens}; \text{text tokens}]
$$

在实际输入中, prompt 中的 `<image>` 会被替换成一组 visual tokens:

```text
USER: <image>
Describe this image.

ASSISTANT:
```

---

### 3.3 具体实现

LLaVA 通常使用 patch features, 而不是只使用 CLIP 的全局 CLS feature.

如果只使用 CLS feature:

$$
z_I \in \mathbb{R}^{d}
$$

模型只能得到整张图的全局语义, 细粒度信息会被压缩.

使用 patch features 时:

$$
X_v = \{v_1,v_2,\dots,v_N\}
$$

每个 patch 都可以作为一个 visual token 进入 LLM, 适合 VQA, OCR, grounding 等细粒度任务.

MLP Projector 常见形式为:

$$
H_v = W_2 \cdot \sigma(W_1X_v)
$$

其中:

- $X_v$: Vision Encoder 输出.
- $H_v$: LLM hidden size 下的 visual tokens.
- $\sigma$: GELU 等非线性激活函数.

---

### 3.4 训练方式

LLaVA 训练通常分为两个阶段, 详见 [Training](./Training.md).

#### 3.4.1 Feature Alignment

第一阶段主要训练 Projector:

1. 冻结 Vision Encoder.
2. 冻结 LLM.
3. 只更新 MLP Projector.

目标是让视觉特征进入 LLM embedding space.

这一步可以理解为先让 LLM 知道:

**这些 visual tokens 大致对应什么语言语义.**

#### 3.4.2 Visual Instruction Tuning

第二阶段使用多模态指令数据进行监督微调:

$$
L_{\text{SFT}} = -\sum_t \log P(y_t|I,x,y_{<t})
$$

其中:

- $I$: 图像.
- $x$: 文本指令.
- $y$: 标准回答.

这个阶段让模型学习:

1. 根据图像回答问题.
2. 遵循多模态指令.
3. 进行图像描述, VQA, 简单推理.
4. 按对话格式输出.

---

### 3.5 关键技巧

#### 3.5.1 使用 Patch Features

Patch features 能保留局部信息, 比 CLS feature 更适合细粒度图像理解.

优点:

1. 局部细节更强.
2. 更适合 VQA 和区域问题.
3. 后续可以扩展到高分辨率 tile.

缺点:

1. visual token 数更多.
2. prefill 成本更高.
3. 高分辨率下 [KV Cache](../Inference/KV_Cache.md) 压力更大.

#### 3.5.2 两阶段训练

先训练 Projector, 再做 Visual Instruction Tuning, 可以降低训练不稳定性.

如果一开始直接训练整个 VLM, LLM 会看到完全陌生的视觉 embedding, 容易出现对齐困难.

#### 3.5.3 简单 MLP Projector

LLaVA 选择 MLP Projector 而不是 Q-Former, 主要原因是:

1. 结构简单.
2. 不改变 LLM 主体结构.
3. 工程实现方便.
4. 适合开源复现和扩展.

---

### 3.6 LLaVA-NeXT / OneVision

早期 LLaVA 的主要问题是固定分辨率容易丢失细节, OCR, 文档和图表能力较弱.

LLaVA-NeXT / OneVision 在 LLaVA 基础上加强:

1. **AnyRes**: 支持任意分辨率图像.
2. **High-Resolution Tiles**: 将大图切成多个 tile.
3. **Global Image + Local Tiles**: 同时保留全局图和局部细节.
4. **Interleaved Multi-Image**: 支持多图和文本交错输入.
5. **Image / Video Task Transfer**: 将图像能力迁移到视频任务.

AnyRes 的基本流程:

$$
\text{Image}
\rightarrow
\text{Global View} + \text{Local Tiles}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{LLM}
$$

这样可以改善:

1. OCR.
2. 文档截图理解.
3. 图表问答.
4. 小目标识别.
5. 多图对比.

但代价是 visual tokens 增加, 推理成本上升.

---

### 3.7 特点和局限

LLaVA 系列的优点:

1. 架构简单.
2. 不需要大改 LLM.
3. 训练流程清晰.
4. 易于复现和扩展.
5. 是很多开源 VLM 的基础范式.

LLaVA 系列的局限:

1. 早期版本 OCR 和 document 能力较弱.
2. 高分辨率输入会显著增加 visual tokens.
3. 效果依赖 Vision Encoder 和指令数据质量.
4. 对 grounding, video, GUI 等复杂任务需要额外增强.

---

## 4. Qwen-VL 系列

### 4.1 核心思想

Qwen-VL 系列更偏向真实世界多模态助手. 它不仅关注图像问答, 还重点强化:

1. OCR.
2. Document QA.
3. Chart / Table Understanding.
4. Visual Grounding.
5. Video Understanding.
6. GUI / Agent 场景.
7. 中文和多语言多模态任务.

它的核心思想是:

**通过 dynamic resolution, visual token 压缩和多维位置编码, 在可控计算成本下保留高分辨率视觉细节.**

---

### 4.2 技术架构

Qwen-VL 系列整体结构仍可抽象为:

$$
\text{Image / Video}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Patch Merger / Projector}
\rightarrow
\text{Qwen LLM}
$$

和基础 LLaVA 路线相比, Qwen-VL 的重点不只是连接 Vision Encoder 和 LLM, 而是让模型处理更复杂的真实场景输入.

---

### 4.3 Dynamic Resolution

固定分辨率 VLM 会把所有图片 resize 到同一尺寸:

$$
I \rightarrow I_{\text{fixed}}
$$

这种方式实现简单, 但会带来问题:

1. 长图被压缩.
2. 文档小字变模糊.
3. 截图布局丢失.
4. 医疗影像细节丢失.

Dynamic Resolution 的做法是让不同尺寸的图片产生不同数量的 visual tokens:

$$
N_v = f(H,W)
$$

而不是固定为常数:

$$
N_v = C
$$

这样可以在高分辨率图像中保留更多细节, 对 OCR, document, chart 和 screenshot 类任务更重要.

### 4.3.1 代价

Dynamic Resolution 会带来:

1. batch 内序列长度不一致.
2. 推理调度更复杂.
3. [KV Cache](../Inference/KV_Cache.md) 更难估算.
4. 高分辨率输入下 prefill 成本更高.

---

### 4.4 Patch Merger

Dynamic Resolution 会产生大量 visual tokens, 因此 Qwen-VL 系列通常需要 token 压缩模块.

Patch Merger 的作用是:

$$
X_v \in \mathbb{R}^{N_v \times d_v}
\rightarrow
H_v \in \mathbb{R}^{N'_v \times d_{\text{LLM}}}
$$

其中:

$$
N'_v < N_v
$$

Patch Merger 的目标:

1. 降低 visual token 数.
2. 控制上下文长度.
3. 降低 prefill 和 KV Cache 成本.
4. 尽量保留空间细节.

简单理解:

**Dynamic Resolution 负责保留更多视觉细节, Patch Merger 负责控制进入 LLM 的 token 数量.**

---

### 4.5 M-RoPE

普通文本 LLM 的位置编码是一维的:

$$
t = 1,2,\dots,n
$$

图像是二维结构:

$$
(h,w)
$$

视频是三维结构:

$$
(t,h,w)
$$

Qwen2-VL 引入 M-RoPE, 目的是让模型同时处理文本, 图像和视频中的不同位置维度.

M-RoPE 主要解决:

1. 文本 token 的顺序关系.
2. 图像 token 的行列关系.
3. 视频 token 的时间关系.
4. 图文混合序列中的位置对齐.

它对 OCR, 表格理解, grounding 和视频理解都很重要.

---

### 4.6 Grounding

Grounding 是指模型不仅要识别图像内容, 还要能定位对应区域.

典型输出形式:

$$
(x_1,y_1,x_2,y_2)
$$

Grounding 要求模型具备:

1. 物体识别能力.
2. 空间位置理解能力.
3. 坐标格式输出能力.
4. 区域标注数据训练.

在实际场景中, grounding 常用于:

1. 指出图片中某个目标的位置.
2. 文档问答中定位证据区域.
3. GUI Agent 中定位按钮或控件.
4. 医疗影像中定位可疑病灶.

相关评估见 [Evaluation](./Evaluation.md).

---

### 4.7 Qwen2.5-VL 和 Qwen3-VL

Qwen2.5-VL 在 Qwen2-VL 基础上进一步强化真实世界任务:

1. OCR 和 Document Understanding.
2. Chart / Table Understanding.
3. Grounding 和坐标输出.
4. Video Understanding.
5. GUI / Agent 场景.
6. 结构化输出.

Qwen3-VL 继续强化:

1. 更长的 interleaved multimodal context.
2. 更强的视觉细节注入.
3. 更统一的图像, 文本和视频位置处理.
4. 更复杂的多模态推理任务.

可以按下面关系理解:

$$
\text{Qwen2-VL}
\rightarrow
\text{Dynamic Resolution + M-RoPE}
\rightarrow
\text{Qwen2.5-VL}
\rightarrow
\text{Real-World Task Enhancement}
\rightarrow
\text{Qwen3-VL}
\rightarrow
\text{Long Multimodal Context + Stronger Reasoning}
$$

---

### 4.8 特点和局限

Qwen-VL 系列的优点:

1. OCR, document, chart 能力强.
2. Grounding 和坐标输出能力较强.
3. 支持图像和视频任务.
4. 中文和多语言场景友好.
5. 更接近通用多模态助手.

Qwen-VL 系列的局限:

1. 高分辨率输入下 visual token 成本高.
2. dynamic resolution 增加推理调度复杂度.
3. 坐标和结构化输出依赖训练数据质量.
4. 服务端 batching 和显存管理难度更高.

---

## 5. InternVL 系列

### 5.1 核心思想

InternVL 系列的重点是:

**通过更强的视觉基础模型, 高分辨率处理方式和系统化训练配方, 提升开源 VLM 的综合能力.**

它和 LLaVA 的主要区别不是公式不同, 而是整体能力构建方式不同:

1. 更强 Vision Encoder.
2. 更系统的 dynamic high resolution.
3. 更多阶段的图文对齐和指令训练.
4. 更强 benchmark 综合能力.
5. 引入多模态偏好优化和 test-time scaling.

---

### 5.2 技术架构

InternVL 系列可以抽象为:

$$
\text{Image}
\rightarrow
\text{InternViT / Vision Foundation Model}
\rightarrow
\text{MLP / Token Processor}
\rightarrow
\text{LLM}
$$

与 LLaVA 相比, InternVL 更强调 Vision Encoder 本身的能力. 如果 Vision Encoder 在高分辨率, 细粒度识别和视觉语义上更强, 后续 Projector 和 LLM 才能获得更好的视觉输入.

---

### 5.3 Dynamic High Resolution

InternVL 系列常使用 dynamic high resolution.

基本流程:

1. 根据图片宽高比选择合适的 tile 布局.
2. 将大图切成多个固定尺寸 tile.
3. 可加入一张 global thumbnail.
4. 每个 tile 经过 Vision Encoder.
5. 将视觉 token 送入 LLM.

其中:

- tile 负责保留局部细节.
- global thumbnail 负责保留整体布局.

这类设计适合:

1. 文档截图.
2. 图表理解.
3. OCR.
4. 医学影像.
5. 小目标识别.

局限是 tile 数越多, visual token 越多, 推理成本越高.

---

### 5.4 强 Vision Foundation Model

LLaVA 通常复用 CLIP 或 OpenCLIP 作为 Vision Encoder. InternVL 系列更强调训练和扩展自己的视觉基础模型.

强 Vision Encoder 的价值在于:

1. 提升细粒度识别.
2. 提升 OCR 和文档理解.
3. 提升 dense perception.
4. 降低领域迁移难度.
5. 为 Projector 提供更好的视觉特征.

如果 Vision Encoder 已经丢失细节, 后续 LLM 很难通过语言推理恢复这些信息.

---

### 5.5 V2PE

InternVL3 引入 V2PE 这类视觉位置编码设计, 主要面向 dynamic high resolution 场景.

问题在于:

1. 图像 tile 数量不固定.
2. tile 之间存在全局空间关系.
3. tile 内部 patch 也有局部位置.
4. global thumbnail 和 local tile 需要共同建模.

V2PE 的目标是让视觉 token 在动态高分辨率输入下拥有更合适的位置表示.

它和 Qwen-VL 中 M-RoPE 的区别:

|对比项|M-RoPE|V2PE|
|---|---|---|
|代表路线|Qwen-VL 系列|InternVL3|
|核心目的|统一 text, image, video 的多维位置|增强动态高分辨率视觉 token 的位置表达|
|重点场景|图文视频混合输入|高分辨率 tile 输入|

---

### 5.6 原生多模态预训练

早期很多 VLM 是先训练 text-only LLM, 再接入视觉模块:

$$
\text{Text-only LLM}
\rightarrow
\text{Add Vision Encoder}
\rightarrow
\text{Multimodal Tuning}
$$

InternVL3 更强调 native multimodal pretraining, 即在更早阶段让模型接触图文数据.

这种方式的优势:

1. 视觉和语言对齐更早发生.
2. 多模态推理更自然.
3. 后续 instruction tuning 压力更小.
4. 有利于提升综合 benchmark 能力.

代价是训练成本更高, 数据清洗更复杂, 对分布式训练要求更高. 相关工程框架可参考 [Megatron](../Framework/Megatron.md) 和 [DeepSpeed](../Framework/DeepSpeed.md).

---

### 5.7 MPO 和 Test-Time Scaling

InternVL3 中还强调多模态偏好优化, 可以理解为将 [DPO](../Align/DPO.md), [RLHF](../Align/RLHF.md) 一类思想扩展到多模态任务.

目标是:

1. 减少多模态幻觉.
2. 提高 OCR, grounding, reasoning 的稳定性.
3. 让模型更偏好高质量回答.
4. 让格式化输出更可靠.

Test-Time Scaling 是在推理阶段增加计算来换取更好效果, 常见方式包括:

1. 多次采样.
2. 多候选 rerank.
3. 更长 [CoT](../Inference/CoT.md).
4. 使用 verifier 或规则检查.

适合可验证任务:

- OCR exact match.
- Chart QA.
- Grounding box.
- Medical multiple choice.

---

### 5.8 InternVL 和 Qwen-VL 的区别

|对比项|Qwen-VL 系列|InternVL 系列|
|---|---|---|
|主要定位|真实世界多模态助手|强视觉塔和系统训练路线|
|重点能力|OCR, document, grounding, video, GUI|高分辨率视觉理解, benchmark 综合能力|
|结构关键词|dynamic resolution, patch merger, M-RoPE|dynamic high resolution, strong ViT, V2PE|
|训练关键词|多任务真实场景数据|原生多模态预训练, MPO, test-time scaling|
|理解重点|应用能力覆盖广|视觉底座和训练配方强|

---

## 6. BLIP-2

### 6.1 核心思想

[BLIP-2](https://arxiv.org/abs/2301.12597) 主要解决:

**如何低成本连接冻结的 Vision Encoder 和冻结的 LLM.**

它的核心模块是 Q-Former.

---

### 6.2 技术架构

BLIP-2 的结构为:

$$
\text{Frozen Vision Encoder}
\rightarrow
\text{Q-Former}
\rightarrow
\text{Frozen LLM}
$$

Q-Former 中有一组 learnable query tokens:

$$
Q = \{q_1,q_2,\dots,q_M\}
$$

这些 query tokens 通过 cross-attention 从视觉特征中提取信息:

$$
Q' = \text{CrossAttention}(Q,X_v)
$$

最后只将 $M$ 个 query outputs 送入 LLM.

---

### 6.3 具体实现

假设图像有 576 个 patch tokens:

$$
X_v \in \mathbb{R}^{576 \times d_v}
$$

Q-Former 不直接把 576 个 token 全部交给 LLM, 而是用少量 query tokens 提取与语言相关的信息:

$$
Q' \in \mathbb{R}^{M \times d_q}
$$

通常:

$$
M \ll 576
$$

这种方式本质上是一种 **query-based visual token compression**.

---

### 6.4 特点和局限

BLIP-2 的优点:

1. 参数效率高.
2. 可以冻结 Vision Encoder 和 LLM.
3. 训练成本相对较低.
4. Q-Former 可以压缩视觉 token.

BLIP-2 的局限:

1. query 数固定, 可能丢失细节.
2. OCR, document 等 dense task 不一定强.
3. Q-Former 结构比 MLP Projector 复杂.
4. 指令遵循能力需要额外数据增强.

---

### 6.5 BLIP-2 和 LLaVA 的区别

|对比项|BLIP-2|LLaVA|
|---|---|---|
|连接模块|Q-Former|MLP Projector|
|视觉 token 处理|先用 query 压缩|patch features 投影后直接进入 LLM|
|训练重点|连接冻结模型|visual instruction tuning|
|优点|参数效率高|简单, 工程友好|
|局限|容易丢细节|visual token 成本较高|

---

## 7. Flamingo

### 7.1 核心思想

[Flamingo](https://arxiv.org/abs/2204.14198) 主要解决:

**如何让冻结 LLM 处理图文交错的多模态上下文.**

它的关键模块包括:

1. Perceiver Resampler.
2. Gated Cross-Attention.
3. Interleaved image-text 输入格式.

---

### 7.2 技术架构

视觉侧:

$$
\text{Image Features}
\rightarrow
\text{Perceiver Resampler}
\rightarrow
\text{Visual Latents}
$$

语言侧:

$$
H_t' = \text{GatedCrossAttention}(H_t,H_v)
$$

也就是说, Flamingo 不是只在 LLM 输入端拼接 visual tokens, 而是在 LLM 中间层通过 cross-attention 读取视觉信息.

---

### 7.3 Perceiver Resampler

Perceiver Resampler 将不同长度的视觉特征压缩成固定数量的 visual latents:

$$
X_v \rightarrow Z
$$

其中 $Z$ 的长度固定.

作用:

1. 控制 visual token 数量.
2. 支持多图输入.
3. 适合 interleaved image-text 场景.
4. 降低不同图像长度不一致带来的复杂性.

---

### 7.4 Gated Cross-Attention

Flamingo 在 LLM 中插入 gated cross-attention layer.

Cross-attention 让文本 hidden states 读取视觉 latents:

$$
H_t' = \text{CrossAttention}(H_t,H_v)
$$

Gate 用于控制视觉信息注入强度, 避免训练初期破坏已经预训练好的 LLM 表示.

---

### 7.5 Interleaved Image-Text

Flamingo 支持图像和文本交错输入:

```text
<image_1> Question 1 Answer 1
<image_2> Question 2 Answer 2
<image_3> Question 3
```

这种格式适合:

1. 多图上下文.
2. multimodal few-shot learning.
3. 图文交错推理.
4. 多轮视觉问答.

---

### 7.6 Flamingo 和 LLaVA 的区别

|对比项|Flamingo|LLaVA|
|---|---|---|
|视觉注入位置|LLM 中间层 cross-attention|LLM 输入侧 visual tokens|
|视觉压缩|Perceiver Resampler|通常使用 MLP Projector|
|多图交错|天然支持|后续版本增强|
|工程复杂度|较高|较低|
|是否修改 LLM 结构|是|通常不需要大改|

---

## 8. CLIP / SigLIP / BLIP

### 8.1 CLIP

[CLIP](https://arxiv.org/abs/2103.00020) 使用图像编码器和文本编码器进行对比学习.

结构:

$$
\text{Image Encoder}(I) \rightarrow z_I
$$

$$
\text{Text Encoder}(T) \rightarrow z_T
$$

训练目标是让匹配图文对相似度更高:

$$
\text{sim}(z_I,z_T) \uparrow
$$

不匹配图文对相似度更低:

$$
\text{sim}(z_I,z_T) \downarrow
$$

CLIP 的价值:

1. 提供强图文对齐表示.
2. 支持 zero-shot classification.
3. 常作为 VLM 的 Vision Encoder.

CLIP 的局限:

1. 本身不是生成式聊天模型.
2. 全局表征对细粒度 OCR 不够强.
3. 高分辨率和专业领域需要额外适配.

---

### 8.2 SigLIP

[SigLIP](https://arxiv.org/abs/2303.15343) 是对 CLIP 训练目标的改进.

CLIP 通常使用 softmax contrastive loss, SigLIP 使用 sigmoid loss.

区别:

|对比项|CLIP|SigLIP|
|---|---|---|
|训练目标|softmax contrastive loss|sigmoid loss|
|batch 依赖|依赖 batch 内对比|更适合大规模分布式训练|
|任务定位|图文对齐|图文对齐|
|是否生成回答|否|否|

SigLIP 常作为现代 VLM 的 Vision Encoder 候选.

---

### 8.3 BLIP

BLIP 类 Encoder-Decoder 架构将图像编码后, 再用文本 decoder 生成语言.

结构:

$$
\text{Image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Text Decoder}
\rightarrow
\text{Caption / Answer}
$$

它比 CLIP 多了生成能力, 适合 caption 和 VQA. 但如果 decoder 不是强 LLM, 语言推理和指令遵循能力会受限.

---

## 9. Video VLM

Video VLM 可以看作 Image VLM 的扩展, 但核心难点不只是多输入几张图.

视频可以表示为:

$$
V = \{I_1,I_2,\dots,I_T\}
$$

每帧经过 Vision Encoder:

$$
I_t \rightarrow X_{v,t}
$$

再将多帧视觉特征压缩或拼接后送入 LLM:

$$
\{X_{v,1},X_{v,2},\dots,X_{v,T}\}
\rightarrow
\text{Video Tokens}
\rightarrow
\text{LLM}
$$

---

### 9.1 帧采样

视频不能把所有帧都送进 LLM, 因此需要帧采样.

常见方式:

1. **Uniform Sampling**: 均匀抽帧.
2. **Keyframe Sampling**: 选择关键帧.
3. **FPS Sampling**: 按固定帧率抽帧.
4. **Adaptive Sampling**: 根据内容变化动态抽帧.

不同采样方式的差异:

|方式|优点|缺点|
|---|---|---|
|Uniform Sampling|实现简单|可能错过关键动作|
|Keyframe Sampling|关注变化帧|需要额外检测策略|
|低 FPS|token 成本低|动作细节容易丢|
|高 FPS|时间细节更完整|visual token 数量大|

---

### 9.2 时间位置编码

图像需要二维位置:

$$
(h,w)
$$

视频还需要时间维度:

$$
(t,h,w)
$$

如果没有时间位置, 模型可能知道每帧有什么, 但无法准确判断:

1. 事件先后顺序.
2. 动作方向.
3. 状态变化.
4. 因果关系.

因此 Video VLM 通常需要 3D position encoding 或类似 M-RoPE 的多维位置编码.

---

### 9.3 Video Token Compression

视频 visual token 数大致为:

$$
N_v = T \times N_{\text{frame}}
$$

其中:

- $T$: 帧数.
- $N_{\text{frame}}$: 每帧 visual token 数.

如果帧数和分辨率都较高, visual tokens 会迅速增加. 因此 Video VLM 常用:

1. 减少采样帧数.
2. 合并 patch tokens.
3. temporal pooling.
4. 选择关键帧.
5. 使用 video-specific encoder.

Video VLM 的主要难点:

1. 空间细节.
2. 时间顺序.
3. 长视频压缩.
4. 计算和显存成本.

---

## 10. 关键结构对比

### 10.1 MLP Projector / Q-Former / Perceiver Resampler

|对比项|MLP Projector|Q-Former|Perceiver Resampler|
|---|---|---|---|
|代表模型|LLaVA|BLIP-2|Flamingo|
|核心方式|逐 patch 映射到 LLM hidden size|用 query tokens 读取视觉特征|用 latent tokens 压缩视觉特征|
|是否压缩 token|通常不明显压缩|明显压缩|明显压缩|
|优点|简单, 易复现|参数效率高|适合多图交错输入|
|缺点|visual token 多|可能丢细节|结构复杂|

---

### 10.2 AnyRes / Dynamic Resolution / Dynamic High Resolution

|概念|代表路线|核心作用|
|---|---|---|
|AnyRes|LLaVA-NeXT / OneVision|将任意分辨率图像切成 global view + local tiles|
|Dynamic Resolution|Qwen-VL 系列|根据图像尺寸动态产生不同数量 visual tokens|
|Dynamic High Resolution|InternVL 系列|动态选择高分辨率 tile 布局, 并保留全局图|

它们共同解决的问题是:

**固定低分辨率会丢失 OCR, 文档, 图表和小目标细节.**

---

### 10.3 M-RoPE / V2PE

|对比项|M-RoPE|V2PE|
|---|---|---|
|代表路线|Qwen-VL 系列|InternVL3|
|核心目标|统一文本, 图像, 视频的多维位置|增强动态高分辨率视觉 token 的位置表达|
|重点维度|1D text, 2D image, 3D video|tile layout 和 visual token position|

---

### 10.4 OCR / Grounding / Document Understanding

|能力|任务形式|关键依赖|
|---|---|---|
|OCR|识别图中文字|高分辨率, 小文字识别, 文本顺序|
|Grounding|定位目标区域|空间位置, 坐标输出, box 数据|
|Document Understanding|理解文档内容|OCR, layout, table, structure extraction|

三者关系:

1. OCR 负责读出内容.
2. Grounding 负责指出位置.
3. Document Understanding 负责结合版面和语义回答问题.

---

## 11. 总结

主流 VLM 架构可以按下面关系理解:

1. **CLIP / SigLIP**: 解决图文表征对齐, 常作为 Vision Encoder.
2. **BLIP**: 使用 Encoder-Decoder 直接生成图像描述或回答.
3. **BLIP-2**: 使用 Q-Former 压缩视觉信息, 低成本连接冻结 LLM.
4. **Flamingo**: 使用 Perceiver Resampler 和 gated cross-attention 处理图文交错输入.
5. **LLaVA**: 使用 CLIP Vision Encoder + MLP Projector + LLM, 形成简单开源 VLM 范式.
6. **LLaVA-NeXT / OneVision**: 在 LLaVA 基础上加强高分辨率, 多图和视频能力.
7. **Qwen-VL**: 通过 dynamic resolution, patch merger, M-RoPE 强化 OCR, document, grounding 和 video.
8. **InternVL**: 通过强 Vision Encoder, dynamic high resolution, V2PE 和系统训练配方提升综合能力.

当前主流 VLM 的核心竞争点不只是把图像接入 LLM, 而是在可控 visual token budget 下保留高分辨率细节, 空间位置, 多图关系和视频时间信息.
