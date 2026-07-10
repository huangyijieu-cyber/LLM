# Vision Encoder

Vision Encoder 是 VLM 中负责理解图像的部分. 它的任务是把原始图像转成一组视觉特征:

$$
\text{Image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Visual Features}
$$

这些视觉特征后续会通过 [Projector](./Projector.md) 映射到 LLM 的 hidden space, 再和文本 token 一起送入 [Transformer](../basic/Transformer%20架构.md).

---

## 1. 为什么需要 Vision Encoder

LLM 只能直接处理离散 token 或 embedding, 不能直接处理像素矩阵.

图片通常是:

$$
I \in \mathbb{R}^{H \times W \times C}
$$

而 LLM 期望输入是:

$$
X \in \mathbb{R}^{N \times d_{\text{model}}}
$$

因此需要 Vision Encoder 完成:

1. 把像素切成 patch 或 region.
2. 提取图像语义特征.
3. 输出一组视觉 embedding.
4. 为后续图文对齐提供基础.

---

## 2. ViT

[ViT](https://arxiv.org/abs/2010.11929) 全称是 Vision Transformer.

它的核心思想是:

**把图片切成 patch, 把每个 patch 当成一个 token, 然后使用 Transformer Encoder 处理.**

### 2.1 Patch Embedding

假设输入图片大小为:

$$
H \times W
$$

patch 大小为:

$$
P \times P
$$

那么 patch 数量为:

$$
N = \frac{H}{P} \times \frac{W}{P}
$$

每个 patch 被展平后通过线性层映射成 embedding:

$$
x_i = W_p \cdot \text{patch}_i
$$

这些 patch embeddings 会送入 Transformer Encoder.

### 2.2 ViT 的特点

1. **结构和 Transformer 统一**: 方便和 LLM 架构对齐.
2. **全局建模能力强**: Self-Attention 可以建模远距离区域关系.
3. **需要大量数据预训练**: 小数据上不如 CNN 稳.
4. **高分辨率成本高**: patch 数量随图像面积增长.

ViT 是现代 VLM vision encoder 的基础.

---

## 3. CLIP Vision Encoder

[CLIP](https://arxiv.org/abs/2103.00020) 使用大规模图文对进行对比学习.

CLIP 中有两个 encoder:

$$
\text{Image Encoder}: I \rightarrow z_I
$$

$$
\text{Text Encoder}: T \rightarrow z_T
$$

训练目标是让匹配图文对相似度更高:

$$
\text{sim}(z_I,z_T) \uparrow
$$

不匹配图文对相似度更低:

$$
\text{sim}(z_I,z_T) \downarrow
$$

### 3.1 为什么 CLIP 常用于 VLM

1. **图文对齐强**: 视觉特征天然靠近语言语义.
2. **zero-shot 能力好**.
3. **开源生态成熟**.
4. **LLaVA 等模型直接复用 CLIP vision tower**.

### 3.2 局限性

1. **全局语义强, 细粒度 OCR 较弱**.
2. **固定分辨率可能丢失细节**.
3. **医学影像等专业域存在 domain gap**.

---

## 4. SigLIP

[SigLIP](https://arxiv.org/abs/2303.15343) 是对 CLIP 对比学习目标的改进.

CLIP 通常使用 softmax contrastive loss, 而 SigLIP 使用 sigmoid loss.

直观理解:

**CLIP 把一个 batch 内的图文 pair 当成多分类问题, SigLIP 把每个 image-text pair 当成独立二分类问题.**

### 4.1 特点

1. **训练更易扩展到大 batch / 分布式场景**.
2. **图文表征质量强**.
3. **常被现代 VLM 用作 vision encoder 候选**.

### 4.2 适合场景

- 通用图文对齐.
- VLM 视觉塔.
- 多语言图文预训练.

---

## 5. DINOv2

[DINOv2](https://arxiv.org/abs/2304.07193) 是自监督视觉表示模型.

和 CLIP 不同, DINOv2 不依赖图文对齐数据, 而是从图像本身学习强视觉特征.

### 5.1 特点

1. **视觉表征强**: 对分类, 分割, 检测等视觉任务泛化好.
2. **不依赖文本监督**.
3. **局部视觉结构更强**.
4. **图文对齐不如 CLIP 直接**.

### 5.2 在 VLM 中的价值

DINOv2 更偏视觉理解, CLIP 更偏图文语义对齐. 一些 VLM 会融合 CLIP 和 DINO 类特征, 试图同时获得:

- 图文对齐能力.
- 细粒度视觉感知能力.

---

## 6. EVA-CLIP 和其他视觉塔

EVA-CLIP, OpenCLIP, ConvNeXt-CLIP 等也常作为 VLM 视觉塔.

这些模型的共同目标是:

1. 提供更强图像特征.
2. 支持更高分辨率.
3. 改善 OCR, grounding, fine-grained perception.

它们不是每个都需要深入掌握, 面试中更重要的是知道:

**VLM 的视觉塔选择会直接影响感知能力, OCR 能力, 细粒度识别和领域迁移能力.**

---

## 7. 输出特征类型

Vision Encoder 输出通常有几种形式.

### 7.1 CLS Feature

ViT 中常用一个 `[CLS]` token 表示整张图.

优点:

- 简洁.
- 适合分类和检索.

缺点:

- 信息压缩太强.
- 不适合细粒度定位和 OCR.

---

### 7.2 Patch Features

保留所有 patch token:

$$
X_v = \{v_1,v_2,\dots,v_N\}
$$

优点:

- 细粒度信息更丰富.
- 适合 grounding, OCR, 图表理解.

缺点:

- token 数多.
- LLM 上下文成本高.

---

### 7.3 Multi-scale Features

多尺度特征保留不同分辨率下的信息.

适合:

- 小目标.
- 医学影像.
- 文档 OCR.
- 遥感图像.

缺点是实现更复杂, 推理成本更高.

---

## 8. 分辨率问题

VLM 中分辨率非常关键.

低分辨率:

- 推理快.
- token 少.
- 但细节丢失.

高分辨率:

- OCR 和细粒度识别更好.
- 医疗影像更有价值.
- 但 visual tokens 变多.
- [KV Cache](../Inference/KV_Cache.md) 和 prefill 成本增加.

现代 VLM 常用:

- AnyRes.
- Dynamic Resolution.
- Image tiling.
- Multi-crop.

---

## 9. Vision Encoder 对比

|Vision Encoder|核心监督|优点|局限|常见用途|
|---|---|---|---|---|
|ViT|分类或自监督|结构统一, 易扩展|需大数据|基础视觉编码器|
|CLIP|图文对比学习|图文对齐强|细粒度和专业域弱|VLM 视觉塔|
|SigLIP|sigmoid 图文对比|扩展性好, 表征强|仍依赖图文数据|现代 VLM 视觉塔|
|DINOv2|自监督视觉学习|视觉细节强|语言对齐弱|感知增强|
|EVA-CLIP|大规模图文预训练|性能强|训练成本高|强 VLM 视觉塔|

一句话总结:

**CLIP / SigLIP 更偏图文语义对齐, DINOv2 更偏视觉感知, 现代强 VLM 往往需要同时兼顾语义对齐和细粒度视觉理解.**
