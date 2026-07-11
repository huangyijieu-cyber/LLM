# Vision Encoder

Vision Encoder 是 VLM 中负责理解图像的部分. 它的任务是把原始图像转成一组视觉特征:

$$
\text{Image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Visual Features}
$$

这些视觉特征后续会通过 [Projector](./Projector.md) 映射到 LLM hidden space, 再和文本 token 一起送入 [Transformer](../basic/Transformer%20架构.md).

Vision Encoder 决定 VLM 能看到什么. 如果视觉塔没有保留文字, 小目标或病灶细节, 后续 [LLM](../basic/Transformer%20架构.md) 很难通过语言推理恢复这些信息.

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
5. 保留 OCR, grounding, medical image 等任务需要的细节.

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

### 2.2 特点

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

### 3.1 为什么 CLIP 常用于 VLM

1. **图文对齐强**: 视觉特征天然靠近语言语义.
2. **zero-shot 能力好**.
3. **开源生态成熟**.
4. **LLaVA 等模型直接复用 CLIP vision tower**.

### 3.2 局限性

1. **全局语义强, 细粒度 OCR 较弱**.
2. **固定分辨率可能丢失细节**.
3. **医学影像等专业域存在 domain gap**.
4. **只做对齐不等于会生成回答**.

---

## 4. SigLIP

[SigLIP](https://arxiv.org/abs/2303.15343) 是对 CLIP 对比学习目标的改进.

CLIP 通常使用 softmax contrastive loss, SigLIP 使用 sigmoid loss. 它将每个 image-text pair 当作独立二分类问题.

### 4.1 特点

1. **训练更易扩展到大 batch / 分布式场景**.
2. **图文表征质量强**.
3. **常被现代 VLM 用作 vision encoder 候选**.
4. **适合大规模图文对齐预训练**.

### 4.2 和 CLIP 的区别

|对比项|CLIP|SigLIP|
|---|---|---|
|训练目标|softmax contrastive loss|sigmoid loss|
|batch 依赖|依赖 batch 内多分类对比|每个 pair 独立二分类|
|扩展性|依赖大 batch|更适合分布式扩展|
|主要用途|图文对齐, VLM 视觉塔|图文对齐, VLM 视觉塔|

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

DINOv2 更偏视觉感知, CLIP / SigLIP 更偏图文语义对齐. 一些 VLM 会融合 CLIP 和 DINO 类特征, 试图同时获得:

1. 图文对齐能力.
2. 细粒度视觉感知能力.
3. dense perception 能力.
4. grounding 能力.

---

## 6. EVA-CLIP / InternViT / 其他视觉塔

EVA-CLIP, OpenCLIP, ConvNeXt-CLIP, InternViT 等也常作为 VLM 视觉塔.

这些模型的共同目标是:

1. 提供更强图像特征.
2. 支持更高分辨率.
3. 改善 OCR, grounding, fine-grained perception.
4. 降低特定领域的 domain gap.

其中 InternVL 系列更强调强 vision foundation model, 详见 [Architecture](./Architecture.md).

---

## 7. 输出特征类型

Vision Encoder 输出通常有几种形式.

### 7.1 CLS Feature

ViT 中常用一个 `[CLS]` token 表示整张图.

优点:

1. 表示简洁.
2. 适合分类和检索.
3. token 成本低.

缺点:

1. 信息压缩太强.
2. 不适合细粒度定位.
3. 不适合 OCR 和文档理解.

---

### 7.2 Patch Features

保留所有 patch token:

$$
X_v = \{v_1,v_2,\dots,v_N\}
$$

优点:

1. 细粒度信息更丰富.
2. 适合 grounding, OCR, 图表理解.
3. 适合高分辨率 tile 扩展.

缺点:

1. token 数多.
2. LLM 上下文成本高.
3. [KV Cache](../Inference/KV_Cache.md) 压力更大.

---

### 7.3 Multi-Scale Features

Multi-scale features 保留不同分辨率下的信息.

适合:

1. 小目标.
2. 医学影像.
3. 文档 OCR.
4. 遥感图像.
5. 病理切片.

缺点是实现更复杂, 推理成本更高.

---

## 8. 分辨率问题

VLM 中分辨率非常关键.

低分辨率:

1. 推理快.
2. visual tokens 少.
3. OCR 和细节容易丢失.

高分辨率:

1. OCR 和细粒度识别更好.
2. 医疗影像更有价值.
3. visual tokens 变多.
4. [KV Cache](../Inference/KV_Cache.md) 和 prefill 成本增加.

现代 VLM 常用:

1. AnyRes.
2. Dynamic Resolution.
3. Image tiling.
4. Multi-crop.
5. Global thumbnail + local tiles.

这些机制在 [Architecture](./Architecture.md) 中有更详细说明.

---

## 9. Vision Encoder 选择和任务能力

|任务|更依赖什么能力|Vision Encoder 关注点|
|---|---|---|
|通用 VQA|语义理解|图文对齐能力|
|OCR|小文字识别|高分辨率, patch features|
|Document QA|布局和文字|高分辨率, 2D 位置, tile|
|Grounding|区域定位|局部视觉特征, spatial feature|
|Video|帧级视觉特征|时间采样和多帧一致性|
|Medical VLM|细微异常识别|高分辨率, 专业域适配|

---

## 10. Vision Encoder 对比

|Vision Encoder|核心监督|优点|局限|常见用途|
|---|---|---|---|---|
|ViT|分类或自监督|结构统一, 易扩展|需大数据|基础视觉编码器|
|CLIP|图文对比学习|图文对齐强|细粒度和专业域弱|VLM 视觉塔|
|SigLIP|sigmoid 图文对比|扩展性好, 表征强|仍依赖图文数据|现代 VLM 视觉塔|
|DINOv2|自监督视觉学习|视觉细节强|语言对齐弱|感知增强|
|EVA-CLIP|大规模图文预训练|性能强|训练成本高|强 VLM 视觉塔|
|InternViT|视觉基础模型训练|高分辨率和综合能力强|训练和部署成本高|InternVL 系列|

---

## 11. 总结

Vision Encoder 的选择会直接影响 VLM 的感知上限.

1. CLIP / SigLIP 更偏图文语义对齐.
2. DINOv2 更偏视觉感知.
3. EVA-CLIP / InternViT 等强视觉塔更适合高性能 VLM.
4. OCR, document, medical image 等任务通常需要更高分辨率和更多局部特征.
5. 高分辨率会带来更多 visual tokens, 需要和 [Projector](./Projector.md), [Inference](./Inference.md) 一起考虑.
