# VLM Architecture

VLM 架构主要解决一个问题:

**视觉信息如何进入语言模型, 并和文本信息一起参与推理和生成.**

不同模型的差异通常体现在三个部分:

1. Vision Encoder 如何选.
2. Vision Feature 如何压缩和对齐.
3. LLM 如何接收 visual tokens.

---

## 1. Dual Encoder 架构

Dual Encoder 是最早被大规模验证的图文对齐范式, 代表模型是 [CLIP](https://arxiv.org/abs/2103.00020).

结构如下:

$$
\text{Image}
\rightarrow
\text{Image Encoder}
\rightarrow
z_I
$$

$$
\text{Text}
\rightarrow
\text{Text Encoder}
\rightarrow
z_T
$$

训练目标是让匹配的图文 pair embedding 更接近, 不匹配的更远.

### 1.1 特点

1. **适合检索和分类**: 例如 image-text retrieval, zero-shot classification.
2. **图文表示强**: CLIP vision encoder 常被后续 VLM 复用.
3. **不是生成式模型**: 它本身不能像 ChatGPT 一样生成长回答.
4. **训练依赖大规模图文对**.

### 1.2 适用场景

- 图文检索.
- 图片分类.
- 图像相似度.
- 给生成式 VLM 提供视觉塔.

### 1.3 局限性

Dual Encoder 通常只得到一个全局 embedding, 对细粒度 OCR, 图表, 多步推理能力较弱.

---

## 2. Encoder-Decoder 架构

Encoder-Decoder 架构把图像编码后, 再用文本 decoder 生成语言.

典型代表包括早期 caption model, BLIP 等.

结构如下:

$$
\text{Image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Text Decoder}
\rightarrow
\text{Caption / Answer}
$$

### 2.1 特点

1. **适合 caption 和 VQA**.
2. **生成能力比 Dual Encoder 更强**.
3. **可以做图文匹配, caption, VQA 多任务训练**.
4. **如果 decoder 不是强 LLM, 语言推理能力会受限**.

---

## 3. Frozen LLM + Visual Connector 架构

现代主流开源 VLM 大多采用这一类路线.

核心结构:

$$
\text{Vision Encoder}
\rightarrow
\text{Connector}
\rightarrow
\text{LLM}
$$

其中:

- Vision Encoder 负责提取视觉特征.
- Connector 负责把视觉特征转成 LLM hidden size.
- LLM 负责语言理解和生成.

这类模型的代表:

- BLIP-2.
- Flamingo.
- LLaVA.
- MiniGPT-4.
- Qwen-VL.
- InternVL.

---

## 4. BLIP-2 / Q-Former 架构

[BLIP-2](https://arxiv.org/abs/2301.12597) 的核心是 Q-Former.

结构如下:

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

这些 query tokens 通过 cross-attention 从图像 patch features 中提取和语言相关的信息.

### 4.1 特点

1. **压缩视觉 token**: 不把所有 patch 都送入 LLM, 而是用少量 query tokens 表示图像.
2. **参数效率较高**: 主要训练 Q-Former.
3. **适合冻结大模型**: vision encoder 和 LLM 都可以冻结.
4. **结构比简单 MLP projector 更复杂**.

### 4.2 适合场景

- 训练资源有限.
- 希望尽量复用冻结 LLM.
- 视觉信息需要压缩后接入 LLM.

---

## 5. Flamingo 架构

[Flamingo](https://arxiv.org/abs/2204.14198) 的特点是支持 interleaved image-text 输入.

它的核心模块包括:

- Vision Encoder.
- Perceiver Resampler.
- Gated Cross-Attention.
- Frozen LLM.

结构可以理解为:

$$
\text{Image Features}
\rightarrow
\text{Perceiver Resampler}
\rightarrow
\text{Visual Latents}
$$

然后 LLM 在若干层中通过 cross-attention 读取 visual latents.

### 5.1 特点

1. **支持多图多文本交错上下文**.
2. **few-shot 多模态能力强**.
3. **适合复杂上下文输入**.
4. **实现和训练成本较高**.

### 5.2 影响

Flamingo 对后续多图输入, interleaved multimodal context, visual resampler 等设计影响很大.

---

## 6. LLaVA 架构

[LLaVA](https://arxiv.org/abs/2304.08485) 是当前最常见的开源 VLM 基础范式之一.

结构如下:

$$
\text{Image}
\rightarrow
\text{CLIP Vision Encoder}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{LLM}
$$

它把投影后的视觉特征当作一串 visual tokens, 插入到文本 prompt 中:

$$
[\text{<image tokens>}; \text{text tokens}]
$$

### 6.1 特点

1. **架构简单**: CLIP + MLP + LLM.
2. **工程友好**: 训练和复现成本相对低.
3. **依赖 visual instruction tuning**.
4. **成为很多开源 VLM 的基础路线**.

### 6.2 训练阶段

1. **Feature Alignment**: 冻结 vision encoder 和 LLM, 训练 projector.
2. **Visual Instruction Tuning**: 训练模型按照图像和指令回答问题.

详见 [Training](./Training.md).

---

## 7. Qwen-VL / InternVL 架构趋势

Qwen-VL 和 InternVL 代表了更强的现代开源 VLM 路线.

它们通常强调:

1. **高分辨率输入**: 动态分辨率, AnyRes, tile-based image encoding.
2. **强 OCR 能力**: 适合文档, 表格, 截图.
3. **多图和视频能力**.
4. **视觉 grounding**: 输出坐标, 框选区域.
5. **更强中文多模态能力**.

### 7.1 LLaVA-NeXT / OneVision 路线

LLaVA 后续路线通常可以理解为:

**保持 Vision Encoder + Projector + LLM 的简单结构, 但通过更强数据, 更高分辨率和更多模态提升能力.**

相比早期 LLaVA, 这类模型更关注:

1. 更高分辨率输入.
2. 多图输入.
3. OCR 和文档理解.
4. 视频帧输入.
5. 更大规模 instruction tuning 数据.

它的意义在于说明:

**VLM 能力提升不一定来自复杂架构, 数据质量, 分辨率策略和训练配方同样关键.**

---

### 7.2 Qwen2-VL / Qwen2.5-VL 路线

Qwen2-VL / Qwen2.5-VL 的代表特点是更强的真实世界视觉理解和中文多模态能力.

主流知识点:

1. **Dynamic Resolution**: 不把所有图像强行缩放到固定尺寸, 而是根据图像大小动态产生 visual tokens.
2. **OCR / Document 能力强**: 对截图, 表格, 文档, 图中文字更友好.
3. **Grounding 能力**: 可以输出框坐标或区域相关答案.
4. **Video 理解**: 通过多帧输入扩展到视频场景.
5. **多语言能力**: 中文场景适配较好.

Qwen-VL 类模型的重点不是单个 projector 多复杂, 而是:

**高质量多模态数据 + 高分辨率视觉输入 + 强 LLM 基座 + 统一多模态训练配方.**

---

### 7.3 InternVL 路线

InternVL 系列可以看作是强调强视觉基础模型和强 LLM 对接的路线.

它通常关注:

1. 更强 vision foundation model.
2. 更强 image-text alignment.
3. 动态高分辨率.
4. 多阶段训练.
5. 多任务和多 benchmark 泛化.

InternVL 的思路和 LLaVA 类似, 都属于 Vision Encoder + Projector + LLM 的主流范式, 但更强调视觉塔规模, 数据配方和系统化训练.

---

### 7.4 Dynamic Resolution

固定分辨率会损失细节. 例如把一张医学影像或文档截图缩到 $224 \times 224$, 文字和细小病灶可能直接丢失.

Dynamic Resolution 的思路是:

1. 保留更高分辨率.
2. 将图像切成多个 tile.
3. 每个 tile 编码为 visual tokens.
4. 再送入 LLM.

优点是细节更强, 缺点是 visual token 数量增加, 推理成本上升.

---

## 8. Video VLM 架构

Video VLM 可以看作 Image VLM 的扩展.

基本流程:

1. 从视频中采样多帧.
2. 每帧通过 vision encoder.
3. 对帧特征进行压缩或融合.
4. 将 video tokens 送入 LLM.

常见处理方式:

- Uniform frame sampling.
- Key frame sampling.
- Temporal pooling.
- Frame-level visual tokens.
- Video projector.

主要难点:

1. 帧数增加导致 visual tokens 暴涨.
2. 需要建模时间顺序.
3. 视频标注数据成本高.
4. 推理延迟明显高于图片 VLM.

---

## 9. 架构对比

|架构|代表模型|核心模块|优点|局限|
|---|---|---|---|---|
|Dual Encoder|CLIP|Image Encoder + Text Encoder|检索强, 表征好|不能直接长文本生成|
|Encoder-Decoder|BLIP|Vision Encoder + Text Decoder|caption / VQA 直接|语言推理依赖 decoder|
|Q-Former|BLIP-2|Q-Former|压缩视觉 token, 参数效率高|结构较复杂|
|Cross-Attention|Flamingo|Perceiver + Gated Cross-Attention|多图交错上下文强|训练成本高|
|MLP Projector|LLaVA|Vision Encoder + MLP + LLM|简单, 主流, 易复现|细粒度能力依赖数据和分辨率|
|Dynamic Resolution|Qwen-VL / InternVL|高分辨率切图 + LLM|OCR, 文档, 细节强|visual tokens 多, 推理成本高|

一句话总结:

**当前最主流的 VLM 工程范式是 Vision Encoder + Projector + LLM, 其中 LLaVA 路线最简单, Qwen-VL / InternVL 路线更强调高分辨率和通用多模态能力.**
