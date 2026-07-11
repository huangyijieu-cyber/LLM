# Vision Encoder

Vision Encoder 是 VLM 的视觉感知模块, 负责把像素转换为一组视觉特征:

$$
I \in \mathbb{R}^{H \times W \times C}
\rightarrow
X_v \in \mathbb{R}^{N_v \times d_v}
$$

这些特征经过 [Projector](./Projector.md) 映射后进入 LLM. Vision Encoder 决定模型能够看到什么: 如果小字, 局部目标或病灶在视觉编码阶段已经丢失, 后续 LLM 很难通过语言推理恢复. 因此选择视觉塔时不能只比较分类性能, 还要同时考虑图文语义, 局部细节, 输入分辨率和 visual token 成本.

---

## 1. ViT 基础结构

[ViT](https://arxiv.org/abs/2010.11929) 是现代 VLM 最常用的视觉骨干. 它把图像切成 $P \times P$ 的 patch, 每个 patch 经过线性映射后作为一个 token 输入 Transformer Encoder. 对尺寸为 $H \times W$ 的图像, patch 数量为:

$$
N_v = \frac{H}{P} \times \frac{W}{P}
$$

patch embedding 可以写为:

$$
x_i = W_p \cdot \mathrm{patch}_i + e_i^{\mathrm{pos}}
$$

其中 $e_i^{\mathrm{pos}}$ 表示空间位置. Self-Attention 使每个 patch 能够读取其他区域的信息, 因而适合建模物体之间的全局关系. 代价是 token 数随图像面积增长, 高分辨率下 Attention 和后续 LLM prefill 都会变贵.

ViT 只是网络结构, 真正决定特征性质的是预训练目标. 分类监督更偏类别判别, 图文对比学习更偏语言语义, 自监督学习更偏通用视觉结构. VLM 中常见的 CLIP, SigLIP, DINOv2 和 InternViT 可以理解为不同预训练路线下的视觉基础模型.

---

## 2. 图文对齐视觉塔

### 2.1 CLIP

[CLIP](https://arxiv.org/abs/2103.00020) 使用 image encoder 和 text encoder 对图文对进行对比学习:

$$
I \rightarrow z_I, \qquad T \rightarrow z_T
$$

训练目标提高匹配图文的相似度, 降低不匹配图文的相似度. 这使视觉特征天然带有语言语义, 容易通过 Projector 接入 LLM. LLaVA 等经典架构因此直接复用 CLIP vision tower.

CLIP 的强项是全局语义, zero-shot 迁移和成熟生态. 它的局限也来自同一训练目标: 网页图文对往往描述显著对象, 对小文字, 细粒度空间关系和专业医学影像的监督不足. 固定低分辨率还会进一步损失 OCR 与小目标信息.

### 2.2 SigLIP

[SigLIP](https://arxiv.org/abs/2303.15343) 将每个 image-text pair 视为独立二分类问题, 使用 sigmoid loss, 而 CLIP 使用 batch 内 softmax contrastive loss. SigLIP 不要求所有 pair 共同组成一个多分类归一化项, 更便于扩展大规模分布式图文训练, 也常被现代 VLM 用作视觉塔.

|对比项|CLIP|SigLIP|
|---|---|---|
|训练目标|Batch 内 softmax contrastive loss|Pair-wise sigmoid loss|
|负样本关系|依赖 batch 内其他样本|每个 pair 独立判断|
|主要能力|图文语义对齐, zero-shot|可扩展的图文语义对齐|
|在 VLM 中的作用|经典视觉塔|现代强视觉塔候选|

CLIP 和 SigLIP 都只提供视觉表征, 并不会自动获得对话和长答案生成能力. 这部分仍需 [Training](./Training.md) 中的生成式预训练与 Multimodal Instruction Tuning.

---

## 3. 视觉自监督和强视觉基础模型

[DINOv2](https://arxiv.org/abs/2304.07193) 不依赖文本监督, 而是从图像本身学习稳定的视觉表示. 相比 CLIP 类模型, DINOv2 更擅长保留局部结构和 dense perception 信息, 对分类, 检测与分割的迁移能力较强; 但它的特征没有天然对齐语言空间, 接入 LLM 时需要更充分的对齐训练.

一些 VLM 会融合图文对齐特征与 DINO 类局部特征, 目的是同时获得语言语义和细粒度感知. 代价是多视觉塔会增加参数, 特征融合和推理开销.

EVA-CLIP, OpenCLIP 和 InternViT 等模型继续扩大视觉预训练规模, 分辨率或模型容量. InternVL 路线尤其强调强 Vision Foundation Model, 因为更好的视觉底座能够提高 OCR, grounding 和高分辨率理解的上限. 但视觉塔越大, 每个请求的固定 image encoding 成本也越高, 不能只看 benchmark 得分而忽略部署吞吐.

---

## 4. 输出特征的选择

ViT 可以输出一个全局 CLS feature, 全部 patch features, 或不同层与不同尺度的特征. 选择哪种输出决定了信息压缩发生在什么位置.

|特征形式|保留的信息|适合任务|主要代价|
|---|---|---|---|
|CLS Feature|整图全局语义|分类, 检索|局部细节损失大|
|Patch Features|每个 patch 的局部表示|VQA, OCR, Grounding|visual tokens 多|
|Multi-Layer Features|浅层细节和深层语义|细粒度识别, dense task|融合结构更复杂|
|Multi-Scale Features|不同分辨率下的目标|文档, 医疗, 遥感, 病理|显存和计算成本高|

生成式 VLM 通常使用 patch features, 因为 LLM 需要知道不同区域的内容. 只使用 CLS token 虽然便宜, 但无法可靠支持文字读取和区域定位. 一些模型还会拼接 Vision Encoder 多层特征, 让 Projector 同时接触局部纹理和高层语义.

---

## 5. 分辨率和位置

固定分辨率会把所有图片 resize 到相同尺寸, 实现和 batching 都比较简单, 但长图, 文档和高分辨率医学影像容易被压缩. AnyRes, Dynamic Resolution 和 Dynamic High Resolution 通过保留原始宽高比或切分 local tiles 改善细节, 具体见 [Architecture](./Architecture.md).

常见的 global thumbnail + local tiles 方案同时保留整体布局和局部细节. Global view 告诉模型各区域之间的关系, local tile 提供小字和小目标信息. 问题在于 tile 数量增加会近似线性增加视觉编码和 LLM 输入成本, 因而通常还要配合 Patch Merger, pooling 或 token pruning.

位置编码同样重要. 图像 patch 需要二维空间位置, 视频还需要时间维度. 当输入被切成多个 tile 时, 模型既要知道 patch 在 tile 内的位置, 也要知道 tile 在原图中的位置. M-RoPE 和 V2PE 等设计就是为复杂空间与时间位置建模服务.

---

## 6. 训练策略和选择原则

早期对齐阶段常冻结 Vision Encoder, 只训练 Projector, 这样训练稳定且不会破坏已有视觉表示. 在高质量多模态指令数据上, 可以解冻视觉塔后部若干层或进行全参数训练, 以适配 OCR, 医疗等新领域. 完全冻结的成本低但领域上限有限; 全量更新的上限高, 但更容易遗忘通用视觉能力, 也需要更小 learning rate 和更谨慎的数据配比.

|任务|优先关注的视觉能力|常见选择方向|
|---|---|---|
|通用 VQA|图文语义和场景理解|CLIP / SigLIP 类视觉塔|
|OCR / Document|高分辨率和 patch 细节|强 ViT + Dynamic Resolution|
|Grounding|局部结构和空间位置|Patch / multi-layer features, DINO 类增强|
|Video|帧级感知和时间位置|高效视觉塔 + frame compression|
|Medical VLM|细微异常和领域适配|高分辨率视觉塔 + 医疗继续训练|

---

## 7. 总结

Vision Encoder 决定 VLM 的感知上限. CLIP 和 SigLIP 强在图文语义对齐, DINOv2 强在视觉结构, EVA-CLIP 与 InternViT 等强视觉塔强调规模和高分辨率能力. 实际选择时需要同时考虑输出特征, 输入分辨率, 是否进行领域微调, 以及 visual tokens 对 [Inference](./Inference.md) 的影响. 视觉信息保留得越多, 后续推理空间越大, 但 Projector 和 LLM 需要承担的计算成本也越高.
