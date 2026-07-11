# VLM Architecture

VLM Architecture 解决的是视觉信息如何进入语言模型并参与生成. 大多数现代 VLM 都可以抽象为:

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

[Vision Encoder](./Vision_Encoder.md) 提取视觉特征, [Projector](./Projector.md) 完成空间对齐和 token 压缩, LLM 则联合处理 visual tokens 与 text tokens. 不同架构表面上使用不同模块, 实际差异集中在四个问题: 图像以什么分辨率编码, 视觉 token 如何压缩, 视觉信息在 LLM 的什么位置注入, 空间与时间位置如何表示.

---

## 1. 架构中的核心变量

对输入图像 $I \in \mathbb{R}^{H \times W \times C}$, Vision Encoder 输出:

$$
X_v = \mathrm{VisionEncoder}(I) \in \mathbb{R}^{N_v \times d_v}
$$

Connector 再将其映射为:

$$
H_v = f_{\mathrm{proj}}(X_v) \in \mathbb{R}^{N'_v \times d_{\mathrm{LLM}}}
$$

$d_v \rightarrow d_{\mathrm{LLM}}$ 是特征维度对齐, $N_v \rightarrow N'_v$ 则决定视觉信息压缩程度. 对 patch size 为 $P \times P$ 的图像, 原始 patch 数近似为:

$$
N_v \approx \frac{H \times W}{P^2}
$$

分辨率越高, OCR, 文档和小目标细节越容易保留, 但 LLM prefill, [KV Cache](../Inference/KV_Cache.md) 和 batch 调度成本也越高. 因此 visual token budget 是理解 VLM 架构的主线: AnyRes 和 Dynamic Resolution 负责获取更多细节, Q-Former, Perceiver Resampler 和 Patch Merger 负责控制这些细节以多少 token 进入 LLM.

视觉信息主要有两种注入方式. LLaVA, Qwen-VL 和 InternVL 将 visual embeddings 作为 LLM 输入 token, 通常替换 prompt 中的 `<image>`; Flamingo 则在 LLM 中间层加入 cross-attention, 让文本 hidden states 读取独立视觉 memory. 前者结构简单并复用 decoder-only LLM, 后者融合更灵活但会修改模型与 serving 实现.

---

## 2. 主流路线总览

|路线|核心结构|重点能力|理解重点|
|---|---|---|---|
|LLaVA|CLIP Vision Encoder + MLP Projector + LLM|通用 VQA, 指令遵循, 高分辨率扩展|简单 Connector 与 Visual Instruction Tuning|
|Qwen-VL|Vision Encoder + Patch Merger + Qwen LLM|OCR, Document, Grounding, Video, GUI|Dynamic Resolution 与 M-RoPE|
|InternVL|强 Vision Foundation Model + token processor + LLM|高分辨率视觉理解和综合 benchmark|Dynamic High Resolution, V2PE, 系统训练配方|
|BLIP-2|Frozen Vision Encoder + Q-Former + Frozen LLM|低成本连接冻结模型|Query-based token compression|
|Flamingo|Perceiver Resampler + Gated Cross-Attention|多图交错与 multimodal few-shot|固定 visual latents 和中间层注入|
|CLIP / SigLIP|Image Encoder + Text Encoder|图文表征对齐|常作为视觉塔, 本身不负责对话生成|

当前需要重点理解 LLaVA, Qwen-VL 和 InternVL. 它们都采用强 Vision Encoder + decoder-only LLM, 但在高分辨率策略, token 压缩, 位置编码和训练配方上走出了不同路线. BLIP-2 与 Flamingo 虽不是当前最常见的通用助手结构, 但 Q-Former, Perceiver Resampler 和 gated cross-attention 仍是重要的 Connector 思想.

---

## 3. LLaVA 系列

### 3.1 Patch Features + MLP Projector

[LLaVA](https://arxiv.org/abs/2304.08485) 的核心结构非常直接:

$$
\text{Image}
\rightarrow
\text{CLIP Vision Encoder}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{LLM}
$$

它通常保留 CLIP 的 patch features, 而不是只使用整图 CLS feature. 若只使用 $z_I \in \mathbb{R}^{d}$ 的全局表示, 局部信息被压进单个向量, 很难支持区域问答和小目标识别. Patch features 则表示为 $X_v=\{v_1,v_2,\dots,v_N\}$, 每个 patch 经过 MLP:

$$
H_v = W_2\sigma(W_1X_v)
$$

投影后作为 visual tokens 插入 LLM. MLP 主要完成 hidden size 与语义空间映射, 通常不显著压缩 token. 这种设计不改 LLM block, 易于复现, 也可以直接复用 causal language modeling 训练和推理代码. 代价是视觉 token 数基本由输入 patch 数决定.

### 3.2 两阶段训练

LLaVA 的第一个关键技巧是把接口对齐与任务学习分开. Feature Alignment 阶段冻结 Vision Encoder 和 LLM, 只训练 MLP Projector, 使 visual embeddings 先进入 LLM 可理解的空间. Visual Instruction Tuning 阶段再使用多模态指令数据训练模型按图回答:

$$
L_{\mathrm{SFT}} = -\sum_{t=1}^{T}\log P(y_t \mid I,x,y_{1:t-1})
$$

两阶段训练降低了直接联合优化的难度. 第一阶段解决 "视觉 token 表示什么", 第二阶段解决 "如何根据视觉 token 执行指令". LLaVA 的影响力主要来自这套简单但有效的工程范式, 而不是复杂的新模块.

### 3.3 LLaVA-NeXT 和 OneVision

早期 LLaVA 使用固定分辨率, 文档小字和细粒度目标容易在 resize 时丢失. [LLaVA-1.5](https://arxiv.org/abs/2310.03744) 改善视觉指令训练与数据配方, LLaVA-NeXT / [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) 则进一步引入 AnyRes 和统一图像 / 多图 / 视频能力.

AnyRes 通常将输入表示为一张 global view 与若干 local tiles:

$$
\text{Image}
\rightarrow
\text{Global View} + \text{Local Tiles}
\rightarrow
\text{Vision Encoder}
$$

Global view 保留整体布局, local tiles 保留 OCR, 图表和小目标细节. OneVision 还加强 interleaved multi-image 输入, 并将图像训练获得的能力迁移到多帧视频. 这些扩展改善了高分辨率能力, 但 MLP 路线仍会把大量 tile tokens 交给 LLM, 因而推理成本明显增加.

LLaVA 的优势是结构清晰, 不大改 LLM, 适合作为 VLM baseline 和二次开发起点. 它的上限高度依赖 Vision Encoder, 分辨率策略和指令数据; OCR, Grounding, Video 与 GUI 等能力不是基础 LLaVA 自动具备的, 需要相应架构和数据增强.

---

## 4. Qwen-VL 系列

### 4.1 Dynamic Resolution 和 Patch Merger

[Qwen-VL](https://arxiv.org/abs/2308.12966) 系列面向真实世界多模态助手, 重点覆盖 OCR, Document QA, Chart, Grounding, Video 和 GUI. [Qwen2-VL](https://arxiv.org/abs/2409.12191) 的重要变化之一是 Dynamic Resolution: 不再把所有图片强制 resize 到同一个固定 token 数, 而是根据原始尺寸和宽高比产生可变数量视觉 token:

$$
N_v = f(H,W)
$$

长图与文档因此可以保留更多小字和布局. 但原始 patch 全部进入 LLM 会非常昂贵, 所以视觉侧使用 Patch Merger 将相邻 patch 聚合:

$$
X_v \in \mathbb{R}^{N_v \times d_v}
\rightarrow
H_v \in \mathbb{R}^{N'_v \times d_{\mathrm{LLM}}},
\qquad N'_v < N_v
$$

两者是一组配套设计: Dynamic Resolution 增加有效视觉信息, Patch Merger 控制最终 token budget. 与 Q-Former 固定少量 query 的全局摘要相比, spatial merging 更强调保留局部二维结构, 因而适合 OCR 和文档任务.

### 4.2 M-RoPE

文本位置是一维顺序 $t$, 图像位置是二维坐标 $(h,w)$, 视频还包含时间维度 $(t,h,w)$. Qwen2-VL 的 M-RoPE 将位置表示扩展到多维结构, 使同一个 LLM 能够处理文本顺序, 图像行列和视频时间.

M-RoPE 的价值不只是支持更多输入类型. OCR 需要知道文字阅读顺序, Grounding 需要理解区域位置, Chart 需要保留二维关系, Video 需要区分事件先后. 如果 visual tokens 只有普通的一维拼接位置, 模型仍能学习部分规律, 但不同 modality 的结构先验表达较弱.

### 4.3 Grounding 和真实任务

Qwen-VL 系列将 Grounding 作为重要能力, 可以根据文字返回 box 或根据区域回答. 一个 box 通常表示为 $(x_1,y_1,x_2,y_2)$, 训练和推理必须使用一致的坐标归一化与 resize 规则. Grounding 可用于文档证据定位, GUI 控件选择和医学病灶提示, 相关评测见 [Evaluation](./Evaluation.md).

[Qwen2.5-VL](https://arxiv.org/abs/2502.13923) 在此基础上继续强化 OCR, Document, Chart, Grounding, Video, GUI Agent 和结构化输出. [Qwen3-VL](https://arxiv.org/abs/2511.21631) 则进一步面向更长的 interleaved multimodal context 与复杂视觉推理. 这些版本差异很多, 但理解主线仍是: **保留真实输入的分辨率与结构, 再通过 token merger 和多维位置编码把成本控制在可用范围内**.

Qwen-VL 的优势是能力覆盖广, 中文与多语言任务友好, 接近通用多模态助手. 局限是可变 visual sequence 使 batching, KV Cache 预算和高并发 serving 更复杂, 坐标与结构化输出也高度依赖数据格式一致性.

---

## 5. InternVL 系列

### 5.1 强 Vision Foundation Model

[InternVL](https://arxiv.org/abs/2312.14238) 系列不仅把现有视觉塔接入 LLM, 还强调扩大和训练视觉基础模型本身. 其结构可以抽象为:

$$
\text{Image}
\rightarrow
\text{InternViT / Vision Foundation Model}
\rightarrow
\text{MLP / Token Processor}
\rightarrow
\text{LLM}
$$

强视觉底座提供更好的细粒度识别, dense perception 和高分辨率特征. 这与单纯增大 LLM 不同: 如果视觉塔没有编码某个小目标, 更强语言模型也只能依赖先验猜测. InternVL 因而把 Vision Encoder 视为综合能力的重要上限.

### 5.2 Dynamic High Resolution

[InternVL1.5](https://arxiv.org/abs/2404.16821) 等版本使用 Dynamic High Resolution. 系统根据宽高比选择 tile 网格, 将大图切成若干固定尺寸 local tiles, 并加入 global thumbnail. Local tiles 保存文档小字, 图表和病灶细节, thumbnail 提供全局布局. 这与 LLaVA AnyRes 的目标相似, 但 InternVL 更强调把高分辨率处理与强视觉塔和整体训练配方结合.

Tile 数不能无限增加. 更多 tile 会同时增加 Vision Encoder forward 和 LLM visual tokens, 因而训练与推理都需要设置最大 tile 数, 按图像复杂度分配预算. 在文档, 医疗和小目标任务中提高 tile 上限有价值, 普通场景图则未必需要相同成本.

### 5.3 V2PE 和 Native Multimodal Pretraining

[InternVL3](https://arxiv.org/abs/2504.10479) 引入 V2PE 等视觉位置设计, 面向动态高分辨率下不固定的 tile 数与 visual token 布局. 它需要同时描述 tile 在原图中的全局位置, patch 在 tile 内的局部位置, 以及 thumbnail 与 local tiles 的关系.

V2PE 与 Qwen-VL 的 M-RoPE 关注点不同. M-RoPE 重点统一 text, image 和 video 的多维位置; V2PE 更聚焦动态高分辨率视觉 token 的位置表达. 两者都说明现代 VLM 不再把图像简单看成一串无结构 token.

InternVL3 还强调 native multimodal pretraining, 即在更早训练阶段让模型接触图文与交错多模态数据, 而不是 text-only LLM 完成后才连接视觉模块. 这种训练使图文融合更自然, 但需要更大数据与计算量, 对数据清洗和 [Megatron](../Framework/Megatron.md), [DeepSpeed](../Framework/DeepSpeed.md) 等分布式系统要求更高.

### 5.4 MPO 和 Test-Time Scaling

InternVL 后续训练还强调多模态偏好优化(MPO), 将 [DPO](../Align/DPO.md) / [RLHF](../Align/RLHF.md) 一类思想扩展到视觉回答, 用于降低幻觉, 改善 OCR, Grounding 和格式稳定性. Test-Time Scaling 则在推理时使用多次采样, candidate rerank, 更长 [CoT](../Inference/CoT.md) 或 verifier, 以额外计算换取更高的可验证任务准确率.

Test-Time Scaling 更适合 OCR exact match, Chart QA, Grounding box 和 Medical multiple choice 等可评分任务. 对开放式视觉描述, verifier 标准不清晰时, 多次采样不一定带来等比例收益.

InternVL 的主线可以概括为强视觉底座, 动态高分辨率, 适配视觉布局的位置编码, 以及更完整的预训练与后训练配方. 与 Qwen-VL 相比, Qwen 更强调真实任务覆盖和统一图像 / 视频输入, InternVL 更强调视觉底座与系统化能力构建, 但两者正在不断融合相似技术.

---

## 6. BLIP-2 和 Flamingo

### 6.1 BLIP-2

[BLIP-2](https://arxiv.org/abs/2301.12597) 解决的是如何低成本连接冻结 Vision Encoder 与冻结 LLM. Q-Former 使用 $M$ 个 learnable queries 通过 cross-attention 读取 $N_v$ 个视觉 patch:

$$
Q' = \mathrm{CrossAttention}(Q,X_v),
\qquad Q' \in \mathbb{R}^{M \times d_q},
\qquad M \ll N_v
$$

最终只有固定数量 query outputs 进入 LLM. 这种 query-based compression 参数效率高, 也使 visual token 成本稳定. 但固定信息瓶颈可能忽略 OCR 和 dense perception 细节. 与 LLaVA 相比, BLIP-2 的重点是用 Q-Former 连接冻结模型, LLaVA 的重点是保留 patch features 并通过 Visual Instruction Tuning 获得助手能力.

### 6.2 Flamingo

[Flamingo](https://arxiv.org/abs/2204.14198) 面向图文交错上下文和 multimodal few-shot learning. Perceiver Resampler 先把可变长度视觉特征压成固定 visual latents, LLM 中间层再通过 gated cross-attention 读取这些 latents:

$$
X_v \rightarrow Z,
\qquad
H_t' = \mathrm{GatedCrossAttention}(H_t,Z)
$$

Gate 控制视觉残差注入强度, 减少训练初期对冻结 LLM 的扰动. 这种架构天然支持 `<image_1> text_1 <image_2> text_2` 形式的 interleaved context, 但需要修改 LLM block, 工程复杂度高于输入侧拼接 visual tokens.

|对比项|LLaVA|BLIP-2|Flamingo|
|---|---|---|---|
|Connector|MLP Projector|Q-Former|Perceiver Resampler|
|Token 处理|Patch 逐个映射|固定 query 压缩|固定 latent 压缩|
|视觉注入|LLM 输入侧|LLM 输入侧|LLM 中间层 cross-attention|
|核心优势|简单, 易训练和扩展|冻结模型连接效率高|多图交错与 few-shot|
|主要代价|visual tokens 多|可能丢失 dense detail|结构与 serving 复杂|

---

## 7. CLIP, SigLIP 和 BLIP

[CLIP](https://arxiv.org/abs/2103.00020) 与 [SigLIP](https://arxiv.org/abs/2303.15343) 主要训练图文表征. CLIP 使用 batch 内 softmax contrastive loss, SigLIP 使用 pair-wise sigmoid loss. 两者可以提供强 Vision Encoder, 但没有 decoder-only 对话模型, 本身不能完成长回答和复杂指令.

BLIP 类 Encoder-Decoder 模型在视觉编码之后增加文本 decoder:

$$
\text{Image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Text Decoder}
\rightarrow
\text{Caption / Answer}
$$

它比 CLIP 多了生成能力, 适合 Caption 与 VQA. 当 decoder 规模和语言预训练不足时, 推理与指令遵循能力会弱于连接强 LLM 的现代路线. 因此 CLIP / SigLIP / BLIP 更适合作为理解视觉预训练演进的基础, 当前通用视觉助手则主要建立在强 LLM 之上.

---

## 8. Video VLM

Video VLM 不是简单输入更多图片, 因为模型还需要理解时间. 视频可表示为 $V=\{I_1,I_2,\dots,I_T\}$, 每帧编码后形成总视觉序列:

$$
\{X_{v,1},X_{v,2},\dots,X_{v,T}\}
\rightarrow
\text{Video Tokens}
\rightarrow
\text{LLM}
$$

如果每帧有 $N_{\mathrm{frame}}$ 个 token, 总量近似为 $T \times N_{\mathrm{frame}}$. Uniform sampling 简单但可能漏掉短动作, keyframe sampling 更关注变化但依赖额外检测, adaptive sampling 可以根据问题选帧但系统更复杂. 除减少帧数外, 还可以使用 temporal pooling, patch merging 或 video-specific encoder 压缩冗余.

视频位置需要表示 $(t,h,w)$, 而不只是图像中的 $(h,w)$. 缺少时间位置时, 模型可能识别每一帧的对象, 却无法判断动作方向, 状态变化和事件先后. 因此 Video VLM 的三项核心能力是帧级空间细节, 跨帧时间建模和长视频 token 控制.

---

## 9. 关键结构对比

### 9.1 高分辨率策略

|方法|代表路线|处理方式|共同代价|
|---|---|---|---|
|AnyRes|LLaVA-NeXT / OneVision|Global view + local tiles|tile 增加 visual tokens|
|Dynamic Resolution|Qwen-VL|按图像尺寸动态产生 token|batch 长度和缓存不固定|
|Dynamic High Resolution|InternVL|动态 tile 网格 + thumbnail|视觉编码与 LLM 成本同时增加|

三者都在解决固定低分辨率丢失 OCR, 文档, 图表和小目标的问题. 它们的名称不同, 但真正需要比较的是 resize / tile 规则, 最大视觉 token 数, 是否保留全局图, 以及后续 Connector 的压缩方式.

### 9.2 Connector 和位置编码

|设计|解决的问题|典型适用场景|
|---|---|---|
|MLP Projector|简单完成视觉到语言映射|通用 VQA, LLaVA baseline|
|Q-Former|用固定 query 压缩视觉信息|冻结模型连接, 低 token budget|
|Perceiver Resampler|将多图特征统一为固定 latents|Interleaved image-text|
|Patch Merger|压缩高分辨率 patch 并保留局部结构|OCR, Document, Chart|
|M-RoPE|统一文本, 图像和视频多维位置|Qwen-VL 图文视频任务|
|V2PE|增强动态高分辨率视觉位置|InternVL tile 输入|

OCR 负责读取文字内容, Grounding 负责指出内容或目标的位置, Document Understanding 则需要结合 OCR, layout, table 与语义完成问答. 三种能力相互关联但不能互相替代, 因而模型对比时应使用不同分项评测.

---

## 10. 总结

主流 VLM 架构的演进可以看成围绕 visual token 的持续优化. LLaVA 证明简单 MLP Projector 与高质量指令微调就能形成强基线; Qwen-VL 用 Dynamic Resolution, Patch Merger 和 M-RoPE 面向真实世界图文视频任务; InternVL 用强 Vision Encoder, Dynamic High Resolution, V2PE 和系统训练配方提高视觉上限. BLIP-2 与 Flamingo 则分别展示了 query compression 和 cross-attention 注入的另一种取舍.

选择架构时不应只记模型名称, 而应沿着 **分辨率 -> 视觉特征 -> token 压缩 -> 注入方式 -> 位置编码 -> 训练数据** 逐层判断. 当前强 VLM 的核心竞争力, 就是在可接受的 [Inference](./Inference.md) 成本下保留足够的视觉细节, 空间关系和时间信息.
