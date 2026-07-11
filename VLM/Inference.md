# VLM Inference

VLM 推理在 LLM 前增加视觉预处理与编码, 并把视觉特征转换成 LLM 上下文:

$$
\text{Image}
\rightarrow
\text{Image Processor}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Projector}
\rightarrow
\text{Visual Tokens}
\rightarrow
\text{LLM}
$$

相比纯 LLM, VLM 多出 Vision Encoder 的固定计算, 也因为 visual tokens 增加了 prefill 与 [KV Cache](../Inference/KV_Cache.md) 成本. 分辨率, 图片数量和视频帧数最终都会汇总成同一个工程问题: **一次请求需要向 LLM 输入多少 visual tokens**.

---

## 1. 推理流程

图像首先按照模型对应的 processor 做 resize, normalize, crop 或 padding. 固定分辨率模型直接得到统一 tensor; AnyRes 或 Dynamic High Resolution 模型还要根据宽高比选择 tile 布局, 通常同时生成 global thumbnail 与 local tiles. 预处理必须和训练时完全一致, 否则不仅会影响视觉特征, 还可能破坏 Grounding 的坐标映射.

处理后的图像进入 [Vision Encoder](./Vision_Encoder.md):

$$
I \rightarrow X_v = \{v_1,v_2,\dots,v_{N_v}\}
$$

随后 [Projector](./Projector.md) 将 hidden size 映射到 LLM space. MLP 通常保留 token 数量, Q-Former 与 Perceiver Resampler 输出固定数量 query / latent, Patch Merger 则按空间邻域压缩 patch. 得到的 visual embeddings 会替换 prompt 中的 `<image>` placeholder, 或通过 LLM 中间层的 cross-attention 被读取.

拼接后的输入进入 LLM prefill. 此时模型一次性处理全部 visual tokens 和 text prompt, 并建立 KV Cache. Decode 阶段再逐 token 生成回答并复用缓存. 因此高分辨率和多图输入主要增加 **首 token 延迟和缓存显存**, 输出长度则主要影响后续 decode 时间.

---

## 2. Visual Token 成本

设文本 token 数为 $N_t$, 投影后的 visual token 数为 $N_v$, 则输入长度为:

$$
N = N_t + N_v
$$

对 patch size 为 $P \times P$ 且没有压缩的视觉塔, 原始 patch 数近似为:

$$
N_v \propto \frac{H \times W}{P^2}
$$

分辨率在高和宽上同时放大时, token 数按面积增长. 更多 visual tokens 会增加 LLM prefill, 扩大每层 KV Cache, 降低可用 batch size, 也使 continuous batching 中不同请求的长度差异更大. Dynamic Resolution 虽然避免把文档和长图压缩到固定低分辨率, 但 sequence length 变成 $N_v=f(H,W)$, 显存与调度更难提前估算.

Visual token budget 不能只追求越小越好. OCR, Chart 和医学影像中的关键信息可能只占很小区域, 过度 pooling 或 pruning 会让模型失去证据. 实际部署通常根据任务设置分辨率上限, tile 上限和每张图片的最大 token 数, 在准确率与吞吐之间选择工作点.

---

## 3. 多图和视频

多图输入的 visual token 数近似为各图片 token 数之和:

$$
N_v = \sum_{i=1}^{K}N_{v,i}
$$

除成本增长外, 多图任务还需要清晰的 image placeholder 与顺序. Prompt 应明确 `image_1`, `image_2` 分别代表什么, 对比问题应指出比较维度. 如果所有图片 token 连续拼接却没有边界和位置标识, 模型容易把不同图片中的对象或文字混在一起. 医学检查前后对比, 多页文档和多视角商品分析都存在这个问题.

Video VLM 通常先把视频采样为 $T$ 帧, 总 token 数近似为:

$$
N_v = T \times N_{\mathrm{frame}}
$$

Uniform sampling 覆盖完整时间轴但可能漏掉短暂关键动作; keyframe 或 scene-change sampling 更关注变化, 却需要额外策略; adaptive sampling 可以根据内容或问题选择帧, 工程上最复杂. 长视频还可以使用 temporal pooling, frame token merging 或分段总结. 无论使用哪种方法, 都需要保留时间顺序和时间戳, 否则模型只能理解多张独立图片, 无法判断事件先后和状态变化.

|输入类型|主要瓶颈|常见控制方式|
|---|---|---|
|单张普通图片|Vision Encoder 固定开销|固定分辨率, 常规 batching|
|高分辨率文档|tile 和 patch tokens 多|Dynamic Resolution 上限, Patch Merger|
|多图|token 数近似线性增加, 图像易混淆|每图 token budget, 清晰 placeholder|
|短视频|帧数和时间位置|均匀 / 关键帧采样, temporal position|
|长视频|大量冗余帧, 上下文过长|分段, 自适应采样, temporal compression|

---

## 4. Serving 框架

[vLLM](../Framework/vllm.md) 的 PagedAttention, continuous batching 和高吞吐调度可以扩展到多模态模型, 但还需要模型专用 image processor, Vision Encoder forward, multimodal input schema 与 visual token insertion. 它适合通用在线服务, 大批量推理和 [RLHF](../Align/RLHF.md) / [GRPO](../Align/GRPO.md) rollout.

[SGLang](../Framework/SGLang.md) 更强调复杂生成程序, prefix cache 和结构化输出, 适合多轮视觉工具调用, GUI Agent, 同一视觉上下文上的多步推理等 workflow. 框架选择不是由模型精度决定, 而是由请求模式决定: 简单独立请求重视 batching 吞吐, 多轮共享图像和 prompt 的任务更重视 prefix / visual feature 复用.

部署时还要确认框架是否完整支持目标模型的 processor, Dynamic Resolution, 多图输入和自定义位置编码. 即使 LLM backbone 名称相同, 不同 VLM 的 placeholder 展开规则和 visual token 顺序也可能不同, 不能只加载文本权重配置.

---

## 5. 推理优化

### 5.1 减少重复和无效计算

同一张图片被多次提问时, 可以缓存 Vision Encoder 输出 $X_v$ 或 Projector 输出 $H_v$, 避免重复视觉 forward. 同一文档多字段抽取, 同一医学影像多轮问答和固定 GUI 截图都适合使用视觉特征缓存. 缓存必须绑定模型与 processor 版本, 图像预处理参数也必须一致.

如果请求共享相同 visual tokens 和系统 prompt, 还可以复用 prefix KV Cache. Visual feature cache 节省视觉编码, prefix cache 进一步节省 LLM prefill, 两者作用位置不同. 高并发服务需要为缓存设置容量, key 和淘汰策略, 并避免不同用户之间的数据泄露.

### 5.2 控制 Visual Tokens

Q-Former, Perceiver Resampler 和 Patch Merger 在模型结构内压缩 token; pooling, token pruning 和关键 tile 选择则可以进一步减少输入. 普通场景图可使用较低分辨率, OCR / Document / Chart 使用高分辨率, 医学影像可使用局部 crop 或多尺度输入, 多图请求则限制每张图的独立 budget. 选择性处理通常比对所有请求统一使用最高分辨率更有效.

### 5.3 Quantization 和并行

INT8, INT4 和 FP8 等量化主要减少模型权重显存并提高吞吐, 但 Vision Encoder, Projector 和 LLM 可能需要不同量化策略. OCR 与医疗场景依赖细粒度差异, 应通过分项评测确认低比特量化是否造成感知或安全回退. 对超大模型还可以使用 tensor parallel, pipeline parallel 或多实例部署, 相关方法见 [Distributed Training](../Distributed_Training/Main.md).

---

## 6. 常见问题定位

|现象|优先检查|可能原因|
|---|---|---|
|小字读不清|输入分辨率, tile 数, OCR benchmark|resize 过小, Vision Encoder 或压缩丢细节|
|编造图像内容|负样本, Grounding, prompt|语言先验强, 视觉证据不足, 对齐数据有幻觉|
|多图内容混淆|placeholder 数量和顺序|图像边界不清, prompt 引用错误|
|首 token 延迟高|Vision forward, visual token 数, prefill|分辨率或图片数过高|
|KV Cache 显存高|投影后 token 数, batch length|visual sequence 过长, padding 浪费|
|坐标系统性偏移|resize / crop 后坐标还原|训练和推理使用不同坐标系|
|回答过度自信|不确定性数据, preference alignment|安全与拒答训练不足|

这类问题需要分别观测 image preprocessing, Vision Encoder, Projector token count 和 LLM prefill, 不能只记录端到端 latency. 例如 "VLM 很慢" 可能是视觉塔本身过大, 也可能是 Projector 没有压缩 high-resolution patches, 两者优化方式完全不同.

---

## 7. VLM 和 LLM 推理对比

|对比项|LLM|VLM|
|---|---|---|
|输入|Text tokens|Text tokens + visual tokens|
|额外模块|无|Image Processor + Vision Encoder + Projector|
|首 token 成本|Text prefill|Vision forward + multimodal prefill|
|缓存|Text KV Cache|Visual / text KV Cache, 可选视觉特征缓存|
|长度变化来源|Prompt 长度|Prompt, 分辨率, 图片数, 视频帧数|
|调度难点|输出长度变化|输入和输出长度都高度变化|

---

## 8. 总结

VLM 推理的核心是把视觉质量要求转换成可控的 visual token budget. Vision Encoder 决定每张图的固定编码成本, Connector 决定多少视觉信息进入 LLM, prefill 和 KV Cache 决定服务吞吐. 单图, 多图, 文档和视频需要不同的分辨率与采样策略; 缓存, token 压缩和量化则分别优化重复计算, 上下文长度和权重显存.
