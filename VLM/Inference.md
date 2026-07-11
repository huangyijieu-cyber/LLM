# VLM Inference

VLM 推理比纯 LLM 多了视觉编码阶段.

纯 LLM 推理:

$$
\text{text tokens}
\rightarrow
\text{LLM}
\rightarrow
\text{answer}
$$

VLM 推理:

$$
\text{image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{visual tokens}
\rightarrow
\text{LLM}
\rightarrow
\text{answer}
$$

VLM 推理的核心成本来自两部分:

1. Vision Encoder forward.
2. visual tokens 带来的 prefill 和 [KV Cache](../Inference/KV_Cache.md) 成本.

---

## 1. 推理流程

### 1.1 Image Preprocessing

图像先经过预处理:

1. Resize.
2. Normalize.
3. Crop 或 padding.
4. 切 tile.
5. 转 tensor.
6. 按模型要求组织 dynamic resolution 或 AnyRes 输入.

不同模型对图像输入格式要求不同. 固定分辨率模型, dynamic resolution 模型和 tile-based 模型的预处理方式不同.

---

### 1.2 Vision Encoder Forward

预处理后的图像进入 [Vision Encoder](./Vision_Encoder.md):

$$
I \rightarrow X_v
$$

输出视觉特征:

$$
X_v = \{v_1,v_2,\dots,v_N\}
$$

Vision Encoder forward 是 VLM 相比纯 LLM 多出来的固定开销.

---

### 1.3 Projector Forward

视觉特征经过 [Projector](./Projector.md):

$$
H_v = f_{\text{proj}}(X_v)
$$

得到 LLM hidden size 下的 visual tokens.

如果使用 Q-Former, Perceiver Resampler 或 Patch Merger, 这一阶段还会进行 visual token compression.

---

### 1.4 Prompt 拼接

文本 prompt 中通常有 `<image>` placeholder:

```text
USER: <image>
What is shown in this image?
```

实际送入模型时, `<image>` 会被替换为 visual tokens:

$$
[\text{visual tokens}; \text{text tokens}]
$$

多图场景可能有多个 image placeholder:

```text
USER: <image_1> <image_2>
Compare these two images.
```

---

### 1.5 LLM Prefill and Decode

拼接后的序列进入 LLM.

Prefill 阶段:

1. 处理全部 visual tokens 和 prompt tokens.
2. 建立 [KV Cache](../Inference/KV_Cache.md).
3. 计算第一步生成所需的上下文表示.

Decode 阶段:

1. 自回归逐 token 生成回答.
2. 每一步复用 KV Cache.
3. decode 成本主要和输出长度有关.

VLM 中高分辨率图片主要影响 prefill, 不是只影响 decode.

---

## 2. Visual Token 成本

VLM 推理的关键问题是 visual tokens 会占用上下文长度.

假设:

- 文本 token 数为 $N_t$.
- visual token 数为 $N_v$.

总输入长度:

$$
N = N_t + N_v
$$

Prefill 计算和 [KV Cache](../Inference/KV_Cache.md) 都会受 $N$ 影响.

---

### 2.1 高分辨率的代价

更高分辨率通常意味着更多 patch:

$$
N_v \propto \frac{H \times W}{P^2}
$$

其中 $P$ 是 patch size.

因此:

1. 分辨率越高, 细节越多.
2. visual token 越多, prefill 越慢.
3. KV Cache 越大.
4. batch size 越小.
5. serving 调度越复杂.

---

### 2.2 Dynamic Resolution 的代价

Qwen-VL 等模型使用 dynamic resolution, 不同图片会产生不同数量 visual tokens:

$$
N_v = f(H,W)
$$

这能保留更多细节, 但会导致:

1. batch 内 sequence length 不一致.
2. padding 浪费增加.
3. KV Cache 大小难以提前估计.
4. continuous batching 更复杂.

---

## 3. 多图推理

多图输入时, visual token 数近似线性增加:

$$
N_v = \sum_{i=1}^{K} N_{v,i}
$$

其中 $K$ 是图片数量.

多图任务包括:

1. 多图对比.
2. 医学检查前后对比.
3. 多页文档理解.
4. Agent 截图序列理解.
5. 商品或病例多视角分析.

主要问题:

1. 上下文变长.
2. 图片顺序需要明确.
3. 图片之间的对应关系需要 prompt 说明.
4. 模型容易混淆不同图片的内容.

---

## 4. Video 推理

Video VLM 通常先采样帧:

$$
V = \{I_1,I_2,\dots,I_T\}
$$

每帧生成 visual tokens, 再输入 LLM.

---

### 4.1 帧采样策略

常见方式:

1. Uniform sampling.
2. Key frame sampling.
3. Scene change sampling.
4. 按时间戳采样.
5. Adaptive sampling.

帧采样的目标是在 token 成本可控的情况下保留关键时间信息.

---

### 4.2 推理难点

1. 帧数多, token 多.
2. 长视频需要压缩.
3. 时间顺序和事件变化难建模.
4. 推理延迟高.
5. 不同帧之间可能存在大量冗余.

Video token 数大致为:

$$
N_v = T \times N_{\text{frame}}
$$

其中:

- $T$: 采样帧数.
- $N_{\text{frame}}$: 每帧 visual token 数.

---

## 5. Serving 框架

### 5.1 vLLM

[vLLM](../Framework/vllm.md) 主要优势是高吞吐 LLM serving.

对于 VLM, 需要额外支持:

1. image processor.
2. Vision Encoder.
3. visual token insertion.
4. multimodal input schema.
5. 多模态请求 batching.

vLLM 适合:

1. 通用 VLM serving.
2. 大规模 batch inference.
3. [RLHF](../Align/RLHF.md) / [GRPO](../Align/GRPO.md) rollout.

---

### 5.2 SGLang

[SGLang](../Framework/SGLang.md) 更适合复杂推理 workflow.

它适合:

1. Agent 截图理解.
2. structured output.
3. 多轮视觉工具调用.
4. prefix cache 复用.
5. 多步骤视觉推理.

---

## 6. 推理优化方向

### 6.1 减少 Visual Tokens

方法:

1. 使用 Q-Former.
2. 使用 Perceiver Resampler.
3. 使用 Patch Merger.
4. pooling.
5. token pruning.
6. 只保留关键 tile.

目标是降低:

1. prefill 时间.
2. KV Cache 显存.
3. batch padding 浪费.
4. 多图和视频输入成本.

---

### 6.2 高分辨率选择性处理

不是所有图片都需要高分辨率.

可以根据任务选择:

1. 普通场景图: 低分辨率.
2. OCR / 文档: 高分辨率.
3. 医学影像: 高分辨率或局部 crop.
4. 图表: 高分辨率.
5. 多图对比: 控制每张图的 token budget.

---

### 6.3 Prefix Cache

如果多个请求共享同一图片或同一 prompt, 可以复用 prefix cache.

例如:

1. 同一张医学影像问多个问题.
2. 同一页文档做多个抽取任务.
3. 同一个系统 prompt 多轮对话.
4. Agent 多轮任务共享截图历史.

这和 [KV Cache](../Inference/KV_Cache.md) 管理直接相关.

---

### 6.4 Quantization

量化可以降低模型权重显存.

常见:

1. INT8.
2. INT4.
3. FP8.

但医疗和 OCR 场景要谨慎, 因为低比特量化可能影响细粒度识别和安全性.

---

### 6.5 Vision Encoder 缓存

如果同一张图片会被多次查询, 可以缓存 Vision Encoder 输出:

$$
I \rightarrow X_v
$$

后续请求直接复用 $X_v$ 或 $H_v$.

适合:

1. 同一文档多问题.
2. 同一医学影像多轮问答.
3. 多轮 GUI 分析.

需要注意:

1. 缓存和模型版本绑定.
2. 图像预处理必须一致.
3. 高并发下缓存需要淘汰策略.

---

## 7. VLM 推理常见问题

1. **看不清文字**: 分辨率不足或 OCR 数据不足.
2. **编造图像内容**: 视觉幻觉.
3. **多图混淆**: prompt 中图片顺序不清楚.
4. **回答太自信**: 缺少不确定性表达.
5. **延迟高**: visual tokens 太多.
6. **显存高**: KV Cache 和 Vision Encoder 都占显存.
7. **坐标错误**: grounding 格式或缩放不一致.

---

## 8. VLM 推理和 LLM 推理对比

|对比项|LLM|VLM|
|---|---|---|
|输入|文本 token|文本 token + visual tokens|
|额外模块|无|Vision Encoder + Projector|
|主要缓存|文本 KV Cache|文本 + visual token KV Cache|
|瓶颈|decode 和 KV Cache|vision forward + prefill + KV Cache|
|高分辨率影响|无|visual tokens 增加|
|多图输入|无|token 成本近似线性增加|
|视频输入|无|帧数导致 token 成本增加|

---

## 9. 总结

VLM 推理的核心问题是 visual tokens.

1. 图像越多, visual tokens 越多.
2. 分辨率越高, visual tokens 越多.
3. 视频帧数越多, visual tokens 越多.
4. visual tokens 会增加 prefill 时间和 KV Cache 显存.
5. 推理优化需要在细节保留和 token 成本之间权衡.

VLM 部署需要同时考虑 Vision Encoder, Projector, LLM prefill, KV Cache 和 serving 调度.
