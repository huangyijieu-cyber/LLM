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
\text{vision encoder}
\rightarrow
\text{visual tokens}
\rightarrow
\text{LLM}
\rightarrow
\text{answer}
$$

---

## 1. 推理流程

### 1.1 Image Preprocessing

图像先经过预处理:

1. Resize.
2. Normalize.
3. Crop 或 padding.
4. 切 tile.
5. 转 tensor.

不同模型对图像输入格式要求不同, 例如固定分辨率或动态分辨率.

---

### 1.2 Vision Encoder Forward

预处理后的图像进入 Vision Encoder:

$$
I \rightarrow X_v
$$

输出视觉特征:

$$
X_v = \{v_1,v_2,\dots,v_N\}
$$

---

### 1.3 Projector Forward

视觉特征经过 [Projector](./Projector.md):

$$
H_v = f_{\text{proj}}(X_v)
$$

得到 LLM hidden size 下的 visual tokens.

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

---

### 1.5 LLM Prefill and Decode

拼接后的序列进入 LLM.

Prefill 阶段:

- 处理全部 visual tokens 和 prompt tokens.
- 建立 [KV Cache](../Inference/KV_Cache.md).

Decode 阶段:

- 自回归逐 token 生成回答.
- 每一步复用 KV Cache.

---

## 2. Visual Token 成本

VLM 推理的一个关键问题是 visual tokens 会占用上下文长度.

假设:

- 文本 token 数为 $N_t$.
- visual token 数为 $N_v$.

总输入长度:

$$
N = N_t + N_v
$$

Prefill 计算和 [KV Cache](../Inference/KV_Cache.md) 都会受 $N$ 影响.

### 2.1 高分辨率的代价

更高分辨率通常意味着更多 patch:

$$
N_v \propto \frac{H \times W}{P^2}
$$

其中 $P$ 是 patch size.

因此:

- 分辨率越高, 细节越多.
- visual token 越多, 推理越慢.
- KV Cache 越大.
- batch size 越小.

---

## 3. 多图推理

多图输入时, visual token 数近似线性增加:

$$
N_v = \sum_{i=1}^{K} N_{v,i}
$$

其中 $K$ 是图片数量.

多图任务包括:

- 多图对比.
- 医学检查前后对比.
- 多页文档理解.
- Agent 截图序列理解.

主要问题:

1. 上下文变长.
2. 图片顺序需要明确.
3. 图片之间的对应关系需要 prompt 说明.

---

## 4. Video 推理

Video VLM 通常先采样帧:

$$
V = \{I_1,I_2,\dots,I_T\}
$$

每帧生成 visual tokens, 再输入 LLM.

### 4.1 帧采样策略

- Uniform sampling.
- Key frame sampling.
- Scene change sampling.
- 按时间戳采样.

### 4.2 推理难点

1. 帧数多, token 多.
2. 长视频需要压缩.
3. 时间顺序和事件变化难建模.
4. 推理延迟高.

---

## 5. Serving 框架

### 5.1 vLLM

[vLLM](../Framework/vllm.md) 主要优势是高吞吐 LLM serving.

对于 VLM, 需要额外支持:

- image processor.
- vision encoder.
- visual token insertion.
- multimodal input schema.

vLLM 适合:

- 通用 VLM serving.
- 大规模 batch inference.
- RL rollout.

---

### 5.2 SGLang

[SGLang](../Framework/SGLang.md) 更适合复杂推理 workflow.

它适合:

- Agent 截图理解.
- structured output.
- 多轮视觉工具调用.
- prefix cache 复用.

---

## 6. 推理优化方向

### 6.1 减少 visual tokens

方法:

- 使用 Q-Former.
- 使用 Perceiver Resampler.
- pooling.
- token pruning.
- 只保留关键 tile.

目标是降低:

- prefill 时间.
- KV Cache 显存.
- decoding 成本.

---

### 6.2 高分辨率选择性处理

不是所有图片都需要高分辨率.

可以根据任务选择:

- 普通场景图: 低分辨率.
- OCR / 文档: 高分辨率.
- 医学影像: 高分辨率或局部 crop.
- 图表: 高分辨率.

---

### 6.3 Prefix Cache

如果多个请求共享同一图片或同一 prompt, 可以复用 prefix cache.

例如:

- 同一张医学影像问多个问题.
- 同一页文档做多个抽取任务.
- 同一个系统 prompt 多轮对话.

这和 [KV Cache](../Inference/KV_Cache.md) 管理直接相关.

---

### 6.4 Quantization

量化可以降低模型权重显存.

常见:

- INT8.
- INT4.
- FP8.

但医疗和 OCR 场景要谨慎, 因为低比特量化可能影响细粒度识别和安全性.

---

## 7. VLM 推理常见问题

1. **看不清文字**: 分辨率不足或 OCR 数据不足.
2. **编造图像内容**: 视觉幻觉.
3. **多图混淆**: prompt 中图片顺序不清楚.
4. **回答太自信**: 缺少不确定性表达.
5. **延迟高**: visual tokens 太多.
6. **显存高**: KV Cache 和 vision encoder 都占显存.

---

## 8. VLM 推理和 LLM 推理对比

|对比项|LLM|VLM|
|---|---|---|
|输入|文本 token|文本 token + visual tokens|
|额外模块|无|vision encoder + projector|
|主要缓存|文本 KV Cache|文本 + visual token KV Cache|
|瓶颈|decode 和 KV Cache|vision forward + prefill + KV Cache|
|高分辨率影响|无|visual tokens 增加|
|多图输入|无|token 成本线性增加|

一句话总结:

**VLM 推理的核心成本来自视觉编码和 visual tokens, 图片越多, 分辨率越高, visual tokens 越多, 推理显存和延迟就越高.**
