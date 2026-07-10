# VLM Architecture

VLM 架构不要按模型名字死记. 更好的记法是先抓住 5 个差异轴:

1. **是否生成文本**: CLIP 只做图文表征, LLaVA / Qwen-VL 这类才是生成式 VLM.
2. **视觉 token 怎么来**: 直接用 patch tokens, 还是用 Q-Former / Resampler 压缩.
3. **视觉信息怎么进 LLM**: 作为 prefix visual tokens 拼进去, 还是在 LLM 中间层做 cross-attention.
4. **是否支持高分辨率 / 多图 / 视频**: 这决定 OCR, 文档, 医疗影像细节能力.
5. **训练重点是什么**: 有的重点是图文对齐, 有的重点是 instruction tuning, 有的重点是系统化数据和高分辨率策略.

一句话:

**VLM 架构的核心差异不是都叫 Vision Encoder + LLM, 而是视觉特征如何被压缩, 如何注入 LLM, 以及为了解决什么场景痛点.**

---

## 1. 先看总分类

|类型|代表模型|是否生成回答|核心用途|一句话记忆|
|---|---|---|---|---|
|Dual Encoder|CLIP, SigLIP|否|图文检索, 表征对齐|把图片和文字拉到同一个向量空间|
|Encoder-Decoder|BLIP|是|Caption, VQA|图像编码后直接用 decoder 生成文本|
|Query Compression|BLIP-2|是|低成本接入冻结 LLM|用 Q-Former 从图像里问出少量 visual tokens|
|Cross-Attention Injection|Flamingo|是|多图交错上下文, few-shot|LLM 中间层通过 cross-attention 读取图像|
|Prefix Visual Tokens|LLaVA|是|通用开源 VLM 基础范式|把图片变成一串 token 直接塞到 LLM 前面|
|Modern High-Res VLM|Qwen-VL, InternVL, LLaVA-NeXT|是|OCR, 文档, 多图, 视频, grounding|在 LLaVA 范式上加高分辨率, 数据和能力工程|

---

## 2. Dual Encoder: CLIP / SigLIP

### 2.1 它解决什么问题

Dual Encoder 解决的是 **图文表征对齐**:

$$
\text{Image Encoder}(I) \rightarrow z_I
$$

$$
\text{Text Encoder}(T) \rightarrow z_T
$$

训练目标是:

$$
\text{sim}(z_I,z_T) \uparrow
$$

对于匹配图文对, 相似度提高. 对于不匹配图文对, 相似度降低.

代表模型:

- [CLIP](https://arxiv.org/abs/2103.00020).
- [SigLIP](https://arxiv.org/abs/2303.15343).

---

### 2.2 和生成式 VLM 的区别

CLIP / SigLIP 本身通常 **不负责生成长回答**.

它们更像:

```text
图片 -> 向量
文本 -> 向量
比较两个向量是否匹配
```

而 LLaVA / Qwen-VL 更像:

```text
图片 + 问题 -> 语言回答
```

所以 CLIP 常用于:

1. 给 VLM 提供 vision encoder.
2. 做 image-text retrieval.
3. 做 zero-shot image classification.

不适合直接做:

1. 多轮视觉问答.
2. 医学报告生成.
3. 长文本解释.

---

### 2.3 CLIP 和 SigLIP 的区别

|对比项|CLIP|SigLIP|
|---|---|---|
|训练目标|softmax contrastive loss|sigmoid loss|
|batch 内关系|把 batch 内样本做多分类对比|每个 image-text pair 独立二分类|
|扩展性|依赖大 batch 对比|更适合大规模分布式扩展|
|记忆点|经典图文对齐模型|CLIP-like 的训练目标改进版|

记忆方式:

**CLIP 学会图片和文字是否匹配, SigLIP 是更适合大规模训练的 CLIP-like 图文对齐模型.**

---

## 3. Encoder-Decoder: BLIP

### 3.1 它解决什么问题

Encoder-Decoder 架构开始让模型直接生成文本:

$$
\text{Image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Text Decoder}
\rightarrow
\text{Caption / Answer}
$$

它比 CLIP 多了生成能力.

---

### 3.2 和 CLIP 的区别

|对比项|CLIP|BLIP 类 Encoder-Decoder|
|---|---|---|
|核心目标|图文向量对齐|根据图像生成文本|
|输出|embedding similarity|caption / answer|
|适合任务|检索, 分类|caption, VQA|
|语言能力|弱, 不负责长回答|比 CLIP 强, 但取决于 decoder|

记忆方式:

**CLIP 判断图文配不配, BLIP 开始根据图像说话.**

---

### 3.3 局限

早期 Encoder-Decoder 的 decoder 通常没有现代 LLM 那么强, 所以:

1. 语言推理能力有限.
2. 指令遵循能力有限.
3. 多轮对话能力有限.

这也是后面 BLIP-2 / LLaVA 要接入强 LLM 的原因.

---

## 4. BLIP-2: 用 Q-Former 低成本连接冻结 LLM

### 4.1 它解决什么问题

BLIP-2 的问题意识是:

**Vision Encoder 很强, LLM 也很强, 但直接把大量视觉 patch token 塞给 LLM 成本高, 也不好对齐.**

于是它加了 Q-Former:

$$
\text{Frozen Vision Encoder}
\rightarrow
\text{Q-Former}
\rightarrow
\text{Frozen LLM}
$$

Q-Former 里有一组 learnable query tokens:

$$
Q = \{q_1,q_2,\dots,q_M\}
$$

这些 query tokens 通过 cross-attention 去读图像 patch features:

$$
Q' = \text{CrossAttention}(Q,X_v)
$$

---

### 4.2 Q-Former 到底在干嘛

如果图像有 576 个 patch tokens, 直接送给 LLM 很贵.

Q-Former 的做法是:

```text
不要把所有 patch 都交给 LLM
而是用几十个 query tokens 从图像里提取和语言相关的信息
```

所以它本质上是一个 **视觉信息压缩器**.

---

### 4.3 和 LLaVA 的区别

|对比项|BLIP-2|LLaVA|
|---|---|---|
|连接器|Q-Former|MLP Projector|
|是否压缩视觉 token|是, query 数固定|通常不主动压缩, patch tokens 投影后直接进 LLM|
|训练重点|训练 Q-Former 接冻结模型|projector alignment + visual instruction tuning|
|结构复杂度|较高|较低|
|记忆点|用 query 问图片, 压缩后给 LLM|把图片 patch 翻译成 LLM token 后直接拼进去|

记忆方式:

**BLIP-2 是先筛选压缩视觉信息再给 LLM, LLaVA 是把视觉 patch 投影后更直接地塞给 LLM.**

---

## 5. Flamingo: 多图交错输入和中间层 Cross-Attention

### 5.1 它解决什么问题

Flamingo 关注的是:

**如何让冻结 LLM 处理多张图片和文本交错出现的上下文.**

例如:

```text
Image 1
Question about image 1
Image 2
Question comparing image 1 and image 2
```

它不是简单把 visual tokens 拼在最前面, 而是在 LLM 中插入 gated cross-attention 层.

---

### 5.2 结构

视觉部分:

$$
\text{Image Features}
\rightarrow
\text{Perceiver Resampler}
\rightarrow
\text{Visual Latents}
$$

语言部分:

$$
H_t' = \text{GatedCrossAttention}(H_t,H_v)
$$

LLM 的文本 hidden states 在中间层读取 visual latents.

---

### 5.3 和 BLIP-2 / LLaVA 的区别

|对比项|BLIP-2|Flamingo|LLaVA|
|---|---|---|---|
|视觉压缩|Q-Former|Perceiver Resampler|一般不明显压缩|
|视觉注入位置|送到 LLM 输入侧|插入 LLM 中间层 cross-attention|作为 prefix visual tokens|
|强项|低成本连接冻结 LLM|多图交错上下文, few-shot|简单开源, instruction tuning|
|代价|Q-Former 结构复杂|改 LLM 结构, 工程更重|visual tokens 多时成本高|

记忆方式:

**Flamingo 不是把图像一次性塞进 prompt, 而是让 LLM 在中间层多次看图.**

---

## 6. LLaVA: 最经典的开源 VLM 基础模板

### 6.1 它解决什么问题

LLaVA 的目标是用最简单工程路线做出能对话的 VLM.

结构:

$$
\text{Image}
\rightarrow
\text{CLIP Vision Encoder}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{LLM}
$$

投影后的视觉特征被当成 visual tokens:

$$
[\text{visual tokens}; \text{text tokens}]
$$

---

### 6.2 它的关键不是结构复杂, 而是数据和指令微调

LLaVA 的结构非常朴素:

```text
CLIP 负责看图
MLP 负责翻译视觉特征
LLM 负责回答
```

真正关键是两阶段训练:

1. **Feature Alignment**: 冻结 vision encoder 和 LLM, 训练 projector, 让 visual tokens 进入 LLM 空间.
2. **Visual Instruction Tuning**: 用多模态指令数据训练模型学会看图回答.

详见 [Training](./Training.md).

---

### 6.3 LLaVA 和 BLIP-2 的区别

|问题|BLIP-2|LLaVA|
|---|---|---|
|视觉信息是否先被 query 压缩|是|通常不是|
|连接模块|Q-Former|MLP Projector|
|LLM 是否强依赖 instruction tuning|相对弱一些|非常依赖|
|工程复杂度|更复杂|更简单|
|适合理解什么|低成本冻结模型连接|现代开源 VLM 基础工程范式|

记忆方式:

**LLaVA 的价值在于简单可复现: 视觉塔 + MLP + LLM + 指令数据.**

---

### 6.4 LLaVA 的短板

早期 LLaVA 的问题:

1. 固定分辨率容易丢 OCR 和细节.
2. 多图能力弱.
3. 文档, 表格, 图表能力不够.
4. grounding 不强.
5. 视频能力不是核心.

这正是 Qwen-VL / InternVL / LLaVA-NeXT 这些现代 VLM 要增强的地方.

---

## 7. 现代强 VLM: Qwen-VL / InternVL / LLaVA-NeXT

### 7.1 它们和 LLaVA 的根本区别

现代强 VLM 不一定推翻 LLaVA 的基础范式, 很多仍然是:

$$
\text{Vision Encoder}
\rightarrow
\text{Projector}
\rightarrow
\text{LLM}
$$

但它们把工程重点放在:

1. **更高分辨率**: 保留 OCR, 文档, 医疗影像细节.
2. **更强视觉塔**: 更好的感知和图文对齐.
3. **更强数据配方**: OCR, grounding, chart, document, video, multi-image.
4. **更强任务能力**: 不只是看图聊天, 还要读字, 看表, 定位, 看视频.
5. **更强系统训练**: 多阶段训练, 多任务混合, 数据清洗和评测闭环.

一句话:

**LLaVA 解决 VLM 从 0 到 1 怎么接起来, Qwen-VL / InternVL 解决 VLM 从能用到好用.**

---

### 7.2 Dynamic Resolution

早期 VLM 常把图片缩到固定尺寸, 例如 $224 \times 224$ 或 $336 \times 336$.

问题是:

```text
图片中文字很小
医学影像异常很细
表格线和数字很密
缩小后关键信息直接没了
```

Dynamic Resolution 的思路是:

1. 根据图片原始比例和大小动态决定输入分辨率.
2. 将大图切成多个 tile.
3. 每个 tile 编码成 visual tokens.
4. 让 LLM 看到更多局部细节.

代价:

1. visual tokens 变多.
2. prefill 更慢.
3. [KV Cache](../Inference/KV_Cache.md) 更大.
4. batch size 更难做大.

---

### 7.3 Qwen2-VL / Qwen2.5-VL 记忆点

Qwen2-VL / Qwen2.5-VL 更适合理解成 **真实世界多模态助手路线**.

它强调:

1. **动态分辨率**: 不同图片产生不同数量 visual tokens.
2. **强 OCR / Document**: 适合截图, 表格, 文档, 网页.
3. **Grounding**: 能输出框坐标或区域相关答案.
4. **Video**: 通过多帧扩展到视频.
5. **中文和多语言场景**: 对中文任务更友好.

和 LLaVA 的区别:

|对比项|LLaVA|Qwen2-VL / Qwen2.5-VL|
|---|---|---|
|核心定位|开源 VLM 基础模板|通用多模态助手|
|图像输入|较朴素, 早期多为固定分辨率|动态分辨率, 高分辨率更强|
|OCR / 文档|不是早期强项|重点能力|
|Grounding|弱一些|更重视坐标和定位|
|多图 / 视频|不是早期核心|更重视|
|记忆点|怎么把图接进 LLM|怎么让 VLM 真正处理复杂真实场景|

---

### 7.4 InternVL 记忆点

InternVL 更适合理解成 **强视觉基础模型 + 强 LLM + 系统训练配方** 的路线.

它强调:

1. 更强 vision foundation model.
2. 高质量 image-text alignment.
3. 动态高分辨率.
4. 多阶段训练.
5. 多 benchmark 综合能力.

和 Qwen-VL 的关系:

```text
二者都属于现代强 VLM
都不是只靠一个 projector 取胜
区别更多在具体视觉塔, 训练数据, 分辨率策略和生态
```

记忆方式:

**InternVL 不是一个新连接方式的代名词, 而是强视觉塔和系统训练路线的代表.**

---

### 7.5 LLaVA-NeXT / OneVision 记忆点

LLaVA-NeXT / OneVision 可以看作 LLaVA 路线的增强版:

1. 保留简单的 Vision Encoder + Projector + LLM 思路.
2. 加强高分辨率.
3. 加强多图和视频.
4. 加强 OCR / 文档 / reasoning 数据.

记忆方式:

**LLaVA 是基础模板, LLaVA-NeXT / OneVision 是把这个模板用更好的数据和多模态能力继续做强.**

---

## 8. Video VLM

Video VLM 可以看作 Image VLM 的扩展, 但难点不是简单多放几张图.

基本流程:

1. 从视频中采样多帧.
2. 每帧通过 vision encoder.
3. 对帧特征进行压缩或融合.
4. 将 video tokens 送入 LLM.

### 8.1 和 Image VLM 的区别

|对比项|Image VLM|Video VLM|
|---|---|---|
|输入|单图或多图|多帧序列|
|核心信息|空间内容|空间内容 + 时间变化|
|token 成本|取决于图片数量和分辨率|帧数一多 token 暴涨|
|主要难点|细节, OCR, grounding|时间顺序, 动作, 长视频压缩|

记忆方式:

**Video VLM 的核心不是看多张图, 而是理解帧之间发生了什么变化.**

---

## 9. 最容易混的几组区别

### 9.1 BLIP-2 vs LLaVA

|问题|BLIP-2|LLaVA|
|---|---|---|
|核心模块|Q-Former|MLP Projector|
|视觉 token 处理|先用 query 压缩|投影后直接作为 visual tokens|
|主要优势|参数效率, 冻结大模型连接|简单, 工程友好, 指令微调强|
|脑内图像|从图里问出少量信息|把图翻译成一串 LLM token|

---

### 9.2 Flamingo vs LLaVA

|问题|Flamingo|LLaVA|
|---|---|---|
|视觉注入位置|LLM 中间层 cross-attention|LLM 输入前缀|
|多图交错|天然强|早期较弱, 后续版本增强|
|工程复杂度|高|低|
|脑内图像|LLM 一边读文本一边在中间层看图|图片先变 token, 再和文本一起进 LLM|

---

### 9.3 LLaVA vs Qwen-VL / InternVL

|问题|LLaVA|Qwen-VL / InternVL|
|---|---|---|
|定位|基础开源 VLM 范式|现代强 VLM 系统路线|
|主要贡献|简单架构 + visual instruction tuning|高分辨率, OCR, grounding, 多图, 视频, 数据工程|
|是否同类范式|是|大多仍是 Vision Encoder + Projector + LLM|
|核心区别|从 0 到 1|从能用到好用|

---

## 10. 总对比表

|模型路线|视觉特征处理|怎么进 LLM|强项|主要代价|
|---|---|---|---|---|
|CLIP / SigLIP|图文编码成全局 embedding|不进 LLM|检索, 分类, 视觉塔|不能直接长回答|
|BLIP|Vision Encoder 输出给 decoder|Encoder-Decoder|caption, VQA|语言推理不如强 LLM|
|BLIP-2|Q-Former query 压缩|少量 query tokens 给 LLM|低成本接冻结 LLM|Q-Former 复杂|
|Flamingo|Perceiver 压缩 visual latents|中间层 cross-attention|多图交错, few-shot|工程复杂|
|LLaVA|CLIP patch features + MLP|prefix visual tokens|简单, 主流, 易复现|高分辨率和 OCR 早期弱|
|Qwen-VL / InternVL|高分辨率, 动态 tile, 强视觉塔|visual tokens + 强训练配方|OCR, 文档, grounding, 多图, 视频|token 多, 系统复杂|

一句话总结:

**CLIP 负责图文对齐, BLIP 开始生成, BLIP-2 用 Q-Former 压缩后接冻结 LLM, Flamingo 用 cross-attention 支持多图交错, LLaVA 用最简单 projector 打通 VLM, Qwen-VL / InternVL 在这个基础上用高分辨率和数据工程把能力做强.**
