# VLM Architecture

VLM Architecture 不要按模型名字死记. 现在主流开源 VLM 大多看起来都像:

$$
\text{Image / Video}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Visual Token Processor}
\rightarrow
\text{LLM}
\rightarrow
\text{Answer}
$$

真正拉开差距的不是这条公式, 而是下面几个问题:

1. **视觉 token 怎么来**: 固定分辨率 patch, dynamic resolution, tile, video frame, 还是 query 压缩.
2. **视觉 token 怎么压缩**: MLP projector, Q-Former, Perceiver Resampler, patch merger, pixel shuffle, token pruning.
3. **视觉信息怎么进 LLM**: prefix visual tokens, interleaved image-text tokens, 中间层 cross-attention.
4. **位置编码怎么做**: 1D text position, 2D image position, 3D video position, M-RoPE, V2PE.
5. **训练数据覆盖什么能力**: caption, VQA, OCR, document, chart, grounding, video, agent screenshot, medical image.

一句话:

**主流 VLM 的核心不是能不能看图, 而是在有限 visual token budget 下, 尽量保留文字, 细节, 空间位置, 多图关系和时间信息.**

相关笔记:

- [Vision Encoder](./Vision_Encoder.md).
- [Projector](./Projector.md).
- [Training](./Training.md).
- [Data Process](./Data_Process.md).
- [VLM Inference](./Inference.md).
- [Evaluation](./Evaluation.md).
- [KV Cache](../Inference/KV_Cache.md).

---

## 1. 先看总分类

截至 2026-07, 开源 VLM 里最值得重点掌握的是三条主线:

1. **LLaVA 系**: 最经典的开源基础模板, 核心是简单, 好复现, instruction tuning 路线清晰.
2. **Qwen-VL 系**: 真实世界多模态助手路线, 重点是 dynamic resolution, OCR, document, grounding, video, agent 场景.
3. **InternVL 系**: 强视觉基础模型 + 强 LLM + 系统训练配方路线, 重点是强 vision tower, dynamic high resolution, V2PE, MPO, test-time scaling.

BLIP-2 和 Flamingo 也很重要, 但更像理解设计取舍的基础:

- BLIP-2 解释了 **为什么需要 Q-Former 这种 query compression**.
- Flamingo 解释了 **为什么 interleaved image-text 和 cross-attention 很重要**.

CLIP / SigLIP / BLIP 是基础组件或历史路线, 需要知道但不必按现代 chat VLM 的标准展开.

|重要程度|路线|代表模型|核心问题|今天怎么记|
|---|---|---|---|---|
|重点|LLaVA 系|LLaVA, LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision|怎么用最简单方式把图接进 LLM|基础模板, 面试必会|
|重点|Qwen-VL 系|Qwen2-VL, Qwen2.5-VL, Qwen3-VL|怎么处理真实世界高分辨率图文视频|OCR / document / grounding / video 强|
|重点|InternVL 系|InternVL2.5, InternVL3|怎么用强视觉塔和训练配方做强开源 VLM|视觉能力和系统训练强|
|中等|BLIP-2|BLIP-2|怎么低成本连接冻结 vision encoder 和 LLM|Q-Former 压缩视觉信息|
|中等|Flamingo|Flamingo|怎么处理多图交错上下文|Perceiver + gated cross-attention|
|基础|CLIP / SigLIP|CLIP, SigLIP|怎么做图文表征对齐|训练 vision tower 和检索能力|
|基础|Encoder-Decoder|BLIP|怎么从图像生成文本|caption / VQA 早期路线|

---

## 2. 主流共同骨架: Vision Encoder + Projector + LLM

现代开源 VLM 最常见的形式是:

$$
X_v = \text{VisionEncoder}(I)
$$

$$
H_v = \text{Projector}(X_v)
$$

$$
Y = \text{LLM}([\text{visual tokens}; \text{text tokens}])
$$

其中:

- $X_v$: vision encoder 输出的 patch features.
- $H_v$: 映射到 LLM hidden size 后的 visual tokens.
- $[\text{visual tokens}; \text{text tokens}]$: 拼接后的多模态序列.

这条主线最大的优点是 **不需要大改 LLM 结构**. 只要 LLM 支持 embedding 输入, visual tokens 就能像文本 token 一样被送进去.

---

### 2.1 为什么不用一个全局 image embedding

早期 CLIP 常用一个全局 embedding 表示整张图:

$$
z_I \in \mathbb{R}^{d}
$$

但 chat VLM 不能只靠一个全局向量, 因为它需要回答:

1. 图片左上角写了什么字.
2. 表格第三列是什么数.
3. 医学影像某个小区域是否异常.
4. 两张图之间细节有什么变化.
5. 视频中动作发生的先后顺序.

所以主流生成式 VLM 更常用 patch/grid features:

$$
X_v = \{v_1,v_2,\dots,v_N\}
$$

这样 LLM 才有机会看见局部细节.

代价是:

$$
N_v \uparrow
\Rightarrow
\text{prefill cost} \uparrow
\Rightarrow
\text{KV Cache} \uparrow
$$

这就是为什么几乎所有主流架构都在做 visual token budget 的平衡.

---

### 2.2 Visual token budget 是 VLM 的核心矛盾

假设 patch size 是 $P$, 图像分辨率是 $H \times W$, 那么视觉 token 数大致是:

$$
N_v \approx \frac{H \times W}{P^2}
$$

如果分辨率翻倍, token 数不是翻倍, 而是接近 4 倍.

因此 VLM 设计永远在做取舍:

|选择|好处|代价|
|---|---|---|
|低分辨率|快, 省显存, batch 大|OCR, 文档, 医疗细节容易丢|
|高分辨率|细节强, OCR 强|prefill 慢, KV Cache 大|
|tile 切图|保留局部细节|多图块会打乱全局关系|
|query 压缩|token 少|可能丢细节|
|patch merger|速度和细节折中|压缩策略影响能力|

和 [VLM Inference](./Inference.md) 关系很大: VLM 慢通常不是 decode 慢, 而是 visual tokens 导致 prefill 很重.

---

### 2.3 Projector 不只是维度对齐

很多人把 [Projector](./Projector.md) 记成:

```text
把 vision hidden size 变成 LLM hidden size
```

这只说对了一半.

它实际还影响:

1. **语义翻译**: 把视觉特征翻译成 LLM 能读的表示.
2. **token 压缩**: 是否减少 visual token 数.
3. **局部信息保留**: 是否保留空间位置和细节.
4. **训练稳定性**: projector 太弱会对齐不充分, 太强会训练复杂.

常见 projector:

|Projector|代表路线|特点|
|---|---|---|
|Linear|早期小模型|简单, 但表达能力有限|
|MLP Projector|LLaVA 系|主流, 简单, 不主动压缩 token|
|Q-Former|BLIP-2|用 query 压缩视觉信息|
|Perceiver Resampler|Flamingo|把任意长度视觉特征压成固定 latent|
|Patch Merger / Token Merger|Qwen-VL 等|减少高分辨率下的 visual tokens|

---

## 3. LLaVA 系: 最经典的开源 VLM 基础模板

LLaVA 系是最应该先掌握的架构, 因为它是很多开源 VLM 的共同模板.

它的核心目标不是发明复杂连接器, 而是证明:

**只要用一个简单 projector 把视觉特征接入 LLM, 再用多模态指令数据训练, 模型就能具备不错的看图对话能力.**

代表:

- [LLaVA](https://arxiv.org/abs/2304.08485).
- [LLaVA-1.5](https://arxiv.org/abs/2310.03744).
- [LLaVA-NeXT / LLaVA-OneVision](https://arxiv.org/abs/2408.03326).
- [LLaVA-OV-2](https://arxiv.org/abs/2605.25979).

---

### 3.1 基础结构

LLaVA 的基本结构:

$$
\text{Image}
\rightarrow
\text{CLIP Vision Encoder}
\rightarrow
\text{MLP Projector}
\rightarrow
\text{LLM}
$$

输入形式类似:

```text
USER: <image>
What is unusual in this image?

ASSISTANT:
```

实际进入 LLM 时, `<image>` 会被替换为一串 visual tokens:

$$
[\text{visual tokens}; \text{text tokens}]
$$

LLaVA 的脑内图像:

```text
CLIP 看图
MLP 把图像 patch 翻译成 LLM token
LLM 负责按语言方式回答
```

---

### 3.2 Trick 1: 用 patch features, 不是只用 CLS

如果只用 CLIP 的全局 CLS embedding, 模型知道 "这大概是什么图", 但很难回答细节问题.

LLaVA 类模型通常使用 patch/grid features:

$$
X_v = [v_1,v_2,\dots,v_N]
$$

这样每个 patch 都能成为 LLM 上下文中的一部分.

好处:

1. 局部细节更好.
2. 适合 VQA.
3. 后续能扩展到 high-resolution tile.

代价:

1. visual tokens 多.
2. prefill 成本高.
3. OCR 能力仍受分辨率限制.

---

### 3.3 Trick 2: 两阶段训练

LLaVA 系最典型的训练是两阶段, 详见 [Training](./Training.md).

第一阶段: **Feature Alignment**.

```text
冻结 vision encoder
冻结 LLM
只训练 projector
```

目标是让 visual tokens 先进入 LLM 的 embedding space.

第二阶段: **Visual Instruction Tuning**.

```text
图像 + 指令 + 回答
训练模型学会看图问答和对话
```

这个设计很重要:

1. 如果一开始就训练全部参数, LLM 会看到完全陌生的视觉 embedding, 训练不稳定.
2. 先对齐 projector, 再做 instruction tuning, 训练更稳.
3. 低成本复用已有 LLM 和 vision encoder.

---

### 3.4 Trick 3: MLP Projector 足够简单

LLaVA 没有用 Q-Former, 而是用 MLP:

$$
H_v = W_2 \cdot \sigma(W_1 X_v)
$$

为什么简单 MLP 也能工作?

1. Vision encoder 已经通过 CLIP / SigLIP 学过图文对齐.
2. LLM 已经具备强语言能力.
3. projector 只需要做空间和维度映射.
4. instruction data 会教模型如何使用视觉 token.

这就是 LLaVA 的核心价值:

**架构不花哨, 但训练流程和数据配方有效.**

---

### 3.5 Trick 4: AnyRes / High-Resolution 扩展

早期 LLaVA 处理高分辨率不强, 因为固定 resize 会丢细节.

LLaVA-NeXT / OneVision 这类后续路线加入 high-resolution / AnyRes 思路:

```text
原图
-> 保留全局缩略图
-> 按网格切成多个高分辨率 tile
-> 每个 tile 走 vision encoder
-> 拼成更长 visual token 序列
```

好处:

1. OCR 更好.
2. 文档截图更好.
3. 图表和表格更好.
4. 小目标更容易保留.

代价:

1. visual tokens 变多.
2. tile 之间全局关系需要靠 LLM 学.
3. 推理更慢.

记忆方式:

**LLaVA 基础版是把一张缩小图接进 LLM, LLaVA-NeXT / OneVision 是把全局图和局部 tile 一起接进 LLM.**

---

### 3.6 Trick 5: Interleaved multi-image / video 统一格式

OneVision 这类路线强调把单图, 多图, 视频都看成统一序列:

```text
<image_1> text <image_2> text <frame_1> <frame_2> ...
```

这样做的目的:

1. 多图比较不再是特殊任务.
2. 视频可以看成多帧图像序列.
3. 训练数据格式更统一.
4. 模型更容易从 image task transfer 到 video task.

它和 Flamingo 的思想有相似处: 都重视 interleaved image-text. 区别是 LLaVA 系通常仍然尽量保持 LLM 输入前缀或序列拼接, 不像 Flamingo 那样大量改 LLM 中间层.

---

### 3.7 LLaVA 系的优缺点

优点:

1. 架构简单, 容易复现.
2. 不大改 LLM, 工程成本低.
3. 适合作为 VLM baseline.
4. instruction tuning 路线清晰.

缺点:

1. 早期 OCR / document 能力弱.
2. visual token 很多时推理成本高.
3. 视觉塔强弱直接限制上限.
4. 需要大量高质量指令数据.

面试里可以这样说:

**LLaVA 系的核心贡献不是 projector 多复杂, 而是把 CLIP vision tower, MLP projector, LLM 和 visual instruction tuning 组合成一个简单有效的开源 VLM 范式.**

---

## 4. Qwen-VL 系: 真实世界高分辨率多模态助手路线

Qwen-VL 系更适合记成:

**面向真实世界任务的 VLM: 读字, 看表, 看文档, 定位, 看视频, 操作 GUI, 多语言问答.**

代表:

- [Qwen-VL](https://arxiv.org/abs/2308.12966).
- [Qwen2-VL](https://arxiv.org/abs/2409.12191).
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923).
- [Qwen3-VL](https://arxiv.org/abs/2511.21631).

---

### 4.1 基础结构

Qwen-VL 系也可以抽象成:

$$
\text{Image / Video}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Patch Merger / Projector}
\rightarrow
\text{Qwen LLM}
$$

但它和基础 LLaVA 的关注点不同.

LLaVA 更像:

```text
怎么把图接进 LLM
```

Qwen-VL 更像:

```text
怎么让 VLM 真正在复杂屏幕, 文档, OCR, 视频和定位任务里好用
```

---

### 4.2 Trick 1: Dynamic Resolution

Qwen2-VL 的一个核心设计是 dynamic resolution.

早期模型常把图像固定 resize 到 $224 \times 224$ 或 $336 \times 336$:

```text
所有图片 -> 同一个尺寸 -> 固定数量 visual tokens
```

问题:

1. 长图被压扁.
2. 文档小字被糊掉.
3. 网页截图布局丢失.
4. 医疗影像细节可能消失.

Dynamic Resolution 的思路:

```text
不同图片 -> 根据原始宽高动态产生不同数量 visual tokens
```

形式上:

$$
N_v = f(H,W)
$$

而不是固定:

$$
N_v = C
$$

好处:

1. 小图不用浪费 token.
2. 大图可以保留更多细节.
3. 对 OCR, document, chart, screenshot 更友好.

代价:

1. batch 内序列长度不一致.
2. 推理调度更复杂.
3. [KV Cache](../Inference/KV_Cache.md) 更难估算.
4. 高分辨率输入会明显变慢.

---

### 4.3 Trick 2: Patch Merger 控制 token 数

Dynamic Resolution 会带来一个问题: visual tokens 太多.

因此 Qwen-VL 系会使用类似 patch merger 的模块, 把邻近 patch 合并或压缩:

$$
X_v \in \mathbb{R}^{N_v \times d_v}
\rightarrow
H_v \in \mathbb{R}^{N'_v \times d_{\text{LLM}}}
$$

其中:

$$
N'_v < N_v
$$

这类设计的目的不是单纯降维, 而是:

1. 控制上下文长度.
2. 控制 prefill 成本.
3. 保留足够空间细节.
4. 让高分辨率输入可用.

记忆方式:

**Dynamic Resolution 负责多看细节, Patch Merger 负责别把 LLM 撑爆.**

---

### 4.4 Trick 3: M-RoPE 处理图像和视频位置

纯文本 LLM 的 position encoding 本质上是一维:

$$
t = 1,2,\dots,n
$$

但图像是二维:

$$
(h,w)
$$

视频是三维:

$$
(t,h,w)
$$

Qwen2-VL 引入 M-RoPE, 可以把位置编码拆成多个维度, 分别表示:

1. 文本的一维顺序.
2. 图像的 height / width.
3. 视频的 temporal / height / width.

直观理解:

```text
文本 token 只需要知道前后顺序
图片 token 需要知道自己在第几行第几列
视频 token 还需要知道自己属于第几帧
```

为什么重要?

1. OCR 需要空间顺序.
2. 表格需要行列关系.
3. grounding 需要坐标关系.
4. 视频需要时间顺序.

---

### 4.5 Trick 4: Grounding 和坐标输出

Qwen-VL 系很重视 grounding.

Grounding 不是简单说 "图里有猫", 而是要定位:

```text
猫在 [x1, y1, x2, y2]
```

这要求模型同时具备:

1. 物体识别.
2. 空间位置理解.
3. 坐标格式输出.
4. 训练数据中的区域标注.

Grounding 对医疗和 Agent 场景也很重要:

- 医疗: 病灶区域在哪里.
- GUI Agent: 要点击哪个按钮.
- 文档: 证据来自哪块文本.

它和 [Evaluation](./Evaluation.md) 关系很大, 因为 grounding 通常用 box IoU 或 point accuracy 评测.

---

### 4.6 Trick 5: Qwen2.5-VL 的真实世界增强

Qwen2.5-VL 可以理解成 Qwen2-VL 的工程强化版.

重点能力:

1. **更强 OCR / Document**: 更适合文档, 表格, 截图, 网页.
2. **更强 grounding**: 支持更细的区域定位和坐标理解.
3. **更强 video**: 对多帧视频和时间信息更友好.
4. **更强 agent 场景**: 对手机截图, PC 截图, UI 元素理解更实用.
5. **结构化输出**: 对 JSON, 坐标, 表格抽取更有价值.

这类模型在面试里不要只说 "多模态大模型", 要说它解决了真实落地里的细任务:

```text
OCR + Document + Chart + Grounding + Video + GUI
```

---

### 4.7 Trick 6: Qwen3-VL 的新趋势

Qwen3-VL 继续强化两个方向:

1. **更长的 interleaved multimodal context**: 图, 文, 视频可以更自然地混在长上下文里.
2. **更强的位置和层级视觉注入**: 例如 Interleaved-MRoPE, DeepStack 等设计, 让视觉信息在更合适的位置影响语言模型.

可以这样理解:

```text
Qwen2-VL: 让任意分辨率图像和视频进入 LLM
Qwen2.5-VL: 强化真实世界任务和 grounding
Qwen3-VL: 强化长上下文, 视觉细节注入和多模态推理
```

注意:

1. 这些新模型的细节会随版本变化.
2. 面试不一定要求背每个模块名.
3. 更重要的是说清楚它们为什么要做 dynamic resolution, M-RoPE, patch merger 和 grounding data.

---

### 4.8 Qwen-VL 系的优缺点

优点:

1. OCR / document / chart 能力强.
2. 中文和多语言场景友好.
3. grounding 和 agent 场景实用.
4. video 支持更系统.
5. 生态和部署适配较好, 常见于 [vLLM](../Framework/vllm.md) 和 [SGLang](../Framework/SGLang.md) 推理场景.

缺点:

1. 动态分辨率会让推理长度不稳定.
2. 高分辨率输入显存压力大.
3. 坐标和结构化输出依赖数据质量.
4. 对服务端 batching 和调度要求更高.

记忆方式:

**Qwen-VL 系不是只把图接进 LLM, 而是把真实世界视觉任务拆成 OCR, document, grounding, video, GUI, 再用动态分辨率和位置编码把这些能力做实.**

---

## 5. InternVL 系: 强视觉塔 + 系统训练配方路线

InternVL 系更适合记成:

**不只做连接器, 而是把 vision foundation model, LLM, 数据配方, 训练策略和 test-time scaling 一起做强.**

代表:

- [InternVL](https://arxiv.org/abs/2312.14238).
- [InternVL1.5](https://arxiv.org/abs/2404.16821).
- [InternVL2.5](https://arxiv.org/abs/2412.05271).
- [InternVL3](https://arxiv.org/abs/2504.10479).

---

### 5.1 基础结构

InternVL 系也可以抽象成:

$$
\text{Image}
\rightarrow
\text{InternViT / Vision Foundation Model}
\rightarrow
\text{MLP / Token Processor}
\rightarrow
\text{LLM}
$$

它和 LLaVA 的区别不是公式不同, 而是:

1. vision encoder 更强.
2. 高分辨率处理更系统.
3. 训练数据和训练阶段更多.
4. 多 benchmark 能力更均衡.
5. 后续版本强调原生多模态预训练和偏好优化.

---

### 5.2 Trick 1: Dynamic High Resolution

InternVL 系常用 dynamic high resolution 思路.

核心做法:

```text
根据图像比例选择 tile 布局
把大图切成多个固定尺寸 tile
可额外加入全局 thumbnail
每个 tile 通过 vision encoder
再把 token 送入 LLM
```

为什么要加 thumbnail?

tile 保留局部细节, 但会削弱全局布局. thumbnail 负责给模型全局视野:

```text
tile: 看清局部小字和细节
thumbnail: 知道整张图的布局
```

这对 document, chart, medical image 很重要.

代价:

1. tile 数越多, token 越多.
2. 模型需要学会 tile 顺序和全局关系.
3. 推理成本高于固定分辨率.

---

### 5.3 Trick 2: 强 Vision Foundation Model

LLaVA 往往复用 CLIP vision tower. InternVL 系更强调训练和扩展自己的视觉基础模型.

这意味着视觉塔本身就更适合:

1. 细粒度视觉理解.
2. OCR / document.
3. dense perception.
4. 跨任务迁移.
5. 多模态对齐.

为什么这重要?

因为 projector 再强, 也救不了一个已经丢细节的 vision encoder.

可以这样记:

**LLaVA 更像把现成视觉塔接到 LLM, InternVL 更强调视觉塔本身也要足够强.**

---

### 5.4 Trick 3: V2PE 让位置更适合高分辨率

InternVL3 引入 V2PE 这类更适合视觉 token 的位置编码设计.

问题背景:

1. 文本是一维顺序.
2. 图像 tile 是二维网格.
3. dynamic high resolution 下, 不同样本 tile 数和布局不同.

如果位置编码处理不好, 模型会混淆:

- tile 之间的顺序.
- 同一 tile 内 patch 的位置.
- 全局图和局部图的关系.

V2PE 的核心动机是让视觉位置编码更适应高分辨率动态视觉输入.

和 Qwen 的 M-RoPE 对比:

|对比项|M-RoPE|V2PE|
|---|---|---|
|代表路线|Qwen-VL 系|InternVL3|
|核心目的|把文本, 图像, 视频位置拆成多维|让视觉 token 在高分辨率动态输入下有更合适的位置表示|
|解决问题|1D / 2D / 3D position 统一|dynamic high-resolution vision position|

---

### 5.5 Trick 4: 原生多模态预训练

很多早期 VLM 是:

```text
先有 text-only LLM
再接 vision encoder
再做 multimodal tuning
```

InternVL3 强调 native multimodal pretraining:

```text
在预训练阶段就让模型接触图文数据
而不是最后才把视觉模块接上
```

好处:

1. 视觉和语言对齐更早发生.
2. 多模态推理更自然.
3. 后续 instruction tuning 压力更小.

代价:

1. 训练成本更高.
2. 数据清洗更复杂.
3. 需要更强分布式训练工程, 可参考 [Megatron](../Framework/Megatron.md) 和 [DeepSpeed](../Framework/DeepSpeed.md).

---

### 5.6 Trick 5: MPO 和 test-time scaling

InternVL3 还强调 mixed preference optimization, 可以理解成多模态版偏好优化.

它和 [DPO](../Align/DPO.md), [RLHF](../Align/RLHF.md) 的关系:

```text
让模型更偏好好的多模态回答
减少幻觉
提高 reasoning, grounding, OCR 等任务的稳定性
```

Test-time scaling 则是在推理阶段增加计算来换效果, 例如:

1. 多次采样.
2. 多候选答案 rerank.
3. 更长 [CoT](../Inference/CoT.md).
4. verifier 或规则检查.

这对可验证任务尤其有效:

- OCR exact match.
- chart QA.
- math diagram.
- grounding box.
- medical multiple choice.

---

### 5.7 InternVL 和 Qwen-VL 的区别

|问题|Qwen-VL 系|InternVL 系|
|---|---|---|
|主线记忆|真实世界多模态助手|强视觉塔 + 系统训练配方|
|强项|OCR, document, grounding, video, agent|视觉基础能力, 高分辨率, benchmark 均衡|
|结构关键词|dynamic resolution, M-RoPE, patch merger|dynamic high resolution, strong ViT, V2PE, MPO|
|训练重点|多任务真实场景数据|视觉基础模型 + 原生多模态预训练 + 偏好优化|
|面试说法|更像应用型通用 VLM|更像系统化堆强视觉和训练 recipes|

记忆方式:

**Qwen-VL 重点是把复杂真实场景做实, InternVL 重点是把视觉底座和训练系统做强.**

---

## 6. BLIP-2: Q-Former 压缩视觉信息

BLIP-2 很重要, 因为它解释了一个经典问题:

**Vision Encoder 很强, LLM 很强, 但中间怎么低成本对齐?**

论文: [BLIP-2](https://arxiv.org/abs/2301.12597).

---

### 6.1 基础结构

BLIP-2 的结构:

$$
\text{Frozen Vision Encoder}
\rightarrow
\text{Q-Former}
\rightarrow
\text{Frozen LLM}
$$

Q-Former 有一组 learnable query tokens:

$$
Q = \{q_1,q_2,\dots,q_M\}
$$

这些 query tokens 通过 cross-attention 读取图像 patch features:

$$
Q' = \text{CrossAttention}(Q,X_v)
$$

最后只把 $M$ 个 query outputs 送给 LLM.

---

### 6.2 Trick 1: Query Compression

假设一张图有 576 个 patch tokens:

$$
X_v \in \mathbb{R}^{576 \times d_v}
$$

BLIP-2 不直接把 576 个都送给 LLM, 而是用 32 个左右 query tokens 去问图像:

```text
哪些视觉信息对语言任务有用?
```

输出变成:

$$
Q' \in \mathbb{R}^{32 \times d_q}
$$

优点:

1. token 数少.
2. 适合冻结大模型.
3. 训练成本低.

缺点:

1. query 数固定, 细节可能丢.
2. OCR / document 这类 dense task 不一定强.
3. Q-Former 比 MLP 更复杂.

---

### 6.3 Trick 2: 冻结大模型降低训练成本

BLIP-2 的一个关键动机是复用已有模型:

```text
冻结 image encoder
冻结 LLM
主要训练 Q-Former
```

这样做很省:

1. 不需要重新训练 vision encoder.
2. 不需要大规模微调 LLM.
3. 对算力友好.

但也有限制:

1. 上限受冻结模块限制.
2. 模型适应新任务的能力不如全链路训练.
3. 指令遵循能力不一定像 LLaVA 系那样直接.

---

### 6.4 BLIP-2 和 LLaVA 的区别

|问题|BLIP-2|LLaVA|
|---|---|---|
|连接模块|Q-Former|MLP Projector|
|视觉 token 处理|先用 query 压缩|patch 投影后直接进 LLM|
|主要优势|参数效率, 冻结模型连接|简单, 指令微调强|
|主要风险|压缩后丢细节|token 多, 推理成本高|
|脑内图像|从图里问出少量摘要|把图翻译成一串 LLM token|

记忆方式:

**BLIP-2 是先筛选再交给 LLM, LLaVA 是尽量直接把视觉 patch 交给 LLM.**

---

## 7. Flamingo: Interleaved 输入和中间层 Cross-Attention

Flamingo 很重要, 因为它解决的是另一个经典问题:

**如何让冻结 LLM 处理图文交错的多模态上下文?**

论文: [Flamingo](https://arxiv.org/abs/2204.14198).

---

### 7.1 基础结构

Flamingo 的视觉侧:

$$
\text{Image Features}
\rightarrow
\text{Perceiver Resampler}
\rightarrow
\text{Visual Latents}
$$

语言侧是在 LLM 中插入 gated cross-attention:

$$
H_t' = \text{GatedCrossAttention}(H_t,H_v)
$$

这意味着 LLM 不是只在输入开头看到图像, 而是在中间层反复读取视觉信息.

---

### 7.2 Trick 1: Perceiver Resampler

不同图片的 patch 数可能不同. Flamingo 用 Perceiver Resampler 把视觉特征压成固定数量的 visual latents:

$$
X_v \rightarrow Z
$$

其中 $Z$ 的长度固定.

好处:

1. 多图输入更可控.
2. token 数更稳定.
3. 可以处理 interleaved image-text.

代价:

1. 结构复杂.
2. 训练成本高.
3. 开源复现难度大.

---

### 7.3 Trick 2: Gated Cross-Attention

Flamingo 不是这样做:

```text
[all visual tokens] + [all text tokens] -> LLM
```

而是:

```text
LLM hidden states 在若干层通过 cross-attention 读取 visual latents
```

gate 的作用是控制视觉信息注入强度:

```text
刚开始不要让视觉特征破坏预训练 LLM
训练后逐步学会什么时候看图
```

直观记忆:

**LLaVA 是把图塞进 LLM 的输入, Flamingo 是让 LLM 在中间层随时查图.**

---

### 7.4 Trick 3: Interleaved image-text few-shot

Flamingo 支持类似:

```text
<image_1> Question 1 Answer 1
<image_2> Question 2 Answer 2
<image_3> Question 3
```

这对 few-shot 很重要.

为什么?

1. 示例和图片可以交错出现.
2. 模型能从前面的图文示例中学习任务格式.
3. 多图上下文更自然.

今天很多多图 / 视频 VLM 的序列格式都能看到 Flamingo 的影子.

---

### 7.5 Flamingo 今天怎么记

Flamingo 本身不一定是你训练开源模型时最常用的路线, 但它贡献了两个重要思想:

1. **Perceiver Resampler**: 控制视觉 token 长度.
2. **Gated Cross-Attention**: 在 LLM 中间层注入视觉信息.

和主流 LLaVA / Qwen / InternVL 路线相比:

|对比项|Flamingo|主流 prefix visual token 路线|
|---|---|---|
|视觉注入位置|LLM 中间层|LLM 输入序列|
|多图交错|天然强|后续通过 interleave 格式增强|
|工程复杂度|高|低|
|是否改 LLM 结构|是|通常尽量少改|

---

## 8. CLIP / SigLIP / BLIP: 基础路线简记

这些不是今天生成式 VLM 的全部, 但它们是理解主流架构的地基.

---

### 8.1 CLIP / SigLIP: 图文表征对齐

CLIP / SigLIP 的目标是把图像和文本拉到同一个向量空间.

代表:

- [CLIP](https://arxiv.org/abs/2103.00020).
- [SigLIP](https://arxiv.org/abs/2303.15343).

形式:

$$
\text{Image Encoder}(I) \rightarrow z_I
$$

$$
\text{Text Encoder}(T) \rightarrow z_T
$$

目标:

$$
\text{sim}(z_I,z_T) \uparrow
$$

CLIP 和 SigLIP 的区别:

|对比项|CLIP|SigLIP|
|---|---|---|
|训练目标|softmax contrastive loss|sigmoid loss|
|batch 依赖|更依赖 batch 内对比|更适合大规模分布式训练|
|输出用途|图文检索, zero-shot 分类, vision tower|同样常用作强 vision tower|
|是否生成回答|否|否|

记忆方式:

**CLIP / SigLIP 解决图文是否匹配, 不负责像 ChatGPT 一样生成长回答.**

---

### 8.2 BLIP: 从图像生成文本

BLIP 类 Encoder-Decoder 路线开始让模型根据图像生成文本:

$$
\text{Image}
\rightarrow
\text{Vision Encoder}
\rightarrow
\text{Text Decoder}
\rightarrow
\text{Caption / Answer}
$$

它比 CLIP 多了生成能力, 但语言推理能力通常不如接入强 LLM 的现代 VLM.

记忆方式:

**CLIP 判断图文配不配, BLIP 开始根据图像说话, BLIP-2 开始把图像压缩后接入冻结 LLM.**

---

## 9. Video VLM: 不只是多放几张图

Video VLM 可以看成 Image VLM 的扩展, 但核心难点不是 "帧数变多" 这么简单.

基本流程:

$$
V = \{I_1,I_2,\dots,I_T\}
$$

$$
I_t \rightarrow X_{v,t}
$$

$$
\{X_{v,1},\dots,X_{v,T}\}
\rightarrow
\text{video tokens}
\rightarrow
\text{LLM}
$$

---

### 9.1 Trick 1: 帧采样

视频不能把所有帧都送进 LLM.

常见采样:

1. Uniform sampling: 均匀抽帧.
2. Keyframe sampling: 选关键帧.
3. FPS sampling: 按固定帧率抽.
4. Adaptive sampling: 根据内容变化抽帧.

取舍:

|采样方式|优点|缺点|
|---|---|---|
|均匀采样|简单稳定|可能错过关键瞬间|
|关键帧采样|更关注变化|需要额外检测策略|
|低 FPS|省 token|动作细节丢失|
|高 FPS|动作细节好|token 爆炸|

---

### 9.2 Trick 2: 时间位置编码

图片只需要空间位置:

$$
(h,w)
$$

视频还需要时间位置:

$$
(t,h,w)
$$

如果没有时间位置, 模型可能知道每帧有什么, 但不知道:

1. 谁先发生.
2. 动作方向是什么.
3. 前后状态如何变化.
4. 因果关系是什么.

所以 Qwen-VL 系的 M-RoPE, video time alignment, 以及其他 3D position encoding 都是在解决这个问题.

---

### 9.3 Trick 3: Video token compression

视频 token 数大致是:

$$
N_v = T \times N_{\text{frame}}
$$

其中 $T$ 是帧数.

如果每帧 256 个 visual tokens, 取 32 帧就是:

$$
8192 \text{ visual tokens}
$$

这对 prefill 和 KV Cache 很贵.

所以 Video VLM 必须做压缩:

1. 减少帧数.
2. 合并 patch tokens.
3. 使用 temporal pooling.
4. 只保留关键帧或关键区域.
5. 使用更高效的 video encoder.

记忆方式:

**Image VLM 的矛盾是分辨率和 token 数, Video VLM 的矛盾是帧数, 分辨率和时间理解三者同时爆炸.**

---

## 10. 几组最容易混的区别

### 10.1 MLP Projector vs Q-Former vs Perceiver Resampler

|问题|MLP Projector|Q-Former|Perceiver Resampler|
|---|---|---|---|
|代表|LLaVA|BLIP-2|Flamingo|
|做法|逐 patch 映射到 LLM hidden size|用 learnable queries 读取图像|用 latent tokens 压缩视觉特征|
|是否压缩 token|通常不明显压缩|明显压缩|明显压缩|
|优点|简单, 主流, 易复现|省 token, 适合冻结模型|适合多图交错|
|缺点|token 多|可能丢细节|结构复杂|

---

### 10.2 AnyRes vs Dynamic Resolution vs Dynamic High Resolution

|概念|常见路线|核心意思|
|---|---|---|
|AnyRes|LLaVA-NeXT / OneVision|把任意分辨率图像切成 tile + 全局图|
|Dynamic Resolution|Qwen-VL 系|根据输入宽高动态产生不同数量 visual tokens|
|Dynamic High Resolution|InternVL 系|动态选择高分辨率 tile 布局, 常配合全局 thumbnail|

它们都在解决同一个问题:

**固定低分辨率会丢细节.**

区别在具体实现和 token 控制策略.

---

### 10.3 M-RoPE vs V2PE

|问题|M-RoPE|V2PE|
|---|---|---|
|代表路线|Qwen-VL 系|InternVL3|
|解决问题|文本, 图像, 视频的多维位置统一|高分辨率动态视觉输入的位置编码|
|关键词|1D text, 2D image, 3D video|dynamic high-resolution vision position|

记忆方式:

**M-RoPE 更强调多模态多维位置统一, V2PE 更强调视觉 token 在动态高分辨率下的位置表达.**

---

### 10.4 OCR vs Grounding vs Document Understanding

|能力|问题形式|需要什么|
|---|---|---|
|OCR|图片里写了什么字|高分辨率, 小文字识别, 文本顺序|
|Grounding|目标在哪里|空间位置, 坐标格式, box/point 数据|
|Document Understanding|文档表达了什么|OCR + layout + 表格 + 结构化抽取|

这三者经常一起出现, 但不是一回事.

例如:

```text
OCR: 读出表格里的数字
Grounding: 指出数字在哪个单元格
Document Understanding: 根据整张表回答业务问题
```

---

### 10.5 Pretraining vs Instruction Tuning vs Preference Alignment

|阶段|目的|对应笔记|
|---|---|---|
|Image-Text Pretraining|学会图文语义对齐|[Training](./Training.md)|
|Projector Alignment|让 visual tokens 进入 LLM 空间|[Projector](./Projector.md)|
|Multimodal Instruction Tuning|学会按指令看图回答|[Training](./Training.md)|
|Preference Alignment|减少幻觉, 提高回答偏好|[DPO](../Align/DPO.md), [RLHF](../Align/RLHF.md)|
|Domain Adaptation|适配医疗, 金融, 工业等领域|[Medical VLM](./Medical_VLM.md)|

---

## 11. 面试速记表

|模型路线|一句话定位|核心 trick|最该说清楚的区别|
|---|---|---|---|
|LLaVA|开源 VLM 基础模板|CLIP + MLP Projector + instruction tuning|简单 projector 也能靠数据和 SFT 做出对话能力|
|LLaVA-NeXT / OneVision|LLaVA 的高分辨率和多模态扩展|AnyRes, interleaved image/video format|解决早期 LLaVA OCR 和多图视频弱的问题|
|Qwen-VL|真实世界多模态助手|dynamic resolution, patch merger, M-RoPE, grounding|重点不是连接器, 而是 OCR/document/video/agent 能力|
|InternVL|强视觉塔和系统训练路线|dynamic high resolution, strong ViT, V2PE, MPO|靠视觉底座和训练 recipes 把整体能力做强|
|BLIP-2|低成本连接冻结模型|Q-Former query compression|先筛选压缩再给 LLM|
|Flamingo|图文交错和 few-shot|Perceiver Resampler, gated cross-attention|在 LLM 中间层读图, 不只是输入前缀|
|CLIP / SigLIP|图文表征对齐|contrastive / sigmoid loss|不是生成式 VLM, 常作为 vision tower|

---

## 12. 最终记忆线

可以按这条线记:

```text
CLIP / SigLIP: 先学会图文对齐
BLIP: 开始根据图像生成文本
BLIP-2: 用 Q-Former 压缩视觉信息接冻结 LLM
Flamingo: 用 Perceiver 和 cross-attention 处理图文交错
LLaVA: 用最简单 projector 打通开源 VLM
LLaVA-NeXT / OneVision: 在 LLaVA 上补高分辨率, 多图, 视频
Qwen-VL: 面向真实世界 OCR, document, grounding, video, GUI
InternVL: 强视觉塔 + 高分辨率 + 系统训练配方
```

一句话总结:

**今天主流 VLM 的竞争点不是能不能把图接进 LLM, 而是谁能用更合理的 visual token 预算, 更好的空间/时间位置编码, 更强的数据配方和更稳定的训练流程, 把高分辨率真实世界视觉任务做稳.**

---

## 13. 参考资料

- [CLIP](https://arxiv.org/abs/2103.00020).
- [SigLIP](https://arxiv.org/abs/2303.15343).
- [BLIP-2](https://arxiv.org/abs/2301.12597).
- [Flamingo](https://arxiv.org/abs/2204.14198).
- [LLaVA](https://arxiv.org/abs/2304.08485).
- [LLaVA-1.5](https://arxiv.org/abs/2310.03744).
- [LLaVA-OneVision](https://arxiv.org/abs/2408.03326).
- [LLaVA-OV-2](https://arxiv.org/abs/2605.25979).
- [Qwen-VL](https://arxiv.org/abs/2308.12966).
- [Qwen2-VL](https://arxiv.org/abs/2409.12191).
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923).
- [Qwen3-VL](https://arxiv.org/abs/2511.21631).
- [InternVL](https://arxiv.org/abs/2312.14238).
- [InternVL1.5](https://arxiv.org/abs/2404.16821).
- [InternVL2.5](https://arxiv.org/abs/2412.05271).
- [InternVL3](https://arxiv.org/abs/2504.10479).
