# VLM Data Process

VLM 的上限很大程度取决于数据. 对于同一个模型结构, 数据质量, 数据覆盖范围和数据配比往往比小的结构改动更重要.

VLM 数据可以粗略分为:

1. Image-text pair.
2. Caption 数据.
3. Multimodal instruction data.
4. OCR / Document 数据.
5. Grounding 数据.
6. Video 数据.
7. Medical multimodal 数据.

这些数据分别对应不同能力. 数据处理的核心不是简单堆数据, 而是让数据覆盖图文对齐, 指令遵循, OCR, grounding, reasoning, video 和领域安全.

---

## 1. Image-Text Pair

Image-text pair 是最基础的数据形式:

$$
(I,T)
$$

其中:

- $I$: 图片.
- $T$: 图片相关文本, 例如 alt-text, caption, title.

这类数据常用于 CLIP / SigLIP 式图文对齐预训练, 见 [Vision_Encoder](./Vision_Encoder.md).

---

### 1.1 优点

1. 数据规模大.
2. 可以从网页, 图库, 文档中收集.
3. 适合训练通用视觉语义表示.
4. 适合构建可复用 Vision Encoder.

---

### 1.2 问题

1. 文本可能和图片弱相关.
2. alt-text 可能很短或很噪.
3. 存在水印, 广告, 低质图.
4. 可能包含隐私和版权风险.
5. 不能直接提供复杂指令能力.

---

### 1.3 清洗重点

1. 去除低分辨率和模糊图片.
2. 过滤广告图, 水印图和纯装饰图.
3. 用 CLIP score 或 VLM judge 过滤图文不相关样本.
4. 去除重复图片和重复文本.
5. 做隐私和安全过滤.

---

## 2. Caption 数据

Caption 数据是图片和描述:

$$
(I,\text{caption})
$$

适合训练模型生成图片描述.

---

### 2.1 人工 Caption

优点:

1. 质量高.
2. 描述准确.
3. 幻觉少.

缺点:

1. 成本高.
2. 规模有限.
3. 风格可能不够多样.

---

### 2.2 合成 Caption

可以用强 VLM 自动生成 caption.

优点:

1. 成本低.
2. 规模大.
3. 可按需求生成详细描述.
4. 可以针对 OCR, chart, medical 等场景生成专项描述.

缺点:

1. 可能有幻觉.
2. 风格单一.
3. 会继承 teacher model 偏差.
4. 需要抽样人工检查.

---

### 2.3 Caption 数据的局限

Caption 数据通常只描述图片显著内容. 如果 caption 占比过高, 模型可能更擅长描述图片, 但不擅长:

1. 按指令回答.
2. 多轮对话.
3. 复杂推理.
4. grounding.
5. 结构化输出.

因此 caption 数据需要和 instruction data 配合使用, 见 [Training](./Training.md).

---

## 3. Multimodal Instruction Data

多模态指令数据形式:

$$
(I,x,y)
$$

其中:

- $I$: 图像.
- $x$: 用户问题或指令.
- $y$: 标准回答.

例如:

```text
USER: <image>
Describe the abnormal finding in this image.

ASSISTANT: ...
```

---

### 3.1 常见任务

1. Caption.
2. VQA.
3. OCR QA.
4. Region QA.
5. 多轮对话.
6. 判断题 / 选择题.
7. 结构化抽取.
8. 多图对比.
9. 医学问答.

---

### 3.2 数据构造方式

常见构造方式:

1. 人工标注.
2. 使用强 VLM 合成 QA.
3. 使用 LLM 根据 caption 生成问题.
4. 从已有 VQA 数据集转换.
5. 从文档, 表格, 医疗报告中构造任务.
6. 从 grounding 标注中构造区域问答.

---

### 3.3 清洗重点

Instruction data 需要检查:

1. 问题是否需要看图才能回答.
2. 答案是否真的能从图像中得到.
3. 是否存在语言先验泄露.
4. 是否存在格式错误.
5. 是否包含幻觉或过度推断.
6. 多轮对话中上下文是否一致.

---

## 4. OCR / Document 数据

OCR 和文档理解是当前 VLM 的重点能力.

数据类型:

1. TextVQA.
2. OCR 图片.
3. 文档截图.
4. 表格.
5. 图表.
6. 网页截图.
7. PDF 页面.
8. 表单和票据.

---

### 4.1 核心挑战

1. 文字小, 需要高分辨率.
2. 布局信息重要.
3. 表格和图表需要结构理解.
4. 需要避免仅靠语言先验猜答案.
5. 多页文档需要跨页信息整合.

这些能力通常需要 [Architecture](./Architecture.md) 中提到的 dynamic resolution, AnyRes 或 dynamic high resolution.

---

### 4.2 清洗要点

1. 去掉文字模糊图片.
2. 过滤 OCR 识别质量差样本.
3. 保留图文位置关系.
4. 检查答案是否可以从图片中找到.
5. 避免训练集中泄露评测答案.
6. 保留表格结构和版面信息.

---

### 4.3 结构化输出

OCR / Document 数据经常需要结构化输出:

```json
{
  "date": "...",
  "amount": "...",
  "items": []
}
```

训练时需要保证:

1. JSON 格式合法.
2. 字段含义稳定.
3. 输出来自图像证据.
4. 对缺失字段允许输出 null 或 unknown.

---

## 5. Grounding 数据

Grounding 数据包含区域标注:

$$
(I,\text{text},\text{box})
$$

或:

$$
(I,\text{region},\text{answer})
$$

常见形式:

1. Bounding box.
2. Point.
3. Segmentation mask.
4. Region caption.
5. Referring expression.

---

### 5.1 作用

1. 训练模型定位物体.
2. 缓解视觉幻觉.
3. 支持 "请指出哪里异常".
4. 支持 GUI click / agent 场景.
5. 医疗影像中可用于病灶定位.

---

### 5.2 清洗重点

1. 检查 box 是否越界.
2. 检查文本描述是否对应区域.
3. 统一坐标格式.
4. 处理不同分辨率下坐标缩放.
5. 区分检测任务和问答任务.

Grounding 数据和 [Evaluation](./Evaluation.md) 中的 IoU, pointing accuracy 直接相关.

---

## 6. Video 数据

Video VLM 数据通常包含:

1. 视频 caption.
2. 视频问答.
3. action recognition.
4. temporal reasoning.
5. 多帧事件描述.
6. 时间戳定位.

---

### 6.1 处理方式

1. 抽帧.
2. 关键帧选择.
3. 帧级 caption.
4. 多帧 QA 构造.
5. 时间戳对齐.
6. 去除无关片段.

---

### 6.2 难点

1. 标注成本高.
2. 帧数多导致 token 多.
3. 时间顺序难建模.
4. 视频中无关帧很多.
5. 长视频需要压缩.

Video 数据处理和 [Inference](./Inference.md) 中的 visual token 成本关系很大.

---

## 7. 医疗多模态数据

医疗 VLM 常见数据:

1. X-ray + report.
2. CT / MRI + diagnosis.
3. Pathology image + caption.
4. Medical VQA.
5. Medical document OCR.
6. 病历文本 + 检查图像.
7. Lesion box / mask.

---

### 7.1 清洗重点

1. **隐私脱敏**: 去除姓名, ID, 电话, 地址等 PHI.
2. **专业术语统一**: 处理同义词和缩写.
3. **报告质量过滤**: 去掉模板化, 空泛, 矛盾报告.
4. **图像质量过滤**: 去掉低质量, 错位, 模糊图像.
5. **标签一致性**: 检查影像和报告是否对应同一患者和检查.
6. **安全边界**: 避免训练模型输出危险医疗建议.

详见 [Medical_VLM](./Medical_VLM.md).

---

## 8. 数据清洗通用方法

### 8.1 去重

需要去重:

1. 完全相同图片.
2. 近似重复图片.
3. 重复 caption.
4. 同一问题重复答案.
5. 与评测集重复的样本.

去重的意义:

1. 避免模型过拟合重复样本.
2. 防止评测集污染.
3. 提高训练效率.

可参考 [Data_Process](../Pretrain/Data_Process.md) 中的数据去重部分.

---

### 8.2 图文相关性过滤

图文不相关会严重影响图文对齐.

常见做法:

1. 使用 CLIP score 过滤.
2. 使用 VLM judge 过滤.
3. 使用 OCR 检查图片文字和文本是否相关.
4. 用规则过滤广告和低质描述.
5. 人工抽样复查高风险来源.

---

### 8.3 安全和隐私过滤

需要过滤:

1. 隐私信息.
2. 不安全内容.
3. 版权风险数据.
4. 医疗敏感信息.
5. 未授权人脸和身份信息.

医疗场景中, 隐私过滤是必须项.

---

### 8.4 质量打分

可以给样本打质量分:

1. 图像清晰度.
2. 文本长度.
3. 图文相关性.
4. 答案准确性.
5. 是否包含可学习信息.
6. 是否需要视觉证据.

训练时可以:

1. 过滤低分样本.
2. 按分数采样.
3. 高质量数据加权.
4. 对不同任务设置不同采样比例.

---

## 9. 数据配比

VLM 训练不是数据越多越好, 还要注意配比.

如果 OCR 数据太少, 模型读字能力弱.

如果 caption 数据太多, 模型容易只描述图片, 不会回答复杂问题.

如果合成数据太多, 模型可能学到 teacher 的固定风格和错误.

常见配比思路:

1. 通用图像理解作为基础.
2. OCR / Document 数据提升读文字能力.
3. Reasoning 数据提升复杂问答.
4. Grounding 数据提升证据定位.
5. Video 数据提升时间理解.
6. 医疗数据用于领域适配.
7. 安全数据用于降低风险.

---

## 10. 数据和能力对应关系

|数据类型|主要提升能力|风险|
|---|---|---|
|Image-text pair|图文对齐|噪声大, 弱相关|
|Caption|描述能力|不一定会指令遵循|
|VQA|问答能力|答案可能过短|
|OCR / Document|读文字, 表格, 文档|高分辨率成本高|
|Grounding|定位和证据对齐|标注成本高|
|Video|时间理解|token 成本高|
|Medical|领域能力|隐私和安全风险高|
|Preference|回答偏好和安全性|标注标准难统一|

---

## 11. 总结

VLM 数据工程的核心是能力覆盖和质量控制.

1. Image-text pair 负责图文对齐.
2. Caption 数据负责生成描述.
3. Instruction data 负责指令遵循和问答.
4. OCR / Document 数据负责读文字和理解版面.
5. Grounding 数据负责区域定位和证据对齐.
6. Video 数据负责时间理解.
7. Medical 数据负责领域适配和安全边界.

不同数据类型需要合理配比, 否则模型容易在某些能力上明显短板.
