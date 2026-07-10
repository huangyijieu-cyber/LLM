# VLM Evaluation

VLM 评测需要同时考察视觉感知, 语言生成, 推理, OCR, grounding 和安全性. 它比纯 LLM 评测更复杂, 因为错误可能来自:

1. 图像没看懂.
2. 文字没读出来.
3. 图文没对齐.
4. 推理错了.
5. 语言表达不安全.
6. 模型产生视觉幻觉.

---

## 1. Caption Evaluation

Caption 任务评估模型生成图片描述的能力.

常见指标:

- BLEU.
- ROUGE.
- METEOR.
- CIDEr.
- SPICE.

### 1.1 局限性

这些指标主要比较生成文本和参考文本的 n-gram 或语义结构, 但不一定能准确判断描述是否真的符合图片.

例如模型说:

```text
A dog is running on grass.
```

参考答案是:

```text
A brown dog plays in a field.
```

指标可能给中等分, 但它们其实语义接近.

因此现代 VLM 更常使用 benchmark QA 和 LLM/VLM judge 评估.

---

## 2. VQA Evaluation

VQA 评估模型根据图像回答问题的能力.

形式:

$$
(I,x) \rightarrow y
$$

常见指标:

- Accuracy.
- Exact Match.
- Multiple Choice Accuracy.

### 2.1 常见任务

1. 图片内容问答.
2. 物体数量.
3. 颜色和属性.
4. 空间关系.
5. 常识推理.

### 2.2 局限性

VQA 数据集有时存在语言偏置. 模型可能不看图, 仅根据问题猜答案.

因此更强评测会设计需要视觉证据的问题.

---

## 3. OCR Evaluation

OCR 能力是当前 VLM 重要能力.

常见评测:

- TextVQA.
- OCRBench.
- DocVQA.
- ChartQA.

评估点:

1. 能否读出图片中文字.
2. 能否理解文档布局.
3. 能否处理表格和图表.
4. 能否回答和文字位置相关的问题.

### 3.1 为什么 OCR 难

1. 文字很小.
2. 字体和背景复杂.
3. 图片压缩导致模糊.
4. 文档布局需要二维结构理解.
5. 高分辨率会增加 visual token 数量.

---

## 4. General Multimodal Benchmark

通用多模态 benchmark 关注综合能力.

常见 benchmark:

- MMBench.
- SEED-Bench.
- MMMU.
- MME.
- MMStar.

### 4.1 MMMU

[MMMU](https://arxiv.org/abs/2311.16502) 强调多学科大学级多模态理解.

它覆盖:

- Art.
- Science.
- Engineering.
- Medicine.
- Business.
- Humanities.

MMMU 更关注专业知识和多模态推理, 难度高于普通 VQA.

---

## 5. Visual Reasoning Evaluation

视觉推理评测关注多步思考能力.

典型任务:

- 图表数学题.
- 几何图形推理.
- 科学图像问答.
- 多图比较.

### 5.1 MathVista

[MathVista](https://arxiv.org/abs/2310.02255) 主要评估视觉数学推理.

它要求模型结合:

- 图像理解.
- 数学知识.
- 多步推理.
- 文本生成.

对 VLM 来说, MathVista 比普通图片问答更接近 reasoning 能力测试.

---

## 6. Grounding Evaluation

Grounding 评估模型能否把文本和图像区域对齐.

常见指标:

- IoU.
- Pointing accuracy.
- Referring expression accuracy.

如果任务是输出 bounding box:

$$
\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}
$$

当 IoU 超过阈值, 通常认为定位正确.

### 6.1 价值

Grounding 可以判断模型回答是否有视觉依据.

对医疗 VLM 来说, 如果模型指出异常, 最好能同时指出异常区域.

---

## 7. Hallucination Evaluation

VLM 幻觉指模型描述了图像中不存在的内容.

例如图片里没有猫, 模型却说:

```text
There is a cat on the sofa.
```

### 7.1 POPE

[POPE](https://arxiv.org/abs/2305.10355) 用来评估 object hallucination.

它通常问模型:

```text
Is there a cat in the image?
```

然后检查模型是否错误地回答 yes.

### 7.2 幻觉来源

1. 语言先验太强.
2. 图像细节没看清.
3. 训练数据中常见共现关系误导.
4. 模型倾向于给用户一个确定回答.

### 7.3 缓解方式

- 使用更高质量视觉编码器.
- 增加 grounding 数据.
- 加入拒答和不确定性数据.
- 使用 verifier 或 self-check.
- 做 preference alignment.

---

## 8. Medical VLM Evaluation

医疗 VLM 评测需要比通用 VLM 更谨慎.

常见数据集:

- VQA-RAD.
- SLAKE.
- PathVQA.
- PMC-VQA.
- MIMIC-CXR report generation.

评估能力:

1. 医学图像识别.
2. 医学术语理解.
3. 报告生成.
4. 病灶定位.
5. 医疗安全和保守回答.

详见 [Medical_VLM](./Medical_VLM.md).

---

## 9. LLM / VLM Judge

对于开放式回答, exact match 不够.

可以用强模型作为 judge:

1. 判断答案是否正确.
2. 判断是否引用了图像证据.
3. 判断是否存在幻觉.
4. 判断医疗建议是否安全.

### 9.1 风险

Judge 也可能错.

因此重要评测最好结合:

- 自动指标.
- 强模型 judge.
- 人工专家评估.
- 规则校验.

---

## 10. 评测矩阵

|能力|典型评测|关注点|
|---|---|---|
|Caption|CIDEr, BLEU, SPICE|图片描述质量|
|VQA|VQA accuracy|根据图像回答问题|
|OCR|TextVQA, OCRBench|读图中文字|
|Document|DocVQA, ChartQA|文档, 表格, 图表|
|General|MMBench, MMMU, SEED-Bench|综合多模态能力|
|Reasoning|MathVista|视觉数学和多步推理|
|Grounding|IoU, pointing accuracy|文本和区域对齐|
|Hallucination|POPE|是否编造不存在物体|
|Medical|VQA-RAD, SLAKE, PathVQA|医学图像理解和安全性|

一句话总结:

**VLM 评测不能只看一个总分, 需要拆成感知, OCR, reasoning, grounding, hallucination 和领域安全多个维度.**
