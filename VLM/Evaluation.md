# VLM Evaluation

VLM 评测需要回答两个问题: **模型具备哪些能力, 以及错误发生在哪一层**. 一条错误回答可能来自图像没有看清, OCR 没有读对, 图文没有对齐, 推理过程错误, 或者语言先验压过了视觉证据. 因此单一综合分数只能用于粗略比较, 不能说明模型是否适合具体任务.

比较完整的评测体系通常包含三层: 基础感知层考察物体, 属性和文字识别; 认知推理层考察文档, 图表和多步推理; 可靠性层考察 Grounding, Hallucination, 安全性和领域泛化. **评测主线是感知 -> 推理 -> 证据 -> 可靠性, 而不是只看一个总分.**

---

## 1. 基础视觉理解

### 1.1 Caption

Caption 任务要求模型生成图片描述. BLEU, ROUGE 和 METEOR 主要比较生成文本与参考文本的词语重合, CIDEr 会提高具有区分性的 n-gram 权重, SPICE 则更关注对象, 属性和关系组成的语义结构.

这些指标适合大规模自动评测, 但不能完全判断描述是否忠于图像. 例如 `A dog is running on grass` 和 `A brown dog plays in a field` 用词不同但语义接近; 反过来, 一段语言流畅且与参考答案相似的描述也可能加入图中不存在的对象. **文本相似度高不等于视觉事实正确**, 因此 Caption 指标需要和视觉事实一致性或人工评估结合使用.

### 1.2 VQA

VQA 的输入与输出形式为:

$$
(I,x) \rightarrow y
$$

封闭式问答通常使用 Accuracy, Exact Match 或 Multiple Choice Accuracy. 开放式回答则需要答案归一化, 语义匹配或 LLM / VLM judge. VQA 可以覆盖对象识别, 数量, 颜色, 属性, 空间关系, 常识推理和多图比较, 是最常用的 VLM 基础评测形式.

VQA 的主要风险是 **语言偏置**. 如果问题模板与答案分布存在强相关, 模型可能不看图也能猜中. 更可靠的数据会要求读取特定视觉细节, 使用对抗选项, 或同时输出证据区域.

---

## 2. OCR, 文档和图表

OCR 评测关注能否读取图片中的文字, Document Evaluation 进一步要求模型理解文字之间的二维布局, 表格结构和跨区域关系. 常见 benchmark 包括 TextVQA, OCRBench, DocVQA, ChartQA 和 InfoVQA.

**OCR 检查是否读对文字, Document QA 还要检查是否理解 layout 和结构.** 对文档问答, 模型需要先识别文字, 再确定文字位于标题, 表格单元格还是正文, 最后结合问题完成抽取或推理. 对 ChartQA, 还需要理解图例, 坐标轴和数值关系. 结构化抽取任务则应同时检查字段值和 JSON 等输出格式是否合法.

OCR 失败通常与分辨率不足, 字体和背景复杂, 图片压缩或 visual token 压缩过强有关. Document QA 还受 layout 建模影响. 这也是 [Architecture](./Architecture.md) 中 AnyRes, Dynamic Resolution, Patch Merger 等设计的重要评测场景.

|能力|典型评测|主要检查内容|
|---|---|---|
|Scene Text|TextVQA, OCRBench|自然图片中的小字和复杂背景文字|
|Document QA|DocVQA, InfoVQA|版面, 字段, 多区域信息整合|
|Chart / Table|ChartQA|图例, 坐标, 单元格关系和数值推理|
|Structured Extraction|自建 exact match / schema check|字段正确性, 格式合法性, 缺失值处理|

---

## 3. 综合理解和视觉推理

MMBench, SEED-Bench, MME 和 MMStar 用多个任务汇总通用视觉理解能力. [MMMU](https://arxiv.org/abs/2311.16502) 进一步覆盖 Art, Science, Engineering, Medicine, Business 和 Humanities 等大学学科, 强调专业知识与多模态推理, 难度高于普通 VQA.

Visual Reasoning Benchmark 关注模型能否在视觉感知之后继续完成多步计算或逻辑推理. [MathVista](https://arxiv.org/abs/2310.02255) 是典型例子, 题目可能同时要求读取图表或几何图形, 调用数学知识并生成最终答案. **视觉推理错误必须区分前端读取错误和后端推理错误.** 评测时最好保留中间识别结果, 例如先检查坐标轴数值, 再检查计算答案, 才能判断真正的能力瓶颈.

综合 benchmark 适合比较模型整体水平, 但容易掩盖任务分布差异. 一个模型在通用选择题上得分高, 不代表它能稳定处理高分辨率文档, 多图比较或专业医学影像. 实际选型仍应以目标场景的子能力为主.

---

## 4. Grounding 和视觉证据

Grounding 评估文本与图像区域是否真正对齐. 模型可能根据 referring expression 输出 bounding box, point 或 mask, 也可能根据指定区域生成描述. 对 bounding box, 常用指标是 Intersection over Union:

$$
\mathrm{IoU} = \frac{\mathrm{Area}(B_{\mathrm{pred}} \cap B_{\mathrm{gt}})}{\mathrm{Area}(B_{\mathrm{pred}} \cup B_{\mathrm{gt}})}
$$

当 IoU 超过设定阈值时认为定位正确. Pointing task 可以检查预测点是否落在目标区域内, referring expression 则同时考察语言理解和空间定位.

**Grounding 的价值不只是定位目标, 还在于检查回答是否存在可定位的视觉依据.** GUI Agent 需要定位正确控件, 医疗 VLM 需要指出可疑病灶, 文档问答可以返回答案所在区域. 需要注意图像 resize, crop 和 tile 会改变坐标系, 如果预处理后的坐标没有正确映射回原图, 模型语义正确也会被评为定位错误.

---

## 5. Hallucination 和可靠性

Visual Hallucination 指模型生成了图像中不存在的对象, 属性或关系. [POPE](https://arxiv.org/abs/2305.10355) 通过对象存在性问答评估 object hallucination, 例如图片中没有猫时询问 `Is there a cat in the image?`, 检查模型是否受常见共现关系影响而错误回答 `yes`.

幻觉通常来自语言先验过强, 视觉分辨率不足, 训练数据中的共现偏差, visual tokens 压缩过度, 或模型不愿表达不确定性. 改善方式包括提高视觉特征质量, 加入 Grounding 与反事实数据, 训练拒答和不确定性表达, 使用 [DPO](../Align/DPO.md) 等偏好对齐, 以及在高风险任务中要求模型给出视觉证据, self-check 或接受 verifier 检查. **语言更流畅不能抵消视觉事实错误.**

可靠性评测不能只统计回答正确率, 还应观察下面几种错误是否被混在一起:

|错误类型|例子|适合的检查方式|
|---|---|---|
|False Positive|图中不存在异常, 模型声称存在|负样本, POPE, specificity|
|False Negative|图中存在目标, 模型回答不存在|召回率, sensitivity|
|Attribute Hallucination|对象存在但颜色, 数量或状态错误|细粒度 VQA|
|Unsupported Reasoning|结论可能正确但证据不来自图像|Grounding, evidence judge|
|Overconfidence|信息不足时仍给出确定结论|校准, 拒答率, 专家评估|

---

## 6. 视频和医疗评测

Video VLM 需要同时评估空间内容和时间关系. Video QA 与 Caption 检查内容理解, Action Recognition 检查动作类别, Temporal Reasoning 检查事件顺序和状态变化, Long Video Understanding 则考察稀疏关键事件能否在长上下文中被保留. **视频评测必须同时报告采样策略, 视频长度和输入帧数**, 因为关键帧未被采样时, 失败不完全等同于语言模型失败. 推理侧影响见 [Inference](./Inference.md).

医疗 VLM 的常见数据集包括 VQA-RAD, SLAKE, PathVQA, PMC-VQA, MIMIC-CXR 和 IU X-Ray. 除医学问答和报告生成分数外, 还需要评估医学事实准确性, 图像证据一致性, 病灶定位, 幻觉率, 不确定性表达和安全性. 报告语言相似度不能替代临床有效性, 高风险结论应由医生复核. 具体见 [Medical_VLM](./Medical_VLM.md).

---

## 7. 评测方法设计

自动评测适合格式明确的任务. 选择题使用 Accuracy, OCR 使用 normalized exact match, Grounding 使用 IoU, 结构化输出可以使用 schema validator. 开放式回答很难由单一字符串指标覆盖, 可以引入强模型 judge, 但 judge 可能受提示词, 答案顺序和自身知识偏差影响.

因此重要评测通常采用 **规则指标 + 模型 judge + 人工或专家抽检**. Judge 负责语义正确性和表达质量, 规则检查数值, 坐标和格式, 专家评估高风险事实与安全边界. 还需要防止训练集与评测集重复, 并在不同分辨率, 医院, 设备或人群上测试分布外泛化.

|维度|代表评测|回答的问题|
|---|---|---|
|通用感知|VQA, MMBench, SEED-Bench|模型是否看懂基本内容|
|文字与文档|OCRBench, DocVQA, ChartQA|模型是否读对文字并理解布局|
|专业推理|MMMU, MathVista|模型能否结合视觉证据完成推理|
|证据定位|IoU, pointing accuracy|结论能否对应到正确区域|
|幻觉控制|POPE, negative set|模型是否编造视觉内容|
|时序理解|Video QA, temporal reasoning|模型是否理解事件变化|
|医疗可靠性|VQA-RAD, MIMIC-CXR, 专家评估|模型是否准确, 保守且安全|

---

## 8. 总结

VLM 评测的重点不是堆 benchmark, 而是建立从 **感知 -> 推理 -> 证据 -> 可靠性** 的诊断链路. 综合分数用于横向比较, OCR, Grounding, Hallucination 和领域评测用于发现具体短板. 面向真实应用时, 评测集必须覆盖实际输入分辨率, 数据分布和失败代价, 否则离线高分不能代表部署后的可靠性.
