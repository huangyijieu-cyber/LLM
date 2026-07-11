# VLM Training

VLM 训练需要逐步解决三个问题: 视觉特征是否具有语言语义, visual tokens 是否能被 LLM 正确读取, 模型是否会依据图像执行具体指令. 因此现代 VLM 通常采用多阶段训练, 而不是一开始就把所有模块和数据混在一起更新.

$$
\text{Vision-Language Pretraining}
\rightarrow
\text{Projector Alignment}
\rightarrow
\text{Multimodal Instruction Tuning}
\rightarrow
\text{Preference Alignment}
\rightarrow
\text{Domain Adaptation}
$$

每个阶段的边界并非固定. 有些模型直接复用 CLIP / SigLIP 视觉塔, 因而跳过视觉预训练; 有些模型进行 native multimodal pretraining, 会较早联合更新 Vision Encoder 和 LLM. **训练实现可以合并阶段, 但图文表征, 接口对齐, 指令能力和偏好约束仍是四类不同目标.**

---

## 1. 训练阶段总览

|阶段|主要数据|常见训练模块|解决的问题|
|---|---|---|---|
|图文表征预训练|大规模 image-text pair|Vision Encoder, Text Encoder|建立图像和文本语义对齐|
|生成式预训练|image-caption / interleaved data|VLM 或 Connector + LLM|学习根据视觉内容生成文本|
|Projector Alignment|caption, 简单 VQA|Projector|把视觉特征映射到 LLM space|
|Instruction Tuning|多任务 multimodal instruction|Projector + LLM, 可选 Vision Encoder|按指令完成 VQA, OCR, Grounding 等任务|
|Preference Alignment|chosen / rejected 或可验证 rollout|VLM|减少幻觉, 改善安全与输出偏好|
|Domain Adaptation|Medical, Document, GUI 等领域数据|按领域选择|适配专业分布和安全边界|

训练流程通常从稳定对齐逐步过渡到能力扩展. 前期数据规模大但监督较弱, 后期数据规模较小但任务和质量要求更高.

---

## 2. 视觉语言预训练

### 2.1 Contrastive Pretraining

[CLIP](https://arxiv.org/abs/2103.00020) 对一批图文对 $\{(I_i,T_i)\}_{i=1}^{B}$ 分别编码, 提高匹配 pair 的相似度并降低不匹配 pair 的相似度. [SigLIP](https://arxiv.org/abs/2303.15343) 将 pair 匹配改为独立的 sigmoid 二分类目标. **对比学习建立图文表示空间, 但不直接训练长回答生成.** 这类训练得到的 [Vision Encoder](./Vision_Encoder.md) 天然带有语言语义, 也保留较好的 zero-shot 与 retrieval 能力.

这类目标更关注全局图文关系, 对 OCR, 文档布局和医学细节未必充分. 因此强视觉塔只是 VLM 的起点, 不能替代后续多模态生成训练.

### 2.2 Generative Pretraining

Caption 或 interleaved image-text pretraining 直接训练模型根据图像预测文本:

$$
L_{\mathrm{gen}} = -\sum_{t=1}^{T}\log P(y_t \mid I,y_{1:t-1})
$$

**生成式预训练让 visual tokens 真正参与自回归预测, 因而比纯对比学习更接近 VQA 和对话模型.** 但 Caption 通常只描述显著内容, 不要求遵循复杂指令, 也不覆盖 Grounding, 多轮对话和结构化输出. 如果 Caption 占比过高, 模型容易形成 "看到图片就描述" 的默认行为.

一些新模型使用 native multimodal pretraining, 在较早阶段把图像, 文本和交错序列一起训练. 这样视觉与语言融合更充分, 但训练成本, 数据清洗和稳定性要求都明显高于复用冻结视觉塔.

---

## 3. Projector Alignment

LLaVA 类模型通常先冻结 Vision Encoder 和 LLM, 只训练 [Projector](./Projector.md):

$$
H_v = f_{\mathrm{proj}}(X_v)
$$

该阶段使用 image-caption pair 或简单 VQA, 让 $H_v$ 落到 LLM 能够解释的 hidden space. **Projector Alignment 解决的是接口可读性, 不是复杂任务能力.** 如果跳过基础对齐, LLM 会直接看到分布陌生的视觉 embedding, 后续 SFT 容易收敛慢或破坏已有语言能力.

Projector Alignment 的目标不是让模型掌握所有任务. 数据过于简单时, 模型只会建立 "视觉特征对应什么文本" 的粗粒度映射, 复杂推理仍然依赖 Instruction Tuning. 对齐数据如果图文弱相关或包含幻觉, 则会从接口层污染后续训练, 表现为模型忽略图像, visual tokens 语义不稳定或 instruction stage 学习效率低.

---

## 4. Multimodal Instruction Tuning

Multimodal Instruction Tuning 是 VLM 从图文生成模型变成视觉助手的关键阶段. 一条样本通常表示为 $(I,x,y)$, 其中 $I$ 是图像, $x$ 是用户指令, $y$ 是目标回答:

```text
USER: <image>
What is abnormal in this X-ray?

ASSISTANT: ...
```

训练目标与 [SFT](../Finetune/Main.md) 相同, 只是在条件中加入视觉输入:

$$
L_{\mathrm{SFT}} = -\sum_{t=1}^{T}\log P(y_t \mid I,x,y_{1:t-1})
$$

Instruction data 需要覆盖 General VQA, OCR QA, Document QA, Chart QA, Grounding, Multi-Image QA, Video QA 和 Medical QA. **不同任务不是同一种能力的不同数据集: OCR 学读取, Grounding 学定位, Video 学时间关系, Medical 学专业域和安全边界.** 具体配比见 [Data_Process](./Data_Process.md).

常见训练配置是冻结 Vision Encoder, 全量训练 Projector, 并对 LLM 使用 [LoRA](../Finetune/PEFT.md) 或全参数微调. 当目标任务与视觉塔预训练分布差异较大, 例如医疗影像或高分辨率文档, 可以解冻 Vision Encoder 后部层. 全模块联合训练的上限更高, 但应对视觉塔和 LLM 使用较小 learning rate, 避免通用能力快速遗忘.

Instruction Tuning 的主要问题来自数据分布. 合成 QA 占比过高会继承 teacher 的语气和错误; Caption 太多会削弱指令遵循; 短答案太多会让模型缺乏解释能力; 某类任务不足则会形成明确短板. 训练 loss 下降并不能证明模型在看图, 还需要用视觉依赖样本和 [Evaluation](./Evaluation.md) 中的分项 benchmark 验证.

---

## 5. Preference Alignment

SFT 让模型模仿标准回答, 但不能直接优化两个都合理回答之间的偏好. 多模态偏好数据通常写为:

$$
(I,x,y_w,y_l)
$$

其中 $y_w$ 比 $y_l$ 更符合视觉证据, 幻觉更少, 格式更稳定或医疗表达更安全. 可以使用 [DPO](../Align/DPO.md) 直接提高 chosen answer 的相对概率. **多模态偏好的差异必须来自视觉正确性, 而不能只来自语言流畅度.** 否则模型可能改善文风却没有减少视觉幻觉.

[RLHF](../Align/RLHF.md) 可以使用 reward model 评价开放式回答. 对有明确验证规则的任务, [GRPO](../Align/GRPO.md) 或 [DAPO](../Align/DAPO.md) 更容易构造可靠 reward, 例如 OCR exact match, Chart QA 数值答案, Grounding box IoU, Medical multiple choice 和 JSON schema. 对同一输入生成多个回答并用 verifier 打分, 可以直接优化任务结果. 大规模 rollout 可结合 [vLLM](../Framework/vllm.md), [SGLang](../Framework/SGLang.md) 与 [verl](../Framework/Verl.md).

偏好优化适合改善幻觉, 拒答, 不确定性和格式, 但不能弥补 Vision Encoder 根本看不清图像的问题. 如果视觉信息在前端已经丢失, 继续增加语言侧 reward 可能只会让模型更自信地猜测.

---

## 6. 专项能力训练

### 6.1 OCR, Document 和 Grounding

OCR / Document 训练通常使用 TextVQA 类数据, 高分辨率截图, 表格, 图表, PDF 页面和多页文档. 除文字内容外, 数据需要保留 layout, 行列关系和字段位置. 结构化抽取任务还应约束 schema, 缺失字段和数值格式. 这类训练与 Dynamic Resolution, Patch Merger 等架构设计共同决定最终效果.

Grounding 数据可以写为 $(\mathrm{image},\mathrm{text},\mathrm{box})$ 或 $(\mathrm{image},\mathrm{region},\mathrm{answer})$, 将文本与 box, point 或 mask 对齐. 它既能训练目标定位, 也能要求模型为回答提供视觉证据. 医疗病灶定位与 GUI click 都依赖同一基础能力, 但坐标系统和评价标准不同. 图像预处理发生 resize 或 crop 时, 训练标签必须同步变换, 否则模型会学到错误位置.

### 6.2 Video

Video VLM 将视频表示为帧序列 $V = \{I_1,I_2,\dots,I_T\}$. Video Caption, Video QA 和 action recognition 提供内容监督, temporal reasoning 强调状态变化和先后关系. **只有问题必须依赖多帧时, Video QA 才能真正训练时间理解.** 训练时还需要记录帧顺序, 加入时间位置, 并用帧采样或 token compression 控制上下文, 相关成本见 [Inference](./Inference.md).

### 6.3 Medical Domain Adaptation

医疗 VLM 通常从通用 VLM 初始化, 再使用 X-ray image-report pair, CT / MRI finding, pathology caption, 医学教材图像和医疗文档 OCR 建立领域对应, 使用医疗指令数据训练问答与报告生成, 最后加入医生偏好, Grounding 和安全数据. 医疗训练不仅要提升术语和病灶识别, 还要教模型区分影像发现与最终诊断, 表达不确定性并建议专业复核. 详见 [Medical_VLM](./Medical_VLM.md).

---

## 7. 工程和稳定性

多模态训练同时包含图像预处理, Vision Encoder, Projector 和 LLM, 显存通常受 activation 与长 visual sequence 影响. 大规模训练可使用 [DeepSpeed](../Framework/DeepSpeed.md) 或 [Megatron](../Framework/Megatron.md), 并结合 gradient checkpointing, mixed precision 和 [LoRA](../Finetune/PEFT.md). Dynamic Resolution 会使 batch 内序列长度变化, 需要按 visual token 数做 batching 或设置每批 token budget, 否则 padding 浪费和 OOM 会很不稳定.

**训练监控不能只记录 total loss.** 更有价值的是分别观察 Caption, VQA, OCR, Grounding 和纯文本数据的 loss 或验证指标, 同时检查模型是否保留 text-only 能力. 当模型开始忽略图像, 常见原因包括语言数据过强, 视觉输入与答案弱相关, Projector 学习不足或 image placeholder / label mask 实现错误.

---

## 8. 总结

VLM 训练的主线是 **先建立视觉语言表示, 再对齐接口, 然后学习任务和偏好**. Projector Alignment 负责让视觉特征可读, Instruction Tuning 决定能力覆盖, Preference Alignment 改善可靠性, 专项训练补充 OCR, Grounding, Video 和 Medical 等能力. 各阶段必须与数据来源和模块更新范围对应, 否则把所有数据混合训练只会让问题更难定位.
