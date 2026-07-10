# SGLang

## 1. 核心概念

SGLang 是一个面向大模型和多模态模型的推理服务框架. 它既提供前端编程接口, 也提供高性能后端 runtime, 主要用于构建 **高吞吐, 低延迟, 可结构化控制** 的 LLM 应用. 它和 [vLLM](./vllm.md) 一样属于推理框架, 不负责模型训练.

SGLang 的核心思想是:

**把复杂的大模型调用流程表达成可编程的 language program, 同时用高性能 runtime 管理 batch, [KV cache](../Inference/KV_Cache.md), prefix reuse 和 decoding.**

它通常适合以下场景:

1. 在线聊天服务.
2. Agent 和多轮工具调用.
3. 结构化输出, 例如 JSON, regex, schema.
4. 多请求共享长前缀的推理任务.
5. reasoning, code, retrieval-augmented generation 等复杂生成流程.

---

## 2. 核心组成

### 2.1 Frontend: SGLang Language

SGLang 的前端提供一种描述大模型程序的方式. 用户可以把多轮对话, 分支, 并行调用, 约束生成等逻辑写成一个程序.

它关注的是:

- Prompt 组织.
- 多轮上下文管理.
- 多个生成步骤之间的依赖关系.
- 工具调用或函数调用.
- 结构化输出约束.

这使得 SGLang 不只是一个简单的 `generate()` 接口, 而是更适合描述复杂 LLM workflow.

---

### 2.2 Backend: Runtime Engine

SGLang 的后端 runtime 负责实际推理执行.

核心能力包括:

1. **Continuous Batching**: 动态合并不同请求, 提升 GPU 利用率.
2. **[KV Cache](../Inference/KV_Cache.md) Management**: 管理请求生成过程中的 KV cache.
3. **Prefix Reuse**: 对共享前缀进行复用, 避免重复 prefill.
4. **Parallel Decoding**: 支持多请求, 多分支或多样本生成.
5. **Model Serving API**: 提供服务化接口, 方便在线部署.

---

### 2.3 RadixAttention

SGLang 的一个重要特点是 **RadixAttention**. 它可以看作是在 [KV Cache](../Inference/KV_Cache.md) 复用上的系统级优化.

普通推理系统中, 如果两个请求有相同前缀, 但不是在同一个 batch 中执行, 往往仍然会重复计算前缀部分.

RadixAttention 的思路是:

**用 radix tree 组织历史请求的 prefix [KV cache](../Inference/KV_Cache.md), 当新请求到来时, 自动查找可复用的最长前缀.**

这对以下任务非常有用:

- 多轮对话中保留长历史.
- RAG 中多个问题共享相同文档上下文.
- Agent 中多个分支共享同一个系统提示词和工具描述.
- Best-of-N 或 self-consistency 中多个采样共享 prompt.

其核心收益是:

1. 减少重复 prefill 计算.
2. 降低长上下文请求延迟.
3. 提高多请求服务吞吐.

---

### 2.4 Structured Output

SGLang 支持结构化输出, 可以约束模型按照指定格式生成.

常见约束包括:

- JSON.
- Regex.
- Choice.
- Schema.
- Function call 格式.

结构化输出的意义在于:

**让 LLM 的输出更容易被程序消费, 减少格式错误, 提高下游系统稳定性.**

例如信息抽取任务中, 我们希望模型输出:

```json
{
  "name": "...",
  "time": "...",
  "location": "..."
}
```

如果没有约束解码, 模型可能输出解释性文字或格式错误 JSON. 使用 structured output 后, runtime 会在生成过程中限制可选 token, 使输出更符合目标格式.

---

### 2.5 多模态和模型支持

SGLang 不只面向纯文本 LLM, 也支持部分多模态模型.

它通常关注:

- Text generation.
- Vision-language model serving.
- Embedding 或 rerank 等辅助任务.
- OpenAI-compatible API.

具体支持哪些模型和特性, 需要参考当前版本文档. 推理框架的模型适配变化较快, 新模型通常需要 runtime 支持其 attention, tokenizer, processor 和 generation config.

---

## 3. 使用流程

### 3.1 启动服务

典型流程是:

1. 选择模型.
2. 启动 SGLang server.
3. 指定 tensor parallel size, context length, memory fraction 等参数.
4. 暴露 HTTP 或 OpenAI-compatible 接口.

服务启动后, 应用侧可以像调用普通 Chat Completion API 一样发送请求.

---

### 3.2 编写生成程序

如果使用 SGLang 前端, 可以把生成逻辑写成一个程序:

1. 构造 system prompt.
2. 添加用户输入.
3. 调用模型生成中间结果.
4. 根据中间结果决定下一步 prompt.
5. 使用 structured output 约束最终答案.

这类方式适合复杂 Agent, 因为它比单次 prompt 更容易管理中间状态.

---

### 3.3 优化推理性能

常见优化方向:

- 开启 prefix cache.
- 合理设置 batch 和并发.
- 使用 tensor parallel.
- 控制最大输出长度.
- 对长上下文场景使用 cache reuse.
- 对结构化任务使用 constrained decoding.

---

## 4. 特点

1. **适合复杂 LLM 程序**: SGLang 不只是 serving engine, 还强调用程序化方式组织多步生成流程.

2. **Prefix reuse 能力强**: RadixAttention 对共享前缀场景非常友好, 可以减少大量重复 prefill.

3. **结构化输出支持好**: 适合 JSON 抽取, function calling, agent tool call 等需要稳定格式的任务.

4. **服务化能力强**: 可以作为线上推理服务, 支持高并发请求.

5. **适合 Agent 场景**: 多轮对话, 多分支生成, 工具调用和约束输出都比较贴合 Agent 工作流. 如果只是追求通用高吞吐 serving, 可以对比 [vLLM](./vllm.md).

---

## 5. 局限性

1. **主要面向推理, 不是训练框架**: SGLang 不负责模型预训练或微调.

2. **复杂特性依赖模型适配**: 并不是所有模型都能完整支持 structured output, 多模态, 高效 cache 等能力.

3. **线上调优有成本**: batch size, [KV cache](../Inference/KV_Cache.md), 显存比例, 并发数等参数需要根据业务负载调优.

4. **对动态工作流更友好, 对简单 batch 推理未必最必要**: 如果只是离线跑简单 `generate()`, 使用 Transformers 或 [vLLM](./vllm.md) 可能已经足够.

5. **版本变化较快**: 推理框架通常更新很快, API 和模型支持列表需要以当前官方文档为准.

---

## 6. SGLang 和 vLLM 的区别

|对比项|SGLang|[vLLM](./vllm.md)|
|---|---|---|
|核心定位|推理 runtime + LLM 编程接口|高吞吐推理 serving engine|
|关键特点|RadixAttention, structured output, language program|PagedAttention, continuous batching, OpenAI API|
|适合场景|Agent, 多步生成, 结构化输出, 共享前缀|通用高吞吐在线服务, rollout, batch serving|
|训练能力|不负责训练|不负责训练|
|上下文复用|强调 radix tree prefix reuse|强调 prefix cache 和 [KV cache](../Inference/KV_Cache.md) 管理|
|应用层表达|更强|较弱|

一句话总结:

**SGLang 更像是一个面向复杂 LLM 应用的推理编程和服务框架, 特别适合共享前缀, 结构化输出和 Agent 工作流.**
