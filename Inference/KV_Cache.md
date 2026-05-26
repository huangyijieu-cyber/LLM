# KV_Cache

KV Cache（键值缓存） 是大模型推理中最基础的优化技术，其核心目的是以空间换时间，消除自回归生成中的冗余计算。

在Transformer的每一层中，输入Token经过投影矩阵 $W_Q$, $W_K$, $W_V$ 变换为查询（Query）、键（Key）和值（Value）向量。

## 1. 运行机制

### 1.1 无缓存推理
每生成一个 Token，都把整个历史序列重新喂给 Transformer。复杂度 $O(T^2)$ 甚至 $O(T^3)$

### 1.2 有缓存推理

- 首Token生成（Prefill）：计算Prompt中所有Token的 $K$, $V$ ，并将其存入显存中的特定区域（Cache）。
- 后续生成（Decode）：当生成第 $n$ 个Token时，仅需计算当前Token的 $q_n$, $k_n$, $v_n$ 。
- 拼接与检索：将 $k_n$, $v_n$ 追加到缓存末尾。注意力计算变为 $q_n$ 与 $K_{cache} + k_n$ 的交互。

通过缓存之前的键值对，我们可以专注于计算新token的注意力。

![img](https://miro.medium.com/v2/resize:fit:1400/1*uyuyOW1VBqmF5Gtv225XHQ.gif)

### 2. KV_Cache 带来的显存开销

虽然KV Cache将计算复杂度从 $O(L^2)$ 降低到了 $O(L)$ （线性扫描历史），但它引入了巨大的显存开销。缓存的大小与以下因素成正比：

- 层数（Layers）
- 批处理大小（Batch Size）
- 序列长度（Sequence Length）
- 注意力头数（Heads）与维度（Head Dimension）

公式如下：

$$
M_{KV} = 2 \times N_{layers} \times N_{heads} \times D_{head} \times L_{seq} \times B_{batch} \times P_{size}
$$

随着Batch Size的增加（为了提高吞吐量）和Sequence Length的延长（长文档分析），KV Cache的体积往往会超过模型权重本身，成为限制并发量的主要瓶颈。这直接催生了注意力架构的演进。

#### 为什么不缓存 Query ?

这是一个常见的误解。在注意力机制中，Query代表当前处理的token对之前所有token的查询，它是随着生成而不断变化的。例如，在生成第10个词时，我们关注的是第10个词与前9个词的关系因此，Query向量在每一步都是全新的，必须重新计算。而历史Token作为被关注的对象（Key）和信息载体（Value），其特征是固定的，因此可以被缓存 。
