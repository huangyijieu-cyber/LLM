# Attention

## 1. Transformer 中的 Attention

在 Transformer 中，所有的 Attention 计算都基于三个核心向量： $Q$ （Query，查询）、 $K$ （Key，键）、 $V$ （Value，值）。

### 1.1 Self Attention: 自注意力

自注意力是指 Q, K, V 均来源于同一个输入张量 **x**.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

1. **x** (`[B, L, D]`) 首先分别经过三个形状为 d_model * d_model (`[D, D]`) 权重矩阵 $W_Q$, $W_K$, $W_V$, 得到的 $Q$, $K$, $V$ 矩阵 (`[B, L, D]`).

2. 将 $Q$ (`[B, L, D]`) 与 $K^T$ (`[B, D, L]`) 进行点积, 得到注意力得分 Score (`[B, L, L]`).

3. 对 Score 进行缩放: $Score = \frac{Score}{\sqrt{D}}$, 维度不变.

4. 在维度 D 上对 Score 应用 Softmax 函数, 得到表示每个 token 对其余 token 关注程度的注意力权重矩阵 (0 到 1 之间的概率).

5. 将得到的注意力权重矩阵 (`[B, L, L]`) 与 $V$ (`[B, L, D]`)矩阵相乘, 得到注意力矩阵(`[B, L, D]`).

#### 为什么要在 Softmax 之前对 Score 除以 ${\sqrt{D}}$ 进行缩放?

这取决于 Softmax 函数的性质, 当数量级较大时, Softmax 会将所有概率分布全都分配给最大值对应的 label. 这使得 Softmax 后得到的注意力权重矩阵接近于一个 one-hot 向量, 梯度消失为0, 参数更新困难.
那为什么要除以 ${\sqrt{D}}$ 而非其余参数呢? 这里涉及一些数学推导, 简单来说就是, 假设 $Q$ 和 $K$ 为相互独立的随机变量(均值为 0, 方差为 1), 则点积 $Q  \cdot K$ 均值和方差则为 0 和 D. 方差越大表示点积的数量级也越大, 则一个自然的做法就是将方差稳定到 1, 即将点击除以 ${\sqrt{D}}$.

### 1.2 Mask Self-Attention: 掩码自注意力

通常应用于 Decoder 中. 为了保持自回归的特性, 在计算 Attention 时引入一个 **掩码矩阵 (Mask)**, 屏蔽当前时间步之后的信息, 使得当前位置 t 的输出只依赖于位置 t 及之前的输入.

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

1. 前期计算与标准的 Self Attention 完全一致, 直到计算出注意力得分 $Score = \frac{Score}{\sqrt{D}}$ (`[B, L, L]`).

2. 构建一个维度为 `[L, L]` 的上三角矩阵 (不包含对角线), 并将其广播到 `[B, L, L]` 上. 被 mask 的位置 (上三角) 设为 $-\infty$, 其余位置设置为 0.

3. 令 $Score_{masked} = Score + Mask$ ,维度始终保持为 `[B, L, L]`.

4. 进行 Softmax 操作, $-\infty$ 的位置变为 0, 未来特征的权重被设置为 0, 不会影响现在的输出.

5. 后续操作和 Self Attention 一样.

### 1.3 Multi-head Attention (MHA): 多头注意力

将 $Q$, $K$, $V$ 投影到 n_head (h) 个不同的低维子空间中并行计算 Attention, 最后将各个头的输出拼接（Concat）并进行一次线性映射以恢复原始维度. 它允许模型同时关注来自 **不同表示子空间** 的信息.

$$
\text{MultiHeadAttention}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

其中每个注意力头为：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

1. 根据输入 **x** 生成 $Q$, $K$, $V$ 矩阵 (`[B, L, D]`).

2. 将 $Q$, $K$, $V$ 在最后一个维度 D 上根据 D = h * $d_k$ (D') 进行拆分, 得到维度为 `[B, L, h, D']` 的矩阵.

3. 为了利用并行矩阵乘法计算 Attention, 交换 L 和 h 的维度,得到维度为 `[B, h, L, D']` 的矩阵. 这可视为 B x h 个独立样本进行 Attention 计算.

4. 后续计算和 Self Attention 一样, 都是计算 Score (`[B, h, L, L]`) 后进行 Softmax, 然后与 $V$ 矩阵相乘, 得到注意力矩阵 (`[B, h, L, D']`).

5. 将注意力矩阵维度转换回来(`[B, L, h, D']`), 然后将多头特征进行拼接(`[B, L, D]`), 最后经过一个维度为 `[D, D]` 的权重矩阵进行特征融合, 得到最终的注意力矩阵(`[B, L, D]`).

### 1.4 Cross Attention: 交叉注意力

在 Decoder 中, 模型需要根据源序列的信息来生成目标序列, 即 $Q$, $K$, $V$ 来源于不同的输入.

其中 $Q$ 来源于目标序列, $K$ 和 $V$ 来源于源序列.

注意力矩阵的计算过程与 Self Attention 基本相同, 仅涉及数据维度不同, 故下面只讨论维度变化:

1. 输入 $Q$ (`[B, L_q, D]`), $K$ (`[B, L_k, D]`), $V$ (`[B, L_k, D]`).

2. 计算得到 Score (`[B, L_q, L_k]`).

3. Softmax 后与 $V$ 矩阵相乘得到注意力矩阵 (`[B, L_q, D]`).

## 2. Attention 架构的改进

### 2.1 Multi-Query Attention (MQA):多查询注意力

MQA 为降低显存开销, 仅保留 $H$ 个Query头,但所有Query头共享同一个Key头和同一个Value头.

- **优势**: 这样做取得的直接好处是KV Cache的大小直接缩小了 $H$ 倍,极大地提高了吞吐量,同时支持超大 batch.
- **劣势**: 由于所有Query头只能在同一个Key-Value子空间中进行注意力检索,模型的表达能力受到限制,容易导致生成质量下降.同时MQA在训练初期容易出现不收敛的情况.

### 2.2 Grouped-Query Attention (GQA):分组查询注意力

GQA是相对于 MHA 和 MQA 的折中方案,现已成为主流LLM的标配. GQA将Query头分成 $G$ 个组(Group),每组包含 $H/G$ 个Query头.每个组共享一个Key头和一个Value头.

GQA可以看作是MHA和MQA的泛化形式.当 $G=H$ 时,它就是MHA;当 $G=1$ 时,它就是MQA.通常选取 $G=8$ 作为一个甜点(Sweet Spot).

- **优势**: GQA在精度上接近MHA,同时在速度上也接近MQA,显著降低了显存需求.