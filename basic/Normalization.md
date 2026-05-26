# Normalization
归一化(Normalization)是成功训练深度神经网络的关键技术之一. 它的演进体现了社区对**训练稳定性、梯度传播和系统硬件效率**的持续探索.
在深度学习训练过程中，数据或中间层的分布会不断变化（**Internal Covariate Shift**），导致：
- **训练不稳定**
- **收敛速度慢**
- **梯度消失 / 梯度爆炸**

这时就需要进行归一化了.

## 1. 常见归一化方法

### 1.1 Batch Normalization(BN)
在计算机视觉（CV）领域，**Batch Normalization (BN)**曾长期占据统治地位。BN 是在 **Batch 维度** 对每个特征(批次内所有样本)进行归一化.

### 1.2 Layer Normalization(LN)
LN 是对**单个样本内部的特征维度**进行归一化。

计算流程:
对于一个维度为 $d$ 的输入向量 $x = (x_1, x_2, \dots, x_d)$
，Layer Normalization 的计算包含两个步骤：标准化（Normalization）和仿射变换（Affine Transformation）。

首先，计算输入向量的均值 $\mu$ 和方差 $\sigma^2$ ：

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

接着，使用这两个统计量对输入进行标准化：

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

其中， $\epsilon$ 是一个微小的常数（如 1e-5），用于防止分母为零带来的数值不稳定。

最后，为了保证模型的表达能力（Expressivity）不被归一化操作限制，引入了可学习的缩放参数 $\gamma$ （Gain）和偏置参数 $\beta$ （Bias）：

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

在初始化阶段，通常将 $\gamma$ 设为 1， $\beta$ 设为 0，使得初始状态下的 LayerNorm 近似为恒等变换。

#### 为什么Transformer选择层归一化而非批次归一化?
1. **NLP中批次内序列的长度不一**, 如果进行 BN, 则需要考虑不同序列长度进行填充, 实现复杂且容易导致特征失真. 而 LN 只对每个 token 进行独立的归一化, 规避了这个问题.
2. **LLM 常因显存限制使用小 Batch**, 而 BN 在小 Batch 上统计噪声大, 而 LN 不受影响.
3. **语义特征不能被破坏**, 在 NLP 任务中, 每个 token 的语义往往依赖于上下文, 但不同样本间的 token 往往没有直接关系, BN 在 batch 层面强制归一化, 一定程度上破坏了 token 特征的独立性。

## 2. Post-Normalization vs Pre-Normalizaiton
在 Transformer 的架构演进史中，归一化层的位置选（Placement）是一个引发长期争论的话题。这一选择直接决定了模型的训练稳定性、收敛速度以及最终的性能上限。

### 2.1 Post-Normalization
在 Google 最初的论文《Attention Is All You Need》以及 BERT 模型中，采用的是 Post-Norm 结构。即归一化层被放置在残差连接（Residual Connection）之后：

$$
x_{l+1} = LN(x_l + Sublayer(x_l))
$$

Post-Norm的特性：

1. **梯度爆炸风险：** 在 Post-Norm 结构中，归一化层位于残差块的末端。从数学推导来看，随着网络层数 $L$ 的增加，输出端的方差是受控的，但在反向传播时，梯度需要穿过一系列的 LayerNorm 层。研究表明，Post-Norm 结构中的梯度范数在靠近输出层时较大，而在靠近输入层时会迅速衰减（梯度消失）或在某些初始化下剧烈震荡（梯度爆炸）。
2. **Warm-up 的必要性：** 由于初始阶段的梯度极不稳定，训练 Post-Norm 模型（尤其是深层模型）必须使用学习率预热（Warm-up）策略，即在训练初期使用极小的学习率，待优化器统计量稳定后再逐步增加。没有 Warm-up，Post-Norm 模型往往在训练初期就会发散。
3. **性能上限：** 尽管训练困难，Post-Norm 的一个显著优势在于，如果在精细调优的超参数下成功收敛，其最终的测试集性能（如 BLEU 分数或 Perplexity）往往略优于 Pre-Norm。这是因为 Post-Norm 保证了每一层的输入都是经过标准化的，充分利用了网络的非线性表达能力。

---

### 2.2 Pre-Normalization
为了解决 Post-Norm 的训练不稳定性，GPT-2、GPT-3 以及随后的 LLaMA、PaLM 等主流大模型转向了 Pre-Norm 结构。即归一化层被放置在子层的输入端，且位于残差分支内部：

$$
x_{l+1} = x_l + Sublayer(LN(x_l))
$$

1. 高速公路（Highway）效应: Pre-Norm 结构最核心的优势在于其创造了一条贯穿整个网络的“恒等路径”（Identity Path）, 在反向传播时，梯度可以直接沿着这条主干路径无损地传导至底层，而不需要经过非线性的归一化层。这极大地改善了梯度流，使得训练超深网络成为可能。

2. 移除 Warm-up: 由于梯度流稳定，Pre-Norm 模型通常不需要 Warm-up 阶段，或者可以使用更激进的学习率策略，显著缩短了训练初期的“爬坡”时间。这一特性对于训练成本高昂的大模型（如 175B 参数）至关重要。

3. “深度诅咒”与表征坍塌: 然而，Pre-Norm 并非完美。最新的研究指出，Pre-Norm 结构存在所谓的“深度诅咒”现象。由于主干路径 $x$ 的方差随着层数累积而线性增长，而残差分支的输出经过 LN 后方差被重置为 1，导致深层网络的输入 $x_l$ 幅度极大。由于 $LN(x)$ 对输入幅度的缩放不变性，实际上深层残差分支对主干的贡献权重会被隐式地缩小（相当于 $\frac{1}{L}$ ）。这意味着，在非常深的模型中，深层网络可能退化为恒等映射，对表征学习的贡献微乎其微。

---

## 3. RMSNorm

目前 LLM 主流的归一化操作是 **RMSNorm**, 它相对于经典的 LayerNorm 移去了均值中心化和偏置项. 原因是在深度神经网络中，激活值的均值通常在 0 附近波动，强制中心化的收益微乎其微，但计算均值却引入了额外的规约（Reduction）操作，增加了计算复杂度。

### 3.1 RMS的计算形式:

RMSNorm 省略了均值计算，直接使用均方根（Root Mean Square）进行归一化：

$$
RMS(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
\bar{x}_i = \frac{x_i}{RMS(x)} \cdot \gamma_i
$$

---

### 3.2 关键实现细节：

- **无偏置项 $\beta$ ：** 在 LLaMA 和 Mistral 的实现中，RMSNorm 进一步移除了仿射变换中的偏置项 $\beta$ ，仅保留缩放因子 $\gamma$ 。这不仅减少了参数量，还简化了算子实现，使其更加轻量化。
- **$\epsilon$ 的位置：** 在数值实现中， $\epsilon$ 必须加在平方和平均之后、开根号之前，以防止分母为零。常见的取值为 $1e^{-6}$ 。
- **Gemma 的修正：** Google 的 Gemma 模型在 RMSNorm 的基础上做了一个微小的调整，即 $RMSNorm(x) \cdot (1 + \gamma)$ 。这种设计使得在初始化阶段（ $\gamma \approx 0$ ）时，归一化层的输出接近于输入本身（Identity），有助于信号在深层网络中的初始传播，保留了类似 Pre-Norm 的直通特性。

### 3.3 相比于 LayerNorm 的优势:

1. **计算效率：** 省去了计算均值的操作, 尽管在总的 FLOPs 中占比较小, 但显著减少了内存带宽的压力(省去了一次全量扫描数据的操作, 显著降低了内存访问的开销), 这归根结底是因为 LayerNorm 是**内存带宽受限**的操作。
2. **工程优化：** RMSNorm 更容易进行算子融合, 同时少了偏置项 $\beta$ 参数。