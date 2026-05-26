# Filtering_Algorithm (过滤算法)

## 1. N-gram 语言模型 (N-gram Language Models)

N-gram模型是过滤流水线的基础工具, 用于评估文本质量.

其核心思想基于马尔可夫假设 (当前词仅依赖前 n-1 个词): 给定一个 Token 序列 $x_1, x_2, ..., x_n$, N-gram 模型将其概率分解为:

```math
P(x_1, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_{i-n+1}, ..., x_{i-1})
```

- **Unigram (n=1)**: $P(x_i)$ - 只看当前词
- **Bigram (n=2)**: $P(x_i | x_{i-1})$ - 看前一个词
- **Trigram (n=3)**: $P(x_i | x_{i-2}, x_{i-1})$ - 看前两个词

## 2. KenLM

KenLM 是一个高度优化的N-gram语言模型库, 支持:

- **修改Kneser-Ney平滑**: 比简单的加一平滑效果更好
- **回退机制 (Backoff)**: 当高阶n-gram未见过时，回退到低阶
- **压缩存储**: 使用Trie结构和量化技术

## 3. fastText

## 4. 重要性重采样 (Data Selection via Importance Resampling, DSIR)