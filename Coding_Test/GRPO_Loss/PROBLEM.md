# GRPO_Loss

## 题目

实现 GRPO Loss

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

先对每组 rewards 做组内标准化优势 A=(R-mean)/(std+1e-8)，std 按样本标准差 N-1；再套 PPO clip，可选加 beta*ref_kl。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：rewards, old_log_probs, new_log_probs, clip_epsilon, beta, ref_kl(可选)。输出标量。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"rewards":[[1,2]],"old_log_probs":[[0,0]],"new_log_probs":[[0,0]],"clip_epsilon":0.2}
```

## 样例期望返回

```json
0.0
```

## 使用方式

在当前目录实现 `GRPO_Loss.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
