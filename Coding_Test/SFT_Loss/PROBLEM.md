# SFT_Loss

## 题目

实现 SFT Loss

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

先把每个样本 prompt_lengths 之前的位置 label 置为 -100，再做 next-token shift，最后计算 ignore_index=-100 的平均交叉熵。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：logits, labels, prompt_lengths。输出标量。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"logits":[[[1,2,3],[2,1,0],[0,0,1]]],"labels":[[0,2,1]],"prompt_lengths":[1]}
```

## 样例期望返回

```json
0.907606
```

## 使用方式

在当前目录实现 `SFT_Loss.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
