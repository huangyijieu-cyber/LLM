# LoRA

## 题目

实现 LoRA Linear 前向计算

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

实现 y = x @ weight + (x @ lora_a @ lora_b) * alpha/rank。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：x, weight, lora_a, lora_b, alpha。矩阵按 [in_dim, out_dim] 给出。输出三维数组。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"x":[[[1,2]]],"weight":[[1,0],[0,1]],"lora_a":[[1],[1]],"lora_b":[[0.5,-0.5]],"alpha":2}
```

## 样例期望返回

```json
[[[4.0,-1.0]]]
```

## 使用方式

在当前目录实现 `LoRA.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
