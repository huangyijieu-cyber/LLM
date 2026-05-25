# ROPE

## 题目

实现 Rotary Position Embedding

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

按 Examples 的 rotate_half 写法，对 xq 和 xk 应用 RoPE：x*cos + rotate_half(x)*sin。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：xq,xk,theta(可选)，形状 [batch, seq, heads, head_dim]。返回字典 {"xq": ..., "xk": ...}。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"xq":[[[[1,0,0,1]]]],"xk":[[[[0,1,1,0]]]]}
```

## 样例期望返回

```json
{"xq":[[[[1.0,0.0,0.0,1.0]]]],"xk":[[[[0.0,1.0,1.0,0.0]]]]}
```

## 使用方式

在当前目录实现 `ROPE.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
