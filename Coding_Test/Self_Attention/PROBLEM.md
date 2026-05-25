# Self_Attention

## 题目

实现 Scaled Dot-Product Attention

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

实现 Attention(Q,K,V)=softmax(QK^T/sqrt(d))V，并支持 mask 中 0 位置不可见。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：q,k,v,mask(可选)，形状为 [batch, heads, seq, dim]。返回字典 {"output": ..., "attn_weights": ...}。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"q":[[[[1,0]]]],"k":[[[[1,0]]]],"v":[[[[5,6]]]]}
```

## 样例期望返回

```json
{"output":[[[[5.0,6.0]]]],"attn_weights":[[[[1.0]]]]}
```

## 使用方式

在当前目录实现 `Self_Attention.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
