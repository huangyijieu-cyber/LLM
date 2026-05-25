# SwiGLU

## 题目

实现 SwiGLU FFN

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

实现 output = (SiLU(x @ w_gate) * (x @ w_up)) @ w_down。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：x, w_gate, w_up, w_down。输出三维数组。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"x":[[[1,2]]],"w_gate":[[1,0],[0,1]],"w_up":[[2,1],[1,2]],"w_down":[[1,0],[0,1]]}
```

## 样例期望返回

```json
[[[2.924234,8.807971]]]
```

## 使用方式

在当前目录实现 `SwiGLU.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
