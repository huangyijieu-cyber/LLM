# Layer_Norm

## 题目

实现 LayerNorm 前向计算

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

给定三维张量 x，对最后一维做均值方差归一化：y=(x-mean)/sqrt(var+eps)*gamma+beta。注意方差除以 N，而不是 N-1。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：x, gamma, beta, eps(可选)。输出归一化后的三维数组。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"x":[[[1,2,3]]],"gamma":[1,1,1],"beta":[0,0,0]}
```

## 样例期望返回

```json
[[[-1.224736,0.0,1.224736]]]
```

## 使用方式

在当前目录实现 `Layer_Norm.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
