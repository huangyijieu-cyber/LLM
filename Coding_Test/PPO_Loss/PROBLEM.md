# PPO_Loss

## 题目

实现 PPO Clip Loss

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

计算 ratio=exp(new-old)，返回 -mean(min(ratio*A, clamp(ratio,1-eps,1+eps)*A))。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：old_log_probs, new_log_probs, advantages, clip_epsilon(可选)。输出标量。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"old_log_probs":[0],"new_log_probs":[0],"advantages":[1],"clip_epsilon":0.2}
```

## 样例期望返回

```json
-1.0
```

## 使用方式

在当前目录实现 `PPO_Loss.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
