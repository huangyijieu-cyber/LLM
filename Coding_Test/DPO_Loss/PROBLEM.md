# DPO_Loss

## 题目

实现 DPO Loss

## 背景

这道题按 `Coding_Test/Examples` 中对应标准答案的前向逻辑构造，适合互联网大厂面试中的手撕深度学习基础模块。

## 要求

计算 logits=(policy_chosen-ref_chosen)-(policy_rejected-ref_rejected)，loss=-logsigmoid(beta*logits) 的均值；支持 label_smoothing。

只需要实现题目文件中的指定函数。测试脚本会直接 import 你的函数，并传入 Python list/dict；你不需要处理标准输入输出。

## 函数输入输出

输入字段：policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, label_smoothing(可选)。输出标量。

所有浮点输出保留合理精度即可，评测允许 `1e-5` 误差。

## 样例函数输入

```json
{"policy_chosen_logps":[1],"policy_rejected_logps":[0],"ref_chosen_logps":[0],"ref_rejected_logps":[0],"beta":1}
```

## 样例期望返回

```json
0.313262
```

## 使用方式

在当前目录实现 `DPO_Loss.py` 中的待实现函数，然后运行：

```powershell
python run_tests.py
```
