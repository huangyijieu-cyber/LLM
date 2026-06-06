# LoRA 线性层

## 题目要求

请补全 `LoRA.py` 中的实现，使公开接口、参数含义、Tensor shape、返回值以及计算结果与对应的 `Coding_Test/Examples` 标准答案完全一致。

## 标准接口

```python
LoRALinear(in_features, out_features, rank=8, alpha=1.0, dropout=0.0)
```

题目使用 PyTorch。测试器会实例化你的实现和 `Examples` 标准实现，将相同的 `state_dict` 加载到两个模块中，再使用相同的 Tensor 输入进行对拍。

## 调用样例

```python
import torch

layer = LoRALinear(16, 32, rank=4)
x = torch.randn(2, 5, 16)
y = layer(x)
```

## 测试范围

- 共包含 5 组固定随机种子的测试样例。
- 模块题覆盖不同 batch、序列长度、维度以及 mask/交叉注意力等场景。
- 浮点结果使用 `rtol=1e-5, atol=1e-5` 比较。
- 返回类型与 `Examples` 不一致也会判错。

## 运行

```powershell
python run_tests.py
```
