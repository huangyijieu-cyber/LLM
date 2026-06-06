# Coding Test Interview Suite

这个目录用于深度学习面试手撕练习。`Coding_Test/Examples` 保存标准答案；正式题目的接口、Tensor shape 和返回格式与对应标准答案完全一致。

## 目录结构

每道题都有一个独立目录，例如 `MoE`：

- `PROBLEM.md`：题目描述、函数输入输出格式、样例。
- `MoE.py`：你要补全的 PyTorch 模块或函数。
- `run_tests.py`：本题测试入口。

公共测试器会加载对应的 `Examples` 标准实现。对于 `nn.Module` 题目，测试器会给你的实现和标准实现加载相同的 `state_dict`，再使用相同 Tensor 输入对拍。

## 使用方式

1. 打开某个题目的 `PROBLEM.md`，先读题。
2. 在同目录的 `<题名>.py` 中实现待实现函数。
3. 进入该目录运行测试：

```powershell
cd Coding_Test\MoE
python run_tests.py
```

也可以在 `Coding_Test` 根目录运行全部题目：

```powershell
python run_all_tests.py
```

注意：初始模板会抛出 `NotImplementedError`，这是正常的。你实现完成后，对应测试才会通过。

## 测试约定

所有题目都采用同一种模式：

- 使用 PyTorch Tensor，不再使用 JSON/list 模拟张量。
- 类名、函数名、初始化参数、forward 参数及返回类型必须与对应 `Examples` 一致。
- 每题包含 5 组固定随机种子的测试；GRPO 会分别测试三个公开函数。
- 浮点结果使用 `rtol=1e-5, atol=1e-5` 比较。
