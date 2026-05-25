# Coding Test Interview Suite

这个目录用于深度学习面试手撕练习。`Coding_Test/Examples` 里保留标准答案风格；每个正式题目目录是函数式单测练习题，更接近真实面试里“给函数签名、写核心逻辑”的形式。

## 目录结构

每道题都有一个独立目录，例如 `MoE`：

- `PROBLEM.md`：题目描述、函数输入输出格式、样例。
- `MoE.py`：你要手写的做题文件，只包含函数签名和待实现函数。
- `cases.json`：5 组测试数据和期望返回值。
- `run_tests.py`：本题测试脚本，会 import `MoE.py` 里的函数并直接调用。

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

- 你只实现题目文件中的指定函数。
- 测试脚本会直接 import 该函数并传入 Python list/dict。
- 函数返回 Python 标量、list 或 dict。
- 浮点误差：测试器允许 `1e-5` 误差。
