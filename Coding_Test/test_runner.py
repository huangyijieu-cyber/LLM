import ast
import importlib.util
import inspect
import math
import sys
import types
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    print("缺少 PyTorch。请在安装了 torch 的 Python 环境中运行测试。")
    raise SystemExit(1)


REFERENCE_FILES = {
    "Layer_Norm": "normalization/LayerNorm.py",
    "RMS_Norm": "normalization/RMSNorm.py",
    "FFN": "ffn/FFN.py",
    "SwiGLU": "ffn/SwiGLUFFN.py",
    "MoE": "ffn/MoE.py",
    "Self_Attention": "attention/ScaledDotProductAttention.py",
    "Multi-Attention": "attention/MultiHeadAttention.py",
    "GQA": "attention/GroupQueryAttention.py",
    "ROPE": "position/RotaryEmbedding.py",
    "LoRA": "peft/LoRALinear.py",
    "SFT_Loss": "loss/SFTLoss.py",
    "DPO_Loss": "loss/DPOLoss.py",
    "PPO_Loss": "loss/PPOLoss.py",
    "GRPO_Loss": "loss/GRPOLoss.py",
}


def find_problem_dir():
    current = Path.cwd()
    if (current / "run_tests.py").exists():
        return current
    for frame in inspect.stack():
        path = Path(frame.filename).resolve()
        if path.name == "run_tests.py":
            return path.parent
    raise RuntimeError("找不到题目目录")


def load_student(path):
    spec = importlib.util.spec_from_file_location("student_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    module = types.ModuleType("reference_solution")
    module.__dict__.update({"torch": torch, "nn": nn, "F": F, "math": math})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def assert_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"返回类型应为 torch.Tensor，实际为 {type(actual).__name__}")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        return
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError("返回序列的类型或长度不一致")
        for actual_item, expected_item in zip(actual, expected):
            assert_close(actual_item, expected_item)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("返回字典的键不一致")
        for key in expected:
            assert_close(actual[key], expected[key])
        return
    if actual != expected:
        raise AssertionError(f"expected={expected}, got={actual}")


def sync_module(student_cls, reference_cls, init_args):
    reference = reference_cls(*init_args)
    student = student_cls(*init_args)
    student.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    student.eval()
    return student, reference


def causal_mask(batch, q_len, k_len):
    mask = torch.tril(torch.ones(q_len, k_len))
    return mask.view(1, 1, q_len, k_len).expand(batch, 1, q_len, k_len)


def module_cases(name):
    if name in {"Layer_Norm", "RMS_Norm"}:
        return [
            ((4,), (torch.randn(2, 3, 4),)),
            ((8,), (torch.randn(1, 5, 8),)),
            ((3, 1e-6), (torch.randn(3, 2, 3),)),
            ((6,), (torch.zeros(2, 1, 6),)),
            ((2,), (torch.tensor([[[1.0, -1.0], [2.0, 4.0]]]),)),
        ]
    if name == "FFN":
        return [
            ((4, 8), (torch.randn(2, 3, 4),)),
            ((6, 12), (torch.randn(1, 5, 6),)),
            ((3, 4), (torch.zeros(2, 2, 3),)),
            ((8, 16), (torch.randn(3, 1, 8),)),
            ((2, 3), (torch.tensor([[[1.0, -1.0], [0.0, 2.0]]]),)),
        ]
    if name == "SwiGLU":
        return [
            ((4, 8), (torch.randn(2, 3, 4),)),
            ((6, 10), (torch.randn(1, 5, 6),)),
            ((3, 5), (torch.zeros(2, 2, 3),)),
            ((8, 12), (torch.randn(3, 1, 8),)),
            ((2, 4), (torch.tensor([[[1.0, -1.0], [0.0, 2.0]]]),)),
        ]
    if name == "Self_Attention":
        return [
            ((0.0,), (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 6), None)),
            ((0.0,), (torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4), causal_mask(1, 5, 5))),
            ((0.0,), (torch.randn(2, 1, 3, 8), torch.randn(2, 1, 4, 8), torch.randn(2, 1, 4, 2), torch.ones(2, 1, 3, 4))),
            ((0.0,), (torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 2, 4), torch.randn(1, 1, 2, 4), None)),
            ((0.0,), (torch.randn(1, 4, 1, 2), torch.randn(1, 4, 1, 2), torch.randn(1, 4, 1, 2), None)),
        ]
    if name == "Multi-Attention":
        return [
            ((8, 2, 0.0), (torch.randn(2, 4, 8), None, None)),
            ((8, 4, 0.0), (torch.randn(1, 5, 8), None, causal_mask(1, 5, 5))),
            ((12, 3, 0.0), (torch.randn(2, 3, 12), torch.randn(2, 5, 12), None)),
            ((4, 1, 0.0), (torch.zeros(1, 2, 4), None, None)),
            ((16, 8, 0.0), (torch.randn(1, 1, 16), torch.randn(1, 3, 16), None)),
        ]
    if name == "GQA":
        return [
            ((8, 4, 2, 0.0), (torch.randn(2, 4, 8), None)),
            ((8, 4, 1, 0.0), (torch.randn(1, 5, 8), causal_mask(1, 5, 5))),
            ((12, 6, 3, 0.0), (torch.randn(2, 3, 12), None)),
            ((4, 2, 2, 0.0), (torch.zeros(1, 2, 4), None)),
            ((16, 8, 2, 0.0), (torch.randn(1, 1, 16), None)),
        ]
    if name == "ROPE":
        return [
            ((4, 16, 10000.0), (torch.randn(2, 5, 3, 4), torch.randn(2, 5, 3, 4))),
            ((8, 32, 10000.0), (torch.randn(1, 7, 2, 8), torch.randn(1, 7, 2, 8))),
            ((2, 8, 1000.0), (torch.randn(3, 4, 1, 2), torch.randn(3, 4, 1, 2))),
            ((6, 16, 10000.0), (torch.zeros(1, 3, 2, 6), torch.zeros(1, 3, 2, 6))),
            ((4, 4, 5000.0), (torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4))),
        ]
    if name == "LoRA":
        return [
            ((4, 6, 2, 4.0, 0.0), (torch.randn(2, 3, 4),)),
            ((8, 4, 4, 8.0, 0.0), (torch.randn(1, 5, 8),)),
            ((3, 3, 1, 1.0, 0.0), (torch.zeros(2, 2, 3),)),
            ((6, 2, 2, 2.0, 0.0), (torch.randn(3, 1, 6),)),
            ((2, 5, 1, 3.0, 0.0), (torch.tensor([[[1.0, -1.0], [0.0, 2.0]]]),)),
        ]
    if name == "MoE":
        return [
            ((4, 3, 1), (torch.randn(2, 3, 4),)),
            ((4, 4, 2), (torch.randn(1, 5, 4),)),
            ((6, 3, 2), (torch.randn(2, 2, 6),)),
            ((3, 2, 1), (torch.zeros(1, 4, 3),)),
            ((8, 4, 3), (torch.randn(1, 1, 8),)),
        ]
    if name == "SFT_Loss":
        return [
            ((), (torch.randn(2, 5, 7), torch.randint(0, 7, (2, 5)), torch.tensor([2, 1]))),
            ((), (torch.randn(1, 4, 3), torch.randint(0, 3, (1, 4)), torch.tensor([0]))),
            ((), (torch.randn(3, 6, 8), torch.randint(0, 8, (3, 6)), torch.tensor([1, 2, 3]))),
            ((), (torch.zeros(2, 3, 4), torch.tensor([[0, 1, 2], [3, 2, 1]]), torch.tensor([1, 1]))),
            ((), (torch.randn(1, 2, 5), torch.tensor([[2, 4]]), torch.tensor([0]))),
        ]
    raise KeyError(name)


def run_module_problem(name, student_module, reference_module):
    class_names = {
        "Layer_Norm": "LayerNorm",
        "RMS_Norm": "RMSNorm",
        "FFN": "FFN",
        "SwiGLU": "SwiGLUFFN",
        "Self_Attention": "ScaledDotProductAttention",
        "Multi-Attention": "MultiHeadAttention",
        "GQA": "GroupQueryAttention",
        "ROPE": "RotaryEmbedding",
        "LoRA": "LoRALinear",
        "MoE": "MoE",
        "SFT_Loss": "SFTLoss",
    }
    class_name = class_names[name]
    student_cls = getattr(student_module, class_name)
    reference_cls = getattr(reference_module, class_name)
    torch.manual_seed(2026)
    for index, (init_args, forward_args) in enumerate(module_cases(name), 1):
        torch.manual_seed(1000 + index)
        student, reference = sync_module(student_cls, reference_cls, init_args)
        with torch.no_grad():
            expected = reference(*forward_args)
            actual = student(*forward_args)
        assert_close(actual, expected)
        print(f"case {index}: PASS")


def function_cases(name):
    if name == "DPO_Loss":
        return [
            (torch.randn(4), torch.randn(4), torch.randn(4), torch.randn(4), 0.1, 0.0),
            (torch.randn(8), torch.randn(8), torch.randn(8), torch.randn(8), 0.5, 0.1),
            (torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3), 1.0, 0.0),
            (torch.tensor([1.0, -1.0]), torch.tensor([0.0, 1.0]), torch.zeros(2), torch.zeros(2), 0.2, 0.2),
            (torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1), 0.01, 0.0),
        ]
    if name == "PPO_Loss":
        return [
            (torch.randn(4), torch.randn(4), torch.randn(4), 0.2),
            (torch.randn(8), torch.randn(8), torch.randn(8), 0.1),
            (torch.zeros(3), torch.zeros(3), torch.ones(3), 0.2),
            (torch.tensor([0.0, 0.0]), torch.tensor([0.5, -0.5]), torch.tensor([1.0, -1.0]), 0.3),
            (torch.randn(1), torch.randn(1), torch.randn(1), 0.05),
        ]
    if name == "GRPO_Loss":
        return [
            (torch.randn(4), torch.randn(4), torch.randn(4), 0.2, 0.01, None),
            (torch.randn(8), torch.randn(8), torch.randn(8), 0.1, 0.05, torch.rand(8)),
            (torch.zeros(3), torch.zeros(3), torch.ones(3), 0.2, 0.01, None),
            (torch.tensor([0.0, 0.0]), torch.tensor([0.5, -0.5]), torch.tensor([1.0, -1.0]), 0.3, 0.1, torch.tensor([0.2, 0.4])),
            (torch.randn(1), torch.randn(1), torch.randn(1), 0.05, 0.0, None),
        ]
    raise KeyError(name)


def run_function_problem(name, student_module, reference_module):
    function_names = {
        "DPO_Loss": ["dpo_loss"],
        "PPO_Loss": ["ppo_clip_loss"],
        "GRPO_Loss": ["compute_grpo_advantages", "grpo_loss", "compute_kl_penalty"],
    }
    torch.manual_seed(2026)
    for function_name in function_names[name]:
        student_fn = getattr(student_module, function_name)
        reference_fn = getattr(reference_module, function_name)
        if function_name == "compute_grpo_advantages":
            cases = [(torch.randn(2, 4),), (torch.zeros(3, 3),), (torch.tensor([[1.0, 2.0, 3.0]]),)]
        elif function_name == "compute_kl_penalty":
            cases = [(torch.randn(4), torch.randn(4)), (torch.zeros(3), torch.zeros(3)), (torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0]))]
        else:
            cases = function_cases(name)
        for index, args in enumerate(cases, 1):
            expected = reference_fn(*args)
            actual = student_fn(*args)
            assert_close(actual, expected)
            print(f"{function_name} case {index}: PASS")


def main():
    problem_dir = find_problem_dir()
    name = problem_dir.name
    root = problem_dir.parent
    student = load_student(problem_dir / f"{name}.py")
    reference = load_reference(root / "Examples" / REFERENCE_FILES[name])
    try:
        if name in {"DPO_Loss", "PPO_Loss", "GRPO_Loss"}:
            run_function_problem(name, student, reference)
        else:
            run_module_problem(name, student, reference)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
