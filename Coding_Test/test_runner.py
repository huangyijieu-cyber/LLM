import copy
import importlib.util
import inspect
import json
from pathlib import Path


CALLS = {
    "Layer_Norm": ("layer_norm", lambda d: (d["x"], d["gamma"], d["beta"], d.get("eps", 1e-5))),
    "RMS_Norm": ("rms_norm", lambda d: (d["x"], d["gamma"], d.get("eps", 1e-8))),
    "FFN": ("ffn", lambda d: (d["x"], d["w_up"], d["b_up"], d["w_down"], d["b_down"])),
    "SwiGLU": ("swiglu", lambda d: (d["x"], d["w_gate"], d["w_up"], d["w_down"])),
    "Self_Attention": ("scaled_dot_product_attention", lambda d: (d["q"], d["k"], d["v"], d.get("mask"))),
    "Multi-Attention": ("multi_head_attention", lambda d: (d,)),
    "GQA": ("group_query_attention", lambda d: (d,)),
    "ROPE": ("rotary_embedding", lambda d: (d["xq"], d["xk"], d.get("theta", 10000.0))),
    "LoRA": ("lora_linear", lambda d: (d["x"], d["weight"], d["lora_a"], d["lora_b"], d["alpha"])),
    "MoE": ("moe_forward", lambda d: (d["x"], d["router"], d["experts"], d["top_k"])),
    "SFT_Loss": ("sft_loss", lambda d: (d["logits"], d["labels"], d["prompt_lengths"])),
    "DPO_Loss": ("dpo_loss", lambda d: (d,)),
    "PPO_Loss": ("ppo_clip_loss", lambda d: (d["old_log_probs"], d["new_log_probs"], d["advantages"], d.get("clip_epsilon", 0.2))),
    "GRPO_Loss": ("grpo_loss", lambda d: (d,)),
}


def close(a, b, tol=1e-5):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


def find_case_dir():
    case_dir = Path.cwd()
    if (case_dir / "cases.json").exists():
        return case_dir

    for frame in inspect.stack():
        candidate = Path(frame.filename).resolve()
        if candidate.name == "run_tests.py":
            return candidate.parent

    raise RuntimeError("找不到 cases.json 所在目录")


def load_solution(script):
    spec = importlib.util.spec_from_file_location("solution", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    case_dir = find_case_dir()
    script = case_dir / f"{case_dir.name}.py"
    func_name, build_args = CALLS[case_dir.name]
    func = getattr(load_solution(script), func_name)
    cases = json.loads((case_dir / "cases.json").read_text(encoding="utf-8"))

    passed = 0
    for i, case in enumerate(cases, 1):
        try:
            got = func(*build_args(copy.deepcopy(case["input"])))
        except Exception as exc:
            print(f"case {i}: runtime error\n{type(exc).__name__}: {exc}")
            return 1

        if not close(got, case["output"]):
            print(f"case {i}: wrong answer\ninput={case['input']}\nexpected={case['output']}\ngot={got}")
            return 1
        passed += 1

    print(f"PASS {passed}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
