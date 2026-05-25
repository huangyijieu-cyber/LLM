import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    problem_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "run_tests.py").exists()
    )

    failed = []
    for problem in problem_dirs:
        proc = subprocess.run(
            [sys.executable, "run_tests.py"],
            cwd=problem,
            text=True,
            capture_output=True,
            timeout=10,
        )
        output = proc.stdout.strip() or proc.stderr.strip()
        print(f"{problem.name}: {output}")
        if proc.returncode != 0:
            failed.append(problem.name)

    if failed:
        print(f"FAILED {len(failed)}/{len(problem_dirs)}: {', '.join(failed)}")
        return 1
    print(f"ALL PASS {len(problem_dirs)}/{len(problem_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
