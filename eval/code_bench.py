"""Code generation benchmark for the GAC harness (MBPP + HumanEval).

For scoring we call out to the community-standard
``bigcode-evaluation-harness`` (Apache-2.0). This gives us
sandboxed pass@k execution without re-implementing the runner.

If ``bigcode-evaluation-harness`` is not installed, the script writes
completions to disk and exits with instructions.

Usage:
    python code_bench.py \\
        --model_path <ckpt> --benchmarks mbpp humaneval \\
        --output_dir ./results --tp_size 4 --n_samples 1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset

from generate_vllm import GenerationConfig, generate
from prompts import CODE_SYSTEM, humaneval_user_prompt, mbpp_user_prompt


PY_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str:
    """Extract a python code block from a completion. Fall back to raw text."""
    m = PY_BLOCK_RE.search(completion)
    if m:
        return m.group(1).strip()
    return completion.strip()


def _load_humaneval() -> list[dict]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "task_id": row["task_id"],
            "prompt": humaneval_user_prompt(row["prompt"]),
            "raw_prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in ds
    ]


def _load_mbpp() -> list[dict]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    return [
        {
            "task_id": f"mbpp_{row['task_id']}",
            "prompt": mbpp_user_prompt(row["text"], row["test_list"]),
            "raw_text": row["text"],
            "test_list": row["test_list"],
            "code": row.get("code"),
        }
        for row in ds
    ]


LOADERS = {"humaneval": _load_humaneval, "mbpp": _load_mbpp}


def _write_bigcode_generations(
    items: list[dict],
    completions: list[list[str]],
    path: Path,
) -> None:
    """Write generations in the shape bigcode-evaluation-harness expects
    when loaded via ``--load_generations_path``: a list-of-lists JSON."""
    all_gens = [[extract_code(c) for c in cs] for cs in completions]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_gens, f)


def _run_bigcode_eval(
    task: str, gen_path: Path, output_dir: Path
) -> dict | None:
    """Call bigcode-evaluation-harness for scoring. Returns metrics dict or None."""
    if shutil.which("accelerate") is None and shutil.which("python") is None:
        return None
    metric_path = output_dir / f"{task}_metrics.json"
    cmd = [
        "python", "-m", "bigcode_eval.main",
        "--tasks", task,
        "--load_generations_path", str(gen_path),
        "--allow_code_execution",
        "--metric_output_path", str(metric_path),
        "--save_generations",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError as e:
        print(f"[code_bench] bigcode-eval-harness failed for {task}:\n{e.stderr[:500]}")
        return None
    if metric_path.exists():
        with open(metric_path) as f:
            return json.load(f)
    return None


def run_benchmark(
    name: str, model_path: str, output_dir: Path, cfg: GenerationConfig
) -> dict:
    items = LOADERS[name]()
    completions = generate(
        [it["prompt"] for it in items], cfg, system_prompt=CODE_SYSTEM
    )

    per_bench_dir = output_dir / name
    per_bench_dir.mkdir(parents=True, exist_ok=True)

    # Save raw predictions.
    with open(per_bench_dir / "predictions.jsonl", "w") as f:
        for it, cs in zip(items, completions):
            f.write(
                json.dumps(
                    {
                        "task_id": it["task_id"],
                        "completions": cs,
                        "extracted_code": [extract_code(c) for c in cs],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Emit bigcode-compatible generations file for scoring.
    gen_path = per_bench_dir / "generations.json"
    _write_bigcode_generations(items, completions, gen_path)

    metrics = _run_bigcode_eval(
        task=name.replace("mbpp", "mbpp").replace("humaneval", "humaneval"),
        gen_path=gen_path,
        output_dir=per_bench_dir,
    )

    if metrics is None:
        summary = {
            "benchmark": name,
            "model_path": model_path,
            "n_total": len(items),
            "seed": cfg.seed,
            "n_samples": cfg.n_samples,
            "score": None,
            "note": (
                "bigcode-evaluation-harness not installed or failed. "
                "Install with:  git clone https://github.com/bigcode-project/bigcode-evaluation-harness && "
                "cd bigcode-evaluation-harness && pip install -e ."
                " Then rerun scoring via:  python -m bigcode_eval.main "
                f"--tasks {name} --load_generations_path {gen_path} "
                "--allow_code_execution"
            ),
        }
    else:
        pass_at_1 = metrics.get(name, {}).get("pass@1")
        summary = {
            "benchmark": name,
            "model_path": model_path,
            "n_total": len(items),
            "pass@1": pass_at_1,
            "raw_metrics": metrics,
            "seed": cfg.seed,
            "n_samples": cfg.n_samples,
        }

    with open(per_bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[code_bench] {name}: summary → {per_bench_dir / 'summary.json'}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--benchmarks", nargs="+", default=list(LOADERS.keys()),
        choices=list(LOADERS.keys()),
    )
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)  # code: lower T
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = GenerationConfig(
        model_path=args.model_path,
        tp_size=args.tp_size,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for name in args.benchmarks:
        all_summaries[name] = run_benchmark(name, args.model_path, out, cfg)

    with open(out / "code_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
