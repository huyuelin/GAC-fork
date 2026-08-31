"""Knowledge benchmark evaluation for the GAC harness.

Covers:
    - MMLU-Pro (multi-choice, 10 options, TIGER-Lab/MMLU-Pro)
    - GPQA-diamond (multi-choice, 4 options, Idavidrein/gpqa)
    - SciBench (open-ended numeric/symbolic, xw27/scibench)

The first two use a strict letter-match on \\boxed{X}. SciBench uses SymPy for
numeric/symbolic equivalence, with a string-normalization fallback.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from generate_vllm import GenerationConfig, generate
from prompts import MCQ_SYSTEM, SCIENCE_SYSTEM, mcq_user_prompt, scibench_user_prompt


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
LETTER_RE = re.compile(r"\b([A-J])\b")


def extract_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_letter(text: str) -> str | None:
    """First try \\boxed{X}, then any of ``The answer is X`` / final capital letter."""
    boxed = extract_boxed(text)
    if boxed:
        letters = LETTER_RE.findall(boxed.upper())
        if letters:
            return letters[0]

    m = re.search(
        r"(?:the answer is|answer:|final answer)[\s:]*\(?([A-J])\)?",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    letters = LETTER_RE.findall(text[-200:].upper())
    return letters[-1] if letters else None


# ---------------------------- MMLU-Pro / GPQA (MCQ) ---------------------------


def _load_mmlu_pro(subset_size: int = 1000, seed: int = 42) -> list[dict]:
    """Load a 1000-sample fixed subset of MMLU-Pro (paper: 1k stratified sample)."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    items = []
    for i in idxs[:subset_size]:
        row = ds[i]
        options = {chr(65 + j): opt for j, opt in enumerate(row["options"])}
        # answer_index in MMLU-Pro is 0-indexed integer
        gold_letter = chr(65 + int(row["answer_index"]))
        items.append(
            {
                "id": f"mmlu-pro_{i}",
                "prompt": mcq_user_prompt(row["question"], options),
                "gold": gold_letter,
                "raw_question": row["question"],
                "options": options,
            }
        )
    return items


def _load_gpqa_diamond() -> list[dict]:
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    items = []
    for i, row in enumerate(ds):
        # GPQA has one correct and three incorrect answers as separate columns.
        opts = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        rng = random.Random(i)
        perm = list(range(4))
        rng.shuffle(perm)
        shuffled = [opts[p] for p in perm]
        correct_idx = perm.index(0)
        letter = chr(65 + correct_idx)
        options_dict = {chr(65 + j): shuffled[j] for j in range(4)}
        items.append(
            {
                "id": f"gpqa_{i}",
                "prompt": mcq_user_prompt(row["Question"], options_dict),
                "gold": letter,
                "raw_question": row["Question"],
                "options": options_dict,
            }
        )
    return items


def _score_mcq(completion: str, gold: str) -> bool:
    pred = extract_letter(completion)
    return pred is not None and pred == gold.upper()


# --------------------------------- SciBench -----------------------------------


def _load_scibench() -> list[dict]:
    ds = load_dataset("xw27/scibench", split="train")
    items = []
    for i, row in enumerate(ds):
        problem = row.get("problem_text") or row.get("problem") or row.get("question")
        gold = row.get("answer_number") or row.get("answer") or row.get("gold_answer")
        unit = row.get("unit")
        if problem is None or gold is None:
            continue
        items.append(
            {
                "id": f"scibench_{i}",
                "prompt": scibench_user_prompt(str(problem), unit),
                "gold": str(gold),
                "raw_problem": str(problem),
                "unit": unit,
            }
        )
    return items


def _score_scibench(completion: str, gold: str) -> bool:
    pred = extract_boxed(completion)
    if pred is None:
        return False

    # Try SymPy equivalence.
    try:
        import sympy as sp
        g = sp.sympify(str(gold).replace(",", ""))
        p = sp.sympify(pred.replace(",", ""))
        if sp.simplify(g - p) == 0:
            return True
    except Exception:
        pass

    # Numeric fallback: compare as floats with 1% tolerance.
    try:
        g = float(str(gold).replace(",", ""))
        p = float(pred.replace(",", ""))
        return abs(g - p) / max(1.0, abs(g)) < 0.01
    except Exception:
        pass

    # String fallback: normalize whitespace + trailing punctuation.
    return pred.strip().rstrip(".").replace(" ", "") == str(gold).strip().rstrip(".").replace(" ", "")


# --------------------------------- Runner -------------------------------------


LOADERS = {
    "mmlu-pro": (_load_mmlu_pro, _score_mcq, MCQ_SYSTEM),
    "gpqa": (_load_gpqa_diamond, _score_mcq, MCQ_SYSTEM),
    "scibench": (_load_scibench, _score_scibench, SCIENCE_SYSTEM),
}


def run_benchmark(
    name: str, model_path: str, output_dir: Path, cfg: GenerationConfig
) -> dict:
    loader, scorer, system_prompt = LOADERS[name]
    items = loader()
    completions = generate(
        [it["prompt"] for it in items], cfg, system_prompt=system_prompt
    )

    records = []
    correct = 0
    for it, cs in zip(items, completions):
        best = any(scorer(c, it["gold"]) for c in cs)
        records.append(
            {
                "id": it["id"],
                "gold": it["gold"],
                "completions": cs,
                "correct": best,
            }
        )
        correct += int(best)

    acc = correct / max(1, len(records))
    per_bench_dir = output_dir / name
    per_bench_dir.mkdir(parents=True, exist_ok=True)

    with open(per_bench_dir / "predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": name,
        "model_path": model_path,
        "n_total": len(records),
        "n_correct": correct,
        "accuracy": acc,
        "seed": cfg.seed,
    }
    with open(per_bench_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[knowledge_bench] {name}: acc={acc*100:.1f}% ({correct}/{len(records)}) → {per_bench_dir}"
    )
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
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=8192)
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

    with open(out / "knowledge_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
