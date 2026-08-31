"""Aggregate per-seed summaries into mean±std tables.

Usage:
    python aggregate.py results/seed_0 results/seed_1 results/seed_2

Reads ``{math,knowledge,code,bbh_logic}_summary.json`` from each directory and
prints a paper-style report table with mean, std, and n_seeds.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


TOPLINES = ["math_summary.json", "knowledge_summary.json", "code_summary.json", "bbh_logic_summary.json"]


def _flatten(summary: dict) -> dict[str, float]:
    """Extract {benchmark: score} from one top-level summary file."""
    out: dict[str, float] = {}
    for name, data in summary.items():
        if not isinstance(data, dict):
            continue
        if name.startswith("_"):
            # e.g. _bbh_logic_avg
            if data.get("accuracy") is not None:
                out[name.lstrip("_")] = float(data["accuracy"])
            continue
        if data.get("accuracy") is not None:
            out[name] = float(data["accuracy"])
        elif data.get("pass@1") is not None:
            out[name] = float(data["pass@1"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed_dirs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    all_seeds: dict[str, list[float]] = defaultdict(list)
    for d in args.seed_dirs:
        if not d.exists():
            print(f"[aggregate] warn: {d} does not exist")
            continue
        for f in TOPLINES:
            path = d / f
            if not path.exists():
                continue
            with open(path) as fh:
                summary = json.load(fh)
            for k, v in _flatten(summary).items():
                all_seeds[k].append(v)

    print(f"{'Benchmark':<24} {'mean':>10} {'std':>10} {'n_seeds':>8}")
    print("-" * 55)
    rows = []
    for k in sorted(all_seeds):
        vs = all_seeds[k]
        mean = statistics.mean(vs) * 100
        std = statistics.stdev(vs) * 100 if len(vs) > 1 else 0.0
        rows.append({"benchmark": k, "mean_pct": mean, "std_pct": std, "n_seeds": len(vs)})
        print(f"{k:<24} {mean:>9.2f}% {std:>9.2f}  {len(vs):>7d}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[aggregate] wrote {args.output}")


if __name__ == "__main__":
    main()
