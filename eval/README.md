# GAC Evaluation Harness

This directory contains the evaluation pipeline used to reproduce the numbers reported in the GAC paper (Tables 1-4). It covers **8 benchmarks across 4 domains**:

| Domain | Benchmarks | Script |
|---|---|---|
| Math | AMC · AIME24 · AIME25 | `math_bench.py` |
| Knowledge | MMLU-Pro · GPQA · SciBench | `knowledge_bench.py` |
| Code | MBPP · HumanEval | `code_bench.py` |
| Logic | BBH (Log-Ded · Obj-Count · Tracking) | `bbh_logic.py` |

## Design

**Generation** is unified via `generate_vllm.py` — a single vLLM-based generator that takes a prompt file (jsonl), a system template, and produces raw completions. This keeps the sampling logic identical across all 8 benchmarks (temperature = 0.6, top-p = 0.95, max_new_tokens = 8192, matching the paper).

**Scoring** uses domain-appropriate tooling:

| Domain | Scorer | Rationale |
|---|---|---|
| Math | [`math-verify`](https://github.com/huggingface/Math-Verify) | Canonical for AIME/AMC/MATH — the same scorer used by DeepSeek-R1, Qwen2.5-Math, LUFFY. Avoids false negatives from equivalent-but-formatted-differently answers (e.g. `\frac{1}{2}` vs `0.5`). |
| Code | [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness) (via subprocess) | Sandboxed pass@k execution with the canonical MBPP/HumanEval prompts and unit tests. |
| Multi-choice (MMLU-Pro, GPQA) | In-house letter matcher | Regex-extract `\boxed{}` / final "The answer is (X)" answer letter; strict A/B/C/D match. |
| Open-ended (SciBench) | In-house SymPy checker + LLM-as-judge fallback | SciBench answers are numeric or expressions — SymPy handles unit normalization; ambiguous cases delegate to a small judge model. |
| BBH-Logic | In-house exact-match (case-insensitive, whitespace-normalized) | BBH answers are short strings ("(A)", "yes", "3") — exact match after normalization is standard. |

## Quick start

### 1 · Install

```bash
pip install -r requirements.txt
```

Additional deps for code eval:
```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
cd bigcode-evaluation-harness && pip install -e .
```

### 2 · Run everything

```bash
bash run_all.sh \
    --model_path /path/to/gac-checkpoint-or-any-hf-model \
    --output_dir ./results \
    --tp_size 4 \
    --seed 0
```

Runs all 8 benchmarks sequentially. On 4×A100/A800, wall-clock is ~90 minutes for a 7B model.

### 3 · Run individual benchmarks

```bash
# Math (AMC/AIME24/AIME25)
python math_bench.py \
    --model_path <model> \
    --benchmarks amc aime24 aime25 \
    --output_dir ./results \
    --tp_size 4

# Knowledge (MMLU-Pro/GPQA/SciBench)
python knowledge_bench.py \
    --model_path <model> \
    --benchmarks mmlu-pro gpqa scibench \
    --output_dir ./results \
    --tp_size 4

# Code (MBPP/HumanEval)
python code_bench.py \
    --model_path <model> \
    --benchmarks mbpp humaneval \
    --output_dir ./results \
    --tp_size 4 \
    --n_samples 1     # pass@1

# Logic (BBH)
python bbh_logic.py \
    --model_path <model> \
    --benchmarks logical_deduction object_counting tracking_shuffled_objects \
    --output_dir ./results \
    --tp_size 4
```

## Reproducing paper numbers

The paper reports mean ± std over **3 seeds**. To reproduce:

```bash
for seed in 0 1 2; do
  bash run_all.sh --model_path <ckpt> --output_dir ./results/seed_${seed} --seed $seed
done
python aggregate.py ./results/seed_{0,1,2}   # prints mean±std across seeds
```

Expected numbers for `GAC + Token-φ` on Qwen2.5-7B:

| Benchmark | GAC + Token-φ (paper) |
|---|---|
| AMC | 67.2 ±0.4 |
| AIME24 | 20.8 ±0.4 |
| AIME25 | 19.8 ±0.5 |
| MMLU-Pro | 58.6 ±0.3 |
| MBPP | 78.8 ±0.5 |
| HumanEval | 83.5 ±0.4 |
| GPQA | 43.5 ±0.5 |
| SciBench | 41.2 ±0.5 |
| BBH-Logic (avg) | 65.7 ±0.5 |

## Data

Benchmark datasets are auto-downloaded from HuggingFace on first run and cached under `~/.cache/huggingface/`. Sources:

| Benchmark | HuggingFace dataset ID | Split |
|---|---|---|
| AMC | `AI-MO/aimo-validation-amc` | test (83 problems) |
| AIME24 | `HuggingFaceH4/aime_2024` | train (30 problems) |
| AIME25 | `math-ai/aime25` | test (30 problems) |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | test (12k problems, we use 1k-sample fixed subset) |
| GPQA | `Idavidrein/gpqa` | main (198 problems, diamond subset) |
| SciBench | `xw27/scibench` | test (695 problems) |
| MBPP | `google-research-datasets/mbpp` | test (500 problems) |
| HumanEval | `openai/openai_humaneval` | test (164 problems) |
| BBH | `lukaemon/bbh` | logical_deduction / object_counting / tracking_shuffled_objects (~250 each) |

Prompt templates are in `prompts.py`. All benchmarks use the same chat template as the base model (Qwen2.5 by default).

## Output format

Each benchmark writes two files per run:

```
results/
├── amc.jsonl            # per-problem: {id, prompt, completion, gold, score}
└── amc_summary.json     # {accuracy, n_correct, n_total, mean_response_length, ...}
```

## Notes on scoring caveats

- **`math-verify`** returns `True/False/None`. `None` means the parser could not extract a candidate from the completion; we count these as incorrect (matching LUFFY / DeepSeek convention).
- **Code eval** requires a sandbox. If running in an untrusted environment, set `HF_ALLOW_CODE_EVAL=1` and consider running inside Docker (see `docker/` in the repo root, coming in v0.4).
- **SymPy fallback** on SciBench uses `sympy.simplify(gold - pred)`; a return of `0` means match. If SymPy raises, we fall back to string comparison after `.replace(' ', '')`.

## What this harness does NOT do (yet)

- **No training-time evaluation loop.** These scripts are for **checkpoint evaluation only** — you point them at a saved `AutoModelForCausalLM` directory and get numbers.
- **No head-to-head baseline runs.** To reproduce the full paper table, you'd re-train HPT / LUFFY / CHORD / SRFT baselines yourself, or use the authors' released checkpoints. We only ship GAC.

Training scripts (VeRL fork) and a Docker image with the full pipeline are planned for v0.2 and v0.4 respectively (see main [README](../README.md#-roadmap)).

## License

Apache 2.0. This harness reuses code from LUFFY, `math-verify`, and `bigcode-evaluation-harness` — see individual files for attribution.
