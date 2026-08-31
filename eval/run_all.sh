#!/usr/bin/env bash
# GAC eval harness — one-command reproduction across 8 benchmarks.
#
# Usage:
#   bash run_all.sh --model_path <ckpt> --output_dir ./results [--tp_size 4] [--seed 0]
#
# Runs, in order:
#   1. math_bench.py       (AMC / AIME24 / AIME25)
#   2. knowledge_bench.py  (MMLU-Pro / GPQA / SciBench)
#   3. code_bench.py       (MBPP / HumanEval)   [needs bigcode-evaluation-harness]
#   4. bbh_logic.py        (BBH logical subsets)
#
# On 4×A100/A800, wall-clock is ~90 min for a 7B model.
set -euo pipefail

MODEL_PATH=""
OUTPUT_DIR=""
TP_SIZE=4
SEED=0
N_SAMPLES=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)  MODEL_PATH="$2"; shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --tp_size)     TP_SIZE="$2";    shift 2 ;;
        --seed)        SEED="$2";       shift 2 ;;
        --n_samples)   N_SAMPLES="$2";  shift 2 ;;
        *)  echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --model_path <ckpt> --output_dir <dir> [--tp_size 4] [--seed 0] [--n_samples 1]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==============================================================="
echo "GAC eval harness"
echo "  model_path : $MODEL_PATH"
echo "  output_dir : $OUTPUT_DIR"
echo "  tp_size    : $TP_SIZE"
echo "  seed       : $SEED"
echo "  n_samples  : $N_SAMPLES"
echo "==============================================================="

echo ""
echo "[1/4] Math (AMC / AIME24 / AIME25)"
python math_bench.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED"

echo ""
echo "[2/4] Knowledge (MMLU-Pro / GPQA / SciBench)"
python knowledge_bench.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED"

echo ""
echo "[3/4] Code (MBPP / HumanEval)"
python code_bench.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED" || echo "[warn] code_bench failed — completions saved, install bigcode-evaluation-harness for scoring"

echo ""
echo "[4/4] BBH-Logic (Logical Ded / Object Count / Tracking)"
python bbh_logic.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --tp_size "$TP_SIZE" \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED"

echo ""
echo "==============================================================="
echo "Done. Per-benchmark summaries live under $OUTPUT_DIR/{math,knowledge,code,bbh_logic}_summary.json"
echo "To aggregate across seeds:"
echo "  python aggregate.py $OUTPUT_DIR/../seed_{0,1,2}"
echo "==============================================================="
