#!/bin/bash
# ============================================================
# ReKV Exp1: Quick-start experiment runners
# Usage: chmod +x scripts.sh && ./scripts.sh [command]
# ============================================================
set -e

DRAFT="meta-llama/Llama-3.2-1B"
TARGET="meta-llama/Llama-3.1-8B-Instruct"

case "${1:-help}" in

  debug)
    echo "=== Debug (5 samples, ~5 min) ==="
    python run.py --preset debug
    ;;

  quick)
    echo "=== Quick (20 samples, ~30 min) ==="
    python run.py --preset quick
    ;;

  full)
    echo "=== Full (200 samples + random baseline, ~4-8 hours) ==="
    python run.py --preset full
    ;;

  sweep)
    echo "=== K Sweep (K=3,5,7,10, 100 samples, ~6 hours) ==="
    python run.py --preset sweep
    ;;

  all-datasets)
    echo "=== All Datasets (MT-Bench + GSM8K + HumanEval) ==="
    for ds in mt_bench gsm8k humaneval; do
      echo "--- $ds ---"
      python run.py \
        --dataset $ds --num_samples 100 --spec_length 5 \
        --output_dir results/dataset_comparison/$ds \
        --collect_random_baseline
    done
    ;;

  custom-model)
    # Example: ./scripts.sh custom-model Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-7B-Instruct
    echo "=== Custom Model Pair ==="
    python run.py \
      --draft_model "${2:?Provide draft model}" \
      --target_model "${3:?Provide target model}" \
      --dataset mt_bench --num_samples 50 --spec_length 5 \
      --output_dir results/custom_models
    ;;

  visualize)
    echo "=== Regenerate plots from saved results ==="
    python -c "
import json, sys
from plots import generate_all_plots
results_dir = '${2:-results/quick/K5}'
events = json.load(open(f'{results_dir}/events_K5.json'))
summary = json.load(open(f'{results_dir}/summary_K5.json'))
generate_all_plots(events, summary, f'{results_dir}/plots_regen')
print('Done')
"
    ;;

  help|*)
    echo "ReKV Experiment 1: Rejected KV Similarity Analysis"
    echo ""
    echo "Usage: ./scripts.sh [command]"
    echo ""
    echo "Commands:"
    echo "  debug          5 samples, ~5 min (sanity check)"
    echo "  quick          20 samples, ~30 min (preliminary results)"
    echo "  full           200 samples + baseline, ~4-8 hours (paper-ready)"
    echo "  sweep          K=3,5,7,10 sweep, ~6 hours"
    echo "  all-datasets   MT-Bench + GSM8K + HumanEval comparison"
    echo "  custom-model   Custom model pair (pass draft & target as args)"
    echo "  visualize      Regenerate plots from saved JSON results"
    echo "  help           Show this message"
    ;;

esac
