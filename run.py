"""
run.py - ReKV Experiment 1 Main Entry Point
=============================================

Usage:
    # Quick sanity check (~5 min)
    python run.py --preset debug

    # Preliminary results (~30 min)
    python run.py --preset quick

    # Full experiment for NeurIPS paper (~4-8 hours)
    python run.py --preset full

    # Custom configuration
    python run.py \
        --draft_model meta-llama/Llama-3.2-1B \
        --target_model meta-llama/Llama-3.1-8B-Instruct \
        --dataset mt_bench \
        --num_samples 100 \
        --spec_length 5 \
        --output_dir results/custom

    # Multi-K sweep
    python run.py \
        --dataset mt_bench --num_samples 100 \
        --spec_length 3 5 7 10 \
        --output_dir results/sweep
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="ReKV Exp1: Rejected KV Similarity Analysis"
    )

    # Presets
    parser.add_argument("--preset", type=str, default=None,
                        choices=["debug", "quick", "full", "sweep"],
                        help="Predefined configurations")

    # Models
    parser.add_argument("--draft_model", type=str,
                        default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--target_model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")

    # Data
    parser.add_argument("--dataset", type=str, default="mt_bench",
                        choices=["debug", "mt_bench", "gsm8k", "humaneval", "alpaca"])
    parser.add_argument("--num_samples", type=int, default=100)

    # SD settings
    parser.add_argument("--spec_length", type=int, nargs="+", default=[5],
                        help="Speculation length K (can pass multiple for sweep)")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Analysis options
    parser.add_argument("--collect_random_baseline", action="store_true",
                        help="Measure random vector baseline (slower)")
    parser.add_argument("--random_baseline_samples", type=int, default=50)

    # Output
    parser.add_argument("--output_dir", type=str, default="results/kv_analysis")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--save_pdf", action="store_true", default=True)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--max_prompt_len", type=int, default=512,
                        help="Max input prompt length (truncation)")

    args = parser.parse_args()

    # Apply presets
    if args.preset == "debug":
        args.dataset = "debug"
        args.num_samples = 5
        args.spec_length = [5]
        args.max_new_tokens = 64
        args.output_dir = "results/debug"
    elif args.preset == "quick":
        args.dataset = "mt_bench"
        args.num_samples = 20
        args.spec_length = [5]
        args.max_new_tokens = 128
        args.output_dir = "results/quick"
    elif args.preset == "full":
        args.dataset = "mt_bench"
        args.num_samples = 200
        args.spec_length = [5]
        args.max_new_tokens = 128
        args.output_dir = "results/full"
        args.collect_random_baseline = True
    elif args.preset == "sweep":
        args.dataset = "mt_bench"
        args.num_samples = 100
        args.spec_length = [3, 5, 7, 10]
        args.max_new_tokens = 128
        args.output_dir = "results/sweep"

    return args


def load_models(args):
    """Draft 諛?Target 紐⑤뜽 濡쒕뱶."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"\n[1/3] Loading tokenizer: {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/3] Loading draft model: {args.draft_model}")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    draft.eval()

    print(f"[3/3] Loading target model: {args.target_model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation="sdpa",
    )
    target.eval()

    # 紐⑤뜽 ?뺣낫 異쒕젰
    d_params = sum(p.numel() for p in draft.parameters()) / 1e9
    t_params = sum(p.numel() for p in target.parameters()) / 1e9
    d_layers = draft.config.num_hidden_layers
    t_layers = target.config.num_hidden_layers
    print(f"\n  Draft:  {d_params:.1f}B params, {d_layers} layers")
    print(f"  Target: {t_params:.1f}B params, {t_layers} layers")
    print(f"  Layer ratio: 1:{t_layers/d_layers:.1f}")

    if hasattr(draft.config, "num_key_value_heads"):
        print(f"  Draft KV heads: {draft.config.num_key_value_heads}, "
              f"Target KV heads: {target.config.num_key_value_heads}")

    return draft, target, tokenizer


def tokenize_prompt(prompt, tokenizer, max_len, device):
    """?꾨＼?꾪듃瑜??좏겙?? Chat template ?곸슜 ?쒕룄."""
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt

    ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_len)
    return ids.to(device)


def run_single_k(args, K, draft, target, tokenizer, prompts):
    """?섎굹??spec_length K??????꾩껜 ?ㅽ뿕 ?섑뻾."""
    from spec_decode import InstrumentedSpecDecoder
    from stats import compute_summary, print_summary, save_results
    from plots import generate_all_plots

    print(f"\n{'='*60}")
    print(f"  K = {K}")
    print(f"{'='*60}")

    decoder = InstrumentedSpecDecoder(
        draft_model=draft,
        target_model=target,
        tokenizer=tokenizer,
        device=args.device,
        collect_random_baseline=args.collect_random_baseline,
        random_baseline_samples=args.random_baseline_samples,
    )

    errors = 0
    t_start = time.time()

    for idx, prompt in enumerate(tqdm(prompts, desc=f"K={K}", ncols=80)):
        input_ids = tokenize_prompt(prompt, tokenizer, args.max_prompt_len, args.device)

        try:
            decoder.run(
                input_ids=input_ids,
                sample_id=idx,
                spec_length=K,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  ??Sample {idx} failed: {e}")
            elif errors == 4:
                print(f"\n  ??Suppressing further error messages...")
            continue

        # Periodic progress
        if (idx + 1) % max(1, len(prompts) // 5) == 0 and decoder.rejection_events:
            n_events = len(decoder.rejection_events)
            k_cos = [e.kv_sim.mean_key_cosine for e in decoder.rejection_events[-50:] if e.kv_sim]
            v_cos = [e.kv_sim.mean_value_cosine for e in decoder.rejection_events[-50:] if e.kv_sim]
            import numpy as np
            print(f"\n  [{idx+1}/{len(prompts)}] events={n_events}, "
                  f"recent K-cos={np.mean(k_cos):.3f}, V-cos={np.mean(v_cos):.3f}")

    elapsed = time.time() - t_start
    print(f"\n  Completed in {elapsed:.1f}s ({errors} errors)")
    print(f"  Total rejections: {len(decoder.rejection_events)}")
    print(f"  Total rounds: {len(decoder.round_stats)}")

    if not decoder.rejection_events:
        print("  ??No rejection events collected. Check model pair.")
        return None

    # Compute summary
    summary = compute_summary(
        decoder.rejection_events,
        decoder.round_stats,
        decoder.random_baselines if args.collect_random_baseline else None,
    )
    print_summary(summary)

    # Save
    tag = f"K{K}"
    out_dir = os.path.join(args.output_dir, tag)
    save_results(decoder.rejection_events, summary, out_dir, tag=tag)

    # Plots
    if not args.skip_plots:
        print("\n  Generating plots...")
        events_dicts = [e.to_dict() for e in decoder.rejection_events]
        plot_dir = os.path.join(out_dir, "plots")
        generate_all_plots(events_dicts, summary, plot_dir, save_pdf=args.save_pdf)

    return summary


def main():
    args = parse_args()

    print("=" * 60)
    print("  ReKV Experiment 1: Rejected KV Similarity Analysis")
    print("=" * 60)
    print(f"  Draft:      {args.draft_model}")
    print(f"  Target:     {args.target_model}")
    print(f"  Dataset:    {args.dataset} ({args.num_samples} samples)")
    print(f"  Spec K:     {args.spec_length}")
    print(f"  Max tokens: {args.max_new_tokens}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Device:     {args.device}, dtype: {args.dtype}")
    print("=" * 60)

    # GPU check
    if args.device == "cuda" and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem:.0f} GB)")
    elif args.device == "cuda":
        print("  ??CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load
    draft, target, tokenizer = load_models(args)

    from datasets_loader import load_prompts
    prompts = load_prompts(args.dataset, args.num_samples, tokenizer)

    # Run for each K
    all_summaries = {}
    for K in args.spec_length:
        summary = run_single_k(args, K, draft, target, tokenizer, prompts)
        if summary:
            all_summaries[f"K={K}"] = summary

    # Combined report
    if all_summaries:
        combined_path = os.path.join(args.output_dir, "combined_results.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"\n  Combined results: {combined_path}")

        # Final GO/NO-GO
        print("\n" + "=" * 60)
        print("  FINAL GO/NO-GO ACROSS ALL K")
        print("=" * 60)
        for k_str, s in all_summaries.items():
            go = s["go_decision"]
            marker = "GO" if go["decision"] == "GO" else "CAUTION" if go["decision"] == "CAUTION" else "NO-GO"
            print(f"  [{marker}] {k_str}: K-cos={go['mean_key_cosine']:.3f}, "
                  f"V-cos={go['mean_value_cosine']:.3f} => {go['decision']}")
        print("=" * 60)


if __name__ == "__main__":
    main()


