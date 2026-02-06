"""
stats.py - Rejection Event Statistics & Summary
=================================================
?섏쭛??RejectionEvent 由ъ뒪?몃줈遺???듦퀎瑜?怨꾩궛?섍퀬, GO/NO-GO ?먯젙???대┝.
"""

import json
import numpy as np
from typing import List, Dict, Any
from kv_analyzer import RejectionEvent, RoundStats


def compute_summary(
    events: List[RejectionEvent],
    round_stats: List[RoundStats],
    random_baselines: List[dict] = None,
) -> Dict[str, Any]:
    """
    紐⑤뱺 rejection event? round stats瑜?醫낇빀?섏뿬 summary ?앹꽦.

    Returns:
        Summary dict with GO/NO-GO decision
    """
    if not events:
        return {"error": "No rejection events collected", "num_events": 0}

    # --- Aggregate cosine similarities ---
    k_cos = [e.kv_sim.mean_key_cosine for e in events if e.kv_sim]
    v_cos = [e.kv_sim.mean_value_cosine for e in events if e.kv_sim]
    k_l2 = [e.kv_sim.mean_key_l2 for e in events if e.kv_sim]
    v_l2 = [e.kv_sim.mean_value_l2 for e in events if e.kv_sim]

    # --- Acceptance stats ---
    total_spec = sum(s.spec_length for s in round_stats) if round_stats else 0
    total_accept = sum(s.accept_length for s in round_stats) if round_stats else 0
    acceptance_rate = total_accept / total_spec if total_spec > 0 else 0
    all_accepted_count = sum(1 for s in round_stats if s.all_accepted)
    all_accepted_rate = all_accepted_count / len(round_stats) if round_stats else 0

    def stats_block(values, name=""):
        if not values:
            return {"mean": 0, "std": 0, "median": 0, "q25": 0, "q75": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # --- Per-position breakdown ---
    pos_breakdown = {}
    for e in events:
        pos = e.rejection_pos_in_window
        if pos not in pos_breakdown:
            pos_breakdown[pos] = {"k_cos": [], "v_cos": [], "count": 0, "confidences": []}
        if e.kv_sim:
            pos_breakdown[pos]["k_cos"].append(e.kv_sim.mean_key_cosine)
            pos_breakdown[pos]["v_cos"].append(e.kv_sim.mean_value_cosine)
        pos_breakdown[pos]["count"] += 1
        pos_breakdown[pos]["confidences"].append(e.draft_confidence)

    position_stats = {}
    for pos, data in sorted(pos_breakdown.items()):
        position_stats[str(pos)] = {
            "count": data["count"],
            "mean_key_cosine": float(np.mean(data["k_cos"])) if data["k_cos"] else 0,
            "mean_value_cosine": float(np.mean(data["v_cos"])) if data["v_cos"] else 0,
            "mean_draft_confidence": float(np.mean(data["confidences"])),
        }

    # --- Per-layer breakdown ---
    if events and events[0].kv_sim and events[0].kv_sim.key_cosine_per_layer:
        num_layers = len(events[0].kv_sim.key_cosine_per_layer)
        layer_stats = {}
        for l in range(num_layers):
            lk = [e.kv_sim.key_cosine_per_layer[l] for e in events
                  if e.kv_sim and l < len(e.kv_sim.key_cosine_per_layer)]
            lv = [e.kv_sim.value_cosine_per_layer[l] for e in events
                  if e.kv_sim and l < len(e.kv_sim.value_cosine_per_layer)]
            layer_stats[str(l)] = {
                "key_cosine_mean": float(np.mean(lk)) if lk else 0,
                "value_cosine_mean": float(np.mean(lv)) if lv else 0,
                "key_cosine_std": float(np.std(lk)) if lk else 0,
                "value_cosine_std": float(np.std(lv)) if lv else 0,
            }
    else:
        layer_stats = {}

    # --- Confidence-binned breakdown ---
    conf_bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    conf_stats = {}
    for lo, hi in conf_bins:
        bin_events = [e for e in events if lo <= e.draft_confidence < hi]
        if bin_events:
            bk = [e.kv_sim.mean_key_cosine for e in bin_events if e.kv_sim]
            bv = [e.kv_sim.mean_value_cosine for e in bin_events if e.kv_sim]
            conf_stats[f"[{lo:.1f},{hi:.1f})"] = {
                "count": len(bin_events),
                "key_cosine_mean": float(np.mean(bk)) if bk else 0,
                "value_cosine_mean": float(np.mean(bv)) if bv else 0,
            }

    # --- Target rank breakdown ---
    ranks = [e.draft_token_rank_in_target for e in events if e.draft_token_rank_in_target >= 0]
    rank_stats = {
        "mean_rank": float(np.mean(ranks)) if ranks else 0,
        "median_rank": float(np.median(ranks)) if ranks else 0,
        "rank_1_fraction": sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0,
        "rank_top5_fraction": sum(1 for r in ranks if r < 5) / len(ranks) if ranks else 0,
        "rank_top10_fraction": sum(1 for r in ranks if r < 10) / len(ranks) if ranks else 0,
    }

    # --- GO/NO-GO decision ---
    mean_k = float(np.mean(k_cos)) if k_cos else 0
    mean_v = float(np.mean(v_cos)) if v_cos else 0
    avg_kv = (mean_k + mean_v) / 2

    if avg_kv > 0.7:
        decision = "GO"
        recommendation = "Strong signal. Proceed with ReKV implementation."
    elif avg_kv > 0.5:
        decision = "CAUTION"
        recommendation = (
            "Moderate signal. Consider selective recycling "
            "(only high-confidence rejections, specific layers)."
        )
    else:
        decision = "NO-GO"
        recommendation = (
            "Weak signal. Rejected KV is not sufficiently similar to target KV. "
            "Pivot to alternative approach or investigate why."
        )

    # --- Random baseline comparison ---
    baseline_info = {}
    if random_baselines:
        baseline_k = float(np.mean([b["key_cosine_mean"] for b in random_baselines]))
        baseline_v = float(np.mean([b["value_cosine_mean"] for b in random_baselines]))
        baseline_info = {
            "random_key_cosine_mean": baseline_k,
            "random_value_cosine_mean": baseline_v,
            "improvement_over_random_key": mean_k - baseline_k,
            "improvement_over_random_value": mean_v - baseline_v,
        }

    return {
        "num_rejection_events": len(events),
        "num_rounds": len(round_stats),
        "acceptance_rate": acceptance_rate,
        "all_accepted_rate": all_accepted_rate,
        "mean_accept_length": float(np.mean([s.accept_length for s in round_stats])) if round_stats else 0,
        "key_cosine_similarity": stats_block(k_cos),
        "value_cosine_similarity": stats_block(v_cos),
        "key_l2_distance": stats_block(k_l2),
        "value_l2_distance": stats_block(v_l2),
        "by_position": position_stats,
        "by_layer": layer_stats,
        "by_confidence": conf_stats,
        "draft_token_rank_in_target": rank_stats,
        "random_baseline": baseline_info,
        "go_decision": {
            "decision": decision,
            "recommendation": recommendation,
            "mean_key_cosine": mean_k,
            "mean_value_cosine": mean_v,
            "avg_kv_cosine": avg_kv,
        },
    }


def print_summary(summary: Dict[str, Any]):
    """Summary瑜??곕??먯뿉 ?쎄린 醫뗪쾶 異쒕젰."""
    print("\n" + "=" * 65)
    print("  ReKV Experiment 1: KV Similarity Analysis Results")
    print("=" * 65)

    print(f"\n  Rejection events : {summary['num_rejection_events']}")
    print(f"  Speculation rounds: {summary['num_rounds']}")
    print(f"  Acceptance rate   : {summary['acceptance_rate']:.4f}")
    print(f"  All-accepted rate : {summary['all_accepted_rate']:.4f}")
    print(f"  Mean accept length: {summary['mean_accept_length']:.2f}")

    ks = summary["key_cosine_similarity"]
    vs = summary["value_cosine_similarity"]
    print(f"\n  {'':20s} {'Key':>10s} {'Value':>10s}")
    print(f"  {'?'*42}")
    for metric in ["mean", "std", "median", "q25", "q75", "min", "max"]:
        print(f"  {metric:20s} {ks[metric]:>10.4f} {vs[metric]:>10.4f}")

    # Draft token rank
    rs = summary.get("draft_token_rank_in_target", {})
    if rs:
        print(f"\n  Draft token rank in target distribution:")
        print(f"    Mean rank: {rs.get('mean_rank', 0):.1f}")
        print(f"    Rank=1 (2nd best): {rs.get('rank_1_fraction', 0):.1%}")
        print(f"    Top-5: {rs.get('rank_top5_fraction', 0):.1%}")
        print(f"    Top-10: {rs.get('rank_top10_fraction', 0):.1%}")

    # By position
    pos = summary.get("by_position", {})
    if pos:
        print(f"\n  By rejection position in window:")
        print(f"  {'Pos':>5s} {'Count':>7s} {'Key cos':>9s} {'Val cos':>9s} {'Confidence':>11s}")
        for p, data in sorted(pos.items(), key=lambda x: int(x[0])):
            print(f"  {p:>5s} {data['count']:>7d} "
                  f"{data['mean_key_cosine']:>9.4f} {data['mean_value_cosine']:>9.4f} "
                  f"{data['mean_draft_confidence']:>11.4f}")

    # By layer (condensed)
    layer = summary.get("by_layer", {})
    if layer and len(layer) > 0:
        print(f"\n  By layer (top-5 highest, top-5 lowest key cosine):")
        sorted_layers = sorted(layer.items(), key=lambda x: x[1]["key_cosine_mean"])
        for l, data in sorted_layers[:5]:
            print(f"    Layer {l:>3s}: K={data['key_cosine_mean']:.4f} V={data['value_cosine_mean']:.4f} (LOW)")
        print(f"    ...")
        for l, data in sorted_layers[-5:]:
            print(f"    Layer {l:>3s}: K={data['key_cosine_mean']:.4f} V={data['value_cosine_mean']:.4f} (HIGH)")

    # By confidence
    conf = summary.get("by_confidence", {})
    if conf:
        print(f"\n  By draft confidence bin:")
        print(f"  {'Bin':>15s} {'Count':>7s} {'Key cos':>9s} {'Val cos':>9s}")
        for bin_name, data in conf.items():
            print(f"  {bin_name:>15s} {data['count']:>7d} "
                  f"{data['key_cosine_mean']:>9.4f} {data['value_cosine_mean']:>9.4f}")

    # Random baseline
    bl = summary.get("random_baseline", {})
    if bl:
        print(f"\n  Random baseline comparison:")
        print(f"    Random key cosine:   {bl['random_key_cosine_mean']:.4f}")
        print(f"    Random value cosine: {bl['random_value_cosine_mean']:.4f}")
        print(f"    Improvement (key):   +{bl['improvement_over_random_key']:.4f}")
        print(f"    Improvement (value): +{bl['improvement_over_random_value']:.4f}")

    # GO/NO-GO
    go = summary["go_decision"]
    print(f"\n  {'='*50}")
    marker = "GO" if go["decision"] == "GO" else "CAUTION" if go["decision"] == "CAUTION" else "NO-GO"
    print(f"  [{marker}] Decision: {go['decision']}")
    print(f"    Mean key cosine:   {go['mean_key_cosine']:.4f}")
    print(f"    Mean value cosine: {go['mean_value_cosine']:.4f}")
    print(f"    Average:           {go['avg_kv_cosine']:.4f}")
    print(f"    {go['recommendation']}")
    print(f"  {'='*50}\n")


def save_results(
    events: List[RejectionEvent],
    summary: Dict[str, Any],
    output_dir: str,
    tag: str = "",
):
    """寃곌낵瑜?JSON?쇰줈 ???"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    suffix = f"_{tag}" if tag else ""

    # Summary
    path_summary = os.path.join(output_dir, f"summary{suffix}.json")
    with open(path_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path_summary}")

    # Events (per-event detail)
    path_events = os.path.join(output_dir, f"events{suffix}.json")
    with open(path_events, "w") as f:
        json.dump([e.to_dict() for e in events], f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path_events}")

