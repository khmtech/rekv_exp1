"""
plots.py - Paper-Quality Visualization
========================================
수집된 rejection events와 summary로부터 논문용 figure 생성.

생성되는 figure:
  fig1: KV cosine similarity 분포 (핵심 motivating figure)
  fig2: Layer별 similarity
  fig3: Draft confidence vs similarity
  fig4: Rejection position vs similarity
  fig5: Draft token rank histogram
  fig6: Accepted tokens per round 분포
"""

import os
import json
import numpy as np
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 공통 스타일 설정
COLORS = {
    "key": "#2C5F8A",
    "value": "#B8541A",
    "go_line": "#2E8B57",
    "mean_line": "#CC2222",
    "random": "#999999",
    "accent": "#6A0DAD",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def generate_all_plots(
    events: List[dict],
    summary: Dict[str, Any],
    output_dir: str,
    save_pdf: bool = True,
):
    """모든 figure를 생성."""
    os.makedirs(output_dir, exist_ok=True)

    plot_similarity_distribution(events, summary, output_dir, save_pdf)
    plot_per_layer(events, summary, output_dir, save_pdf)
    plot_confidence_vs_similarity(events, output_dir, save_pdf)
    plot_position_vs_similarity(events, output_dir, save_pdf)
    plot_rank_histogram(events, output_dir, save_pdf)
    plot_acceptance_distribution(summary, output_dir, save_pdf)

    print(f"  All plots saved to: {output_dir}")


# ============================================================
# Fig 1: KV Similarity Distribution (핵심 figure)
# ============================================================

def plot_similarity_distribution(events, summary, output_dir, save_pdf=True):
    """Rejected KV vs Target KV cosine similarity 분포."""
    k_cos = [e["mean_key_cosine"] for e in events if "mean_key_cosine" in e]
    v_cos = [e["mean_value_cosine"] for e in events if "mean_value_cosine" in e]

    if not k_cos:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vals, label, color in [
        (axes[0], k_cos, "Key Cache", COLORS["key"]),
        (axes[1], v_cos, "Value Cache", COLORS["value"]),
    ]:
        ax.hist(vals, bins=50, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        mean_val = np.mean(vals)
        ax.axvline(x=mean_val, color=COLORS["mean_line"], linestyle="--", linewidth=2,
                    label=f"Mean: {mean_val:.3f}")
        ax.axvline(x=0.7, color=COLORS["go_line"], linestyle=":", linewidth=2,
                    label="GO threshold (0.7)")

        # Random baseline if available
        bl = summary.get("random_baseline", {})
        if bl:
            bl_key = "random_key_cosine_mean" if "Key" in label else "random_value_cosine_mean"
            if bl_key in bl:
                ax.axvline(x=bl[bl_key], color=COLORS["random"], linestyle="-.",
                            linewidth=1.5, label=f"Random: {bl[bl_key]:.3f}")

        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: Rejected Draft vs Target KV")
        ax.legend(loc="upper left")
        ax.set_xlim(-0.15, 1.05)

    plt.tight_layout()
    _save(fig, output_dir, "fig1_kv_similarity_distribution", save_pdf)


# ============================================================
# Fig 2: Per-Layer Similarity
# ============================================================

def plot_per_layer(events, summary, output_dir, save_pdf=True):
    """Layer별 cosine similarity."""
    layer_data = summary.get("by_layer", {})
    if not layer_data:
        return

    layers = sorted(layer_data.keys(), key=int)
    k_means = [layer_data[l]["key_cosine_mean"] for l in layers]
    v_means = [layer_data[l]["value_cosine_mean"] for l in layers]
    k_stds = [layer_data[l].get("key_cosine_std", 0) for l in layers]
    v_stds = [layer_data[l].get("value_cosine_std", 0) for l in layers]
    x = range(len(layers))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(x, k_means, yerr=k_stds, fmt="o-", color=COLORS["key"],
                linewidth=2, markersize=5, capsize=3, label="Key similarity")
    ax.errorbar(x, v_means, yerr=v_stds, fmt="s-", color=COLORS["value"],
                linewidth=2, markersize=5, capsize=3, label="Value similarity")
    ax.axhline(y=0.7, color=COLORS["go_line"], linestyle=":", linewidth=1.5,
               alpha=0.7, label="GO threshold")

    ax.set_xlabel("Draft Model Layer Index")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("KV Similarity by Layer (Draft→Target Proportional Mapping)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x)[::max(1, len(layers)//15)])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    _save(fig, output_dir, "fig2_per_layer_similarity", save_pdf)


# ============================================================
# Fig 3: Confidence vs Similarity
# ============================================================

def plot_confidence_vs_similarity(events, output_dir, save_pdf=True):
    """Draft confidence와 KV similarity의 관계."""
    confs = [e["draft_confidence"] for e in events]
    k_cos = [e.get("mean_key_cosine", 0) for e in events]
    v_cos = [e.get("mean_value_cosine", 0) for e in events]

    if not confs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, vals, label, color in [
        (axes[0], k_cos, "Key Cosine Similarity", COLORS["key"]),
        (axes[1], v_cos, "Value Cosine Similarity", COLORS["value"]),
    ]:
        ax.scatter(confs, vals, alpha=0.25, s=12, color=color, edgecolors="none")

        # Binned trend line
        bins = np.linspace(0, 1, 11)
        bin_centers = []
        bin_means = []
        bin_stds = []
        for i in range(len(bins) - 1):
            mask = [(bins[i] <= c < bins[i+1]) for c in confs]
            bin_vals = [v for v, m in zip(vals, mask) if m]
            if len(bin_vals) >= 3:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_means.append(np.mean(bin_vals))
                bin_stds.append(np.std(bin_vals))

        if bin_centers:
            ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="D-",
                        color="black", linewidth=2, markersize=6, capsize=4,
                        label="Binned mean ± std", zorder=5)

        ax.set_xlabel("Draft Model Confidence")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Draft Confidence")
        ax.legend(loc="upper left")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 1.05)

    plt.tight_layout()
    _save(fig, output_dir, "fig3_confidence_vs_similarity", save_pdf)


# ============================================================
# Fig 4: Position in Window vs Similarity
# ============================================================

def plot_position_vs_similarity(events, output_dir, save_pdf=True):
    """Speculation window 내 위치별 KV similarity."""
    pos_data = {}
    for e in events:
        pos = e["rejection_pos_in_window"]
        if pos not in pos_data:
            pos_data[pos] = {"k": [], "v": []}
        pos_data[pos]["k"].append(e.get("mean_key_cosine", 0))
        pos_data[pos]["v"].append(e.get("mean_value_cosine", 0))

    if not pos_data:
        return

    positions = sorted(pos_data.keys())
    k_means = [np.mean(pos_data[p]["k"]) for p in positions]
    v_means = [np.mean(pos_data[p]["v"]) for p in positions]
    k_stds = [np.std(pos_data[p]["k"]) for p in positions]
    v_stds = [np.std(pos_data[p]["v"]) for p in positions]
    counts = [len(pos_data[p]["k"]) for p in positions]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.errorbar(positions, k_means, yerr=k_stds, fmt="o-", color=COLORS["key"],
                 linewidth=2, capsize=5, markersize=7, label="Key cosine")
    ax1.errorbar(positions, v_means, yerr=v_stds, fmt="s-", color=COLORS["value"],
                 linewidth=2, capsize=5, markersize=7, label="Value cosine")
    ax1.axhline(y=0.7, color=COLORS["go_line"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Rejection Position in Speculation Window")
    ax1.set_ylabel("Mean Cosine Similarity")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)

    # 두 번째 y축: count
    ax2 = ax1.twinx()
    ax2.bar(positions, counts, alpha=0.15, color="gray", width=0.4, label="Count")
    ax2.set_ylabel("Rejection Count", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title("KV Similarity by Rejection Position in Window")
    plt.tight_layout()
    _save(fig, output_dir, "fig4_position_vs_similarity", save_pdf)


# ============================================================
# Fig 5: Rejected Token Rank Histogram
# ============================================================

def plot_rank_histogram(events, output_dir, save_pdf=True):
    """Target 분포에서 rejected draft token의 rank 분포."""
    ranks = [e["draft_token_rank_in_target"] for e in events
             if e.get("draft_token_rank_in_target", -1) >= 0]

    if not ranks:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    # Top-20까지 히스토그램
    max_rank = min(max(ranks), 50)
    bins = range(0, max_rank + 2)
    ax.hist(ranks, bins=bins, color=COLORS["accent"], alpha=0.8, edgecolor="white")
    ax.set_xlabel("Rank of Draft Token in Target Distribution")
    ax.set_ylabel("Count")
    ax.set_title("How Close Was the Rejected Token? (Rank in Target Distribution)")
    ax.set_xlim(-0.5, min(max_rank, 30) + 0.5)

    # 텍스트 annotation
    top1 = sum(1 for r in ranks if r == 0) / len(ranks)
    top5 = sum(1 for r in ranks if r < 5) / len(ranks)
    top10 = sum(1 for r in ranks if r < 10) / len(ranks)
    ax.text(0.97, 0.95,
            f"Rank 0 (correct): {top1:.1%}\n"
            f"Top-5: {top5:.1%}\n"
            f"Top-10: {top10:.1%}\n"
            f"Mean rank: {np.mean(ranks):.1f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    _save(fig, output_dir, "fig5_rank_histogram", save_pdf)


# ============================================================
# Fig 6: Acceptance Length Distribution
# ============================================================

def plot_acceptance_distribution(summary, output_dir, save_pdf=True):
    """Per-position breakdown을 bar chart로."""
    pos_data = summary.get("by_position", {})
    if not pos_data:
        return

    positions = sorted(pos_data.keys(), key=int)
    counts = [pos_data[p]["count"] for p in positions]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([int(p) for p in positions], counts, color=COLORS["key"],
                  alpha=0.8, edgecolor="white")

    # 각 bar 위에 퍼센트 표시
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        if pct >= 3:  # 3% 이상만 표시
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("First Rejection Position")
    ax.set_ylabel("Count")
    ax.set_title("Where Do Rejections Happen in the Speculation Window?")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    _save(fig, output_dir, "fig6_rejection_position_distribution", save_pdf)


# ============================================================
# Utility
# ============================================================

def _save(fig, output_dir, name, save_pdf=True):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    if save_pdf:
        fig.savefig(os.path.join(output_dir, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
