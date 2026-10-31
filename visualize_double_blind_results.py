#!/usr/bin/env python3
"""
Module: visualize_double_blind_results.py
Description:
    Publication-grade visualization for the Three-Condition Double-Blind Attractor Benchmark.

    Reads results/double_blind_results.json (produced by eval_double_blind_attractor.py)
    and generates:
      Figure 1: Grouped barplot of Conditions A / B / C + A→B degradation drop (dual y-axis).
      Figure 2: Closed-Set vs Open-Set robustness comparison on Condition B.
      Figure 3: Model × Condition accuracy heatmap.

    All figures are written as vector PDF (default) or PNG at 300 DPI.

Usage:
    python visualize_double_blind_results.py \
        --results results/double_blind_results.json \
        --output_dir figures/ \
        --format pdf
"""

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# 1. GLOBAL STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})


# Colorblind-safe palette (Okabe-Ito + accent)
PALETTE = {
    "A": "#2b5c8f",       # Deep slate blue — Forward Binding
    "B": "#d95f02",       # Vermillion — Adversarial Inversion
    "C": "#7570b3",       # Soft purple — Neutral Control
    "Drop": "#e7298a",    # Magenta — A→B drop
    "CAFT": "#1b9e77",    # Forest teal — Ours highlight
    "Baseline": "#888888",# Neutral gray — baselines
}


# =============================================================================
# 2. UTILITIES
# =============================================================================

def load_results(json_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Double-blind result file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    return data


def save_figure(fig, output_dir: str, name: str, fmt: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(path, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure Exported] -> {path}")


def is_caft(model_name: str) -> bool:
    return "caft" in model_name.lower()


def extract_condition_accs(item: Dict[str, Any]) -> Dict[str, float]:
    pc = item.get("per_condition", {})
    return {
        "A": pc.get("A_Forward_Binding", {}).get("acc", 0.0),
        "B": pc.get("B_Adversarial_Inversion", {}).get("acc", 0.0),
        "C": pc.get("C_Neutral_Control", {}).get("acc", 0.0),
    }


# =============================================================================
# 3. FIGURE 1: GROUPED BARPLOT (A / B / C) + A→B DROP
# =============================================================================

def plot_main_benchmark_figure(
    data: List[Dict[str, Any]],
    output_dir: str,
    fmt: str = "pdf",
) -> None:
    """
    Figure 1: Grouped barplot of Conditions A / B / C, with dual-y-axis for A→B drop.
    CAFT is highlighted with teal color + bold black edge.
    """
    # Filter closed-set results (fallback to all if none)
    closed = [d for d in data if d.get("candidate_type", "closed") == "closed"]
    if not closed:
        closed = data

    # Sort: baselines first, CAFT last (so CAFT stands out on the right)
    def sort_key(item):
        name = item.get("model_name", "")
        return (1 if is_caft(name) else 0, name)

    closed = sorted(closed, key=sort_key)

    model_names = [d.get("model_name", "Unknown") for d in closed]
    accs = [extract_condition_accs(d) for d in closed]
    drops = []
    for d, a in zip(closed, accs):
        drop = d.get("a_to_b_drop", a["A"] - a["B"])
        drops.append(drop)

    x = np.arange(len(model_names))
    width = 0.22

    fig, ax1 = plt.subplots(figsize=(10, 5.2), dpi=300)

    # Bars
    for idx, name in enumerate(model_names):
        highlight = is_caft(name)
        edge = "black" if highlight else "none"
        lw = 1.6 if highlight else 0.5

        for j, cond in enumerate(["A", "B", "C"]):
            alpha = {"A": 0.55, "B": 0.95, "C": 0.75}[cond]
            color = PALETTE["CAFT"] if highlight else PALETTE[cond]
            ax1.bar(
                x[idx] + (j - 1) * width,
                accs[idx][cond],
                width,
                color=color,
                alpha=alpha,
                edgecolor=edge,
                linewidth=lw,
                zorder=3,
            )

    # Annotate B (adversarial) values
    for idx, a in enumerate(accs):
        ax1.annotate(
            f"{a['B']:.1f}",
            xy=(x[idx], a["B"]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color="#222222",
        )

    ax1.set_ylabel("Resolution Accuracy (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    # Dual axis: A→B drop
    ax2 = ax1.twinx()
    ax2.plot(
        x, drops,
        color=PALETTE["Drop"], marker="o", markersize=7,
        linewidth=2.0, label="A$\\to$B Drop (pp)", zorder=5,
    )
    ax2.set_ylabel("A→B Drop (pp)", color=PALETTE["Drop"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["Drop"])
    ax2.set_ylim(0, max(80, max(drops) * 1.15 if drops else 80))

    # Legend
    legend_handles = [
        mpatches.Patch(color=PALETTE["A"], alpha=0.55, label="Cond A (Forward)"),
        mpatches.Patch(color=PALETTE["B"], alpha=0.95, label="Cond B (Adversarial)"),
        mpatches.Patch(color=PALETTE["C"], alpha=0.75, label="Cond C (Neutral)"),
        plt.Line2D([0], [0], color=PALETTE["Drop"], marker="o",
                   linewidth=2.0, label="A→B Drop"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", framealpha=0.95)

    plt.title("Double-Blind Attractor Benchmark: A/B/C Accuracy and A→B Degradation")
    fig.tight_layout()

    save_figure(fig, output_dir, "Fig1_main_benchmark", fmt)


# =============================================================================
# 4. FIGURE 2: CLOSED-SET VS OPEN-SET (CONDITION B)
# =============================================================================

def plot_closed_vs_open_figure(
    data: List[Dict[str, Any]],
    output_dir: str,
    fmt: str = "pdf",
) -> None:
    """
    Figure 2: Condition B accuracy comparison between closed-set (2-way) and open-set (Top-10).
    """
    # Build per-base-model dict
    base_models: Dict[str, Dict[str, float]] = {}
    for d in data:
        name = d.get("model_name", "Unknown")
        base = name.replace(" (closed)", "").replace(" (open)", "").strip()
        cand = d.get("candidate_type", "closed")
        pc = d.get("per_condition", {})
        acc_b = pc.get("B_Adversarial_Inversion", {}).get("acc", 0.0)

        if base not in base_models:
            base_models[base] = {}
        base_models[base][cand] = acc_b

    models = sorted(base_models.keys(),
                    key=lambda m: (1 if is_caft(m) else 0, m))
    closed_accs = [base_models[m].get("closed", 0.0) for m in models]
    open_accs = [base_models[m].get("open", 0.0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=300)
    bars_c = ax.bar(x - width / 2, closed_accs, width,
                    label="Closed-Set (2-way)",
                    color=PALETTE["A"], edgecolor="black", linewidth=0.6, zorder=3)
    bars_o = ax.bar(x + width / 2, open_accs, width,
                    label="Open-Set (Top-10)",
                    color=PALETTE["B"], edgecolor="black", linewidth=0.6, zorder=3)

    # Highlight CAFT
    for idx, m in enumerate(models):
        if is_caft(m):
            bars_c[idx].set_edgecolor(PALETTE["CAFT"])
            bars_c[idx].set_linewidth(2.0)
            bars_o[idx].set_edgecolor(PALETTE["CAFT"])
            bars_o[idx].set_linewidth(2.0)

    # Annotate values
    for rect in list(bars_c) + list(bars_o):
        h = rect.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("Condition B Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.legend(loc="upper right", framealpha=0.95)

    plt.title("Robustness to Adversarial Attractors: Closed-Set vs Open-Set")
    fig.tight_layout()

    save_figure(fig, output_dir, "Fig2_closed_vs_open", fmt)


# =============================================================================
# 5. FIGURE 3: HEATMAP OF MODEL × CONDITION
# =============================================================================

def plot_model_condition_heatmap(
    data: List[Dict[str, Any]],
    output_dir: str,
    fmt: str = "pdf",
) -> None:
    """
    Figure 3: Heatmap of model × condition accuracy (closed-set).
    """
    closed = [d for d in data if d.get("candidate_type", "closed") == "closed"]
    if not closed:
        closed = data

    closed = sorted(closed,
                    key=lambda d: (1 if is_caft(d.get("model_name", "")) else 0,
                                   d.get("model_name", "")))

    model_names = [d.get("model_name", "Unknown") for d in closed]
    conditions = ["A_Forward_Binding", "B_Adversarial_Inversion", "C_Neutral_Control"]
    condition_labels = ["A\n(Forward)", "B\n(Adversarial)", "C\n(Neutral)"]

    matrix = np.zeros((len(closed), len(conditions)))
    for i, d in enumerate(closed):
        pc = d.get("per_condition", {})
        for j, c in enumerate(conditions):
            matrix[i, j] = pc.get(c, {}).get("acc", 0.0)

    fig, ax = plt.subplots(
        figsize=(5.0, max(3.0, 0.45 * len(model_names))), dpi=300
    )
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(condition_labels)))
    ax.set_xticklabels(condition_labels, fontsize=9)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=8)

    for i in range(len(model_names)):
        for j in range(len(condition_labels)):
            v = matrix[i, j]
            color = "black" if 20 < v < 80 else "white"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)")

    ax.set_title("Model × Condition Accuracy Heatmap")
    fig.tight_layout()

    save_figure(fig, output_dir, "Fig3_heatmap", fmt)


# =============================================================================
# 6. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Double-Blind Attractor Benchmark Results."
    )
    parser.add_argument("--results", type=str,
                        default="results/double_blind_results.json",
                        help="Path to the double_blind_results.json file.")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save figures.")
    parser.add_argument("--format", type=str, choices=["pdf", "png", "svg"],
                        default="pdf", help="Output figure format.")
    parser.add_argument("--figures", type=str, default="1,2,3",
                        help="Comma-separated figure indices to generate (e.g. '1,2').")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_results(args.results)
    print(f"[INFO] Loaded {len(data)} entries from {args.results}")

    figs_to_gen = set(t.strip() for t in args.figures.split(","))

    if "1" in figs_to_gen:
        plot_main_benchmark_figure(data, args.output_dir, args.format)

    if "2" in figs_to_gen:
        plot_closed_vs_open_figure(data, args.output_dir, args.format)

    if "3" in figs_to_gen:
        plot_model_condition_heatmap(data, args.output_dir, args.format)

    print(f"[DONE] All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()