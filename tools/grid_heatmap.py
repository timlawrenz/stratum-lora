#!/usr/bin/env python3
"""Generate heatmap visualization from grid search results CSV.

Usage:
    python tools/grid_heatmap.py --csv grid_results.csv --output heatmap.png
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser(description="Grid search heatmap")
    parser.add_argument("--csv", required=True, help="Path to grid_results.csv")
    parser.add_argument("--output", default="heatmap.png", help="Output path")
    parser.add_argument("--metric", default="avg_cosine",
                        help="Metric column to plot (avg_cosine, std_cosine)")
    parser.add_argument("--step", type=int, default=3000,
                        help="Step to plot (default: 3000 = final)")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Control baseline value for Δ coloring")
    args = parser.parse_args()

    # Load data
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        data = [{k: float(v) if k in ("dim", "step", "avg_cosine", "std_cosine", "faces_detected", "num_images") else v
                 for k, v in row.items()}
                for row in reader]

    # Filter to requested step
    step_data = [r for r in data if r["step"] == args.step]
    if not step_data:
        print(f"No data for step={args.step}")
        return

    methods = sorted(set(r["method"] for r in step_data))
    dims = sorted(set(int(r["dim"]) for r in step_data))

    # Build matrix: methods × dims
    matrix = np.zeros((len(methods), len(dims)))
    for r in step_data:
        mi = methods.index(r["method"])
        di = dims.index(int(r["dim"]))
        matrix[mi, di] = r[args.metric]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                              gridspec_kw={"width_ratios": [1, 1, 0.05]})

    # --- Raw heatmap ---
    ax1 = axes[0]
    im1 = ax1.imshow(matrix, cmap="YlOrRd", aspect="auto",
                     vmin=matrix.min() * 0.95, vmax=matrix.max() * 1.05)
    ax1.set_xticks(range(len(dims)))
    ax1.set_xticklabels(dims)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_title(f"Identity Fidelity (cosine ↑)\nStep {args.step}")
    ax1.set_xlabel("LoRA Dimension")
    for mi in range(len(methods)):
        for di in range(len(dims)):
            ax1.text(di, mi, f"{matrix[mi, di]:.4f}",
                     ha="center", va="center", fontsize=11,
                     color="white" if matrix[mi, di] > matrix.mean() else "black")

    # --- Δ vs Control ---
    ax2 = axes[1]
    if "control" in methods:
        ci = methods.index("control")
        delta = matrix - matrix[ci]  # (methods, dims)
        delta_norm = TwoSlopeNorm(vcenter=0, vmin=delta.min(), vmax=delta.max())
        cmap = plt.cm.RdBu
        im2 = ax2.imshow(delta, cmap=cmap, aspect="auto", norm=delta_norm)
        ax2.set_xticks(range(len(dims)))
        ax2.set_xticklabels(dims)
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels(methods)
        ax2.set_title(f"Δ vs Control ({args.metric})\nStep {args.step}")
        ax2.set_xlabel("LoRA Dimension")
        for mi in range(len(methods)):
            for di in range(len(dims)):
                ax2.text(di, mi, f"{delta[mi, di]:+.4f}",
                         ha="center", va="center", fontsize=10,
                         color="white" if abs(delta[mi, di]) > 0.005 else "black")
        cbar2 = plt.colorbar(im2, cax=axes[2])
        cbar2.set_label("Δ cosine")
    else:
        plt.colorbar(im1, cax=axes[2])

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {args.output}")

    # Also generate step-progression line chart
    fig2, ax = plt.subplots(figsize=(14, 5))
    all_data = data
    all_steps = sorted(set(r["step"] for r in all_data))
    colors = {"control": "tab:blue", "auraface": "tab:orange", "dinov3": "tab:green"}
    markers = {"control": "s", "auraface": "o", "dinov3": "^"}
    linestyles = {32: "--", 64: "-.", 128: "-"}

    for method in methods:
        for dim in dims:
            pts = [(r["step"], r["avg_cosine"]) for r in all_data
                   if r["method"] == method and int(r["dim"]) == dim]
            pts.sort()
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=colors.get(method, "gray"),
                        linestyle=linestyles.get(dim, "-"),
                        marker=markers.get(method, "o"),
                        label=f"{method} dim{dim}")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Avg Cosine Similarity ↑")
    ax.set_title("Identity Fidelity vs Training Steps")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    line_path = args.output.replace(".png", "_lines.png")
    plt.tight_layout()
    plt.savefig(line_path, dpi=150, bbox_inches="tight")
    print(f"Line chart saved to {line_path}")


if __name__ == "__main__":
    main()
