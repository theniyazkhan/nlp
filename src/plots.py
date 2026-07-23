"""
Generate IEEE-style visualization figures for copy baseline and entities-only baseline evaluations.
Excludes artifact outliers (entity_share_chrf > 1.0) and focuses metrics on chrF++.
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.titlesize'] = 9


def plot_fig1_copy_by_script(df, output_path="figures/fig1_copy_baseline_by_script.png"):
    df_sorted = df.sort_values(by=["script", "copy_chrf"], ascending=[True, True]).copy()
    unique_scripts = sorted(df_sorted["script"].unique())
    colors = plt.get_cmap("tab20", len(unique_scripts))
    script_color_map = {script: colors(i) for i, script in enumerate(unique_scripts)}
    bar_colors = [script_color_map[s] for s in df_sorted["script"]]

    fig, ax = plt.subplots(figsize=(3.5, 4.8), dpi=300)
    ax.barh(df_sorted["language"], df_sorted["copy_chrf"], color=bar_colors, edgecolor="none", height=0.7)

    legend_elements = [Patch(facecolor=script_color_map[s], label=s) for s in unique_scripts]
    ax.legend(handles=legend_elements, title="Script", loc="lower right", framealpha=0.8, fontsize=6.5, title_fontsize=7)

    ax.set_xlabel("Copy chrF++ Score")
    ax.set_ylabel("Language")
    ax.set_title("FLORES+ Copy Baseline chrF++ by Script", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated Figure 1: '{output_path}'")


def plot_fig2_entities_by_tier(df, output_path="figures/fig2_entities_only_by_tier.png"):
    tier_order = ["high", "mid", "low"]
    tier_data = [df[df["resource_tier"] == tier]["entities_only_chrf"].dropna().values for tier in tier_order]

    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
    box = ax.boxplot(tier_data, tick_labels=["High", "Mid", "Low"], patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.2))

    tier_colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(box['boxes'], tier_colors):
        patch.set_facecolor(color)

    for i, tier in enumerate(tier_order, start=1):
        vals = df[df["resource_tier"] == tier]["entities_only_chrf"]
        x = [i] * len(vals)
        ax.scatter(x, vals, color="black", alpha=0.6, s=12, zorder=3)

    ax.set_xlabel("Resource Tier")
    ax.set_ylabel("Entities-Only chrF++ Score")
    ax.set_title("Entities-Only chrF++ by Resource Tier", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated Figure 2: '{output_path}'")


def plot_fig3_entity_share(df, output_path="figures/fig3_entity_share.png"):
    excluded = df[df["entity_share_chrf"] > 1.0]
    excluded_langs = excluded["language"].tolist() if not excluded.empty else []

    df_filtered = df[df["entity_share_chrf"] <= 1.0].sort_values(by="entity_share_chrf", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(3.5, 5.0), dpi=300)
    colors = ["#2b5c8f" if is_lat else "#d95f02" for is_lat in df_filtered["is_latin"]]

    ax.barh(df_filtered["language"], df_filtered["entity_share_chrf"], color=colors, height=0.7)

    legend_elements = [
        Patch(facecolor="#2b5c8f", label="Latin Script"),
        Patch(facecolor="#d95f02", label="Non-Latin Script")
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8, fontsize=6.5)

    ax.set_xlabel("Entity Share of Copy chrF++ (Entities / Copy)")
    ax.set_ylabel("Language")
    
    title = "Entity Share of Baseline Copy Score"
    if excluded_langs:
        title += f"\n(Excludes brevity artifact: {', '.join(excluded_langs)})"
    ax.set_title(title, fontweight="bold", fontsize=7.5)

    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated Figure 3: '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Generate Baseline Evaluation Figures")
    parser.add_argument("--comparison-csv", type=str, default="results/baseline_comparison.csv", help="Input comparison CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.comparison_csv):
        raise FileNotFoundError(f"Comparison CSV not found: {args.comparison_csv}")

    df = pd.read_csv(args.comparison_csv)

    os.makedirs("figures", exist_ok=True)
    plot_fig1_copy_by_script(df)
    plot_fig2_entities_by_tier(df)
    plot_fig3_entity_share(df)
    print("All figures successfully generated in figures/!")


if __name__ == "__main__":
    main()
