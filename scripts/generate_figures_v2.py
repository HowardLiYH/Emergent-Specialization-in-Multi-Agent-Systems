#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for NeurIPS Paper.

Figures:
1. Cross-Domain MSE Comparison Bar Chart
2. Specialization Index by Domain
3. Diverse vs Homogeneous Performance Scatter
"""

import json
from pathlib import Path
import numpy as np

# Use Agg backend for non-GUI environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (NeurIPS style)
COLORS = {
    'diverse': '#2E86AB',  # Blue
    'homo': '#A23B72',     # Magenta
    'naive': '#F18F01',    # Orange
    'ma': '#C73E1D',       # Red
}


def load_results():
    """Load experimental results."""
    results_path = Path(__file__).parent.parent / "results" / "unified_prediction_v2" / "results.json"
    with open(results_path) as f:
        return json.load(f)


def fig1_cross_domain_mse():
    """Create bar chart comparing MSE across domains."""
    data = load_results()
    domains = list(data["domains"].keys())
    domains = [d for d in domains if "error" not in data["domains"][d]]
    
    # Normalize MSE by Naive baseline for fair comparison
    diverse_normalized = []
    homo_normalized = []
    naive_normalized = []
    
    for domain in domains:
        res = data["domains"][domain]["strategies"]
        naive_mse = res["Naive"]["mse_mean"]
        diverse_normalized.append(res["Diverse"]["mse_mean"] / naive_mse)
        homo_normalized.append(res["Homogeneous"]["mse_mean"] / naive_mse)
        naive_normalized.append(1.0)  # Baseline
    
    x = np.arange(len(domains))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, diverse_normalized, width, label='Diverse Population', 
                   color=COLORS['diverse'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, homo_normalized, width, label='Homogeneous (Best Single)', 
                   color=COLORS['homo'], edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, naive_normalized, width, label='Naive Baseline', 
                   color=COLORS['naive'], edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Naive = 1.0')
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('MSE / Naive MSE (lower is better)')
    ax.set_title('Cross-Domain Prediction Performance\n(Normalized by Naive Baseline)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(homo_normalized), 2.0) * 1.15)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Save
    output_dir = Path(__file__).parent.parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "fig1_cross_domain_mse.pdf", format='pdf')
    plt.savefig(output_dir / "fig1_cross_domain_mse.png", format='png')
    plt.close()
    
    print(f"Saved: fig1_cross_domain_mse.pdf/png")


def fig2_specialization_index():
    """Create bar chart of Specialization Index by domain."""
    data = load_results()
    domains = list(data["domains"].keys())
    domains = [d for d in domains if "error" not in data["domains"][d]]
    
    si_values = [data["domains"][d]["specialization"]["si_mean"] for d in domains]
    si_stds = [data["domains"][d]["specialization"]["si_std"] for d in domains]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(domains))
    bars = ax.bar(x, si_values, yerr=si_stds, capsize=5,
                  color=[COLORS['diverse']] * len(domains),
                  edgecolor='black', linewidth=0.5)
    
    # Color bars by SI level
    for bar, si in zip(bars, si_values):
        if si >= 0.6:
            bar.set_color('#2E7D32')  # Green (high)
        elif si >= 0.4:
            bar.set_color(COLORS['diverse'])  # Blue (medium)
        else:
            bar.set_color(COLORS['naive'])  # Orange (low)
    
    # Add value labels
    for bar, si in zip(bars, si_values):
        height = bar.get_height()
        ax.annotate(f'{si:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Specialization Index (SI)')
    ax.set_title('Agent Specialization Across Domains')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.set_ylim(0, 1.0)
    
    # Add threshold lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='SI = 0.5 (moderate)')
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='SI = 0.7 (high)')
    
    ax.legend(loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Save
    output_dir = Path(__file__).parent.parent / "results" / "figures"
    plt.savefig(output_dir / "fig2_specialization_index.pdf", format='pdf')
    plt.savefig(output_dir / "fig2_specialization_index.png", format='png')
    plt.close()
    
    print(f"Saved: fig2_specialization_index.pdf/png")


def fig3_improvement_waterfall():
    """Create waterfall chart showing improvement over Homogeneous."""
    data = load_results()
    domains = list(data["domains"].keys())
    domains = [d for d in domains if "error" not in data["domains"][d]]
    
    improvements = [data["domains"][d]["comparison"]["diverse_vs_homo_pct"] for d in domains]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(domains))
    colors = ['#2E7D32' if imp > 0 else '#C62828' for imp in improvements]
    
    bars = ax.bar(x, improvements, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        ax.annotate(f'{imp:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=1)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Improvement vs Homogeneous (%)')
    ax.set_title('Diverse Population Performance Improvement\n(Positive = Diverse beats Homogeneous)')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    green_patch = mpatches.Patch(color='#2E7D32', label='Diverse outperforms')
    red_patch = mpatches.Patch(color='#C62828', label='Diverse underperforms')
    ax.legend(handles=[green_patch, red_patch], loc='upper right')
    
    # Save
    output_dir = Path(__file__).parent.parent / "results" / "figures"
    plt.savefig(output_dir / "fig3_improvement.pdf", format='pdf')
    plt.savefig(output_dir / "fig3_improvement.png", format='png')
    plt.close()
    
    print(f"Saved: fig3_improvement.pdf/png")


def fig4_summary_heatmap():
    """Create summary heatmap of all metrics."""
    data = load_results()
    domains = list(data["domains"].keys())
    domains = [d for d in domains if "error" not in data["domains"][d]]
    
    # Metrics: Improvement %, SI, Cohen's d
    metrics = []
    for d in domains:
        res = data["domains"][d]
        metrics.append([
            res["comparison"]["diverse_vs_homo_pct"],
            res["specialization"]["si_mean"] * 100,  # Scale to percentage
            abs(res["comparison"]["cohens_d"]) * 10,  # Scale effect size
        ])
    
    metrics = np.array(metrics)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Normalize each column
    metrics_normalized = metrics.copy()
    for i in range(metrics.shape[1]):
        col_min, col_max = metrics[:, i].min(), metrics[:, i].max()
        if col_max > col_min:
            metrics_normalized[:, i] = (metrics[:, i] - col_min) / (col_max - col_min)
        else:
            metrics_normalized[:, i] = 0.5
    
    im = ax.imshow(metrics_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Improvement %', 'SI × 100', '|Cohen\'s d| × 10'])
    ax.set_yticks(np.arange(len(domains)))
    ax.set_yticklabels([d.capitalize() for d in domains])
    
    # Add values
    for i in range(len(domains)):
        for j in range(3):
            val = metrics[i, j]
            text = f'{val:.1f}' if abs(val) >= 1 else f'{val:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_title('Cross-Domain Performance Summary')
    plt.colorbar(im, ax=ax, label='Normalized Score')
    
    # Save
    output_dir = Path(__file__).parent.parent / "results" / "figures"
    plt.savefig(output_dir / "fig4_summary_heatmap.pdf", format='pdf')
    plt.savefig(output_dir / "fig4_summary_heatmap.png", format='png')
    plt.close()
    
    print(f"Saved: fig4_summary_heatmap.pdf/png")


def main():
    """Generate all figures."""
    print("="*60)
    print("Generating Publication-Quality Figures")
    print("="*60)
    
    fig1_cross_domain_mse()
    fig2_specialization_index()
    fig3_improvement_waterfall()
    fig4_summary_heatmap()
    
    print("\nAll figures generated successfully!")
    print(f"Output directory: results/figures/")


if __name__ == "__main__":
    main()

