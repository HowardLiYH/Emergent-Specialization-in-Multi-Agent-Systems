#!/usr/bin/env python3
"""
Generate Hypothesis Test Table with Bonferroni Correction.

Creates a publication-ready table comparing Diverse vs Homogeneous vs Naive
across all 4 domains with proper statistical rigor.
"""

import json
from pathlib import Path
import numpy as np


def generate_latex_table():
    """Generate LaTeX table from results with baseline comparisons."""

    results_path = Path(__file__).parent.parent / "results" / "unified_prediction_v2" / "results.json"

    with open(results_path) as f:
        data = json.load(f)

    domains = data["domains"]
    n_domains = len([d for d in domains.values() if "error" not in d])
    alpha = 0.05
    bonferroni_alpha = alpha / n_domains

    print("="*90)
    print("TABLE 5: Multi-Domain Specialization with Baseline Comparisons")
    print("="*90)
    print(f"Bonferroni-corrected α = {bonferroni_alpha:.4f}")
    print()

    # LaTeX table
    latex = r"""
\begin{table}[h!]
\centering
\caption{Cross-Domain Prediction Performance. MSE (lower is better) with 95\% bootstrap CI.
Statistical significance tested with paired t-test, Bonferroni-corrected $\alpha = 0.0125$.
$\Delta$\% shows improvement of Diverse over Homogeneous.}
\label{tab:cross_domain}
\begin{tabular}{lccccccc}
\toprule
\textbf{Domain} & \textbf{Diverse MSE} & \textbf{95\% CI} & \textbf{Homo MSE} & \textbf{Naive MSE} & \textbf{$\Delta$\% vs Homo} & \textbf{SI} & \textbf{Sig.} \\
\midrule
"""

    print(f"{'Domain':<10} {'Diverse MSE':<16} {'95% CI':<24} {'Homo MSE':<16} {'Naive MSE':<16} {'Δ% vs Homo':<12} {'SI':<8} {'Sig?'}")
    print("-" * 120)

    for domain, res in domains.items():
        if "error" in res:
            continue

        s = res["strategies"]
        c = res["comparison"]
        si = res["specialization"]["si_mean"]

        diverse_mse = s["Diverse"]["mse_mean"]
        diverse_ci = f"[{s['Diverse']['ci_lower']:.4f}, {s['Diverse']['ci_upper']:.4f}]"
        homo_mse = s["Homogeneous"]["mse_mean"]
        naive_mse = s["Naive"]["mse_mean"]
        delta = c["diverse_vs_homo_pct"]
        sig = "✓" if c["significant_bonferroni"] else "✗"

        # Format numbers based on scale
        if diverse_mse > 1000:
            diverse_str = f"{diverse_mse:,.0f}"
            homo_str = f"{homo_mse:,.0f}"
            naive_str = f"{naive_mse:,.0f}"
            ci_str = f"[{s['Diverse']['ci_lower']:,.0f}, {s['Diverse']['ci_upper']:,.0f}]"
        elif diverse_mse > 1:
            diverse_str = f"{diverse_mse:.2f}"
            homo_str = f"{homo_mse:.2f}"
            naive_str = f"{naive_mse:.2f}"
            ci_str = f"[{s['Diverse']['ci_lower']:.2f}, {s['Diverse']['ci_upper']:.2f}]"
        else:
            diverse_str = f"{diverse_mse:.6f}"
            homo_str = f"{homo_mse:.6f}"
            naive_str = f"{naive_mse:.6f}"
            ci_str = f"[{s['Diverse']['ci_lower']:.6f}, {s['Diverse']['ci_upper']:.6f}]"

        print(f"{domain.capitalize():<10} {diverse_str:<16} {ci_str:<24} {homo_str:<16} {naive_str:<16} {delta:>+10.1f}% {si:>6.3f}   {sig}")

        # LaTeX row
        sig_latex = r"\checkmark" if c["significant_bonferroni"] else r"$\times$"
        latex += f"{domain.capitalize()} & {diverse_str} & {ci_str} & {homo_str} & {naive_str} & {delta:+.1f}\\% & {si:.3f} & {sig_latex} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    print()
    print("="*80)
    print("LaTeX Table:")
    print("="*80)
    print(latex)

    # Save to file
    output_path = Path(__file__).parent.parent / "paper" / "tables" / "cross_domain_results.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(latex)

    print(f"\nSaved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    improvements = []
    sis = []
    for domain, res in domains.items():
        if "error" not in res:
            improvements.append(res["comparison"]["diverse_vs_homo_pct"])
            sis.append(res["specialization"]["si_mean"])

    print(f"Average improvement over Homogeneous: {np.mean(improvements):.1f}%")
    print(f"Average Specialization Index: {np.mean(sis):.3f}")
    print(f"All significant at Bonferroni α=0.0125: {all(domains[d]['comparison']['significant_bonferroni'] for d in domains if 'error' not in domains[d])}")

    # Domains where Diverse beats Homo
    positive_domains = [d for d in domains if "error" not in domains[d] and domains[d]["comparison"]["diverse_vs_homo_pct"] > 0]
    print(f"Domains where Diverse > Homo: {len(positive_domains)}/{len(domains)} ({', '.join(d.capitalize() for d in positive_domains)})")

    return latex


if __name__ == "__main__":
    generate_latex_table()
