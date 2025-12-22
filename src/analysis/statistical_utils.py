"""
Statistical Utilities for Rigorous Experiment Analysis.

Provides:
- Bootstrap confidence intervals
- Bonferroni correction for multiple comparisons
- Effect size calculations (Cohen's d)
- Standardized result formatting
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f} (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"

    def to_latex(self, precision: int = 3) -> str:
        """Format for LaTeX tables."""
        return f"${self.mean:.{precision}f} \\pm {self.std:.{precision}f}$"


@dataclass
class ComparisonResult:
    """Container for comparison test results."""
    group1_mean: float
    group2_mean: float
    difference: float
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool
    significance_level: str  # "", "*", "**", "***"

    def __str__(self) -> str:
        return (
            f"Diff: {self.difference:.4f}, "
            f"t={self.t_statistic:.2f}, p={self.p_value:.4e}, "
            f"d={self.cohens_d:.2f}{self.significance_level}"
        )


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Sample data
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        statistic: "mean" or "median"
        seed: Random seed

    Returns:
        Tuple of (lower, upper) confidence bounds
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    # Bootstrap sampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        if statistic == "mean":
            bootstrap_stats[i] = np.mean(sample)
        elif statistic == "median":
            bootstrap_stats[i] = np.median(sample)

    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return lower, upper


def compute_stats(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
) -> StatisticalResult:
    """
    Compute comprehensive statistics for a sample.

    Args:
        data: Sample data
        confidence: Confidence level

    Returns:
        StatisticalResult with mean, std, and CI
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample std

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(data, confidence=confidence)

    return StatisticalResult(
        mean=float(mean),
        std=float(std),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n=n,
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Cohen's d = (M1 - M2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compare_groups(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    alpha: float = 0.05,
    paired: bool = True,
) -> ComparisonResult:
    """
    Compare two groups with statistical tests.

    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level
        paired: Whether to use paired t-test

    Returns:
        ComparisonResult with test statistics
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # T-test
    if paired:
        t_stat, p_value = stats.ttest_rel(group1, group2)
    else:
        t_stat, p_value = stats.ttest_ind(group1, group2)

    # Effect size
    d = cohens_d(group1, group2)

    # Significance
    significant = p_value < alpha
    if p_value < 0.001:
        sig_level = "***"
    elif p_value < 0.01:
        sig_level = "**"
    elif p_value < 0.05:
        sig_level = "*"
    else:
        sig_level = ""

    return ComparisonResult(
        group1_mean=float(np.mean(group1)),
        group2_mean=float(np.mean(group2)),
        difference=float(np.mean(group1) - np.mean(group2)),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        cohens_d=float(d),
        significant=significant,
        significance_level=sig_level,
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        List of (adjusted_p, significant) tuples
    """
    n = len(p_values)
    adjusted_alpha = alpha / n

    return [
        (p * n, p < adjusted_alpha)  # Adjusted p-value and significance
        for p in p_values
    ]


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool, int]]:
    """
    Apply Holm-Bonferroni stepwise correction (more powerful than Bonferroni).

    Returns:
        List of (adjusted_p, significant, rank) tuples
    """
    n = len(p_values)

    # Sort p-values with original indices
    sorted_indices = np.argsort(p_values)
    results = [None] * n

    for rank, idx in enumerate(sorted_indices):
        p = p_values[idx]
        adjusted_alpha = alpha / (n - rank)
        adjusted_p = min(p * (n - rank), 1.0)

        # Once we fail to reject, all subsequent are also not rejected
        if rank > 0:
            prev_idx = sorted_indices[rank - 1]
            if not results[prev_idx][1]:
                results[idx] = (adjusted_p, False, rank + 1)
                continue

        results[idx] = (adjusted_p, p < adjusted_alpha, rank + 1)

    return results


def multiple_comparisons(
    groups: Dict[str, Union[List[float], np.ndarray]],
    baseline_key: str,
    alpha: float = 0.05,
    correction: str = "holm",
) -> Dict[str, ComparisonResult]:
    """
    Compare multiple groups against a baseline with correction.

    Args:
        groups: Dict of group_name -> data
        baseline_key: Key of baseline group
        alpha: Significance level
        correction: "bonferroni" or "holm"

    Returns:
        Dict of group_name -> ComparisonResult
    """
    baseline = np.array(groups[baseline_key])
    comparison_keys = [k for k in groups if k != baseline_key]

    # First pass: compute raw p-values
    raw_results = {}
    p_values = []

    for key in comparison_keys:
        group = np.array(groups[key])
        result = compare_groups(baseline, group, alpha=alpha, paired=True)
        raw_results[key] = result
        p_values.append(result.p_value)

    # Apply correction
    if correction == "bonferroni":
        corrected = bonferroni_correction(p_values, alpha)
    else:
        corrected = holm_bonferroni_correction(p_values, alpha)

    # Update results with corrected significance
    final_results = {}
    for i, key in enumerate(comparison_keys):
        result = raw_results[key]
        adj_p, significant = corrected[i][:2]

        # Update significance level based on adjusted p
        if adj_p < 0.001:
            sig_level = "***"
        elif adj_p < 0.01:
            sig_level = "**"
        elif adj_p < 0.05:
            sig_level = "*"
        else:
            sig_level = ""

        final_results[key] = ComparisonResult(
            group1_mean=result.group1_mean,
            group2_mean=result.group2_mean,
            difference=result.difference,
            t_statistic=result.t_statistic,
            p_value=adj_p,
            cohens_d=result.cohens_d,
            significant=significant,
            significance_level=sig_level,
        )

    return final_results


def format_results_table(
    results: Dict[str, StatisticalResult],
    title: str = "Results",
) -> str:
    """Format results as ASCII table."""
    lines = [
        f"\n{title}",
        "=" * 60,
        f"{'Group':<20} | {'Mean ± Std':>20} | {'95% CI':>18}",
        "-" * 60,
    ]

    for name, r in results.items():
        lines.append(
            f"{name:<20} | {r.mean:>8.4f} ± {r.std:.4f} | [{r.ci_lower:.4f}, {r.ci_upper:.4f}]"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def format_comparison_table(
    results: Dict[str, ComparisonResult],
    baseline_name: str,
    title: str = "Comparison",
) -> str:
    """Format comparison results as ASCII table."""
    lines = [
        f"\n{title} (vs {baseline_name})",
        "=" * 70,
        f"{'Group':<20} | {'Diff':>10} | {'t':>8} | {'p':>10} | {'d':>6}",
        "-" * 70,
    ]

    for name, r in results.items():
        lines.append(
            f"{name:<20} | {r.difference:>+10.4f} | {r.t_statistic:>8.2f} | "
            f"{r.p_value:>10.4e} | {r.cohens_d:>5.2f}{r.significance_level}"
        )

    lines.append("=" * 70)
    lines.append("* p < 0.05, ** p < 0.01, *** p < 0.001")
    return "\n".join(lines)
