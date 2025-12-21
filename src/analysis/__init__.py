"""
Analysis module: Specialization and diversity metrics.

Core metrics:
- Specialization Index (SI): How focused is an agent on specific methods?
- Population Diversity Index (PDI): How different are agents from each other?
- Regime Win Rate Matrix: Which agents win in which regimes?
- Statistical Tests: Hypothesis testing utilities
"""

from .specialization import (
    compute_specialization_index,
    compute_population_diversity,
    compute_method_coverage,
    SpecializationTracker,
)
from .statistical_tests import (
    paired_t_test,
    anova_with_tukey,
    bootstrap_confidence_interval,
)

__all__ = [
    "compute_specialization_index",
    "compute_population_diversity",
    "compute_method_coverage",
    "SpecializationTracker",
    "paired_t_test",
    "anova_with_tukey",
    "bootstrap_confidence_interval",
]
