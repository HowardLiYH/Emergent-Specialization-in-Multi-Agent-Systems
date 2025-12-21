"""
Analysis module: Metrics, statistical tests, and figure generation.
"""

from .specialization import (
    compute_specialization_index,
    compute_population_diversity,
    compute_regime_coverage,
    SpecializationTracker,
    SpecializationMetrics,
)

from .statistical_tests import (
    paired_t_test,
    one_sample_t_test,
    bootstrap_confidence_interval,
    bonferroni_correction,
    TestResult,
)

__all__ = [
    # Specialization metrics
    "compute_specialization_index",
    "compute_population_diversity",
    "compute_regime_coverage",
    "SpecializationTracker",
    "SpecializationMetrics",
    # Statistical tests
    "paired_t_test",
    "one_sample_t_test",
    "bootstrap_confidence_interval",
    "bonferroni_correction",
    "TestResult",
]
