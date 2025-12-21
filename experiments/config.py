"""
Experiment Configuration.

Central configuration for all experiments to ensure consistency.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""

    # Experiment identification
    experiment_name: str = "default"

    # Random seeds
    base_seed: int = 42
    n_trials: int = 100

    # Environment settings
    n_bars: int = 500
    regime_names: List[str] = field(
        default_factory=lambda: ["trend_up", "trend_down", "mean_revert", "volatile"]
    )
    regime_duration_mean: int = 50
    regime_duration_std: int = 15

    # Population settings
    n_agents: int = 5
    max_methods: int = 3
    transfer_frequency: int = 10
    transfer_tau: float = 0.1

    # Analysis settings
    checkpoint_iterations: List[int] = field(
        default_factory=lambda: [0, 50, 100, 200, 300, 400, 500]
    )

    # Output
    results_dir: str = "results"
    save_checkpoints: bool = True
    verbose: bool = True


# Experiment-specific configurations
EXP1_CONFIG = ExperimentConfig(
    experiment_name="exp1_emergence",
    n_trials=100,
    n_bars=500,
)

EXP2_CONFIG = ExperimentConfig(
    experiment_name="exp2_diversity_value",
    n_trials=100,
    n_bars=500,
)

EXP3_CONFIG = ExperimentConfig(
    experiment_name="exp3_population_size",
    n_trials=50,
    n_bars=500,
)

EXP4_CONFIG = ExperimentConfig(
    experiment_name="exp4_transfer_frequency",
    n_trials=50,
    n_bars=500,
)

EXP5_CONFIG = ExperimentConfig(
    experiment_name="exp5_regime_transitions",
    n_trials=50,
    n_bars=300,
    regime_duration_mean=100,  # Longer regimes for clear transitions
    regime_duration_std=5,     # More consistent duration
)

EXP6_CONFIG = ExperimentConfig(
    experiment_name="exp6_real_data",
    n_trials=1,  # Single run on real data
)
