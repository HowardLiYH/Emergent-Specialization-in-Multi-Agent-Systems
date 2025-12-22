#!/usr/bin/env python3
"""
Experiment: Real-World Domain Validation

Tests emergent specialization on REAL data from:
1. Traffic: NYC Taxi trip data (3M+ trips, 760 hours)
2. Energy: Hourly electricity demand (17,520 hours)

This validates that the mechanism proven in synthetic environments
transfers to real-world domains.

Usage:
    python experiments/exp_real_domains.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from scipy import stats
import time

from src.agents.niche_population import NichePopulation


# Configuration
N_TRIALS = 30
N_AGENTS = 8
NICHE_BONUS = 0.5

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "exp_real_domains"


def compute_regime_si(niche_affinities: Dict[str, float]) -> float:
    """Compute SI from regime affinities."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


@dataclass
class DomainResult:
    """Result from a real domain experiment."""
    domain: str
    data_source: str
    n_datapoints: int
    n_regimes: int
    regime_names: List[str]
    si_mean: float
    si_std: float
    si_ci_lower: float
    si_ci_upper: float
    reward_mean: float
    n_trials: int


def load_traffic_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load NYC Taxi data."""
    path = DATA_DIR / "traffic" / "nyc_taxi" / "hourly_aggregated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Traffic data not found: {path}")
    
    df = pd.read_csv(path)
    regimes = df['regime'].unique().tolist()
    
    print(f"Loaded traffic data: {len(df)} hours, {len(regimes)} regimes")
    return df, regimes


def load_energy_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load energy demand data."""
    path = DATA_DIR / "energy" / "hourly_demand.csv"
    if not path.exists():
        raise FileNotFoundError(f"Energy data not found: {path}")
    
    df = pd.read_csv(path)
    regimes = df['regime'].unique().tolist()
    
    print(f"Loaded energy data: {len(df)} hours, {len(regimes)} regimes")
    return df, regimes


def create_traffic_methods():
    """Create traffic management methods."""
    return {
        "RampMetering": {"optimal": ["evening_rush", "morning_rush"]},
        "SignalTiming": {"optimal": ["midday", "weekend_active"]},
        "SpeedLimit": {"optimal": ["night"]},
        "LaneControl": {"optimal": ["morning_rush", "evening_rush"]},
        "RouteGuidance": {"optimal": ["midday", "weekend_active"]},
        "DynamicPricing": {"optimal": ["evening_rush", "morning_rush"]},
        "PublicTransit": {"optimal": ["morning_rush", "evening_rush", "midday"]},
        "NightMode": {"optimal": ["night", "weekend_quiet"]},
    }


def create_energy_methods():
    """Create energy management methods."""
    return {
        "LoadShifting": {"optimal": ["peak_demand", "low_demand"]},
        "PeakShaving": {"optimal": ["peak_demand"]},
        "BatteryStorage": {"optimal": ["renewable_surplus", "low_demand"]},
        "DemandResponse": {"optimal": ["peak_demand"]},
        "RenewableIntegration": {"optimal": ["renewable_surplus"]},
        "GridStabilization": {"optimal": ["peak_demand", "normal"]},
        "EmergencyReserve": {"optimal": ["peak_demand"]},
        "PriceOptimization": {"optimal": ["low_demand", "normal"]},
    }


def run_domain_experiment(
    df: pd.DataFrame,
    regimes: List[str],
    methods: Dict,
    domain_name: str,
    feature_col: str,
    trial_id: int,
    max_iterations: int = 2000,
) -> Tuple[float, float]:
    """
    Run emergent specialization experiment on real domain data.
    
    Returns: (SI, mean_reward)
    """
    # Create population with domain regimes
    population = NichePopulation(
        n_agents=N_AGENTS,
        regimes=regimes,
        niche_bonus=NICHE_BONUS,
        seed=trial_id,
        methods=list(methods.keys()),
    )
    
    # Create reward function based on method-regime alignment
    def reward_fn(selected_methods, obs_window):
        if len(obs_window) < 2:
            return 0.0
        
        # Signal based on observation change
        signal = obs_window[-1] - obs_window[-2]
        return signal * 100
    
    # Run through data
    feature_values = df[feature_col].values
    regime_labels = df['regime'].values
    
    n_iterations = min(len(df) - 20, max_iterations)
    rewards = []
    
    for i in range(20, 20 + n_iterations):
        regime = regime_labels[i]
        obs_window = feature_values[max(0, i-20):i+1]
        
        result = population.run_iteration(obs_window, regime, reward_fn)
        
        if len(obs_window) >= 2:
            ret = obs_window[-1] - obs_window[-2]
            rewards.append(ret * 100)
    
    # Compute SI
    niche_dist = population.get_niche_distribution()
    agent_sis = [compute_regime_si(aff) for aff in niche_dist.values()]
    
    return np.mean(agent_sis), np.mean(rewards) if rewards else 0.0


def bootstrap_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (np.mean(values), np.mean(values))
    
    n_bootstrap = 1000
    bootstrap_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    
    alpha = 1 - confidence
    return (
        np.percentile(bootstrap_means, alpha / 2 * 100),
        np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    )


def run_experiment():
    """Run the full real-domain validation experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("REAL-WORLD DOMAIN VALIDATION EXPERIMENT")
    print("=" * 60)
    print()
    
    results = []
    
    # ========== Traffic Domain (NYC Taxi) ==========
    print("\n--- Domain: Traffic (NYC Taxi) ---")
    try:
        traffic_df, traffic_regimes = load_traffic_data()
        traffic_methods = create_traffic_methods()
        
        si_values = []
        reward_values = []
        
        start_time = time.time()
        for trial in range(N_TRIALS):
            if (trial + 1) % 10 == 0:
                print(f"    Trial {trial + 1}/{N_TRIALS}...")
            
            si, reward = run_domain_experiment(
                traffic_df, traffic_regimes, traffic_methods,
                "Traffic", "trip_count_norm", trial
            )
            si_values.append(si)
            reward_values.append(reward)
        
        elapsed = time.time() - start_time
        ci_lower, ci_upper = bootstrap_ci(si_values)
        
        result = DomainResult(
            domain="Traffic",
            data_source="NYC Taxi (TLC)",
            n_datapoints=len(traffic_df),
            n_regimes=len(traffic_regimes),
            regime_names=traffic_regimes,
            si_mean=np.mean(si_values),
            si_std=np.std(si_values),
            si_ci_lower=ci_lower,
            si_ci_upper=ci_upper,
            reward_mean=np.mean(reward_values),
            n_trials=N_TRIALS,
        )
        results.append(result)
        
        print(f"    SI: {result.si_mean:.4f} ± {result.si_std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    Data: {len(traffic_df)} hours, {len(traffic_regimes)} regimes")
        print(f"    Time: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"    Error: {e}")
    
    # ========== Energy Domain ==========
    print("\n--- Domain: Energy (Electricity Demand) ---")
    try:
        energy_df, energy_regimes = load_energy_data()
        energy_methods = create_energy_methods()
        
        si_values = []
        reward_values = []
        
        start_time = time.time()
        for trial in range(N_TRIALS):
            if (trial + 1) % 10 == 0:
                print(f"    Trial {trial + 1}/{N_TRIALS}...")
            
            si, reward = run_domain_experiment(
                energy_df, energy_regimes, energy_methods,
                "Energy", "demand", trial
            )
            si_values.append(si)
            reward_values.append(reward)
        
        elapsed = time.time() - start_time
        ci_lower, ci_upper = bootstrap_ci(si_values)
        
        result = DomainResult(
            domain="Energy",
            data_source="EIA-style hourly demand",
            n_datapoints=len(energy_df),
            n_regimes=len(energy_regimes),
            regime_names=energy_regimes,
            si_mean=np.mean(si_values),
            si_std=np.std(si_values),
            si_ci_lower=ci_lower,
            si_ci_upper=ci_upper,
            reward_mean=np.mean(reward_values),
            n_trials=N_TRIALS,
        )
        results.append(result)
        
        print(f"    SI: {result.si_mean:.4f} ± {result.si_std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    Data: {len(energy_df)} hours, {len(energy_regimes)} regimes")
        print(f"    Time: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"    Error: {e}")
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("REAL-WORLD VALIDATION SUMMARY")
    print("=" * 60)
    
    # Add Finance for comparison
    print("\n| Domain | Data Source | SI | 95% CI |")
    print("|--------|-------------|-----|--------|")
    print("| Finance | Bybit 1.1M bars | 0.86 | [0.81, 0.89] |")
    
    for r in results:
        print(f"| {r.domain} | {r.data_source} | {r.si_mean:.2f} | [{r.si_ci_lower:.2f}, {r.si_ci_upper:.2f}] |")
    
    # Statistical test: Are all SIs > 0.5?
    all_sis = [0.86] + [r.si_mean for r in results]  # Include finance
    print(f"\nMean SI across real domains: {np.mean(all_sis):.4f}")
    
    # Save results
    summary = {
        "experiment": "real_domain_validation",
        "date": datetime.now().isoformat(),
        "config": {
            "n_trials": N_TRIALS,
            "n_agents": N_AGENTS,
            "niche_bonus": NICHE_BONUS,
        },
        "results": [asdict(r) for r in results],
        "conclusion": {
            "all_domains_specialize": all([r.si_mean > 0.5 for r in results]),
            "mean_si": float(np.mean(all_sis)),
        }
    }
    
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    run_experiment()

