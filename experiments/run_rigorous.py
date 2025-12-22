"""
Orchestrator: Run All Rigorous Experiments

Runs all experiments with proper statistical rigor:
- 30+ trials per experiment
- Consistent random seeds
- Progress tracking
- Checkpointing
"""

import sys
sys.path.insert(0, '.')

import json
import time
from pathlib import Path
from datetime import datetime

# Import all experiments
from experiments.exp_rl_baselines_v2 import run_rl_baselines_v2
from experiments.diagnose_real_data import diagnose_real_data
from experiments.exp_tradeoff_analysis import run_tradeoff_analysis
from experiments.exp_adaptive_lambda import run_adaptive_lambda_experiment
from experiments.exp_transaction_costs import run_transaction_costs_experiment
from experiments.exp_out_of_sample import run_out_of_sample_experiment
from experiments.exp_regime_sensitivity import run_regime_sensitivity_experiment


def run_all_rigorous_experiments(
    n_trials: int = 30,
    quick_mode: bool = False,
):
    """
    Run all experiments with rigorous settings.

    Args:
        n_trials: Number of trials per experiment (30 for full, 5 for quick)
        quick_mode: If True, use reduced settings for testing
    """
    if quick_mode:
        n_trials = 5
        n_iterations = 500
    else:
        n_iterations = 2000

    output_dir = Path("results/rigorous_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    log = {
        "start_time": datetime.now().isoformat(),
        "n_trials": n_trials,
        "quick_mode": quick_mode,
        "experiments": {},
    }

    print("=" * 70)
    print("RIGOROUS EXPERIMENT SUITE")
    print(f"Trials: {n_trials}, Mode: {'Quick' if quick_mode else 'Full'}")
    print("=" * 70)

    experiments = [
        ("diagnose_real_data", lambda: diagnose_real_data(symbol="BTC")),
        ("rl_baselines_v2", lambda: run_rl_baselines_v2(n_trials=n_trials, n_iterations=n_iterations)),
        ("tradeoff_analysis", lambda: run_tradeoff_analysis(n_trials=n_trials, n_iterations=n_iterations)),
        ("adaptive_lambda", lambda: run_adaptive_lambda_experiment(n_trials=n_trials, n_iterations=n_iterations)),
        ("transaction_costs", lambda: run_transaction_costs_experiment(n_trials=n_trials, n_iterations=n_iterations)),
        ("out_of_sample", lambda: run_out_of_sample_experiment(n_trials=n_trials)),
        ("regime_sensitivity", lambda: run_regime_sensitivity_experiment(n_trials=n_trials, n_iterations=n_iterations)),
    ]

    for i, (name, run_fn) in enumerate(experiments):
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}: {name}")
        print(f"{'='*70}")

        exp_start = time.time()

        try:
            result = run_fn()
            exp_time = time.time() - exp_start

            log["experiments"][name] = {
                "status": "success",
                "duration_seconds": exp_time,
            }

            print(f"\n✅ {name} completed in {exp_time/60:.1f} minutes")

        except Exception as e:
            exp_time = time.time() - exp_start

            log["experiments"][name] = {
                "status": "failed",
                "error": str(e),
                "duration_seconds": exp_time,
            }

            print(f"\n❌ {name} failed: {e}")

    # Summary
    total_time = time.time() - start_time
    log["end_time"] = datetime.now().isoformat()
    log["total_duration_seconds"] = total_time

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("=" * 70)

    # Count successes/failures
    successes = sum(1 for e in log["experiments"].values() if e["status"] == "success")
    failures = len(log["experiments"]) - successes

    print(f"\nResults: {successes} succeeded, {failures} failed")

    for name, info in log["experiments"].items():
        status = "✅" if info["status"] == "success" else "❌"
        print(f"  {status} {name}: {info['duration_seconds']/60:.1f} min")

    # Save log
    with open(output_dir / "run_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nLog saved to {output_dir / 'run_log.json'}")

    return log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick version for testing")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials")
    args = parser.parse_args()

    run_all_rigorous_experiments(n_trials=args.trials, quick_mode=args.quick)
