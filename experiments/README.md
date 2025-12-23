# Experiments

This folder contains the key experiments for the paper "Emergent Specialization in Multi-Agent Systems."

## Quick Start: Reproduce All Results

```bash
# From repository root
python experiments/exp_unified_pipeline.py
```

This runs all core experiments across all 6 domains with identical configuration.

## Experiment Files

| File | Purpose | Key Finding |
|------|---------|-------------|
| `exp_unified_pipeline.py` | **Main reproducibility script** | Runs all experiments on all 6 domains |
| `exp_method_specialization.py` | Method specialization analysis | Agents specialize in prediction methods (+26.5% improvement) |
| `exp_hypothesis_tests.py` | Formal hypothesis testing (H1-H4) | All 4 hypotheses pass (p < 0.001) |
| `exp_marl_comparison.py` | MARL baseline comparison | NichePopulation achieves 4.3× higher SI than MARL |
| `exp_lambda_ablation.py` | λ ablation study | Competition alone (λ=0) induces SI > 0.25 |
| `exp_lambda_zero_real.py` | λ=0 on real domains | Validates core thesis on real data |
| `exp_mechanism_ablation.py` | Isolate competition vs bonus effects | Competition is primary driver |
| `exp_regime_shuffle.py` | Negative control test | Specialization robust to regime relabeling |

## Configuration

All experiments use identical settings (see `config.py`):

| Parameter | Value |
|-----------|-------|
| Trials | 30 |
| Iterations | 500 |
| Agents | 8 |
| Default λ | 0.3 |

## Results

Results are saved to `../results/` with JSON files and figures.
