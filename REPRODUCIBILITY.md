# Reproducibility Guide

This document provides complete instructions to reproduce all results in the paper "Emergent Specialization in Multi-Agent Systems."

## Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems.git
cd Emergent-Specialization-in-Multi-Agent-Systems

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all experiments
python experiments/exp_unified_pipeline.py
```

## Repository Structure

```
emergent_specialization/
├── paper/              # LaTeX source
│   ├── main.tex        # Full paper
│   └── figures/        # Publication figures
├── src/                # Core library
│   ├── agents/         # NichePopulation algorithm
│   ├── baselines/      # MARL baselines (QMIX, MAPPO, IQL)
│   ├── domains/        # 6 domain implementations
│   └── theory/         # Formal propositions
├── experiments/        # Reproducible experiments
│   └── exp_unified_pipeline.py  # Main entry point
├── data/               # Real-world data (6 domains)
├── results/            # Experiment outputs
└── scripts/            # Data download & figure generation
```

## System Requirements

- **Python**: 3.8+
- **Memory**: 4GB RAM
- **Storage**: 500MB for data
- **Time**: ~30 minutes for full experiments

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
```

## Experiments

### Main Experiment: Unified Pipeline

Runs all core experiments across all 6 domains:

```bash
python experiments/exp_unified_pipeline.py
```

**Output**: `results/unified_pipeline/results.json`

### Individual Experiments

| Experiment | Command | Output |
|------------|---------|--------|
| Hypothesis Tests | `python experiments/exp_hypothesis_tests.py` | `results/hypothesis_tests/` |
| Method Specialization | `python experiments/exp_method_specialization.py` | `results/method_specialization/` |
| MARL Comparison | `python experiments/exp_marl_standalone.py` | `results/marl_comparison/` |
| λ Ablation | `python experiments/exp_lambda_ablation.py` | `results/lambda_ablation/` |

### Generate Figures

```bash
python scripts/generate_neurips_figures.py
```

## Data

All data is **100% real-world data** from verified public sources:

| Domain | Source | Records |
|--------|--------|---------|
| Crypto | Bybit Exchange | 44,000+ |
| Commodities | FRED (US Gov) | 5,630 |
| Weather | Open-Meteo | 9,105 |
| Solar | Open-Meteo | 116,834 |
| Traffic | NYC TLC | 2,879 |
| Air Quality | Open-Meteo | 2,880 |

See `data/README.md` for download instructions.

## Expected Results

Running the unified pipeline should produce:

| Domain | SI (Niche) | vs Homo | Cohen's d |
|--------|------------|---------|-----------|
| Crypto | 0.786 ± 0.06 | +26% | 20.05 |
| Commodities | 0.773 ± 0.06 | +25% | 19.89 |
| Weather | 0.758 ± 0.05 | +24% | 23.44 |
| Solar | 0.764 ± 0.04 | +25% | 25.71 |
| Traffic | 0.573 ± 0.05 | +18% | 15.86 |
| Air Quality | 0.826 ± 0.04 | +27% | 32.06 |

All differences should be statistically significant (p < 0.001).

## Random Seeds

All experiments use fixed random seeds for reproducibility:
- Base seed: 42
- Per-trial seed: 42 + trial_index

## Statistical Methods

- **Trials per experiment**: 30
- **Statistical test**: Welch's t-test (two-sided)
- **Significance threshold**: α = 0.001
- **Effect size**: Cohen's d
- **Multiple testing**: Bonferroni correction where applicable

## Docker

```bash
docker build -t emergent-specialization .
docker run emergent-specialization python experiments/exp_unified_pipeline.py
```

## Contact

For questions about reproducibility:
- **Author**: Yuhao Li
- **Email**: li88@sas.upenn.edu
- **Repository**: https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems
