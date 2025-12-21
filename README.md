# Emergent Specialization in Multi-Agent Trading

**NeurIPS 2025 Submission**

## Abstract

We demonstrate that populations of learning agents in financial markets exhibit emergent specialization without explicit supervision. Agents naturally partition the regime space through competitive selection pressure, resembling Evolutionary Stable Strategies (ESS) in biological systems.

## Key Research Questions

1. **RQ1**: Do agents naturally specialize without supervision?
2. **RQ2**: Does diversity improve collective performance?
3. **RQ3**: How does population size affect emergence speed?
4. **RQ4**: What is the optimal knowledge transfer frequency?

## Project Structure

```
emergent_specialization/
├── src/                          # Core implementation
│   ├── environment/              # Synthetic market environment
│   │   ├── synthetic_market.py   # Main environment class
│   │   └── regime_generators.py  # Market regime generation
│   ├── agents/                   # Agent implementations
│   │   ├── method_selector.py    # Thompson Sampling agent
│   │   ├── population.py         # Population dynamics
│   │   └── inventory.py          # Trading method inventory
│   ├── analysis/                 # Analysis tools
│   │   ├── specialization.py     # Specialization metrics
│   │   └── statistical_tests.py  # Statistical tests
│   └── baselines/                # Baseline implementations
│       ├── oracle.py             # Oracle Specialist
│       ├── homogeneous.py        # Single-agent population
│       └── random_selection.py   # Random baseline
├── experiments/                  # Experiment scripts
│   ├── exp1_emergence.py         # Emergence experiment
│   ├── config.py                 # Experiment configuration
│   └── runner.py                 # Unified runner
├── paper/                        # Paper materials
│   ├── figures/                  # Generated figures
│   └── tables/                   # Result tables
├── results/                      # Experiment results
└── tests/                        # Unit tests
```

## Core Concepts

### Synthetic Market Environment

We use a controlled synthetic environment with 4 market regimes:
- **Trend Up**: Positive drift, low volatility
- **Trend Down**: Negative drift, low volatility
- **Mean Revert**: No drift, mean reverting
- **Volatile**: High volatility, no clear direction

This enables rigorous, reproducible experiments with ground-truth regime labels.

### Method Inventory

Each agent selects from 11 trading methods, optimized for different regimes:

| Method | Category | Best Regime |
|--------|----------|-------------|
| RSI_Oversold | Momentum | Mean Revert |
| MACD_Cross | Momentum | Trend Up/Down |
| Bollinger_Break | Mean Reversion | Mean Revert |
| HMM_Regime | Statistical | Volatile |
| ... | ... | ... |

### Specialization Metrics

1. **Specialization Index (SI)**: How concentrated is method usage? (0=uniform, 1=single method)
2. **Regime Purity (RP)**: Does the agent win in one regime or many?
3. **Population Diversity (PD)**: How different are agents from each other?
4. **Niche Overlap (NO)**: Do agents compete for the same regimes?
5. **Regime Coverage (RC)**: Are all regimes covered by specialists?

## Experiments

### Experiment 1: Emergence of Specialists
- **Hypothesis**: After training, agents specialize (SI > 0.6)
- **Protocol**: 100 trials × 500 iterations
- **Analysis**: Paired t-test, effect size, 95% CI

### Experiment 2: Value of Diversity
- **Hypothesis**: Diverse populations outperform homogeneous
- **Protocol**: Compare 5-agent diverse vs 5× single-agent
- **Baseline**: Oracle Specialist (knows true regime)

### Experiment 3: Population Size
- **Hypothesis**: Optimal population size exists
- **Protocol**: Test N ∈ {3, 5, 7, 10, 15, 20}

### Experiment 4: Transfer Frequency
- **Hypothesis**: Too-frequent transfer prevents specialization
- **Protocol**: Test τ ∈ {1, 5, 10, 25, 50}

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Running Experiments

```bash
# Run all experiments (full suite)
python -m experiments.runner --all

# Run specific experiment
python -m experiments.runner -e exp1

# Run multiple experiments
python -m experiments.runner -e exp1 exp2 exp3

# Quick test (10 trials instead of 100)
python -m experiments.runner -e exp1 --trials 10

# Custom seed
python -m experiments.runner --all --seed 123

# Custom output directory
python -m experiments.runner --all -o my_results/
```

### Running Individual Experiments

```bash
# Experiment 1: Emergence
python experiments/exp1_emergence.py --trials 100

# Experiment 2: Diversity Value
python experiments/exp2_diversity_value.py --trials 100

# Experiment 3: Population Size
python experiments/exp3_population_size.py --trials 50 --sizes 3 5 7 10

# Experiment 4: Transfer Frequency
python experiments/exp4_transfer_frequency.py --trials 50

# Experiment 5: Regime Transitions
python experiments/exp5_regime_transitions.py --trials 50

# Experiment 6: Real Data Validation
python experiments/exp6_real_data.py --data-dir ../MAS_Final_With_Agents/data/bybit
```

## Generating Figures

```bash
# Generate all paper figures
python -c "from src.analysis.figures import generate_all_figures; generate_all_figures()"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_core.py::TestPopulation -v
```

## Expected Results

| Experiment | Key Metric | Expected Value | p-value |
|------------|------------|----------------|---------|
| Exp 1: Emergence | Final SI | 0.65 ± 0.10 | < 0.001 |
| Exp 2: Diversity | Diverse vs Homogeneous | +35% | < 0.001 |
| Exp 3: Pop Size | Optimal N* | 5-7 | - |
| Exp 4: Transfer | Optimal τ* | 10-25 | - |
| Exp 5: Transitions | Switch Rate | >0.8 | < 0.05 |
| Exp 6: Real Data | SI matches synthetic | Yes | - |

## Citation

```bibtex
@inproceedings{emergent_specialization_2025,
  title={Emergent Specialization in Multi-Agent Trading:
         A Population-Based Approach to Market Regime Adaptation},
  author={MAS Finance Research Team},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details.
