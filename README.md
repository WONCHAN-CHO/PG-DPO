# PG-DPO Experiments
![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains a standalone Python script (`Infinite_to_finite.py`) that trains investment and consumption policies with stochastic labor income using actor–critic reinforcement learning and a hedging-only optimizer for convergence checks.

## Project structure
- `Infinite_to_finite.py` — full experiment script with model definitions, training loops, and plotting utilities.

## Requirements
- Python 3.10+
- PyTorch, NumPy, Matplotlib, and Seaborn (GPU is optional)

You can install dependencies with:
```bash
pip install torch numpy matplotlib seaborn
```

## Quick start
1. Ensure dependencies are installed and an optional CUDA device is available.
2. Run the full experiment suite:
```bash
python Infinite_to_finite.py
```
The script will seed the random number generators, execute the main actor–critic experiments, run convergence checks for varying horizons, and verify the Merton benchmark. Plots are displayed inline.

## Key components
- **Config**: Centralizes market, income, utility, and training hyperparameters. Device selection is automatic, preferring CUDA when available.
- **ActorNet / CriticNet**: Neural networks for policy and value prediction. The actor outputs investment and consumption ratios; the critic approximates the value function.
- **train_actor_critic**: Performs rollouts over a finite horizon `T`, optimizing both networks with correlated income and asset shocks (`rho`) and configurable income volatility (`sigma_y`).
- **evaluate_networks**: Generates policy and value predictions on a grid of income-to-wealth ratios for analysis and plotting.
- **run_ppgdpo_experiment**: Trains a hedging-focused policy to study convergence and sensitivity to income risk, parameterized by horizon, risk aversion `gamma`, correlation `rho`, and income volatility `sigma_y`.
- **experiment_3_convergence_T / experiment_4_merton_verification**: Reproduce convergence toward the infinite-horizon solution and verify the Merton ratio when income volatility is zero.

## Tips for use
- Adjust horizons in `Config.horizons` or the `T_horizon` argument of `run_ppgdpo_experiment` to explore different planning windows.
- Modify `sigma_y` and `rho` to study how labor-income risk affects hedging demand. The hedging term is linear in the income-to-wealth ratio `z`.
- Training duration is controlled by `Config.n_epochs`; reduce it for quicker exploratory runs.

## Reproducibility
The script fixes seeds via `set_seed` in the `__main__` block. For repeated runs with a different seed, call `set_seed(<seed_value>)` before invoking the experiment functions.
