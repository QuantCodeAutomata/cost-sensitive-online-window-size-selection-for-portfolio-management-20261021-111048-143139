# Cost-Sensitive Online Window Size Selection for Portfolio Management

Reproduction of the paper:
> **Cost-Sensitive Online Window Size Selection for Portfolio Management**

---

## Overview

This repository implements the two-level portfolio management framework from the paper:

1. **Expert layer**: For each candidate window size `j ∈ N`, solve a cost-sensitive mean-variance optimisation at each date `t`:
   ```
   max_{w ∈ W}  μ̂_j(t)ᵀw  −  0.5 wᵀΣ̂_j(t)w  −  c ‖w − w_j(t−1)‖₁
   ```
   over the long-only simplex W = {w ≥ 0, Σw = 1}.

2. **Aggregation layer**: Online algorithms (Hedge, Fixed Share) combine expert recommendations using cost-sensitive expert losses:
   ```
   ℓ(f_j(t), y(t)) = −f_j(t)ᵀy(t) + c ‖f_j(t) − f_j(t−1)‖₁
   ```

---

## Experiments

| ID | Title | Data | API Required |
|----|-------|------|-------------|
| exp_1 | Synthetic regime-shift | 32 synthetic stocks, 2000 days | No |
| exp_2 | DJIA historical study | 30 DJIA stocks + BIL, 2021–2024 | Yes |
| exp_3 | S&P 500 historical study | 481 S&P 500 + BIL, 2020–2024 | Yes |
| exp_4 | Additional synthetic analyses | Same DGP, varied regimes/windows | No |

---

## Repository Structure

```
├── src/
│   ├── dgp.py          — Synthetic DGP (Section 4.1)
│   ├── expert.py       — Cost-sensitive MV expert (Problem 2.1)
│   ├── aggregator.py   — Hedge & Fixed Share algorithms
│   ├── metrics.py      — Tracking regret, account value, performance
│   └── baselines.py    — UP, EG, PAMR, CWMR, OLMAR baselines
├── exp/
│   ├── exp1_synthetic.py
│   ├── exp2_djia.py
│   ├── exp3_sp500.py
│   └── exp4_sensitivity.py
├── tests/              — pytest unit tests
├── data/               — Data fetched at runtime (not committed)
├── results/            — Output metrics, plots, RESULTS.md
├── config.py           — Paper-derived parameters
└── reproduce.sh        — Entry point
```

---

## Setup

```bash
pip install -r requirements.txt
```

**For historical experiments (exp_2, exp_3):**
```bash
export DATALOADER_API_KEY="<your-key-from-quantx.limex.pro>"
export MASSIVE_API_KEY="<your-key-from-massive.com>"
```

---

## Reproduce

```bash
chmod +x reproduce.sh

# Run all (exp_1, exp_4 run without API; exp_2, exp_3 require API keys):
./reproduce.sh

# Run only synthetic experiments (no API required):
./reproduce.sh --exp1 --exp4

# Run only unit tests:
./reproduce.sh --tests
```

Results, metrics, and plots are saved to `results/`.

---

## Parameters (from paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Transaction cost `c` | 0.001 | Section 2.1, eq. (1) |
| DJIA tickers | 30 + BIL | Appendix C |
| DJIA date range | 2021-03-04 to 2024-02-25 | Section 4.2 / Appendix B.4 |
| S&P 500 components | 481 + BIL | Appendix B.5 |
| S&P 500 date range | 2020-01-02 to 2024-08-01 | Appendix B.5 |
| Synthetic: 32 stocks | 2000 days | Section 4.1 |
| Synthetic windows | {5, 10, 21, 63} | Section 4.1 |
| Expanded windows | {5, 10, 21, 30, 41, 50, 63, 126} | Appendix B.2 |

### Assumed Parameters (not specified by paper)

| Parameter | Assumed Value | Justification |
|-----------|--------------|---------------|
| `mu` (synthetic DGP) | 5×10⁻⁴ | Small positive daily return |
| `alpha` (DGP signal ratio) | 0.3 | Moderate mean-reversion |
| `sigma` (noise std) | 0.01 | ~1% daily noise |
| Random seed | 42 | Reproducibility |
| Cov. regularisation λ | 1×10⁻⁴ | Ridge for positive definiteness |
| Fixed Share epsilon | 0.5 | K(T) formula exponent |

---

## Key Results

See `results/RESULTS.md` for full experimental outcomes.

---

## Assumptions and Deviations

See `results/RESULTS.md` (Assumptions section) and inline docstrings for all
deviations from the paper and unresolved details.
