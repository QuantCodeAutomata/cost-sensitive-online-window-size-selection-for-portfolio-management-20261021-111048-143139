"""
Experiment 4: Additional synthetic analyses — regime-shift timing and expanded expert set.

Paper: Appendix B.2 "Additional Synthetic Experiments"
Two sub-experiments:
  4a: Vary timing of regime shift with N = {5, 10, 21, 63} (4 experts)
  4b: Expand expert set to N = {5, 10, 21, 30, 41, 50, 63, 126} (8 experts)

Confirmed: paper shows weight-evolution plots (Figures 3 and 4).
Assumed: exact shift dates and DGP parameters not specified; uses same parameters
         as exp_1 (mu, alpha, sigma, seed).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from config import (
    SYNTHETIC_NUM_ASSETS, SYNTHETIC_TOTAL_DAYS, SYNTHETIC_MU, SYNTHETIC_ALPHA,
    SYNTHETIC_SIGMA, SYNTHETIC_RANDOM_SEED, WINDOWS_4, WINDOWS_8,
    TRANSACTION_COST_C, FIXED_SHARE_EPSILON, COV_REGULARIZATION_LAMBDA,
)
from exp.exp1_synthetic import run_exp1

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp4a_timing_sensitivity(
    shift_fractions: list[float] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Experiment 4a: Test sensitivity of expert weights to regime-shift timing.

    Paper: Appendix B.2, Figure 3 — "Different switching timing"
    Runs multiple regime-shift scenarios with different break points.

    Parameters
    ----------
    shift_fractions : list[float] or None
        Fraction of total days at which the regime shifts.
        If None, uses [0.25, 0.50, 0.75] (early, mid, late switch).
        Assumed: paper shows "different switching timing" without specifying values.
    verbose : bool

    Returns
    -------
    all_results : list[dict]
    """
    if shift_fractions is None:
        shift_fractions = [0.25, 0.50, 0.75]

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXP4a: Regime-Shift Timing Sensitivity")
        print(f"  Shift fractions: {shift_fractions}")
        print(f"  Windows (4-expert): {WINDOWS_4}")
        print(f"{'='*70}\n")

    all_results = []
    for frac in shift_fractions:
        break_day = int(SYNTHETIC_TOTAL_DAYS * frac)
        if break_day <= max(WINDOWS_4) or break_day >= SYNTHETIC_TOTAL_DAYS - max(WINDOWS_4):
            break_day = max(int(SYNTHETIC_TOTAL_DAYS * frac), max(WINDOWS_4) + 1)
        regime_lengths = [break_day, SYNTHETIC_TOTAL_DAYS - break_day]
        tag = f"exp4a_shift{int(frac*100)}pct"

        if verbose:
            print(f"\nRunning shift at {frac*100:.0f}% ({break_day} days)...")

        results = run_exp1(
            windows=WINDOWS_4,
            regime_windows=[63, 5],     # same regime structure as exp_1
            regime_lengths=regime_lengths,
            tag=tag,
            verbose=False,
        )
        results["shift_fraction"] = frac
        results["break_day"] = break_day
        all_results.append(results)
        if verbose:
            print(f"  tag={tag}: Hedge regret={results['hedge_cumulative_regret_final']:.4f}, "
                  f"FS regret={results['fs_cumulative_regret_final']:.4f}")

    # Save combined timing results
    out_path = os.path.join(RESULTS_DIR, "exp4a_timing_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    if verbose:
        print(f"\nExp4a timing metrics saved to {out_path}")

    # Plot comparison of expert weights for different shift timings
    _plot_timing_comparison(all_results, shift_fractions, save_dir=RESULTS_DIR)
    return all_results


def run_exp4b_expanded_expert_set(verbose: bool = True) -> dict:
    """
    Experiment 4b: Expand expert set from 4 to 8 windows.

    Paper: Appendix B.2, Figure 4 — "Different number of window sizes"
    Uses same synthetic DGP (midpoint regime shift) with 8-window expert set.

    Confirmed: N expanded to {5, 10, 21, 30, 41, 50, 63, 126} (paper Appendix B.2).
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"EXP4b: Expanded Expert Set Sensitivity")
        print(f"  Windows (8-expert): {WINDOWS_8}")
        print(f"{'='*70}\n")

    results = run_exp1(
        windows=WINDOWS_8,
        regime_windows=[63, 5],
        regime_lengths=[SYNTHETIC_TOTAL_DAYS // 2, SYNTHETIC_TOTAL_DAYS - SYNTHETIC_TOTAL_DAYS // 2],
        tag="exp4b_8experts",
        verbose=verbose,
    )

    # Save results
    out_path = os.path.join(RESULTS_DIR, "exp4b_expanded_experts_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nExp4b metrics saved to {out_path}")
    return results


def _plot_timing_comparison(all_results: list[dict], shift_fractions: list[float], save_dir: str):
    """
    Plot comparison of final aggregation weights across different shift timings.
    """
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4), sharey=True)
    if len(all_results) == 1:
        axes = [axes]
    windows = WINDOWS_4
    for ax, res, frac in zip(axes, all_results, shift_fractions):
        hedge_q = res["hedge_final_expert_q"]
        fs_q = res["fs_final_expert_q"]
        x = np.arange(len(windows))
        width = 0.35
        ax.bar(x - width / 2, hedge_q, width, label="Hedge", alpha=0.7)
        ax.bar(x + width / 2, fs_q, width, label="Fixed Share", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={w}" for w in windows], fontsize=8)
        ax.set_title(f"Shift at {frac*100:.0f}%\n(day {res['break_day']})")
        ax.set_ylim(0, 1)
        if ax == axes[0]:
            ax.set_ylabel("Final aggregation weight $q_j$")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Expert Weights at End of Simulation — Different Shift Timings", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "exp4a_timing_weight_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def run_exp4(verbose: bool = True) -> dict:
    """Run both sub-experiments of Experiment 4."""
    timing_results = run_exp4a_timing_sensitivity(verbose=verbose)
    expanded_results = run_exp4b_expanded_expert_set(verbose=verbose)
    return {
        "exp4a_timing": timing_results,
        "exp4b_expanded": expanded_results,
    }


if __name__ == "__main__":
    results = run_exp4()
    print("\nExp4 completed successfully.")
