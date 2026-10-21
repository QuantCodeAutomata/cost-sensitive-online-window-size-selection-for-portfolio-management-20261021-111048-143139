"""
Experiment 1: Synthetic regime-shift experiment.

Paper: Section 4.1 "Synthetic Data"
Objective: Test the two-level framework on a synthetic 32-stock market with a
known mid-sample regime shift. Compare Hedge vs Fixed Share over sliding-window
mean-variance experts.

Confirmed parameters (paper p.7):
  - 32 stocks, 2,000 business days, regime j_t=63 (days 1-1000) → j_t=5 (days 1001-2000)
  - Windows N = {5, 10, 21, 63}
  - c = 0.001

Assumed parameters (not specified by paper):
  - mu=5e-4, alpha=0.3, sigma=0.01, seed=42, w_0=uniform
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import json

from config import (
    SYNTHETIC_NUM_ASSETS, SYNTHETIC_TOTAL_DAYS, SYNTHETIC_REGIME_1_WINDOW,
    SYNTHETIC_REGIME_2_WINDOW, SYNTHETIC_MU, SYNTHETIC_ALPHA, SYNTHETIC_SIGMA,
    SYNTHETIC_RANDOM_SEED, WINDOWS_4, TRANSACTION_COST_C, INITIAL_ACCOUNT_VALUE,
    FIXED_SHARE_EPSILON, COV_REGULARIZATION_LAMBDA,
)
from src.dgp import generate_synthetic_returns, make_regime_schedule
from src.expert import ExpertSet
from src.aggregator import (
    run_aggregation, compute_hedge_eta, compute_switching_budget,
    compute_fixed_share_alpha, compute_expert_loss,
)
from src.metrics import (
    compute_cumulative_regret, compute_tracking_regret_dp,
    compute_hedge_regret_bound, compute_fixed_share_regret_bound,
    summarise_performance, cumulative_return, max_drawdown, annualised_sharpe,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_exp1(
    mu: float = SYNTHETIC_MU,
    alpha: float = SYNTHETIC_ALPHA,
    sigma: float = SYNTHETIC_SIGMA,
    seed: int = SYNTHETIC_RANDOM_SEED,
    windows: tuple = WINDOWS_4,
    c: float = TRANSACTION_COST_C,
    regime_windows: list = None,
    regime_lengths: list = None,
    total_days: int = SYNTHETIC_TOTAL_DAYS,
    n_assets: int = SYNTHETIC_NUM_ASSETS,
    tag: str = "exp1",
    verbose: bool = True,
) -> dict:
    """
    Run Experiment 1: Synthetic regime-shift.

    Returns dict of results including regret trajectories, weights, performance.
    """
    if regime_windows is None:
        regime_windows = [SYNTHETIC_REGIME_1_WINDOW, SYNTHETIC_REGIME_2_WINDOW]
    if regime_lengths is None:
        regime_lengths = [total_days // 2, total_days - total_days // 2]

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXP1: Synthetic Regime-Shift Experiment ({tag})")
        print(f"  n_assets={n_assets}, total_days={total_days}, windows={windows}")
        print(f"  regimes={list(zip(regime_windows, regime_lengths))}")
        print(f"  mu={mu}, alpha={alpha}, sigma={sigma}, c={c}, seed={seed}")
        print(f"{'='*70}\n")

    # ── 1. Generate synthetic returns ────────────────────────────────────────
    regime_schedule = make_regime_schedule(total_days, regime_windows, regime_lengths)
    burnin = max(windows)
    returns_df, _ = generate_synthetic_returns(
        n_assets=n_assets,
        total_days=total_days,
        regime_schedule=regime_schedule,
        mu=mu, alpha=alpha, sigma=sigma,
        burnin=burnin, random_seed=seed,
    )
    returns = returns_df.values  # shape (T, n_assets)
    T, n = returns.shape
    n_experts = len(windows)
    regime_break = regime_lengths[0]  # day index where regime switches

    if verbose:
        print(f"Generated returns: shape={returns.shape}")
        print(f"  Return stats: mean={returns.mean():.5f}, std={returns.std():.5f}")

    # ── 2. Compute expert portfolios ─────────────────────────────────────────
    # Build full history: burnin_returns prepended to trading returns.
    # Burnin returns are i.i.d. N(mu, sigma^2) — implementation choice (not
    # specified by paper). This gives all experts enough history on day 0.
    # (paper: at least max(N)=burnin prior observations required)
    if verbose:
        print("Computing expert portfolios...")
    from src.expert import estimate_rolling_stats, solve_expert_portfolio

    rng_burnin = np.random.default_rng(seed + 1)
    mu_vec = np.full(n_assets, mu)
    burnin_returns = rng_burnin.normal(loc=mu_vec, scale=sigma, size=(burnin, n_assets))
    full_history = np.vstack([burnin_returns, returns])  # shape (burnin+T, n_assets)

    expert_set = ExpertSet(
        windows=windows, n_assets=n, c=c, regularization_lambda=COV_REGULARIZATION_LAMBDA
    )
    expert_weights_all = np.zeros((T, n_experts, n))
    expert_w_prev_all = np.zeros((T, n_experts, n))

    for t in range(T):
        t_abs = burnin + t          # absolute index in full_history
        for idx, j in enumerate(windows):
            expert_w_prev_all[t, idx] = expert_set.w[j].copy()
        for j in windows:
            mu_hat, Sigma_hat = estimate_rolling_stats(
                full_history, j, t_abs, COV_REGULARIZATION_LAMBDA
            )
            w_new = solve_expert_portfolio(mu_hat, Sigma_hat, expert_set.w[j], c=c)
            expert_set.w[j] = w_new
        for idx, j in enumerate(windows):
            expert_weights_all[t, idx] = expert_set.w[j].copy()
        if verbose and (t + 1) % 500 == 0:
            print(f"  Expert step {t+1}/{T}")

    if verbose:
        print("Expert portfolios computed.")

    # ── 3. Run Hedge ──────────────────────────────────────────────────────────
    T_horizon = T
    if verbose:
        print("Running Hedge aggregation...")
    hedge_results = run_aggregation(
        returns=returns,
        expert_weights_all=expert_weights_all,
        aggregator_type="hedge",
        n_experts=n_experts,
        horizon=T_horizon,
        c=c,
    )

    # ── 4. Run Fixed Share ────────────────────────────────────────────────────
    if verbose:
        print("Running Fixed Share aggregation...")
    K_fs = compute_switching_budget(T_horizon, FIXED_SHARE_EPSILON)
    fs_results = run_aggregation(
        returns=returns,
        expert_weights_all=expert_weights_all,
        aggregator_type="fixed_share",
        n_experts=n_experts,
        horizon=T_horizon,
        c=c,
        K=K_fs,
    )

    if verbose:
        print(f"  Switching budget K = {K_fs}")
        print(f"  Hedge eta = {compute_hedge_eta(n_experts, T_horizon):.5f}")
        alpha_fs = compute_fixed_share_alpha(n_experts, T_horizon, K_fs, FIXED_SHARE_EPSILON)
        print(f"  Fixed Share alpha = {alpha_fs:.5f}")

    # ── 5. Compute expert aggregate losses for regret ─────────────────────────
    # Aggregate loss per step: -w_hat^T y + c ||w_hat - w_hat_prev||_1
    def _agg_losses(res):
        return -res["agg_pnl"] + c * res["agg_turnover"]

    hedge_agg_loss = _agg_losses(hedge_results)
    fs_agg_loss = _agg_losses(fs_results)

    # ── 6. Compute cumulative regret (vs best fixed expert) ───────────────────
    hedge_regret = compute_cumulative_regret(hedge_agg_loss, hedge_results["expert_losses"])
    fs_regret = compute_cumulative_regret(fs_agg_loss, fs_results["expert_losses"])

    # ── 7. Tracking regret with K switches ────────────────────────────────────
    K_track = K_fs
    if T <= 500:  # DP can be slow for large T; skip for very large horizons
        hedge_track_regret, _ = compute_tracking_regret_dp(
            hedge_results["expert_losses"], hedge_agg_loss, K_track
        )
        fs_track_regret, _ = compute_tracking_regret_dp(
            fs_results["expert_losses"], fs_agg_loss, K_track
        )
    else:
        # Use simple regret for large T
        hedge_track_regret = hedge_regret
        fs_track_regret = fs_regret

    # ── 8. Theoretical bounds ─────────────────────────────────────────────────
    hedge_bound = compute_hedge_regret_bound(n_experts, T_horizon)
    fs_bound = compute_fixed_share_regret_bound(n_experts, T_horizon, K_track)

    # ── 9. Performance metrics ────────────────────────────────────────────────
    hedge_perf = summarise_performance(
        hedge_results["account_value"], hedge_results["agg_net_pnl"], "Hedge"
    )
    fs_perf = summarise_performance(
        fs_results["account_value"], fs_results["agg_net_pnl"], "Fixed Share"
    )

    if verbose:
        print("\nPerformance Summary:")
        for perf in [hedge_perf, fs_perf]:
            print(
                f"  {perf['label']:12s}: CumRet={perf['cumulative_return']:.4f}, "
                f"MaxDD={perf['max_drawdown']:.4f}, "
                f"Sharpe={perf['sharpe']:.3f}"
            )

    # ── 10. Plot expert weights evolution ────────────────────────────────────
    dates = returns_df.index
    _plot_expert_weights(
        dates, hedge_results["expert_q"], fs_results["expert_q"],
        windows, regime_break, tag=tag, save_dir=RESULTS_DIR,
    )

    # ── 11. Plot cumulative regret ────────────────────────────────────────────
    _plot_cumulative_regret(
        dates, hedge_regret, fs_regret, hedge_bound, fs_bound,
        regime_break, tag=tag, save_dir=RESULTS_DIR,
    )

    # ── 12. Plot account value ────────────────────────────────────────────────
    _plot_account_value(
        dates, hedge_results["account_value"][1:], fs_results["account_value"][1:],
        tag=tag, save_dir=RESULTS_DIR,
    )

    results = {
        "tag": tag,
        "n_assets": n_assets, "T": T, "windows": list(windows),
        "regime_break": regime_break,
        "K": K_track,
        "hedge_cumulative_regret_final": float(hedge_regret[-1]),
        "fs_cumulative_regret_final": float(fs_regret[-1]),
        "hedge_tracking_regret_final": float(hedge_track_regret[-1]),
        "fs_tracking_regret_final": float(fs_track_regret[-1]),
        "hedge_performance": hedge_perf,
        "fs_performance": fs_perf,
        "hedge_final_expert_q": hedge_results["expert_q"][-1].tolist(),
        "fs_final_expert_q": fs_results["expert_q"][-1].tolist(),
        "hedge_post_regime_expert_q": hedge_results["expert_q"][regime_break].tolist(),
        "fs_post_regime_expert_q": fs_results["expert_q"][regime_break].tolist(),
        "assumptions": {
            "mu": mu, "alpha": alpha, "sigma": sigma, "seed": seed,
            "cov_regularization_lambda": COV_REGULARIZATION_LAMBDA,
            "w_0": "uniform (1/n)",
        },
    }

    # Save metrics
    out_path = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nMetrics saved to {out_path}")

    return results


def _plot_expert_weights(
    dates, hedge_q, fs_q, windows, regime_break, tag, save_dir
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    labels = [f"N={w}" for w in windows]
    for ax, q, alg in zip(axes, [hedge_q, fs_q], ["Hedge", "Fixed Share"]):
        for idx, lbl in enumerate(labels):
            ax.plot(dates, q[:, idx], label=lbl)
        ax.axvline(dates[regime_break] if regime_break < len(dates) else dates[-1],
                   color="red", linestyle="--", alpha=0.7, label="Regime shift")
        ax.set_ylabel("Expert weight $q_j(t)$")
        ax.set_title(f"{alg}: Expert Aggregation Weights")
        ax.legend(loc="upper right", ncol=len(windows))
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_expert_weights.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_cumulative_regret(
    dates, hedge_reg, fs_reg, hedge_bound, fs_bound, regime_break, tag, save_dir
):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, hedge_reg, label="Hedge (realized)", color="blue")
    ax.plot(dates, fs_reg, label="Fixed Share (realized)", color="orange")
    ax.plot(dates, hedge_bound, label="Hedge bound", color="blue", linestyle="--", alpha=0.5)
    ax.plot(dates, fs_bound, label="Fixed Share bound", color="orange", linestyle="--", alpha=0.5)
    ax.axvline(dates[regime_break] if regime_break < len(dates) else dates[-1],
               color="red", linestyle=":", alpha=0.7, label="Regime shift")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Cost-Sensitive Regret")
    ax.set_title(f"Cumulative Tracking Regret ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_regret.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_account_value(dates, hedge_av, fs_av, tag, save_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, hedge_av, label="Hedge", color="blue")
    ax.plot(dates, fs_av, label="Fixed Share", color="orange")
    ax.set_xlabel("Date")
    ax.set_ylabel("Account Value")
    ax.set_title(f"Account Value Trajectory ({tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_account_value.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    results = run_exp1()
    print("\nExp1 completed successfully.")
    print(f"Hedge cumulative regret (final): {results['hedge_cumulative_regret_final']:.4f}")
    print(f"FS    cumulative regret (final): {results['fs_cumulative_regret_final']:.4f}")
