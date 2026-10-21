"""
Experiment 2: Historical DJIA study.

Paper: Section 4.2 / Appendix B.4 "Empirical Studies: Dow Jones 30"
Date range: 2021/03/04 to 2024/02/25
Tickers: 30 DJIA constituents + BIL (risk-free asset)
Windows N = {5, 10, 21, 30, 41, 50, 63, 126}
Transaction cost c = 0.001

Data note: Loaded via QuantX Dataloader REST API. Requires DATALOADER_API_KEY
and MASSIVE_API_KEY environment variables.
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
import logging

from config import (
    DJIA_TICKERS, DJIA_START_DATE, DJIA_END_DATE, WINDOWS_8,
    TRANSACTION_COST_C, MAX_WINDOW, FIXED_SHARE_EPSILON,
    COV_REGULARIZATION_LAMBDA, INITIAL_ACCOUNT_VALUE,
    EG_ETA, PAMR_EPSILON, PAMR_C, CWMR_PHI, CWMR_EPSILON,
    OLMAR_WINDOW, OLMAR_EPSILON,
)
from src.data_loader import load_prices, prices_to_returns
from src.expert import (
    ExpertSet, estimate_rolling_stats, solve_expert_portfolio,
    compute_expert_portfolios_parallel,
)
from src.aggregator import (
    run_aggregation, compute_hedge_eta, compute_switching_budget,
    compute_fixed_share_alpha,
)
from src.metrics import (
    compute_cumulative_regret, compute_tracking_regret_dp,
    compute_fixed_share_regret_bound, compute_hedge_regret_bound,
    compute_stock_sequence_regret, summarise_performance,
    cumulative_return, max_drawdown, annualised_sharpe,
    compute_account_value,
)
from src.baselines import run_all_baselines

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "djia_prices.pkl")


def load_djia_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DJIA adjusted close prices and compute returns.

    Pre-sample lookback of MAX_WINDOW (126) trading days is fetched to
    initialise the largest window expert.

    Returns
    -------
    prices : pd.DataFrame
        Full adjusted close price panel (includes pre-sample).
    returns : pd.DataFrame
        Simple daily returns for the evaluation period only.
    """
    tickers = list(DJIA_TICKERS)
    prices = load_prices(
        tickers=tickers,
        start_date=DJIA_START_DATE,
        end_date=DJIA_END_DATE,
        lookback_days=MAX_WINDOW + 10,  # extra buffer for weekends/holidays
        cache_path=DATA_CACHE if use_cache else None,
    )

    # Validate
    if prices.empty:
        raise RuntimeError("No price data returned for DJIA tickers.")

    missing_tickers = [t for t in tickers if t not in prices.columns]
    if missing_tickers:
        logger.warning("Missing tickers: %s", missing_tickers)

    # Fill minor gaps with forward fill then backward fill
    prices = prices.ffill().bfill()

    # Drop any remaining all-NaN columns
    prices = prices.dropna(axis=1, how="all")

    logger.info("Loaded prices: %s, date range: %s to %s",
                prices.shape, prices.index[0], prices.index[-1])

    # Compute returns
    returns_all = prices_to_returns(prices)

    # Split: pre-sample (first MAX_WINDOW rows) and evaluation
    start_eval = pd.to_datetime(DJIA_START_DATE)
    eval_mask = returns_all.index >= start_eval
    returns_eval = returns_all[eval_mask].copy()

    logger.info("Evaluation returns: %s", returns_eval.shape)
    return prices, returns_eval


def run_exp2(use_cache: bool = True, verbose: bool = True) -> dict:
    """
    Run Experiment 2: DJIA historical study.

    Returns dict of results including regret and performance metrics.
    """
    tag = "exp2_djia"

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXP2: Historical DJIA Study")
        print(f"  Date range: {DJIA_START_DATE} to {DJIA_END_DATE}")
        print(f"  Tickers: {len(DJIA_TICKERS)}, Windows: {WINDOWS_8}")
        print(f"  c={TRANSACTION_COST_C}")
        print(f"{'='*70}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if verbose:
        print("Loading DJIA price data...")
    prices, returns_eval = load_djia_data(use_cache=use_cache)
    returns = returns_eval.values
    T, n = returns.shape
    dates = returns_eval.index
    asset_names = list(returns_eval.columns)
    n_experts = len(WINDOWS_8)
    windows = WINDOWS_8

    if verbose:
        print(f"Returns shape: {returns.shape}, dates: {dates[0]} to {dates[-1]}")

    # Build full history including pre-sample for rolling estimation
    # Pre-sample = rows in returns_all before DJIA_START_DATE
    returns_all = prices_to_returns(prices).fillna(0.0)
    returns_all_arr = returns_all.values

    # Find offset: start of eval period in returns_all
    start_eval = pd.to_datetime(DJIA_START_DATE)
    t_offset = (returns_all.index >= start_eval).argmax()

    if verbose:
        print(f"Pre-sample rows available: {t_offset}")

    # ── 2. Compute expert portfolios (fast parallel OSQP) ─────────────────────
    if verbose:
        print("Computing expert portfolios (fast parallel OSQP, warm-started)...")
    expert_weights_all = compute_expert_portfolios_parallel(
        returns_all=returns_all_arr,
        windows=windows,
        t_offset=t_offset,
        T=T,
        c=TRANSACTION_COST_C,
        reg_lambda=COV_REGULARIZATION_LAMBDA,
        n_jobs=-1,
        verbose=verbose,
    )
    if verbose:
        print(f"  Expert portfolios computed for {T} days.    ")

    # ── 3. Run Hedge and Fixed Share ──────────────────────────────────────────
    K_fs = compute_switching_budget(T, FIXED_SHARE_EPSILON)
    if verbose:
        print(f"\nRunning Hedge (K={K_fs}, eta={compute_hedge_eta(n_experts,T):.5f})...")
    hedge_results = run_aggregation(
        returns=returns,
        expert_weights_all=expert_weights_all,
        aggregator_type="hedge",
        n_experts=n_experts,
        horizon=T,
        c=TRANSACTION_COST_C,
    )
    if verbose:
        print(f"Running Fixed Share (alpha={compute_fixed_share_alpha(n_experts,T,K_fs,FIXED_SHARE_EPSILON):.5f})...")
    fs_results = run_aggregation(
        returns=returns,
        expert_weights_all=expert_weights_all,
        aggregator_type="fixed_share",
        n_experts=n_experts,
        horizon=T,
        c=TRANSACTION_COST_C,
        K=K_fs,
    )

    # ── 4. Run baselines ──────────────────────────────────────────────────────
    if verbose:
        print("Running baseline algorithms (UP, EG, PAMR, CWMR, OLMAR)...")
    baseline_results = run_all_baselines(
        returns=returns,
        c=TRANSACTION_COST_C,
        eg_eta=EG_ETA,
        pamr_epsilon=PAMR_EPSILON,
        pamr_C=PAMR_C,
        cwmr_phi=CWMR_PHI,
        cwmr_epsilon=CWMR_EPSILON,
        olmar_m=OLMAR_WINDOW,
        olmar_epsilon=OLMAR_EPSILON,
    )

    # ── 5. Compute regret ─────────────────────────────────────────────────────
    def _agg_loss(res):
        return -res["agg_pnl"] + TRANSACTION_COST_C * res["agg_turnover"]

    hedge_agg_loss = _agg_loss(hedge_results)
    fs_agg_loss = _agg_loss(fs_results)
    hedge_regret = compute_cumulative_regret(hedge_agg_loss, hedge_results["expert_losses"])
    fs_regret = compute_cumulative_regret(fs_agg_loss, fs_results["expert_losses"])

    # ── 6. Stock-sequence tracking regret R_stock ─────────────────────────────
    K_stock = K_fs
    # Use subset of stocks (exclude BIL for stock-sequence comparison if desired)
    if verbose:
        print(f"Computing stock-sequence regret (K={K_stock})...")
    # For efficiency limit DP to manageable dimension
    if n <= 35 and T <= 800:
        hedge_r_stock, _ = compute_stock_sequence_regret(
            hedge_results["agg_net_pnl"], returns, K_stock
        )
        fs_r_stock, _ = compute_stock_sequence_regret(
            fs_results["agg_net_pnl"], returns, K_stock
        )
    else:
        hedge_r_stock = np.zeros(T)
        fs_r_stock = np.zeros(T)
        if verbose:
            print("  Skipping stock-sequence DP (too large); reporting zeros.")

    # ── 7. Compute performance metrics ────────────────────────────────────────
    hedge_perf = summarise_performance(
        hedge_results["account_value"], hedge_results["agg_net_pnl"], "Hedge"
    )
    fs_perf = summarise_performance(
        fs_results["account_value"], fs_results["agg_net_pnl"], "Fixed Share"
    )
    baseline_perf = {
        name: summarise_performance(res["account_value"], res["net_pnl"], name)
        for name, res in baseline_results.items()
    }

    if verbose:
        print("\nPerformance Summary:")
        for perf in [hedge_perf, fs_perf] + list(baseline_perf.values()):
            print(
                f"  {perf['label']:12s}: CumRet={perf['cumulative_return']:.4f}, "
                f"MaxDD={perf['max_drawdown']:.4f}, "
                f"Sharpe={perf['sharpe']:.3f}"
            )

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    _plot_expert_weights(dates, hedge_results["expert_q"], fs_results["expert_q"],
                         windows, tag=tag, save_dir=RESULTS_DIR)
    _plot_cumulative_regret(dates, hedge_regret, fs_regret,
                            compute_hedge_regret_bound(n_experts, T),
                            compute_fixed_share_regret_bound(n_experts, T, K_fs),
                            tag=tag, save_dir=RESULTS_DIR)
    _plot_account_value(dates, hedge_results, fs_results, baseline_results, tag=tag, save_dir=RESULTS_DIR)

    # ── 9. Save results ───────────────────────────────────────────────────────
    results = {
        "tag": tag,
        "n_assets": n, "T": T, "windows": list(windows),
        "K": K_stock,
        "hedge_cumulative_regret_final": float(hedge_regret[-1]),
        "fs_cumulative_regret_final": float(fs_regret[-1]),
        "hedge_r_stock_final": float(hedge_r_stock[-1]),
        "fs_r_stock_final": float(fs_r_stock[-1]),
        "hedge_performance": hedge_perf,
        "fs_performance": fs_perf,
        "baseline_performance": {k: v for k, v in baseline_perf.items()},
        "hedge_final_expert_q": hedge_results["expert_q"][-1].tolist(),
        "fs_final_expert_q": fs_results["expert_q"][-1].tolist(),
    }
    out_path = os.path.join(RESULTS_DIR, f"{tag}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    if verbose:
        print(f"\nMetrics saved to {out_path}")
    return results


def _plot_expert_weights(dates, hedge_q, fs_q, windows, tag, save_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    labels = [f"N={w}" for w in windows]
    for ax, q, alg in zip(axes, [hedge_q, fs_q], ["Hedge", "Fixed Share"]):
        for idx, lbl in enumerate(labels):
            ax.plot(dates, q[:, idx], label=lbl, linewidth=0.8)
        ax.set_ylabel("Expert weight $q_j(t)$")
        ax.set_title(f"{alg}: Expert Aggregation Weights (DJIA)")
        ax.legend(loc="upper right", ncol=4, fontsize=8)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_expert_weights.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_cumulative_regret(dates, hedge_reg, fs_reg, hedge_bound, fs_bound, tag, save_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, hedge_reg, label="Hedge", color="blue")
    ax.plot(dates, fs_reg, label="Fixed Share", color="orange")
    ax.plot(dates, hedge_bound, "--", color="blue", alpha=0.4, label="Hedge bound")
    ax.plot(dates, fs_bound, "--", color="orange", alpha=0.4, label="FS bound")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Regret")
    ax.set_title(f"Cumulative Cost-Sensitive Regret (DJIA, {tag})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_regret.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def _plot_account_value(dates, hedge_res, fs_res, baseline_res, tag, save_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, hedge_res["account_value"][1:], label="Hedge", linewidth=1.5)
    ax.plot(dates, fs_res["account_value"][1:], label="Fixed Share", linewidth=1.5)
    colors = ["green", "purple", "brown", "gray", "pink"]
    for (name, res), col in zip(baseline_res.items(), colors):
        ax.plot(dates, res["account_value"][1:], label=name, linewidth=0.8, color=col)
    ax.set_xlabel("Date"); ax.set_ylabel("Account Value")
    ax.set_title(f"Account Value Comparison (DJIA, {tag})")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_account_value.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    results = run_exp2()
    print("\nExp2 completed.")
