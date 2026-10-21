"""
Experiment 3: Historical S&P 500 study.

Paper: Appendix B.5 "Empirical Studies: S&P 500"
Date range: 2020/01/02 to 2024/08/01 (paper states 2026/08/01; use available data)
Universe: 481 S&P 500 components (fixed, no constituent changes); + BIL
Windows N = {5, 10, 21, 30, 41, 50, 63, 126}
Transaction cost c = 0.001

NOTE: The paper states 481 components over 2020/01/02 to 2026/08/01. Since
live data may not extend to 2026, we use the available range up to 2024-08-01.
The S&P 500 ticker list is loaded from data/sp500_tickers.txt. Tickers that
were continuously listed (no constituency changes) during the sample are retained.

Uncertainty (paper confirmed): exact ticker list, data vendor, and K not stated.
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
    SP500_START_DATE, SP500_END_DATE, WINDOWS_8,
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
    compute_cumulative_regret,
    compute_fixed_share_regret_bound, compute_hedge_regret_bound,
    summarise_performance,
)
from src.baselines import run_all_baselines

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SP500_TICKERS_FILE = os.path.join(DATA_DIR, "sp500_tickers.txt")
DATA_CACHE = os.path.join(DATA_DIR, "sp500_prices.pkl")


def load_sp500_tickers() -> list[str]:
    """
    Load S&P 500 ticker list.

    The paper uses 481 components that were continuously listed during
    2020/01/02 to the end of the study. BIL is included as risk-free.
    Tickers are read from data/sp500_tickers.txt.
    """
    if os.path.exists(SP500_TICKERS_FILE):
        with open(SP500_TICKERS_FILE) as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers

    # Fallback: use a representative subset of S&P 500 components
    # Note: this is an assumption — the paper does not supply the exact list
    # in machine-readable form. We use a curated subset known to be
    # continuously listed throughout 2020-2024.
    logger.warning(
        "sp500_tickers.txt not found. Using a representative 50-stock subset. "
        "This deviates from the paper's 481-stock universe."
    )
    subset = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "JNJ",
        "V", "PG", "UNH", "HD", "MA", "DIS", "BAC", "ADBE", "NFLX", "CMCSA",
        "XOM", "VZ", "T", "INTC", "KO", "PFE", "MRK", "PEP", "ABBV", "CSCO",
        "CVX", "WMT", "ABT", "TMO", "ACN", "MCD", "NKE", "NEE", "TXN", "LIN",
        "MDT", "AMGN", "HON", "PM", "UPS", "LOW", "IBM", "GS", "AXP", "CAT",
        "BIL",
    ]
    return subset


def run_exp3(
    use_cache: bool = True,
    max_tickers: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment 3: S&P 500 historical study.

    Parameters
    ----------
    use_cache : bool
        Whether to use cached price data.
    max_tickers : int or None
        Limit number of tickers for faster testing (None = use all).
    verbose : bool

    Returns
    -------
    results : dict of metrics.
    """
    tag = "exp3_sp500"

    tickers = load_sp500_tickers()
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
        if "BIL" not in tickers:
            tickers.append("BIL")
        tag = f"exp3_sp500_n{len(tickers)}"

    if verbose:
        print(f"\n{'='*70}")
        print(f"EXP3: Historical S&P 500 Study ({tag})")
        print(f"  Date range: {SP500_START_DATE} to {SP500_END_DATE}")
        print(f"  Tickers: {len(tickers)}, Windows: {WINDOWS_8}")
        print(f"  c={TRANSACTION_COST_C}")
        print(f"{'='*70}\n")

    # ── 1. Load price data ────────────────────────────────────────────────────
    cache_path = DATA_CACHE if (use_cache and max_tickers is None) else None
    if verbose:
        print("Loading S&P 500 price data (may take time for large universes)...")
    try:
        prices = load_prices(
            tickers=tickers,
            start_date=SP500_START_DATE,
            end_date=SP500_END_DATE,
            lookback_days=MAX_WINDOW + 20,
            cache_path=cache_path,
        )
    except Exception as e:
        logger.error("Failed to load S&P 500 data: %s", e)
        print(f"ERROR: {e}")
        return {"tag": tag, "error": str(e)}

    if prices.empty:
        logger.error("No price data loaded for S&P 500")
        return {"tag": tag, "error": "Empty price data"}

    prices = prices.ffill().bfill()
    prices = prices.dropna(axis=1, how="all")

    # Remove columns that have any remaining NaN in eval period
    eval_start = pd.to_datetime(SP500_START_DATE)
    eval_mask = prices.index >= eval_start
    prices_eval = prices[eval_mask]
    valid_cols = prices_eval.columns[prices_eval.isna().sum() == 0]
    prices = prices[valid_cols]

    if verbose:
        print(f"Loaded {len(prices.columns)} tickers, {len(prices)} days total")

    returns_all = prices_to_returns(prices).fillna(0.0)
    t_offset = (returns_all.index >= eval_start).argmax()
    returns_eval = returns_all[eval_start:].copy()
    returns = returns_eval.values
    T, n = returns.shape
    dates = returns_eval.index
    asset_names = list(returns_eval.columns)
    n_experts = len(WINDOWS_8)
    windows = WINDOWS_8
    returns_all_arr = returns_all.values

    if verbose:
        print(f"Evaluation: T={T} days, n={n} assets")

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
        print(f"  Expert portfolios computed.      ")

    # ── 3. Hedge and Fixed Share ──────────────────────────────────────────────
    K_fs = compute_switching_budget(T, FIXED_SHARE_EPSILON)
    if verbose:
        print(f"\nRunning Hedge and Fixed Share (K={K_fs})...")
    hedge_results = run_aggregation(
        returns=returns,
        expert_weights_all=expert_weights_all,
        aggregator_type="hedge",
        n_experts=n_experts,
        horizon=T,
        c=TRANSACTION_COST_C,
    )
    fs_results = run_aggregation(
        returns=returns,
        expert_weights_all=expert_weights_all,
        aggregator_type="fixed_share",
        n_experts=n_experts,
        horizon=T,
        c=TRANSACTION_COST_C,
        K=K_fs,
    )

    # ── 4. Baselines ──────────────────────────────────────────────────────────
    if verbose:
        print("Running baselines (UP, EG, PAMR, CWMR, OLMAR)...")
    baseline_results = run_all_baselines(
        returns=returns, c=TRANSACTION_COST_C,
        eg_eta=EG_ETA, pamr_epsilon=PAMR_EPSILON, pamr_C=PAMR_C,
        cwmr_phi=CWMR_PHI, cwmr_epsilon=CWMR_EPSILON,
        olmar_m=OLMAR_WINDOW, olmar_epsilon=OLMAR_EPSILON,
    )

    # ── 5. Regret ─────────────────────────────────────────────────────────────
    def _agg_loss(res):
        return -res["agg_pnl"] + TRANSACTION_COST_C * res["agg_turnover"]

    hedge_agg_loss = _agg_loss(hedge_results)
    fs_agg_loss = _agg_loss(fs_results)
    hedge_regret = compute_cumulative_regret(hedge_agg_loss, hedge_results["expert_losses"])
    fs_regret = compute_cumulative_regret(fs_agg_loss, fs_results["expert_losses"])

    # ── 6. Performance metrics ────────────────────────────────────────────────
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

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    _plot_expert_weights(dates, hedge_results["expert_q"], fs_results["expert_q"],
                         windows, tag=tag, save_dir=RESULTS_DIR)
    _plot_cumulative_regret(dates, hedge_regret, fs_regret,
                            compute_hedge_regret_bound(n_experts, T),
                            compute_fixed_share_regret_bound(n_experts, T, K_fs),
                            tag=tag, save_dir=RESULTS_DIR)
    _plot_account_value(dates, hedge_results, fs_results, baseline_results,
                        tag=tag, save_dir=RESULTS_DIR)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    results = {
        "tag": tag,
        "n_assets": n, "T": T, "windows": list(windows), "K": K_fs,
        "hedge_cumulative_regret_final": float(hedge_regret[-1]),
        "fs_cumulative_regret_final": float(fs_regret[-1]),
        "hedge_performance": hedge_perf,
        "fs_performance": fs_perf,
        "baseline_performance": {k: v for k, v in baseline_perf.items()},
        "hedge_final_expert_q": hedge_results["expert_q"][-1].tolist(),
        "fs_final_expert_q": fs_results["expert_q"][-1].tolist(),
        "n_tickers_loaded": n,
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
        ax.set_title(f"{alg}: Expert Aggregation Weights (S&P 500)")
        ax.legend(loc="upper right", ncol=4, fontsize=8)
        ax.set_ylim([0, 1]); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_expert_weights.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"Saved: {path}")


def _plot_cumulative_regret(dates, hedge_reg, fs_reg, hedge_bound, fs_bound, tag, save_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, hedge_reg, label="Hedge"); ax.plot(dates, fs_reg, label="Fixed Share")
    ax.plot(dates, hedge_bound, "--", alpha=0.4, label="Hedge bound")
    ax.plot(dates, fs_bound, "--", alpha=0.4, label="FS bound")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Regret")
    ax.set_title(f"Cumulative Cost-Sensitive Regret (S&P 500, {tag})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_regret.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"Saved: {path}")


def _plot_account_value(dates, hedge_res, fs_res, baseline_res, tag, save_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, hedge_res["account_value"][1:], label="Hedge", linewidth=1.5)
    ax.plot(dates, fs_res["account_value"][1:], label="Fixed Share", linewidth=1.5)
    for name, res in baseline_res.items():
        ax.plot(dates, res["account_value"][1:], label=name, linewidth=0.8)
    ax.set_xlabel("Date"); ax.set_ylabel("Account Value")
    ax.set_title(f"Account Value Comparison (S&P 500, {tag})")
    ax.legend(loc="upper left", ncol=3, fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_account_value.png")
    plt.savefig(path, dpi=150); plt.close(); print(f"Saved: {path}")


if __name__ == "__main__":
    results = run_exp3()
    print("\nExp3 completed.")
