"""
Metrics for evaluating cost-sensitive online window-size aggregation.

Paper references:
  - Problem 2.4: Cost-sensitive tracking regret R_T(K)
  - Section 3.1: Hedge regret bound
  - Appendix B.1: Account-value evaluation
  - Appendix B.3: Stock-sequence tracking regret R_stock
"""

from __future__ import annotations

import numpy as np
from itertools import combinations


# ─── Cost-Sensitive Tracking Regret ──────────────────────────────────────────

def compute_cumulative_regret(
    agg_losses: np.ndarray,
    expert_losses: np.ndarray,
) -> np.ndarray:
    """
    Compute cumulative cost-sensitive regret vs best fixed expert.

    Regret(T) = sum_{t=1}^{T} ell_agg(t) - min_j sum_{t=1}^{T} ell_j(t)

    Parameters
    ----------
    agg_losses : np.ndarray, shape (T,)
        Aggregate portfolio losses over time.
    expert_losses : np.ndarray, shape (T, n_experts)
        Per-expert losses over time.

    Returns
    -------
    cumulative_regret : np.ndarray, shape (T,)
        Cumulative regret up to each date.
    """
    T = len(agg_losses)
    cum_agg = np.cumsum(agg_losses)
    cum_expert = np.cumsum(expert_losses, axis=0)   # (T, n_experts)
    best_expert_cum = cum_expert.min(axis=1)          # (T,)
    return cum_agg - best_expert_cum


def compute_tracking_regret_dp(
    expert_losses: np.ndarray,
    agg_losses: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cost-sensitive tracking regret against best K-switch expert sequence.

    Uses dynamic programming to solve:
      min over sigma in Sigma_{T,K}: sum_{t=1}^T ell_{sigma(t)}(t)

    where Sigma_{T,K} = {sequences of expert indices with at most K switches}.

    Paper reference: Problem 2.4.

    Parameters
    ----------
    expert_losses : np.ndarray, shape (T, n_experts)
    agg_losses : np.ndarray, shape (T,)
    K : int
        Maximum number of switches (comparator budget).

    Returns
    -------
    tracking_regret : np.ndarray, shape (T,)
        Cumulative tracking regret R_T(K) up to each date t.
    best_seq_cumcost : np.ndarray, shape (T,)
        Cumulative cost of the best K-switch expert sequence.
    """
    T, n_experts = expert_losses.shape
    # dp[t, k] = minimum cumulative loss of best sequence up to t with at most k switches
    # dp[0, k] = 0 for all k
    # Transition: dp[t, k] = min over j of (dp[t-1, k] + expert_losses[t-1,j])
    #   or (dp[t-1, k-1] + expert_losses[t-1,j]) if we switch expert at t

    # Simplified DP: track (t, switches_used) states
    # For efficiency: dp_cost[k, j] = min cum loss using k switches, currently holding expert j
    INF = 1e18
    K_max = min(K, T - 1)

    # dp[sw, j] = min cumulative loss having made sw switches and currently using expert j
    dp = np.full((K_max + 1, n_experts), INF)
    dp[0] = expert_losses[0]    # at t=0, no switches, pick any expert
    for j in range(n_experts):
        dp[0, j] = expert_losses[0, j]
    # Also allow switching at t=0 (costs 0 switches since starting position)
    # Actually switches = number of times we change expert after initial choice

    best_seq_cumcost = np.zeros(T)
    best_seq_cumcost[0] = dp.min()

    for t in range(1, T):
        dp_new = np.full((K_max + 1, n_experts), INF)
        for sw in range(K_max + 1):
            for j in range(n_experts):
                # Stay with expert j (no switch)
                if dp[sw, j] < INF:
                    dp_new[sw, j] = min(dp_new[sw, j], dp[sw, j] + expert_losses[t, j])
                # Switch from any expert k to j (uses 1 switch)
                if sw > 0:
                    best_prev = dp[sw - 1].min()
                    if best_prev < INF:
                        dp_new[sw, j] = min(dp_new[sw, j], best_prev + expert_losses[t, j])
        dp = dp_new
        best_seq_cumcost[t] = dp.min()

    cum_agg = np.cumsum(agg_losses)
    tracking_regret = cum_agg - best_seq_cumcost
    return tracking_regret, best_seq_cumcost


def compute_hedge_regret_bound(
    n_experts: int,
    horizon: int,
    loss_max: float = 1.0,
) -> np.ndarray:
    """
    Compute Hedge theoretical regret bound.

    Bound: R_T <= sqrt(2 * T * ln(n)) for losses in [0, 1]
    (standard Hedge bound with eta* = sqrt(2 ln n / T)).

    Paper reference: Section 3.1 bound.

    Parameters
    ----------
    n_experts : int
    horizon : int
    loss_max : float
        Maximum single-period loss (assumed: 1.0 for normalised losses).

    Returns
    -------
    bound_at_each_t : np.ndarray, shape (horizon,)
    """
    t_vals = np.arange(1, horizon + 1, dtype=float)
    bound = loss_max * np.sqrt(2.0 * t_vals * np.log(n_experts))
    return bound


def compute_fixed_share_regret_bound(
    n_experts: int,
    horizon: int,
    K: int,
    loss_max: float = 1.0,
) -> np.ndarray:
    """
    Compute Fixed Share theoretical tracking-regret bound.

    Standard Fixed Share bound (Herbster & Warmuth 1998):
      R_T(K) <= sqrt(2 T (K+1) ln n + 2 T K ln(T/(K+1)))
    (approximate; paper cites a similar minimax form).

    Parameters
    ----------
    n_experts : int
    horizon : int
    K : int
    loss_max : float

    Returns
    -------
    bound_at_each_t : np.ndarray, shape (horizon,)
    """
    t_vals = np.arange(1, horizon + 1, dtype=float)
    inner = (K + 1) * np.log(n_experts) + K * np.log(np.maximum(t_vals / (K + 1), 1.0))
    bound = loss_max * np.sqrt(2.0 * t_vals * inner)
    return bound


# ─── Account Value ────────────────────────────────────────────────────────────

def compute_account_value(
    returns: np.ndarray,
    portfolio_weights: np.ndarray,
    c: float = 0.001,
    initial_value: float = 1.0,
) -> np.ndarray:
    """
    Compute account value trajectory.

    V(t+1) = V(t) * (1 + w_hat(t)^T y(t) - c * ||w_hat(t) - w_hat(t-1)||_1)

    Paper reference: Appendix B.1.

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)
    portfolio_weights : np.ndarray, shape (T, n_assets)
    c : float
    initial_value : float

    Returns
    -------
    account_value : np.ndarray, shape (T+1,)
    """
    T = len(returns)
    av = np.zeros(T + 1)
    av[0] = initial_value
    w_prev = portfolio_weights[0]  # assume starts at first weight
    for t in range(T):
        gross = float(portfolio_weights[t] @ returns[t])
        turnover = float(np.sum(np.abs(portfolio_weights[t] - (w_prev if t == 0 else portfolio_weights[t - 1]))))
        net = gross - c * turnover
        av[t + 1] = av[t] * (1.0 + net)
    return av


# ─── Performance Metrics ─────────────────────────────────────────────────────

def cumulative_return(account_value: np.ndarray) -> float:
    """Total return: (V_T / V_0) - 1."""
    return float(account_value[-1] / account_value[0] - 1.0)


def max_drawdown(account_value: np.ndarray) -> float:
    """
    Maximum drawdown of an account-value series.

    MDD = max_{s<=t} (V(s) - V(t)) / V(s)
    """
    peak = np.maximum.accumulate(account_value)
    drawdown = (peak - account_value) / np.where(peak > 0, peak, 1.0)
    return float(drawdown.max())


def annualised_sharpe(net_pnl: np.ndarray, trading_days_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio from daily net portfolio returns.

    Sharpe = mean(r) / std(r) * sqrt(252)

    Parameters
    ----------
    net_pnl : np.ndarray, shape (T,)
        Daily net portfolio returns (after costs).
    trading_days_per_year : int

    Returns
    -------
    sharpe : float
    """
    std = float(net_pnl.std())
    if std < 1e-12:
        return 0.0
    return float(net_pnl.mean() / std * np.sqrt(trading_days_per_year))


def summarise_performance(
    account_value: np.ndarray,
    net_pnl: np.ndarray,
    label: str,
) -> dict:
    """
    Compute summary performance statistics.

    Returns dict with keys: label, cumret, max_drawdown, sharpe.
    """
    return {
        "label": label,
        "cumulative_return": cumulative_return(account_value),
        "max_drawdown": max_drawdown(account_value),
        "sharpe": annualised_sharpe(net_pnl),
        "final_value": float(account_value[-1]),
    }


# ─── Stock-Sequence Tracking Regret ──────────────────────────────────────────

def compute_stock_sequence_regret(
    portfolio_net_pnl: np.ndarray,
    stock_returns: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute stock-sequence tracking regret R_stock against best K-switch stock.

    Loss for buy-and-hold stock j at time t: -log(1 + y_j(t)).
    Aggregate loss: -log(1 + w_hat^T y(t) - c*turnover).

    Uses the same DP as compute_tracking_regret_dp but with log-return loss.

    Paper reference: Appendix B.3.

    Parameters
    ----------
    portfolio_net_pnl : np.ndarray, shape (T,)
        Aggregate net PnL per period.
    stock_returns : np.ndarray, shape (T, n_stocks)
        Individual stock returns.
    K : int
        Switching budget.

    Returns
    -------
    r_stock : np.ndarray, shape (T,)
        Cumulative R_stock up to each date.
    best_seq_cumcost : np.ndarray, shape (T,)
        Cumulative negative log-return of best K-switch stock sequence.
    """
    # Portfolio log-return loss
    eps = 1e-8
    agg_log_losses = -np.log(np.maximum(1.0 + portfolio_net_pnl, eps))
    # Stock log-return losses
    stock_log_losses = -np.log(np.maximum(1.0 + stock_returns, eps))
    tracking_regret, best_cumcost = compute_tracking_regret_dp(
        stock_log_losses, agg_log_losses, K
    )
    return tracking_regret, best_cumcost
