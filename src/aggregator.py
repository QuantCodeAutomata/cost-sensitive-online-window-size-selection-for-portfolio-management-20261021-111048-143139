"""
Online aggregation algorithms: Hedge and Fixed Share.

Paper references:
  - Section 3.1: Hedge (exponential weights)
  - Section 3.2: Fixed Share for window selection
  - Expert losses: ell(f_j(t), y(t)) = -f_j(t)^T y(t) + c ||f_j(t) - f_j(t-1)||_1
    (Section 4.1, cost-sensitive negative PnL loss)
  - Update rule (Hedge): q_j(t+1) = q_j(t) * exp(-eta * loss_j(t)) / Z
  - Fixed Share: q_j^exp -> (1-alpha)*q_j^exp + alpha/n (share step)
  - Learning rate optimisation: eta* = sqrt(8 ln(n) / T) for bounded losses in [-1,1]
  - Fixed Share mixing: alpha* from switching-budget formula
"""

from __future__ import annotations

import math
import numpy as np


def compute_expert_loss(
    w_j: np.ndarray,
    w_j_prev: np.ndarray,
    y: np.ndarray,
    c: float = 0.001,
) -> float:
    """
    Cost-sensitive expert loss (paper Section 4.1 / Problem 2.4).

    ell(f_j(t), y(t)) = -f_j(t)^T y(t) + c * ||f_j(t) - f_j(t-1)||_1

    Parameters
    ----------
    w_j : np.ndarray, shape (n,)
        Current expert portfolio recommendation f_j(t).
    w_j_prev : np.ndarray, shape (n,)
        Previous expert portfolio recommendation f_j(t-1).
    y : np.ndarray, shape (n,)
        Realised returns y(t).
    c : float
        Transaction cost rate.

    Returns
    -------
    loss : float
        Expert cost-sensitive loss value.
    """
    pnl = float(w_j @ y)
    turnover = float(np.sum(np.abs(w_j - w_j_prev)))
    return -pnl + c * turnover


def compute_hedge_eta(n_experts: int, horizon: int) -> float:
    """
    Compute Hedge learning rate from minimax optimisation.

    eta* = sqrt(8 * ln(n) / T)
    Valid for bounded losses in [-1, 1] (Freund & Schapire 1997).
    Paper's optimisation: losses are turnover-inclusive negative PnL.
    Assumed: losses normalised to [-1,1] range.

    Parameters
    ----------
    n_experts : int
        Number of experts.
    horizon : int
        Total number of trading days T.

    Returns
    -------
    eta : float
    """
    if horizon <= 0:
        return 0.01
    return math.sqrt(8.0 * math.log(n_experts) / horizon)


def compute_fixed_share_alpha(
    n_experts: int,
    horizon: int,
    K: int,
    epsilon: float = 0.5,
) -> float:
    """
    Compute Fixed Share mixing parameter from switching budget.

    Paper (Section 3.2): alpha* is derived from the minimax regret
    over comparators switching at most K times in T rounds.
    K(T) = floor(T / (log(T) * (log(log(T)))^{1+eps})) — paper's asymptotic formula.

    Practical formula: alpha = K / (K + T - 1) is a standard result
    (Herbster & Warmuth 1998) aligned with the paper's framework.
    Assumed: epsilon not critical for small horizons; K computed from paper's formula.

    Parameters
    ----------
    n_experts : int
        Number of experts.
    horizon : int
        Total number of trading days T.
    K : int
        Switching budget.
    epsilon : float
        Exponent offset > 0 in K(T) formula (assumed: 0.5).

    Returns
    -------
    alpha : float
    """
    if K <= 0 or horizon <= 0:
        return 1e-4  # essentially no mixing
    # Standard Herbster-Warmuth optimal: alpha = K/(K+T)
    alpha = K / (K + horizon)
    # Also compute eta for Fixed Share (same as Hedge but scaled by switching)
    return float(np.clip(alpha, 1e-6, 0.5))


def compute_switching_budget(horizon: int, epsilon: float = 0.5) -> int:
    """
    Compute K(T) = floor(T / (log(T) * (log(log(T)))^{1+epsilon})).

    Paper reference: Section 3.2, K(T) formula for large T.
    epsilon > 0 is unspecified by paper; assumed = 0.5.

    Parameters
    ----------
    horizon : int
    epsilon : float

    Returns
    -------
    K : int
    """
    if horizon <= 2:
        return 1
    log_T = math.log(horizon)
    log_log_T = math.log(max(log_T, 1.0))
    K = int(horizon / (log_T * (log_log_T ** (1.0 + epsilon))))
    return max(K, 1)


class HedgeAggregator:
    """
    Hedge (Exponential Weights) online aggregation.

    Paper Section 3.1:
      q_j(t+1) = q_j(t) * exp(-eta * ell_j(t)) / Z
    where Z is the normalisation constant.

    Tracks cumulative weights and computes aggregate portfolio
    w_hat(t) = sum_j q_j(t) * f_j(t).
    """

    def __init__(self, n_experts: int, horizon: int, eta: float | None = None) -> None:
        """
        Parameters
        ----------
        n_experts : int
            Number of experts.
        horizon : int
            Total horizon T (used for eta calculation if eta is None).
        eta : float or None
            Learning rate. If None, uses eta* = sqrt(8 ln(n) / T).
        """
        self.n_experts = n_experts
        self.horizon = horizon
        self.eta = eta if eta is not None else compute_hedge_eta(n_experts, horizon)
        # Initialise uniform weights q_j(1) = 1/n (paper: uniform initialisation)
        self.q = np.full(n_experts, 1.0 / n_experts)

    def aggregate(
        self, expert_portfolios: np.ndarray
    ) -> np.ndarray:
        """
        Form aggregate portfolio w_hat(t) = sum_j q_j(t) * f_j(t).

        Parameters
        ----------
        expert_portfolios : np.ndarray, shape (n_experts, n_assets)
            Expert portfolio recommendations at time t.

        Returns
        -------
        w_hat : np.ndarray, shape (n_assets,)
        """
        return self.q @ expert_portfolios  # shape (n_assets,)

    def update(self, losses: np.ndarray) -> None:
        """
        Update expert weights using Hedge rule.

        q_j(t+1) = q_j(t) * exp(-eta * loss_j(t)) / Z

        Parameters
        ----------
        losses : np.ndarray, shape (n_experts,)
            Expert losses at time t.
        """
        self.q = self.q * np.exp(-self.eta * losses)
        self.q /= self.q.sum()

    @property
    def weights(self) -> np.ndarray:
        return self.q.copy()


class FixedShareAggregator:
    """
    Fixed Share online aggregation (Herbster & Warmuth 1998).

    Paper Section 3.2:
      After exponential update: q_j^exp(t) = q_j(t) * exp(-eta * ell_j(t)) / Z
      Fixed Share step: q_j(t+1) = (1 - alpha) * q_j^exp(t) + alpha / n

    This allows tracking non-stationary best expert with up to K switches.
    """

    def __init__(
        self,
        n_experts: int,
        horizon: int,
        K: int | None = None,
        eta: float | None = None,
        alpha: float | None = None,
        epsilon: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        n_experts : int
        horizon : int
        K : int or None
            Switching budget. If None, computed from K(T) formula.
        eta : float or None
            Learning rate. If None, computed from minimax formula.
        alpha : float or None
            Mixing parameter. If None, computed from K and T.
        epsilon : float
            Exponent offset for K(T) formula (assumed: 0.5).
        """
        self.n_experts = n_experts
        self.horizon = horizon
        if K is None:
            K = compute_switching_budget(horizon, epsilon)
        self.K = K
        self.eta = eta if eta is not None else compute_hedge_eta(n_experts, horizon)
        self.alpha = alpha if alpha is not None else compute_fixed_share_alpha(
            n_experts, horizon, K, epsilon
        )
        # Initialise uniform weights q_j(1) = 1/n
        self.q = np.full(n_experts, 1.0 / n_experts)

    def aggregate(self, expert_portfolios: np.ndarray) -> np.ndarray:
        """Form aggregate portfolio w_hat(t) = sum_j q_j(t) * f_j(t)."""
        return self.q @ expert_portfolios

    def update(self, losses: np.ndarray) -> None:
        """
        Update weights: exponential step then fixed-share mixing.

        Parameters
        ----------
        losses : np.ndarray, shape (n_experts,)
        """
        # Exponential (Hedge) step
        q_exp = self.q * np.exp(-self.eta * losses)
        q_exp /= q_exp.sum()
        # Fixed Share mixing step
        self.q = (1.0 - self.alpha) * q_exp + self.alpha / self.n_experts

    @property
    def weights(self) -> np.ndarray:
        return self.q.copy()


def run_aggregation(
    returns: np.ndarray,
    expert_weights_all: np.ndarray,
    aggregator_type: str,
    n_experts: int,
    horizon: int,
    c: float = 0.001,
    eta: float | None = None,
    alpha: float | None = None,
    K: int | None = None,
    epsilon: float = 0.5,
) -> dict:
    """
    Run online aggregation (Hedge or Fixed Share) over all trading days.

    Parameters
    ----------
    returns : np.ndarray, shape (T, n_assets)
        Realised returns for all trading days.
    expert_weights_all : np.ndarray, shape (T, n_experts, n_assets)
        Expert portfolio recommendations for all days.
    aggregator_type : str
        'hedge' or 'fixed_share'.
    n_experts : int
    horizon : int
    c : float
        Transaction cost rate.
    eta : float or None
    alpha : float or None
    K : int or None
    epsilon : float

    Returns
    -------
    results : dict with keys:
        'agg_weights': shape (T, n_assets) aggregate portfolio
        'expert_q': shape (T, n_experts) aggregation weights over time
        'expert_losses': shape (T, n_experts) per-expert losses
        'agg_pnl': shape (T,) aggregate gross PnL
        'agg_turnover': shape (T,) aggregate turnover
        'agg_net_pnl': shape (T,) aggregate net PnL after costs
        'account_value': shape (T+1,) account value trajectory
    """
    T, n_assets = returns.shape
    assert expert_weights_all.shape == (T, n_experts, n_assets)

    if aggregator_type == "hedge":
        agg = HedgeAggregator(n_experts, horizon, eta=eta)
    elif aggregator_type == "fixed_share":
        agg = FixedShareAggregator(n_experts, horizon, K=K, eta=eta, alpha=alpha, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")

    agg_weights = np.zeros((T, n_assets))
    expert_q = np.zeros((T, n_experts))
    expert_losses = np.zeros((T, n_experts))
    agg_pnl = np.zeros(T)
    agg_turnover = np.zeros(T)
    account_value = np.zeros(T + 1)
    account_value[0] = 1.0

    w_hat_prev = np.full(n_assets, 1.0 / n_assets)  # assumed: uniform initial aggregate

    for t in range(T):
        # Record current aggregation weights
        expert_q[t] = agg.weights

        # Expert portfolios at time t
        f_j = expert_weights_all[t]  # shape (n_experts, n_assets)

        # Aggregate portfolio
        w_hat = agg.aggregate(f_j)
        agg_weights[t] = w_hat

        # Realised return
        y = returns[t]

        # Compute expert losses: ell(f_j(t), y(t)) = -f_j^T y + c ||f_j - f_j_prev||_1
        if t == 0:
            f_j_prev = np.tile(np.full(n_assets, 1.0 / n_assets), (n_experts, 1))
        else:
            f_j_prev = expert_weights_all[t - 1]
        for j_idx in range(n_experts):
            expert_losses[t, j_idx] = compute_expert_loss(f_j[j_idx], f_j_prev[j_idx], y, c)

        # Aggregate gross PnL and turnover
        agg_pnl[t] = float(w_hat @ y)
        agg_turnover[t] = float(np.sum(np.abs(w_hat - w_hat_prev)))

        # Account value update: V(t+1) = V(t) * (1 + w_hat^T y - c ||w_hat - w_hat_prev||_1)
        # (paper Appendix B.1)
        net_return = agg_pnl[t] - c * agg_turnover[t]
        account_value[t + 1] = account_value[t] * (1.0 + net_return)

        # Update aggregation weights
        agg.update(expert_losses[t])
        w_hat_prev = w_hat

    return {
        "agg_weights": agg_weights,
        "expert_q": expert_q,
        "expert_losses": expert_losses,
        "agg_pnl": agg_pnl,
        "agg_turnover": agg_turnover,
        "agg_net_pnl": agg_pnl - c * agg_turnover,
        "account_value": account_value,
    }
