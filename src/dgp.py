"""
Synthetic data-generating process (DGP) for the regime-shift experiments.

Paper reference: Section 4.1 "Synthetic Data"
DGP: y(t) = mu + alpha * ((1/j_t) * sum_{k=1}^{j_t} y(t-k) - mu) + epsilon(t)
where epsilon(t) ~ N(0, sigma^2 * I), j_t is the target window for the current regime.

Assumed parameters (not specified by paper — documented as implementation choice):
  mu, alpha, sigma^2, initial history generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_returns(
    n_assets: int,
    total_days: int,
    regime_schedule: list[int],
    mu: float | np.ndarray,
    alpha: float,
    sigma: float,
    burnin: int = 126,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic return panel following the paper's DGP.

    DGP (paper p.7, Section 4.1):
      y_i(t) = mu_i + alpha * ((1/j_t) * sum_{k=1}^{j_t} y_i(t-k) - mu_i) + eps_i(t)
      eps_i(t) ~ N(0, sigma^2) i.i.d. across assets and time.

    Parameters
    ----------
    n_assets : int
        Number of synthetic assets (paper: 32).
    total_days : int
        Number of trading days to generate (paper: 2,000).
    regime_schedule : list[int]
        For each trading day t in [0, total_days), the target window j_t.
        Must have len == total_days.
    mu : float or np.ndarray
        Baseline daily expected return. If scalar, broadcast to all assets.
        Assumed: not specified by paper.
    alpha : float
        Signal ratio, 0 < alpha < 1. Assumed: not specified by paper.
    sigma : float
        Noise standard deviation per asset per day. Assumed: not specified.
    burnin : int
        Number of initial lagged returns to generate before the first trading day.
        Must be >= max(regime_schedule). Assumed: 126 (largest window across
        all experiments).
    random_seed : int
        Random seed for reproducibility. Assumed: not specified by paper.

    Returns
    -------
    returns_df : pd.DataFrame
        DataFrame of shape (total_days, n_assets) with daily returns.
        Index is a DatetimeIndex of business days starting from 2020-01-01.
    all_dates : pd.DatetimeIndex
        All business-day dates including burn-in period.
    """
    assert len(regime_schedule) == total_days, "regime_schedule length must equal total_days"
    assert 0.0 < alpha < 1.0, "alpha must be in (0, 1)"
    assert sigma > 0.0, "sigma must be positive"
    assert burnin >= max(regime_schedule), (
        f"burnin={burnin} must be >= max window {max(regime_schedule)}"
    )

    rng = np.random.default_rng(random_seed)
    mu_vec = np.full(n_assets, mu) if np.isscalar(mu) else np.asarray(mu, dtype=float)
    assert mu_vec.shape == (n_assets,)

    # Allocate full history: burnin + total_days rows
    full_len = burnin + total_days
    Y = np.zeros((full_len, n_assets))

    # Initialise burn-in as i.i.d. N(mu, sigma^2) draws
    Y[:burnin] = rng.normal(loc=mu_vec, scale=sigma, size=(burnin, n_assets))

    # Generate trading-day returns using the recursive DGP
    for t_local, j_t in enumerate(regime_schedule):
        t = burnin + t_local  # index into full history
        # Rolling mean of last j_t returns prior to t
        past = Y[t - j_t : t]          # shape (j_t, n_assets)
        rolling_mean = past.mean(axis=0)
        eps = rng.normal(loc=0.0, scale=sigma, size=n_assets)
        Y[t] = mu_vec + alpha * (rolling_mean - mu_vec) + eps

    # Build business-day date index
    all_dates = pd.bdate_range(start="2020-01-01", periods=full_len)

    returns_df = pd.DataFrame(
        Y[burnin:], index=all_dates[burnin:], columns=[f"SYN_{i+1}" for i in range(n_assets)]
    )
    return returns_df, all_dates


def returns_to_prices(
    returns_df: pd.DataFrame,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """
    Convert simple return panel to price panel.

    S_i(t+1) = S_i(t) * (1 + y_i(t)).

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, shape (T, n_assets).
    initial_price : float
        Starting price for all assets (assumed: 100.0).

    Returns
    -------
    prices_df : pd.DataFrame
        Daily prices, shape (T+1, n_assets). First row is initial_price.
    """
    prices = np.empty((len(returns_df) + 1, returns_df.shape[1]))
    prices[0] = initial_price
    for t in range(len(returns_df)):
        prices[t + 1] = prices[t] * (1.0 + returns_df.iloc[t].values)
    assert (prices > 0).all(), "Negative prices detected — check DGP parameters"

    price_index = returns_df.index.insert(0, returns_df.index[0] - pd.tseries.offsets.BDay(1))
    return pd.DataFrame(prices, index=price_index, columns=returns_df.columns)


def make_regime_schedule(
    total_days: int,
    regime_windows: list[int],
    regime_lengths: list[int],
) -> list[int]:
    """
    Build j_t schedule from a list of (window, length) pairs.

    Parameters
    ----------
    total_days : int
        Total number of trading days.
    regime_windows : list[int]
        Window size for each regime, e.g. [63, 5].
    regime_lengths : list[int]
        Duration in days for each regime, e.g. [1000, 1000].

    Returns
    -------
    schedule : list[int]
        Length total_days list of j_t values.
    """
    assert sum(regime_lengths) == total_days, "Regime lengths must sum to total_days"
    schedule: list[int] = []
    for w, L in zip(regime_windows, regime_lengths):
        schedule.extend([w] * L)
    return schedule
