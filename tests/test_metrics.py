"""Tests for portfolio performance metrics."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import (
    compute_cumulative_regret,
    compute_tracking_regret_dp,
    compute_hedge_regret_bound,
    compute_fixed_share_regret_bound,
    compute_account_value,
    cumulative_return,
    max_drawdown,
    annualised_sharpe,
    summarise_performance,
    compute_stock_sequence_regret,
)


# ── compute_cumulative_regret ────────────────────────────────────────────────

def test_cumulative_regret_zero_when_agg_equals_best():
    T, n = 20, 3
    losses = np.ones((T, n)) * 0.01
    agg_losses = np.ones(T) * 0.01  # equals every expert
    regret = compute_cumulative_regret(agg_losses, losses)
    np.testing.assert_allclose(regret, 0.0, atol=1e-10)


def test_cumulative_regret_shape():
    T, n = 50, 4
    expert_losses = np.random.rand(T, n) * 0.01
    agg_losses = np.random.rand(T) * 0.01
    regret = compute_cumulative_regret(agg_losses, expert_losses)
    assert regret.shape == (T,)


def test_cumulative_regret_non_decreasing_if_agg_always_worst():
    T, n = 20, 3
    expert_losses = np.ones((T, n)) * 0.005
    agg_losses = np.ones(T) * 0.01  # always worse than all experts
    regret = compute_cumulative_regret(agg_losses, expert_losses)
    assert (np.diff(regret) >= -1e-10).all(), "Regret should not decrease when agg is always worse"


def test_cumulative_regret_increases_monotonically_when_agg_loses():
    T, n = 10, 2
    expert_losses = np.zeros((T, n))
    agg_losses = np.ones(T) * 0.01
    regret = compute_cumulative_regret(agg_losses, expert_losses)
    np.testing.assert_allclose(regret, np.arange(1, T + 1) * 0.01, atol=1e-10)


# ── compute_tracking_regret_dp ───────────────────────────────────────────────

def test_tracking_regret_dp_K0_equals_best_fixed():
    T, n = 20, 4
    expert_losses = np.random.rand(T, n) * 0.01
    agg_losses = np.random.rand(T) * 0.01
    regret_k0, _ = compute_tracking_regret_dp(expert_losses, agg_losses, K=0)
    regret_fixed = compute_cumulative_regret(agg_losses, expert_losses)
    # K=0 means no switches → should equal best fixed expert
    np.testing.assert_allclose(regret_k0, regret_fixed, atol=1e-6)


def test_tracking_regret_dp_K1_leq_K0():
    T, n = 20, 4
    expert_losses = np.random.rand(T, n) * 0.01
    agg_losses = np.random.rand(T) * 0.01
    regret_k1, _ = compute_tracking_regret_dp(expert_losses, agg_losses, K=1)
    regret_k0, _ = compute_tracking_regret_dp(expert_losses, agg_losses, K=0)
    # Larger K => lower comparator cost => higher or equal regret
    assert regret_k1[-1] >= regret_k0[-1] or np.isclose(regret_k1[-1], regret_k0[-1], atol=1e-6)


def test_tracking_regret_dp_shape():
    T, n = 30, 3
    expert_losses = np.random.rand(T, n) * 0.01
    agg_losses = np.random.rand(T) * 0.01
    regret, best_cost = compute_tracking_regret_dp(expert_losses, agg_losses, K=2)
    assert regret.shape == (T,)
    assert best_cost.shape == (T,)


def test_tracking_regret_dp_perfect_switch():
    """K=1 allows one switch: best K=1 sequence should not exceed best fixed."""
    T, n = 10, 2
    expert_losses = np.zeros((T, n))
    expert_losses[:5, 0] = 1.0  # expert 0 is bad for first half
    expert_losses[5:, 1] = 1.0  # expert 1 is bad for second half
    # Best fixed: either expert has loss 5.0 over T=10
    # Best switch: switch at t=5, total loss = 0.0
    agg_losses = np.zeros(T)
    regret_k1, best_k1 = compute_tracking_regret_dp(expert_losses, agg_losses, K=1)
    assert best_k1[-1] < 5.0  # should improve over best fixed


# ── compute_hedge_regret_bound ───────────────────────────────────────────────

def test_hedge_bound_shape():
    bound = compute_hedge_regret_bound(n_experts=4, horizon=100)
    assert bound.shape == (100,)


def test_hedge_bound_positive():
    bound = compute_hedge_regret_bound(4, 100)
    assert (bound > 0).all()


def test_hedge_bound_non_decreasing():
    bound = compute_hedge_regret_bound(4, 100)
    assert (np.diff(bound) >= 0).all()


def test_hedge_bound_increases_with_experts():
    bound_2 = compute_hedge_regret_bound(2, 100)
    bound_8 = compute_hedge_regret_bound(8, 100)
    assert (bound_8 > bound_2).all()


# ── compute_fixed_share_regret_bound ─────────────────────────────────────────

def test_fixed_share_bound_shape():
    bound = compute_fixed_share_regret_bound(4, 100, K=2)
    assert bound.shape == (100,)


def test_fixed_share_bound_positive():
    bound = compute_fixed_share_regret_bound(4, 100, K=2)
    assert (bound > 0).all()


def test_fixed_share_bound_K0_vs_K1():
    bound_k0 = compute_fixed_share_regret_bound(4, 100, K=0)
    bound_k1 = compute_fixed_share_regret_bound(4, 100, K=1)
    # K=1 allows switching → higher regret bound (worse worst case)
    assert bound_k1[-1] >= bound_k0[-1]


# ── compute_account_value ────────────────────────────────────────────────────

def test_account_value_shape():
    T, n = 30, 4
    ret = np.random.randn(T, n) * 0.01
    w = np.tile(np.full(n, 0.25), (T, 1))
    av = compute_account_value(ret, w, c=0.001)
    assert av.shape == (T + 1,)


def test_account_value_initial():
    T, n = 10, 3
    ret = np.zeros((T, n))
    w = np.tile(np.full(n, 1.0 / n), (T, 1))
    av = compute_account_value(ret, w, initial_value=1.0)
    assert av[0] == 1.0


def test_account_value_zero_return_no_turnover():
    """Zero returns and no turnover → account value constant."""
    T, n = 10, 3
    ret = np.zeros((T, n))
    w = np.tile(np.full(n, 1.0 / n), (T, 1))
    av = compute_account_value(ret, w, c=0.0)
    np.testing.assert_allclose(av[1:], 1.0, atol=1e-10)


# ── cumulative_return, max_drawdown, annualised_sharpe ───────────────────────

def test_cumulative_return_flat():
    av = np.ones(11)  # no change
    assert cumulative_return(av) == 0.0


def test_cumulative_return_doubled():
    av = np.array([1.0, 2.0])
    assert cumulative_return(av) == 1.0


def test_max_drawdown_no_drawdown():
    av = np.cumprod(np.ones(10) * 1.01)  # monotonically increasing
    assert abs(max_drawdown(av)) < 1e-6


def test_max_drawdown_known_case():
    av = np.array([1.0, 2.0, 1.0])  # peak=2, trough=1 → MDD=50%
    assert abs(max_drawdown(av) - 0.5) < 1e-6


def test_max_drawdown_between_0_and_1():
    rng = np.random.default_rng(0)
    av = np.cumprod(1 + rng.normal(0, 0.01, 200))
    mdd = max_drawdown(av)
    assert 0.0 <= mdd <= 1.0


def test_sharpe_flat_returns_zero():
    """Constant returns → std=0 → Sharpe=0."""
    pnl = np.ones(252) * 0.001
    sharpe = annualised_sharpe(pnl)
    assert sharpe == 0.0


def test_sharpe_reasonable_scale():
    rng = np.random.default_rng(1)
    pnl = rng.normal(0.0005, 0.01, 252)
    sharpe = annualised_sharpe(pnl)
    assert -5.0 <= sharpe <= 5.0


# ── summarise_performance ────────────────────────────────────────────────────

def test_summarise_performance_keys():
    av = np.cumprod(1 + np.random.randn(200) * 0.01)
    pnl = np.random.randn(200) * 0.01
    result = summarise_performance(av, pnl, "test")
    assert "cumulative_return" in result
    assert "max_drawdown" in result
    assert "sharpe" in result
    assert "final_value" in result
    assert result["label"] == "test"


# ── compute_stock_sequence_regret ────────────────────────────────────────────

def test_stock_regret_shape():
    T, n = 30, 5
    net_pnl = np.random.randn(T) * 0.005
    stock_returns = np.random.randn(T, n) * 0.01
    r_stock, best_cost = compute_stock_sequence_regret(net_pnl, stock_returns, K=2)
    assert r_stock.shape == (T,)
    assert best_cost.shape == (T,)
