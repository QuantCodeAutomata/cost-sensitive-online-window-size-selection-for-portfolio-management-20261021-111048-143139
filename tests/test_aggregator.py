"""Tests for Hedge and Fixed Share online aggregation algorithms."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.aggregator import (
    HedgeAggregator, FixedShareAggregator, run_aggregation,
    compute_expert_loss, compute_hedge_eta, compute_fixed_share_alpha,
    compute_switching_budget,
)


# ── compute_expert_loss ───────────────────────────────────────────────────────

def test_expert_loss_no_turnover():
    w = np.array([0.5, 0.5])
    w_prev = np.array([0.5, 0.5])
    y = np.array([0.01, 0.02])
    loss = compute_expert_loss(w, w_prev, y, c=0.001)
    expected = -(0.5 * 0.01 + 0.5 * 0.02)  # = -0.015
    assert abs(loss - expected) < 1e-10


def test_expert_loss_with_turnover():
    w = np.array([1.0, 0.0])
    w_prev = np.array([0.5, 0.5])
    y = np.array([0.01, 0.0])
    # PnL = 1.0*0.01 = 0.01; turnover = |1-0.5| + |0-0.5| = 1.0
    loss = compute_expert_loss(w, w_prev, y, c=0.001)
    expected = -0.01 + 0.001 * 1.0
    assert abs(loss - expected) < 1e-10


def test_expert_loss_zero_return():
    w = np.array([0.5, 0.5])
    w_prev = np.array([0.5, 0.5])
    y = np.zeros(2)
    loss = compute_expert_loss(w, w_prev, y, c=0.001)
    assert loss == 0.0


# ── compute_hedge_eta ─────────────────────────────────────────────────────────

def test_hedge_eta_positive():
    eta = compute_hedge_eta(n_experts=4, horizon=100)
    assert eta > 0.0


def test_hedge_eta_decreases_with_horizon():
    eta_small = compute_hedge_eta(4, 100)
    eta_large = compute_hedge_eta(4, 10000)
    assert eta_small > eta_large


def test_hedge_eta_increases_with_experts():
    eta_few = compute_hedge_eta(2, 100)
    eta_many = compute_hedge_eta(8, 100)
    assert eta_many > eta_few


# ── compute_switching_budget ──────────────────────────────────────────────────

def test_switching_budget_at_least_1():
    K = compute_switching_budget(100, epsilon=0.5)
    assert K >= 1


def test_switching_budget_increases_with_horizon():
    K_small = compute_switching_budget(100, 0.5)
    K_large = compute_switching_budget(10000, 0.5)
    assert K_large >= K_small


# ── HedgeAggregator ───────────────────────────────────────────────────────────

def test_hedge_initial_uniform():
    agg = HedgeAggregator(n_experts=4, horizon=100)
    np.testing.assert_allclose(agg.weights, np.full(4, 0.25), atol=1e-10)


def test_hedge_weights_sum_to_one_after_update():
    agg = HedgeAggregator(n_experts=4, horizon=100)
    losses = np.array([0.01, 0.02, -0.01, 0.0])
    agg.update(losses)
    assert abs(agg.weights.sum() - 1.0) < 1e-10


def test_hedge_weights_nonnegative():
    agg = HedgeAggregator(n_experts=4, horizon=100)
    for _ in range(10):
        losses = np.random.randn(4) * 0.1
        agg.update(losses)
    assert (agg.weights >= 0).all()


def test_hedge_penalises_high_loss_experts():
    """Expert with highest loss should receive lowest weight."""
    agg = HedgeAggregator(n_experts=4, horizon=100, eta=1.0)
    losses = np.array([1.0, 0.0, 0.0, 0.0])  # expert 0 has highest loss
    agg.update(losses)
    assert agg.weights[0] < agg.weights[1]


def test_hedge_aggregate_shape():
    agg = HedgeAggregator(n_experts=4, horizon=100)
    experts = np.random.dirichlet(np.ones(6), size=4)  # (4, 6)
    w_hat = agg.aggregate(experts)
    assert w_hat.shape == (6,)


def test_hedge_aggregate_sums_to_one():
    agg = HedgeAggregator(n_experts=4, horizon=100)
    experts = np.array([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]])
    w_hat = agg.aggregate(experts)
    assert abs(w_hat.sum() - 1.0) < 1e-10


# ── FixedShareAggregator ──────────────────────────────────────────────────────

def test_fixed_share_initial_uniform():
    agg = FixedShareAggregator(n_experts=4, horizon=100, K=2)
    np.testing.assert_allclose(agg.weights, np.full(4, 0.25), atol=1e-10)


def test_fixed_share_weights_sum_to_one():
    agg = FixedShareAggregator(n_experts=4, horizon=100, K=2)
    losses = np.array([0.01, 0.02, -0.01, 0.0])
    agg.update(losses)
    assert abs(agg.weights.sum() - 1.0) < 1e-10


def test_fixed_share_weights_nonneg():
    agg = FixedShareAggregator(n_experts=4, horizon=200, K=3)
    for _ in range(20):
        losses = np.abs(np.random.randn(4)) * 0.05
        agg.update(losses)
    assert (agg.weights >= 0).all()


def test_fixed_share_mixing_spreads_weight():
    """Alpha > 0 should spread weight away from degenerate distribution."""
    agg = FixedShareAggregator(n_experts=4, horizon=100, K=5, alpha=0.3)
    # Force all weight to expert 0
    agg.q = np.array([1.0, 0.0, 0.0, 0.0])
    losses = np.zeros(4)
    agg.update(losses)
    # After mixing, other experts should have non-zero weight
    assert (agg.weights[1:] > 0).all()


# ── run_aggregation ───────────────────────────────────────────────────────────

def _make_aggregation_inputs(T=50, n_experts=4, n_assets=5, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.001, 0.01, (T, n_assets))
    expert_weights_all = np.zeros((T, n_experts, n_assets))
    for t in range(T):
        for j in range(n_experts):
            w = rng.dirichlet(np.ones(n_assets))
            expert_weights_all[t, j] = w
    return returns, expert_weights_all


def test_run_hedge_returns_keys():
    returns, experts = _make_aggregation_inputs()
    res = run_aggregation(returns, experts, "hedge", 4, 50)
    for key in ("agg_weights", "expert_q", "expert_losses", "agg_pnl",
                "agg_turnover", "agg_net_pnl", "account_value"):
        assert key in res


def test_run_hedge_account_value_shape():
    returns, experts = _make_aggregation_inputs()
    res = run_aggregation(returns, experts, "hedge", 4, 50)
    assert res["account_value"].shape == (51,)
    assert res["account_value"][0] == 1.0


def test_run_fixed_share_account_value_positive():
    returns, experts = _make_aggregation_inputs()
    res = run_aggregation(returns, experts, "fixed_share", 4, 50, K=2)
    assert (res["account_value"] > 0).all()


def test_run_agg_weights_sum_to_one():
    returns, experts = _make_aggregation_inputs()
    res = run_aggregation(returns, experts, "hedge", 4, 50)
    sums = res["agg_weights"].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_run_expert_q_sums_to_one():
    returns, experts = _make_aggregation_inputs()
    res = run_aggregation(returns, experts, "hedge", 4, 50)
    sums = res["expert_q"].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_run_unknown_type_raises():
    returns, experts = _make_aggregation_inputs()
    with pytest.raises(ValueError):
        run_aggregation(returns, experts, "unknown", 4, 50)


def test_run_single_expert_hedge():
    """With 1 expert, aggregate == expert portfolio."""
    T, n = 20, 4
    returns = np.zeros((T, n))
    experts = np.ones((T, 1, n)) * 0.25  # uniform, 1 expert
    res = run_aggregation(returns, experts, "hedge", 1, T)
    np.testing.assert_allclose(res["agg_weights"], 0.25, atol=1e-8)
