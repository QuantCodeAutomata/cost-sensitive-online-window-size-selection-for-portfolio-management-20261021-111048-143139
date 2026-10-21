"""Tests for cost-sensitive mean-variance expert portfolio optimisation."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.expert import estimate_rolling_stats, solve_expert_portfolio, ExpertSet


# ── estimate_rolling_stats ────────────────────────────────────────────────────

def _make_returns(T=100, n=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.01, (T, n))


def test_rolling_stats_shape():
    R = _make_returns(100, 5)
    mu, Sigma = estimate_rolling_stats(R, window=20, t=20)
    assert mu.shape == (5,)
    assert Sigma.shape == (5, 5)


def test_rolling_stats_positive_definite():
    R = _make_returns(100, 5)
    _, Sigma = estimate_rolling_stats(R, window=20, t=50, regularization_lambda=1e-4)
    eigenvalues = np.linalg.eigvals(Sigma)
    assert (eigenvalues > 0).all(), "Regularised covariance should be positive definite"


def test_rolling_stats_mean_correct():
    R = np.zeros((20, 3))
    R[10:20] = 1.0  # last 10 rows are 1.0
    mu, _ = estimate_rolling_stats(R, window=10, t=20)
    np.testing.assert_allclose(mu, 1.0, rtol=1e-10)


def test_rolling_stats_raises_if_not_enough_history():
    R = _make_returns(10, 5)
    with pytest.raises(AssertionError):
        estimate_rolling_stats(R, window=20, t=10)


def test_rolling_stats_single_row():
    R = _make_returns(100, 5)
    mu, Sigma = estimate_rolling_stats(R, window=1, t=10)
    assert mu.shape == (5,)
    # single-row cov falls back to identity
    assert Sigma.shape == (5, 5)


# ── solve_expert_portfolio ────────────────────────────────────────────────────

def _get_problem_data(n=5, seed=1):
    rng = np.random.default_rng(seed)
    mu_hat = rng.normal(0.001, 0.01, n)
    A = rng.normal(0, 1, (n, n))
    Sigma_hat = A @ A.T / n + 1e-4 * np.eye(n)
    w_prev = np.full(n, 1.0 / n)
    return mu_hat, Sigma_hat, w_prev


def test_solve_expert_simplex_constraint():
    mu, Sigma, w_prev = _get_problem_data(5)
    w = solve_expert_portfolio(mu, Sigma, w_prev, c=0.001)
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights should sum to 1; got {w.sum()}"
    assert (w >= -1e-6).all(), "All weights should be non-negative"


def test_solve_expert_result_shape():
    mu, Sigma, w_prev = _get_problem_data(5)
    w = solve_expert_portfolio(mu, Sigma, w_prev, c=0.001)
    assert w.shape == (5,)


def test_solve_expert_no_turnover_cost():
    """With c=0, solution may differ from c=0.001."""
    mu, Sigma, w_prev = _get_problem_data(5)
    w_c0 = solve_expert_portfolio(mu, Sigma, w_prev, c=0.0)
    w_c1 = solve_expert_portfolio(mu, Sigma, w_prev, c=0.001)
    # Both should be valid simplex points
    assert abs(w_c0.sum() - 1.0) < 1e-5
    assert abs(w_c1.sum() - 1.0) < 1e-5


def test_solve_expert_high_cost_stays_near_prev():
    """Very high transaction cost should keep portfolio near w_prev."""
    mu, Sigma, w_prev = _get_problem_data(5)
    w = solve_expert_portfolio(mu, Sigma, w_prev, c=100.0)
    # Should not move far from w_prev
    assert np.sum(np.abs(w - w_prev)) < 0.1 + 1e-5


def test_solve_expert_single_asset():
    mu = np.array([0.01])
    Sigma = np.array([[0.01]])
    w_prev = np.array([1.0])
    w = solve_expert_portfolio(mu, Sigma, w_prev, c=0.001)
    np.testing.assert_allclose(w, [1.0], atol=1e-5)


def test_solve_expert_all_positive_return():
    """With all positive expected returns and no cost, should concentrate on highest."""
    n = 4
    mu = np.array([0.001, 0.002, 0.01, 0.001])
    A = np.eye(n) * 0.01
    Sigma = A @ A.T + 1e-4 * np.eye(n)
    w_prev = np.full(n, 0.25)
    w = solve_expert_portfolio(mu, Sigma, w_prev, c=0.0)
    assert abs(w.sum() - 1.0) < 1e-5
    assert w[2] >= 0.0  # highest return asset should have non-trivial weight


# ── ExpertSet ─────────────────────────────────────────────────────────────────

def test_expert_set_initial_uniform():
    es = ExpertSet(windows=(5, 10, 21), n_assets=4, c=0.001)
    for j in (5, 10, 21):
        np.testing.assert_allclose(es.w[j], np.full(4, 0.25), atol=1e-10)


def test_expert_set_step_returns_dict():
    es = ExpertSet(windows=(5, 10, 21), n_assets=4, c=0.001)
    R = _make_returns(50, 4)
    portfolios = es.step(R, t=21)
    assert set(portfolios.keys()) == {5, 10, 21}
    for j in (5, 10, 21):
        w = portfolios[j]
        assert abs(w.sum() - 1.0) < 1e-5


def test_expert_set_reset():
    es = ExpertSet(windows=(5, 10), n_assets=4, c=0.001)
    R = _make_returns(50, 4)
    es.step(R, t=20)
    es.reset()
    for j in (5, 10):
        np.testing.assert_allclose(es.w[j], np.full(4, 0.25), atol=1e-10)


def test_expert_set_not_enough_history_fallback():
    """If t < window, expert should return previous weights unchanged."""
    es = ExpertSet(windows=(5, 10, 21), n_assets=4, c=0.001)
    R = _make_returns(50, 4)
    portfolios = es.step(R, t=5)  # only window=5 can compute; 10 and 21 can't
    # window 5: t=5 >= 5 so can compute
    assert abs(portfolios[5].sum() - 1.0) < 1e-5
    # window 10: t=5 < 10, should return initial uniform
    np.testing.assert_allclose(portfolios[10], np.full(4, 0.25), atol=1e-5)
