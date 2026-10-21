"""Tests for baseline online portfolio algorithms."""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines import (
    UniversalPortfolio, ExponentiatedGradient, PAMR, CWMR, OLMAR,
    run_baseline,
)


def _make_returns(T=100, n=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.01, (T, n))


# ── project_simplex sanity via ExponentiatedGradient ─────────────────────────

def test_eg_simplex_projection():
    eg = ExponentiatedGradient(n_assets=5)
    v = np.array([-0.5, 1.5, 0.3, -0.1, 0.8])
    proj = eg.project_simplex(v)
    assert abs(proj.sum() - 1.0) < 1e-8
    assert (proj >= -1e-10).all()


# ── UniversalPortfolio ────────────────────────────────────────────────────────

def test_up_initial_uniform():
    up = UniversalPortfolio(n_assets=4)
    np.testing.assert_allclose(up.w, np.full(4, 0.25), atol=1e-10)


def test_up_update_stays_on_simplex():
    up = UniversalPortfolio(n_assets=4)
    y = np.array([0.01, -0.02, 0.005, 0.03])
    w_new = up.update(y)
    assert abs(w_new.sum() - 1.0) < 1e-6
    assert (w_new >= -1e-8).all()


def test_up_update_shape():
    up = UniversalPortfolio(n_assets=5)
    y = np.zeros(5)
    w = up.update(y)
    assert w.shape == (5,)


# ── ExponentiatedGradient ─────────────────────────────────────────────────────

def test_eg_update_simplex():
    eg = ExponentiatedGradient(n_assets=4, eta=0.05)
    y = np.array([0.01, 0.02, -0.01, 0.005])
    w = eg.update(y)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()


def test_eg_favours_high_return_assets():
    eg = ExponentiatedGradient(n_assets=3, eta=1.0)
    y = np.array([0.1, 0.0, 0.0])  # asset 0 best
    for _ in range(10):
        eg.update(y)
    assert eg.w[0] > eg.w[1]


# ── PAMR ──────────────────────────────────────────────────────────────────────

def test_pamr_initial_uniform():
    pamr = PAMR(n_assets=5)
    np.testing.assert_allclose(pamr.w, np.full(5, 0.2), atol=1e-10)


def test_pamr_update_simplex():
    pamr = PAMR(n_assets=4)
    y = np.array([0.01, -0.01, 0.02, 0.0])
    w = pamr.update(y)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()


# ── CWMR ──────────────────────────────────────────────────────────────────────

def test_cwmr_update_simplex():
    cwmr = CWMR(n_assets=4)
    y = np.array([0.01, 0.02, -0.005, 0.015])
    w = cwmr.update(y)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()


# ── OLMAR ─────────────────────────────────────────────────────────────────────

def test_olmar_update_simplex():
    olmar = OLMAR(n_assets=4, m=3)  # m is the window size parameter
    returns = _make_returns(20, 4)
    # Warmup: OLMAR maintains internal price window state
    for t in range(3):
        olmar.update(returns[t])
    w = olmar.update(returns[3])
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-8).all()


# ── run_baseline ──────────────────────────────────────────────────────────────

def test_run_baseline_eg_keys():
    returns = _make_returns(50, 5)
    result = run_baseline("EG", returns, c=0.001)
    for key in ("account_value", "net_pnl", "agg_weights"):
        assert key in result


def test_run_baseline_up_account_value_positive():
    returns = _make_returns(50, 5)
    result = run_baseline("UP", returns, c=0.001)
    assert (result["account_value"] > 0).all()


def test_run_baseline_pamr_shape():
    returns = _make_returns(30, 4)
    result = run_baseline("PAMR", returns, c=0.001)
    assert result["account_value"].shape == (31,)
    assert result["net_pnl"].shape == (30,)


def test_run_baseline_cwmr():
    returns = _make_returns(30, 4)
    result = run_baseline("CWMR", returns, c=0.001)
    assert result["account_value"][0] == 1.0


def test_run_baseline_olmar():
    returns = _make_returns(50, 4)
    result = run_baseline("OLMAR", returns, c=0.001)
    assert result["account_value"].shape == (51,)


def test_run_baseline_unknown_raises():
    returns = _make_returns(30, 4)
    with pytest.raises(ValueError):
        run_baseline("NONEXISTENT", returns)


def test_run_baseline_agg_weights_simplex():
    returns = _make_returns(40, 5)
    result = run_baseline("EG", returns, c=0.001)
    sums = result["agg_weights"].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)
