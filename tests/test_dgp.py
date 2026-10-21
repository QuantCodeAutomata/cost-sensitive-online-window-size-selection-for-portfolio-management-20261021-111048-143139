"""Tests for the synthetic data-generating process (DGP)."""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dgp import generate_synthetic_returns, returns_to_prices, make_regime_schedule


# ── make_regime_schedule ──────────────────────────────────────────────────────

def test_make_regime_schedule_lengths():
    schedule = make_regime_schedule(100, [63, 5], [50, 50])
    assert len(schedule) == 100

def test_make_regime_schedule_values():
    schedule = make_regime_schedule(6, [10, 5], [3, 3])
    assert schedule == [10, 10, 10, 5, 5, 5]

def test_make_regime_schedule_raises_on_mismatch():
    with pytest.raises(AssertionError):
        make_regime_schedule(100, [63, 5], [60, 50])  # 110 != 100


# ── generate_synthetic_returns ────────────────────────────────────────────────

def _make_schedule(total=200, w1=63, w2=5, split=100):
    return make_regime_schedule(total, [w1, w2], [split, total - split])

def test_returns_shape():
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(
        n_assets=4, total_days=200, regime_schedule=schedule,
        mu=1e-4, alpha=0.3, sigma=0.01, burnin=63, random_seed=0,
    )
    assert df.shape == (200, 4)

def test_returns_are_dataframe():
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(
        n_assets=4, total_days=200, regime_schedule=schedule,
        mu=1e-4, alpha=0.3, sigma=0.01, burnin=63, random_seed=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)

def test_returns_have_correct_columns():
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(
        n_assets=4, total_days=200, regime_schedule=schedule,
        mu=1e-4, alpha=0.3, sigma=0.01, burnin=63, random_seed=0,
    )
    assert list(df.columns) == ["SYN_1", "SYN_2", "SYN_3", "SYN_4"]

def test_returns_no_nan():
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(
        n_assets=4, total_days=200, regime_schedule=schedule,
        mu=1e-4, alpha=0.3, sigma=0.01, burnin=63, random_seed=0,
    )
    assert not df.isna().any().any()

def test_returns_reproducible():
    schedule = _make_schedule()
    df1, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, random_seed=42)
    df2, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, random_seed=42)
    np.testing.assert_array_equal(df1.values, df2.values)

def test_different_seeds_give_different_returns():
    schedule = _make_schedule()
    df1, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, random_seed=42)
    df2, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, random_seed=99)
    assert not np.allclose(df1.values, df2.values)

def test_returns_reasonable_magnitude():
    """Returns should be small (daily); not exploding."""
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, random_seed=0)
    assert df.abs().max().max() < 0.5, "Returns look too large — DGP issue?"

def test_burnin_raises_when_too_small():
    schedule = _make_schedule()  # max window = 63
    with pytest.raises(AssertionError):
        generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, burnin=10, random_seed=0)

def test_alpha_boundary():
    schedule = _make_schedule()
    with pytest.raises(AssertionError):
        generate_synthetic_returns(4, 200, schedule, 1e-4, 0.0, 0.01, 63, 0)  # alpha=0 not in (0,1)
    with pytest.raises(AssertionError):
        generate_synthetic_returns(4, 200, schedule, 1e-4, 1.0, 0.01, 63, 0)  # alpha=1 not in (0,1)


# ── returns_to_prices ─────────────────────────────────────────────────────────

def test_prices_shape():
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, 0)
    prices = returns_to_prices(df)
    assert prices.shape == (201, 4)

def test_prices_positive():
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, 0)
    prices = returns_to_prices(df)
    assert (prices.values > 0).all(), "All prices should be positive"

def test_prices_initial_value():
    schedule = _make_schedule()
    df, _ = generate_synthetic_returns(4, 200, schedule, 1e-4, 0.3, 0.01, 63, 0)
    prices = returns_to_prices(df, initial_price=50.0)
    assert (prices.values[0] == 50.0).all()

def test_32_assets():
    """Paper: 32 assets, 2000 days (small burn-in for speed)."""
    schedule = make_regime_schedule(100, [63, 5], [50, 50])
    df, _ = generate_synthetic_returns(32, 100, schedule, 5e-4, 0.3, 0.01, 63, 0)
    assert df.shape == (100, 32)
