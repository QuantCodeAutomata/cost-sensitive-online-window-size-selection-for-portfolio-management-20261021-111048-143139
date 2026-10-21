"""
Cost-sensitive mean-variance expert portfolio optimization.

Paper reference: Section 2.1 "Sliding-Window Expert and Cost-Sensitive Mean-Variance Model"

Problem (paper eq. 1):
  max_{w in W} mu_hat_j(t)^T w - 0.5 * w^T Sigma_hat_j(t) w - c * ||w - w_prev||_1
  subject to: w >= 0, sum(w) == 1

Linearised for CVXPY (implementation note, paper p.2):
  Introduce u >= 0 such that u >= w - w_prev and u >= -(w - w_prev),
  then penalise c * sum(u).

# Using CVXPY for expert mean-variance with L1 turnover — Context7 confirmed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import cvxpy as cp
import osqp


def estimate_rolling_stats(
    returns: np.ndarray,
    window: int,
    t: int,
    regularization_lambda: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling mean and regularised covariance over previous j returns.

    Parameters
    ----------
    returns : np.ndarray
        Return matrix of shape (T_history, n_assets). Rows are time steps.
    window : int
        Rolling window length j.
    t : int
        Current time index (exclusive). Uses returns[t-j:t].
    regularization_lambda : float
        Ridge regularisation added to diagonal of Sigma_hat to ensure
        positive definiteness. Implementation choice — not specified by paper.

    Returns
    -------
    mu_hat : np.ndarray, shape (n_assets,)
    Sigma_hat : np.ndarray, shape (n_assets, n_assets)
    """
    assert t >= window, f"Need t >= window; got t={t}, window={window}"
    slice_ = returns[t - window : t]          # shape (window, n_assets)
    mu_hat = slice_.mean(axis=0)
    # Sample covariance (divides by window-1)
    if window > 1:
        Sigma_hat = np.cov(slice_.T, ddof=1)
    else:
        Sigma_hat = np.eye(returns.shape[1])
    # Ridge regularisation — implementation choice, documented
    Sigma_hat += regularization_lambda * np.eye(Sigma_hat.shape[0])
    # Symmetrize to satisfy CVXPY quad_form requirement (removes float rounding asymmetry)
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0
    return mu_hat, Sigma_hat


def solve_expert_portfolio(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    w_prev: np.ndarray,
    c: float = 0.001,
    solver: str | None = None,
) -> np.ndarray:
    """
    Solve the cost-sensitive mean-variance problem (paper eq. 1).

    max_{w in W} mu_hat^T w - 0.5 w^T Sigma_hat w - c ||w - w_prev||_1
    s.t. w >= 0, sum(w) == 1

    Linearised L1: introduce u >= w - w_prev and u >= -(w - w_prev), u >= 0.
    Objective becomes: max mu^T w - 0.5 w^T Sigma w - c 1^T u.

    Parameters
    ----------
    mu_hat : np.ndarray, shape (n,)
        Estimated mean returns.
    Sigma_hat : np.ndarray, shape (n, n)
        Estimated (regularised) covariance matrix.
    w_prev : np.ndarray, shape (n,)
        Previous expert portfolio weights.
    c : float
        Transaction cost rate (paper: 0.001).
    solver : str or None
        CVXPY solver name. None uses CVXPY default (OSQP or CLARABEL).

    Returns
    -------
    w_opt : np.ndarray, shape (n,)
        Optimal portfolio weights on the simplex.
    """
    n = len(mu_hat)
    w = cp.Variable(n, nonneg=True)
    u = cp.Variable(n, nonneg=True)          # L1 linearisation auxiliary variable
    delta = w - w_prev

    # Use psd_wrap to skip ARPACK PSD eigenvalue test; Sigma_hat is already
    # regularised (Sigma_hat += lambda*I) so it is PSD by construction.
    objective = cp.Maximize(
        mu_hat @ w - 0.5 * cp.quad_form(w, cp.psd_wrap(Sigma_hat)) - c * cp.sum(u)
    )
    constraints = [
        cp.sum(w) == 1,
        u >= delta,
        u >= -delta,
    ]
    problem = cp.Problem(objective, constraints)
    try:
        if solver is not None:
            problem.solve(solver=solver, warm_start=True)
        else:
            problem.solve(warm_start=True)
    except cp.SolverError:
        # Fallback: return previous weights if solver fails
        return w_prev.copy()

    if w.value is None or problem.status in {"infeasible", "unbounded"}:
        # Fallback to previous weights on solver failure
        return w_prev.copy()

    w_opt = np.asarray(w.value, dtype=float)
    w_opt = np.clip(w_opt, 0.0, None)
    w_opt /= w_opt.sum()          # re-normalise due to numerical noise
    return w_opt


class ExpertSet:
    """
    Manages a collection of sliding-window mean-variance experts.

    Each expert j ∈ N maintains its own previous portfolio w_j(t-1)
    and produces a new recommendation w_j(t) at each date.
    """

    def __init__(
        self,
        windows: tuple[int, ...],
        n_assets: int,
        c: float = 0.001,
        regularization_lambda: float = 1e-4,
        solver: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        windows : tuple[int, ...]
            Candidate window sizes N.
        n_assets : int
            Number of assets.
        c : float
            Transaction cost rate.
        regularization_lambda : float
            Ridge for covariance (implementation choice).
        solver : str or None
            CVXPY solver.
        """
        self.windows = windows
        self.n_assets = n_assets
        self.c = c
        self.regularization_lambda = regularization_lambda
        self.solver = solver
        self.n_experts = len(windows)
        # Initialise expert portfolios uniformly (assumed: uniform w_0)
        self.w: dict[int, np.ndarray] = {
            j: np.full(n_assets, 1.0 / n_assets) for j in windows
        }

    def step(
        self, returns_history: np.ndarray, t: int
    ) -> dict[int, np.ndarray]:
        """
        Compute expert portfolios for date t.

        Parameters
        ----------
        returns_history : np.ndarray
            Full return history up to (but not including) t.
            Shape (T_total, n_assets). Row t is y(t).
        t : int
            Current time index.

        Returns
        -------
        portfolios : dict[int, np.ndarray]
            Mapping from window size to portfolio weights.
        """
        portfolios: dict[int, np.ndarray] = {}
        for j in self.windows:
            if t < j:
                # Not enough history; use previous weights unchanged
                portfolios[j] = self.w[j].copy()
                continue
            mu_hat, Sigma_hat = estimate_rolling_stats(
                returns_history, j, t, self.regularization_lambda
            )
            w_new = solve_expert_portfolio(
                mu_hat, Sigma_hat, self.w[j], c=self.c, solver=self.solver
            )
            self.w[j] = w_new
            portfolios[j] = w_new.copy()
        return portfolios

    def reset(self) -> None:
        """Reset all expert portfolios to uniform."""
        self.w = {j: np.full(self.n_assets, 1.0 / self.n_assets) for j in self.windows}


# ── Fast direct-OSQP solver (warm-started, bypasses CVXPY overhead) ───────────

class FastExpertSolver:
    """
    Direct OSQP warm-started solver for the cost-sensitive MV expert problem.

    Maintains a persistent OSQP instance and updates P, q, l at each step
    using osqp.OSQP.update(), which preserves the primal/dual warm-start.

    Implementation note: uses OSQP directly (bypassing CVXPY) for speed.
    Benchmarked at ~0.03–0.20 s per solve (depending on n and polishing).

    Constraint layout (2n decision variables: [w; u]):
        Row 0:          sum(w) == 1
        Rows 1..n:      w >= 0  (u part unconstrained in the variable box)
        Rows n+1..2n:   u >= 0
        Rows 2n+1..3n:  u - w >= -w_prev  (u >= w - w_prev)
        Rows 3n+1..4n:  u + w >= w_prev   (u >= -(w - w_prev))

    P (upper-triangular, block-diagonal [Sigma, 0]):
        Objective: -mu^T w + 0.5 w^T Sigma w + c * 1^T u (minimization form).
    """

    def __init__(self, n_assets: int, c: float, reg_lambda: float) -> None:
        self.n = n_assets
        self.c = c
        self.reg_lambda = reg_lambda
        self._prob: osqp.OSQP | None = None
        self._O = sp.csc_matrix((n_assets, n_assets))
        self._A = self._build_A_static()
        self._u_static = np.concatenate([
            [1.0], np.ones(n_assets), np.full(3 * n_assets, np.inf)
        ])

    def _build_A_static(self) -> sp.csc_matrix:
        n = self.n
        I = sp.eye(n, format='csc')
        O = self._O
        return sp.vstack([
            sp.hstack([sp.csc_matrix(np.ones((1, n))), sp.csc_matrix(np.zeros((1, n)))]),
            sp.hstack([I, O]),
            sp.hstack([O, I]),
            sp.hstack([-I, I]),
            sp.hstack([I, I]),
        ], format='csc')

    def _build_P(self, Sigma: np.ndarray) -> sp.csc_matrix:
        Sigma_triu = sp.triu(sp.csc_matrix(Sigma), format='csc')
        return sp.block_diag([Sigma_triu, self._O], format='csc')

    def solve(self, mu_hat: np.ndarray, Sigma_hat: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        """Solve for one expert at one time step (warm-starts from previous call)."""
        n = self.n
        P = self._build_P(Sigma_hat)
        q = np.concatenate([-mu_hat, self.c * np.ones(n)])
        l = np.concatenate([[1.0], np.zeros(2 * n), -w_prev, w_prev])

        if self._prob is None:
            self._prob = osqp.OSQP()
            self._prob.setup(
                P, q, self._A, l, self._u_static,
                verbose=False, polish=False,
                warm_starting=True,
                eps_abs=1e-5, eps_rel=1e-5,
                max_iter=8000,
            )
        else:
            self._prob.update(Px=P.data, q=q, l=l)

        res = self._prob.solve()
        if res.info.status not in ("solved", "solved_inaccurate"):
            return w_prev.copy()

        w = np.maximum(res.x[:n], 0.0)
        s = w.sum()
        return w / s if s > 1e-10 else np.ones(n) / n


def _compute_window_portfolios(
    returns_all: np.ndarray,
    window: int,
    t_offset: int,
    T: int,
    c: float,
    reg_lambda: float,
) -> np.ndarray:
    """
    Compute all T portfolios for a single window (designed for joblib worker).

    Uses FastExpertSolver with warm starting across consecutive time steps.
    """
    n = returns_all.shape[1]
    portfolios = np.zeros((T, n))
    solver = FastExpertSolver(n, c, reg_lambda)
    w_prev = np.ones(n) / n

    for t_local in range(T):
        t_abs = t_offset + t_local
        if t_abs < window:
            portfolios[t_local] = w_prev
            continue
        mu_hat, Sigma_hat = estimate_rolling_stats(returns_all, window, t_abs, reg_lambda)
        w = solver.solve(mu_hat, Sigma_hat, w_prev)
        portfolios[t_local] = w
        w_prev = w

    return portfolios


def compute_expert_portfolios_parallel(
    returns_all: np.ndarray,
    windows: tuple[int, ...],
    t_offset: int,
    T: int,
    c: float = 0.001,
    reg_lambda: float = 1e-4,
    n_jobs: int = -1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute expert portfolios for all windows, parallelised by window.

    Uses FastExpertSolver (direct OSQP, warm-started) per window, running
    each window's time-series sequentially while windows run in parallel.

    Parameters
    ----------
    returns_all : np.ndarray, shape (T_all, n_assets)
        Full return history including pre-sample rows.
    windows : tuple[int, ...]
        Candidate window sizes.
    t_offset : int
        Row index in returns_all where the evaluation period begins.
    T : int
        Number of evaluation trading days.
    c : float
        Transaction cost rate.
    reg_lambda : float
        Covariance ridge regularisation.
    n_jobs : int
        Number of parallel workers (-1 = all CPUs).

    Returns
    -------
    expert_weights_all : np.ndarray, shape (T, n_experts, n_assets)
    """
    from joblib import Parallel, delayed

    if verbose:
        print(f"  Parallel expert computation: {len(windows)} windows, {T} days, n={returns_all.shape[1]}")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_compute_window_portfolios)(returns_all, window, t_offset, T, c, reg_lambda)
        for window in windows
    )

    n = returns_all.shape[1]
    expert_weights_all = np.zeros((T, len(windows), n))
    for j_idx, w_arr in enumerate(results):
        expert_weights_all[:, j_idx, :] = w_arr

    return expert_weights_all
