"""Allocation strategies: 1/N, GMVP, Markowitz, and the efficient frontier."""

import cvxpy as cp
import numpy as np


def _psd(Sigma):
    """Symmetrize a covariance matrix and wrap it for cvxpy as PSD.

    Input: Sigma, np.ndarray (N x N).
    Output: cvxpy constant (N x N) flagged as positive semidefinite.
    """
    S = 0.5 * (Sigma + Sigma.T)
    return cp.psd_wrap(S)


def _solve(problem, w):
    """Solve a cvxpy problem and return the optimal weights.

    Input: problem, cp.Problem; w, cp.Variable (N,).
    Output: np.ndarray (N,) optimal weights. Raises RuntimeError on failure.
    """
    problem.solve()
    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise RuntimeError("cvxpy failed: status " + str(problem.status))
    return np.asarray(w.value, dtype=float).ravel()


def equal_weight(n):
    """1/N portfolio.

    Input: n, int, number of assets.
    Output: w, np.ndarray (n,) with w_i = 1/n.
    """
    return np.full(n, 1.0 / n)


def gmvp(Sigma, long_only=True):
    """Global minimum variance portfolio: minimize w^T Sigma w s.t. sum(w)=1 (and w>=0).

    Input: Sigma, np.ndarray (N x N); long_only, bool (default True; False is
           used only by the closed-form verification check).
    Output: w, np.ndarray (N,).
    """
    n = Sigma.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, _psd(Sigma))), constraints)
    return _solve(problem, w)


def markowitz(mu, Sigma, lam):
    """Markowitz mean-variance portfolio: maximize w^T mu - (lam/2) w^T Sigma w
    s.t. sum(w)=1, w>=0.

    Input: mu, np.ndarray (N,); Sigma, np.ndarray (N x N); lam, float > 0.
    Output: w, np.ndarray (N,).
    """
    n = Sigma.shape[0]
    w = cp.Variable(n)
    objective = cp.Maximize(w @ mu - (lam / 2.0) * cp.quad_form(w, _psd(Sigma)))
    problem = cp.Problem(objective, [cp.sum(w) == 1, w >= 0])
    return _solve(problem, w)


def portfolio_stats(w, mu, Sigma):
    """In-sample annualized expected return and volatility of a portfolio.

    Input: w, np.ndarray (N,); mu, np.ndarray (N,); Sigma, np.ndarray (N x N).
    Output: (expected_return, volatility), two floats.
    """
    return float(w @ mu), float(np.sqrt(w @ Sigma @ w))


def efficient_frontier(mu, Sigma, lam_grid):
    """Long-only efficient frontier traced by Markowitz over a grid of lam.

    Input: mu, np.ndarray (N,); Sigma, np.ndarray (N x N);
           lam_grid, np.ndarray (K,) of lam > 0.
    Output: (rets, vols), two np.ndarray (K,) of in-sample annualized
            expected return and volatility, one per lam.
    """
    rets = np.empty(len(lam_grid))
    vols = np.empty(len(lam_grid))
    for k, lam in enumerate(lam_grid):
        w = markowitz(mu, Sigma, lam)
        rets[k], vols[k] = portfolio_stats(w, mu, Sigma)
    return rets, vols
