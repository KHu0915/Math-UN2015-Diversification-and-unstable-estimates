"""Annualized estimates of the mean vector and covariance matrix."""

import numpy as np
from sklearn.covariance import LedoitWolf

import config


def estimate_mu(returns):
    """Annualized sample mean of daily log returns.

    Input: returns, pd.DataFrame or np.ndarray (T x N) of daily log returns.
    Output: mu_hat, np.ndarray (N,), mean x 252.
    """
    R = np.asarray(returns, dtype=float)
    return R.mean(axis=0) * config.TRADING_DAYS


def estimate_sigma_sample(returns):
    """Annualized sample covariance (divisor T-1) of daily log returns.

    Input: returns, pd.DataFrame or np.ndarray (T x N).
    Output: Sigma_hat, np.ndarray (N x N), covariance x 252.
    """
    R = np.asarray(returns, dtype=float)
    return np.cov(R, rowvar=False, ddof=1) * config.TRADING_DAYS


def estimate_sigma_lw(returns):
    """Annualized Ledoit-Wolf shrinkage covariance of daily log returns.

    Input: returns, pd.DataFrame or np.ndarray (T x N).
    Output: Sigma_lw, np.ndarray (N x N), shrinkage covariance x 252.
    """
    R = np.asarray(returns, dtype=float)
    lw = LedoitWolf().fit(R)
    return lw.covariance_ * config.TRADING_DAYS
