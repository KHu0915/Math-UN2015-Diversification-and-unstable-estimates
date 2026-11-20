"""Buy-and-hold backtest and the three evaluation metrics."""

import numpy as np
import pandas as pd

import config


def buy_and_hold_value(w, prices_test):
    """Buy-and-hold portfolio value with V_0 = 1 and no rebalancing.

    V_t = sum_i w_i * P_{i,t} / P_{i,0}, where P_{i,0} is the price on the
    first test day.

    Input: w, np.ndarray (N,); prices_test, pd.DataFrame (T x N).
    Output: V, pd.Series (T,) of portfolio values indexed by date.
    """
    P = prices_test.to_numpy(dtype=float)
    V = (P / P[0]) @ w
    return pd.Series(V, index=prices_test.index, name="V")


def daily_log_returns(V):
    """Daily portfolio log returns ln(V_t / V_{t-1}).

    Input: V, pd.Series (T,).
    Output: np.ndarray (T-1,).
    """
    v = np.asarray(V, dtype=float)
    return np.log(v[1:] / v[:-1])


def sharpe_ratio(V):
    """Annualized Sharpe ratio with r_f = 0:
    mean(daily log return) / std(daily log return) * sqrt(252).

    Input: V, pd.Series (T,).
    Output: float.
    """
    r = daily_log_returns(V)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(config.TRADING_DAYS))


def max_drawdown(V):
    """Maximum drawdown max_t (1 - V_t / max_{s<=t} V_s), as a positive fraction.

    Input: V, pd.Series (T,).
    Output: float in [0, 1).
    """
    v = np.asarray(V, dtype=float)
    running_max = np.maximum.accumulate(v)
    return float(np.max(1.0 - v / running_max))


def cumulative_return(V):
    """Cumulative return V_T / V_0 - 1.

    Input: V, pd.Series (T,).
    Output: float.
    """
    v = np.asarray(V, dtype=float)
    return float(v[-1] / v[0] - 1.0)
