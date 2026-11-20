"""Download / load cached prices, compute log returns, split train/test."""

import os

import numpy as np
import pandas as pd

import config


def download_prices():
    """Download adjusted close prices from yfinance.

    Input: none (uses config.TICKERS, config.START_DATE, config.END_DATE).
    Output: pd.DataFrame (T x N) of adjusted close prices, indexed by date,
            columns in config.TICKERS order.
    """
    import yfinance as yf

    end_exclusive = (pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        config.TICKERS,
        start=config.START_DATE,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
    )
    prices = raw["Adj Close"][config.TICKERS]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    return prices


def load_prices():
    """Load prices from the cache, downloading and caching them on first use.

    Input: none.
    Output: pd.DataFrame (T x N) of adjusted close prices without missing values.
    Raises ValueError if any stock has a gap in its price history.
    """
    if os.path.exists(config.PRICES_PATH):
        prices = pd.read_csv(config.PRICES_PATH, index_col="Date", parse_dates=True)
    else:
        prices = download_prices()
        os.makedirs(config.DATA_DIR, exist_ok=True)
        prices.to_csv(config.PRICES_PATH)
    prices = prices[config.TICKERS]
    if prices.isna().any().any():
        missing = prices.isna().sum()
        raise ValueError("Incomplete price history:\n" + str(missing[missing > 0]))
    return prices


def log_returns(prices):
    """Compute daily log returns l_t = ln(P_t / P_{t-1}).

    Input: prices, pd.DataFrame (T x N).
    Output: pd.DataFrame ((T-1) x N) of daily log returns.
    """
    return np.log(prices / prices.shift(1)).iloc[1:]


def split_train_test(prices):
    """Chronological split of trading days: first 70% train, last 30% test.

    Input: prices, pd.DataFrame (T x N).
    Output: (prices_train, prices_test), two pd.DataFrames. The training
            returns are computed from prices_train; the test period starts on
            the first day of prices_test (the buy-and-hold investment day).
    """
    n_train = int(round(config.TRAIN_FRACTION * len(prices)))
    prices_train = prices.iloc[:n_train]
    prices_test = prices.iloc[n_train:]
    return prices_train, prices_test
