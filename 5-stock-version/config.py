"""Configuration constants for the 5-stock portfolio optimization demo."""

import numpy as np

# Five large-cap S&P 500 members, one per distinct GICS sector.
# All have complete daily price histories over the full date range.
TICKERS = ["AAPL", "JPM", "JNJ", "XOM", "PG"]
SECTORS = {
    "AAPL": "Information Technology",
    "JPM": "Financials",
    "JNJ": "Health Care",
    "XOM": "Energy",
    "PG": "Consumer Staples",
}

# Date range (inclusive start, end); yfinance treats `end` as exclusive,
# so data.py passes the day after END_DATE.
START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

# Chronological train/test split: first 70% of trading days = training.
TRAIN_FRACTION = 0.70

# Annualization factor.
TRADING_DAYS = 252

# Lambda grid for the efficient frontier (log-spaced, lam > 0).
LAM_GRID = np.logspace(-1, 3, 50)

# Comparison lambda for the strategy comparison.
# Chosen so that the Markowitz weights lie clearly between the highest-mean
# single stock and the GMVP. On the training data, lam <= 2 puts 100% in XOM
# (highest mu_hat), and lam >= 100 is close to the GMVP. At lam = 10 the
# weights are about AAPL 14%, JPM 11%, JNJ 0%, XOM 42%, PG 33% (both
# covariance estimators) -- XOM is roughly halfway between 100% and its GMVP
# weight of ~13%.
LAM_COMPARE = 10.0

# Paths.
DATA_DIR = "data"
PRICES_PATH = "data/prices.csv"
RESULTS_DIR = "results"
