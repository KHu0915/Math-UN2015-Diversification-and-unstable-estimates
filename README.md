# Portfolio Management Optimization — 5-stock demo (MATH UN2015, Group 7)

Out-of-sample comparison of three long-only allocation strategies — **Markowitz mean-variance**, **Global Minimum Variance (GMVP)** and **equal weighting (1/N)** — on five S&P 500 stocks, using a single 70/30 chronological train/test split. Each strategy is evaluated by Sharpe ratio, maximum drawdown and cumulative return. The one varied ingredient is the covariance estimator: sample covariance vs. Ledoit-Wolf shrinkage.

## How to run

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify.py                 # must print PASS for all checks
python run_demo.py
```

The first run downloads prices with `yfinance` and caches them to `data/prices.csv`; later runs read the cache.

## Data

| Ticker | Company | GICS sector |
|---|---|---|
| AAPL | Apple | Information Technology |
| JPM | JPMorgan Chase | Financials |
| JNJ | Johnson & Johnson | Health Care |
| XOM | Exxon Mobil | Energy |
| PG | Procter & Gamble | Consumer Staples |

- Adjusted close prices, 2021-01-01 to 2025-12-31 (1255 trading days, no gaps).
- Daily log returns, annualized with 252 trading days.
- Training: 2021-01-04 to 2024-07-01 (first 70%, 878 days). Test: 2024-07-02 to 2025-12-31 (377 days).

## Comparison λ

**λ = 10.**

On the training data, the highest-mean stock is XOM (annualized μ̂ ≈ 0.34). For λ ≤ 2 the Markowitz portfolio is 100% XOM; for λ ≥ 100 it is nearly the GMVP (XOM ≈ 13%, JNJ ≈ 40%, PG ≈ 32%). At λ = 10 the weights are about AAPL 14%, JPM 11%, JNJ 0%, XOM 42%, PG 33% for both covariance estimators — XOM sits roughly halfway between 100% and its GMVP weight, so the portfolio is clearly between the two extremes, not a single stock and not the GMVP.

## Verification (`verify.py`)

1. cvxpy GMVP without `w >= 0` matches the closed form `Σ⁻¹1 / (1ᵀΣ⁻¹1)`.
2. Every weight vector produced satisfies `sum(w) = 1`, `w >= 0`.
3. Σ̂ and Σ_LW are positive semidefinite (minimum eigenvalue ≥ −tolerance).
4. Two-asset prospectus example reproduced: GMVP weight on asset 1 = 24.14%, GMVP vol = 8.14%, 1/N vol = 10.25%, GMVP expected return = 7.45%.

## Outputs (`results/`)

- `metrics.csv` — Sharpe ratio (r_f = 0), max drawdown, cumulative return per (strategy, covariance estimator); 1/N once.
- `weights.csv` — weights of every strategy for each estimator.
- `frontier.png` — in-sample efficient frontier (sample and Ledoit-Wolf), with GMVP, 1/N and Markowitz λ = 10 marked.
- `test_value.png` — test-period buy-and-hold value curves.
- `metrics.png` — bar charts of Sharpe ratio, max drawdown and cumulative return for every strategy.
