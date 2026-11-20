# CLAUDE.md — Portfolio Management Optimization (MATH UN2015, Group 7) — 5-stock demo

## Project context

Course project for MATH UN2015 Linear Algebra & Probability (Direction 5). We compare three long-only allocation strategies out of sample on real S&P 500 stock data:

1. **Markowitz mean-variance optimization** (risk-aversion parameter λ)
2. **Global Minimum Variance Portfolio (GMVP)**
3. **Equal weighting (1/N)** — a control benchmark; comparison against 1/N is only one part of the evaluation

Each strategy is evaluated with three measures: **Sharpe ratio, maximum drawdown, cumulative return**.

This file describes a **small demo**: 5 stocks, one train/test split. Stay strictly within the scope below. Do not add features, analyses or metrics that are not listed here.

## Scope of the demo

### Data
- **5 stocks, one per distinct GICS sector**, large-cap S&P 500 members. You choose them. Every stock must have a complete daily price history over the whole date range (no gaps, no IPO inside the range).
- Record the chosen tickers and their sectors in `config.py`.
- **Date range: 2021-01-01 to 2025-12-31** (5 years of daily data). Keep it as a constant in `config.py`.
- Source: `yfinance`, **adjusted close** prices. Download once and cache to `data/prices.csv`; later runs read the cache.

### Returns and units
- Use **daily log returns**: `l_t = ln(P_t / P_{t-1})`.
- Annualize with **252 trading days**: mean × 252, covariance × 252, volatility × sqrt(252).
- All reported numbers are annualized.

### Train/test split
- **Single chronological split: first 70% of trading days = training, last 30% = testing.** No shuffling.
- All estimates (μ̂, Σ̂) use training data only. The test data is used only for evaluation.

### Estimation
- `mu_hat`: sample mean of training log returns (annualized).
- `Sigma_hat`: sample covariance of training log returns (annualized, divisor T−1).
- `Sigma_lw`: **Ledoit-Wolf shrinkage** covariance (`sklearn.covariance.LedoitWolf`), annualized the same way.
- The covariance estimator (sample vs. Ledoit-Wolf) is the **one variable varied** in this demo. Everything else is held fixed. Run every strategy once with `Sigma_hat` and once with `Sigma_lw`.

### Strategies
Weights `w` are an N×1 vector. Constraints for **all** strategies: `sum(w) = 1`, `w >= 0` (long-only, no short selling).

- **1/N**: `w_i = 1/N` for every i. Uses no estimates.
- **GMVP**: minimize `w^T Σ w` subject to the constraints. Solve with `cvxpy`.
- **Markowitz**: maximize `w^T μ − (λ/2) w^T Σ w` subject to the constraints. Solve with `cvxpy`.
  - **Efficient frontier**: solve over a grid of λ > 0 (log-spaced, e.g. `np.logspace(-1, 3, 50)`); for each λ record in-sample annualized expected return and volatility. Plot the frontier.
  - **Comparison λ**: pick **one** fixed λ for the strategy comparison. Choose a middle value such that the Markowitz weights lie clearly between the highest-mean single stock and the GMVP (not ≈100% one stock, not ≈GMVP). Write down the chosen value and the reason in `config.py`.

### Out-of-sample backtest
- **Buy-and-hold**: on the first test day, invest according to `w`; never rebalance during the test period. Weights drift with prices.
- Portfolio value series (initial value 1): `V_t = sum_i w_i * P_{i,t} / P_{i,0}`, where `P_{i,0}` is stock i's price on the first test day.
- Daily portfolio log return: `ln(V_t / V_{t-1})`.

### Metrics (test period only)
- **Sharpe ratio** (annualized, risk-free rate **r_f = 0**): `mean(daily portfolio log return) / std(daily portfolio log return) * sqrt(252)`.
- **Maximum drawdown**: `max over t of (1 − V_t / max_{s<=t} V_s)`, reported as a positive fraction.
- **Cumulative return**: `V_T / V_0 − 1`.

### Verification (must pass; put in `verify.py`)
1. **Closed-form benchmark**: solve the GMVP **without** the `w >= 0` constraint in `cvxpy`, and compare with the closed form `w = Σ^{-1} 1 / (1^T Σ^{-1} 1)`. They must agree to numerical tolerance.
2. **Constraint checks**: for every weight vector produced, `sum(w) = 1` and `w >= 0` (within tolerance).
3. **Positive semidefiniteness**: confirm `x^T Σ̂ x >= 0` for Σ̂ and Σ_lw (e.g. check all eigenvalues >= −tolerance).
4. **Two-asset example from our prospectus** (hypothetical annual parameters): μ = (0.12, 0.06), σ = (0.20, 0.10), ρ = −0.20. Reproduce: GMVP weight on asset 1 = 24.14%, GMVP volatility = 8.14%, equal-weight volatility = 10.25%, GMVP expected return = 7.45%.

### Outputs
- `results/metrics.csv`: one row per (strategy, covariance estimator) with Sharpe ratio, max drawdown, cumulative return. 1/N appears once (it uses no covariance).
- `results/weights.csv`: weights of every strategy for each covariance estimator.
- `results/frontier.png`: in-sample efficient frontier (sample covariance and Ledoit-Wolf on the same axes), with GMVP, 1/N and the comparison-λ Markowitz portfolio marked.
- `results/test_value.png`: test-period value curves `V_t` of all strategies.
- Print a short summary table to the terminal.

## Out of scope — do NOT implement
- More than 5 stocks, or all 500 S&P 500 stocks.
- Rolling windows, multiple splits, or simulation from a known μ and Σ.
- Eigenvalue / condition-number analysis beyond the PSD check above.
- Inverse-volatility or any other strategy not listed.
- Any metric other than Sharpe ratio, max drawdown, cumulative return.
- Rebalancing during the test period, transaction costs, a non-zero risk-free rate.

## Code layout

```
config.py       # tickers + sectors, date range, split ratio, λ grid, comparison λ, paths
data.py         # download / load cached prices, compute log returns, split train/test
estimate.py     # mu_hat, Sigma_hat, Sigma_lw (annualized)
strategies.py   # equal_weight, gmvp, markowitz (cvxpy), efficient_frontier
backtest.py     # buy-and-hold value series, sharpe, max_drawdown, cumulative_return
verify.py       # the four verification checks; prints PASS/FAIL for each
run_demo.py     # runs everything end to end, writes results/
requirements.txt
```

## Coding rules
- Python 3, `numpy`, `pandas`, `cvxpy`, `scikit-learn`, `matplotlib`, `yfinance`. List them in `requirements.txt`.
- Use numpy for all linear algebra. Use named `def` functions; no lambda expressions.
- Every function gets a short docstring: what goes in (with types/shapes), what comes out.
- Keep variable names consistent with the math above: `w`, `mu_hat`, `Sigma_hat`, `Sigma_lw`, `lam`, `V`.
- Do not rename variables in existing code without asking.
- Shell commands that write into an existing file must append (`>>`), never overwrite (`>`).
- Run `python verify.py` before `python run_demo.py`; if any check fails, fix it before producing results.

## How to run
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify.py
python run_demo.py
```
