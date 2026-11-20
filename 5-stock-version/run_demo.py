"""Run the full demo end to end and write results/."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
import config
import data
import estimate
import strategies

COV_LABELS = {"sample": "Sample", "ledoit_wolf": "Ledoit-Wolf"}


def compute_weights(mu_hat, Sigma_hat, Sigma_lw):
    """Compute the weights of every (strategy, covariance estimator) pair.

    Input: mu_hat, np.ndarray (N,); Sigma_hat, Sigma_lw, np.ndarray (N x N).
    Output: dict mapping (strategy, estimator) -> w, np.ndarray (N,).
            1/N appears once with estimator "none".
    """
    n = len(mu_hat)
    weights = {("1/N", "none"): strategies.equal_weight(n)}
    for est, Sigma in (("sample", Sigma_hat), ("ledoit_wolf", Sigma_lw)):
        weights[("GMVP", est)] = strategies.gmvp(Sigma)
        weights[("Markowitz", est)] = strategies.markowitz(mu_hat, Sigma, config.LAM_COMPARE)
    return weights


def plot_frontier(mu_hat, Sigma_hat, Sigma_lw, weights, path):
    """Plot the in-sample efficient frontier for both covariance estimators,
    marking GMVP, 1/N and the comparison-lam Markowitz portfolio.

    Input: mu_hat, np.ndarray (N,); Sigma_hat, Sigma_lw, np.ndarray (N x N);
           weights, dict from compute_weights; path, str output file.
    Output: none (writes a PNG).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {"sample": ("tab:blue", "-"), "ledoit_wolf": ("tab:orange", "--")}
    for est, Sigma in (("sample", Sigma_hat), ("ledoit_wolf", Sigma_lw)):
        color, ls = styles[est]
        rets, vols = strategies.efficient_frontier(mu_hat, Sigma, config.LAM_GRID)
        ax.plot(vols, rets, ls, color=color, label="Frontier (" + COV_LABELS[est] + ")")
        r, v = strategies.portfolio_stats(weights[("GMVP", est)], mu_hat, Sigma)
        ax.scatter(v, r, marker="s", s=70, color=color, edgecolor="k", zorder=3,
                   label="GMVP (" + COV_LABELS[est] + ")")
        r, v = strategies.portfolio_stats(weights[("Markowitz", est)], mu_hat, Sigma)
        ax.scatter(v, r, marker="^", s=80, color=color, edgecolor="k", zorder=3,
                   label="Markowitz lam=%g (%s)" % (config.LAM_COMPARE, COV_LABELS[est]))
        r, v = strategies.portfolio_stats(weights[("1/N", "none")], mu_hat, Sigma)
        ax.scatter(v, r, marker="o", s=70, color=color, edgecolor="k", zorder=3,
                   label="1/N (" + COV_LABELS[est] + " vol)")
    ax.set_xlabel("Annualized volatility (in-sample)")
    ax.set_ylabel("Annualized expected log return (in-sample)")
    ax.set_title("Long-only efficient frontier, training period")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_test_values(values, path):
    """Plot test-period value curves V_t of all strategies.

    Input: values, dict mapping (strategy, estimator) -> V, pd.Series;
           path, str output file.
    Output: none (writes a PNG).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for (strategy, est), V in values.items():
        label = strategy if est == "none" else strategy + " (" + COV_LABELS[est] + ")"
        ls = "--" if est == "ledoit_wolf" else "-"
        ax.plot(V.index, V.to_numpy(), ls, label=label)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value V_t (V_0 = 1)")
    ax.set_title("Out-of-sample buy-and-hold, test period")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


STRATEGY_COLORS = {"1/N": "#2a78d6", "GMVP": "#eb6834", "Markowitz": "#1baf7a"}


def plot_metrics(metrics_df, path):
    """Bar charts of the three test-period metrics, one panel per metric.

    Bars are colored by strategy; Ledoit-Wolf bars are hatched to separate
    them from sample-covariance bars.

    Input: metrics_df, pd.DataFrame with columns strategy, covariance,
           sharpe_ratio, max_drawdown, cumulative_return; path, str output file.
    Output: none (writes a PNG).
    """
    panels = [
        ("sharpe_ratio", "Sharpe ratio (annualized, r_f = 0)", "%.2f", 1.0),
        ("max_drawdown", "Maximum drawdown (lower is better)", "%.1f%%", 100.0),
        ("cumulative_return", "Cumulative return", "%.1f%%", 100.0),
    ]
    order = {"1/N": 0, "GMVP": 1, "Markowitz": 2}
    cov_order = {"none": 0, "sample": 1, "ledoit_wolf": 2}
    metrics_df = metrics_df.assign(
        _o=metrics_df["strategy"].map(order),
        _c=metrics_df["covariance"].map(cov_order),
    ).sort_values(["_o", "_c"]).reset_index(drop=True)
    labels = []
    for strategy, est in zip(metrics_df["strategy"], metrics_df["covariance"]):
        labels.append(strategy if est == "none" else strategy + "\n(" + COV_LABELS[est] + ")")
    colors = [STRATEGY_COLORS[s] for s in metrics_df["strategy"]]
    hatches = ["///" if est == "ledoit_wolf" else "" for est in metrics_df["covariance"]]
    x = np.arange(len(metrics_df))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title, fmt, scale) in zip(axes, panels):
        values = metrics_df[col].to_numpy() * scale
        bars = ax.bar(x, values, width=0.7, color=colors, edgecolor="white", linewidth=2)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for xi, v in zip(x, values):
            ax.text(xi, v, fmt % v, ha="center", va="bottom", fontsize=9, color="#0b0b0b")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, values.max() * 1.15)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if scale == 100.0:
            ax.set_ylabel("%")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STRATEGY_COLORS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#52514e", hatch="///"))
    fig.legend(handles, list(STRATEGY_COLORS.keys()) + ["Ledoit-Wolf covariance"],
               loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Test-period metrics (2024-07-02 to 2025-12-31)", y=0.98, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Load data, estimate, build portfolios, backtest, write outputs, print summary.

    Input: none.
    Output: none.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    prices = data.load_prices()
    prices_train, prices_test = data.split_train_test(prices)
    returns_train = data.log_returns(prices_train)

    mu_hat = estimate.estimate_mu(returns_train)
    Sigma_hat = estimate.estimate_sigma_sample(returns_train)
    Sigma_lw = estimate.estimate_sigma_lw(returns_train)

    weights = compute_weights(mu_hat, Sigma_hat, Sigma_lw)

    weight_rows = []
    metric_rows = []
    values = {}
    for (strategy, est), w in weights.items():
        weight_rows.append(dict({"strategy": strategy, "covariance": est},
                                **dict(zip(config.TICKERS, w))))
        V = backtest.buy_and_hold_value(w, prices_test)
        values[(strategy, est)] = V
        metric_rows.append({
            "strategy": strategy,
            "covariance": est,
            "sharpe_ratio": backtest.sharpe_ratio(V),
            "max_drawdown": backtest.max_drawdown(V),
            "cumulative_return": backtest.cumulative_return(V),
        })

    weights_df = pd.DataFrame(weight_rows)
    metrics_df = pd.DataFrame(metric_rows)
    weights_df.to_csv(os.path.join(config.RESULTS_DIR, "weights.csv"), index=False)
    metrics_df.to_csv(os.path.join(config.RESULTS_DIR, "metrics.csv"), index=False)

    plot_frontier(mu_hat, Sigma_hat, Sigma_lw, weights,
                  os.path.join(config.RESULTS_DIR, "frontier.png"))
    plot_test_values(values, os.path.join(config.RESULTS_DIR, "test_value.png"))
    plot_metrics(metrics_df, os.path.join(config.RESULTS_DIR, "metrics.png"))

    print("Train: %s to %s (%d days) | Test: %s to %s (%d days)" % (
        prices_train.index[0].date(), prices_train.index[-1].date(), len(prices_train),
        prices_test.index[0].date(), prices_test.index[-1].date(), len(prices_test)))
    print("Comparison lam = %g" % config.LAM_COMPARE)
    print()
    print("Weights:")
    print(weights_df.round(3).to_string(index=False))
    print()
    print("Test-period metrics (annualized Sharpe, r_f = 0):")
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
