"""The four verification checks; prints PASS/FAIL for each."""

import sys

import numpy as np

import config
import data
import estimate
import strategies

TOL = 1e-6


def report(name, passed, detail=""):
    """Print a PASS/FAIL line.

    Input: name, str; passed, bool; detail, str (optional extra info).
    Output: passed, bool (unchanged).
    """
    status = "PASS" if passed else "FAIL"
    print("[" + status + "] " + name + ((": " + detail) if detail else ""))
    return passed


def load_estimates():
    """Load cached prices and compute training estimates.

    Input: none.
    Output: (mu_hat, Sigma_hat, Sigma_lw); np.ndarray (N,), (N x N), (N x N).
    """
    prices = data.load_prices()
    prices_train, _ = data.split_train_test(prices)
    returns_train = data.log_returns(prices_train)
    mu_hat = estimate.estimate_mu(returns_train)
    Sigma_hat = estimate.estimate_sigma_sample(returns_train)
    Sigma_lw = estimate.estimate_sigma_lw(returns_train)
    return mu_hat, Sigma_hat, Sigma_lw


def closed_form_gmvp(Sigma):
    """Unconstrained-sign GMVP closed form w = Sigma^{-1} 1 / (1^T Sigma^{-1} 1).

    Input: Sigma, np.ndarray (N x N), invertible.
    Output: w, np.ndarray (N,).
    """
    ones = np.ones(Sigma.shape[0])
    x = np.linalg.solve(Sigma, ones)
    return x / (ones @ x)


def check_closed_form(Sigma_hat, Sigma_lw):
    """Check 1: cvxpy GMVP without w>=0 equals the closed form.

    Input: Sigma_hat, Sigma_lw, np.ndarray (N x N).
    Output: bool.
    """
    ok = True
    for label, Sigma in (("Sigma_hat", Sigma_hat), ("Sigma_lw", Sigma_lw)):
        w_cvx = strategies.gmvp(Sigma, long_only=False)
        w_cf = closed_form_gmvp(Sigma)
        err = float(np.max(np.abs(w_cvx - w_cf)))
        ok = report("1. Closed-form GMVP (" + label + ")", err < 1e-5,
                    "max |w_cvxpy - w_closed| = %.2e" % err) and ok
    return ok


def check_constraints(mu_hat, Sigma_hat, Sigma_lw):
    """Check 2: every produced weight vector has sum(w)=1 and w>=0.

    Input: mu_hat, np.ndarray (N,); Sigma_hat, Sigma_lw, np.ndarray (N x N).
    Output: bool.
    """
    n = len(mu_hat)
    weights = {"1/N": strategies.equal_weight(n)}
    for label, Sigma in (("sample", Sigma_hat), ("ledoit_wolf", Sigma_lw)):
        weights["GMVP/" + label] = strategies.gmvp(Sigma)
        weights["Markowitz/" + label] = strategies.markowitz(mu_hat, Sigma, config.LAM_COMPARE)
        for lam in config.LAM_GRID:
            weights["Frontier(lam=%.3g)/%s" % (lam, label)] = strategies.markowitz(mu_hat, Sigma, lam)
    bad = []
    for name, w in weights.items():
        if abs(w.sum() - 1.0) > TOL or w.min() < -TOL:
            bad.append(name)
    detail = "%d weight vectors checked" % len(weights)
    if bad:
        detail += "; violations: " + ", ".join(bad)
    return report("2. Constraints sum(w)=1, w>=0", not bad, detail)


def check_psd(Sigma_hat, Sigma_lw):
    """Check 3: Sigma_hat and Sigma_lw are positive semidefinite.

    Input: Sigma_hat, Sigma_lw, np.ndarray (N x N).
    Output: bool.
    """
    ok = True
    for label, Sigma in (("Sigma_hat", Sigma_hat), ("Sigma_lw", Sigma_lw)):
        eig_min = float(np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T)).min())
        ok = report("3. PSD (" + label + ")", eig_min >= -TOL,
                    "min eigenvalue = %.3e" % eig_min) and ok
    return ok


def check_two_asset():
    """Check 4: reproduce the two-asset example from the prospectus.

    mu = (0.12, 0.06), sigma = (0.20, 0.10), rho = -0.20.
    Expected: GMVP weight on asset 1 = 24.14%, GMVP vol = 8.14%,
              equal-weight vol = 10.25%, GMVP expected return = 7.45%.

    Input: none.
    Output: bool.
    """
    mu = np.array([0.12, 0.06])
    sigma = np.array([0.20, 0.10])
    rho = -0.20
    Sigma = np.array([[sigma[0] ** 2, rho * sigma[0] * sigma[1]],
                      [rho * sigma[0] * sigma[1], sigma[1] ** 2]])
    w = strategies.gmvp(Sigma)
    ret, vol = strategies.portfolio_stats(w, mu, Sigma)
    _, vol_eq = strategies.portfolio_stats(strategies.equal_weight(2), mu, Sigma)
    got = np.round(100 * np.array([w[0], vol, vol_eq, ret]), 2)
    expected = np.array([24.14, 8.14, 10.25, 7.45])
    detail = "w1=%.2f%%, vol=%.2f%%, vol_1/N=%.2f%%, E[r]=%.2f%%" % tuple(got)
    return report("4. Two-asset prospectus example", bool(np.allclose(got, expected, atol=1e-9)), detail)


def main():
    """Run all four checks; exit with status 1 if any fails.

    Input: none.
    Output: none.
    """
    mu_hat, Sigma_hat, Sigma_lw = load_estimates()
    results = [
        check_closed_form(Sigma_hat, Sigma_lw),
        check_constraints(mu_hat, Sigma_hat, Sigma_lw),
        check_psd(Sigma_hat, Sigma_lw),
        check_two_asset(),
    ]
    if all(results):
        print("All checks passed.")
    else:
        print("Some checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
