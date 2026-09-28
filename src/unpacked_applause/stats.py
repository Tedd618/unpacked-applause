"""Inference helpers: event-study significance tests, wild cluster bootstrap, permutation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def caar_tests(car: np.ndarray, scar: np.ndarray, est_pos_frac: np.ndarray | None = None,
               n_boot: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Cross-sectional t-test, BMP standardised test, generalised sign test, bootstrap CI."""
    ok = np.isfinite(car) & np.isfinite(scar)
    car, scar = car[ok], scar[ok]
    n = len(car)
    out: dict[str, float] = {"n": n}
    if n < 3:
        return out
    mean, sd = car.mean(), car.std(ddof=1)
    t = mean / (sd / np.sqrt(n))
    z = scar.mean() / (scar.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    boots = car[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    out.update(
        mean=mean, median=float(np.median(car)), sd=sd,
        t=t, p_t=2 * stats.t.sf(abs(t), n - 1),
        bmp_t=z, p_bmp=2 * stats.t.sf(abs(z), n - 1),
        ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
        pos_share=float((car > 0).mean()),
    )
    # Generalised sign test (Cowan 1992): null share of positives = estimation-period share.
    p0 = float(np.nanmean(est_pos_frac[ok])) if est_pos_frac is not None else 0.5
    w = (car > 0).sum()
    zs = (w - n * p0) / np.sqrt(n * p0 * (1 - p0))
    out.update(sign_z=zs, p_sign=2 * stats.norm.sf(abs(zs)))
    return out


def _cluster_t(X: np.ndarray, Y: np.ndarray, k: int, onehot: np.ndarray) -> np.ndarray:
    """CR1 t-statistics of coefficient k for every column of Y (vectorised over bootstrap draws)."""
    n, p = X.shape
    g = onehot.shape[0]
    xtx_inv = np.linalg.inv(X.T @ X)
    B = xtx_inv @ X.T @ Y
    U = Y - X @ B
    h = X @ xtx_inv[:, k]  # influence of each observation on beta_k
    scores = onehot @ (h[:, None] * U)  # (G, draws)
    c = g / (g - 1) * (n - 1) / (n - p)
    se = np.sqrt(c * (scores**2).sum(axis=0))
    return B[k] / se


def wild_cluster_bootstrap(y: np.ndarray, X: np.ndarray, clusters: np.ndarray, k: int,
                           n_boot: int = 9_999, seed: int = 0) -> dict[str, float]:
    """Wild cluster restricted (WCR) bootstrap-t p-value for H0: beta_k = 0.

    Cameron, Gelbach & Miller (2008) with Webb (2023) six-point weights, which behave well
    with few clusters (here: nine companies).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    labels, cl = np.unique(clusters, return_inverse=True)
    onehot = np.zeros((len(labels), len(y)))
    onehot[cl, np.arange(len(y))] = 1.0
    t_obs = float(_cluster_t(X, y[:, None], k, onehot)[0])

    Xr = np.delete(X, k, axis=1)
    beta_r, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    fitted, resid = Xr @ beta_r, y - Xr @ beta_r
    rng = np.random.default_rng(seed)
    weights = rng.choice(WEBB_WEIGHTS, size=(len(labels), n_boot))
    Y = fitted[:, None] + resid[:, None] * weights[cl]
    t_boot = _cluster_t(X, Y, k, onehot)
    p = float((np.abs(t_boot) >= abs(t_obs)).mean())
    return {"t_cr1": t_obs, "p_wcb": p, "n_clusters": len(labels)}


def design(df: pd.DataFrame, regressor: str, controls: list[str] | None = None,
           fe: str | None = "company") -> tuple[np.ndarray, list[str]]:
    """Design matrix: [regressor, controls..., intercept or fixed-effect dummies]."""
    cols = [regressor] + (controls or [])
    parts = [df[cols].to_numpy(float)]
    names = list(cols)
    if fe:
        dummies = pd.get_dummies(df[fe], prefix=fe, dtype=float)
        parts.append(dummies.to_numpy())
        names += list(dummies.columns)
    else:
        parts.append(np.ones((len(df), 1)))
        names.append("const")
    return np.column_stack(parts), names


def permutation_pvalue(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                       n_perm: int = 9_999, seed: int = 0) -> tuple[float, float]:
    """Spearman rho and a two-sided permutation p-value (shuffling x within groups)."""
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    groups = np.zeros(len(x), int) if groups is None else pd.factorize(groups)[0]
    rank_y = stats.rankdata(y)
    xs = np.array(x, float)
    count = 0
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for _ in range(n_perm):
        perm = xs.copy()
        for m in members:
            perm[m] = xs[rng.permutation(m)]
        r = np.corrcoef(stats.rankdata(perm), rank_y)[0, 1]
        count += abs(r) >= abs(rho) - 1e-12
    return float(rho), (count + 1) / (n_perm + 1)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out
