"""Daily event study: abnormal returns, CARs and abnormal volume around each keynote.

Conventions
-----------
* ``t0`` is the first trading session whose close is after the keynote starts, so a
  keynote during market hours has t0 = that day and an evening keynote has t0 = next day.
* Normal returns are estimated over trading days [-250, -30] relative to t0 (>= 120 obs).
* SCAR = CAR / (sigma_eps * sqrt(L)), the standardisation used by Boehmer, Musumeci and
  Poulsen (1991) without the (small, n~220) forecast-error correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    COMPANIES, ESTIMATION_WINDOW, EVENT_WINDOWS, MIN_ESTIMATION_OBS, PATH_WINDOW,
)
from .market import stock_panel

MODELS = ("market", "market_adj", "mean", "ff3")


def locate_t0(close_utc: pd.Series, start_utc: pd.Timestamp) -> int | None:
    """Position of the first session closing after ``start_utc`` (None if beyond the data)."""
    pos = int(pd.DatetimeIndex(close_utc).searchsorted(start_utc, side="right"))
    return pos if pos < len(close_utc) else None


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = np.sqrt(resid @ resid / (len(y) - X.shape[1]))
    return coef, float(sigma)


def abnormal_returns(panel: pd.DataFrame, pos: int, model: str = "market",
                     est: tuple[int, int] = ESTIMATION_WINDOW,
                     path: tuple[int, int] = PATH_WINDOW) -> dict | None:
    """ARs for relative days path[0]..path[1] (NaN where data run out) plus model diagnostics."""
    lo, hi = pos + est[0], pos + est[1]
    if lo < 0 or pos + path[0] < 0:
        return None
    est_df = panel.iloc[lo : hi + 1]
    rel = np.arange(path[0], path[1] + 1)
    idx = pos + rel
    ok = idx < len(panel)
    ev_df = panel.iloc[idx[ok]]

    r_est, r_ev = est_df["ret"].to_numpy(), ev_df["ret"].to_numpy()
    if model == "market":
        X_est = np.column_stack([np.ones(len(est_df)), est_df["mkt"].to_numpy()])
        X_ev = np.column_stack([np.ones(len(ev_df)), ev_df["mkt"].to_numpy()])
        y_est, y_ev = r_est, r_ev
    elif model == "ff3":
        if "mkt_rf" not in panel:
            return None
        cols = ["mkt_rf", "smb", "hml"]
        X_est = np.column_stack([np.ones(len(est_df)), est_df[cols].to_numpy()])
        X_ev = np.column_stack([np.ones(len(ev_df)), ev_df[cols].to_numpy()])
        y_est = r_est - est_df["rf"].to_numpy()
        y_ev = r_ev - ev_df["rf"].to_numpy()
    elif model == "market_adj":
        X_est = X_ev = None
        y_est = r_est - est_df["mkt"].to_numpy()
        y_ev = r_ev - ev_df["mkt"].to_numpy()
    elif model == "mean":
        X_est = X_ev = None
        y_est, y_ev = r_est, r_ev
    else:
        raise ValueError(f"unknown model {model}")

    if X_est is not None:
        good = np.isfinite(y_est) & np.isfinite(X_est).all(axis=1)
        if good.sum() < MIN_ESTIMATION_OBS:
            return None
        coef, sigma = _ols(y_est[good], X_est[good])
        ar_ev = y_ev - X_ev @ coef
        resid_est = y_est[good] - X_est[good] @ coef
    else:
        good = np.isfinite(y_est)
        if good.sum() < MIN_ESTIMATION_OBS:
            return None
        mu = 0.0 if model == "market_adj" else y_est[good].mean()
        coef = np.array([mu])
        resid_est = y_est[good] - mu
        sigma = float(resid_est.std(ddof=1))
        ar_ev = y_ev - mu

    ar = np.full(len(rel), np.nan)
    ar[ok] = ar_ev
    return {
        "ar": pd.Series(ar, index=rel),
        "sigma": sigma,
        "alpha": float(coef[0]),
        "beta": float(coef[1]) if len(coef) > 1 else np.nan,
        "n_est": int(good.sum()),
        "est_pos_frac": float((resid_est > 0).mean()),
    }


def car(ar: pd.Series, window: tuple[int, int]) -> float:
    vals = ar.loc[window[0] : window[1]]
    return float(vals.sum()) if vals.notna().all() and len(vals) else np.nan


def abnormal_volume(panel: pd.DataFrame, pos: int, window=(0, 1),
                    est: tuple[int, int] = ESTIMATION_WINDOW) -> tuple[float, float]:
    """Mean log-volume deviation from the estimation window, raw and standardised."""
    lv = panel["logvol"].to_numpy()
    base = lv[pos + est[0] : pos + est[1] + 1]
    base = base[np.isfinite(base) & (base > 0)]
    lo, hi = pos + window[0], pos + window[1]
    if hi >= len(lv) or len(base) < MIN_ESTIMATION_OBS:
        return np.nan, np.nan
    dev = lv[lo : hi + 1].mean() - base.mean()
    sd = base.std(ddof=1)
    return float(dev), float(dev / sd) if sd > 0 else np.nan


def run_event_study(events: pd.DataFrame, models=MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (event-level table, long table of daily ARs for the primary market model)."""
    panels = {slug: stock_panel(c) for slug, c in COMPANIES.items()
              if slug in set(events["company"])}
    rows, paths = [], []
    for ev in events.itertuples():
        panel = panels[ev.company]
        pos = locate_t0(panel["close_utc"], ev.start_utc)
        row = {"event_id": ev.event_id}
        if pos is None:
            rows.append(row)
            continue
        row["t0"] = panel.index[pos]
        row["days_to_t0"] = (panel.index[pos] - ev.start_utc.tz_convert(None).normalize()).days
        for model in models:
            res = abnormal_returns(panel, pos, model)
            if res is None:
                continue
            for name, win in EVENT_WINDOWS.items():
                c = car(res["ar"], win)
                row[f"{name}_{model}"] = c
                row[f"s{name}_{model}"] = c / (res["sigma"] * np.sqrt(win[1] - win[0] + 1))
            if model == "market":
                row.update(sigma=res["sigma"], alpha=res["alpha"], beta=res["beta"],
                           n_est=res["n_est"], est_pos_frac=res["est_pos_frac"])
                paths.append(pd.DataFrame({"event_id": ev.event_id, "rel_day": res["ar"].index,
                                           "ar": res["ar"].to_numpy(),
                                           "sar": res["ar"].to_numpy() / res["sigma"]}))
        row["abn_logvol_0_1"], row["sav_0_1"] = abnormal_volume(panel, pos)
        row["ret_0_1_raw"] = float(np.prod(1 + panel["ret"].iloc[pos : pos + 2]) - 1)
        row["mkt_0_1_raw"] = float(np.prod(1 + panel["mkt"].iloc[pos : pos + 2]) - 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    path_df = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    return table, path_df
