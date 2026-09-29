"""Cross-sectional analysis: do louder keynotes come with different abnormal returns?

Implements docs/analysis_plan.md. Every function returns plain DataFrames / dicts so the
notebooks and the report can reuse them.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from .catalog import load_events
from .config import COMPANIES, ESTIMATION_WINDOW, PROCESSED, SQL_DIR
from .event_study import abnormal_returns, car
from .market import load_prices, stock_panel
from .stats import (
    benjamini_hochberg,
    caar_tests,
    design,
    permutation_pvalue,
    wild_cluster_bootstrap,
)

FEATURES_FILE = PROCESSED / "applause_features.parquet"
STUDY_FILE = PROCESSED / "event_study.parquet"
PATHS_FILE = PROCESSED / "ar_paths.parquet"

PRIMARY_Y = "scar_0_1_market"
PRIMARY_X = "z_applause_share"
Z_MEASURES = ["z_applause_share", "z_applause_per_min", "z_applause_max", "z_clap_mass",
              "z_cheer_share"]
MEASURE_LABELS = {
    "z_applause_share": "Applause share of keynote time",
    "z_applause_per_min": "Ovations per minute",
    "z_applause_max": "Longest ovation (log seconds)",
    "z_clap_mass": "Mean clap probability",
    "z_cheer_share": "Cheering share",
    "hype": "Hype index (1st principal component)",
}
OUTCOME_LABELS = {
    "scar_0_0_market": "SCAR[0,0]",
    "scar_m1_1_market": "SCAR[-1,+1]",
    "scar_0_5_market": "SCAR[0,+5]",
    "scar_2_20_market": "SCAR[+2,+20]",
    "abs_scar_0_1": "|SCAR[0,+1]|",
    "sav_0_1": "Abnormal log volume [0,+1] (std.)",
}


# ----------------------------------------------------------------------------- panel

def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all keynotes, live-audience analysis sample) built with DuckDB (sql/build_panel.sql)."""
    con = duckdb.connect()
    events = load_events()
    events["date_local"] = events["date_local"].dt.date
    vix = load_prices("^VIX")[["close"]].rename(columns={"close": "vix_close"}).reset_index()
    con.register("events", events)
    con.register("features", pd.read_parquet(FEATURES_FILE))
    con.register("study", pd.read_parquet(STUDY_FILE))
    con.register("vix", vix)
    con.execute((SQL_DIR / "build_panel.sql").read_text())
    panel = con.execute("SELECT * FROM panel ORDER BY start_utc").df()
    live = con.execute("SELECT * FROM live ORDER BY start_utc").df()
    for df in (panel, live):
        df["abs_scar_0_1"] = df[PRIMARY_Y].abs()
        df["log_vix"] = np.log(df["vix_prev"])
        df["log_keynote_min"] = np.log(df["keynote_min"])
        df["official"] = df["official_channel"].astype(float)
        df["is_launch"] = (df["event_type"] == "product_launch").astype(float)
    live = add_hype_index(live)
    return panel, live


def add_hype_index(live: pd.DataFrame) -> pd.DataFrame:
    """First principal component of the within-company z-scored applause measures."""
    Z = live[Z_MEASURES].fillna(0.0).to_numpy()
    Zc = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    w = vt[0] if vt[0][0] >= 0 else -vt[0]
    pc1 = Zc @ w
    out = live.copy()
    out["hype"] = (pc1 - pc1.mean()) / pc1.std(ddof=1)
    out.attrs["hype_loadings"] = dict(zip(Z_MEASURES, w.round(3).tolist(), strict=True))
    out.attrs["hype_explained"] = float(s[0] ** 2 / (s**2).sum())
    return out


# ------------------------------------------------------------------------ regressions

def ols_fe(df: pd.DataFrame, y: str, x: str, controls: list[str] | None = None,
           fe: str | None = "company", n_boot: int = 9_999, seed: int = 0) -> dict:
    controls = controls or []
    d = df.dropna(subset=[y, x, *controls])
    X, _ = design(d, x, controls, fe)
    yv = d[y].to_numpy(float)
    fit = sm.OLS(yv, X).fit(cov_type="HC3")
    wcb = wild_cluster_bootstrap(yv, X, d["company"].to_numpy(), 0, n_boot, seed)
    ci = np.asarray(fit.conf_int())[0]
    return {
        "beta": float(fit.params[0]), "se_hc3": float(fit.bse[0]),
        "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "p_hc3": float(fit.pvalues[0]),
        "t_cr1": wcb["t_cr1"], "p_wcb": wcb["p_wcb"],
        "n": len(d), "n_clusters": wcb["n_clusters"],
    }


def primary(live: pd.DataFrame) -> dict:
    return ols_fe(live, PRIMARY_Y, PRIMARY_X)


def secondary_outcomes(live: pd.DataFrame) -> pd.DataFrame:
    rows = [{"outcome": OUTCOME_LABELS[y], "column": y, **ols_fe(live, y, PRIMARY_X)}
            for y in OUTCOME_LABELS]
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_wcb"].to_numpy())
    return out


def secondary_measures(live: pd.DataFrame) -> pd.DataFrame:
    rows = [{"measure": MEASURE_LABELS[x], "column": x, **ols_fe(live, PRIMARY_Y, x)}
            for x in MEASURE_LABELS if x != PRIMARY_X]
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_wcb"].to_numpy())
    return out


def robustness(live: pd.DataFrame) -> pd.DataFrame:
    specs: list[tuple[str, str, dict]] = [
        ("Normal-return model", "Primary: market model (S&P 500 / KOSPI)", {}),
        ("Normal-return model", "Market-adjusted (beta = 1)", {"y": "scar_0_1_market_adj"}),
        ("Normal-return model", "Constant mean", {"y": "scar_0_1_mean"}),
        ("Normal-return model", "Nasdaq-100 as market", {"y": "scar_0_1_market_ndx"}),
        ("Normal-return model", "Fama-French 3-factor (US listings)", {"y": "scar_0_1_ff3"}),
        ("Controls", "+ pre-event run-up CAR[-10,-1]", {"controls": ["car_m10_m1_market"]}),
        ("Controls", "+ log VIX on t0-1", {"controls": ["log_vix"]}),
        ("Controls", "+ log keynote length", {"controls": ["log_keynote_min"]}),
        ("Controls", "+ official-channel video", {"controls": ["official"]}),
        ("Controls", "+ product-launch dummy", {"controls": ["is_launch"]}),
        ("Controls", "All controls", {"controls": ["car_m10_m1_market", "log_vix",
                                                   "log_keynote_min", "official", "is_launch"]}),
        ("Sample", "Without Tesla", {"sample": lambda d: d[d.company != "tesla"]}),
        ("Sample", "Without Samsung", {"sample": lambda d: d[d.company != "samsung"]}),
        ("Sample", "Product launches only", {"sample": lambda d: d[d.is_launch == 1]}),
        ("Sample", "Official-channel videos only", {"sample": lambda d: d[d.official == 1]}),
        ("Exploratory", "Without earnings releases within +-1 session",
         {"sample": lambda d: d[~d.earnings_near.astype(bool)]}),
        ("Exploratory", "Without low-confidence start times",
         {"sample": lambda d: d[d.start_time_confidence != "low"]}),
    ]
    rows = []
    for group, label, spec in specs:
        d = spec.get("sample", lambda d: d)(live)
        y = spec.get("y", PRIMARY_Y)
        rows.append({"group": group, "spec": label,
                     **ols_fe(d, y, PRIMARY_X, spec.get("controls"))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- non-parametric

def spearman_within(live: pd.DataFrame, x: str = PRIMARY_X, y: str = PRIMARY_Y) -> dict:
    d = live.dropna(subset=[x, y])
    rho, p = permutation_pvalue(d[x].to_numpy(), d[y].to_numpy(), d["company"].to_numpy())
    return {"rho": rho, "p_perm": p, "n": len(d)}


def terciles(live: pd.DataFrame, x: str = PRIMARY_X, y: str = PRIMARY_Y,
             n_boot: int = 10_000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    d = live.dropna(subset=[x, y]).copy()
    pct = d.groupby("company")[x].rank(pct=True, method="first")
    d["tercile"] = np.ceil(pct * 3).clip(1, 3).astype(int)
    table = d.groupby("tercile")[y].agg(["mean", "std", "count"]).reset_index()
    table["se"] = table["std"] / np.sqrt(table["count"])
    top, bottom = d.loc[d.tercile == 3, y].to_numpy(), d.loc[d.tercile == 1, y].to_numpy()
    diff = top.mean() - bottom.mean()
    rng = np.random.default_rng(seed)
    boots = (rng.choice(top, (n_boot, len(top))).mean(1)
             - rng.choice(bottom, (n_boot, len(bottom))).mean(1))
    welch = sps.ttest_ind(top, bottom, equal_var=False)
    return table, {"diff": diff, "ci_lo": float(np.percentile(boots, 2.5)),
                   "ci_hi": float(np.percentile(boots, 97.5)), "p_welch": float(welch.pvalue)}


# ----------------------------------------------------------------------------- placebo

def placebo(live: pd.DataFrame, panel: pd.DataFrame, n_draws: int = 1_000, seed: int = 0,
            exclusion: int = 10) -> dict:
    """Move each keynote's applause to a random non-event day of the same company."""
    rng = np.random.default_rng(seed)
    pools: dict[str, np.ndarray] = {}
    for slug in live["company"].unique():
        sp = stock_panel(COMPANIES[slug])
        event_pos = sp.index.get_indexer(pd.to_datetime(panel.loc[panel.company == slug, "t0"].dropna()))
        banned = np.zeros(len(sp), bool)
        for p in event_pos[event_pos >= 0]:
            banned[max(p - exclusion, 0) : p + exclusion + 1] = True
        first = -ESTIMATION_WINDOW[0]
        values = []
        for pos in range(first, len(sp) - 2):
            if banned[pos]:
                continue
            res = abnormal_returns(sp, pos, "market", path=(0, 1))
            if res is not None:
                c = car(res["ar"], (0, 1))
                if np.isfinite(c):
                    values.append(c / (res["sigma"] * np.sqrt(2)))
        pools[slug] = np.asarray(values)

    d = live.dropna(subset=[PRIMARY_X, PRIMARY_Y]).reset_index(drop=True)
    x_dm = d[PRIMARY_X] - d.groupby("company")[PRIMARY_X].transform("mean")
    x_dm = x_dm.to_numpy()
    Y = np.column_stack([rng.choice(pools[c], n_draws) for c in d["company"]]).T  # (n, draws)
    groups = pd.factorize(d["company"])[0]
    means = np.zeros((groups.max() + 1, n_draws))
    np.add.at(means, groups, Y)
    means /= np.bincount(groups)[:, None]
    Y_dm = Y - means[groups]
    betas = x_dm @ Y_dm / (x_dm @ x_dm)
    real = primary(d)["beta"]
    return {"betas": betas, "beta_real": real,
            "p_placebo": float((np.abs(betas) >= abs(real)).mean()),
            "pool_sizes": {k: len(v) for k, v in pools.items()}}


# ----------------------------------------------------------------- average event effects

def average_effects(panel: pd.DataFrame) -> pd.DataFrame:
    windows = {"[0,0]": "0_0", "[0,+1]": "0_1", "[-1,+1]": "m1_1", "[0,+5]": "0_5",
               "[+2,+20]": "2_20", "[-10,-1]": "m10_m1"}
    groups = {"All keynotes": panel, **{f"Audience: {a}": panel[panel.audience == a]
                                        for a in ("live", "hybrid", "virtual")}}
    rows = []
    for gname, g in groups.items():
        for wname, suffix in windows.items():
            res = caar_tests(g[f"car_{suffix}_market"].to_numpy(float),
                             g[f"scar_{suffix}_market"].to_numpy(float),
                             g["est_pos_frac"].to_numpy(float))
            rows.append({"sample": gname, "window": wname, **res})
    return pd.DataFrame(rows)


def by_company(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slug, g in panel.groupby("company"):
        res = caar_tests(g["car_0_1_market"].to_numpy(float), g["scar_0_1_market"].to_numpy(float),
                         g["est_pos_frac"].to_numpy(float))
        rows.append({"company": COMPANIES[slug].name, **res})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def caar_path(panel: pd.DataFrame, paths: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean cumulative abnormal return from day PATH_WINDOW[0] with 95% CIs."""
    wide = paths.pivot(index="event_id", columns="rel_day", values="ar")
    wide = wide.loc[wide.notna().all(axis=1)]
    cum = wide.cumsum(axis=1)
    keys = panel.set_index("event_id")[by] if by else pd.Series("All", index=cum.index)
    out = []
    for key, idx in cum.groupby(keys.reindex(cum.index)).groups.items():
        c = cum.loc[idx]
        n = len(c)
        mean, se = c.mean(), c.std(ddof=1) / np.sqrt(n)
        out.append(pd.DataFrame({"group": key, "rel_day": c.columns, "caar": mean.to_numpy(),
                                 "lo": (mean - 1.96 * se).to_numpy(),
                                 "hi": (mean + 1.96 * se).to_numpy(), "n": n}))
    return pd.concat(out, ignore_index=True)


def live_vs_virtual(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in [("scar_0_1_market", "SCAR[0,+1]"), ("abs_scar_0_1", "|SCAR[0,+1]|"),
                       ("sav_0_1", "Abnormal log volume [0,+1] (std.)")]:
        a = panel.loc[panel.audience == "live", col].dropna()
        b = panel.loc[panel.audience == "virtual", col].dropna()
        t = sps.ttest_ind(a, b, equal_var=False)
        rows.append({"outcome": label, "live_mean": a.mean(), "virtual_mean": b.mean(),
                     "diff": a.mean() - b.mean(), "p_welch": float(t.pvalue),
                     "n_live": len(a), "n_virtual": len(b)})
    return pd.DataFrame(rows)



# ------------------------------------------------------------------------------ driver

def run_all(n_boot: int = 9_999, n_placebo: int = 1_000) -> dict:
    """Run every pre-registered analysis, write tables to reports/tables, return a summary."""
    from .config import TABLES

    TABLES.mkdir(parents=True, exist_ok=True)
    panel, live = load_panel()
    paths = pd.read_parquet(PATHS_FILE)

    prim = ols_fe(live, PRIMARY_Y, PRIMARY_X, n_boot=n_boot)
    sec_out = secondary_outcomes(live)
    sec_meas = secondary_measures(live)
    rob = robustness(live)
    spear = spearman_within(live)
    terc_table, terc = terciles(live)
    plac = placebo(live, panel, n_draws=n_placebo)
    avg = average_effects(panel)
    comp = by_company(panel)
    lvv = live_vs_virtual(panel)
    path_all = caar_path(panel, paths)
    path_aud = caar_path(panel, paths, by="audience")
    path_terc = caar_path(live.assign(tercile=np.ceil(
        live.groupby("company")[PRIMARY_X].rank(pct=True, method="first") * 3).clip(1, 3)
        .astype("Int64").astype(str)), paths, by="tercile")

    for name, df in {
        "secondary_outcomes": sec_out, "secondary_measures": sec_meas, "robustness": rob,
        "terciles": terc_table, "average_effects": avg, "by_company": comp,
        "live_vs_virtual": lvv, "caar_path_all": path_all, "caar_path_audience": path_aud,
        "caar_path_tercile": path_terc,
    }.items():
        df.to_csv(TABLES / f"{name}.csv", index=False)
    pd.DataFrame({"beta": plac["betas"]}).to_csv(TABLES / "placebo_betas.csv", index=False)
    live.to_csv(TABLES / "analysis_sample.csv", index=False)

    return {
        "n_keynotes": len(panel),
        "n_with_video": int((panel["youtube_id"] != "").sum()),
        "n_live_sample": len(live),
        "n_companies_live": int(live["company"].nunique()),
        "primary": prim,
        "spearman": spear,
        "terciles": terc,
        "placebo": {k: v for k, v in plac.items() if k != "betas"},
        "hype_loadings": live.attrs.get("hype_loadings"),
        "hype_explained": live.attrs.get("hype_explained"),
    }
