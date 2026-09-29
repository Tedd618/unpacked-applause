"""Report figures (static, light surface) following one consistent visual system.

Palette and chrome come from the validated reference palette (see README > Figures):
categorical slots are used in fixed order and never more than three per chart; companies
(nine of them) are shown as small multiples instead of nine hues.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import COMPANIES, FIGURES, PROCESSED, TABLES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated, all-pairs)
BLUE, ORANGE, AQUA = SERIES
BLUE_LIGHT = "#86b6ef"  # sequential step 250
DEEMPH = "#c3c2b7"
AUDIENCE_COLORS = {"live": BLUE, "hybrid": ORANGE, "virtual": AQUA}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "Helvetica Neue",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9.5,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def dot(ax, x, y, color=BLUE, size=34, alpha=1.0, zorder=3, **kw):
    """Filled marker with a 2 px surface ring (legible where marks overlap)."""
    return ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=1.2,
                      alpha=alpha, zorder=zorder, **kw)


def pct(ax, axis="y", decimals=0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# --------------------------------------------------------------------------- figures

def fig_coverage(panel: pd.DataFrame) -> None:
    """Keynotes per year by audience type (stacked columns, 2 px surface gaps)."""
    df = panel.assign(year=pd.to_datetime(panel["date_local"]).dt.year)
    counts = df.pivot_table(index="year", columns="audience", values="event_id",
                            aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=["live", "hybrid", "virtual"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bottom = np.zeros(len(counts))
    for aud in counts.columns:
        ax.bar(counts.index, counts[aud], bottom=bottom, width=0.62,
               color=AUDIENCE_COLORS[aud], edgecolor=SURFACE, linewidth=1.5, label=aud)
        bottom += counts[aud].to_numpy()
    ax.set_xticks(counts.index[::2])
    ax.set_ylabel("Keynotes")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0), handlelength=1, handleheight=1)
    ax.set_title("Keynotes in the catalogue, by year and audience", pad=18)
    subtitle(ax, "Pandemic-era keynotes (2020–22) were virtual; Apple never went back")
    save(fig, "coverage_by_year")


def fig_timeline(event_id: str, caption_tags: pd.DataFrame | None, tl: pd.DataFrame,
                 episodes: list[tuple[int, int]], span: tuple[int, int], title: str) -> None:
    """Clap score over a keynote with detected ovations and human caption tags."""
    from .features import clap_score

    score = pd.Series(clap_score(tl)).rolling(5, center=True, min_periods=1).mean()
    minutes = tl["t"].to_numpy() / 60
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for a, b in episodes:
        ax.axvspan(a / 60, b / 60, color=BLUE_LIGHT, alpha=0.35, lw=0, zorder=1)
    ax.plot(minutes, score, color=BLUE, lw=1.1, zorder=2)
    if caption_tags is not None and len(caption_tags):
        tags = caption_tags[caption_tags["is_applause"]]
        ax.scatter(tags["start_s"] / 60, np.full(len(tags), -0.07), marker="|", s=40,
                   color=INK_2, lw=1, zorder=3, clip_on=False)
        ax.text(1.0, -0.07, " human caption tags", transform=ax.get_yaxis_transform(),
                va="center", color=INK_2, fontsize=8.5)
    ax.set_xlim(span[0] / 60, span[1] / 60)
    ax.set_ylim(-0.12, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Minutes into the video")
    ax.set_ylabel("Clap probability")
    ax.set_title(title, pad=18)
    subtitle(ax, "PANNs clap score (5 s smoothing); shaded = detected ovations; ticks = [Applause] captions")
    save(fig, f"timeline_{event_id}")


def fig_detector(validation: dict, per_event: pd.DataFrame, panel: pd.DataFrame) -> None:
    """(a) episode F1 across thresholds on dev videos; (b) applause share by audience type."""
    grid = pd.read_csv(TABLES / "detector_grid.csv")
    best = validation["params"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.35})
    sel = grid[(grid["min_dur"] == best["min_dur"])
               & np.isclose(grid["off"] / grid["on"], best["off"] / best["on"])]
    sel = sel.sort_values("on")
    for col, color, label in [("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"),
                              ("f1", AQUA, "F1")]:
        ax1.plot(sel["on"], sel[col], color=color, marker="o", ms=4.5, mec=SURFACE, mew=1,
                 label=label)
        ax1.text(sel["on"].iloc[-1] + 0.01, sel[col].iloc[-1], label, color=INK_2,
                 va="center", fontsize=8.5)
    ax1.axvline(best["on"], color=BASELINE, lw=1)
    ax1.set_xlabel("Onset threshold on clap probability")
    ax1.set_ylim(0, 1.02)
    ax1.set_xlim(sel["on"].min() - 0.02, sel["on"].max() + 0.1)
    ax1.set_title("Tuning on human captions", pad=18)
    subtitle(ax1, f"Test videos: P {validation['test_precision']:.2f} · "
                  f"R {validation['test_recall']:.2f} · AUC {validation['test_auc_pooled']:.2f}")

    order = ["live", "hybrid", "virtual"]
    rng = np.random.default_rng(0)
    ticks = []
    for i, aud in enumerate(order):
        vals = panel.loc[panel.audience == aud, "applause_share"].dropna().to_numpy()
        ticks.append(f"{aud.capitalize()}\nn={len(vals)}")
        if not len(vals):
            continue
        dot(ax2, rng.uniform(-0.18, 0.18, len(vals)) + i, vals, color=AUDIENCE_COLORS[aud],
            size=18, alpha=0.85)
        ax2.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2, color=INK, lw=1.5, zorder=4)
    ax2.set_xticks(range(3), ticks)
    pct(ax2)
    ax2.set_ylabel("Share of keynote in applause")
    ax2.set_title("Negative control", pad=18)
    subtitle(ax2, "Virtual keynotes should be near zero; bar = median")
    save(fig, "detector_validation")


def fig_caar(path_all: pd.DataFrame, path_aud: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True,
                                   gridspec_kw={"wspace": 0.12})
    a = path_all
    ax1.fill_between(a.rel_day, a.lo, a.hi, color=BLUE, alpha=0.10, lw=0)
    ax1.plot(a.rel_day, a.caar, color=BLUE)
    ax1.set_title("All keynotes", pad=18)
    subtitle(ax1, f"n = {int(a.n.iloc[0])}; band = 95% CI")
    for aud in ["live", "virtual"]:
        g = path_aud[path_aud.group == aud]
        if g.empty:
            continue
        ax2.fill_between(g.rel_day, g.lo, g.hi, color=AUDIENCE_COLORS[aud], alpha=0.10, lw=0)
        ax2.plot(g.rel_day, g.caar, color=AUDIENCE_COLORS[aud], label=f"{aud} (n={int(g.n.iloc[0])})")
        ax2.text(g.rel_day.iloc[-1] + 0.4, g.caar.iloc[-1], aud, color=INK_2, va="center",
                 fontsize=8.5)
    ax2.legend(loc="lower left")
    ax2.set_title("Live vs virtual audience", pad=18)
    subtitle(ax2, "Cumulative abnormal return, market model")
    for ax in (ax1, ax2):
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.axvline(0, color=BASELINE, lw=0.8)
        ax.set_xlabel("Trading days relative to the keynote (t0)")
        pct(ax, decimals=1)
    ax1.set_ylabel("CAAR")
    save(fig, "caar_paths")


def fig_scatter(live: pd.DataFrame, prim: dict, label_top: int = 3) -> None:
    from .analysis import PRIMARY_X, PRIMARY_Y

    d = live.dropna(subset=[PRIMARY_X, PRIMARY_Y])
    x, y = d[PRIMARY_X].to_numpy(), d[PRIMARY_Y].to_numpy()
    # within-company demeaned outcome so the plotted slope equals the fixed-effects beta
    y_dm = y - d.groupby("company")[PRIMARY_Y].transform("mean").to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    dot(ax, x, y_dm, size=30, alpha=0.9)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, prim["beta"] * xs, color=INK, lw=1.5)
    ax.fill_between(xs, prim["ci_lo"] * xs, prim["ci_hi"] * xs, color=INK, alpha=0.06, lw=0)
    ax.axhline(0, color=BASELINE, lw=0.8)
    ext = np.argsort(-np.abs(y_dm))[:label_top]
    for i in ext:
        ax.annotate(d["event_name"].iloc[i], (x[i], y_dm[i]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.set_xlabel("Applause share, z-score within company")
    ax.set_ylabel("SCAR[0,+1], demeaned within company")
    ax.set_title("More applause, different reaction?", pad=18)
    subtitle(ax, f"Live keynotes, n = {prim['n']}; slope = {prim['beta']:+.3f} "
                 f"(wild-cluster bootstrap p = {prim['p_wcb']:.2f})")
    save(fig, "primary_scatter")


def fig_company_multiples(live: pd.DataFrame) -> None:
    from .analysis import PRIMARY_Y

    slugs = live.groupby("company").size().sort_values(ascending=False).index.tolist()
    n = len(slugs)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.4, 2.1 * rows), sharey=True)
    for ax, slug in zip(axes.flat, slugs, strict=False):
        g = live[live.company == slug]
        dot(ax, g["applause_share"], g[PRIMARY_Y], size=16)
        if len(g) >= 3 and g["applause_share"].std() > 0:
            b, a = np.polyfit(g["applause_share"], g[PRIMARY_Y], 1)
            xs = np.linspace(g["applause_share"].min(), g["applause_share"].max(), 10)
            ax.plot(xs, a + b * xs, color=INK, lw=1.2)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"{COMPANIES[slug].name}  (n={len(g)})", fontsize=9.5, pad=4)
        pct(ax, "x")
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.supxlabel("Share of keynote in applause (raw)", fontsize=9.5, color=INK_2)
    fig.supylabel("SCAR[0,+1]", fontsize=9.5, color=INK_2)
    fig.suptitle("The applause–return slope, company by company", x=0.02, ha="left",
                 fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout()
    save(fig, "company_multiples")


def fig_terciles(table: pd.DataFrame, terc: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = table["tercile"].to_numpy()
    ax.bar(x, table["mean"], width=0.42, color=BLUE)
    ax.errorbar(x, table["mean"], yerr=1.96 * table["se"], fmt="none", ecolor=INK_2,
                elinewidth=1.2, capsize=0)
    for xi, m in zip(x, table["mean"], strict=True):
        ax.text(xi, m + (0.02 if m >= 0 else -0.02), f"{m:+.2f}", ha="center",
                va="bottom" if m >= 0 else "top", color=INK, fontsize=9)
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_xticks(x, ["Quietest third", "Middle", "Loudest third"])
    ax.set_ylabel("Mean SCAR[0,+1]")
    ax.set_title("Returns by applause tercile (within company)", pad=18)
    subtitle(ax, f"Loud − quiet = {terc['diff']:+.2f} σ, 95% CI [{terc['ci_lo']:+.2f}, "
                 f"{terc['ci_hi']:+.2f}]")
    save(fig, "terciles")


def fig_robustness(rob: pd.DataFrame, sec_meas: pd.DataFrame, sec_out: pd.DataFrame) -> None:
    rows = [(r.spec, r.beta, r.ci_lo, r.ci_hi, r.group) for r in rob.itertuples()]
    rows += [(f"Measure: {r.measure}", r.beta, r.ci_lo, r.ci_hi, "Applause measure")
             for r in sec_meas.itertuples()]
    fig, ax = plt.subplots(figsize=(6.8, 0.26 * len(rows) + 1.0))
    ys = np.arange(len(rows))[::-1]
    for yv, (label, b, lo, hi, _group) in zip(ys, rows, strict=True):
        primary = label.startswith("Primary")
        color = BLUE if primary else INK_2
        ax.plot([lo, hi], [yv, yv], color=color if primary else DEEMPH, lw=2)
        dot(ax, [b], [yv], color=color, size=30)
    ax.axvline(0, color=BASELINE, lw=0.8)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("β: SCAR[0,+1] per 1 SD more applause (95% CI, HC3)")
    ax.set_title("Robustness of the primary estimate", pad=18)
    subtitle(ax, "Blue = pre-registered primary specification")
    save(fig, "robustness")


def fig_placebo(betas: np.ndarray, real: float, p: float) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.hist(betas, bins=40, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(real, color=INK, lw=1.5)
    ax.text(real, ax.get_ylim()[1] * 0.95, f"  actual keynote days: {real:+.3f}", color=INK,
            fontsize=8.5, va="top", ha="left" if real < np.percentile(betas, 70) else "right")
    ax.set_xlabel("β from pseudo-event days")
    ax.set_ylabel("Draws")
    ax.set_title("Placebo: applause moved to random non-event days", pad=18)
    subtitle(ax, f"{len(betas):,} draws; share at least as extreme = {p:.2f}")
    save(fig, "placebo")


def fig_applause_by_company(live: pd.DataFrame) -> None:
    order = live.groupby("company")["applause_share"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rng = np.random.default_rng(1)
    for i, slug in enumerate(order):
        vals = live.loc[live.company == slug, "applause_share"].to_numpy()
        dot(ax, vals, rng.uniform(-0.15, 0.15, len(vals)) + i, size=16, alpha=0.9)
        ax.plot([np.median(vals)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.5, zorder=4)
    ax.set_yticks(range(len(order)), [COMPANIES[s].name for s in order])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    pct(ax, "x")
    ax.set_xlabel("Share of keynote in applause")
    ax.set_title("How much do audiences clap?", pad=18)
    subtitle(ax, "Live keynotes; bar = company median")
    save(fig, "applause_by_company")


def make_all() -> None:
    from .analysis import load_panel
    from .captions import caption_track, parse_json3
    from .detect import load_timeline
    from .features import EpisodeParams, clap_score, episodes, keynote_span

    setup()
    panel, live = load_panel()
    results = json.loads((TABLES / "results.json").read_text())
    validation = json.loads((TABLES / "detector_validation.json").read_text())
    per_event = pd.read_csv(TABLES / "detector_per_event.csv")
    params = EpisodeParams(**json.loads((PROCESSED / "detector_params.json").read_text()))

    fig_coverage(panel)
    fig_detector(validation, per_event, panel)
    fig_caar(pd.read_csv(TABLES / "caar_path_all.csv"),
             pd.read_csv(TABLES / "caar_path_audience.csv"))
    fig_scatter(live, results["primary"])
    fig_company_multiples(live)
    fig_terciles(pd.read_csv(TABLES / "terciles.csv"), results["terciles"])
    fig_robustness(pd.read_csv(TABLES / "robustness.csv"),
                   pd.read_csv(TABLES / "secondary_measures.csv"),
                   pd.read_csv(TABLES / "secondary_outcomes.csv"))
    placebo_betas = pd.read_csv(TABLES / "placebo_betas.csv")["beta"].to_numpy()
    fig_placebo(placebo_betas, results["placebo"]["beta_real"], results["placebo"]["p_placebo"])
    fig_applause_by_company(live)

    for event_id in ["apple_2017-09-12_iphone-x", "samsung_2013-03-14_galaxy-s4"]:
        match = panel[panel.event_id.str.startswith(event_id[:18])]
        if match.empty:
            continue
        eid = match.event_id.iloc[0]
        try:
            tl = load_timeline(eid)
        except FileNotFoundError:
            continue
        path, kind = caption_track(eid)
        tags = parse_json3(path) if kind == "manual" else None
        eps = episodes(clap_score(tl), params)
        fig_timeline(eid, tags, tl, eps, keynote_span(tl), match.event_name.iloc[0])
