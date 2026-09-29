"""Generate the analysis notebooks from plain Python so they stay reviewable in diffs.

Run `python notebooks/build_notebooks.py`, then `make notebooks` to execute them in place.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = """\
%load_ext autoreload
%autoreload 2
import json
import numpy as np
import pandas as pd
from IPython.display import Image, display
from unpacked_applause.config import FIGURES, PROCESSED, TABLES
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")"""


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


NOTEBOOKS = {
    "01_event_catalog.ipynb": [
        md("""
# 1 · The keynote catalogue

Every row of `data/catalog/events.csv` is one keynote: when it started (with its IANA time
zone, because the event day depends on whether the market was still open), whether a live
audience was present, and which YouTube video carries the full broadcast audio. The
catalogue was researched company by company and validated by `applause catalog`
(`src/unpacked_applause/catalog.py`); manual decisions are listed in
`data/catalog/curation_log.md`.
"""),
        code(SETUP),
        code("""
import duckdb
from unpacked_applause.catalog import load_events
events = load_events()
con = duckdb.connect()
con.register("events", events.assign(date_local=events["date_local"].dt.date))
con.execute(open("../sql/coverage.sql").read()).df()
"""),
        md("""
## When do keynotes happen relative to the trading day?

`t0` is the first session whose close is after the keynote start. Apple, Google and
Microsoft usually present at 10:00 Pacific, i.e. during US trading hours, so the reaction
starts the same day; Tesla presents in the evening and Samsung's Unpacked is overnight in
Seoul, so their `t0` is the next session.
"""),
        code("""
study = pd.read_parquet(PROCESSED / "event_study.parquet")
ev = events.merge(study[["event_id", "t0", "days_to_t0"]], on="event_id", how="left")
ev["t0_same_day"] = ev["days_to_t0"] == 0
ev.groupby("company")["t0_same_day"].mean().rename("share with t0 = keynote day").to_frame()
"""),
        code("""display(Image(filename=str(FIGURES / "coverage_by_year.png")))"""),
        code("""
events.groupby(["company", "audience"]).size().unstack(fill_value=0)
"""),
    ],
    "02_applause_detection.ipynb": [
        md("""
# 2 · Measuring applause

Each keynote's audio is decoded to 32 kHz mono and passed through PANNs
`Cnn14_DecisionLevelMax`, a CNN trained on AudioSet's 527 sound classes. The model emits a
probability per class every 0.32 s; we keep 13 classes (speech, applause, clapping,
cheering, laughter, music, ...) averaged to one row per second. An ovation is a stretch where
the clap score `max(P(applause), P(clapping))` crosses an onset threshold, with hysteresis.

The thresholds are tuned **only** on human-written caption tags (`[Applause]`,
`[Cheering and applause]`, ...) that professional captioners added to official uploads.
Half of the caption-labelled videos tune the thresholds, the other half measure accuracy.
"""),
        code(SETUP),
        code("""
validation = json.loads((TABLES / "detector_validation.json").read_text())
{k: v for k, v in validation.items() if not k.endswith("_ids")}
"""),
        code("""
grid = pd.read_csv(TABLES / "detector_grid.csv")
grid.head(10)
"""),
        code("""
per_event = pd.read_csv(TABLES / "detector_per_event.csv")
per_event.describe()[["auc", "ap", "label_share"]]
"""),
        code("""display(Image(filename=str(FIGURES / "detector_validation.png")))"""),
        md("""
## An example timeline

Detected ovations (shaded) against the human caption tags (ticks under the axis).
"""),
        code("""
for p in sorted(FIGURES.glob("timeline_*.png")):
    display(Image(filename=str(p)))
"""),
        md("""
## Event-level applause measures
"""),
        code("""
features = pd.read_parquet(PROCESSED / "applause_features.parquet")
from unpacked_applause.catalog import load_events
f = load_events()[["event_id", "company", "audience"]].merge(features, on="event_id")
f.groupby("audience")[["applause_share", "applause_per_min", "applause_max_s", "cheer_share"]].median()
"""),
        code("""display(Image(filename=str(FIGURES / "applause_by_company.png")))"""),
    ],
    "03_event_study.ipynb": [
        md("""
# 3 · The event study

Normal returns come from a market model estimated over trading days [-250, -30] before
`t0` (S&P 500 for the US listings, KOSPI for Samsung Electronics). Abnormal returns are
cumulated over the event windows and standardised by the estimation-window residual
volatility (SCAR), which puts Tesla's and Microsoft's reactions on a common scale.
"""),
        code(SETUP),
        code("""
avg = pd.read_csv(TABLES / "average_effects.csv")
avg[avg["sample"] == "All keynotes"][["window", "n", "mean", "ci_lo", "ci_hi", "p_t", "p_bmp", "pos_share", "p_sign"]]
"""),
        code("""
avg[avg["window"] == "[0,+1]"][["sample", "n", "mean", "p_t", "p_bmp", "pos_share"]]
"""),
        code("""pd.read_csv(TABLES / "by_company.csv")[["company", "n", "mean", "ci_lo", "ci_hi", "p_t", "pos_share"]]"""),
        code("""display(Image(filename=str(FIGURES / "caar_paths.png")))"""),
        code("""pd.read_csv(TABLES / "live_vs_virtual.csv")"""),
    ],
    "04_applause_vs_returns.ipynb": [
        md("""
# 4 · Does applause line up with the stock's reaction?

This notebook runs the pre-registered plan in `docs/analysis_plan.md`: the primary
regression of SCAR[0,+1] on the within-company z-score of applause share with company fixed
effects and a wild cluster bootstrap over the nine companies, then the secondary families
(Benjamini–Hochberg corrected), robustness checks, a tercile sort and a placebo test.
"""),
        code(SETUP),
        code("""
results = json.loads((TABLES / "results.json").read_text())
pd.Series(results["primary"])
"""),
        code("""display(Image(filename=str(FIGURES / "primary_scatter.png")))"""),
        code("""pd.read_csv(TABLES / "secondary_outcomes.csv")[["outcome", "beta", "ci_lo", "ci_hi", "p_hc3", "p_wcb", "q_bh", "n"]]"""),
        code("""pd.read_csv(TABLES / "secondary_measures.csv")[["measure", "beta", "ci_lo", "ci_hi", "p_hc3", "p_wcb", "q_bh", "n"]]"""),
        code("""pd.read_csv(TABLES / "robustness.csv")[["group", "spec", "beta", "ci_lo", "ci_hi", "p_wcb", "n"]]"""),
        code("""display(Image(filename=str(FIGURES / "robustness.png")))"""),
        code("""
print(results["terciles"])
display(Image(filename=str(FIGURES / "terciles.png")))
"""),
        code("""
print(results["spearman"])
print(results["placebo"])
display(Image(filename=str(FIGURES / "placebo.png")))
"""),
        code("""display(Image(filename=str(FIGURES / "company_multiples.png")))"""),
    ],
}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                        "language": "python"}
        nbf.write(nb, HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
