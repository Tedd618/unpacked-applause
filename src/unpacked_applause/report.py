"""Render the research report: Jinja2 HTML template -> headless Chrome -> PDF."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import COMPANIES, FIGURES, PROCESSED, REPORTS, TABLES, TIMELINES_DIR

TEMPLATE_DIR = REPORTS / "template"
BUILD_DIR = REPORTS / "build"
PDF_PATH = REPORTS / "unpacked_applause_report.pdf"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise FileNotFoundError("Chrome/Chromium is needed to print the PDF")


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def context() -> dict:
    tables = {p.stem: pd.read_csv(p) for p in TABLES.glob("*.csv")}
    timelines = list(TIMELINES_DIR.glob("*.parquet"))
    timeline_figs = sorted(FIGURES.glob("timeline_*.svg"))
    fig_name = timeline_figs[0].stem if timeline_figs else ""
    return {
        "results": json.loads((TABLES / "results.json").read_text()),
        "validation": json.loads((TABLES / "detector_validation.json").read_text()),
        "params": json.loads((PROCESSED / "detector_params.json").read_text()),
        "t": tables,
        "fig": lambda name: (FIGURES / f"{name}.svg").as_uri(),
        "fmt_p": fmt_p,
        "pct": lambda v, d=2: f"{100 * v:+.{d}f}%",
        "companies": {k: c.name for k, c in COMPANIES.items()},
        "build_date": date.today().strftime("%B %Y"),
        "timeline_kb": sum(p.stat().st_size for p in timelines) / max(len(timelines), 1) / 1e3,
        "timeline_fig": fig_name,
        "timeline_caption": TIMELINE_CAPTION,
        # row lookups for the narrative: A["All keynotes|[0,+1]"].mean, C["Apple"].p_t, ...
        "A": {f"{r.sample}|{r.window}": r for r in tables["average_effects"].itertuples()},
        "C": {r.company: r for r in tables["by_company"].itertuples()},
        "R": {r.spec: r for r in tables["robustness"].itertuples()},
        "LV": {r.outcome: r for r in tables["live_vs_virtual"].itertuples()},
        "sec_min_q": min(tables["secondary_outcomes"]["q_bh"].min(),
                         tables["secondary_measures"]["q_bh"].min()),
        "rob_min_p": tables["robustness"]["p_wcb"].min(),
        "rob_beta_range": (tables["robustness"]["beta"].min(), tables["robustness"]["beta"].max()),
        # typical two-day residual volatility, to translate SCAR units into returns
        "typ_sd2": float(tables["analysis_sample"]["sigma"].median() * 2 ** 0.5),
    }


TIMELINE_CAPTION = (
    "Fifteen minutes of one keynote as the pipeline sees it: the per-second clap score, the "
    "detected ovations (shaded) and, underneath, the [Applause] tags a human captioner wrote. "
    "The two sources were produced independently. The keynote shown is the caption-labelled "
    "video whose detection accuracy (ROC-AUC) is closest to the median, not the best one.")


def narrative(env: Environment, ctx: dict) -> dict[str, str]:
    """Sections of narrative.md (## key) rendered with the same context, one <p> per paragraph."""
    text = (TEMPLATE_DIR / "narrative.md").read_text()
    out = {}
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        key, body = block.split("\n", 1)
        rendered = env.from_string(body).render(**ctx).strip()
        paras = [q.strip().replace("\n", " ") for q in re.split(r"\n\s*\n", rendered) if q.strip()]
        out[key.strip()] = "\n".join(p if p.startswith("<") else f"<p>{p}</p>" for p in paras)
    return out


def build_report(pdf: bool = True) -> Path:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR),
                      autoescape=select_autoescape(["html"]))
    ctx = context()
    text = narrative(Environment(), ctx)
    ctx["text"] = {k: (v if k != "abstract" else re.sub(r"</?p>", "", v)) for k, v in text.items()}
    html = env.get_template("report.html.j2").render(**ctx)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    html_path = BUILD_DIR / "report.html"
    html_path.write_text(html)
    if pdf:
        subprocess.run([
            find_chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
            f"--print-to-pdf={PDF_PATH}", html_path.as_uri(),
        ], check=True, capture_output=True)
        print(f"report: {PDF_PATH}")
    return html_path
