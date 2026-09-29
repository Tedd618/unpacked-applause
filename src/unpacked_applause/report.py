"""Render the research report: Jinja2 HTML template -> headless Chrome -> PDF."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import FIGURES, PROCESSED, REPORTS, TABLES

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
    return {
        "results": json.loads((TABLES / "results.json").read_text()),
        "validation": json.loads((TABLES / "detector_validation.json").read_text()),
        "params": json.loads((PROCESSED / "detector_params.json").read_text()),
        "t": tables,
        "fig": lambda name: (FIGURES / f"{name}.svg").as_uri(),
        "fmt_p": fmt_p,
    }


def build_report(pdf: bool = True) -> Path:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR),
                      autoescape=select_autoescape(["html"]))
    html = env.get_template("report.html.j2").render(**context())
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
