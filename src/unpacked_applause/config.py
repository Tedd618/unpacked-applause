"""Project-wide paths, company metadata and analysis constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
CATALOG_DIR = DATA / "catalog"
CATALOG_PARTS = CATALOG_DIR / "parts"
EVENTS_CSV = CATALOG_DIR / "events.csv"

RAW = DATA / "raw"  # never committed
AUDIO_DIR = RAW / "audio"
CAPTIONS_DIR = RAW / "captions"
PRICES_DIR = RAW / "prices"

PROCESSED = DATA / "processed"  # small derived tables, committed
TIMELINES_DIR = PROCESSED / "timelines"

MODELS = ROOT / "models"
PANNS_CHECKPOINT = MODELS / "Cnn14_DecisionLevelMax_mAP=0.385.pth"
AUDIOSET_LABELS = MODELS / "class_labels_indices.csv"

SQL_DIR = ROOT / "sql"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"


@dataclass(frozen=True)
class Company:
    slug: str
    name: str
    ticker: str
    exchange: str  # exchange_calendars code
    market: str  # benchmark index ticker (Yahoo Finance)
    currency: str
    ipo: str  # first date an event can enter the sample


COMPANIES: dict[str, Company] = {
    c.slug: c
    for c in [
        Company("apple", "Apple", "AAPL", "XNYS", "^GSPC", "USD", "1980-12-12"),
        Company("samsung", "Samsung Electronics", "005930.KS", "XKRX", "^KS11", "KRW", "1975-06-11"),
        Company("google", "Alphabet", "GOOGL", "XNYS", "^GSPC", "USD", "2004-08-19"),
        Company("microsoft", "Microsoft", "MSFT", "XNYS", "^GSPC", "USD", "1986-03-13"),
        Company("meta", "Meta Platforms", "META", "XNYS", "^GSPC", "USD", "2012-05-18"),
        Company("amazon", "Amazon", "AMZN", "XNYS", "^GSPC", "USD", "1997-05-15"),
        Company("nvidia", "NVIDIA", "NVDA", "XNYS", "^GSPC", "USD", "1999-01-22"),
        Company("tesla", "Tesla", "TSLA", "XNYS", "^GSPC", "USD", "2010-06-29"),
        Company("amd", "AMD", "AMD", "XNYS", "^GSPC", "USD", "1972-09-27"),
    ]
}

# Extra series downloaded alongside the stocks (robustness benchmarks and controls).
EXTRA_TICKERS = ["^GSPC", "^NDX", "^KS11", "^VIX"]

SAMPLE_START = "2010-01-01"
SAMPLE_END = "2026-08-31"
PRICE_START = "2008-06-01"  # leaves a full estimation window before the first event

# --- Event-study design (fixed in docs/analysis_plan.md before outcomes were computed) ---
ESTIMATION_WINDOW = (-250, -30)  # trading days relative to t0, inclusive
MIN_ESTIMATION_OBS = 120
EVENT_WINDOWS: dict[str, tuple[int, int]] = {
    "car_0_0": (0, 0),
    "car_0_1": (0, 1),  # primary
    "car_m1_1": (-1, 1),
    "car_0_5": (0, 5),
    "car_2_20": (2, 20),  # post-event drift
    "car_m10_m1": (-10, -1),  # pre-event run-up (control)
}
PRIMARY_WINDOW = "car_0_1"
PATH_WINDOW = (-10, 20)  # days kept for CAAR path plots

# --- Audio analysis ---
SAMPLE_RATE = 32_000  # PANNs Cnn14 input rate
TIMELINE_CLASSES = {
    # column name -> AudioSet display name
    "speech": "Speech",
    "applause": "Applause",
    "clapping": "Clapping",
    "cheering": "Cheering",
    "crowd": "Crowd",
    "chatter": "Chatter",
    "hubbub": "Hubbub, speech noise, speech babble",
    "laughter": "Laughter",
    "whoop": "Whoop",
    "yell": "Yell",
    "music": "Music",
    "horn": "Vehicle horn, car horn, honking",
    "silence": "Silence",
}
