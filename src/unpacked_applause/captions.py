"""Parse YouTube json3 captions into applause intervals (weak labels for detector validation)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CAPTIONS_DIR

# Bracketed sound descriptions such as "[Applause]", "( cheers and applause )", "[CLAPPING]".
TAG_RE = re.compile(r"[\[\(]([^\]\)]{1,60})[\]\)]")
APPLAUSE_RE = re.compile(r"applau|clap|cheer|ovation|whoop", re.IGNORECASE)


def parse_json3(path: Path) -> pd.DataFrame:
    """All bracketed sound tags as intervals: columns start_s, end_s, tag, is_applause."""
    events = json.loads(Path(path).read_text()).get("events", [])
    rows = []
    for ev in events:
        text = "".join(seg.get("utf8", "") for seg in ev.get("segs") or [])
        for tag in TAG_RE.findall(text):
            start = ev.get("tStartMs", 0) / 1000
            rows.append({
                "start_s": start,
                "end_s": start + ev.get("dDurationMs", 0) / 1000,
                "tag": tag.strip().lower(),
                "is_applause": bool(APPLAUSE_RE.search(tag)),
            })
    return pd.DataFrame(rows, columns=["start_s", "end_s", "tag", "is_applause"])


def caption_track(event_id: str) -> tuple[Path | None, str]:
    """Best caption file for an event and its kind ('manual', 'auto' or 'none')."""
    manual_langs = set()
    marker = CAPTIONS_DIR / f"{event_id}.manual"
    if marker.exists():
        manual_langs = {x for x in marker.read_text().split() if x}
    files = {p.name.split(".")[-2]: p for p in CAPTIONS_DIR.glob(f"{event_id}.*.json3")}
    for lang in ("en", "en-US", "en-GB"):
        if lang in files and lang in manual_langs:
            return files[lang], "manual"
    for lang in ("en-orig", "en", "en-US", "en-GB"):
        if lang in files:
            return files[lang], "auto"
    return None, "none"


def applause_labels(tags: pd.DataFrame, n_seconds: int) -> np.ndarray:
    """1 for every whole second overlapped by an applause tag, else 0."""
    y = np.zeros(n_seconds, dtype=np.int8)
    for row in tags[tags["is_applause"]].itertuples():
        a, b = int(np.floor(row.start_s)), int(np.ceil(row.end_s))
        y[max(a, 0) : min(b, n_seconds)] = 1
    return y


def tagged_seconds(tags: pd.DataFrame, n_seconds: int) -> np.ndarray:
    """1 for every second overlapped by any bracketed sound tag (music, laughter, ...)."""
    y = np.zeros(n_seconds, dtype=np.int8)
    for row in tags.itertuples():
        a, b = int(np.floor(row.start_s)), int(np.ceil(row.end_s))
        y[max(a, 0) : min(b, n_seconds)] = 1
    return y
