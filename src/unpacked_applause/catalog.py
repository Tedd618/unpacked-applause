"""Assemble and validate the event catalog (data/catalog/events.csv)."""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from .config import CATALOG_PARTS, COMPANIES, EVENTS_CSV, SAMPLE_END, SAMPLE_START

COLUMNS = [
    "event_id", "company", "ticker", "event_name", "series", "event_type",
    "headline_products", "date_local", "start_time_local", "tz", "start_time_confidence",
    "city", "venue", "audience", "audience_notes", "youtube_id", "youtube_title",
    "youtube_channel", "youtube_duration_s", "youtube_upload_date", "official_channel",
    "vod_notes", "sources",
]
AUDIENCES = {"live", "virtual", "hybrid"}
EVENT_TYPES = {"product_launch", "developer_conference", "enterprise_conference", "investor_event"}
CONFIDENCE = {"high", "medium", "low"}
EVENT_ID_RE = re.compile(r"^[a-z]+_\d{4}-\d{2}-\d{2}_[a-z0-9-]+$")
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class CatalogError(ValueError):
    pass


def read_parts() -> pd.DataFrame:
    frames = []
    for path in sorted(CATALOG_PARTS.glob("*.csv")):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise CatalogError(f"{path.name}: missing columns {sorted(missing)}")
        frames.append(df[COLUMNS])
    if not frames:
        raise CatalogError(f"no catalog parts found in {CATALOG_PARTS}")
    df = pd.concat(frames, ignore_index=True)
    return df.apply(lambda s: s.str.strip())


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems: list[str] = []

    def bad(mask: pd.Series, msg: str) -> None:
        for eid in df.loc[mask, "event_id"]:
            problems.append(f"{eid}: {msg}")

    bad(~df["event_id"].str.match(EVENT_ID_RE), "malformed event_id")
    dup = df["event_id"].duplicated(keep=False)
    bad(dup, "duplicate event_id")

    bad(~df["company"].isin(list(COMPANIES)), "unknown company")
    expected_ticker = df["company"].map({k: c.ticker for k, c in COMPANIES.items()})
    bad(df["ticker"] != expected_ticker, "ticker does not match company")

    dates = pd.to_datetime(df["date_local"], format="%Y-%m-%d", errors="coerce")
    bad(dates.isna(), "unparseable date_local")
    bad((dates < SAMPLE_START) | (dates > SAMPLE_END), "date outside sample period")
    ipo = pd.to_datetime(df["company"].map({k: c.ipo for k, c in COMPANIES.items()}))
    bad(dates < ipo, "event before IPO")
    id_date = df["event_id"].str.extract(r"_(\d{4}-\d{2}-\d{2})_")[0]
    bad(id_date != df["date_local"], "event_id date differs from date_local")

    bad(~df["start_time_local"].str.match(TIME_RE), "start_time_local not HH:MM")
    bad(~df["start_time_confidence"].isin(CONFIDENCE), "bad start_time_confidence")

    def tz_ok(name: str) -> bool:
        try:
            ZoneInfo(name)
            return True
        except (ZoneInfoNotFoundError, ValueError):
            return False

    bad(~df["tz"].map(tz_ok), "unknown IANA time zone")
    bad(~df["audience"].isin(AUDIENCES), "bad audience value")
    bad(~df["event_type"].isin(EVENT_TYPES), "bad event_type")

    has_vod = df["youtube_id"] != ""
    bad(has_vod & ~df["youtube_id"].str.match(YOUTUBE_ID_RE), "malformed youtube_id")
    dur = pd.to_numeric(df["youtube_duration_s"], errors="coerce")
    bad(has_vod & dur.isna(), "youtube_duration_s missing")
    dup_vod = has_vod & df["youtube_id"].duplicated(keep=False)
    bad(dup_vod, "youtube_id used by more than one event")
    bad(~df["official_channel"].str.lower().isin(["true", "false"]), "official_channel not true/false")
    return problems


def add_start_utc(df: pd.DataFrame) -> pd.DataFrame:
    def to_utc(row: pd.Series) -> pd.Timestamp:
        local = pd.Timestamp(f"{row.date_local} {row.start_time_local}")
        return local.tz_localize(ZoneInfo(row.tz)).tz_convert("UTC")

    out = df.copy()
    out["start_utc"] = out.apply(to_utc, axis=1)
    return out


def build_catalog() -> pd.DataFrame:
    df = read_parts()
    problems = validate(df)
    if problems:
        raise CatalogError("catalog validation failed:\n  " + "\n  ".join(problems))
    df = add_start_utc(df)
    df["official_channel"] = df["official_channel"].str.lower() == "true"
    df = df.sort_values(["start_utc", "company"]).reset_index(drop=True)
    df.to_csv(EVENTS_CSV, index=False)
    return df


def load_events(with_vod_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(EVENTS_CSV, dtype=str, keep_default_na=False)
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    df["date_local"] = pd.to_datetime(df["date_local"])
    df["youtube_duration_s"] = pd.to_numeric(df["youtube_duration_s"], errors="coerce")
    df["official_channel"] = df["official_channel"].str.lower() == "true"
    if with_vod_only:
        df = df[df["youtube_id"] != ""].reset_index(drop=True)
    return df
