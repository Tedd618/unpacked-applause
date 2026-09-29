"""Command-line entry point: ``applause <step>`` (see the Makefile for the full pipeline)."""

from __future__ import annotations

import argparse
import json
import random
import time

import pandas as pd

from .config import AUDIO_DIR, CAPTIONS_DIR, PROCESSED, TABLES


def _events(args, with_vod_only=True) -> pd.DataFrame:
    from .catalog import load_events

    ev = load_events(with_vod_only=with_vod_only)
    if getattr(args, "company", None):
        ev = ev[ev["company"].isin(args.company)]
    if getattr(args, "limit", None):
        ev = ev.head(args.limit)
    return ev


def _fetch(args, what: str) -> None:
    from .audio import audio_path, download_audio, download_captions

    ev = _events(args)
    failures = []
    for i, row in enumerate(ev.itertuples(), 1):
        if what == "audio" and audio_path(row.event_id) is not None:
            continue
        try:
            fn = download_audio if what == "audio" else download_captions
            fn(row.event_id, row.youtube_id)
            print(f"[{i}/{len(ev)}] {what} ok: {row.event_id}", flush=True)
        except Exception as exc:  # keep going; failures are logged for a retry pass
            failures.append({"event_id": row.event_id, "youtube_id": row.youtube_id,
                             "error": str(exc)[:300]})
            print(f"[{i}/{len(ev)}] {what} FAILED: {row.event_id}: {exc}", flush=True)
        time.sleep(random.uniform(2, 5))  # be polite to YouTube
    out_dir = AUDIO_DIR if what == "audio" else CAPTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(args.company) if args.company else "all"  # one log per parallel run
    pd.DataFrame(failures, columns=["event_id", "youtube_id", "error"]).to_csv(
        out_dir / f"_failures_{tag}.csv", index=False)
    print(f"{what}: {len(failures)} failures")


def cmd_catalog(args) -> None:
    import duckdb

    from .catalog import build_catalog, load_events
    from .config import SQL_DIR

    print(f"catalog: {len(build_catalog())} keynotes written")
    con = duckdb.connect()
    events = load_events()
    con.register("events", events.assign(date_local=events["date_local"].dt.date))
    print(con.execute((SQL_DIR / "coverage.sql").read_text()).df().to_string(index=False))


def cmd_detect(args) -> None:
    from .detect import detect_all

    detect_all(_events(args)["event_id"].tolist(), device=args.device, overwrite=args.overwrite)


def cmd_validate(args) -> None:
    from dataclasses import asdict

    from .validation import validate_detector

    res = validate_detector(_events(args)["event_id"].tolist())
    TABLES.mkdir(parents=True, exist_ok=True)
    res["grid"].to_csv(TABLES / "detector_grid.csv", index=False)
    res["per_event"].to_csv(TABLES / "detector_per_event.csv", index=False)
    summary = {k: v for k, v in res.items() if k not in {"grid", "per_event", "params"}}
    summary["params"] = asdict(res["params"])
    (TABLES / "detector_validation.json").write_text(json.dumps(summary, indent=2, default=float))
    (PROCESSED / "detector_params.json").write_text(json.dumps(asdict(res["params"]), indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_ids")},
                     indent=2, default=float))


def cmd_features(args) -> None:
    from .detect import load_timeline, timeline_path
    from .features import EpisodeParams, event_features

    params_file = PROCESSED / "detector_params.json"
    params = EpisodeParams(**json.loads(params_file.read_text())) if params_file.exists() \
        else EpisodeParams()
    print(f"features: using {params}")
    rows = [{"event_id": e, **event_features(load_timeline(e), params)}
            for e in _events(args)["event_id"] if timeline_path(e).exists()]
    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED / "applause_features.parquet", index=False)
    df.to_csv(PROCESSED / "applause_features.csv", index=False)
    print(f"features: {len(df)} keynotes")


def cmd_market(args) -> None:
    from .market import download_ff_factors, download_prices

    download_prices()
    download_ff_factors()


def cmd_study(args) -> None:
    from .analysis import PATHS_FILE, STUDY_FILE
    from .event_study import run_event_study

    table, paths = run_event_study(_events(args, with_vod_only=False))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_parquet(STUDY_FILE, index=False)
    paths.to_parquet(PATHS_FILE, index=False)
    print(f"study: {table['t0'].notna().sum()} of {len(table)} keynotes mapped to a trading day")


def cmd_analyze(args) -> None:
    from .analysis import run_all

    summary = run_all(n_boot=args.n_boot, n_placebo=args.n_placebo)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


def cmd_figures(args) -> None:
    from .plots import make_all

    make_all()


def cmd_report(args) -> None:
    from .report import build_report

    build_report()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="applause", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(func=fn)
        p.add_argument("--company", nargs="*", help="restrict to these company slugs")
        p.add_argument("--limit", type=int, help="process at most N keynotes")
        return p

    add("catalog", cmd_catalog, "merge and validate data/catalog/parts/*.csv")
    add("fetch-audio", lambda a: _fetch(a, "audio"), "download keynote audio (yt-dlp)")
    add("fetch-captions", lambda a: _fetch(a, "captions"), "download YouTube captions")
    p = add("detect", cmd_detect, "run the PANNs applause detector")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    add("validate", cmd_validate, "validate/tune the detector against caption tags")
    add("features", cmd_features, "event-level applause measures")
    add("market", cmd_market, "download prices and Fama-French factors")
    add("study", cmd_study, "run the event study")
    p = add("analyze", cmd_analyze, "cross-sectional analysis (pre-registered plan)")
    p.add_argument("--n-boot", type=int, default=9_999)
    p.add_argument("--n-placebo", type=int, default=1_000)
    add("figures", cmd_figures, "render report figures")
    add("report", cmd_report, "build the HTML/PDF report")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
