"""Validate the applause detector against human caption tags and tune episode thresholds."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .captions import applause_labels, caption_track, parse_json3, tagged_seconds
from .detect import load_timeline, timeline_path
from .features import EpisodeParams, clap_score, episodes, keynote_span, runs

TOLERANCE_S = 2
ON_GRID = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
OFF_RATIO_GRID = [0.4, 0.6, 0.8]
MIN_DUR_GRID = [1, 2, 3]


def labelled_events(event_ids: list[str], min_tags: int = 10) -> list[str]:
    """Events with a human caption track that actually annotates applause."""
    out = []
    for eid in event_ids:
        path, kind = caption_track(eid)
        if kind != "manual" or not timeline_path(eid).exists():
            continue
        if parse_json3(path)["is_applause"].sum() >= min_tags:
            out.append(eid)
    return out


def load_labelled(event_id: str) -> dict:
    tl = load_timeline(event_id)
    path, _ = caption_track(event_id)
    tags = parse_json3(path)
    s0, s1 = keynote_span(tl)
    y = applause_labels(tags, len(tl))
    return {"event_id": event_id, "tl": tl, "y": y, "span": (s0, s1),
            "any_tag": tagged_seconds(tags, len(tl))}


def _overlaps(a: tuple[int, int], b: tuple[int, int], tol: int = TOLERANCE_S) -> bool:
    return a[0] < b[1] + tol and b[0] < a[1] + tol


def episode_scores(item: dict, params: EpisodeParams) -> dict[str, float]:
    s0, s1 = item["span"]
    det = [e for e in episodes(clap_score(item["tl"]), params) if s0 <= e[0] < s1]
    ref = [r for r in runs(item["y"].astype(bool)) if s0 <= r[0] < s1]
    tp_det = sum(any(_overlaps(d, r) for r in ref) for d in det)
    tp_ref = sum(any(_overlaps(r, d) for d in det) for r in ref)
    return {"n_det": len(det), "n_ref": len(ref), "tp_det": tp_det, "tp_ref": tp_ref}


def prf(rows: list[dict]) -> tuple[float, float, float]:
    n_det = sum(r["n_det"] for r in rows)
    n_ref = sum(r["n_ref"] for r in rows)
    precision = sum(r["tp_det"] for r in rows) / n_det if n_det else np.nan
    recall = sum(r["tp_ref"] for r in rows) / n_ref if n_ref else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def split_dev_test(event_ids: list[str], seed: int = 7) -> tuple[list[str], list[str]]:
    """Deterministic 50/50 split, stratified by company."""
    rng = np.random.default_rng(seed)
    dev, test = [], []
    by_company: dict[str, list[str]] = {}
    for eid in sorted(event_ids):
        by_company.setdefault(eid.split("_")[0], []).append(eid)
    for ids in by_company.values():
        ids = list(rng.permutation(ids))
        half = (len(ids) + 1) // 2
        dev += ids[:half]
        test += ids[half:]
    return dev, test


def tune(items: list[dict]) -> tuple[EpisodeParams, pd.DataFrame]:
    grid = []
    for on, ratio, min_dur in itertools.product(ON_GRID, OFF_RATIO_GRID, MIN_DUR_GRID):
        p = EpisodeParams(on=on, off=round(on * ratio, 3), min_dur=min_dur, merge_gap=1)
        precision, recall, f1 = prf([episode_scores(it, p) for it in items])
        grid.append({"on": p.on, "off": p.off, "min_dur": min_dur,
                     "precision": precision, "recall": recall, "f1": f1})
    grid = pd.DataFrame(grid).sort_values("f1", ascending=False, ignore_index=True)
    best = grid.iloc[0]
    return EpisodeParams(on=float(best.on), off=float(best.off), min_dur=int(best.min_dur)), grid


def second_level_metrics(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        s0, s1 = it["span"]
        y = it["y"][s0:s1]
        score = clap_score(it["tl"])[s0:s1]
        if y.min() == y.max():
            continue
        rows.append({"event_id": it["event_id"], "auc": roc_auc_score(y, score),
                     "ap": average_precision_score(y, score), "label_share": y.mean()})
    return pd.DataFrame(rows)


def validate_detector(event_ids: list[str]) -> dict:
    ids = labelled_events(event_ids)
    dev_ids, test_ids = split_dev_test(ids)
    items = {eid: load_labelled(eid) for eid in ids}
    params, grid = tune([items[e] for e in dev_ids])
    test_rows = [episode_scores(items[e], params) for e in test_ids]
    precision, recall, f1 = prf(test_rows)
    per_event = second_level_metrics([items[e] for e in ids])
    pooled_y = np.concatenate([items[e]["y"][slice(*items[e]["span"])] for e in test_ids])
    pooled_s = np.concatenate([clap_score(items[e]["tl"])[slice(*items[e]["span"])]
                               for e in test_ids])
    return {
        "params": params,
        "grid": grid,
        "dev_ids": dev_ids,
        "test_ids": test_ids,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_auc_pooled": roc_auc_score(pooled_y, pooled_s),
        "test_ap_pooled": average_precision_score(pooled_y, pooled_s),
        "per_event": per_event,
    }
