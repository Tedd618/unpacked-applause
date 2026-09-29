"""Event-level applause measures computed from a 1 Hz sound-event timeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

CLAP_COLS = ["applause", "clapping"]
CHEER_COLS = ["cheering", "whoop", "yell"]


@dataclass(frozen=True)
class EpisodeParams:
    """Hysteresis rule that turns the per-second clap score into applause episodes.

    Runs of seconds with score >= ``off`` that are separated by at most ``merge_gap`` seconds
    are merged; a merged run becomes an episode if it reaches ``on`` somewhere and lasts at
    least ``min_dur`` seconds. Thresholds are tuned against human caption tags only
    (see notebooks/02_applause_detection.ipynb), never against returns.
    """

    on: float = 0.5
    off: float = 0.25
    min_dur: int = 2
    merge_gap: int = 1


@dataclass(frozen=True)
class SpanParams:
    """The keynote span runs from the first to the last minute that is mostly speech.

    A closing music performance (Apple booked Foo Fighters, U2, Sia, ...) is cut off: if a run
    of at least ``closing_music_s`` music-dominated seconds starts in the last
    ``closing_zone`` of the span, the span ends where that run starts.
    """

    speech_thr: float = 0.5
    window_s: int = 60
    min_speech_share: float = 0.3
    music_thr: float = 0.5
    closing_music_s: int = 120
    closing_zone: float = 0.25


DEFAULT_EPISODE = EpisodeParams()
DEFAULT_SPAN = SpanParams()


def clap_score(tl: pd.DataFrame) -> np.ndarray:
    return tl[CLAP_COLS].max(axis=1).to_numpy()


def cheer_score(tl: pd.DataFrame) -> np.ndarray:
    return tl[CHEER_COLS].max(axis=1).to_numpy()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index runs where ``mask`` is True."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def episodes(score: np.ndarray, p: EpisodeParams = DEFAULT_EPISODE) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for a, b in runs(score >= p.off):  # a brief dip below `off` does not end an ovation
        if merged and a - merged[-1][1] <= p.merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if score[a:b].max() >= p.on and b - a >= p.min_dur]


def episode_mask(n: int, eps: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for a, b in eps:
        mask[a:b] = True
    return mask


def keynote_span(tl: pd.DataFrame, p: SpanParams = DEFAULT_SPAN) -> tuple[int, int]:
    """Trim countdown pre-rolls and post-show music: keep the speech-dominated stretch."""
    speaking = (tl["speech"].to_numpy() >= p.speech_thr).astype(float)
    share = pd.Series(speaking).rolling(p.window_s, center=True, min_periods=1).mean().to_numpy()
    active = np.flatnonzero(share >= p.min_speech_share)
    if len(active) == 0:
        return 0, len(tl)
    s0, s1 = int(active[0]), int(active[-1]) + 1
    music = (tl["music"].to_numpy() >= p.music_thr).astype(float)
    dominated = pd.Series(music).rolling(30, center=True, min_periods=1).mean().to_numpy() >= 0.8
    zone_start = s1 - int(p.closing_zone * (s1 - s0))
    for a, b in runs(dominated[s0:]):
        if s0 + a >= zone_start and b - a >= p.closing_music_s:
            return s0, s0 + a
    return s0, s1


def event_features(tl: pd.DataFrame, ep: EpisodeParams = DEFAULT_EPISODE,
                   sp: SpanParams = DEFAULT_SPAN) -> dict[str, float]:
    s0, s1 = keynote_span(tl, sp)
    seg = tl.iloc[s0:s1].reset_index(drop=True)
    n = len(seg)
    minutes = n / 60
    clap = clap_score(seg)
    cheer = cheer_score(seg)
    eps = episodes(clap, ep)
    durs = np.array([b - a for a, b in eps], dtype=float)
    in_applause = episode_mask(n, eps)

    rms = seg["rms_db"].to_numpy()
    speech_only = (seg["speech"].to_numpy() >= 0.5) & ~in_applause
    rel_db = (np.median(rms[in_applause]) - np.median(rms[speech_only])
              if in_applause.any() and speech_only.any() else np.nan)
    head = min(300, n)

    return {
        "span_start_s": s0,
        "span_end_s": s1,
        "keynote_min": minutes,
        "n_applause": len(eps),
        "applause_per_min": len(eps) / minutes if minutes else np.nan,
        "applause_share": in_applause.mean() if n else np.nan,
        "applause_mean_s": durs.mean() if len(durs) else 0.0,
        "applause_max_s": durs.max() if len(durs) else 0.0,
        "applause_p90_s": np.percentile(durs, 90) if len(durs) else 0.0,
        "clap_mass": clap.mean() if n else np.nan,
        "cheer_share": (cheer >= ep.on).mean() if n else np.nan,
        "laughter_share": (seg["laughter"].to_numpy() >= ep.on).mean() if n else np.nan,
        "horn_share": (seg["horn"].to_numpy() >= ep.on).mean() if n else np.nan,
        "music_share": (seg["music"].to_numpy() >= 0.5).mean() if n else np.nan,
        "speech_share": (seg["speech"].to_numpy() >= 0.5).mean() if n else np.nan,
        "applause_rel_db": rel_db,
        "opening_applause_s": float(in_applause[:head].sum()),
        "closing_applause_s": float(in_applause[n - head :].sum()),
    }


def params_dict(ep: EpisodeParams, sp: SpanParams) -> dict[str, float]:
    return {**{f"episode_{k}": v for k, v in asdict(ep).items()},
            **{f"span_{k}": v for k, v in asdict(sp).items()}}
