"""Turn keynote audio into a 1 Hz timeline of sound-event probabilities with PANNs."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch

from .audio import audio_path, decode, rms_db_per_second
from .config import (
    AUDIOSET_LABELS, PANNS_CHECKPOINT, SAMPLE_RATE, TIMELINE_CLASSES, TIMELINES_DIR,
)
from .panns import SEGMENT_SAMPLES, load_model

CORE_SEGMENTS = 188  # ~60 s of output kept per window
CONTEXT_SEGMENTS = 10  # ~3.2 s of context on each side, discarded to avoid edge effects
SEGMENT_SECONDS = SEGMENT_SAMPLES / SAMPLE_RATE  # 0.32 s
QUANT = 255  # probabilities are stored as uint8 (1/255 resolution) to keep the repo small


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def class_indices() -> dict[str, int]:
    labels = pd.read_csv(AUDIOSET_LABELS)
    by_name = dict(zip(labels["display_name"], labels["index"], strict=True))
    return {col: int(by_name[name]) for col, name in TIMELINE_CLASSES.items()}


def segment_probabilities(
    wave: np.ndarray, model: torch.nn.Module, device: str, class_idx: list[int], batch_size: int = 4
) -> np.ndarray:
    """Probabilities (n_segments, n_classes) for an int16 waveform, one row per 0.32 s.

    The waveform is cut into overlapping windows; each window contributes only its central
    CORE_SEGMENTS outputs so that no output is computed near a zero-padded edge.
    """
    n_segments = math.ceil(len(wave) / SEGMENT_SAMPLES)
    n_windows = math.ceil(n_segments / CORE_SEGMENTS)
    core = CORE_SEGMENTS * SEGMENT_SAMPLES
    win = (CORE_SEGMENTS + 2 * CONTEXT_SEGMENTS) * SEGMENT_SAMPLES
    padded = np.zeros(n_windows * core + 2 * CONTEXT_SEGMENTS * SEGMENT_SAMPLES, dtype=np.int16)
    start = CONTEXT_SEGMENTS * SEGMENT_SAMPLES
    padded[start : start + len(wave)] = wave

    idx = torch.tensor(class_idx, device=device)
    out = np.zeros((n_windows * CORE_SEGMENTS, len(class_idx)), dtype=np.float32)
    with torch.inference_mode():
        for b0 in range(0, n_windows, batch_size):
            ws = range(b0, min(b0 + batch_size, n_windows))
            batch = np.stack([padded[w * core : w * core + win] for w in ws])
            x = torch.from_numpy(batch.astype(np.float32) / 32768.0).to(device)
            probs = model(x)[:, CONTEXT_SEGMENTS : CONTEXT_SEGMENTS + CORE_SEGMENTS]
            probs = probs.index_select(2, idx).cpu().numpy()
            for j, w in enumerate(ws):
                out[w * CORE_SEGMENTS : (w + 1) * CORE_SEGMENTS] = probs[j]
    return out[:n_segments]


def to_seconds(seg_probs: np.ndarray, n_seconds: int) -> np.ndarray:
    """Average the 0.32 s segments that start inside each whole second."""
    sec = np.minimum((np.arange(len(seg_probs)) * SEGMENT_SECONDS).astype(int), n_seconds - 1)
    sums = np.zeros((n_seconds, seg_probs.shape[1]), dtype=np.float64)
    np.add.at(sums, sec, seg_probs)
    counts = np.bincount(sec, minlength=n_seconds)
    return (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)


def build_timeline(event_id: str, model: torch.nn.Module, device: str,
                   classes: dict[str, int]) -> pd.DataFrame:
    path = audio_path(event_id)
    if path is None:
        raise FileNotFoundError(f"no audio for {event_id}")
    wave = decode(path)
    rms = rms_db_per_second(wave)
    per_sec = to_seconds(segment_probabilities(wave, model, device, list(classes.values())), len(rms))
    df = pd.DataFrame({"t": np.arange(len(rms), dtype=np.int32), "rms_db": rms})
    for j, col in enumerate(classes):
        df[col] = np.round(per_sec[:, j] * QUANT).astype(np.uint8)
    return df


def timeline_path(event_id: str):
    return TIMELINES_DIR / f"{event_id}.parquet"


def load_timeline(event_id: str) -> pd.DataFrame:
    df = pd.read_parquet(timeline_path(event_id))
    for col in TIMELINE_CLASSES:
        df[col] = df[col].astype(np.float32) / QUANT
    return df


def detect_all(event_ids: list[str], device: str | None = None, overwrite: bool = False) -> None:
    device = device or pick_device()
    TIMELINES_DIR.mkdir(parents=True, exist_ok=True)
    todo = [e for e in event_ids
            if audio_path(e) is not None and (overwrite or not timeline_path(e).exists())]
    if not todo:
        print("detect: nothing to do")
        return
    model = load_model(PANNS_CHECKPOINT, device)
    classes = class_indices()
    for i, event_id in enumerate(todo, 1):
        t0 = time.time()
        df = build_timeline(event_id, model, device, classes)
        df.to_parquet(timeline_path(event_id), compression="zstd", index=False)
        dt = time.time() - t0
        print(f"[{i}/{len(todo)}] {event_id}: {len(df) / 60:.0f} min audio in {dt:.0f} s "
              f"({len(df) / max(dt, 1e-9):.0f}x real time)", flush=True)
