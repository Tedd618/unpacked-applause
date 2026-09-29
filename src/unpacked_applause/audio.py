"""Fetch keynote audio from YouTube (audio-only) and decode it with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import yt_dlp

from .config import AUDIO_DIR, CAPTIONS_DIR, SAMPLE_RATE

# Low-bitrate Opus first (~20 MB per hour, full 48 kHz bandwidth); AAC as fallback. Videos
# with several audio tracks (YouTube auto-dubbing) only expose suffixed ids such as "251-10",
# so they fall through to the best track that is neither machine-dubbed nor a "DRC"
# (YouTube stable-volume) variant, whose compression would flatten applause loudness.
AUDIO_FORMAT = "249/250/251/140/139/bestaudio[format_note!*=?dubbed][format_note!*=?DRC]"
CAPTION_LANGS = ["en", "en-US", "en-GB", "en-orig", "en-en"]


def youtube_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def audio_path(event_id: str) -> Path | None:
    for path in AUDIO_DIR.glob(f"{event_id}.*"):
        if path.suffix not in {".part", ".ytdl", ".json"}:
            return path
    return None


def download_audio(event_id: str, youtube_id: str) -> Path:
    """Download the audio stream once; later calls hit the local cache."""
    cached = audio_path(event_id)
    if cached is not None:
        return cached
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": AUDIO_FORMAT,
        "outtmpl": str(AUDIO_DIR / f"{event_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 10,
        "fragment_retries": 10,
        "sleep_interval_requests": 1,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(youtube_url(youtube_id), download=True)
    path = audio_path(event_id)
    if path is None:
        raise FileNotFoundError(f"download produced no audio file for {event_id}")
    write_track_info(event_id, info)
    return path


def write_track_info(event_id: str, info: dict) -> None:
    """Record which audio track was downloaded (guards against dubbed tracks)."""
    fmt = (info.get("requested_formats") or [info])[0]
    keep = ["format_id", "language", "format_note", "acodec", "abr", "asr"]
    track = {k: fmt.get(k) for k in keep} | {"youtube_id": info.get("id")}
    (AUDIO_DIR / f"{event_id}.json").write_text(json.dumps(track, indent=1))


def download_captions(event_id: str, youtube_id: str) -> list[Path]:
    """Save human and auto-generated English captions (json3) for detector validation."""
    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CAPTIONS_DIR.glob(f"{event_id}.*json3"))
    if existing or (CAPTIONS_DIR / f"{event_id}.none").exists():
        return existing
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": CAPTION_LANGS,
        "subtitlesformat": "json3",
        "outtmpl": {
            "subtitle": str(CAPTIONS_DIR / f"{event_id}.%(ext)s"),
            "default": str(CAPTIONS_DIR / f"{event_id}.%(ext)s"),
        },
        "quiet": True,
        "no_warnings": True,
        "sleep_interval_requests": 1,
        "sleep_interval_subtitles": 2,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(youtube_url(youtube_id), download=True)
    # Remember which tracks were human-made so the validation can prefer them
    # (yt-dlp picks the human track over the automatic one for the same language).
    manual = sorted(set(info.get("subtitles") or {}) & set(CAPTION_LANGS))
    paths = sorted(CAPTIONS_DIR.glob(f"{event_id}.*json3"))
    (CAPTIONS_DIR / f"{event_id}.manual").write_text("\n".join(manual))
    if not paths:
        (CAPTIONS_DIR / f"{event_id}.none").touch()
    return paths


def decode(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio file to mono int16 PCM at ``sample_rate`` (kept as int16 to halve RAM)."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode()[-500:]}")
    return np.frombuffer(proc.stdout, dtype="<i2")


def rms_db_per_second(wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Loudness of each whole second in dBFS (int16 input)."""
    n_sec = int(np.ceil(len(wave) / sample_rate))
    padded = np.zeros(n_sec * sample_rate, dtype=np.float32)
    padded[: len(wave)] = wave.astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(padded.reshape(n_sec, sample_rate) ** 2, axis=1))
    return (20 * np.log10(rms + 1e-7)).astype(np.float32)
