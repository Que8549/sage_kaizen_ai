"""
rag_v1/media/audio_analysis.py

Per-file audio attribute extraction.  Results are stored as jsonb keys in
media_files.metadata so they can be queried with SQL operators.

Extracted attributes
--------------------
  bpm            float    — beats per minute (librosa beat_track)
  key            str      — musical key, e.g. "C major" (Krumhansl-Schmuckler)
  has_vocals     bool     — CLAP zero-shot: audio embedding vs text probes
  is_explicit    bool     — lyrics keyword scan (requires lyrics table row)

Vocal detection strategy
------------------------
  The CLAP model (laion/clap-htsat-unfused) is jointly trained on audio and
  text.  We embed two probe strings — "song with singing and vocals" and
  "instrumental music no singing no vocals" — via the running CLAP service
  (port 8040), then compute cosine similarity against the stored audio
  embedding for each file.  The side with higher similarity wins.

  This requires:
    1. The file already has a row in audio_embeddings (Phase 1 complete).
    2. The CLAP service (port 8040) is running.

  No audio re-processing is needed — the stored embedding is reused.

Thread safety
-------------
  extract_bpm_key() is CPU-bound (librosa/numpy).  It is safe to call from
  multiple ThreadPoolExecutor workers; each call loads audio independently.
  clap_vocal_probes() makes a single HTTP call to the CLAP service and is
  also thread-safe (httpx client per call).
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_LOG = logging.getLogger("sage_kaizen.audio_analysis")


# ---------------------------------------------------------------------------
# OS-level stderr suppressor
# ---------------------------------------------------------------------------
# libsndfile (used by soundfile for MP3) writes "Illegal Audio-MPEG-Header"
# warnings directly to OS file descriptor 2, bypassing Python's sys.stderr.
# contextlib.redirect_stderr() cannot suppress them — only dup2(devnull, 2)
# works at the kernel level.
#
# This is safe here because:
#   - Any real decode failure raises an exception (caught by the caller).
#   - The suppression window is limited to the single librosa.load() call.
#   - Python's sys.stderr is restored via the saved duplicate fd.
#
# On Windows, os.devnull = 'nul'; on Linux/macOS = '/dev/null'.

@contextlib.contextmanager
def _suppress_c_stderr():
    """Suppress C-library stderr output (e.g. mpg123 decode warnings)."""
    try:
        # Save a duplicate of the real stderr fd before redirecting
        saved_fd = os.dup(2)
    except OSError:
        yield
        return
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        yield
    finally:
        # Restore the real stderr fd
        os.dup2(saved_fd, 2)
        os.close(saved_fd)

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles
# ---------------------------------------------------------------------------
_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float32)

_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float32)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

# ---------------------------------------------------------------------------
# Explicit content keyword list
# ---------------------------------------------------------------------------
_EXPLICIT_WORDS: frozenset[str] = frozenset({
    "fuck", "fuckin", "fucking", "motherfuck", "motherfucking",
    "muthafuck", "muthafucka", "shit", "bullshit",
    "bitch", "bitches", "pussy", "pussies",
    "nigga", "niggas", "nigger", "niggers",
    "dick", "cock", "cunt", "ass", "asshole",
    "hoe", "hoes", "whore", "whores", "slut", "slutty",
    "sex", "suck", "suckin", "blowjob", "cum",
})


# ---------------------------------------------------------------------------
# BPM + Key
# ---------------------------------------------------------------------------

def extract_bpm_key(path: Path) -> tuple[float | None, str | None]:
    """
    Load audio with librosa and return (bpm, key_string).

    Returns (None, None) on any failure so callers can skip gracefully.
    Only the first 90 s of audio is analysed — enough for reliable tempo
    and key estimation without loading the entire file.
    """
    try:
        import librosa  # noqa: PLC0415
    except ImportError:
        _LOG.warning("librosa not installed — BPM/key extraction skipped")
        return None, None

    try:
        # Load at native sample rate, mono, first 90 s only.
        # Suppress C-level stderr to squelch libsndfile/mpg123 "Illegal
        # Audio-MPEG-Header" notes that appear on MP3s with embedded LYRICS3
        # tags (harmless — the decoder resyncs and loads the audio correctly).
        with _suppress_c_stderr():
            y, sr = librosa.load(str(path), sr=None, mono=True, duration=90.0)
    except Exception as exc:
        _LOG.debug("librosa.load failed for %s: %s", path.name, exc)
        return None, None

    # ── BPM ──────────────────────────────────────────────────────────────
    bpm: float | None = None
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # beat_track returns a 0-d array in newer librosa versions
        bpm = float(np.asarray(tempo).ravel()[0])
    except Exception as exc:
        _LOG.debug("BPM extraction failed for %s: %s", path.name, exc)

    # ── Key (Krumhansl-Schmuckler) ────────────────────────────────────────
    key: str | None = None
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1).astype(np.float32)

        best_r = -np.inf
        best_label = "C major"
        for i, note in enumerate(_NOTE_NAMES):
            maj = np.roll(_MAJOR_PROFILE, i)
            mn  = np.roll(_MINOR_PROFILE, i)
            r_maj = float(np.corrcoef(chroma_mean, maj)[0, 1])
            r_min = float(np.corrcoef(chroma_mean, mn)[0, 1])
            if r_maj > best_r:
                best_r = r_maj
                best_label = f"{note} major"
            if r_min > best_r:
                best_r = r_min
                best_label = f"{note} minor"
        key = best_label
    except Exception as exc:
        _LOG.debug("Key extraction failed for %s: %s", path.name, exc)

    return bpm, key


# ---------------------------------------------------------------------------
# Vocal detection (CLAP zero-shot)
# ---------------------------------------------------------------------------

_VOCAL_PROBE    = "song with singing and vocals"
_INST_PROBE     = "instrumental music no singing no vocals"
_VOCAL_THRESH   = 0.02   # sim(vocals) must exceed sim(instrumental) by this margin


def clap_vocal_probes(
    clap_host: str = "127.0.0.1",
    clap_port: int = 8040,
) -> tuple[list[float], list[float]] | None:
    """
    Embed the two vocal probe strings via the CLAP service.
    Returns (vocal_vec, instrumental_vec) or None if service unavailable.
    Call once per ingest run; reuse the vectors for all files.
    """
    import httpx  # noqa: PLC0415

    url = f"http://{clap_host}:{clap_port}/embed/text"
    try:
        r = httpx.post(
            url,
            json={"texts": [_VOCAL_PROBE, _INST_PROBE]},
            timeout=15.0,
        )
        r.raise_for_status()
        vecs = r.json()["embeddings"]
        return vecs[0], vecs[1]
    except Exception as exc:
        _LOG.warning("CLAP vocal probe failed: %s", exc)
        return None


def classify_vocals(
    audio_vec: list[float],
    vocal_probe: list[float],
    inst_probe: list[float],
) -> bool:
    """
    Given a stored audio embedding and the two probe vectors, return True
    if the audio resembles vocals more than instrumental.

    Uses cosine similarity (vectors are already L2-normalised by CLAP service).
    """
    a  = np.array(audio_vec,   dtype=np.float32)
    vp = np.array(vocal_probe, dtype=np.float32)
    ip = np.array(inst_probe,  dtype=np.float32)
    sim_vocal = float(np.dot(a, vp))
    sim_inst  = float(np.dot(a, ip))
    return (sim_vocal - sim_inst) > _VOCAL_THRESH


# ---------------------------------------------------------------------------
# Explicit content (lyrics keyword scan)
# ---------------------------------------------------------------------------

def is_explicit_from_lyrics(lyrics_text: str) -> bool:
    """
    Return True if the lyrics contain explicit-content words.
    Case-insensitive, whole-word matching via simple tokenisation.
    """
    words = set(lyrics_text.lower().split())
    return bool(words & _EXPLICIT_WORDS)
