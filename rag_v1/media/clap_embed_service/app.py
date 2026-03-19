"""
rag_v1/media/clap_embed_service/app.py

FastAPI service: laion/clap-htsat-unfused audio + text embeddings.

Endpoints
---------
GET  /health              — {"loaded": true} once model is ready
POST /embed/text          — embed a list of strings → 512-dim L2-normalized vectors
POST /embed/audio         — embed a list of base64-encoded raw audio byte strings → 512-dim vectors

Audio requirements
------------------
  - Raw bytes of any format readable by soundfile (WAV, FLAC, OGG, MP3 via libsndfile)
  - The service resamples to 48 kHz mono before feeding CLAP
  - Do NOT pre-process audio; send raw file bytes encoded as base64

Environment variables
---------------------
  CLAP_MODEL_DIR   — path to the local clap-htsat-unfused directory
                     default: E:/clap-htsat-unfused
  CLAP_DEVICE      — torch device string, default: cuda:1
  CLAP_PORT        — port to listen on, default: 8040
  CLAP_HOST        — host to bind, default: 127.0.0.1

Run (standalone):
    python -m rag_v1.media.clap_embed_service.app

Run (via uvicorn directly):
    uvicorn rag_v1.media.clap_embed_service.app:app --port 8040
"""
from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_LOG = logging.getLogger("sage_kaizen.clap_embed")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ──────────────────────────────────────────────────────────────────────────── #
# Config from environment                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

_MODEL_DIR  = os.environ.get("CLAP_MODEL_DIR", "E:/clap-htsat-unfused")
_DEVICE_STR = os.environ.get("CLAP_DEVICE",    "cuda:1")
_PORT       = int(os.environ.get("CLAP_PORT",  "8040"))
_HOST       = os.environ.get("CLAP_HOST",      "127.0.0.1")

_TARGET_SR  = 48_000          # CLAP expects 48 kHz mono
_MAX_DUR_S  = 10.0            # clip audio to 10 s to avoid OOM

# ──────────────────────────────────────────────────────────────────────────── #
# Model globals (set in lifespan)                                                #
# ──────────────────────────────────────────────────────────────────────────── #

_model:     Any = None
_processor: Any = None
_device:    torch.device | None = None
_loaded:    bool = False


# ──────────────────────────────────────────────────────────────────────────── #
# Lifespan                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor, _device, _loaded

    from transformers import ClapModel, ClapProcessor

    _device = torch.device(_DEVICE_STR if torch.cuda.is_available() else "cpu")
    _LOG.info("[clap_embed] Loading CLAP model on %s", _device)
    _LOG.info("[clap_embed] Model dir: %s", _MODEL_DIR)

    _processor = ClapProcessor.from_pretrained(_MODEL_DIR, local_files_only=True)
    _model = ClapModel.from_pretrained(_MODEL_DIR, local_files_only=True)
    _model.eval()
    _model.to(_device)

    _loaded = True
    _LOG.info("[clap_embed] CLAP model ready on %s", _device)

    yield

    _LOG.info("[clap_embed] Shutting down.")


app = FastAPI(title="CLAP Embed Service", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────────── #
# Request / response schemas                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

class TextRequest(BaseModel):
    texts: list[str]


class AudioRequest(BaseModel):
    # Each item is a base64-encoded string of the raw audio file bytes.
    audios_b64: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


# ──────────────────────────────────────────────────────────────────────────── #
# Audio decoding helpers                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

def _decode_audio(raw: bytes) -> np.ndarray:
    """
    Decode raw audio bytes to a float32 mono array at _TARGET_SR (48 kHz).

    Uses soundfile for decoding and torchaudio.functional.resample for SRC.
    Returns a 1-D float32 numpy array clipped to _MAX_DUR_S.
    """
    import soundfile as sf

    with io.BytesIO(raw) as buf:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)

    # Mix down to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample to 48 kHz if needed
    if sr != _TARGET_SR:
        import torchaudio.functional as taf
        t = torch.from_numpy(data).unsqueeze(0)          # (1, T)
        t = taf.resample(t, orig_freq=sr, new_freq=_TARGET_SR)
        data = t.squeeze(0).numpy()

    # Clip to max duration
    max_samples = int(_TARGET_SR * _MAX_DUR_S)
    if len(data) > max_samples:
        data = data[:max_samples]

    return data.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────── #
# Embedding helpers                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

def _embed_texts(texts: list[str]) -> list[list[float]]:
    inputs = _processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        # transformers >= 4.40: get_text_features returns BaseModelOutputWithPooling;
        # .pooler_output is already L2-normalized inside the model.
        feats = _model.get_text_features(**inputs).pooler_output  # (B, 512)

    return feats.cpu().float().tolist()


def _embed_audios(arrays: list[np.ndarray]) -> list[list[float]]:
    inputs = _processor(
        audio=arrays,
        return_tensors="pt",
        padding=True,
        sampling_rate=_TARGET_SR,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        # transformers >= 4.40: get_audio_features returns BaseModelOutputWithPooling;
        # .pooler_output is already L2-normalized inside the model.
        feats = _model.get_audio_features(**inputs).pooler_output  # (B, 512)

    return feats.cpu().float().tolist()


# ──────────────────────────────────────────────────────────────────────────── #
# Routes                                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

@app.get("/health")
def health():
    return {"loaded": _loaded, "device": str(_device), "model_dir": _MODEL_DIR}


@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(req: TextRequest):
    if not _loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not req.texts:
        return EmbedResponse(embeddings=[])
    try:
        vecs = _embed_texts(req.texts)
    except Exception as exc:
        _LOG.exception("Text embed failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return EmbedResponse(embeddings=vecs)


@app.post("/embed/audio", response_model=EmbedResponse)
def embed_audio(req: AudioRequest):
    if not _loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not req.audios_b64:
        return EmbedResponse(embeddings=[])

    arrays: list[np.ndarray] = []
    for i, b64 in enumerate(req.audios_b64):
        try:
            raw = base64.b64decode(b64)
            arrays.append(_decode_audio(raw))
        except Exception as exc:
            _LOG.warning("Audio decode failed for item %d: %s", i, exc)
            # Return a zero vector for failed items so batch indices stay aligned
            arrays.append(np.zeros(_TARGET_SR, dtype=np.float32))

    try:
        vecs = _embed_audios(arrays)
    except Exception as exc:
        _LOG.exception("Audio embed failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return EmbedResponse(embeddings=vecs)


# ──────────────────────────────────────────────────────────────────────────── #
# Entrypoint                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    uvicorn.run(
        "rag_v1.media.clap_embed_service.app:app",
        host=_HOST,
        port=_PORT,
        log_level="info",
    )
