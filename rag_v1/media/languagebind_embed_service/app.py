"""
rag_v1/media/languagebind_embed_service/app.py

FastAPI service that loads LanguageBind models once at startup and exposes
text, image, and audio embedding endpoints in a shared 768-dim CLIP-aligned
vector space.

Three modalities are supported (text, image, audio).  Video embedding is
deferred — decord has no Python 3.14 wheel; a torchvision shim will be added
in a future update.

All supported modalities share the same output dimension (768), so a single
pgvector cosine search retrieves matching assets across all media types.

Configuration is read from config/brains/brains.yaml (media_embed: section).
Device and port can be overridden at runtime via environment variables:
    MEDIA_EMBED_DEVICE    — e.g. "cuda:0" or "cpu"  (default: config value)
    MEDIA_EMBED_PORT      — e.g. "8040"              (default: config value)
    LANGUAGEBIND_REPO_DIR — path to the cloned LanguageBind repo (see below)

SETUP — runs in the main Sage Kaizen Python environment (no separate venv):
---------------------------------------------------------------------------
1. Clone the LanguageBind repo (NOT on PyPI — must be cloned manually):

   git clone https://github.com/PKU-YuanGroup/LanguageBind F:/Projects/sage_kaizen_ai/languagebind_repo

2. Install the one missing dep in the main env:

   pip install soundfile

   All other deps (torch, torchvision, transformers, accelerate, einops,
   fastapi, uvicorn, pydantic, tenacity, pyyaml) are already present in
   requirements.txt.

3. Set the repo path (or rely on media_embed.repo_dir in brains.yaml):

   set LANGUAGEBIND_REPO_DIR=F:/Projects/sage_kaizen_ai/languagebind_repo

4. The service is auto-started by MediaRetriever on first search.
   To start manually from the project root:

   python -m rag_v1.media.languagebind_embed_service.app

The service degrades gracefully (returns HTTP 503) if LanguageBind is
unavailable, so the main Sage Kaizen chat pipeline is never blocked.
---------------------------------------------------------------------------
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# ── LanguageBind repo path injection ──────────────────────────────────────── #
# languagebind is NOT on PyPI.  The repo must be cloned from GitHub and its   #
# path added to sys.path before `import languagebind` is attempted.           #
#                                                                              #
# Resolution order (first found wins):                                        #
#   1. LANGUAGEBIND_REPO_DIR environment variable                             #
#   2. media_embed.repo_dir in config/brains/brains.yaml                     #
#   3. Default fallback: F:/Projects/sage_kaizen_ai/languagebind_repo                                #
def _resolve_lb_repo() -> str:
    if env := os.environ.get("LANGUAGEBIND_REPO_DIR"):
        return env
    try:
        import yaml as _yaml
        _root = Path(__file__).resolve().parents[3]   # languagebind_embed_service → media → rag_v1 → project root
        data = _yaml.safe_load((_root / "config" / "brains" / "brains.yaml").read_text(encoding="utf-8"))
        return data.get("media_embed", {}).get("repo_dir", "F:/Projects/sage_kaizen_ai/languagebind_repo")
    except Exception:
        return "F:/Projects/sage_kaizen_ai/languagebind_repo"

_LB_REPO = _resolve_lb_repo()
if _LB_REPO and Path(_LB_REPO).is_dir() and _LB_REPO not in sys.path:
    sys.path.insert(0, _LB_REPO)
# ─────────────────────────────────────────────────────────────────────────── #

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────── #
# Config loader                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def _load_cfg() -> dict:
    """Return the media_embed section from brains.yaml."""
    from pathlib import Path as _P
    import yaml as _yaml
    _root = _P(__file__).resolve().parents[3]   # languagebind_embed_service → media → rag_v1 → project root
    data = _yaml.safe_load((_root / "config" / "brains" / "brains.yaml").read_text(encoding="utf-8"))
    return data["media_embed"]


# ──────────────────────────────────────────────────────────────────────────── #
# Globals populated at startup                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

_model: Any             = None
_tokenizer: Any         = None
_device: str            = "cpu"
_cache_dir: str         = ""
_text_batch: int        = 64
_image_batch: int       = 8
_audio_batch: int       = 4

# Video (LanguageBind_Video_V1.5_FT) requires decord which has no Python 3.14
# wheel.  It will be added once the torchvision shim is implemented.
_CLIP_TYPES = {
    "image": "LanguageBind/LanguageBind_Image",
    "audio": "LanguageBind/LanguageBind_Audio_FT",
}


# ──────────────────────────────────────────────────────────────────────────── #
# Lifespan                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device, _cache_dir
    global _text_batch, _image_batch, _audio_batch

    cfg = _load_cfg()
    svc = cfg["service"]

    _device      = os.environ.get("MEDIA_EMBED_DEVICE") or svc.get("device", "cpu")
    _cache_dir   = str(cfg.get("model_cache", "E:/languagebind_models"))
    _text_batch  = int(svc.get("text_batch",  64))
    _image_batch = int(svc.get("image_batch",  8))
    _audio_batch = int(svc.get("audio_batch",  4))

    print(f"[media_embed] Loading LanguageBind models on {_device} …", flush=True)
    print(f"[media_embed] Model cache: {_cache_dir}", flush=True)

    try:
        from languagebind import LanguageBind, LanguageBindImageTokenizer  # type: ignore[import]

        _model = LanguageBind(
            clip_type=_CLIP_TYPES,
            cache_dir=_cache_dir,
        )
        _model = _model.to(_device).eval()

        # Tokenizer from the image variant (shared text encoder across all modalities)
        _tokenizer = LanguageBindImageTokenizer.from_pretrained(
            _CLIP_TYPES["image"],
            cache_dir=_cache_dir,
        )
        print("[media_embed] LanguageBind ready.", flush=True)

    except ImportError as exc:
        print(
            f"[media_embed] ERROR: languagebind repo not found or not importable.\n"
            f"  Clone: git clone https://github.com/PKU-YuanGroup/LanguageBind F:/Projects/sage_kaizen_ai/languagebind_repo\n"
            f"  Detail: {exc}",
            flush=True,
        )
        # Service starts but all endpoints will return 503
        _model = None
        _tokenizer = None

    yield
    # GPU memory released on process exit


app = FastAPI(title="Sage Kaizen — Cross-Modal LanguageBind Embed Service", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────────── #
# Request / response models                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

class TextRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


class MediaRequest(BaseModel):
    items_b64: list[str]    # base64-encoded file bytes
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int = 768


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _check_model() -> None:
    if _model is None or _tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="LanguageBind models not loaded. Clone https://github.com/PKU-YuanGroup/LanguageBind and set LANGUAGEBIND_REPO_DIR.",
        )


def _normalize_t(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t.float(), dim=-1)


def _to_list(t: torch.Tensor) -> list[list[float]]:
    return t.cpu().tolist()


def _write_temp(raw: bytes, suffix: str) -> str:
    """Write bytes to a named temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except Exception:
        os.close(fd)
        raise
    return path


# ──────────────────────────────────────────────────────────────────────────── #
# Endpoints                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

@app.get("/health")
def health() -> dict:
    return {
        "status":  "ok" if _model is not None else "degraded",
        "device":  _device,
        "model":   "LanguageBind",
        "loaded":  _model is not None,
    }


@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(req: TextRequest) -> EmbedResponse:
    """Embed text queries into the 768-dim shared space."""
    _check_model()
    if not req.texts:
        return EmbedResponse(embeddings=[], dim=768)
    if len(req.texts) > _text_batch:
        raise HTTPException(status_code=422, detail=f"Batch too large: {len(req.texts)} > {_text_batch}")

    try:
        inputs = _tokenizer(
            req.texts,
            max_length=77,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model({"language": inputs})
        embs = out["language"].float()
        if req.normalize:
            embs = _normalize_t(embs)
        return EmbedResponse(embeddings=_to_list(embs), dim=embs.shape[-1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text embed failed: {exc}") from exc


@app.post("/embed/image", response_model=EmbedResponse)
def embed_image(req: MediaRequest) -> EmbedResponse:
    """Embed PNG/JPG images into the 768-dim shared space."""
    _check_model()
    if not req.items_b64:
        return EmbedResponse(embeddings=[], dim=768)
    if len(req.items_b64) > _image_batch:
        raise HTTPException(status_code=422, detail=f"Batch too large: {len(req.items_b64)} > {_image_batch}")

    tmp_paths: list[str] = []
    try:
        from languagebind import LanguageBindImageProcessor  # type: ignore[import]
        processor = LanguageBindImageProcessor.from_pretrained(
            _CLIP_TYPES["image"], cache_dir=_cache_dir
        )

        for b64 in req.items_b64:
            raw = base64.b64decode(b64)
            path = _write_temp(raw, suffix=".png")
            tmp_paths.append(path)

        inputs = processor(tmp_paths, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model({"image": inputs})
        embs = out["image"].float()
        if req.normalize:
            embs = _normalize_t(embs)
        return EmbedResponse(embeddings=_to_list(embs), dim=embs.shape[-1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image embed failed: {exc}") from exc
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


@app.post("/embed/audio", response_model=EmbedResponse)
def embed_audio(req: MediaRequest) -> EmbedResponse:
    """Embed WAV/MP3/FLAC audio clips into the 768-dim shared space."""
    _check_model()
    if not req.items_b64:
        return EmbedResponse(embeddings=[], dim=768)
    if len(req.items_b64) > _audio_batch:
        raise HTTPException(status_code=422, detail=f"Batch too large: {len(req.items_b64)} > {_audio_batch}")

    tmp_paths: list[str] = []
    try:
        from languagebind import LanguageBindAudioProcessor  # type: ignore[import]
        processor = LanguageBindAudioProcessor.from_pretrained(
            _CLIP_TYPES["audio"], cache_dir=_cache_dir
        )

        for b64 in req.items_b64:
            raw = base64.b64decode(b64)
            # Write as WAV; processor handles resampling internally
            path = _write_temp(raw, suffix=".wav")
            tmp_paths.append(path)

        inputs = processor(tmp_paths, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model({"audio": inputs})
        embs = out["audio"].float()
        if req.normalize:
            embs = _normalize_t(embs)
        return EmbedResponse(embeddings=_to_list(embs), dim=embs.shape[-1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio embed failed: {exc}") from exc
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────────── #
# CLI entrypoint                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import uvicorn

    cfg = _load_cfg()
    svc = cfg["service"]
    port = int(os.environ.get("MEDIA_EMBED_PORT") or svc.get("port", 8040))
    host = svc.get("host", "127.0.0.1")

    uvicorn.run(
        "rag_v1.media.languagebind_embed_service.app:app",
        host=host,
        port=port,
        log_level="info",
    )
