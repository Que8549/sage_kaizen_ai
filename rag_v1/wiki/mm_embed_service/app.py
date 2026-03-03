"""
rag_v1/wiki/mm_embed_service/app.py

FastAPI service that loads jina-clip-v2 once at startup from a local directory
and exposes text and image embedding endpoints in a shared 1024-dim vector space.

Configuration is read from config/brains/brains.yaml (wiki_embed: section).
No environment variables are required.

Run standalone:
    python -m rag_v1.wiki.mm_embed_service.app
"""
from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel

from rag_v1.wiki.wiki_embed_config import load_wiki_embed_config

# ──────────────────────────────────────────────────────────────────────────── #
# Globals (populated at startup)                                                #
# ──────────────────────────────────────────────────────────────────────────── #

_model: Any = None
_device: str = "cuda:0"
_text_batch: int = 32
_image_batch: int = 8


# ──────────────────────────────────────────────────────────────────────────── #
# Lifespan (load model once, release at shutdown)                               #
# ──────────────────────────────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _text_batch, _image_batch

    cfg = load_wiki_embed_config()
    model_path  = str(cfg.model)
    _device      = os.environ.get("WIKI_EMBED_DEVICE") or cfg.device
    _text_batch  = cfg.text_batch
    _image_batch = cfg.image_batch

    print(f"[mm_embed_service] Loading jina-clip-v2 from {model_path!r} on {_device} …", flush=True)
    _model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    _model = _model.to(_device).eval()
    print("[mm_embed_service] Model ready.", flush=True)

    yield
    # No explicit cleanup needed; process exit handles GPU memory.


app = FastAPI(title="Sage Kaizen — Wiki Multimodal Embed Service", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────────────────── #
# Request / response models                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

class TextRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


class ImageRequest(BaseModel):
    images_b64: list[str]   # base64-encoded image bytes
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), dim=-1)


def _to_embeddings(raw) -> list[list[float]]:
    """Convert model output (numpy / tensor) to list[list[float]]."""
    if not isinstance(raw, torch.Tensor):
        raw = torch.tensor(raw, dtype=torch.float32)
    return raw.cpu().tolist()


# ──────────────────────────────────────────────────────────────────────────── #
# Endpoints                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": _device, "model": "jina-clip-v2"}


@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(req: TextRequest) -> EmbedResponse:
    if not req.texts:
        return EmbedResponse(embeddings=[])
    if len(req.texts) > _text_batch:
        raise HTTPException(
            status_code=422,
            detail=f"Too many texts: {len(req.texts)} > batch limit {_text_batch}. "
                   "Split into smaller batches.",
        )
    try:
        with torch.no_grad():
            raw = _model.encode_text(req.texts)
        embs = torch.tensor(raw, dtype=torch.float32) if not isinstance(raw, torch.Tensor) else raw.float()
        if req.normalize:
            embs = _normalize(embs)
        embs = torch.nan_to_num(embs, nan=0.0)
        return EmbedResponse(embeddings=embs.cpu().tolist())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc


@app.post("/embed/image", response_model=EmbedResponse)
def embed_image(req: ImageRequest) -> EmbedResponse:
    if not req.images_b64:
        return EmbedResponse(embeddings=[])
    if len(req.images_b64) > _image_batch:
        raise HTTPException(
            status_code=422,
            detail=f"Too many images: {len(req.images_b64)} > batch limit {_image_batch}. "
                   "Split into smaller batches.",
        )
    try:
        pil_images: list[Image.Image] = []
        for b64 in req.images_b64:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_images.append(img)

        with torch.no_grad():
            raw = _model.encode_image(pil_images)
        embs = torch.tensor(raw, dtype=torch.float32) if not isinstance(raw, torch.Tensor) else raw.float()
        if req.normalize:
            embs = _normalize(embs)
        embs = torch.nan_to_num(embs, nan=0.0)
        return EmbedResponse(embeddings=embs.cpu().tolist())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image embedding failed: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────── #
# CLI entrypoint: python -m rag_v1.wiki.mm_embed_service.app                   #
# ──────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import uvicorn

    cfg  = load_wiki_embed_config()
    port = int(os.environ.get("WIKI_EMBED_PORT") or cfg.port)
    uvicorn.run(
        "rag_v1.wiki.mm_embed_service.app:app",
        host=cfg.host,
        port=port,
        log_level="info",
    )
