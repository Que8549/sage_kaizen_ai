"""
rag_v1/wiki/mm_embed_service/app.py

FastAPI service that loads jina-clip-v2 once at startup from a local directory
and exposes text and image embedding endpoints in a shared 1024-dim vector space.

Configuration is read from config/brains/brains.yaml (wiki_embed: section).
The project-root .env file is loaded at startup so that HF_TOKEN is available
to huggingface_hub before transformers is imported (suppresses unauthenticated
request warning even when local_files_only=True).

Run standalone:
    python -m rag_v1.wiki.mm_embed_service.app

Verbosity
---------
By default the service only logs key lifecycle milestones (loading, ready,
errors).  Set WIKI_EMBED_VERBOSE=1 in the environment to enable full output,
including transformers weight-loading progress bars, Python deprecation
warnings, and uvicorn access logs.  This is useful for diagnosing model load
failures or GPU/CUDA issues.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ── Load .env BEFORE any HuggingFace/transformers import ────────────────────
# huggingface_hub checks for HF_TOKEN at module-initialisation time and emits
# "unauthenticated requests" warning if the token is absent — even when
# local_files_only=True and no network calls are made.  Loading .env here (the
# only place guaranteed to run before `from transformers import …`) silences it.
# override=False: existing env vars (e.g. set by the parent process) take
# precedence so the service can still be launched with a different token.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
except Exception:
    pass  # python-dotenv not installed or .env missing — not a hard failure

# ── Verbosity gate ──────────────────────────────────────────────────────────
# Must be evaluated before any import that triggers tqdm or warnings.
_VERBOSE: bool = os.environ.get("WIKI_EMBED_VERBOSE", "").strip().lower() in (
    "1", "true", "yes", "y", "on",
)

if not _VERBOSE:
    # Suppress tqdm progress bars emitted by transformers during from_pretrained.
    # TQDM_DISABLE is checked at tqdm import time; setting it here (before
    # transformers is imported below) suppresses all tqdm output in this process.
    os.environ.setdefault("TQDM_DISABLE", "1")

    # Suppress Python warnings (flash-attn / xFormers / torch_dtype deprecation).
    import warnings
    warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC support unavailable; non-HEIC images unaffected
from pydantic import BaseModel
from transformers import AutoModel
import transformers as _transformers

# Suppress transformers INFO/WARNING logs (weight-loading messages, deprecation
# notices) unless verbose mode is active.  This must happen after importing
# transformers but before any from_pretrained call.
if not _VERBOSE:
    _transformers.logging.set_verbosity_error()

from rag_v1.wiki.wiki_embed_config import load_wiki_embed_config
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.wiki_embed_service")

# ──────────────────────────────────────────────────────────────────────────── #
# Globals (populated at startup)                                                #
# ──────────────────────────────────────────────────────────────────────────── #

_model: Any = None           # torch.compile wrapper — used for all inference
_model_base: Any = None     # pre-compiled AutoModel — used for device migration (.to())
_device: str = "cuda:0"
_device_type: str = "cuda"       # device family; derived from _device at startup
_text_batch: int = 32
_image_batch: int = 8
_loaded: bool = False
_dtype = torch.bfloat16          # BF16: native on Blackwell; recommended by jina-clip-v2

# ── Idle-offload state ─────────────────────────────────────────────────────
# After _idle_timeout_s seconds of no requests, jina-clip-v2 is moved from
# GPU to CPU, freeing ~2 GB on CUDA0 for ARCHITECT.  On the next request the
# model is automatically restored to GPU before inference runs.
#
# Threading model:
#   _restore_lock   — threading.Lock: guards the offload / restore path.
#                     Endpoint threads (FastAPI thread pool) call _ensure_on_gpu()
#                     which acquires this lock; the async idle monitor calls
#                     _offload_model() which also acquires it.  The lock window
#                     is short (one .to() + empty_cache call) so event-loop
#                     blocking from the async context is negligible.
#   _model_offloaded — bool flag: True when params are on CPU, False on GPU.
#   _last_request   — monotonic timestamp of the most recent embed request.
#   _idle_timeout_s — seconds; 0 disables idle offload entirely.
_model_offloaded: bool = False
_last_request: float = 0.0
_idle_timeout_s: float = 120.0
_restore_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────── #
# Lifespan (load model once, release at shutdown)                               #
# ──────────────────────────────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_base, _device, _device_type, _text_batch, _image_batch
    global _loaded, _idle_timeout_s, _last_request

    cfg = load_wiki_embed_config()
    model_path   = str(cfg.model)
    _device      = os.environ.get("WIKI_EMBED_DEVICE") or cfg.device
    _device_type = _device.split(":")[0]
    _text_batch  = cfg.text_batch
    _image_batch = cfg.image_batch
    _idle_timeout_s = float(
        os.environ.get("WIKI_EMBED_IDLE_TIMEOUT_S") or cfg.idle_timeout_s
    )

    # ── PyTorch backend tuning ──────────────────────────────────────────────
    #
    # cudnn.benchmark=False: transformer models receive variable sequence lengths
    # per batch; re-benchmarking on every new shape causes net slowdowns.
    torch.backends.cudnn.benchmark = False

    # TF32 for residual FP32 ops (matmuls outside autocast scope).
    # Uses set_float32_matmul_precision — the forward-compatible API that replaces
    # torch.backends.cuda.matmul.allow_tf32, which is deprecated in PyTorch 2.9+.
    torch.set_float32_matmul_precision("high")

    # SDPA backend priority for Blackwell (RTX 5090/5080, SM_120):
    #   cuDNN attention > Flash SDP > Memory-Efficient SDP > (math disabled)
    # cuDNN 9 (bundled with PyTorch 2.7+ cu128) is specifically optimised for
    # Blackwell and provides ~10% E2E improvement over Flash SDP alone.
    # PyTorch PR #145602 added native sm_120 support to Flash and Mem-Efficient
    # backends, so the external flash-attn package is not needed.
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)   # unoptimised fallback; never needed

    # ── Model load ─────────────────────────────────────────────────────────
    _LOG.info(
        "Loading jina-clip-v2 from %r on %s (bf16, sdpa) …", model_path, _device,
    )
    # device_map streams weights directly from disk to GPU — no CPU staging copy.
    # This sets low_cpu_mem_usage automatically, halving the peak RAM footprint.
    # use_safetensors prefers model.safetensors over pytorch_model.bin (faster,
    # no pickle deserialization).
    _model_base = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        dtype=_dtype,
        device_map={"": _device},
    )
    _model_base = _model_base.eval()

    # ── torch.compile ──────────────────────────────────────────────────────
    # max-autotune-no-cudagraphs: benchmarks Triton tile configs to find the
    # fastest matmul/attention kernel for our exact batch shapes (one-time cost
    # at startup; cache persists across restarts via TORCHINDUCTOR_CACHE_DIR).
    # "no-cudagraphs" variant avoids pytorch/pytorch#171672 where CUDA graphs
    # are re-instantiated on every iteration rather than replayed, causing
    # reduce-overhead / max-autotune to be slower than default on affected builds.
    # fullgraph=False: jina-clip-v2 uses trust_remote_code custom Python paths
    # (LoRA adapters, task instructions) that contain graph breaks; False is safe.
    _LOG.info("Compiling model with torch.compile (max-autotune-no-cudagraphs) …")
    _model = torch.compile(_model_base, mode="max-autotune-no-cudagraphs", fullgraph=False)

    # ── Warmup ─────────────────────────────────────────────────────────────
    # Triggers JIT compilation so the first real request is not delayed.
    _LOG.info("Running warmup inference (triggers JIT compilation) …")
    with torch.inference_mode(), torch.autocast(_device_type, dtype=_dtype):
        _ = _model.encode_text(["warmup"])

    _loaded = True
    _last_request = time.monotonic()
    _LOG.info(
        "Model ready — device=%s idle_timeout_s=%.0f verbose=%s",
        _device, _idle_timeout_s, _VERBOSE,
    )

    # ── Idle monitor ───────────────────────────────────────────────────────
    # Polls every 30 s; offloads GPU memory when idle_timeout_s is exceeded.
    # Disabled when idle_timeout_s == 0 (keep model resident).
    if _idle_timeout_s > 0:
        asyncio.create_task(_idle_monitor())

    yield
    # No explicit cleanup needed; process exit handles GPU memory.


# ──────────────────────────────────────────────────────────────────────────── #
# Idle offload / GPU restore                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

async def _idle_monitor() -> None:
    """
    Background asyncio task: offload jina-clip-v2 to CPU after idle_timeout_s
    seconds of no embed requests.  Runs every 30 s; terminates when the server
    shuts down (asyncio.CancelledError on lifespan exit).
    """
    global _model_offloaded
    while True:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return
        if _idle_timeout_s <= 0 or _model_offloaded:
            continue
        idle_secs = time.monotonic() - _last_request
        if idle_secs >= _idle_timeout_s:
            _offload_model()


def _offload_model() -> None:
    """
    Move jina-clip-v2 parameters from GPU to CPU and empty the CUDA cache.
    No-op if already offloaded.  Thread-safe via _restore_lock.
    """
    global _model_offloaded
    with _restore_lock:
        if _model_offloaded:
            return
        _LOG.info(
            "jina-clip-v2 idle for %.0f s — offloading to CPU to free ~2 GB on CUDA0 …",
            time.monotonic() - _last_request,
        )
        _model_base.to("cpu")
        if _device_type == "cuda":
            torch.cuda.empty_cache()
        _model_offloaded = True
        _LOG.info("jina-clip-v2 offloaded to CPU.  CUDA0 VRAM freed.")


def _ensure_on_gpu() -> None:
    """
    Restore jina-clip-v2 to GPU if it was offloaded, then update _last_request.
    Called at the start of every embed endpoint.  Thread-safe via _restore_lock.

    The warmup forward pass inside the lock re-traces the compiled graph for
    the GPU device.  TORCHINDUCTOR_CACHE_DIR means subsequent traces are fast
    (kernel configs cached from the initial startup compile).
    """
    global _model_offloaded, _last_request
    if not _model_offloaded:
        _last_request = time.monotonic()
        return
    with _restore_lock:
        if not _model_offloaded:          # another thread restored it first
            _last_request = time.monotonic()
            return
        _LOG.info("Restoring jina-clip-v2 to %s …", _device)
        _model_base.to(_device)
        # Warmup: triggers Dynamo re-tracing on the GPU device.
        # Uses _model (compiled wrapper) which delegates to _model_base.
        with torch.inference_mode(), torch.autocast(_device_type, dtype=_dtype):
            _ = _model.encode_text(["warmup"])
        _model_offloaded = False
        _last_request = time.monotonic()
        _LOG.info("jina-clip-v2 restored to %s and ready.", _device)


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


def _to_float_tensor(raw: Any) -> torch.Tensor:
    """Coerce model output (numpy array or tensor) to a float32 CPU tensor."""
    if isinstance(raw, torch.Tensor):
        return raw.float()
    return torch.tensor(raw, dtype=torch.float32)


def _ready_or_503() -> None:
    """Raise 503 if the model has not finished loading yet."""
    if not _loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Retry shortly.")


# ──────────────────────────────────────────────────────────────────────────── #
# Endpoints                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": _device,
        "model": "jina-clip-v2",
        "loaded": _loaded,
        "offloaded": _model_offloaded,
        "idle_timeout_s": _idle_timeout_s,
    }


@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(req: TextRequest) -> EmbedResponse:
    _ready_or_503()
    _ensure_on_gpu()
    if not req.texts:
        return EmbedResponse(embeddings=[])
    if len(req.texts) > _text_batch:
        raise HTTPException(
            status_code=422,
            detail=f"Too many texts: {len(req.texts)} > batch limit {_text_batch}. "
                   "Split into smaller batches.",
        )
    try:
        with torch.inference_mode(), torch.autocast(_device_type, dtype=_dtype):
            raw = _model.encode_text(req.texts)
        embs = _to_float_tensor(raw)
        if req.normalize:
            embs = _normalize(embs)
        embs = torch.nan_to_num(embs, nan=0.0)
        return EmbedResponse(embeddings=embs.cpu().tolist())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc


@app.post("/embed/image", response_model=EmbedResponse)
def embed_image(req: ImageRequest) -> EmbedResponse:
    _ready_or_503()
    _ensure_on_gpu()
    if not req.images_b64:
        return EmbedResponse(embeddings=[])
    if len(req.images_b64) > _image_batch:
        raise HTTPException(
            status_code=422,
            detail=f"Too many images: {len(req.images_b64)} > batch limit {_image_batch}. "
                   "Split into smaller batches.",
        )
    try:
        pil_images: list[Image.Image] = [
            Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            for b64 in req.images_b64
        ]
        with torch.inference_mode(), torch.autocast(_device_type, dtype=_dtype):
            raw = _model.encode_image(pil_images)
        embs = _to_float_tensor(raw)
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
        # In quiet mode: suppress uvicorn startup banners and all access logs.
        # In verbose mode: full uvicorn output including per-request access logs.
        log_level="info" if _VERBOSE else "warning",
        access_log=_VERBOSE,
    )
