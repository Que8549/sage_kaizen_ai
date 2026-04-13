"""
chat_service.py

Encapsulates the full single-turn lifecycle:
    route → ensure servers → build prompts → RAG inject → stream response

The UI layer (ui_streamlit_server.py) calls this class and renders the
yielded chunks.  It has no direct knowledge of routing internals, prompt
assembly, or RAG.

Multimodal support:
    Attach MediaAttachment objects to a turn via TurnConfig.media_attachments.
    Images and audio are serialised as OpenAI-compatible content-part arrays
    and sent to llama-server's /v1/chat/completions endpoint.
    Video is handled client-side (frame extraction) and arrives here as
    multiple image (video_frame) attachments.

    Routing:
      - Images / video frames → FAST (Qwen2.5-Omni has combined audio+vision mmproj;
                                       ARCHITECT mmproj disabled — loading it kills speculative
                                       decoding, cache_reuse, and swa_full)
      - Audio                 → FAST (Qwen2.5-Omni audio encoder; Qwen3.5-27B has none)
      - Documents (txt/code/docx/xlsx) → ARCHITECT (128K context; deep analysis)
"""
from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import router as _router
from document_parser import DocumentAttachment, format_document_context
from inference_session import InferenceSession
from memory.bundle_builder import format_bundle_prompt
from memory.models import EpisodeWriteRequest, MemoryContextRequest
from memory.writer import write_episode
from openai_client import HttpTimeouts, LlamaServerError, stream_chat_completions
from prompt_library import (
    TemplateKey,
    build_system_only,
    sage_architect_core,
    sage_fast_core,
)
from rag_v1.runtime.context_injector import apply_rag_and_wiki_parallel
from router import RouteDecision, route as heuristic_route, heuristic_is_ambiguous
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.chat_service")

# ---------------------------------------------------------------------------
# Memory service — lazy singleton with graceful degradation.
# If the memory schema has not been applied yet (scripts/memory_schema.sql),
# MemoryService() will fail on first use.  The singleton catches that and
# disables memory for the session rather than crashing the app.
# ---------------------------------------------------------------------------
_MEMORY_SVC: Optional["MemoryService"] = None  # type: ignore[name-defined]
_MEMORY_DISABLED = False   # set True after a failed init so we don't retry every turn
_MEMORY_LOCK = threading.Lock()


def _get_memory() -> Optional[object]:
    global _MEMORY_SVC, _MEMORY_DISABLED
    if _MEMORY_DISABLED:
        return None
    if _MEMORY_SVC is not None:
        return _MEMORY_SVC
    with _MEMORY_LOCK:
        # Double-checked locking: another thread may have initialised while we waited.
        if _MEMORY_SVC is None and not _MEMORY_DISABLED:
            try:
                from memory.service import MemoryService
                _MEMORY_SVC = MemoryService()
                _LOG.info("memory | MemoryService initialised")
            except Exception as exc:
                _MEMORY_DISABLED = True
                _LOG.warning(
                    "memory | MemoryService unavailable — memory disabled for this session. "
                    "Apply scripts/memory_schema.sql to enable. Error: %s", exc,
                )
    return _MEMORY_SVC

# Thinking-token cap applied automatically to creative writing turns when the
# user has not set an explicit budget (thinking_budget == -1).  Creative tasks
# benefit from a short planning phase but 11+ minutes of uncapped thinking adds
# no measurable quality gain — the output is driven by base model training.
# Set to 0 to disable thinking entirely for creative routes by default.
_CREATIVE_THINKING_CAP = 2048

# Keyword lists for auto-template selection
_TEACH_HINTS      = ("teach", "tutor", "explain like", "for a 3rd grader", "for a 10th grader")
_KNOWLEDGE_HINTS  = ("history", "civilization", "ancient", "timeline", "religion", "religious")
_PHILOSOPHY_HINTS = ("philosophy", "theology", "ethics", "meaning", "debate")


# ─────────────────────────────────────────────────────────────────────────── #
# MediaAttachment                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class MediaAttachment:
    """
    A single piece of media to include in the user turn.

    Attributes
    ----------
    kind       : "image" | "audio" | "video_frame"
                 video_frame is a still extracted from a video file and treated
                 identically to "image" when building the content array.
    data_b64   : Base64-encoded raw bytes of the file.
    mime_type  : MIME type string, e.g. "image/jpeg", "audio/wav".
    label      : Short human-readable label shown in the UI (filename, etc.).
    frame_index: For video_frame attachments, the 0-based frame number.
    """
    kind: str               # "image" | "audio" | "video_frame"
    data_b64: str           # base64-encoded bytes
    mime_type: str          # e.g. "image/jpeg", "audio/wav"
    label: str = ""
    frame_index: int = 0    # only meaningful for kind="video_frame"

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        kind: str,
        mime_type: str,
        label: str = "",
        frame_index: int = 0,
    ) -> "MediaAttachment":
        return cls(
            kind=kind,
            data_b64=base64.b64encode(raw).decode("ascii"),
            mime_type=mime_type,
            label=label,
            frame_index=frame_index,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# TurnConfig                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class TurnConfig:
    """
    All per-turn generation parameters supplied by the user via the sidebar.
    Frozen so it can be passed around safely.

    Sampling notes (model card recommendations):
      FAST  (Qwen2.5-Omni-7B):  temp=0.7, top_p=0.80, top_k=40,  min_p=0.05
      ARCH  (Qwen3.5-27B think): temp=0.6, top_p=0.95, top_k=20,  min_p=0.0
      ARCH  (Qwen3.5-27B plain): temp=0.7, top_p=0.80, top_k=20,  min_p=0.0
    """
    deep_mode: bool
    auto_escalate: bool
    auto_templates: bool
    override_templates: Tuple[TemplateKey, ...]
    temperature_q5: float
    temperature_q6: float
    top_p_q5: float
    top_p_q6: float
    top_k_q5: int
    top_k_q6: int
    min_p_q5: float
    min_p_q6: float
    max_tokens_q5: int
    max_tokens_q6: int
    # ARCHITECT thinking budget (-1 = unlimited, 0 = off, N > 0 = cap at N tokens).
    # Sent as reasoning_budget in the chat completions payload.
    # Creative turns auto-cap at _CREATIVE_THINKING_CAP when this is -1.
    thinking_budget: int = -1
    # Multimodal attachments for this turn (empty tuple = text-only)
    media_attachments: Tuple[MediaAttachment, ...] = field(default_factory=tuple)
    # Text-based document attachments (txt, code files, docx, xlsx).
    # Content is extracted to plain text and injected into the user message.
    # Presence of any document attachment forces routing to ARCHITECT (27B, 128K context).
    document_attachments: Tuple[DocumentAttachment, ...] = field(default_factory=tuple)


# ─────────────────────────────────────────────────────────────────────────── #
# ChatService                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

class ChatService:
    """
    Owns the business logic for a single conversation turn.

    Usage (in the Streamlit UI):

        service = ChatService(session, CONFIG.system_prompt, timeouts)

        decision  = service.decide_route(user_text, cfg)
        templates = service.select_templates(user_text, cfg)
        messages, rag_sources, wiki_images, search_evidence, music_context = service.prepare_messages(
            user_text, history, decision, templates,
            media_attachments=cfg.media_attachments,
        )

        for chunk in service.stream_response(messages, decision, cfg):
            accumulated += chunk
    """

    def __init__(
        self,
        session: InferenceSession,
        system_prompt: str,
        timeouts: HttpTimeouts,
    ) -> None:
        self._session = session
        self._system_prompt = system_prompt
        self._timeouts = timeouts
        self._status_timeouts = HttpTimeouts(connect_s=2.0, read_s=5.0)

    # ------------------------------------------------------------------ #
    # Routing                                                              #
    # ------------------------------------------------------------------ #

    def decide_route(self, user_text: str, cfg: TurnConfig) -> RouteDecision:
        """
        Determine which brain to use for this turn.

        Priority order:
          1. Empty input → FAST
          2. Image/video/audio attachments → FAST (Qwen2.5-Omni has combined audio+vision
             mmproj; ARCHITECT mmproj is disabled to preserve speculative decoding)
          3. auto_escalate disabled → respect deep_mode toggle only
          4. deep_mode on → ARCHITECT (unconditional)
          5. Q5 is live → ask FAST brain to classify (LLM routing)
          6. Q5 not live → keyword-scoring heuristic (no server needed)
        """
        if not user_text and not cfg.media_attachments and not cfg.document_attachments:
            return RouteDecision(brain="FAST", reasons=["empty_input"], score=0)

        # Document routing:
        #   Any text/code/Office document attachment → ARCHITECT (Qwen3.5-27B, 128K context).
        #   FAST (Qwen2.5-Omni-7B) only has 16K context; large documents would overflow it.
        #   Documents need deep analysis that ARCHITECT handles more reliably than FAST.
        if cfg.document_attachments:
            n_docs  = len(cfg.document_attachments)
            total_c = sum(a.char_count for a in cfg.document_attachments)
            types   = sorted({a.doc_type for a in cfg.document_attachments})
            return RouteDecision(
                brain="ARCHITECT",
                reasons=[f"document_upload:{n_docs}_files", f"doc_chars:{total_c}"],
                score=999,
                modality=f"document:{','.join(types)}",
            )

        # Multimodal routing:
        #   All media (image, video_frame, audio) → FAST (Qwen2.5-Omni-7B).
        #   Qwen2.5-Omni has a combined audio+vision mmproj encoder.
        #   ARCHITECT mmproj is disabled: loading it silently kills ngram-map-k speculative
        #   decoding, cache_reuse, and swa_full at startup, halving generation throughput.
        if cfg.media_attachments:
            kinds = sorted({a.kind for a in cfg.media_attachments})
            has_vision = any(a.kind in ("image", "video_frame") for a in cfg.media_attachments)
            has_audio  = any(a.kind == "audio" for a in cfg.media_attachments)
            if has_vision and has_audio:
                modality = "multimodal"
            elif has_vision:
                modality = "video" if any(a.kind == "video_frame" for a in cfg.media_attachments) else "image"
            else:
                modality = "audio"
            return RouteDecision(
                brain="FAST",
                reasons=[f"multimodal:{','.join(kinds)}", "fast_mmproj"],
                score=0,
                modality=modality,
            )

        if not cfg.auto_escalate:
            return self._manual_decision(cfg.deep_mode)

        if cfg.deep_mode:
            return RouteDecision(brain="ARCHITECT", reasons=["manual_deep_mode"], score=999)

        # Two-tier routing:
        #   1. Run instant keyword heuristic first (0 ms, no server call).
        #   2. Only for ambiguous scores (1-2) call the FAST brain to classify.
        #      Clear FAST (score=0) and clear ARCHITECT (score≥3) skip the LLM
        #      round-trip entirely, saving ~500ms per non-ambiguous turn.
        heuristic = heuristic_route(user_text, force_architect=False)
        if not heuristic_is_ambiguous(heuristic.score):
            return heuristic

        # Ambiguous zone — ask FAST brain for a tie-break (only when it's running)
        q5_up, _ = self._session.health_q5(self._status_timeouts)
        if q5_up:
            try:
                return _router.llm_route(
                    user_text=user_text,
                    fast_base_url=self._session.q5_url,
                    model_id=self._session.q5_model_id,
                    timeouts=self._status_timeouts,
                )
            except Exception:
                _LOG.warning("LLM routing failed; using heuristic result (score=%d)", heuristic.score)

        return heuristic

    @staticmethod
    def _manual_decision(deep_mode: bool) -> RouteDecision:
        if deep_mode:
            return RouteDecision(brain="ARCHITECT", reasons=["manual_deep_mode"], score=999)
        return RouteDecision(brain="FAST", reasons=["manual_fast_mode"], score=0)

    # ------------------------------------------------------------------ #
    # Template selection                                                   #
    # ------------------------------------------------------------------ #

    def select_templates(
        self, user_text: str, cfg: TurnConfig
    ) -> Tuple[TemplateKey, ...]:
        if cfg.override_templates:
            return cfg.override_templates
        if not cfg.auto_templates:
            return ()
        return self._auto_templates(user_text)

    @staticmethod
    def _auto_templates(user_text: str) -> Tuple[TemplateKey, ...]:
        txt = (user_text or "").lower()
        keys: List[TemplateKey] = [
            TemplateKey.UNIVERSAL_DEPTH_ANCHOR,
            TemplateKey.AUTO_ADAPTIVE_META,
        ]
        if any(k in txt for k in _TEACH_HINTS):
            keys.append(TemplateKey.TEACHING_TUTORING)
        if any(k in txt for k in _KNOWLEDGE_HINTS):
            keys.append(TemplateKey.STRUCTURED_KNOWLEDGE)
        if any(k in txt for k in _PHILOSOPHY_HINTS):
            keys.append(TemplateKey.PHILOSOPHY_DEEP_THINKING)
        return tuple(dict.fromkeys(keys))

    # ------------------------------------------------------------------ #
    # Message assembly                                                     #
    # ------------------------------------------------------------------ #

    def prepare_messages(
        self,
        user_text: str,
        history: List[dict],
        decision: RouteDecision,
        templates: Tuple[TemplateKey, ...],
        wiki_enabled: bool = True,
        media_attachments: Tuple[MediaAttachment, ...] = (),
        document_attachments: Tuple[DocumentAttachment, ...] = (),
        session_id: Optional[str] = None,
        user_id: str = "alquin",
    ) -> Tuple[List[dict], list, list, object, str]:
        """
        Build the full OpenAI-style messages list for this turn.

        When media_attachments is non-empty the user message content is a
        list of content-part dicts (OpenAI vision / audio format) instead of
        a plain string.  llama-server (with --mmproj) routes these to the
        Qwen2.5-Omni encoders.

        When document_attachments is non-empty, extracted document text is
        prepended to the user message as <document> blocks so ARCHITECT has
        the full file context before reading the user's question.

        Returns:
            (messages, rag_sources, wiki_images, search_evidence, music_context)
            search_evidence is a SearchEvidence or None when search was not triggered.
            music_context is "" when music retrieval was not triggered.
        """
        core = sage_architect_core if decision.brain == "ARCHITECT" else sage_fast_core
        system_content = build_system_only(
            system_prompt=self._system_prompt,
            core_prompt=core,
            templates=templates,
        )

        # Memory injection — retrieve bundle and prepend to system message.
        # Token caps: FAST=600, ARCHITECT=1500 (enforced inside bundle_builder).
        # Gracefully skipped if MemoryService is unavailable (schema not applied).
        memory_svc = _get_memory()
        if memory_svc is not None and user_text:
            try:
                mem_req = MemoryContextRequest(
                    user_id=user_id,
                    project_id="sage_kaizen",
                    session_id=session_id,
                    query_text=user_text,
                    route_target=decision.brain.lower(),
                )
                bundle = memory_svc.get_memory_bundle(mem_req)
                memory_segment = format_bundle_prompt(bundle)
                if memory_segment:
                    system_content = memory_segment + "\n\n" + system_content if system_content else memory_segment
                    _LOG.debug(
                        "prepare_messages | memory injected: %d items ~%d tokens",
                        bundle.total_items, bundle.estimated_tokens,
                    )
            except Exception as exc:
                _LOG.warning("prepare_messages | memory retrieval failed (non-fatal): %s", exc)
                err_str = str(exc).lower()
                if "permission denied" in err_str or "does not exist" in err_str:
                    global _MEMORY_DISABLED
                    _MEMORY_DISABLED = True
                    _LOG.warning("prepare_messages | memory disabled (schema not accessible); will not retry")

        messages: List[dict] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})

        if history:
            messages.extend(history[:-1])

        # Build document context block — prepended before the user query so the
        # model has the full file content in view when it reads the question.
        doc_context = format_document_context(document_attachments)
        if doc_context:
            _LOG.info(
                "prepare_messages: injecting %d document(s), total ~%d chars",
                len(document_attachments),
                sum(a.char_count for a in document_attachments),
            )

        # Build user content — plain string or multimodal content-part list
        if media_attachments:
            user_content = _build_multimodal_content(
                user_text, media_attachments, doc_context=doc_context
            )
        else:
            # Prepend document blocks to the plain text query when documents are present
            if doc_context:
                user_content = doc_context + "\n\n" + user_text.strip()
            else:
                user_content = user_text.strip()

        messages.append({"role": "user", "content": user_content})

        # RAG injection operates on the text query regardless of modality.
        # It appends context to the last user turn's text portion.
        # Prefer the dedicated CPU summarizer when configured (decouples
        # summarization from the FAST brain's single inference slot).
        messages, rag_sources, wiki_images, search_evidence, music_context = apply_rag_and_wiki_parallel(
            messages, user_text, decision, wiki_enabled,
            fast_base_url=self._session.q5_url,
            fast_model_id=self._session.q5_model_id,
            summarizer_base_url=self._session.summarizer_url or None,
            summarizer_model_id=self._session.summarizer_model_id or None,
        )
        return messages, rag_sources, wiki_images, search_evidence, music_context

    # ------------------------------------------------------------------ #
    # Post-turn memory write                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def write_episode_background(
        user_text: str,
        assistant_text: str,
        decision: RouteDecision,
        session_id: Optional[str] = None,
        user_id: str = "alquin",
    ) -> None:
        """
        Fire-and-forget post-turn episodic memory write.

        Runs in a daemon thread so it never blocks the UI.
        Applies the selective write policy (memory/policy.py) — short acks
        and greetings are dropped automatically.

        Call this after collecting all streamed chunks:

            for chunk in service.stream_response(...):
                accumulated += chunk
            ChatService.write_episode_background(user_text, accumulated, decision,
                                                 session_id=session_id, user_id=user_id)
        """
        memory_svc = _get_memory()
        if memory_svc is None:
            return

        def _bg() -> None:
            try:
                req = EpisodeWriteRequest(
                    user_id=user_id,
                    project_id="sage_kaizen",
                    session_id=session_id,
                    scope="project",
                    event_type="general",
                    summary_text=user_text,        # full text — policy enforces selective write
                    raw_excerpt=assistant_text,    # full assistant response
                    importance=0.5,
                    confidence=0.6,
                )
                result = write_episode(req)
                if result:
                    _LOG.debug("memory | episode written id=%s brain=%s", result, decision.brain)
            except Exception as exc:
                _LOG.warning("memory | post-turn episode write failed (non-fatal): %s", exc)

        threading.Thread(target=_bg, name="memory-episode-write", daemon=True).start()

    # ------------------------------------------------------------------ #
    # Streaming                                                            #
    # ------------------------------------------------------------------ #

    def stream_response(
        self,
        messages: List[dict],
        decision: RouteDecision,
        cfg: TurnConfig,
    ) -> Iterator[str]:
        """
        Yield text chunks from the selected llama-server.

        For ARCHITECT turns, resolves the effective thinking budget:
          - cfg.thinking_budget != -1 → use it as-is (explicit user override)
          - cfg.thinking_budget == -1 and creative route → cap at _CREATIVE_THINKING_CAP
          - cfg.thinking_budget == -1 and non-creative → unlimited (-1)

        Raises LlamaServerError if the server returns an HTTP error.
        """
        base_url = self._session.url_for_brain(decision.brain)
        model_id = self._session.model_id_for_brain(decision.brain)
        is_arch = decision.brain == "ARCHITECT"

        # Resolve thinking budget for ARCHITECT turns only.
        thinking_budget: int = -1
        if is_arch:
            if cfg.thinking_budget != -1:
                # Explicit user setting — honour it directly.
                thinking_budget = cfg.thinking_budget
            elif any(r.startswith("creative:") for r in decision.reasons):
                # Auto-cap: creative writing doesn't benefit from unlimited thinking.
                thinking_budget = _CREATIVE_THINKING_CAP
                _LOG.info(
                    "stream_response: auto-capping thinking to %d tokens (creative route)",
                    _CREATIVE_THINKING_CAP,
                )
            # else: -1 (unlimited) — code review, architecture, analysis, etc.

        yield from stream_chat_completions(
            base_url=base_url,
            model=model_id,
            messages=messages,
            temperature=float(cfg.temperature_q6 if is_arch else cfg.temperature_q5),
            top_p=float(cfg.top_p_q6 if is_arch else cfg.top_p_q5),
            top_k=int(cfg.top_k_q6 if is_arch else cfg.top_k_q5),
            min_p=float(cfg.min_p_q6 if is_arch else cfg.min_p_q5),
            max_tokens=int(cfg.max_tokens_q6 if is_arch else cfg.max_tokens_q5),
            thinking_budget=thinking_budget,
            timeouts=self._timeouts,
        )


# ─────────────────────────────────────────────────────────────────────────── #
# Multimodal content builder                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_multimodal_content(
    user_text: str,
    attachments: Tuple[MediaAttachment, ...],
    *,
    doc_context: str = "",
) -> list:
    """
    Build an OpenAI-compatible multimodal content-part list.

    Images / video frames → image_url data URI (base64 JPEG/PNG).
    Audio → input_audio with base64 data and format string.
    Text → text part appended last so the model sees media first.
    Document context → prepended to the text part when present.

    llama-server (with --mmproj) parses these parts and routes each to
    the appropriate encoder inside the mmproj GGUF.
    """
    parts: list = []

    for att in attachments:
        if att.kind in ("image", "video_frame"):
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{att.mime_type};base64,{att.data_b64}",
                },
            })
        elif att.kind == "audio":
            # Extract format from mime_type: "audio/wav" → "wav"
            fmt = att.mime_type.split("/")[-1].lower()
            if fmt == "mpeg":
                fmt = "mp3"
            parts.append({
                "type": "input_audio",
                "input_audio": {
                    "data": att.data_b64,
                    "format": fmt,
                },
            })
        else:
            _LOG.warning("Unknown attachment kind %r — skipped", att.kind)

    # Text always goes last so the model receives media context before the query.
    # When documents are also attached, their content is prepended to the text part.
    query_text = user_text.strip()
    if doc_context:
        query_text = doc_context + ("\n\n" + query_text if query_text else "")
    if query_text:
        parts.append({"type": "text", "text": query_text})

    return parts
