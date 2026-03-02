"""
chat_service.py

Encapsulates the full single-turn lifecycle:
    route → ensure servers → build prompts → RAG inject → stream response

The UI layer (ui_streamlit_server.py) calls this class and renders the
yielded chunks.  It has no direct knowledge of routing internals, prompt
assembly, or RAG.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import router as _router
from inference_session import InferenceSession
from openai_client import HttpTimeouts, LlamaServerError, stream_chat_completions
from prompt_library import (
    TemplateKey,
    build_messages,
    build_system_only,
    sage_architect_core,
    sage_fast_core,
)
from router import RouteDecision, route as heuristic_route
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.chat_service")

# Keyword lists for auto-template selection (kept here; UI no longer needs them)
_TEACH_HINTS = ("teach", "tutor", "explain like", "for a 3rd grader", "for a 10th grader")
_KNOWLEDGE_HINTS = ("history", "civilization", "ancient", "timeline", "religion", "religious")
_PHILOSOPHY_HINTS = ("philosophy", "theology", "ethics", "meaning", "debate")


# ─────────────────────────────────────────────────────────────────────────── #
# Data classes                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class TurnConfig:
    """
    All per-turn generation parameters supplied by the user via the sidebar.
    Frozen so it can be passed around safely.
    """
    deep_mode: bool
    auto_escalate: bool
    auto_templates: bool
    override_templates: Tuple[TemplateKey, ...]
    temperature_q5: float
    temperature_q6: float
    top_p_q5: float
    top_p_q6: float
    max_tokens_q5: int
    max_tokens_q6: int


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
        messages  = service.prepare_messages(user_text, history, decision, templates)

        for chunk in service.stream_response(messages, decision, cfg):
            accumulated += chunk
            live.markdown(accumulated)
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
        # Short timeouts used for health checks and LLM routing (non-blocking)
        self._status_timeouts = HttpTimeouts(connect_s=2.0, read_s=5.0)

    # ------------------------------------------------------------------ #
    # Routing                                                              #
    # ------------------------------------------------------------------ #

    def decide_route(self, user_text: str, cfg: TurnConfig) -> RouteDecision:
        """
        Determine which brain to use for this turn.

        Priority order:
          1. Empty input → FAST
          2. auto_escalate disabled → respect deep_mode toggle only
          3. deep_mode on → ARCHITECT (unconditional)
          4. Q5 is live → ask FAST brain to classify (LLM routing)
          5. Q5 not live → keyword-scoring heuristic (no server needed)
        """
        if not user_text:
            return RouteDecision(brain="FAST", reasons=["empty_input"], score=0)

        if not cfg.auto_escalate:
            return self._manual_decision(cfg.deep_mode)

        if cfg.deep_mode:
            return RouteDecision(brain="ARCHITECT", reasons=["manual_deep_mode"], score=999)

        # LLM routing: only works when Q5 is already running
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
                _LOG.warning("LLM routing failed; falling back to heuristic")

        return heuristic_route(user_text, force_architect=False)

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
        """
        Return the template tuple for this turn.

        If the user has selected override_templates in the sidebar, those
        are used as-is and auto-selection is skipped entirely.
        """
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
    ) -> Tuple[List[dict], list, list]:
        """
        Build the full OpenAI-style messages list for this turn:
            [system + core + templates] + prior_history + [current user turn]

        History is inserted BEFORE the current user message so the last message
        is always role=user — required by Qwen3 (enable_thinking rejects assistant prefill).

        RAG context is injected into the user turn (not the system message) so
        the model treats it as ephemeral data rather than a persistent instruction.

        Returns:
            (messages, rag_sources, wiki_images)
            - rag_sources: list[RetrievedChunk] for inline citations
            - wiki_images: list[WikiImage] for Streamlit image rendering
        """
        core = sage_architect_core if decision.brain == "ARCHITECT" else sage_fast_core
        system_content = build_system_only(
            system_prompt=self._system_prompt,
            core_prompt=core,
            templates=templates,
        )

        messages: List[dict] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Prior turns go BEFORE the current user message.
        # history[-1] is the current user turn (already captured in user_text).
        if history:
            messages.extend(history[:-1])

        messages.append({"role": "user", "content": user_text.strip()})

        messages, rag_sources = _router.apply_rag(messages, user_text, decision)
        messages, wiki_images = _router.apply_wiki_rag(messages, user_text, decision, wiki_enabled)
        return messages, rag_sources, wiki_images

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

        Raises LlamaServerError if the server returns an HTTP error.
        The caller is responsible for catching and displaying the error.
        """
        base_url = self._session.url_for_brain(decision.brain)
        model_id = self._session.model_id_for_brain(decision.brain)
        is_arch = decision.brain == "ARCHITECT"

        yield from stream_chat_completions(
            base_url=base_url,
            model=model_id,
            messages=messages,
            temperature=float(cfg.temperature_q6 if is_arch else cfg.temperature_q5),
            top_p=float(cfg.top_p_q6 if is_arch else cfg.top_p_q5),
            max_tokens=int(cfg.max_tokens_q6 if is_arch else cfg.max_tokens_q5),
            timeouts=self._timeouts,
        )
