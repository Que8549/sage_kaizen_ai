from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time
from typing import List, Tuple

from rapidfuzz import fuzz as _fuzz

from openai_client import HttpTimeouts, stream_chat_completions
from sk_logging import get_logger

# ── Routing thresholds (tune here without touching logic) ──────────────────
ARCHITECT_THRESHOLD : int = 3    # heuristic score required to select ARCHITECT
VOICE_BIAS_THRESHOLD: int = 150  # chars below which voice_mode applies a -1 score bias
VERY_LONG_INPUT     : int = 2000 # chars above which +4 score is applied
LONG_INPUT          : int = 800  # chars above which +2 score is applied
LLM_CAP_CHARS       : int = 500  # max chars passed to the LLM classifier

DEPTH_HINTS = (
    "explain", "analyze", "compare", "why", "how", "history", "philosophy", "theology",
    "deep", "in depth", "detailed", "step-by-step", "teach", "tutor", "architecture",
    "design", "tradeoff", "pros and cons", "evaluate", "optimize", "tune", "take time to think",
    "double check your answer", "religious", "religion", "psychology",
)

CODE_HINTS = (
    "code", "python", "c#", "typescript", "javascript", "rust", "go lang", "golang",
    "sql", "bash", "shell script", "powershell", "regex", "function", "class definition",
    "debug", "stack trace", "error", "traceback", "exception", "refactor", "unit test",
    "postgresql",
)

FAST_HINTS = ("summarize", "tl;dr", "quick", "brief", "short answer", "keep it short", "bullet", "one sentence", "in one paragraph")

# Creative writing — long-form narrative needs ARCHITECT for language stability.
# Qwen2.5-Omni-7B (FAST brain) code-switches to Chinese mid-response on long
# creative tasks. Qwen3.5-27B (ARCHITECT) is far more stable at this length.
CREATIVE_HINTS = (
    "write a story", "write a short story", "write a poem", "write a song",
    "write a scene", "write a chapter", "write a script", "write a screenplay",
    "write a novel", "write an essay", "write a blog post",
    "write me a story", "write me a poem", "write me a song", "write me a scene",
    "write me a short story", "write me an essay",
    "tell me a story", "compose a poem", "compose a song",
    "creative writing", "fiction story", "short fiction",
)

_VS_RE     = re.compile(r"(?:^|[\s\W])vs(?:$|[\s\W])")
_VERSUS_RE = re.compile(r"(?:^|[\s\W])versus(?:$|[\s\W])")

# ---------------------------------------------------------------------------
# Search detection — temporal keywords and explicit search intent
# ---------------------------------------------------------------------------

# Temporal phrases that strongly imply a need for live/current information.
# Intentionally conservative to avoid false positives on common words like
# "today I want to learn..." (no search needed) vs "what happened today" (search needed).
_SEARCH_TEMPORAL_HINTS: Tuple[str, ...] = (
    "today's news", "latest news", "breaking news",
    "current events", "right now", "this week's", "this month's",
    "just announced", "recently announced", "just released", "newly released",
    "trending now", "live score", "live results", "stock price", "current price",
    # Weather — present tense; no static model can answer these
    "current weather", "weather right now", "weather outside",
    "current temperature", "temperature right now", "temperature outside",
    "is it raining", "is it snowing", "is it hot", "is it cold",
    "what is the weather", "what's the weather", "how's the weather",
    "weather in", "weather for", "weather near", "weather at",
    "temperature in", "temperature for",
    # Weather — future/forecast; no static model can answer these either
    "tomorrow's weather", "weather tomorrow",
    "will it rain", "will it snow", "will it be cold", "will it be hot", "will it be warm",
    "is it going to rain", "is it going to snow",
    "weekend weather", "this weekend's weather", "next week's weather",
    "what will the weather be",
    "chance of rain", "chance of snow",
    "hourly forecast", "7-day forecast", "10-day forecast",
    # Sports results — outcome is always live data
    "final score", "game results", "match results", "who won",
    "today's game", "tonight's game", "yesterday's game",
    # Finance — live market/energy data
    "bitcoin price", "crypto price", "gas price", "oil price",
    "exchange rate", "market today", "stock market today",
    # News — recent past and ongoing updates
    "what happened yesterday", "latest on",
    # News — natural "top news" variants that don't use "today's news" word order
    "top news", "today's top", "what's the news", "what is the news",
    "what's happening today", "what's going on today",
)

# Explicit search-intent phrases — user is clearly asking for a web search.
_SEARCH_INTENT_HINTS: Tuple[str, ...] = (
    "search for", "search the web", "search online", "find online",
    "look up online", "google that", "find me the latest",
    "what is the latest", "what are the latest", "is there news about",
    "any news on", "any updates on", "what's happening with", "top stories",
    "today's weather", "weather forecast", "what happened today",
    "weather today", "forecast today", "forecast for", "current",
    # Version/release queries — always live data
    "latest version of", "current version of", "newest version of",
)

# Keyword → category mapping for category inference.
# Only maps when the keyword is specific enough to imply a category.
_CATEGORY_KEYWORDS: dict[str, Tuple[str, ...]] = {
    "news":       ("news", "headlines", "breaking", "politics", "election",
                   "current events", "latest events",
                   "final score", "game results", "match results", "who won",
                   "today's game", "tonight's game"),
    "science":    ("arxiv", "research paper", "scientific study", "journal article",
                   "nasa discovery", "new study", "new research"),
    "technology": ("tech news", "software release", "github release", "hardware release",
                   "ai news", "product launch", "new update", "new version released",
                   "latest version of", "current version of"),
    "general":    ("weather", "temperature", "forecast", "rain", "snow",
                   "humidity", "wind speed", "uv index",
                   "bitcoin", "crypto", "gas price", "oil price", "exchange rate",
                   "stock market"),
}

_DEFAULT_SEARCH_CATEGORIES: Tuple[str, ...] = ("general", "news")

# ---------------------------------------------------------------------------
# Fuzzy phrase matching — catches paraphrases the exact lists miss
# ---------------------------------------------------------------------------
# Threshold calibration:
#   76 is the lowest value that avoids the most common false positive:
#   "latest Python version" (token_set_ratio vs "latest news" ≈ 73).
#   It still catches clear paraphrases like "what's today's top news?" vs
#   "top news" (ratio = 100 when one phrase's tokens are a full subset).
_FUZZY_SEARCH_THRESHOLD = 76


def _fuzzy_matches_any(
    txt: str,
    phrases: Tuple[str, ...],
    threshold: int = _FUZZY_SEARCH_THRESHOLD,
) -> bool:
    """
    Two-stage phrase match against a tuple of reference phrases.

    Stage 1 — exact substring (zero overhead, handles all pre-enumerated phrases).
    Stage 2 — token_set_ratio fuzzy match, applied only to phrases with ≥ 2 words.
      Single-word phrases are excluded from fuzzy to prevent false positives
      (e.g. "current" matching "current best practice").

    token_set_ratio splits both strings into word tokens, sorts them, and
    computes the ratio of the sorted-intersection vs. sorted-remainder pairs.
    When one phrase's tokens are fully contained in the other, the score is 100,
    making it ideal for catching word-order variants and natural extensions
    (e.g. "today's top news stories" matching "top news").
    """
    for phrase in phrases:
        if phrase in txt:
            return True
    for phrase in phrases:
        if len(phrase.split()) >= 2 and _fuzz.token_set_ratio(txt, phrase) >= threshold:
            return True
    return False

# ---------------------------------------------------------------------------
# Music query detection
# ---------------------------------------------------------------------------

_MUSIC_HINTS: Tuple[str, ...] = (
    "play something", "play me a song", "find songs", "find a song", "find me a song",
    "list songs", "list tracks", "music for", "songs about", "songs with",
    "tracks about", "tracks with", "what songs", "which songs", "any songs",
    "song that says", "song with the lyric", "lyrics about", "contains the word",
    "sings about", "song where someone", "songs like", "sounds like",
    "similar to", "more like this", "find more like",
    "make a playlist", "create a playlist", "generate a playlist", "playlist for",
    "instrumental", "with vocals", "no vocals", "bpm", "beats per minute",
    "in the key of", "key of c", "key of d", "key of e", "key of f",
    "key of g", "key of a", "key of b", "explicit songs", "clean songs",
    "fast songs", "slow songs", "upbeat tracks", "songs that sound",
    "group my music", "music cluster", "songs in my library",
)


def _detect_music(txt: str) -> bool:
    """Return True when the query looks like a music library search."""
    return any(h in txt for h in _MUSIC_HINTS)


_LOG = get_logger("sage_kaizen.router")


@dataclass(frozen=True)
class RouteDecision:
    brain: str                 # "FAST" (5080) or "ARCHITECT" (5090)
    reasons: List[str]
    score: int
    needs_search: bool = False                      # True → run live web search this turn
    search_categories: Tuple[str, ...] = field(default_factory=tuple)  # SearXNG categories to query
    needs_music: bool = False                       # True → run music retrieval this turn
    modality: str = "text"                          # "text" | "image" | "audio" | "video" | "multimodal"


def heuristic_is_ambiguous(score: int) -> bool:
    """
    True when the heuristic score falls in the ambiguous zone (1 to ARCHITECT_THRESHOLD-1).
    Score 0   = clear FAST;  ≥ARCHITECT_THRESHOLD = clear ARCHITECT;  in-between = uncertain.
    Only the ambiguous zone benefits from the extra LLM classification round-trip.
    """
    return 0 < score < ARCHITECT_THRESHOLD


def _log_decision(decision: "RouteDecision", user_text: str, processing_time_ms: float = 0.0) -> None:
    reasons = ",".join(decision.reasons[:8]) if decision.reasons else ""
    _LOG.info(
        "route | brain=%s | score=%s | modality=%s | needs_search=%s | categories=%s | reasons=[%s] | input_chars=%s | processing_time_ms=%s",
        decision.brain, decision.score, decision.modality, decision.needs_search,
        list(decision.search_categories), reasons, len(user_text), processing_time_ms,
    )
    _LOG.info(
        "route_json %s",
        json.dumps({
            "route":               decision.brain.lower(),
            "score":               decision.score,
            "modality":            decision.modality,
            "reasons":             decision.reasons[:8],
            "rag_needed":          decision.score >= ARCHITECT_THRESHOLD or decision.needs_search,
            "search_used":         decision.needs_search,
            "search_categories":   list(decision.search_categories),
            "music_used":          decision.needs_music,
            "input_chars":         len(user_text),
            "processing_time_ms":  processing_time_ms,
        }),
    )


def _detect_search(txt: str) -> Tuple[bool, Tuple[str, ...]]:
    """
    Heuristic: decide whether this query needs live web data and which
    SearXNG categories to query.

    Returns (needs_search, search_categories).
    Uses two-stage matching: exact substring first, then token-set fuzzy
    match for multi-word phrases (see _fuzzy_matches_any).
    """
    if _fuzzy_matches_any(txt, _SEARCH_TEMPORAL_HINTS) or \
       _fuzzy_matches_any(txt, _SEARCH_INTENT_HINTS):
        categories = _infer_categories(txt)
        return True, categories
    return False, ()


def _infer_categories(txt: str) -> Tuple[str, ...]:
    """Map query text to the most relevant SearXNG categories."""
    cats: list[str] = []
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in txt for kw in keywords):
            cats.append(cat)
    return tuple(cats) if cats else _DEFAULT_SEARCH_CATEGORIES


def route(
    user_text: str,
    force_architect: bool = False,
    voice_mode: bool = False,
) -> RouteDecision:
    """
    Returns a routing decision:
      - FAST      -> 5080 (Qwen2.5-Omni-7B Q8_0, multimodal)
      - ARCHITECT -> 5090 (Qwen3.5-27B-Uncensored Q6_K, thinking+multimodal)

    voice_mode: when True, short queries (<VOICE_BIAS_THRESHOLD chars) receive a -1 score bias
                toward FAST, reflecting the conversational nature of voice input.
                This prevents common depth-hint words like "explain" or "why"
                from escalating brief voice questions to ARCHITECT.

    Also logs the decision to logs/sage_kaizen.log.
    """
    _t0 = time.perf_counter()

    if not user_text:
        decision = RouteDecision(brain="FAST", reasons=["empty_input"], score=0)
        _log_decision(decision, user_text, round((time.perf_counter() - _t0) * 1000, 2))
        return decision

    if force_architect:
        decision = RouteDecision(brain="ARCHITECT", reasons=["force_architect"], score=999)
        _log_decision(decision, user_text, round((time.perf_counter() - _t0) * 1000, 2))
        return decision

    txt = user_text.lower()
    score = 0
    reasons: List[str] = []

    # Live-search detection (independent of brain routing)
    needs_search, search_categories = _detect_search(txt)
    if needs_search:
        reasons.append("search:live_data")

    # Music query detection (independent of brain routing)
    needs_music = _detect_music(txt)
    if needs_music:
        reasons.append("music:library_query")

    # Length heuristics
    n = len(txt)
    if n > VERY_LONG_INPUT:
        score += 4
        reasons.append("very_long_input")
    elif n > LONG_INPUT:
        score += 2
        reasons.append("long_input")

    # Voice-mode bias: short conversational queries lean FAST.
    # Voice input is inherently more concise than typed text; a brief spoken
    # question containing "explain" or "why" is typically FAST territory.
    if voice_mode and n < VOICE_BIAS_THRESHOLD:
        score -= 1
        reasons.append("voice_short_query")

    # Depth hints (moderate)
    for k in DEPTH_HINTS:
        if k in txt:
            score += 2
            reasons.append(f"depth:{k}")
            break

    # Code hints (strong)
    for k in CODE_HINTS:
        if k in txt:
            score += 3
            reasons.append(f"code:{k}")
            break

    # Creative writing (strong) — routes to ARCHITECT for language stability.
    # Qwen2.5-Omni-7B code-switches to Chinese on long-form creative tasks;
    # Qwen3.5-27B handles multi-paragraph narrative reliably in English.
    for k in CREATIVE_HINTS:
        if k in txt:
            score += 3
            reasons.append(f"creative:{k}")
            break

    # Multi-part markers (weak)
    if " and " in txt or " also " in txt:
        score += 1
        reasons.append("multi_part_marker")

    if _VS_RE.search(txt) or _VERSUS_RE.search(txt):
        score += 1
        reasons.append("comparison_marker")

    # Fast intent markers (counterweight)
    for k in FAST_HINTS:
        if k in txt:
            score -= 2
            reasons.append(f"fast_intent:{k}")
            break

    # Final threshold
    if score >= ARCHITECT_THRESHOLD:
        decision = RouteDecision(
            brain="ARCHITECT",
            reasons=reasons or ["score_threshold"],
            score=score,
            needs_search=needs_search,
            search_categories=search_categories,
            needs_music=needs_music,
        )
        _log_decision(decision, user_text, round((time.perf_counter() - _t0) * 1000, 2))
        return decision

    decision = RouteDecision(
        brain="FAST",
        reasons=reasons or ["default_fast"],
        score=score,
        needs_search=needs_search,
        search_categories=search_categories,
        needs_music=needs_music,
    )
    _log_decision(decision, user_text, round((time.perf_counter() - _t0) * 1000, 2))
    return decision


# ---------------------------------------------------
# LLM-assisted routing
# ---------------------------------------------------

_CLASSIFY_SYSTEM = (
    "Route queries. Reply with ONE label only: FAST, ARCHITECT, SEARCH, or ARCHITECT+SEARCH.\n"
    "FAST: simple questions, summaries, casual chat, quick facts, basic math, short creative tasks.\n"
    "ARCHITECT: deep analysis, code review, system design, multi-step reasoning, long writing.\n"
    "SEARCH: needs live/current data (news, prices, scores, recent events) — no deep analysis required.\n"
    "ARCHITECT+SEARCH: complex analysis that ALSO requires live current data "
    "(e.g. 'deeply analyze today's AI news trends', 'compare this week's stock movements').\n"
    "No explanation. One label."
)


def llm_route(
    user_text: str,
    fast_base_url: str,
    model_id: str,
    timeouts: HttpTimeouts,
    force_architect: bool = False,
) -> RouteDecision:
    """
    Ask the FAST brain to classify query complexity in ~1 token.

    Requires the FAST brain (Q5) to already be running.  If the call fails
    for any reason (server error, timeout, unexpected output) this function
    falls back to the keyword-scoring heuristic route() automatically.

    Parameters
    ----------
    user_text:       The raw user query.
    fast_base_url:   Base URL of the FAST brain (e.g. http://127.0.0.1:8011).
    model_id:        Model alias reported by /v1/models.
    timeouts:        HttpTimeouts — use short values (e.g. connect=2s, read=5s).
    force_architect: If True, skip classification and return ARCHITECT immediately.
    """
    if force_architect:
        decision = RouteDecision(brain="ARCHITECT", reasons=["force_architect"], score=999)
        _log_decision(decision, user_text)
        return decision

    if not user_text:
        decision = RouteDecision(brain="FAST", reasons=["empty_input"], score=0)
        _log_decision(decision, user_text)
        return decision

    classify_messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        # Cap at LLM_CAP_CHARS — the classifier only needs enough context to judge
        {"role": "user", "content": user_text[:LLM_CAP_CHARS]},
    ]

    _t0 = time.perf_counter()

    try:
        chunks = list(stream_chat_completions(
            base_url=fast_base_url,
            model=model_id,
            messages=classify_messages,
            temperature=0.0,   # deterministic classification
            top_p=1.0,
            max_tokens=10,     # we only need one label (1–3 tokens)
            timeouts=timeouts,
        ))
        label = "".join(chunks).strip().upper()

        # "ARCHITECT+SEARCH" → brain=ARCHITECT, needs_search=True
        # "ARCHITECT"        → brain=ARCHITECT, needs_search=False
        # "SEARCH"           → brain=FAST,      needs_search=True
        # "FAST"             → brain=FAST,       needs_search=False
        needs_search = "SEARCH" in label
        brain        = "ARCHITECT" if "ARCHITECT" in label else "FAST"

        search_categories: Tuple[str, ...] = ()
        if needs_search:
            _, search_categories = _detect_search(user_text.lower())
            if not search_categories:
                search_categories = _infer_categories(user_text.lower())

        needs_music = _detect_music(user_text.lower())
        processing_time_ms = round((time.perf_counter() - _t0) * 1000, 2)

        _LOG.info(
            "llm_route | brain=%s | needs_search=%s | needs_music=%s | label=%r | input_chars=%s | processing_time_ms=%s",
            brain, needs_search, needs_music, label, len(user_text), processing_time_ms,
        )
        score = 999 if brain == "ARCHITECT" else 0
        decision = RouteDecision(
            brain=brain,
            reasons=["llm_classification"],
            score=score,
            needs_search=needs_search,
            search_categories=search_categories,
            needs_music=needs_music,
        )
        return decision

    except Exception:
        _LOG.warning("llm_route failed; falling back to heuristic route()")
        return route(user_text, force_architect=False)

