from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Tuple

from openai_client import HttpTimeouts, stream_chat_completions
from sk_logging import get_logger

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

FAST_HINTS = ("summarize", "tl;dr", "quick", "brief", "short", "bullet", "one sentence", "in one paragraph")

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
)

# Explicit search-intent phrases — user is clearly asking for a web search.
_SEARCH_INTENT_HINTS: Tuple[str, ...] = (
    "search for", "search the web", "search online", "find online",
    "look up online", "google that", "find me the latest",
    "what is the latest", "what are the latest", "is there news about",
    "any news on", "any updates on", "what's happening with", "top stories",
    "today's weather", "weather forecast", "what happened today", 
)

# Keyword → category mapping for category inference.
# Only maps when the keyword is specific enough to imply a category.
_CATEGORY_KEYWORDS: dict[str, Tuple[str, ...]] = {
    "news":       ("news", "headlines", "breaking", "politics", "election",
                   "current events", "latest events"),
    "science":    ("arxiv", "research paper", "scientific study", "journal article",
                   "nasa discovery", "new study", "new research"),
    "technology": ("tech news", "software release", "github release", "hardware release",
                   "ai news", "product launch", "new update", "new version released"),
}

_DEFAULT_SEARCH_CATEGORIES: Tuple[str, ...] = ("general", "news")

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


def heuristic_is_ambiguous(score: int) -> bool:
    """
    True when the heuristic score falls in the ambiguous zone (1–2).
    Score 0   = clear FAST;  ≥3 = clear ARCHITECT;  1–2 = uncertain.
    Only the ambiguous zone benefits from the extra LLM classification round-trip.
    """
    return score in (1, 2)


def _log_decision(decision: "RouteDecision", user_text: str) -> None:
    reasons = ",".join(decision.reasons[:8]) if decision.reasons else ""
    _LOG.info(
        "route | brain=%s | score=%s | needs_search=%s | categories=%s | reasons=[%s] | input_chars=%s",
        decision.brain, decision.score, decision.needs_search,
        list(decision.search_categories), reasons, len(user_text),
    )


def _detect_search(txt: str) -> Tuple[bool, Tuple[str, ...]]:
    """
    Heuristic: decide whether this query needs live web data and which
    SearXNG categories to query.

    Returns (needs_search, search_categories).
    Conservative by design — only triggers on clear signals to avoid
    unnecessary latency on every turn.
    """
    for phrase in _SEARCH_TEMPORAL_HINTS:
        if phrase in txt:
            categories = _infer_categories(txt)
            return True, categories

    for phrase in _SEARCH_INTENT_HINTS:
        if phrase in txt:
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

    voice_mode: when True, short queries (<150 chars) receive a -1 score bias
                toward FAST, reflecting the conversational nature of voice input.
                This prevents common depth-hint words like "explain" or "why"
                from escalating brief voice questions to ARCHITECT.

    Also logs the decision to logs/sage_kaizen.log.
    """
    if not user_text:
        decision = RouteDecision(brain="FAST", reasons=["empty_input"], score=0)
        _log_decision(decision, user_text)
        return decision

    if force_architect:
        decision = RouteDecision(brain="ARCHITECT", reasons=["force_architect"], score=999)
        _log_decision(decision, user_text)
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
    if n > 2000:
        score += 4
        reasons.append("very_long_input")
    elif n > 800:
        score += 2
        reasons.append("long_input")

    # Voice-mode bias: short conversational queries lean FAST.
    # Voice input is inherently more concise than typed text; a brief spoken
    # question containing "explain" or "why" is typically FAST territory.
    if voice_mode and n < 150:
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
    if score >= 3:
        decision = RouteDecision(
            brain="ARCHITECT",
            reasons=reasons or ["score_threshold"],
            score=score,
            needs_search=needs_search,
            search_categories=search_categories,
            needs_music=needs_music,
        )
        _log_decision(decision, user_text)
        return decision

    decision = RouteDecision(
        brain="FAST",
        reasons=reasons or ["default_fast"],
        score=score,
        needs_search=needs_search,
        search_categories=search_categories,
        needs_music=needs_music,
    )
    _log_decision(decision, user_text)
    return decision


# ---------------------------------------------------
# LLM-assisted routing
# ---------------------------------------------------

_CLASSIFY_SYSTEM = (
    "Route queries. Reply with ONE word only: FAST, ARCHITECT, or SEARCH.\n"
    "FAST: simple questions, summaries, casual chat, quick facts, basic math, short creative tasks.\n"
    "ARCHITECT: deep analysis, code review, system design, multi-step reasoning, long writing.\n"
    "SEARCH: needs live/current data — news, prices, scores, recent events, new releases.\n"
    "No explanation. One word."
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
        # Cap at 500 chars — the classifier only needs enough context to judge
        {"role": "user", "content": user_text[:500]},
    ]

    try:
        chunks = list(stream_chat_completions(
            base_url=fast_base_url,
            model=model_id,
            messages=classify_messages,
            temperature=0.0,   # deterministic classification
            top_p=1.0,
            max_tokens=10,     # we only need one word (1–2 tokens)
            timeouts=timeouts,
        ))
        label = "".join(chunks).strip().upper()

        # SEARCH classification: route to FAST brain (light/quick turn) but
        # flag needs_search so context_injector runs a live web fetch.
        # Infer categories from the query text for SEARCH labels.
        needs_search   = "SEARCH" in label
        search_categories: Tuple[str, ...] = ()
        if needs_search:
            _, search_categories = _detect_search(user_text.lower())
            if not search_categories:
                search_categories = _infer_categories(user_text.lower())
            brain = "FAST"
        else:
            brain = "ARCHITECT" if "ARCHITECT" in label else "FAST"

        needs_music = _detect_music(user_text.lower())

        _LOG.info(
            "llm_route | brain=%s | needs_search=%s | needs_music=%s | label=%r | input_chars=%s",
            brain, needs_search, needs_music, label, len(user_text),
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

