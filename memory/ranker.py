"""
memory/ranker.py
Reciprocal Rank Fusion (RRF) + multi-signal scoring for episode retrieval.

Design:
- Combines lexical (FTS) and vector (HNSW) result lists via RRF.
- Applies multi-signal final score: semantic + lexical + recency + importance + confidence + scope.
- Filters contradictions and duplicates before returning top-k.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .schemas import EpisodeRow

# RRF constant — controls influence of rank position.
# k=60 is the standard literature value; use 20 for shorter lists.
_RRF_K = 60


def rrf_fuse(
    lexical: list[tuple[EpisodeRow, float]],
    vector: list[tuple[EpisodeRow, float]],
) -> list[tuple[EpisodeRow, float]]:
    """
    Reciprocal Rank Fusion of two ranked lists (lexical + vector).

    RRF formula: score(d) = sum_r [ 1.0 / (k + rank_r(d)) ]
    where rank is 1-based.

    Returns a merged list sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = {}
    id_to_row: dict[str, EpisodeRow] = {}

    for rank, (row, _) in enumerate(lexical, start=1):
        rrf_scores[row.id] = rrf_scores.get(row.id, 0.0) + 1.0 / (_RRF_K + rank)
        id_to_row[row.id] = row

    for rank, (row, _) in enumerate(vector, start=1):
        rrf_scores[row.id] = rrf_scores.get(row.id, 0.0) + 1.0 / (_RRF_K + rank)
        id_to_row[row.id] = row

    merged = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(id_to_row[eid], score) for eid, score in merged]


def score_episodes(
    fused: list[tuple[EpisodeRow, float]],
    scope_filter: str | None = None,
) -> list[tuple[EpisodeRow, float]]:
    """
    Apply multi-signal final scoring:

        final = 0.35*semantic + 0.20*lexical + 0.15*recency + 0.10*importance
               + 0.10*confidence + 0.10*scope_match
    where 'semantic' and 'lexical' are approximated via the RRF score (normalised).

    Returns the same list re-scored and re-sorted.
    """
    if not fused:
        return []

    now = datetime.now(tz=timezone.utc)
    max_rrf = fused[0][1] or 1.0   # normalisation denominator

    scored: list[tuple[EpisodeRow, float]] = []
    for row, rrf_score in fused:
        norm_rrf = rrf_score / max_rrf   # 0..1

        # Recency: exponential decay — 0 days → 1.0, 90 days → ~0.05
        age_days = (now - row.created_at.replace(tzinfo=timezone.utc)).days
        recency = max(0.0, 1.0 - age_days / 180.0)

        scope_match = 1.0 if (scope_filter is None or row.scope == scope_filter) else 0.5

        final = (
            0.45 * norm_rrf          # combined RRF signal
            + 0.15 * recency
            + 0.15 * row.importance
            + 0.15 * row.confidence
            + 0.10 * scope_match
        )

        # Bonus for explicit user corrections and preferences
        if row.was_user_correction:
            final = min(1.0, final + 0.10)
        if row.was_explicit_preference:
            final = min(1.0, final + 0.07)

        scored.append((row, final))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def filter_contradictions(
    scored: list[tuple[EpisodeRow, float]],
) -> list[tuple[EpisodeRow, float]]:
    """
    Suppress lower-scored items that share a contradiction_group with a
    higher-scored item.  The first (highest-scored) item in each group wins.
    """
    seen_groups: set[str] = set()
    result: list[tuple[EpisodeRow, float]] = []
    for row, score in scored:
        if row.contradiction_group:
            if row.contradiction_group in seen_groups:
                continue     # dominated by a better item in the same group
            seen_groups.add(row.contradiction_group)
        result.append((row, score))
    return result


def deduplicate(
    scored: list[tuple[EpisodeRow, float]],
    similarity_threshold: float = 0.92,
) -> list[tuple[EpisodeRow, float]]:
    """
    Simple text-based deduplication: suppress items whose summary_text is
    very similar to a higher-ranked item already in the result set.

    Uses character-level Jaccard similarity on word sets (no heavy deps).
    """
    result: list[tuple[EpisodeRow, float]] = []
    seen_tokens: list[set[str]] = []

    for row, score in scored:
        tokens = set(row.summary_text.lower().split())
        duplicate = False
        for existing in seen_tokens:
            if not existing:
                continue
            intersection = len(tokens & existing)
            union = len(tokens | existing)
            if union > 0 and intersection / union >= similarity_threshold:
                duplicate = True
                break
        if not duplicate:
            result.append((row, score))
            seen_tokens.append(tokens)

    return result
