"""
rag_diagnostics.py -- one-shot RAG retrieval diagnostic.

Run from the project root:
    python rag_diagnostics.py

What it does:
  1. Embeds the test query using the live embed server (port 8020).
  2. Runs the exact pgvector query from retriever.py.
  3. Prints every returned row: distance, source, chunk_id, content preview.
  4. Checks if common stop-words from the query appear in each chunk.
"""
from __future__ import annotations

import io
import os
import sys
import textwrap

# Force UTF-8 output so Windows cp1252 terminal does not blow up on
# special chars that may appear in file paths or content previews.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_URL    = os.getenv("SAGE_EMBED_BASE_URL", "http://127.0.0.1:8020/v1")
EMBED_MODEL  = os.getenv("SAGE_EMBED_MODEL",    "bge-m3-embed")
MAX_DISTANCE = float(os.getenv("SAGE_RAG_MAX_DISTANCE", "0.5"))
TOP_K        = int(os.getenv("SAGE_RAG_TOP_K", "10"))

PG_USER = os.getenv("PG_USER", "sage")
PG_PASS = os.getenv("PG_PASSWORD", "")
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB", "sage_kaizen")
PG_DSN  = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# ── Test query ────────────────────────────────────────────────────────────────
QUERY = (
    "Write a short story titled Last Night in Atlanta.\n"
    "Requirements:\n"
    "- Blend jazz music and urban atmosphere\n"
    "- Include zombies\n"
    "- Use sensory imagery\n"
    "- Include one recurring symbolic motif\n"
    "- Tone: reflective, not sentimental\n"
    "- Length: 900-1100 words"
)

# Words we suspect might cause false-positive retrieval
SUSPECT_WORDS = [
    "write", "short", "story", "titled", "last", "night", "atlanta",
    "requirements", "blend", "jazz", "music", "urban", "atmosphere",
    "include", "zombies", "use", "sensory", "imagery", "recurring",
    "symbolic", "motif", "tone", "reflective", "sentimental", "length",
    "words", "in", "not", "one",
]

DOMAIN_WORDS = {"jazz", "zombie", "zombies", "atlanta", "sensory", "motif", "urban"}

SQL = """
SELECT source_id, chunk_id, content, metadata,
       (embedding <=> %s::vector) AS distance
FROM rag_chunks
WHERE embedding IS NOT NULL
  AND (embedding <=> %s::vector) < %s
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

SEP = "=" * 72
DIV = "-" * 72

# ── Helpers ───────────────────────────────────────────────────────────────────
def embed(text: str) -> list[float]:
    r = httpx.post(
        f"{EMBED_URL}/embeddings",
        json={"model": EMBED_MODEL, "input": [text]},
        timeout=60.0,
    )
    r.raise_for_status()
    return [float(x) for x in r.json()["data"][0]["embedding"]]


def relevance(distance: float) -> float:
    return round(1.0 / (1.0 + distance), 4)


def display_source(source_id: str) -> str:
    for prefix in ("localfile:", "web:", "rss_item:"):
        if source_id.startswith(prefix):
            tail = source_id[len(prefix):]
            if prefix == "localfile:":
                from pathlib import Path
                return Path(tail).name
            return tail[:80]
    return source_id[:80]


def keyword_hits(content: str) -> list[str]:
    lower = content.lower()
    return [w for w in SUSPECT_WORDS if w in lower]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(SEP)
    print("RAG RETRIEVAL DIAGNOSTIC")
    print(SEP)
    print()
    print("Query:")
    for line in QUERY.splitlines():
        print(f"  {line}")
    print()
    print(f"Embed URL    : {EMBED_URL}")
    print(f"Max distance : {MAX_DISTANCE}  (min score ~= {relevance(MAX_DISTANCE):.3f})")
    print(f"Top-K        : {TOP_K}")
    print(f"PG DSN       : {PG_DSN}")
    print()

    # Step 1 -- embed
    print("Step 1: Embedding query ... ", end="", flush=True)
    try:
        q_emb = embed(QUERY)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    print(f"OK  (dim={len(q_emb)})")

    # pgvector text format '[0.1, 0.2, ...]' — always works regardless of
    # pgvector cast availability; psycopg3 sends this as a plain text value
    # and PostgreSQL applies text::vector, which is defined in every pgvector version.
    q_vec = "[" + ",".join(repr(x) for x in q_emb) + "]"

    # Step 2 -- query
    print("Step 2: Running pgvector query ...")
    print()
    try:
        with psycopg.connect(PG_DSN, autocommit=True) as conn:
            conn.execute("SET hnsw.ef_search = 100")
            # Use an explicit cursor with row_factory=dict_row so Pylance
            # can infer Cursor[DictRow] via cursor()'s own CursorRow TypeVar,
            # which is free (not the class-level Row default TupleRow).
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(SQL, (q_vec, q_vec, MAX_DISTANCE, q_vec, TOP_K))
                rows = cur.fetchall()
    except Exception as e:
        print(f"DB ERROR: {e}")
        sys.exit(1)

    if not rows:
        print("  *** No rows returned -- nothing within max_distance. ***")
        print(f"\n  Try raising SAGE_RAG_MAX_DISTANCE (currently {MAX_DISTANCE}).")
        return

    print(f"  {len(rows)} row(s) returned.")
    print()
    print(DIV)

    for i, r in enumerate(rows, 1):
        dist    = float(r["distance"])
        s       = relevance(dist)
        src     = display_source(r["source_id"])
        content = (r["content"] or "").strip()
        hits    = keyword_hits(content)

        print(f"[{i}] Source  : {src}")
        print(f"     Chunk   : {r['chunk_id']}")
        print(f"     Distance: {dist:.6f}  ->  score: {s:.4f} ({int(s*100)}%)")
        print("     Content preview (first 300 chars):")
        wrapped = textwrap.fill(content[:300], width=66)
        for line in wrapped.splitlines():
            print(f"       {line}")
        print(f"     Keyword hits: {hits if hits else '(none)'}")
        print()

    # Step 3 -- summary
    print(SEP)
    print("KEYWORD BLEED SUMMARY")
    print(SEP)
    print()
    print("Interpretation guide:")
    print("  Domain hits  : jazz, zombie, atlanta, sensory, motif, urban")
    print("                 -> genuine semantic match")
    print("  Generic hits : in, write, include, story, requirements, etc.")
    print("                 -> possible false positive (semantic drift or")
    print("                    max_distance threshold too permissive)")
    print()

    print("Per-row verdict:")
    for i, r in enumerate(rows, 1):
        dist    = float(r["distance"])
        s       = relevance(dist)
        content = (r["content"] or "").strip()
        hits    = keyword_hits(content)
        src     = display_source(r["source_id"])

        domain_hits  = [h for h in hits if h in DOMAIN_WORDS]
        generic_hits = [h for h in hits if h not in DOMAIN_WORDS]

        if not hits:
            verdict = "NO keyword overlap -- pure vector similarity (may still be coincidental)"
        elif domain_hits:
            verdict = f"Domain match: {domain_hits}  <- likely a REAL hit"
        else:
            verdict = f"Generic/stop-word overlap only: {generic_hits}  <- LIKELY FALSE POSITIVE"

        print(f"  [{i}] score={s:.3f}  {src}")
        print(f"       {verdict}")
        print()

    print(SEP)
    print("NOISE-CLUSTER GATE SIMULATION")
    print(SEP)
    print()
    print("Active thresholds (from rag_settings.py / env vars):")
    print()

    # Mirror the defaults from rag_settings.py; override via env vars.
    CLUSTER_MIN_SIZE   = int(os.getenv("CLUSTER_MIN_SIZE",   "3"))
    CLUSTER_MAX_SPREAD = float(os.getenv("CLUSTER_MAX_SPREAD", "0.030"))
    CLUSTER_TOP1_FLOOR = float(os.getenv("CLUSTER_TOP1_FLOOR", "0.800"))

    print(f"  CLUSTER_MIN_SIZE   = {CLUSTER_MIN_SIZE}")
    print(f"  CLUSTER_MAX_SPREAD = {CLUSTER_MAX_SPREAD:.3f}")
    print(f"  CLUSTER_TOP1_FLOOR = {CLUSTER_TOP1_FLOOR:.3f}")
    print()

    if not rows:
        print("  No rows to evaluate.")
    else:
        kept_scores = [relevance(float(r["distance"])) for r in rows]
        n = len(kept_scores)
        spread = max(kept_scores) - min(kept_scores)
        top1 = max(kept_scores)

        print(f"  Results : {n}")
        print(f"  Spread  : {spread:.4f}  (max_score - min_score)")
        print(f"  Top-1   : {top1:.4f}")
        print()

        cond_size   = n >= CLUSTER_MIN_SIZE
        cond_spread = spread < CLUSTER_MAX_SPREAD
        cond_top1   = top1 < CLUSTER_TOP1_FLOOR
        gate_fires  = cond_size and cond_spread and cond_top1

        print("  Gate conditions:")
        print(f"    len >= {CLUSTER_MIN_SIZE}      : {str(cond_size):<5}  ({n} >= {CLUSTER_MIN_SIZE})")
        print(f"    spread < {CLUSTER_MAX_SPREAD:.3f} : {str(cond_spread):<5}  ({spread:.4f} < {CLUSTER_MAX_SPREAD})")
        print(f"    top1 < {CLUSTER_TOP1_FLOOR:.3f}   : {str(cond_top1):<5}  ({top1:.4f} < {CLUSTER_TOP1_FLOOR})")
        print()
        if gate_fires:
            print("  VERDICT: GATE TRIGGERED -- all chunks REJECTED.")
            print("  No RAG context will be injected for this query.")
        else:
            failed = []
            if not cond_size:
                failed.append(f"size ({n} < {CLUSTER_MIN_SIZE})")
            if not cond_spread:
                failed.append(f"spread ({spread:.4f} >= {CLUSTER_MAX_SPREAD})")
            if not cond_top1:
                failed.append(f"top1 ({top1:.4f} >= {CLUSTER_TOP1_FLOOR})")
            print(f"  VERDICT: gate NOT triggered (failed: {', '.join(failed)}).")
            print(f"  {n} chunk(s) would be injected as RAG context.")


if __name__ == "__main__":
    main()
