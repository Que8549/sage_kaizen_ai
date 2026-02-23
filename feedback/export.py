from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from feedback.db import fetch_kto_rows, fetch_stats, get_conn
from feedback.settings import FeedbackSettings
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.feedback.export")


# ──────────────────────────────────────────────────────────────────────────── #
# TRL KTO format builder                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_kto_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one DB row to a TRL KTOTrainer record.

    TRL KTO conversational format:
        {
          "prompt":     [{"role": "system", ...}, ..., {"role": "user", ...}],
          "completion": [{"role": "assistant", "content": "<response>"}],
          "label":      true   # true = thumbs-up (+1), false = thumbs-down (-1)
        }

    The non-standard "_meta" block is ignored by TRL but useful for debugging.
    """
    prompt_messages: List[dict] = row["prompt_messages"]
    if isinstance(prompt_messages, str):
        prompt_messages = json.loads(prompt_messages)

    return {
        "prompt": prompt_messages,
        "completion": [{"role": "assistant", "content": row["assistant_text"]}],
        "label": bool(row["thumb"] == 1),
        "_meta": {
            "id": row["id"],
            "brain": row["brain"],
            "model_id": row.get("model_id") or "",
            "ts_utc": str(row["ts_utc"]),
        },
    }


# ──────────────────────────────────────────────────────────────────────────── #
# Export function                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def export_kto_jsonl(
    out_path: Path,
    *,
    dsn: str,
    brain: Optional[str] = None,
    thumb: Optional[int] = None,
    min_chars: int = 50,
    verbose: bool = True,
) -> Dict[str, int]:
    """Export rated responses to a JSONL file in TRL KTO conversational format.

    Returns stats dict: total_rows, exported.
    """
    conn = get_conn(dsn)
    rows = fetch_kto_rows(conn, brain=brain, thumb=thumb, min_chars=min_chars)
    conn.close()

    records = [_build_kto_record(r) for r in rows]

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {"total_rows": len(rows), "exported": len(records)}

    if verbose:
        pos = sum(1 for r in rows if r["thumb"] == 1)
        neg = len(rows) - pos
        print(f"Exported {len(records)} KTO records to {out_path}")
        print(f"  label=true  (thumbs-up):  {pos}")
        print(f"  label=false (thumbs-down): {neg}")

    _LOG.info(
        "kto_export | out=%s | exported=%d | brain=%s | thumb=%s",
        out_path, len(records), brain, thumb,
    )
    return stats


# ──────────────────────────────────────────────────────────────────────────── #
# Stats printer                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def print_stats(dsn: str) -> None:
    """Print dataset statistics to stdout."""
    conn = get_conn(dsn)
    stats = fetch_stats(conn)
    conn.close()

    total = stats.get("total", 0)
    up = stats.get("thumbs_up", 0)
    down = stats.get("thumbs_down", 0)
    balance = f"{up / total * 100:.0f}% pos" if total else "n/a"

    print("=== Feedback Dataset Stats ===")
    print(f"  Total rated:    {total}  ({balance})")
    print(f"  Thumbs up  👍:  {up}")
    print(f"  Thumbs down 👎: {down}")
    print(f"  FAST  up/down:  {stats.get('fast_up', 0)} / {stats.get('fast_down', 0)}")
    print(f"  ARCH  up/down:  {stats.get('arch_up', 0)} / {stats.get('arch_down', 0)}")
    if total and abs(up - down) / total > 0.7:
        print(
            "  WARNING: label imbalance > 70%. "
            "KTO training is sensitive to severe imbalance."
        )
