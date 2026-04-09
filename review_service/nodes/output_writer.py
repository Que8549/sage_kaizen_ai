"""
review_service/nodes/output_writer.py — Write review artifacts to disk.

Only reachable when state["approved"] == True (enforced by conditional
edge in graph.py). Writes:

  reviews/YYYY-MM-DD-HHMM-{mode}-review.md   — always
  docs/03-DECISIONS/ADR-YYYY-MM-DD-HHMM-architect-review.md
      — only when architect_findings contains CRITICAL or HIGH architectural risks
  reviews/patches/YYYY-MM-DD-HHMM-{slug}.patch
      — one per suggested patch extracted from architect_findings

No autonomous git commit. No branch creation. Files only.
"""
from __future__ import annotations

from sk_logging import get_logger
from ..state import ReviewState
from ..output.review_writer import write_review_file
from ..output.adr_writer import write_adr_if_needed
from ..output.patch_writer import write_patch_files

_LOG = get_logger("sage_kaizen.review_service.output_writer")


async def output_writer_node(state: ReviewState) -> dict:
    """
    Write all review artifacts. Returns updated output_paths list.
    """
    if not state.get("approved"):
        # Safety guard — graph conditional edge should prevent this path
        _LOG.warning("output_writer reached without approval — skipping all writes")
        return {"output_paths": []}

    paths: list[str] = []

    # 1. Main review markdown
    review_path = write_review_file(state)
    paths.append(review_path)
    _LOG.info("review.output | review_file=%s", review_path)

    # 2. ADR (only for CRITICAL/HIGH architectural findings)
    adr_path = write_adr_if_needed(state)
    if adr_path:
        paths.append(adr_path)
        _LOG.info("review.output | adr_file=%s", adr_path)

    # 3. Patch files
    patch_paths = write_patch_files(state)
    paths.extend(patch_paths)
    if patch_paths:
        _LOG.info("review.output | patch_files=%s", patch_paths)

    _LOG.info("review.approved | thread_id=%s | files_written=%d", state.get("mode"), len(paths))
    return {"output_paths": paths}
