from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Callable


def load_list_from_env_and_file(
    *,
    env_csv_var: str,
    env_file_var: str,
    normalize: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """
    Generic helper to load a list from:
      - ENV CSV:  VAR="a,b,c"
      - ENV FILE: VAR_FILE="C:\\path\\file.txt" (one entry per line)

    Lines beginning with '#' are ignored.
    Duplicates are removed, preserving order.
    """
    csv_val = (os.environ.get(env_csv_var) or "").strip()
    file_val = (os.environ.get(env_file_var) or "").strip()

    items: List[str] = []

    if file_val:
        path = os.path.abspath(file_val)
        for line in open(path, "r", encoding="utf-8", errors="ignore").read().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)

    if csv_val:
        for part in csv_val.split(","):
            s = part.strip()
            if s:
                items.append(s)

    if normalize is not None:
        items = [normalize(x) for x in items]

    # de-dupe preserving order
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class CommitBatcher:
    """
    Tracks pending "units" (chunks/rows) and commits when threshold is reached.

    Usage:
        batcher = CommitBatcher(commit_every=200)
        ...
        batcher.add(n_chunks)
        if batcher.should_commit():
            conn.commit()
            batcher.reset()

        # at end:
        batcher.commit_if_needed(conn)
    """
    commit_every: int = 200
    pending: int = 0

    def add(self, units: int) -> None:
        self.pending += int(units)

    def should_commit(self) -> bool:
        return self.pending >= self.commit_every

    def reset(self) -> None:
        self.pending = 0

    def commit_if_needed(self, conn) -> None:
        if self.pending > 0:
            conn.commit()
            self.pending = 0
