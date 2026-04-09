"""
review_service/runner.py — ReviewRunner

Runs the LangGraph review graph in a background daemon thread with its own
isolated asyncio event loop, preventing conflict with Streamlit's main thread.

Lifecycle
---------
    IDLE
      → start() → RUNNING
          → scope/subprocess/web/architect/... nodes execute
          → AWAITING_APPROVAL  (interrupt() fired; synthesis ready)
              → resume(approved=True)  → RUNNING → DONE
              → resume(approved=False) → REJECTED
          → ERROR  (any unhandled exception)

Communication between the background thread and Streamlit is via plain
Python attributes (str/bool/list) protected by threading.Lock.
Streamlit polls status via st_autorefresh (2s interval in the UI widget).
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from typing import Optional

from langgraph.types import Command

from .checkpointer import make_checkpointer
from .graph import build_review_graph
from .state import ReviewState, default_state
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.review_service.runner")


class ReviewRunner:
    """
    Manages one review run at a time.

    Attributes (read by Streamlit UI, written only under _lock)
    -----------------------------------------------------------
    status          : str  — "idle" | "running" | "awaiting_approval" | "done" | "rejected" | "error"
    thread_id       : str  — LangGraph thread_id for this run (shown in UI)
    interrupt_payload: dict — {"synthesis": str, "prompt": str} when awaiting
    output_paths    : list[str] — written file paths when done
    error           : str  — error message when status == "error"
    """

    def __init__(self) -> None:
        self.status: str = "idle"
        self.thread_id: Optional[str] = None
        self.interrupt_payload: Optional[dict] = None
        self.output_paths: list[str] = []
        self.error: Optional[str] = None

        self._lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, mode: str, target: str = "") -> str:
        """
        Launch a new review run in a background thread.
        Returns the thread_id for display in the UI.
        """
        thread_id = f"review-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{mode}"

        with self._lock:
            self.thread_id         = thread_id
            self.status            = "running"
            self.interrupt_payload = None
            self.output_paths      = []
            self.error             = None

        _LOG.info("review.start | thread_id=%s mode=%s target=%r", thread_id, mode, target)

        self._bg_thread = threading.Thread(
            target=self._run_sync,
            args=(default_state(mode=mode, target=target), thread_id),
            daemon=True,
            name=f"review-{thread_id}",
        )
        self._bg_thread.start()
        return thread_id

    def resume(self, approved: bool) -> None:
        """
        Resume the graph after the human gate interrupt.
        approved=True  → output_writer writes files.
        approved=False → graph ends without writing anything.
        """
        _LOG.info("review.resume | thread_id=%s approved=%s", self.thread_id, approved)

        with self._lock:
            self.status = "running"

        self._bg_thread = threading.Thread(
            target=self._resume_sync,
            args=(approved,),
            daemon=True,
            name=f"review-resume-{self.thread_id}",
        )
        self._bg_thread.start()

    # ── Initial run (background thread) ──────────────────────────────────

    def _run_sync(self, initial_state: ReviewState, thread_id: str) -> None:
        """Entry point for the background thread — creates its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_async(initial_state, thread_id))
        except Exception as exc:
            _LOG.exception("review.error | thread_id=%s | %s", thread_id, exc)
            with self._lock:
                self.status = "error"
                self.error  = str(exc)
        finally:
            loop.close()

    async def _run_async(self, initial_state: ReviewState, thread_id: str) -> None:
        async with make_checkpointer() as checkpointer:
            graph  = build_review_graph(checkpointer)
            config = {"configurable": {"thread_id": thread_id}}
            result = await graph.ainvoke(dict(initial_state), config=config)

        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            _LOG.info("review.interrupt | thread_id=%s | synthesis ready", thread_id)
            with self._lock:
                self.status            = "awaiting_approval"
                self.interrupt_payload = payload if isinstance(payload, dict) else {"synthesis": str(payload)}
        else:
            # Graph completed without interruption (e.g. error path returned early)
            with self._lock:
                self.status       = "done"
                self.output_paths = result.get("output_paths", [])

    # ── Resume (background thread) ────────────────────────────────────────

    def _resume_sync(self, approved: bool) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._resume_async(approved))
        except Exception as exc:
            _LOG.exception("review.resume.error | thread_id=%s | %s", self.thread_id, exc)
            with self._lock:
                self.status = "error"
                self.error  = str(exc)
        finally:
            loop.close()

    async def _resume_async(self, approved: bool) -> None:
        async with make_checkpointer() as checkpointer:
            graph  = build_review_graph(checkpointer)
            config = {"configurable": {"thread_id": self.thread_id}}
            result = await graph.ainvoke(Command(resume=approved), config=config)

        if approved:
            paths = result.get("output_paths", [])
            _LOG.info("review.approved | thread_id=%s | files=%s", self.thread_id, paths)
            with self._lock:
                self.status       = "done"
                self.output_paths = paths
        else:
            _LOG.info("review.rejected | thread_id=%s", self.thread_id)
            with self._lock:
                self.status = "rejected"

    # ── Helpers ───────────────────────────────────────────────────────────

    def is_busy(self) -> bool:
        """Return True if a review is actively running or awaiting approval."""
        return self.status in ("running", "awaiting_approval")

    def reset(self) -> None:
        """Reset to idle state (call after done/rejected/error dismissal)."""
        with self._lock:
            self.status            = "idle"
            self.thread_id         = None
            self.interrupt_payload = None
            self.output_paths      = []
            self.error             = None
