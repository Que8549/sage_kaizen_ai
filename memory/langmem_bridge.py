"""
memory/langmem_bridge.py
Phase 1 shortcut: LangMem + LangGraph AsyncPostgresStore memory bridge.

This module provides a production-ready alternative to the full custom memory
service using LangMem SDK + LangGraph's AsyncPostgresStore.

Architecture:
  - AsyncPostgresStore (langgraph-checkpoint-postgres) → PostgreSQL persistence
  - BGE-M3 (port 8020) → custom aembed_texts function for local embeddings
  - ARCHITECT brain (port 8012) → create_memory_store_manager LLM
  - ReflectionExecutor → debounced background memory extraction

Usage (in an async context, e.g. review_service daemon thread pattern):

    from memory.langmem_bridge import LangMemBridge
    bridge = await LangMemBridge.create()

    # After each turn (fire-and-forget background extraction):
    bridge.submit_turn(messages, user_id="alquin")

    # Retrieve memories for prompt injection:
    memories = await bridge.search(query="preferred code style", user_id="alquin")

    # Graceful shutdown:
    await bridge.close()

Namespace convention:
    ("sage_kaizen", "memories", "<user_id>")

LangMem stores items as JSON documents; each document has a "content" field
(the extracted memory text) and optional metadata.

Installation note:
    langmem is not installed by default.  Install with:
        pip install "langmem>=0.0.30"
    langgraph-checkpoint-postgres is already installed (used by review_service).

Sync wrapper for Streamlit integration:
    Use LangMemBridgeSync which runs the async bridge in a daemon-thread event
    loop following the same pattern as ReviewRunner.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from memory.embedder import aembed_texts as _bge_aembed  # shared async embed — no duplicate HTTP
from pg_settings import PgSettings
from sk_logging import get_logger

_LOG = get_logger("sage_kaizen.memory.langmem_bridge")


# ---------------------------------------------------------------------------
# LangMemBridge — async interface
# ---------------------------------------------------------------------------

class LangMemBridge:
    """
    Async LangMem + LangGraph memory bridge.

    Creates and manages:
    - AsyncPostgresStore: persists memories as JSON docs in PostgreSQL
    - create_memory_store_manager: ARCHITECT-brain-powered memory extractor
    - ReflectionExecutor: debounced background extraction (non-blocking)

    All methods are async.  For Streamlit integration use LangMemBridgeSync.
    """

    def __init__(self) -> None:
        self._store: Any = None
        self._executor: Any = None
        self._manager: Any = None
        self._user_id: str = "alquin"   # default; overridden per call

    @classmethod
    async def create(cls, user_id: str = "alquin") -> "LangMemBridge":
        """
        Async factory.  Call once at startup and reuse the instance.

            bridge = await LangMemBridge.create()
        """
        # Guard against missing langmem at import time — give a clear error.
        try:
            from langmem import ReflectionExecutor, create_memory_store_manager
            from langgraph.store.postgres import AsyncPostgresStore
        except ImportError as exc:
            raise ImportError(
                "LangMem bridge requires 'langmem>=0.0.30' and "
                "'langgraph-checkpoint-postgres>=3.0.5'. "
                "Install with: pip install 'langmem>=0.0.30'"
            ) from exc

        bridge = cls()
        bridge._user_id = user_id

        dsn = PgSettings().pg_dsn

        # Build AsyncPostgresStore with BGE-M3 embed function.
        # index.fields=["$"] embeds the entire document JSON.
        # dims=1024 matches BGE-M3 FP16 output.
        store = await AsyncPostgresStore.from_conn_string(
            dsn,
            index={
                "dims": 1024,
                "embed": _bge_aembed,
                "fields": ["$"],
            },
        )
        await store.setup()   # idempotent: creates langmem tables if missing
        bridge._store = store
        _LOG.info("langmem_bridge | AsyncPostgresStore ready dsn=%s...%s", dsn[:20], dsn[-10:])

        # Build memory extraction LLM — ARCHITECT brain via OpenAI-compatible API.
        # temperature=0.3 for deterministic extraction; no thinking tokens needed here.
        llm = ChatOpenAI(
            base_url="http://127.0.0.1:8012/v1",
            api_key="not-required",
            model="qwen3.5-27b",    # alias must match brains.yaml; llama-server ignores it
            temperature=0.3,
            max_tokens=512,
        )

        # create_memory_store_manager extracts long-term memories from conversations.
        # namespace uses {user_id} template — dynamically resolved per-call via configurable.
        manager = create_memory_store_manager(
            llm,
            namespace=("sage_kaizen", "memories", "{user_id}"),
            instructions=(
                "Extract only durable, reusable memories from this conversation. "
                "Focus on: user corrections, stated preferences, architecture decisions, "
                "model selections, approved code patterns, and explicit prohibitions. "
                "Do NOT extract: ephemeral requests, one-time questions, or trivial acks. "
                "Keep each memory concise (one sentence). "
                "Scope: Sage Kaizen project on Windows 11 Pro with RTX 5090/5080 stack."
            ),
        )
        bridge._manager = manager

        # ReflectionExecutor handles debounced background memory writes.
        # Default debounce: 2 seconds.  Adjust with submit(debounce_seconds=N).
        bridge._executor = ReflectionExecutor(manager, store=store)
        _LOG.info("langmem_bridge | ReflectionExecutor ready")

        return bridge

    def submit_turn(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
        debounce_seconds: float = 2.0,
    ) -> None:
        """
        Submit a completed conversation turn for background memory extraction.

        Non-blocking: the ReflectionExecutor queues the task and debounces
        rapid messages before processing.  Safe to call from the Streamlit
        thread without awaiting.

        messages: list of {"role": "...", "content": "..."} dicts
                  (OpenAI-compatible format, same as what llama-server receives)
        """
        uid = user_id or self._user_id
        config = {"configurable": {"user_id": uid}}
        try:
            self._executor.submit(
                {"messages": messages},
                config=config,
                # ReflectionExecutor.submit accepts debounce_seconds in newer versions
            )
            _LOG.debug("langmem_bridge | turn submitted user=%s msgs=%d", uid, len(messages))
        except Exception as exc:
            _LOG.warning("langmem_bridge | submit_turn failed (non-fatal): %s", exc)

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over stored memories for a user.

        Returns a list of memory dicts with 'value' (the extracted memory doc)
        and 'score' (cosine similarity to query).
        """
        uid = user_id or self._user_id
        namespace = ("sage_kaizen", "memories", uid)
        try:
            items = await self._store.asearch(
                namespace,
                query=query,
                limit=limit,
            )
            results = [
                {
                    "key":   item.key,
                    "value": item.value,
                    "score": getattr(item, "score", 0.0),
                }
                for item in items
            ]
            _LOG.debug(
                "langmem_bridge | search user=%s query=%r → %d results",
                uid, query[:40], len(results),
            )
            return results
        except Exception as exc:
            _LOG.warning("langmem_bridge | search failed: %s", exc)
            return []

    async def format_memories_for_prompt(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 6,
        max_chars: int = 1500,
    ) -> str:
        """
        Retrieve memories and format them as a compact prompt segment.
        Returns an empty string if no memories are found.
        """
        results = await self.search(query=query, user_id=user_id, limit=limit)
        if not results:
            return ""

        lines = ["[SAGE KAIZEN MEMORIES]"]
        total_chars = 0
        for item in results:
            # Memory docs may have {"content": "..."} or be plain strings
            value = item.get("value", {})
            if isinstance(value, dict):
                text = value.get("content", str(value))
            else:
                text = str(value)
            line = f"- {text}"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    async def forget(self, key: str, user_id: Optional[str] = None) -> None:
        """Delete a specific memory item by key."""
        uid = user_id or self._user_id
        namespace = ("sage_kaizen", "memories", uid)
        try:
            await self._store.adelete(namespace, key)
            _LOG.info("langmem_bridge | deleted key=%s user=%s", key, uid)
        except Exception as exc:
            _LOG.warning("langmem_bridge | forget failed: %s", exc)

    async def close(self) -> None:
        """Graceful shutdown — close store connections."""
        try:
            if self._store is not None:
                await self._store.aclose()
                _LOG.info("langmem_bridge | store closed")
        except Exception as exc:
            _LOG.warning("langmem_bridge | close error: %s", exc)


# ---------------------------------------------------------------------------
# LangMemBridgeSync — sync wrapper for Streamlit / main thread
#
# Runs the async bridge in a dedicated daemon-thread event loop following the
# same pattern used by ReviewRunner in review_service/runner.py.
# ---------------------------------------------------------------------------

class LangMemBridgeSync:
    """
    Sync wrapper around LangMemBridge for use from Streamlit (sync) context.

    Usage:
        bridge = LangMemBridgeSync()
        bridge.start()   # call once at app startup

        # Per-turn (non-blocking):
        bridge.submit_turn(messages, user_id="alquin")

        # For retrieval before prompt assembly:
        segment = bridge.get_memory_prompt(query, user_id="alquin")
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._bridge: Optional[LangMemBridge] = None
        self._ready = threading.Event()
        self._error: Optional[Exception] = None

    def start(self) -> None:
        """Spin up the daemon thread and initialise the async bridge."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="langmem-bridge",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=30.0):
            raise TimeoutError("LangMemBridgeSync: bridge did not start within 30s")
        if self._error:
            raise self._error

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._bridge = self._loop.run_until_complete(LangMemBridge.create())
            self._ready.set()
            self._loop.run_forever()
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def _run_async(self, coro: Any) -> Any:
        if self._loop is None or self._bridge is None:
            raise RuntimeError("LangMemBridgeSync not started; call start() first")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30.0)

    def submit_turn(
        self,
        messages: List[Dict[str, str]],
        user_id: Optional[str] = None,
    ) -> None:
        """Non-blocking: schedule background memory extraction for a completed turn."""
        if self._bridge is None:
            return
        self._bridge.submit_turn(messages, user_id=user_id)

    def get_memory_prompt(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 6,
        max_chars: int = 1500,
    ) -> str:
        """
        Synchronously retrieve and format memories for prompt injection.
        Returns '' if bridge is not started or no memories found.
        """
        if self._bridge is None:
            return ""
        try:
            return self._run_async(
                self._bridge.format_memories_for_prompt(
                    query=query, user_id=user_id, limit=limit, max_chars=max_chars
                )
            )
        except Exception as exc:
            _LOG.warning("langmem_bridge_sync | get_memory_prompt failed: %s", exc)
            return ""

    def stop(self) -> None:
        """Stop the daemon event loop."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_bridge_singleton: Optional[LangMemBridgeSync] = None


def get_langmem_bridge() -> LangMemBridgeSync:
    """
    Get or create the module-level LangMemBridgeSync singleton.

    Call start() before first use.  Safe to call multiple times.
    """
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = LangMemBridgeSync()
    return _bridge_singleton
