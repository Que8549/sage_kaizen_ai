"""
memory/
Sage Kaizen Memory Service — local, governed, auditable long-term memory.

Quick start:
    from memory.service import MemoryService
    svc = MemoryService()           # pools opened on first access
    bundle = svc.get_memory_bundle(request)

Phase 1 LangMem shortcut:
    from memory.langmem_bridge import LangMemBridge
    bridge = LangMemBridge()        # async; use asyncio or daemon-thread pattern
"""
