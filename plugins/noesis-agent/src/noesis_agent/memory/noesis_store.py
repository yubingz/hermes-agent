"""NOESIS memory interface: stub for NOESIS-II integration."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class NoesisMemoryStore:
    """Interface to NOESIS-II hybrid memory system.

    Phase 1 provides a stub that can be replaced with the full
    NOESIS-II implementation (hybrid retrieval + adaptive forgetting).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._store: dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize memory backend."""
        # Phase 2: connect to vector DB + graph store
        self._initialized = True
        logger.info("NOESIS memory store initialized (stub mode)")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant memories for a query."""
        # Phase 2: hybrid vector + keyword retrieval
        return []

    def store(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a new memory."""
        import uuid
        mem_id = str(uuid.uuid4())
        self._store[mem_id] = {
            "content": content,
            "metadata": metadata or {},
        }
        return mem_id

    def get_planner_context(self, user_message: str) -> str:
        """Get relevant memory context for the planner."""
        memories = self.retrieve(user_message)
        if not memories:
            return ""
        return "\n".join(f"- {m.get('content', '')}" for m in memories)

    def record_step_result(self, step_id: int, result: str) -> None:
        """Record a step result for future context."""
        self.store(
            content=f"Step {step_id} result: {result[:500]}",
            metadata={"type": "step_result", "step_id": step_id},
        )
