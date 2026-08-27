"""NOESIS-Agent Hermes plugin entry point."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .planner.task_planner import TaskPlanner
from .planner.step_schema import TaskPlan
from .router.tier_classifier import TierClassifier
from .router.model_resolver import ModelResolver
from .router.tool_scoper import ToolScoper
from .middleware.step_executor import StepExecutor
from .memory.noesis_store import NoesisMemoryStore
from .tools.plan_preview import create_plan_preview_tool

logger = logging.getLogger(__name__)

# Global singletons (initialized on register)
_planner: TaskPlanner | None = None
_classifier: TierClassifier | None = None
_resolver: ModelResolver | None = None
_tool_scoper: ToolScoper | None = None
_executor: StepExecutor | None = None
_memory: NoesisMemoryStore | None = None


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return load_config()
    except Exception:
        pass
    path = _hermes_home() / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _preview_handler(args: dict[str, Any], **_kwargs: Any) -> str:
    """Handle noesis_plan_preview tool calls."""
    message = args.get("message", "")
    if not message:
        return json.dumps({"error": "message is required"}, ensure_ascii=False)

    plan = _planner.plan(message, available_tools=None, llm_call_fn=None)

    for step in plan.steps:
        final_tier, reason = _classifier.validate_step(
            step.description, step.tier, step.confidence
        )
        step.tier = final_tier
        step.metadata["validation"] = reason

    result = {
        "request": message[:200],
        "total_steps": len(plan.steps),
        "local_steps": plan.local_steps,
        "cloud_steps": plan.cloud_steps,
        "estimated_savings_pct": round(plan.estimated_savings * 100, 1),
        "steps": [s.to_dict() for s in plan.steps],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def register(ctx) -> None:
    """Register NOESIS-Agent plugin with Hermes.

    Initializes all components and registers:
    1. The plan preview tool
    2. The middleware hook for per-step model switching
    3. Memory store (if enabled)
    """
    global _planner, _classifier, _resolver, _tool_scoper, _executor, _memory

    config = _load_config()
    noesis_cfg = config.get("noesis", {})

    # Initialize components
    _planner = TaskPlanner(config)
    _classifier = TierClassifier(config)
    _resolver = ModelResolver(config)
    _tool_scoper = ToolScoper(config)
    _executor = StepExecutor(_resolver, _tool_scoper, config)
    _memory = NoesisMemoryStore(config)

    if noesis_cfg.get("memory", {}).get("enabled", False):
        _memory.initialize()

    # Register preview tool
    ctx.register_tool(
        name="noesis_plan_preview",
        toolset="noesis_agent",
        schema={
            "name": "noesis_plan_preview",
            "description": (
                "Preview how NOESIS would decompose a request into steps, "
                "assign routing tiers, and estimate token savings. "
                "Does not execute anything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The user request to plan.",
                    },
                },
                "required": ["message"],
            },
        },
        handler=_preview_handler,
        description="Preview NOESIS task plan and routing decisions.",
        emoji="🧠",
    )

    # Register middleware hook if supported by Hermes version
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware(
            "wrap_model_call",
            _executor.wrap_model_call,
            priority=100,  # High priority to intercept before other middleware
        )
        logger.info("NOESIS-Agent: middleware hook registered")
    else:
        logger.info(
            "NOESIS-Agent: middleware not available in this Hermes version. "
            "Core integration patch required for automatic routing. "
            "Use noesis_plan_preview for manual inspection."
        )

    logger.info(
        "NOESIS-Agent v0.1.0 registered (planner=%s, middleware=%s)",
        noesis_cfg.get("planner", {}).get("enabled", True),
        hasattr(ctx, "register_middleware"),
    )


def get_executor() -> StepExecutor | None:
    """Get the global step executor (for core integration)."""
    return _executor


def get_planner() -> TaskPlanner | None:
    """Get the global task planner."""
    return _planner


def get_resolver() -> ModelResolver | None:
    """Get the global model resolver."""
    return _resolver
