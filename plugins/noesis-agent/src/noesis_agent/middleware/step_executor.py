"""Step executor: middleware hook for per-step model switching.

This module integrates with Hermes' middleware architecture (#626)
to intercept model calls and route them to the appropriate model
based on the current execution step.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..planner.step_schema import TaskPlan, PlanStep
from ..router.model_resolver import ModelResolver
from ..router.tool_scoper import ToolScoper

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Tracks the current execution state across steps."""
    plan: TaskPlan | None = None
    current_step_index: int = 0
    step_results: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    total_local_calls: int = 0
    total_cloud_calls: int = 0
    total_tokens_used: int = 0

    @property
    def current_step(self) -> PlanStep | None:
        if self.plan and 0 <= self.current_step_index < len(self.plan.steps):
            return self.plan.steps[self.current_step_index]
        return None

    def advance(self) -> PlanStep | None:
        """Move to next step. Returns the new current step or None if done."""
        self.current_step_index += 1
        return self.current_step

    def record_result(self, step_id: int, result: str, tokens: int, was_local: bool):
        self.step_results.append({
            "step_id": step_id,
            "result": result,
            "tokens": tokens,
            "was_local": was_local,
            "duration": time.time() - self.start_time,
        })
        self.total_tokens_used += tokens
        if was_local:
            self.total_local_calls += 1
        else:
            self.total_cloud_calls += 1


class StepExecutor:
    """Middleware-based executor that switches models per step.

    Hooks into Hermes' wrap_model_call middleware to:
    1. Look up the current step's assigned tier
    2. Resolve the tier to a concrete model
    3. Switch the model before the LLM call
    4. Scope tool schemas to only what the step needs
    5. Record metrics after the call
    """

    def __init__(
        self,
        model_resolver: ModelResolver,
        tool_scoper: ToolScoper,
        config: dict[str, Any] | None = None,
    ):
        self.model_resolver = model_resolver
        self.tool_scoper = tool_scoper
        self.config = config or {}
        self.context = ExecutionContext()
        self._original_model_config: dict[str, Any] = {}
        self._on_model_switch: Callable | None = None

    def set_plan(self, plan: TaskPlan) -> None:
        """Set the task plan to execute."""
        self.context = ExecutionContext(plan=plan)
        logger.info(
            "NOESIS: plan set with %d steps (%d local, %d cloud), est. savings %.0f%%",
            len(plan.steps), plan.local_steps, plan.cloud_steps,
            plan.estimated_savings * 100,
        )

    def set_model_switch_callback(self, callback: Callable) -> None:
        """Set callback for actual model switching.

        The callback receives (model_config: dict) and should
        perform the actual model switch in Hermes.
        """
        self._on_model_switch = callback

    def wrap_model_call(self, next_handler, request: dict[str, Any]) -> dict[str, Any]:
        """Middleware hook: intercept and modify model calls.

        This is called before each LLM request. It:
        1. Determines current step's tier
        2. Resolves the model for that tier
        3. Swaps model/runtime in the request
        4. Scopes tool schemas
        5. Passes to next handler
        6. Records metrics from response
        """
        step = self.context.current_step

        if step is None:
            # No active plan, pass through unchanged
            return next_handler(request)

        # Resolve model for this step's tier
        model_config = self.model_resolver.resolve(step.tier)

        if not model_config.get("fallback", False):
            # Apply model switch
            self._apply_model_to_request(request, model_config)

            # Scope tools
            if "tools" in request and step.required_tools is not None:
                all_tools = request["tools"]
                scoped_tools = self.tool_scoper.scope_tools(
                    all_tools, step.required_tools, step.description,
                )
                request["tools"] = scoped_tools
                logger.debug(
                    "NOESIS step %d: %d tools scoped from %d (tier=%s)",
                    step.step_id, len(scoped_tools), len(all_tools), step.tier,
                )

        # Execute the call
        response = next_handler(request)

        # Record metrics
        usage = response.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        self.context.record_result(
            step_id=step.step_id,
            result=str(response.get("content", ""))[:500],
            tokens=tokens,
            was_local=model_config.get("is_local", False),
        )

        return response

    def _apply_model_to_request(self, request: dict[str, Any], config: dict[str, Any]) -> None:
        """Apply model configuration to a model request."""
        if config.get("model"):
            request["model"] = config["model"]
        if config.get("provider"):
            request["provider"] = config["provider"]
        if config.get("base_url"):
            request["base_url"] = config["base_url"]
        if config.get("api_key"):
            request["api_key"] = config["api_key"]
        if config.get("max_tokens"):
            request["max_tokens"] = config["max_tokens"]

        # Apply reasoning config
        reasoning = config.get("reasoning_config", {})
        if reasoning:
            request["reasoning"] = reasoning

        # Notify Hermes to switch model
        if self._on_model_switch:
            try:
                self._on_model_switch(config)
            except Exception as exc:
                logger.warning("NOESIS: model switch callback failed: %s", exc)

    def get_metrics(self) -> dict[str, Any]:
        """Return execution metrics."""
        ctx = self.context
        return {
            "total_steps": len(ctx.plan.steps) if ctx.plan else 0,
            "completed_steps": len(ctx.step_results),
            "local_calls": ctx.total_local_calls,
            "cloud_calls": ctx.total_cloud_calls,
            "total_tokens": ctx.total_tokens_used,
            "elapsed_seconds": round(time.time() - ctx.start_time, 2),
            "estimated_savings_pct": round(
                ctx.plan.estimated_savings * 100, 1
            ) if ctx.plan else 0,
        }

    def reset(self) -> None:
        """Reset execution context."""
        self.context = ExecutionContext()
