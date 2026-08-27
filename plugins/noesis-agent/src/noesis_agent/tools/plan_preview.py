"""NOESIS plan preview tool for debugging and inspection."""
from __future__ import annotations

import json
from typing import Any


def create_plan_preview_tool(planner, classifier, tool_scoper, all_tool_defs=None):
    """Create the noesis_plan_preview tool handler.

    This tool lets users see how NOESIS would decompose and route
    a request without actually executing it.
    """

    def handler(args: dict[str, Any], **_kwargs) -> str:
        message = args.get("message", "")
        if not message:
            return json.dumps({"error": "message is required"}, ensure_ascii=False)

        # Generate plan using heuristic (no LLM call for preview)
        plan = planner.plan(message, available_tools=None, llm_call_fn=None)

        # Validate tiers
        for step in plan.steps:
            final_tier, reason = classifier.validate_step(
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
            "steps": [],
        }

        for step in plan.steps:
            step_info = step.to_dict()
            # Count scoped tools if we have tool definitions
            if all_tool_defs:
                scoped = tool_scoper.scope_tools(
                    all_tool_defs, step.required_tools, step.description,
                )
                step_info["scoped_tool_count"] = len(scoped)
                step_info["total_tool_count"] = len(all_tool_defs)
            result["steps"].append(step_info)

        return json.dumps(result, ensure_ascii=False, indent=2)

    return handler
