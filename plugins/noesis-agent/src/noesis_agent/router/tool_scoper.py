"""Tool scoper: filters tool schemas to only what a step needs."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolScoper:
    """Scopes the tool set for each execution step.

    The biggest token waste in Hermes is sending all 54+ tool JSON schemas
    (~27k tokens) with every single LLM request. For per-step execution,
    most steps only need 0-3 tools. This class filters schemas accordingly.
    """

    # Tools that are safe to include for most steps
    SAFE_DEFAULTS = {
        "read_file",
        "list_directory",
    }

    # Tools that should never be auto-included (require explicit request)
    RESTRICTED_TOOLS = {
        "sessions_spawn",
        "calendar_delete",
        "email_request",
        "computer_use",
        "create_project",
    }

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        noesis_cfg = self.config.get("noesis", {})
        self.include_safe_defaults = noesis_cfg.get("include_safe_defaults", True)
        self.max_tools_per_step = noesis_cfg.get("max_tools_per_step", 6)

    def scope_tools(
        self,
        all_tool_defs: list[dict[str, Any]],
        required_tools: list[str],
        step_description: str = "",
    ) -> list[dict[str, Any]]:
        """Filter tool definitions to only those needed for a step.

        Args:
            all_tool_defs: Full list of tool JSON schema definitions.
            required_tools: Tool names explicitly requested by the planner.
            step_description: Step description for heuristic tool inference.

        Returns:
            Filtered list of tool definitions (subset of all_tool_defs).
        """
        if not all_tool_defs:
            return []

        # Build name → definition index
        tool_map = {}
        for tool_def in all_tool_defs:
            name = self._extract_name(tool_def)
            if name:
                tool_map[name] = tool_def

        # Determine which tools to include
        selected_names = set()

        # 1. Planner-requested tools (filtered for restricted tools)
        for name in required_tools:
            if name in tool_map and name not in self.RESTRICTED_TOOLS:
                selected_names.add(name)

        # 2. Heuristic inference from step description
        if step_description:
            inferred = self._infer_tools(step_description, set(tool_map.keys()))
            selected_names.update(inferred)

        # 3. Safe defaults (low-cost tools useful for most steps)
        if self.include_safe_defaults:
            for name in self.SAFE_DEFAULTS:
                if name in tool_map:
                    selected_names.add(name)

        # 4. Cap at max tools per step
        if len(selected_names) > self.max_tools_per_step:
            # Prioritize planner-required tools (filtered for restricted)
            required_set = set(required_tools) & selected_names
            # If even required tools exceed cap, take the first N required
            if len(required_set) > self.max_tools_per_step:
                required_list = [n for n in required_tools if n in selected_names]
                selected_names = set(required_list[:self.max_tools_per_step])
            else:
                remaining = selected_names - required_set
                selected_names = set(required_set)
                for name in sorted(remaining):
                    if len(selected_names) >= self.max_tools_per_step:
                        break
                    selected_names.add(name)

        # Build result, preserving original order
        result = [
            tool_def for tool_def in all_tool_defs
            if self._extract_name(tool_def) in selected_names
        ]

        logger.debug(
            "NOESIS tool scope: %d tools → %d tools (selected: %s)",
            len(all_tool_defs), len(result), sorted(selected_names),
        )

        return result

    @staticmethod
    def _extract_name(tool_def: dict[str, Any]) -> str:
        """Extract tool name from various schema formats."""
        if "function" in tool_def:
            return tool_def["function"].get("name", "")
        return tool_def.get("name", "")

    @staticmethod
    def _infer_tools(description: str, available: set[str]) -> set[str]:
        """Infer needed tools from step description text."""
        inferred = set()
        desc_lower = description.lower()

        tool_keywords = {
            "search_web": ["search", "find online", "look up", "web", "internet", "搜索", "查找"],
            "fetch_web": ["url", "http", "webpage", "read page", "链接", "网页"],
            "read_file": ["read file", "open file", "load file", "读取", "打开文件"],
            "write_file": ["write", "save", "create file", "写入", "保存"],
            "edit_file": ["edit", "modify", "update file", "编辑", "修改"],
            "bash": ["run command", "execute", "shell", "terminal", "运行", "执行命令"],
        }

        for tool_name, keywords in tool_keywords.items():
            if tool_name in available:
                if any(kw in desc_lower for kw in keywords):
                    inferred.add(tool_name)

        return inferred

    def estimate_schema_savings(
        self,
        all_tool_defs: list[dict[str, Any]],
        plan_steps: Any,
    ) -> dict[str, Any]:
        """Estimate token savings from tool scoping across all plan steps.

        Returns:
            Dict with baseline_tokens, optimized_tokens, savings_pct.
        """
        import json

        full_schema = json.dumps(all_tool_defs, ensure_ascii=False)
        baseline_per_call = len(full_schema) // 3  # Rough token estimate

        total_baseline = baseline_per_call * len(plan_steps.steps if hasattr(plan_steps, 'steps') else plan_steps)

        total_optimized = 0
        steps = plan_steps.steps if hasattr(plan_steps, 'steps') else plan_steps
        for step in steps:
            tools = step.required_tools if hasattr(step, 'required_tools') else step.get('required_tools', [])
            desc = step.description if hasattr(step, 'description') else step.get('description', '')
            scoped = self.scope_tools(all_tool_defs, tools, desc)
            scoped_schema = json.dumps(scoped, ensure_ascii=False)
            total_optimized += len(scoped_schema) // 3

        savings_pct = (
            (1 - total_optimized / total_baseline) * 100
            if total_baseline > 0 else 0
        )

        return {
            "baseline_per_call": baseline_per_call,
            "total_baseline": total_baseline,
            "total_optimized": total_optimized,
            "savings_pct": round(savings_pct, 1),
            "full_tool_count": len(all_tool_defs),
        }
