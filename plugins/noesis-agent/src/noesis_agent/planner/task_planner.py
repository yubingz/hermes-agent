"""NOESIS Task Planner: LLM-based task decomposition and tiering."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .step_schema import PlanStep, TaskPlan

logger = logging.getLogger(__name__)

# System prompt for the planner model
PLANNER_SYSTEM_PROMPT = """You are a NOESIS task planner. Your job is to break down user requests into an ordered chain of execution steps, and assign each step a routing tier based on its complexity.

## Routing Tiers
- **local**: Simple text operations, format conversion, keyword extraction, file read/write, basic math. No reasoning needed. Can be handled by a small local model (7B).
- **cheap**: Simple Q&A, basic search queries, short summaries, straightforward lookups.
- **standard**: Explanations, medium analysis, single-tool operations, moderate complexity.
- **code**: Code generation, debugging, refactoring, test writing, system administration commands.
- **reasoning**: Complex multi-step analysis, architecture decisions, research synthesis, creative problem solving, trade-off evaluation.

## Rules
1. Break the task into the MINIMUM number of steps needed. Simple requests = 1 step.
2. Each step should be independently executable.
3. Assign tiers conservatively - when unsure, use a higher tier.
4. List which tools each step needs (tool names only, not schemas).
5. Steps execute sequentially; output of step N feeds into step N+1.
6. Estimate tokens needed for each step's LLM call.

## Output Format
Respond ONLY with valid JSON, no markdown fences:
{
  "steps": [
    {
      "step_id": 1,
      "description": "What this step does",
      "tier": "local|cheap|standard|code|reasoning",
      "required_tools": ["tool_name"],
      "input_hint": "What this step expects as input",
      "output_format": "text|json|code|markdown",
      "estimated_tokens": 500,
      "confidence": 0.9
    }
  ]
}
"""

# Fallback heuristic planning when LLM planner is unavailable
SIMPLE_PATTERNS = {
    "code": [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bdebug\b",
        r"\berror\b", r"\btraceback\b", r"\brefactor\b", r"\btest\b",
        r"\bimplement\b", r"\bpython\b", r"\bjavascript\b", r"\bapi\b",
        r"代码", r"函数", r"调试", r"修复", r"重构", r"报错",
    ],
    "reasoning": [
        r"\banalyze\b", r"\bcompare\b", r"\bevaluate\b", r"\bstrategy\b",
        r"\barchitecture\b", r"\bdesign\b", r"\bresearch\b", r"\bexplain why\b",
        r"\btrade.?off\b", r"\bpros and cons\b", r"\bshould i\b",
        r"分析", r"比较", r"评估", r"策略", r"架构", r"研究", r"为什么",
    ],
    "local": [
        r"\bformat\b", r"\bconvert\b", r"\bcount\b", r"\bextract\b",
        r"\brename\b", r"\bsort\b", r"\buppercase\b", r"\blowercase\b",
        r"转换", r"格式化", r"统计", r"提取", r"排序",
    ],
}

LOCAL_TOOL_INDICATORS = {
    "read_file", "write_file", "edit_file", "bash",
    "file_to_url", "list_directory",
}

CODE_TOOL_INDICATORS = {
    "bash", "computer_use", "sessions_spawn",
}


class TaskPlanner:
    """Plans task decomposition and per-step routing tiers.

    The planner uses an LLM call (to the primary model) to decompose
    complex requests into steps with routing decisions. For simple
    requests, it falls back to heuristic classification to avoid
    an extra LLM round-trip.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.planner_cfg = self.config.get("noesis", {}).get("planner", {})
        self.enabled = self.planner_cfg.get("enabled", True)
        # Only use LLM planner for requests above this complexity threshold
        self.min_chars_for_planning = self.planner_cfg.get("min_chars", 50)
        self.max_steps = self.planner_cfg.get("max_steps", 8)

    def plan(
        self,
        user_message: str,
        available_tools: list[str] | None = None,
        llm_call_fn=None,
    ) -> TaskPlan:
        """Create a task plan from a user message.

        Args:
            user_message: The user's request text.
            available_tools: List of tool names available in this session.
            llm_call_fn: Async callable(prompt, system) -> str for LLM planning.
                         If None, uses heuristic fallback.

        Returns:
            TaskPlan with ordered steps and tier assignments.
        """
        if not self.enabled or not user_message.strip():
            return self._single_step_plan(user_message, "standard")

        text = user_message.strip()

        # For very short messages, skip LLM planning and use heuristic
        if len(text) < self.min_chars_for_planning or not llm_call_fn:
            return self._heuristic_plan(text, available_tools)

        # Try LLM-based planning
        try:
            plan = self._llm_plan(text, available_tools, llm_call_fn)
            if plan and plan.steps:
                return self._finalize_plan(plan, text)
        except Exception as exc:
            logger.warning("NOESIS planner LLM failed, falling back: %s", exc)

        return self._heuristic_plan(text, available_tools)

    def _llm_plan(
        self,
        text: str,
        available_tools: list[str] | None,
        llm_call_fn,
    ) -> TaskPlan | None:
        """Use LLM to create a task plan."""
        tools_hint = ""
        if available_tools:
            tools_hint = f"\n\nAvailable tools: {', '.join(available_tools)}"

        prompt = f"User request: {text}{tools_hint}"

        import asyncio
        if asyncio.iscoroutinefunction(llm_call_fn):
            response = asyncio.get_event_loop().run_until_complete(
                llm_call_fn(prompt, PLANNER_SYSTEM_PROMPT)
            )
        else:
            response = llm_call_fn(prompt, PLANNER_SYSTEM_PROMPT)

        # Parse JSON response (strip any markdown fences)
        cleaned = self._extract_json(response)
        data = json.loads(cleaned)

        steps = []
        for i, step_data in enumerate(data.get("steps", [])[:self.max_steps], 1):
            steps.append(PlanStep(
                step_id=i,
                description=step_data["description"],
                tier=step_data.get("tier", "standard"),
                required_tools=step_data.get("required_tools", []),
                input_hint=step_data.get("input_hint", ""),
                output_format=step_data.get("output_format", "text"),
                estimated_tokens=step_data.get("estimated_tokens", 500),
                confidence=step_data.get("confidence", 0.7),
            ))

        return TaskPlan(user_request=text, steps=steps)

    def _heuristic_plan(
        self,
        text: str,
        available_tools: list[str] | None,
    ) -> TaskPlan:
        """Fallback heuristic planning without LLM call."""
        tier = self._classify_tier(text)
        tools = self._infer_tools(text, available_tools)
        est_tokens = min(len(text) * 2, 2000)

        step = PlanStep(
            step_id=1,
            description=text[:200],
            tier=tier,
            required_tools=tools,
            estimated_tokens=est_tokens,
            confidence=0.6,
        )
        plan = TaskPlan(user_request=text, steps=[step])
        return self._finalize_plan(plan, text)

    def _single_step_plan(self, text: str, tier: str) -> TaskPlan:
        step = PlanStep(step_id=1, description=text[:200], tier=tier)
        return TaskPlan(user_request=text, steps=[step])

    def _classify_tier(self, text: str) -> str:
        """Classify text into a routing tier using regex patterns."""
        lowered = text.lower()

        for tier, patterns in SIMPLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, lowered, re.IGNORECASE):
                    return tier

        # Length-based fallback
        word_count = len(text.split())
        if word_count <= 15:
            return "cheap"
        elif word_count <= 80:
            return "standard"
        return "reasoning"

    def _infer_tools(
        self,
        text: str,
        available_tools: list[str] | None,
    ) -> list[str]:
        """Infer which tools might be needed based on message content."""
        if not available_tools:
            return []

        lowered = text.lower()
        inferred = []

        # File-related
        if any(w in lowered for w in ["file", "read", "write", "save", "open", "文件", "读取", "保存"]):
            for t in ["read_file", "write_file", "edit_file"]:
                if t in available_tools and t not in inferred:
                    inferred.append(t)

        # Code/execution
        if any(w in lowered for w in ["run", "execute", "command", "script", "运行", "执行"]):
            if "bash" in available_tools:
                inferred.append("bash")

        # Search
        if any(w in lowered for w in ["search", "find", "look up", "搜索", "查找"]):
            for t in ["search_web", "fetch_web"]:
                if t in available_tools and t not in inferred:
                    inferred.append(t)

        return inferred[:5]  # Cap at 5 tools per step

    def _finalize_plan(self, plan: TaskPlan, text: str) -> TaskPlan:
        """Compute aggregate stats and validate the plan."""
        local_tiers = {"local"}
        plan.local_steps = sum(1 for s in plan.steps if s.tier in local_tiers)
        plan.cloud_steps = len(plan.steps) - plan.local_steps
        plan.total_estimated_tokens = sum(s.estimated_tokens for s in plan.steps)

        # Safety: ensure at least one step
        if not plan.steps:
            return self._single_step_plan(text, "standard")

        return plan

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try direct parse first
        text = text.strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end + 1]

        return text
