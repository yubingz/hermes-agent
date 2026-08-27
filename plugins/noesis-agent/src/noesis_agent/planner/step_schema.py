"""Step data structure for NOESIS task plans."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PlanStep:
    """A single step in a NOESIS task plan.

    Attributes:
        step_id: Sequential step identifier (1-indexed).
        description: What this step needs to accomplish.
        tier: Routing tier - local/cheap/standard/code/reasoning.
        required_tools: Tool names this step is allowed to use.
                        Empty list means no tools needed (pure LLM).
        input_hint: What input this step expects from previous step(s).
        output_format: Expected output format (text/json/code/markdown).
        estimated_tokens: Rough estimate of tokens needed.
        confidence: Planner's confidence in tier assignment (0.0-1.0).
        metadata: Additional metadata for this step.
    """
    step_id: int
    description: str
    tier: str = "standard"
    required_tools: list[str] = field(default_factory=list)
    input_hint: str = ""
    output_format: str = "text"
    estimated_tokens: int = 500
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        return cls(
            step_id=data["step_id"],
            description=data["description"],
            tier=data.get("tier", "standard"),
            required_tools=data.get("required_tools", []),
            input_hint=data.get("input_hint", ""),
            output_format=data.get("output_format", "text"),
            estimated_tokens=data.get("estimated_tokens", 500),
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskPlan:
    """Complete task plan produced by the NOESIS Planner.

    Attributes:
        user_request: Original user message.
        steps: Ordered list of plan steps.
        total_estimated_tokens: Sum of all step estimates.
        local_steps: Count of steps routed to local model.
        cloud_steps: Count of steps routed to cloud models.
        plan_version: Schema version for future compatibility.
    """
    user_request: str
    steps: list[PlanStep] = field(default_factory=list)
    total_estimated_tokens: int = 0
    local_steps: int = 0
    cloud_steps: int = 0
    plan_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "plan_version": self.plan_version,
            "total_estimated_tokens": self.total_estimated_tokens,
            "local_steps": self.local_steps,
            "cloud_steps": self.cloud_steps,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskPlan:
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            user_request=data["user_request"],
            steps=steps,
            total_estimated_tokens=data.get("total_estimated_tokens", 0),
            local_steps=data.get("local_steps", 0),
            cloud_steps=data.get("cloud_steps", 0),
            plan_version=data.get("plan_version", "1.0"),
        )

    @property
    def estimated_savings(self) -> float:
        """Estimated token savings vs sending everything to primary model."""
        if not self.steps:
            return 0.0
        # All 54 tools schema ≈ 27k tokens per request
        full_schema_cost = 27000
        # Per-step scoped schema ≈ 2-5k tokens
        scoped_schema_avg = 3000

        baseline = (full_schema_cost + self.total_estimated_tokens) * len(self.steps)
        optimized = sum(
            (scoped_schema_avg + s.estimated_tokens)
            * (0.0 if s.tier == "local" else 1.0)  # local = zero API cost
            for s in self.steps
        )
        return 1.0 - (optimized / baseline) if baseline > 0 else 0.0
