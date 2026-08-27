"""Tier classifier: estimates step difficulty and validates tier assignments."""
from __future__ import annotations

import re
from typing import Any

# Signals that suggest a step needs stronger reasoning
REASONING_SIGNALS = [
    r"\bbecause\b", r"\btherefore\b", r"\bhowever\b", r"\balthough\b",
    r"\btrade.?off\b", r"\bimplications?\b", r"\brisk\b", r"\bimpact\b",
    r"\bcompare\b", r"\bcontrast\b", r"\bevaluate\b", r"\bjustify\b",
    r"\bwhy\b", r"\bhow does\b", r"\bwhat if\b",
    r"为什么", r"如何", r"比较", r"评估", r"影响", r"风险",
]

CODE_SIGNALS = [
    r"```", r"\bdef\b", r"\bclass\b", r"\bimport\b", r"\bfunction\b",
    r"\breturn\b", r"\bawait\b", r"\basync\b", r"\btraceback\b",
    r"\berror\b", r"\bexception\b", r"\bdebug\b", r"\brefactor\b",
]

LOCAL_SIGNALS = [
    r"^\s*(convert|format|count|sort|rename|extract|list|print|echo)\b",
    r"^\s*(转换|格式化|统计|排序|重命名|提取|列出|打印)\b",
]

# Maximum tokens per tier (rough guide for planner estimates)
TIER_TOKEN_LIMITS = {
    "local": 2048,
    "cheap": 4096,
    "standard": 8192,
    "code": 16384,
    "reasoning": 32768,
}

TIER_ORDER = ["local", "cheap", "standard", "code", "reasoning"]


class TierClassifier:
    """Classifies and validates routing tiers for plan steps.

    Can be used standalone for heuristic classification or to
    validate/upgrade LLM-generated tier assignments.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        noesis_cfg = self.config.get("noesis", {})
        tier_cfg = noesis_cfg.get("tiers", {})

        # Allow config overrides for tier thresholds
        self.local_max_words = tier_cfg.get("local", {}).get("max_words", 30)
        self.cheap_max_words = tier_cfg.get("cheap", {}).get("max_words", 100)

        # Confidence thresholds for LLM planner
        self.upgrade_threshold = noesis_cfg.get("upgrade_threshold", 0.5)
        self.downgrade_threshold = noesis_cfg.get("downgrade_threshold", 0.95)

    def classify(self, text: str) -> tuple[str, float]:
        """Classify text into a tier with confidence.

        Returns:
            Tuple of (tier_name, confidence 0.0-1.0).
        """
        if not text or not text.strip():
            return "standard", 0.3

        lowered = text.lower()
        word_count = len(text.split())

        # Check for code markers (highest priority)
        code_hits = sum(1 for p in CODE_SIGNALS if re.search(p, lowered, re.MULTILINE))
        if code_hits >= 2:
            return "code", min(0.9, 0.6 + code_hits * 0.1)

        # Check for reasoning signals
        reasoning_hits = sum(1 for p in REASONING_SIGNALS if re.search(p, lowered))
        if reasoning_hits >= 2:
            return "reasoning", min(0.9, 0.6 + reasoning_hits * 0.08)

        # Check for local/simple patterns
        if word_count <= self.local_max_words:
            local_hits = sum(1 for p in LOCAL_SIGNALS if re.search(p, lowered, re.MULTILINE))
            if local_hits > 0 or word_count <= 10:
                return "local", 0.8

        # Length-based classification
        if word_count <= self.local_max_words:
            return "cheap", 0.7
        elif word_count <= self.cheap_max_words:
            return "standard", 0.65
        else:
            return "reasoning", 0.5

    def validate_step(self, description: str, assigned_tier: str, confidence: float) -> tuple[str, str]:
        """Validate a tier assignment and upgrade if necessary.

        The planner may underestimate difficulty. This acts as a safety net.

        Returns:
            Tuple of (final_tier, reason).
        """
        heuristic_tier, heuristic_conf = self.classify(description)
        assigned_idx = TIER_ORDER.index(assigned_tier) if assigned_tier in TIER_ORDER else 2
        heuristic_idx = TIER_ORDER.index(heuristic_tier)

        # If heuristic suggests a higher tier with good confidence, upgrade
        if heuristic_idx > assigned_idx and heuristic_conf >= self.upgrade_threshold:
            return heuristic_tier, f"upgraded-from-{assigned_tier}-heuristic"

        # If planner has very high confidence, trust it even if heuristic disagrees
        if confidence >= self.downgrade_threshold:
            return assigned_tier, "planner-high-confidence"

        # If they agree, confirm
        if assigned_tier == heuristic_tier:
            return assigned_tier, "confirmed"

        # Default: trust planner but don't go below heuristic
        if heuristic_idx > assigned_idx:
            return heuristic_tier, f"upgraded-from-{assigned_tier}-safety"
        return assigned_tier, "planner-decision"

    def estimate_tokens(self, text: str, tier: str) -> int:
        """Estimate token usage for a step given its tier."""
        base = min(len(text) * 1.5, TIER_TOKEN_LIMITS.get(tier, 4096))
        # Reasoning/code tend to use more tokens for thinking
        multiplier = {"local": 0.5, "cheap": 0.7, "standard": 1.0, "code": 1.5, "reasoning": 2.0}
        return int(base * multiplier.get(tier, 1.0))
