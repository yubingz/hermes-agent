"""Unit tests for NOESIS-Agent."""
import sys
import os
import json
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from noesis_agent.planner.step_schema import PlanStep, TaskPlan
from noesis_agent.planner.task_planner import TaskPlanner
from noesis_agent.router.tier_classifier import TierClassifier
from noesis_agent.router.model_resolver import ModelResolver
from noesis_agent.router.tool_scoper import ToolScoper


class TestPlanStep(unittest.TestCase):
    def test_step_creation(self):
        step = PlanStep(
            step_id=1,
            description="Read the config file",
            tier="local",
            required_tools=["read_file"],
        )
        self.assertEqual(step.step_id, 1)
        self.assertEqual(step.tier, "local")
        self.assertIn("read_file", step.required_tools)

    def test_step_serialization(self):
        step = PlanStep(step_id=1, description="Test", tier="cheap")
        d = step.to_dict()
        self.assertEqual(d["step_id"], 1)
        restored = PlanStep.from_dict(d)
        self.assertEqual(restored.description, "Test")

    def test_task_plan_savings(self):
        steps = [
            PlanStep(step_id=1, description="Read file", tier="local", estimated_tokens=200),
            PlanStep(step_id=2, description="Analyze data", tier="reasoning", estimated_tokens=2000),
        ]
        plan = TaskPlan(user_request="test", steps=steps)
        plan.local_steps = 1
        plan.cloud_steps = 1
        savings = plan.estimated_savings
        self.assertGreater(savings, 0.3)  # At least 30% savings


class TestTaskPlanner(unittest.TestCase):
    def setUp(self):
        self.planner = TaskPlanner({"noesis": {"planner": {"enabled": True, "min_chars": 1000}}})

    def test_short_message_heuristic(self):
        plan = self.planner.plan("hello", available_tools=None, llm_call_fn=None)
        self.assertEqual(len(plan.steps), 1)
        self.assertIn(plan.steps[0].tier, ["cheap", "local", "standard"])

    def test_code_message_routing(self):
        plan = self.planner.plan(
            "Please debug this traceback and fix the Python function",
            available_tools=None,
            llm_call_fn=None,
        )
        self.assertEqual(plan.steps[0].tier, "code")

    def test_reasoning_message_routing(self):
        plan = self.planner.plan(
            "Analyze and compare these two architecture strategies, evaluate the trade-offs",
            available_tools=None,
            llm_call_fn=None,
        )
        self.assertEqual(plan.steps[0].tier, "reasoning")

    def test_empty_message(self):
        plan = self.planner.plan("", llm_call_fn=None)
        self.assertEqual(len(plan.steps), 1)

    def test_tool_inference(self):
        plan = self.planner.plan(
            "Search for the latest news about AI",
            available_tools=["search_web", "fetch_web", "read_file"],
            llm_call_fn=None,
        )
        tools = plan.steps[0].required_tools
        self.assertIn("search_web", tools)

    def test_llm_planner_with_mock(self):
        mock_response = json.dumps({
            "steps": [
                {"step_id": 1, "description": "Search info", "tier": "cheap",
                 "required_tools": ["search_web"], "estimated_tokens": 300, "confidence": 0.9},
                {"step_id": 2, "description": "Write report", "tier": "standard",
                 "required_tools": [], "estimated_tokens": 800, "confidence": 0.8},
            ]
        })

        def mock_llm(prompt, system):
            return mock_response

        # Need message longer than min_chars (1000)
        long_msg = "Please do a comprehensive research " * 50
        plan = self.planner.plan(long_msg, available_tools=["search_web"], llm_call_fn=mock_llm)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tier, "cheap")
        self.assertEqual(plan.local_steps + plan.cloud_steps, 2)


class TestTierClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TierClassifier()

    def test_code_classification(self):
        tier, conf = self.classifier.classify("Fix this traceback error in the Python function def")
        self.assertEqual(tier, "code")
        self.assertGreater(conf, 0.5)

    def test_reasoning_classification(self):
        tier, conf = self.classifier.classify(
            "Analyze and compare the trade-offs between these two strategies, evaluate the risks"
        )
        self.assertEqual(tier, "reasoning")

    def test_local_classification(self):
        tier, conf = self.classifier.classify("convert this to uppercase")
        self.assertEqual(tier, "local")
        self.assertGreater(conf, 0.7)

    def test_validate_upgrade(self):
        # Planner says "local" but text has code markers
        final, reason = self.classifier.validate_step(
            "debug this traceback error and fix the function",
            "local",
            confidence=0.4,
        )
        self.assertIn(final, ["code", "reasoning"])

    def test_validate_trust_high_confidence(self):
        final, reason = self.classifier.validate_step(
            "simple greeting",
            "cheap",
            confidence=0.98,
        )
        self.assertEqual(final, "cheap")


class TestToolScoper(unittest.TestCase):
    def setUp(self):
        self.scoper = ToolScoper()
        self.all_tools = [
            {"type": "function", "function": {"name": "read_file", "description": "Read a file"}},
            {"type": "function", "function": {"name": "write_file", "description": "Write a file"}},
            {"type": "function", "function": {"name": "search_web", "description": "Search the web"}},
            {"type": "function", "function": {"name": "bash", "description": "Run a command"}},
            {"type": "function", "function": {"name": "sessions_spawn", "description": "Spawn agent"}},
            {"type": "function", "function": {"name": "email_request", "description": "Send email"}},
        ]

    def test_scope_by_required_tools(self):
        result = self.scoper.scope_tools(self.all_tools, ["read_file", "search_web"])
        names = [t["function"]["name"] for t in result]
        self.assertIn("read_file", names)
        self.assertIn("search_web", names)
        self.assertNotIn("email_request", names)

    def test_restricted_tools_excluded(self):
        result = self.scoper.scope_tools(self.all_tools, ["sessions_spawn", "email_request"])
        names = [t["function"]["name"] for t in result]
        self.assertNotIn("sessions_spawn", names)
        self.assertNotIn("email_request", names)

    def test_safe_defaults_included(self):
        result = self.scoper.scope_tools(self.all_tools, ["search_web"])
        names = [t["function"]["name"] for t in result]
        self.assertIn("read_file", names)  # safe default

    def test_heuristic_inference(self):
        result = self.scoper.scope_tools(
            self.all_tools, [], "Search the web for latest news and read the results file"
        )
        names = [t["function"]["name"] for t in result]
        self.assertIn("search_web", names)
        self.assertIn("read_file", names)

    def test_max_tools_cap(self):
        scoper = ToolScoper({"noesis": {"max_tools_per_step": 2, "include_safe_defaults": False}})
        result = scoper.scope_tools(
            self.all_tools,
            ["read_file", "write_file", "search_web", "bash"],
        )
        self.assertLessEqual(len(result), 2)


class TestModelResolver(unittest.TestCase):
    def test_resolve_primary_fallback(self):
        resolver = ModelResolver()
        resolver.set_primary_runtime({
            "model": "gpt-4",
            "provider": "openai",
            "api_key": "test-key",
        })
        result = resolver.resolve("standard")
        self.assertEqual(result["model"], "gpt-4")

    def test_resolve_local(self):
        resolver = ModelResolver()
        result = resolver.resolve("local")
        # May fall back if Ollama not running
        self.assertIn(result["tier"], ["local", "standard"])
        if not result.get("fallback"):
            self.assertTrue(result["is_local"])

    def test_missing_api_key_fallback(self):
        resolver = ModelResolver({
            "noesis": {
                "models": {
                    "code": {
                        "provider": "deepseek",
                        "model": "deepseek-coder",
                        "api_key_env": "NONEXISTENT_KEY_12345",
                    }
                }
            }
        })
        resolver.set_primary_runtime({"model": "primary-model", "provider": "openai"})
        result = resolver.resolve("code")
        self.assertTrue(result.get("fallback"))


if __name__ == "__main__":
    unittest.main()
