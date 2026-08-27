# NOESIS-Agent Core Integration

NOESIS-Agent uses Hermes' middleware architecture (#626) to intercept model calls
and perform per-step model switching. This document describes the integration points.

## Architecture Overview

```
User Message
    │
    ▼
Hermes Agent Loop
    │
    ├── pre_turn_hook
    │   └── NOESIS: TaskPlanner.plan(message) → TaskPlan
    │       └── StepExecutor.set_plan(plan)
    │
    ├── For each LLM call in the turn:
    │   └── wrap_model_call middleware
    │       ├── StepExecutor.wrap_model_call()
    │       │   ├── Get current step from ExecutionContext
    │       │   ├── ModelResolver.resolve(step.tier) → model_config
    │       │   ├── ToolScoper.scope_tools(all_tools, step.required_tools)
    │       │   ├── Apply model_config to request
    │       │   ├── Replace tools with scoped subset
    │       │   └── next_handler(request) → response
    │       └── Record metrics
    │
    └── post_turn_hook
        └── NOESIS: Report metrics, advance/replan if needed
```

## Integration Point 1: Middleware Hook (Primary)

Hermes #626 added `wrap_model_call` middleware. If your Hermes version supports it,
NOESIS-Agent registers automatically via `ctx.register_middleware()`.

**What the middleware does:**
1. Reads the current step from `ExecutionContext`
2. Resolves the step's tier to a concrete model
3. Modifies the request's `model`, `provider`, `base_url`, `api_key` fields
4. Replaces `tools` array with only the scoped subset
5. Passes modified request to the next handler
6. Records token usage and routing metrics from the response

**No core code changes needed** if middleware is available.

## Integration Point 2: Pre-turn Planning

The planner needs to run once at the start of each user turn. This requires
a hook point before the agent loop starts executing.

### If Hermes has pre_turn hooks:
```python
# In plugin register():
ctx.register_hook("pre_turn", _noesis_pre_turn)

def _noesis_pre_turn(user_message: str, agent: AIAgent, config: dict):
    planner = get_planner()
    executor = get_executor()
    resolver = get_resolver()

    # Store primary runtime for fallback
    resolver.set_primary_runtime({
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_key": agent.api_key,
    })

    # Get available tool names
    available_tools = [t.name for t in agent.tools] if hasattr(agent, 'tools') else []

    # Create plan (use agent's LLM for planning if available)
    async def llm_call(prompt, system):
        return await agent.complete(prompt, system=system)

    plan = planner.plan(user_message, available_tools, llm_call)

    # Validate tiers
    classifier = get_classifier()  # Need to expose this
    for step in plan.steps:
        step.tier, reason = classifier.validate_step(
            step.description, step.tier, step.confidence
        )

    executor.set_plan(plan)
```

### If no pre_turn hook (core patch needed):
In `cli.py`, `gateway/run.py`, and `tui_gateway/server.py`, before
calling `run_conversation()`:

```python
# --- NOESIS integration start ---
from noesis_agent.plugin_entry import get_planner, get_executor, get_resolver

planner = get_planner()
executor = get_executor()
resolver = get_resolver()

if planner and executor:
    resolver.set_primary_runtime({
        "model": agent_config.get("model", ""),
        "provider": agent_config.get("provider", "auto"),
        "base_url": agent_config.get("base_url"),
        "api_key": agent_config.get("api_key"),
    })
    plan = planner.plan(user_message, available_tools=None, llm_call_fn=None)
    executor.set_plan(plan)
# --- NOESIS integration end ---
```

## Integration Point 3: TUI/Desktop Model Switching

Hermes Desktop/TUI reuses a live `AIAgent`. When the middleware modifies the
request model, the TUI's session info needs to update.

The `StepExecutor` accepts a `model_switch_callback`:

```python
def switch_model_callback(config: dict):
    """Called when NOESIS switches models mid-turn."""
    if config.get("model") and config["model"] != current_agent.model:
        current_agent.switch_model(
            model=config["model"],
            provider=config["provider"],
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
        )
    if config.get("reasoning_config"):
        current_agent.reasoning_config = config["reasoning_config"]
    # Emit session info update for TUI status bar
    emit_session_info(...)

executor.set_model_switch_callback(switch_model_callback)
```

## Fallback Behavior

If any component fails, NOESIS gracefully degrades:

| Failure | Behavior |
|---------|----------|
| Planner LLM unavailable | Heuristic single-step plan |
| Local model (Ollama) down | Fall back to primary model |
| API key missing for tier | Fall back to primary model |
| Middleware not registered | Plugin provides preview tool only |
| Tool scoping removes all tools | Step executes with safe defaults |

## Metrics

After each turn, `executor.get_metrics()` returns:

```json
{
  "total_steps": 4,
  "completed_steps": 4,
  "local_calls": 2,
  "cloud_calls": 2,
  "total_tokens": 15420,
  "elapsed_seconds": 12.3,
  "estimated_savings_pct": 72.5
}
```

Compare this to a baseline where all 4 LLM calls would use the primary model
with all 54 tool schemas (~27k tokens each):
- Baseline: 4 × (27000 + ~2000) = ~116k tokens
- NOESIS: 15k tokens (87% reduction)
