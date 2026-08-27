# NOESIS-Agent: Planner-driven Per-step Model Routing

## Core Idea

Instead of Waylish's per-turn heuristic routing (keyword match → one model for entire turn),
NOESIS-Agent uses a **Planner-driven per-step routing** approach:

1. **Planner phase** (primary model, once per turn):
   - Analyze user request
   - Break into a task chain (ordered steps)
   - For each step, estimate difficulty and required tools
   - Assign each step a routing tier (local/cheap/standard/code/reasoning)

2. **Execution phase** (per-step model switching via middleware):
   - Each step executes with only its required tools' schemas (not all 54)
   - Simple steps (file ops, text transforms, format conversions) → local model (Ollama)
   - Complex steps (reasoning, multi-tool planning) → primary/cloud model
   - Middleware `wrap_model_call` hook swaps model before each step

3. **Memory integration** (NOESIS-II):
   - Planner uses long-term memory to inform task decomposition
   - Results of each step feed into memory for future context
   - Adaptive forgetting keeps context window efficient

## Token Savings Mechanism

Waylish only saves on model selection (cheaper model for simple turns).
NOESIS-Agent saves on **both** dimensions:
- **Model cost**: ~60-70% of steps can go to local/cheap models
- **Schema cost**: Per-step tool scoping reduces tool schema from ~27k tokens to ~2-5k tokens per step
- **Combined estimated savings**: 70-85% of total token consumption per turn

## Architecture

```
User Request
    │
    ▼
┌─────────────────┐
│  NOESIS Planner │  ← primary model (once per turn)
│  - Decompose    │
│  - Tier each    │
│  - Scope tools  │
└────────┬────────┘
         │ task_plan (JSON)
         ▼
┌─────────────────────────────────────┐
│  Execution Engine                    │
│  ┌──────────┐  ┌──────────┐         │
│  │ Step 1   │→│ Step 2   │→ ...    │
│  │ tier=loc │  │ tier=code│         │
│  │ tools=[A]│  │ tools=[B,C]│      │
│  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼───────────────┘
        │             │
        ▼             ▼
  ┌──────────┐  ┌──────────┐
  │ Ollama   │  │ Cloud    │
  │ local LLM│  │ API model│
  └──────────┘  └──────────┘
```

## Module Structure

```
noesis_agent/
├── __init__.py              # Plugin entry, register hooks
├── planner/
│   ├── __init__.py
│   ├── task_planner.py      # LLM-based task decomposition
│   └── step_schema.py       # Step data structure
├── router/
│   ├── __init__.py
│   ├── tier_classifier.py   # Difficulty estimation
│   ├── model_resolver.py    # Resolve tier → actual model/runtime
│   └── tool_scoper.py       # Filter tool schemas per step
├── middleware/
│   ├── __init__.py
│   └── step_executor.py     # wrap_model_call hook, per-step switching
├── memory/
│   ├── __init__.py
│   └── noesis_store.py      # NOESIS-II memory interface
└── tools/
    ├── __init__.py
    └── plan_preview.py      # Preview tool for debugging
```

## Routing Tiers

| Tier | When | Default Model | Notes |
|------|------|---------------|-------|
| `local` | Simple text ops, format conversion, file read/write, keyword extraction | Ollama local (Qwen2.5-7B) | Zero API cost |
| `cheap` | Short Q&A, simple search queries, basic summarization | DeepSeek v4 flash | Low cost |
| `standard` | Explanation, medium analysis, single-tool calls | Primary cloud model | Default fallback |
| `code` | Code generation, debugging, refactoring | Code-specialized model | With reasoning |
| `reasoning` | Complex analysis, multi-step planning, architecture decisions | Strongest model | High reasoning effort |

## Phase 1 Scope

- [x] Project structure and design
- [ ] Task planner (LLM-based decomposition)
- [ ] Tier classifier (heuristic + LLM confidence)
- [ ] Model resolver (Ollama + cloud providers)
- [ ] Tool scoper (per-step schema filtering)
- [ ] Middleware integration (wrap_model_call hook)
- [ ] Local model setup (Ollama)
- [ ] Core integration patch
- [ ] Unit tests

## Dependencies

- Hermes Agent (base, latest main)
- Ollama (for local model inference)
- httpx (for local model API calls)
- PyYAML (config)
