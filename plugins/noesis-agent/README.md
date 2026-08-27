# NOESIS-Agent

Planner-driven per-step model routing for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## What It Does

Instead of sending every LLM request through the strongest (most expensive) model with all 54+ tool schemas (~27k tokens per call), NOESIS-Agent:

1. **Plans**: Breaks a user request into an ordered chain of steps (once, via primary model)
2. **Routes**: Assigns each step a tier — local (Ollama), cheap, standard, code, or reasoning — based on complexity
3. **Scopes**: Sends only the tools each step needs (typically 0-3 tools instead of 54)
4. **Executes**: Middleware switches models per step automatically

**Estimated token savings: 70-85%** per turn, depending on task composition.

## How It's Different from Waylish/hermes-smart-model-routing

| | Waylish | NOESIS-Agent |
|---|---|---|
| Routing granularity | Per-turn (whole message → one model) | Per-step (each subtask → optimal model) |
| Routing logic | Keyword/heuristic matching | LLM planner + heuristic validation |
| Tool schemas | All tools sent every call | Scoped to per-step requirements |
| Local model | Not supported | First-class Ollama support |
| Memory | None | NOESIS-II memory integration (Phase 2) |

## Architecture

```
User Request → [Planner] → Task Plan (steps with tiers)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
         Step 1: local     Step 2: code       Step 3: reasoning
         tools=[read]      tools=[bash]       tools=[]
         Ollama 7B         DeepSeek Coder     Primary model
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                         Merged Result
```

## Installation

### Prerequisites
- Hermes Agent >= v0.20.0 with middleware support (#626)
- Python >= 3.10
- [Ollama](https://ollama.ai) (for local model tier, optional)

### Install

```bash
git clone -b feature/noesis-agent https://github.com/yubingz/hermes-agent.git
cd hermes-agent/plugins/noesis-agent
pip install -e .
```

Or install directly:

```bash
pip install git+https://github.com/yubingz/hermes-agent.git@feature/noesis-agent#subdirectory=plugins/noesis-agent
```

### Configure

Copy the config example into your Hermes config:

```bash
cp config.example.yaml ~/.hermes/noesis-config.yaml
```

Then merge the `noesis:` section into your Hermes `config.yaml`.

### Set up local model (optional but recommended)

```bash
# Install Ollama, then pull a small model
ollama pull qwen2.5:7b
```

## Usage

### Preview routing without executing

In Hermes:
```
/noesis_plan_preview "Fix the bug in auth.py and write tests"
```

### Automatic routing

Once the core integration is applied, routing happens automatically.
See [patches/core-integration.md](patches/core-integration.md) for details.

## Routing Tiers

| Tier | Use Case | Default Model |
|------|----------|---------------|
| `local` | File ops, format conversion, simple text | Ollama qwen2.5:7b |
| `cheap` | Short Q&A, search queries, summaries | DeepSeek chat |
| `standard` | Explanation, medium analysis | Primary model |
| `code` | Debug, refactor, test writing | DeepSeek coder |
| `reasoning` | Architecture, research, trade-offs | Primary model (high effort) |

All tiers are configurable via `config.yaml`.

## Development

```bash
# Run tests
python -m pytest tests/ -v

# All 22 tests should pass
```

## Roadmap

- [x] Core planner, router, tool scoper, middleware
- [x] Unit tests (22 passing)
- [x] Plugin entry point with preview tool
- [ ] LLM-based planner integration with Hermes
- [ ] Core integration patch for pre-turn hook
- [ ] Ollama auto-discovery and model selection
- [ ] NOESIS-II memory integration
- [ ] local_trace observability integration
- [ ] DSP security layer integration
- [ ] Cost tracking dashboard
- [ ] Adaptive routing based on success/failure feedback

## License

MIT
