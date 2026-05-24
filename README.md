# BYOK — Bring Your Own Key

> Local-first model routing for AI agent frameworks.
> Use the right model for each task. Lower costs, avoid limits, keep sensitive work local, and stay in control of your own keys.

BYOK is an experimental OpenAI-compatible proxy that sits between an AI agent and your model providers. Instead of sending every task to the same model, BYOK classifies the task, checks your configured model pool, respects spend/privacy rules, and routes the request to the best available option.

---

## Why this exists

Most AI agent workflows have the same problem:

- one model is better for coding
- another is cheaper for simple tasks
- local models are better for privacy
- paid APIs hit limits or get expensive
- no single provider is always the best choice

So the question becomes:

> What if your agent had a small routing layer that decided which model should handle each task?

That's what BYOK is trying to become.

---

## Example

```text
Your Agent (Hermes / OpenClaw / Cursor / any OpenAI-compatible client)
      │
      │  "Write a SQL query that joins 3 tables"
      ▼
┌─────────────────────────────────────────┐
│              BYOK ROUTER                │
│                                         │
│  Task type:  coding                     │
│  Difficulty: medium                     │
│  Privacy:    normal                     │
│                                         │
│  Candidate models:                      │
│  - local-gemma                          │
│  - gpt-4o-mini                          │
│  - claude-sonnet  ← selected            │
│                                         │
│  Reason: strongest enabled coding model │
│          under configured spend limit   │
└─────────────────────────────────────────┘
      │
      ▼
  selected model answers the request
```

Simple chat can go to a local/free model. Complex coding can go to a stronger model. Sensitive requests can be forced local-only. Models at their spend limit can be skipped automatically.

---

## Current status

This is an early portfolio/learning project. The core shape is here:

- task classifier
- model registry from YAML
- rule-based router
- spend tracker
- provider interfaces
- FastAPI OpenAI-compatible proxy skeleton
- CLI commands
- tests for the routing pieces

It is not production-ready yet. I'm building it in public as I learn.

---

## Features

- **OpenAI-compatible proxy** — point compatible clients at `http://localhost:8000/v1`
- **Configurable model pool** — define available models in `config/models.yaml`
- **Task-aware routing** — coding, reasoning, math, writing, summarization, tool-calling, simple chat
- **Spend limits** — set a monthly USD cap per model
- **Privacy mode** — force sensitive/private requests to local models only
- **Local routing log** — record decisions, estimated cost, and chosen model in SQLite
- **Provider flexibility** — OpenAI, Anthropic, Ollama/local models, and OpenAI-compatible APIs
- **Transparent rules** — routing logic is simple, inspectable, and easy to change

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/0xisraekl/byok.git
cd byok
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Add API keys

```bash
cp .env.example .env
# Edit .env and add only the keys you want to use
```

### 3. Configure models

Edit `config/models.yaml` and enable the models you want BYOK to route to.

Each model can define:

- provider
- model ID
- strengths
- context window
- estimated cost
- local/cloud flag
- tool support
- monthly spend limit
- priority

### 4. Try routing without making an API call

```bash
byok route "Write a Python function that parses a CSV file"
byok route "Summarize these notes into action items"
byok route "Analyze this private customer document" --private
```

### 5. Start the proxy

```bash
byok serve
```

Then point an OpenAI-compatible client at:

```python
base_url = "http://localhost:8000/v1"
api_key = "byok"  # any non-empty string for local use
```

---

## CLI commands

```bash
# Start the proxy server
byok serve

# Check local setup/config before running the proxy
byok doctor

# Preview routing decisions
byok route "Write a function to parse XML"
byok route "Analyze this confidential document" --private
byok route "Search and summarize today's news" --tools
byok route "Draft a quick email" --mode cheap
byok route "Debug this production issue" --mode quality
byok route "Summarize this customer data" --mode private

# Inspect configured models
byok models

# Show recent routing decisions
byok log

# Show monthly spend vs. limits
byok spend
```

---

## How routing works

1. **Classify** the incoming request by task type and difficulty.
2. **Filter** out models that are disabled, over budget, missing required tool support, too small for the context, or not local when privacy mode is required.
3. **Score** the remaining models by task fit, local preference, latency, cost, and configured priority.
4. **Select** the highest-scoring model.
5. **Log** the decision locally for debugging and spend tracking.

See [`docs/architecture.md`](docs/architecture.md) for a deeper walkthrough.

---

## Project structure

```text
byok/
├── byok/
│   ├── core/
│   │   ├── classifier.py      # Task type + difficulty detection
│   │   ├── registry.py        # Loads config/models.yaml
│   │   └── router.py          # Filtering, scoring, model selection
│   ├── providers/             # Provider interfaces
│   ├── proxy/                 # FastAPI OpenAI-compatible server
│   ├── storage/               # SQLite spend/routing tracker
│   └── cli/                   # byok CLI
├── config/
│   └── models.yaml            # Model pool configuration
├── docs/
│   └── architecture.md
├── examples/
│   └── hermes_agent_integration.md
└── tests/
```

---

## Adding a model

Add a block to `config/models.yaml`:

```yaml
- name: "groq-llama"
  provider: openai_compatible
  model_id: "llama-3.3-70b-versatile"
  base_url: "https://api.groq.com/openai/v1"
  api_key_env: GROQ_API_KEY
  strengths:
    - simple_chat
    - reasoning
    - coding
  context_window: 128000
  cost_per_1k_input: 0.00059
  cost_per_1k_output: 0.00079
  latency: low
  supports_tools: true
  local: false
  enabled: true
  spend_limit_monthly_usd: 15.0
  priority: 1
```

Then add the key to `.env`:

```bash
GROQ_API_KEY=your_key_here
```

Restart `byok serve`.

---

## Similar tools

| Tool | What it does | How BYOK is different |
|---|---|---|
| LiteLLM | Multi-provider unified API | BYOK focuses on task-aware routing decisions |
| OpenRouter | Hosted model marketplace/router | BYOK is local-first and uses your own keys |
| RouteLLM | Learned routing between strong/weak models | BYOK is rule-based, transparent, and beginner-friendly |

---

## Roadmap

- [x] Add `byok doctor` for config/key/local-service diagnostics
- [x] Add routing modes for cheap, quality, private, and speed preferences
- [ ] Improve OpenAI-compatible chat/completions coverage
- [ ] Add richer provider health checks
- [ ] Add real token/cost accounting from provider responses
- [ ] Add a web dashboard for routing decisions and spend
- [ ] Add benchmark prompts for comparing routing quality
- [ ] Add config validation and better error messages
- [ ] Add example integrations with Hermes and other agent frameworks

---

## Contributing

This is a learning/public-build project. Issues, ideas, and PRs are welcome.

Good first contributions:

- add a provider adapter
- improve task classification rules
- improve docs/examples
- add tests for edge cases
- suggest routing heuristics

---

## License

MIT — free to use, fork, and build on.
