# BYOK — Bring Your Own Key

> Intelligent model routing for AI agent frameworks.
> Use the right model for each task. Lower costs. Higher quality. Your keys, your control.

---

## The Problem

Most AI agents use one model for everything.

That means you pay GPT-4o prices to answer "what's the capital of France?" and you use a cheap model to write production code. Both are wrong.

## The Solution

BYOK sits between your agent and your models. It reads each incoming task, figures out what kind of task it is, and routes it to the best model in **your configured pool**.

```
Your Agent (Hermes / OpenClaw / any framework)
      │
      │  "Write a SQL query that joins 3 tables"
      ▼
┌─────────────────────────────────────────┐
│             BYOK ROUTER                 │
│                                         │
│  Task type:  coding   (hard)            │
│  Pool:       gemma3-local               │
│              gpt-4o-mini                │
│              claude-3-5-sonnet  ← picks │
│              deepseek-reasoner          │
│                                         │
│  Reason: strong at coding, fits context │
└─────────────────────────────────────────┘
      │
      ▼
  claude-3-5-sonnet answers the task
```

**Simple chat?** → free local model.
**Complex reasoning?** → your best reasoning model.
**Model at its spend limit?** → auto-fallback to the next best.
**Private/sensitive data?** → local-only, never sent to the cloud.

---

## Features

- **OpenAI-compatible proxy** — point any agent at `http://localhost:8000/v1`, no code changes
- **Your model pool** — define exactly which models BYOK can use, in `config/models.yaml`
- **Task-aware routing** — classifies: coding, reasoning, math, writing, summarization, tool-calling, simple chat
- **Spend limits** — set a monthly USD cap per model; BYOK stops routing there when reached
- **Privacy mode** — keyword detection forces local-only routing for sensitive requests
- **Full routing log** — every decision is stored locally with the reason, tokens, and cost
- **Zero cloud dependencies** — runs entirely on your machine

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/byok
cd byok
pip install -e .
```

### 2. Add your API keys

```bash
cp .env.example .env
# Edit .env and add your keys
```

### 3. Configure your model pool

Edit `config/models.yaml`. Enable the models you have keys for.
Each model has `enabled: true/false` — flip it on for the ones you want.

### 4. Start BYOK

```bash
byok serve
```

### 5. Point your agent at BYOK

In your Hermes Agent (or any OpenAI-compatible framework):
```python
base_url = "http://localhost:8000/v1"
api_key  = "byok"   # any non-empty string
```

That's it. BYOK handles the rest.

---

## CLI Commands

```bash
# Start the proxy server
byok serve

# See what model would be chosen (without making any API call)
byok route "Write a function to parse XML"
byok route "Analyze this confidential document" --private
byok route "Search and summarize today's news" --tools

# Show your model pool + spend status
byok models

# Show recent routing decisions
byok log

# Show monthly spend vs. limits
byok spend
```

---

## Adding a Model

Open `config/models.yaml` and add a block:

```yaml
- name: "my-groq-llama"
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

Then add `GROQ_API_KEY=your_key` to `.env`. Restart `byok serve`. Done.

---

## Supported Providers

| Provider | Type | Notes |
|---|---|---|
| OpenAI | `openai` | GPT-4o, GPT-4o-mini, etc. |
| Anthropic | `anthropic` | Claude 3.5 Sonnet, Haiku, etc. |
| Ollama | `ollama` | Any local model — free and private |
| DeepSeek | `openai_compatible` | Strong reasoning, cheap |
| Groq | `openai_compatible` | Very fast inference |
| Any OpenAI-compatible API | `openai_compatible` | Set `base_url` |

---

## How Routing Works

1. **Classify** — Read the incoming messages. Match keyword patterns to detect task type (coding, reasoning, summarization, etc.) and difficulty.

2. **Filter** — Remove models that can't handle the task: wrong context size, no tool support, at spend limit, not local when privacy is required.

3. **Score** — Each remaining model gets a score:
   - +10 if the task type matches a strength
   - +5 for secondary matches
   - +3 for local (free) models
   - −2 for high-latency models
   - −3 if the model seems too cheap for a hard task
   - −∞ (disqualified) if at spend limit

4. **Select** — Highest score wins. Ties broken by `priority` in config.

5. **Log** — Every decision is recorded in `byok.db` with reason, tokens, and cost.

---

## Project Structure

```
byok/
├── byok/
│   ├── core/
│   │   ├── classifier.py   # Task type detection
│   │   ├── router.py       # Scoring and model selection
│   │   └── registry.py     # Loads models.yaml
│   ├── providers/
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── ollama_provider.py
│   ├── proxy/
│   │   └── server.py       # FastAPI OpenAI-compatible server
│   ├── storage/
│   │   └── spend_tracker.py  # SQLite log
│   └── cli/
│       └── main.py         # byok CLI
├── config/
│   └── models.yaml         # Your model pool (edit this)
├── examples/
│   └── hermes_agent_integration.md
└── .env.example
```

---

## Compared to Similar Tools

| Tool | What it does | BYOK difference |
|---|---|---|
| LiteLLM | Multi-provider unified API | BYOK adds task-aware routing intelligence |
| OpenRouter | Hosted routing service | BYOK is local-first, no third party, your keys |
| RouteLLM | ML-based routing | BYOK is rule-based + config, transparent and beginner-friendly |

---

## License

MIT — free to use, fork, and build on.

---

## Contributing

This is a portfolio/learning project. Issues and PRs welcome.
See `docs/architecture.md` for a deeper technical walkthrough.
