# BYOK architecture

BYOK is designed as a small routing layer between AI clients and model providers.

The goal is not to hide model choice completely. The goal is to make model choice explicit, configurable, and easier to reason about.

---

## High-level flow

```text
OpenAI-compatible client
        │
        ▼
FastAPI proxy
        │
        ▼
Task classifier
        │
        ▼
Model registry
        │
        ▼
Router / scorer
        │
        ▼
Provider adapter
        │
        ▼
OpenAI / Anthropic / Ollama / OpenAI-compatible provider
```

---

## Core components

### 1. Proxy

The proxy exposes an OpenAI-compatible API surface so agent frameworks can point to BYOK with minimal changes.

Initial target:

```text
POST /v1/chat/completions
```

The proxy should:

1. accept incoming chat requests,
2. extract the task content and metadata,
3. ask the router for a model decision,
4. forward the request to the selected provider,
5. return the provider response in OpenAI-compatible format,
6. log the decision and estimated cost.

---

### 2. Task classifier

The classifier converts a user request into routing signals.

Example signals:

- task type: `coding`, `reasoning`, `math`, `writing`, `summarization`, `extraction`, `data_analysis`, `tool_calling`, `simple_chat`
- difficulty: `easy`, `medium`, `hard`
- privacy requirement: normal vs. private/local-only
- tool requirement: whether function/tool calling is needed
- context requirement: estimated context size

The first implementation is intentionally rule-based so the logic is easy to inspect and improve.

---

### 3. Model registry

The registry loads `config/models.yaml` and turns it into a model pool.

Each model can define:

- name
- provider
- model ID
- base URL
- API key env var
- strengths
- general `quality_score`
- per-task `task_quality` overrides
- context window
- cost estimates
- latency class
- tool support
- local/cloud flag
- enabled flag
- monthly spend limit
- priority

The registry is the source of truth for what BYOK is allowed to use.

---

### 4. Router

The router does three jobs:

1. **Filter** models that cannot handle the task.
2. **Score** the remaining candidates.
3. **Select** the best model and explain why.

Common filters:

- disabled model
- missing API key
- over spend limit
- projected next call would exceed spend limit
- request-level cost ceiling cannot fit the prompt/output budget
- insufficient context window
- lacks tool support
- cloud model when privacy mode requires local

Routing modes add preferences on top of the default scoring:

- `balanced`: default; pick near-best task quality at lower cost when possible
- `cheap`: prefer the cheapest model that clears the task's quality floor
- `quality`: prefer the strongest specialist and tolerate higher cost
- `private`: force local-only routing
- `speed`: prefer low-latency models

Scoring considers:

- task/type fit
- configured strengths
- per-task quality priors
- estimated request cost
- estimated output length by task/difficulty
- optional request-level cost ceiling
- latency
- local preference
- priority
- difficulty fit

The router also reports:

- selected model
- task-specific quality estimate
- estimated request cost
- best-quality model in the available pool
- premium reference cost
- estimated savings when a cheaper near-best model wins

The router should always return a human-readable reason for the selection.

---

### 4.1 Sub-agent routing policy

`config/routing_policy.yaml` defines defaults for detected sub-agent roles.

Example:

```yaml
agents:
  coding_agent:
    task: coding
    mode: balanced
    max_cost_usd: 0.004
    max_output_tokens: 1200
```

The policy layer runs after classification and before routing:

1. The classifier detects a role from prompt text, such as `coding_agent`.
2. BYOK loads the matching policy defaults.
3. Policy can set task type, routing mode, max cost, and max output tokens.
4. Explicit request metadata still overrides policy defaults.

This makes Hermes/OpenClaw multi-agent routing practical: each sub-agent can
inherit its own budget and routing mode without modifying framework internals.

---

### 5. Provider adapters

Provider adapters translate BYOK's internal request into provider-specific calls.

Planned adapter types:

- OpenAI
- Anthropic
- Ollama
- OpenAI-compatible APIs such as Groq, DeepSeek, OpenRouter-compatible endpoints, etc.

Provider adapters should hide provider-specific differences from the router.

---

### 6. Spend tracker

The spend tracker records routing decisions and estimated cost in SQLite.

Useful fields:

- timestamp
- selected model
- provider
- task type
- estimated input tokens
- estimated output tokens
- estimated cost
- routing reason
- optional run/session id

Early versions may use estimated token counts. Later versions should use actual provider usage fields when available.

For multi-agent workflows, BYOK can enforce a run-level budget. The proxy stores
`run_id` on every logged request, computes spend already used by that run, and
passes the remaining `max_run_cost_usd` to the router as the effective cost
ceiling. This lets planner, coder, tool, and writer sub-agents share one budget
without the agent framework needing native BYOK integration.

---

## Design principles

### Local-first

BYOK should run on the user's machine and keep routing data local by default.

### Bring your own keys

BYOK should not require a hosted account. Users configure their own provider keys.

### Transparent routing

Every decision should be explainable. If BYOK chooses a model, the user should be able to see why.

### Config over magic

The user should be able to edit `config/models.yaml` and understand what changed.

### Beginner-friendly

The project should stay readable enough for a beginner to learn from it while still being useful.

---

## Future ideas

- dashboard for spend and routing history
- provider health checks
- benchmark suite for routing quality
- per-project routing profiles
- sensitive-data detector beyond keyword rules
- automatic fallback when a provider errors or rate-limits
- model quality scores learned from user feedback
