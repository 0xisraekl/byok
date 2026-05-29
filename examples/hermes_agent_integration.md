# Hermes Agent + BYOK Integration

## What happens when you connect them

Without BYOK:
```
Hermes Agent → GPT-4o (always, for everything)
```

With BYOK:
```
Hermes Agent → BYOK Router → gemma3-local   (for simple chat)
                           → claude-3-5-sonnet (for complex reasoning)
                           → gpt-4o           (for tool calling)
                           → deepseek-reasoner (for math)
```

You don't change how you write your Hermes agents.
You just change one line in your config.

---

## Setup (2 minutes)

### Step 1 — Start BYOK
```bash
cd /path/to/byok
byok serve
```

You should see:
```
BYOK Proxy Server
  URL:  http://127.0.0.1:8000
  Models in your pool: 3
  • gpt-4o-mini (openai)
  • claude-3-5-sonnet (anthropic)
  • gemma3-local (ollama)
```

### Step 2 — Point Hermes Agent at BYOK

Find where Hermes Agent is configured (usually an `.env`, `config.py`, or agent init).

Change the model/API settings to:
```python
base_url = "http://localhost:8000/v1"
api_key  = "byok"   # any non-empty string works
model    = "auto"   # BYOK ignores this field and picks the best model
```

### Step 3 — Run your agent normally

Nothing else changes. Your Hermes agent code stays exactly the same.
BYOK handles the routing transparently.

---

## Verifying it works

After running your agent, check what BYOK decided:

```bash
byok log
```

Output:
```
Last 15 Routing Decisions
┌─────────────────────┬──────────────┬────────┬──────────────────┬───────────┐
│ Time                │ Task Type    │ Diff.  │ Model            │ Cost      │
├─────────────────────┼──────────────┼────────┼──────────────────┼───────────┤
│ 2026-05-23 14:32:01 │ coding       │ hard   │ claude-3-5-sonnet│ $0.00243  │
│ 2026-05-23 14:31:55 │ simple_chat  │ easy   │ gemma3-local     │ $0.00000  │
│ 2026-05-23 14:31:48 │ reasoning    │ medium │ gpt-4o           │ $0.00189  │
└─────────────────────┴──────────────┴────────┴──────────────────┴───────────┘
```

Or check the last routing decision via API:
```bash
curl http://localhost:8000/v1/routing/last | python -m json.tool
```

---

## Testing routing decisions without running your agent

```bash
# See what model would be chosen for a coding task
byok route "Write a Python function to parse and validate JSON"

# Test a summarization task
byok route "Summarize this 5000-word document"

# Force privacy routing (must use local model)
byok route "Analyze this confidential contract" --private

# Simulate a task with tool calling
byok route "Search the web for the latest AI news" --tools
```

---

## Multi-agent / sub-agent routing

For Hermes workflows with sub-agents, give each sub-agent a clear role in its system prompt. BYOK reads the system prompt and uses it as a routing signal.

```python
coding_agent_system_prompt = """
You are a coding agent and senior software engineer.
Implement, debug, refactor, and review code.
"""

research_agent_system_prompt = """
You are a research agent and analyst.
Compare options, reason through tradeoffs, and surface uncertainty.
"""

writer_agent_system_prompt = """
You are a writer sub-agent.
Rewrite, polish, and produce the final user-facing response.
"""
```

With those prompts:

```text
coding agent   → coding-specialist model
research agent → reasoning / analysis model
writer agent   → writing-specialist model
tool agent     → tool-capable model
private task   → local model only
```

If your Hermes setup can attach extra request metadata, BYOK also accepts explicit routing metadata:

```json
{
  "model": "auto",
  "messages": [
    {"role": "user", "content": "Handle this next implementation step."}
  ],
  "byok": {
    "task": "coding",
    "agent": "coder",
    "mode": "quality",
    "max_cost_usd": 0.004,
    "max_output_tokens": 1200
  }
}
```

If Hermes only lets you control prompt text, add a compact hint:

```text
[byok:task=coding,agent=coder,mode=quality] Handle this next implementation step.
```

This is optional. BYOK still classifies normal prompts automatically, but hints make short sub-agent tasks much more reliable.

For cost distribution, give each sub-agent a per-call budget:

```text
planner agent  → {"byok": {"task": "reasoning", "mode": "quality", "max_cost_usd": 0.010}}
coder agent    → {"byok": {"task": "coding", "mode": "balanced", "max_cost_usd": 0.004}}
tool agent     → {"byok": {"task": "tool_calling", "mode": "speed", "max_cost_usd": 0.003}}
writer agent   → {"byok": {"task": "writing", "mode": "cheap", "max_cost_usd": 0.001}}
```

BYOK uses that budget during routing. If a premium model would exceed the limit, it tries a cheaper capable model. If the selected model can fit only with a shorter answer, BYOK lowers `max_tokens` before forwarding the request.

For a whole Hermes task tree, pass the same run id and run budget to every sub-agent request:

```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "You are a coding agent and senior software engineer."},
    {"role": "user", "content": "Implement the next function."}
  ],
  "byok": {
    "run_id": "hermes-task-42",
    "max_run_cost_usd": 0.02
  }
}
```

BYOK tracks all calls with `run_id = "hermes-task-42"` in SQLite. If the planner, coder, tool, and writer sub-agents have already spent `$0.017`, the next call only has `$0.003` left. The router uses that remaining amount as the effective cost ceiling.

Fallbacks also respect the remaining budget. If the first-choice provider is down and BYOK tries a different backup model, it recalculates `max_tokens` using the backup model's own token price before forwarding the request.

If your Hermes/OpenAI-compatible client supports custom headers more easily than custom JSON fields, use:

```text
x-byok-run-id: hermes-task-42
x-byok-run-budget: 0.02
```

You can avoid repeating those budgets by editing `config/routing_policy.yaml`:

```yaml
agents:
  coding_agent:
    task: coding
    mode: balanced
    max_cost_usd: 0.004
    max_output_tokens: 1200

  research_agent:
    task: reasoning
    mode: quality
    max_cost_usd: 0.010
    max_output_tokens: 2000

  tool_agent:
    task: tool_calling
    mode: speed
    max_cost_usd: 0.003
    max_output_tokens: 700

  writing_agent:
    task: writing
    mode: cheap
    max_cost_usd: 0.002
    max_output_tokens: 900
```

Now a Hermes sub-agent with a system prompt like `You are a coding agent and senior software engineer` automatically gets the coding task, balanced mode, `$0.004` cost ceiling, and `1200` max output token cap unless the request explicitly overrides it.
