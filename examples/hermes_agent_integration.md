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
