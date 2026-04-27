# CLI reference

The CLI installs as **two binary names** that point at the same entry —
`fewwords` (the brand) and `trajeval` (the package). Use whichever
matches the rest of your tooling.

## `fewwords run <trace.json>`

Run the Gate against a trace using a contract from `.trajeval.yml`
(or the one passed via `--config`). Prints a one-line summary plus
the per-check verdicts; exits non-zero on any failed check so it
slots straight into CI.

```bash
fewwords run examples/quickstart/simple_agent.trace.json \
  --config examples/quickstart/simple_agent.trajeval.yml
# -> fewwords: PASS — 6/6 checks passed
```

## `fewwords check <trace.json>`

Run a single inline assertion (``--assertion never_calls:drop_database``,
``--assertion must_visit:search,reply``, etc.) against a trace.
Useful when you want a one-shot probe rather than a full
``.trajeval.yml`` config.

## `fewwords scenario <dir>`

Regression runner. Every `*.yml` under the directory is a
(trace + config + expected-verdict) scenario.

## `fewwords init`

Auto-write a starter `.trajeval.yml` from the current repo. Detects the
agent framework (LangGraph, OpenAI, OTel, LiteLLM, AutoGen, LlamaIndex),
reads any trace JSON / JSONL files under conventional paths
(`./traces/`, `./logs/`, `./tmp/traces/`), mines banned-tool inferences
from destructive-keyword names, builds an allowed-tool list from observed
calls, and writes the result with a commented rationale per rule.

```bash
fewwords init
fewwords init --path ./my-agent/
fewwords init --traces ./recent-runs/
```

## `fewwords discover`

Records traces to `.trajeval/traces/` while your agent runs, then
synthesizes contracts once the threshold is met (default 10 traces).
Use the `Discovery` context manager for hands-off integration.

```bash
fewwords discover --threshold 10
```
