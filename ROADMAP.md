# fewwords roadmap

This is the public roadmap. Internal planning lags this by a sprint or two; if a date here looks stale, file an issue.

## Shipped (v0.1.0, April 2026)

- LTL/Büchi runtime over agent trajectories: 4 formula shapes (`GloballyNever`, `Eventually`, `Precedes`, `Response`)
- `SymbolicState` typed-state engine for preconditions / postconditions
- 16-category check engine via `run_checks` (banned tools, ordering, schemas, retry storms, prior-work, dangerous-input patterns, user consent, intent, cascade, fault resilience, stop-on-error, PII, allowed-tools, tool-repeat, gates, cost-budget)
- Pre-execution guard with `check_pre` / `check_post` and tamper-evident state threading
- Cryptographic attestation: HMAC-SHA256 signed receipts with append-only ledger and chain integrity
- 13 vertical YAML packs (coding, GTM, customer service, browser, devops, financial, healthcare, legal, RAG, data pipelines, support, edit safety, generic)
- Trace adapters: OpenAI, OpenTelemetry GenAI spans, LangGraph events, LiteLLM logs, Anthropic Claude Code session logs, native fewwords JSON
- LangChain / LangGraph auto-patch via `init(mode="guard")`
- 14 reconstructed real-world incidents (Replit DROP DATABASE, Amazon Kiro env delete, Air Canada hallucinated policy, AutoGen retry storm, TCPA Air AI settlement, etc.)
- PocketOS-shape incident demo at `examples/incidents/`
- 155 unit tests passing (LTL semantics, state evaluation, guard pre/post, attestation chain, action engine, assertions, auto-contract, auto-detect)

## v0.2 (target: end of Q2 2026)

- **Public benchmark dashboard.** The hosted leaderboard at `fewword-ai.fly.dev/scores` ships fully open with reproducible methodology, raw per-trace results, and the published Fleiss' kappa from the external blinded raters (target: 2026-05-15).
- **Coverage to 60%** of τ-bench's 72 violation patterns. Currently 26%; the gap is structural-pattern expansion plus an LLM-graded fallback for postconditions outside the formal-methods-expressible subset.
- **Move `langchain-core` to `[langchain]` extras.** Lazy-import in `sdk/callback.py` so `pip install fewwords` doesn't pull the LangChain stack for users on raw OpenAI / Anthropic SDK.
- **Learned per-agent drift baseline.** Today's drift is check-fire-rate proxy; the learned baseline engine does per-config trajectory embedding + distribution model + anomaly-score against rolling baseline.
- **MCP-native server.** Expose `fewwords.check` as an MCP tool so any Claude Code / Cursor / Aider session can wire it as a PreToolUse hook with one config line.

## v0.3 (target: end of Q3 2026)

- **TypeScript SDK** at `npm install fewwords-ai`. Same `.fewwords.yml` grammar; same verdicts; same attestation receipts. Required for non-Python agent runtimes.
- **Insurance-feed format.** Underwriter-grade exposure data (frequency × severity × correlation per agent config), specified as JSON-schema-validated rows. Coalition / At-Bay / Marsh-shape consumption.
- **VAS-001 v2.** Agent identity passport with scoped delegation; revocation event protocol with operator-side enforcement + public revocation feed for compliance verification.
- **OWASP LLM01:2025-derived prompt-injection corpus.** ~200 cases scored against fewwords + LLM-judge baselines.

## Q4 2026 and beyond

- **Authorization corpus + scoring** for the agent-overstep dimension.
- **Recovery corpus** for chaos-engineering / fault-injection traces.
- **Notified-body licensing partnerships** (Big 4-shape, EU AI Act Article 43 conformity-assessment integration).
- **Rust runtime engine** for sub-100µs latency on the hot path (today's 0.14ms is in-process Python).

## What we deliberately won't ship

- **A managed dashboard product that competes with Vanta / Drata.** fewwords is the verification primitive; governance dashboards consume our signed receipts as evidence. Not the product we're building.
- **An agent runtime.** fewwords composes with whatever agent stack you're already on (LangGraph, OpenAI, raw HTTP). We don't build a competing agent framework.
- **An LLM-as-judge baseline as a primary product.** We benchmark against LLM judges; we don't sell one.

## How to influence the roadmap

- Open an issue with the `[FEATURE]` template if you want something added.
- Send a real production trace (sanitized) and the failure mode it exposed; if your trace surfaces a category we don't cover, the corresponding pattern moves up the queue.
- The single highest-leverage feedback we can get is *"my agent did X and fewwords didn't catch it; here's the trace."*
