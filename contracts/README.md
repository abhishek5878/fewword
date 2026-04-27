# fewwords contract packs

Pre-built `.trajeval.yml` configs for the most common agent verticals. Each pack ships **banned tools, required workflow steps, ordering contracts, schemas, and HITL consent gates** that catch the failure modes specific to that vertical.

**One-command onboarding.** Pick the pack closest to your agent, drop it in, ship.

```bash
# List available packs (with their aliases)
fewwords init --list-packs

# Print the pack to stdout (inspect before committing)
fewwords init --pack code-agent

# Write to .trajeval.yml in the current directory
fewwords init --pack code-agent --write
```

## Pick this pack if your agent does…

| If your agent… | Pick | Catches |
|---|---|---|
| Edits code, runs shell, manages a repo (Cursor, Claude Code, Replit, Aider, Devin, Codex) | [`code-agent`](code_agents.yml) | DROP DATABASE, rm -rf, destroy_stack, infinite retries, blind edits |
| Edits files specifically (a narrower variant of code-agent) | [`edit-safety`](edit_safety.yml) | Read-before-Edit ordering, runaway thrash. Stack with `code-agent` for full coverage. |
| Drives a browser (Browser-Use, Stagehand, Computer Use, Operator) | [`browser-agent`](browser_agent.yml) | Form submission without consent, unauthorized JS exec, navigation loops, PII into wrong domain |
| Manages infrastructure (Kiro, Terraform CDK, AWS CDK, k8s autopilot) | [`devops`](devops.yml) | Production deletion, plan-without-apply, force-push to main, credential exfil |
| Talks to customers (chatbots, refund agents, claim adjusters) | [`customer-service`](customer_service.yml) | Air Canada–style hallucinated policy, refund without identity verification, PII leakage |
| Runs ETL / text-to-SQL / data-pipeline orchestration | [`data-pipeline`](data_pipeline.yml) | Schema drift, dropped data, missing PII flow tracking |
| Does outbound sales / GTM (Ruzo-style, Clay-adjacent) | [`sales`](sales.yml) | GDPR/CAN-SPAM violations, over-personalization spam, banned-cadence loops |
| Triages support tickets (lighter than `customer-service`) | [`support`](support.yml) | Mis-routing, escalation-loop, agent-says-feature-exists hallucination |
| Handles money / payments / banking | [`financial`](financial.yml) | Unauthorized transfers, transaction without confirmation, missing audit trail |
| Retrieves over a corpus and answers (RAG, doc search, Q&A) | [`rag`](rag.yml) | Hallucinated source, retrieval-without-citation, answer-before-retrieve |
| Operates in healthcare context | [`healthcare`](healthcare.yml) | HIPAA exposure, drug-interaction missed, prior-auth missing |
| Operates in legal context | [`legal`](legal.yml) | Privilege leakage, jurisdiction mismatch, unauthorized practice |
| Doesn't fit any of the above | [`generic`](generic.yml) | Universal checks only: retry storms, repetition, stop-on-error, PII |

Aliases (e.g. `cs`, `iac`, `etl`, `gtm`) all map to a canonical pack, `fewwords init --list-packs` shows the full mapping.

## What's in each pack

Every pack uses the same `.trajeval.yml` schema (parsed by [`trajeval.action.load_config`](../src/trajeval/action.py)):

- `banned_tools`: globbable patterns; the agent calling `rm_rf_recursive` matches `*rm_rf*`.
- `required_tools`: every trajectory must include at least one call to each.
- `cost_budget_usd`: total session cost cap.
- `max_retries` / `max_tool_repeat`: retry-storm + repetition control.
- `stop_on_error`: fail loud after the first 4xx/5xx tool response.
- `contracts`: natural-language ordering rules ("X before Y") compiled to LTL.
- `schemas`: JSON-schema validation on tool outputs.
- `require_user_consent_before`: list of write-action tools that need explicit user "yes" before firing.
- `strict_consent_only`: reject permissive phrasings ("sure", "okay") and require literal yes / confirm / proceed family. Recommended for compliance-heavy domains.
- `check_pii`: PII regex over tool outputs.

See any pack file for working examples.

## Stacking packs

Some packs are **complementary**, designed to layer:

- `code-agent` (banned-tools focus) + `edit-safety` (read-before-edit ordering) → full code-agent coverage
- `customer-service` (HITL + identity) + `financial` (banking-specific banned tools) → fintech-CS bot
- Any pack + `generic` → if `generic` is layered last, you get the universal safety net even when the vertical pack is silent

**To stack:** copy both files into your repo and let the three-scope merge resolve overlaps (per [src/trajeval/action.py](../src/trajeval/action.py)). Roadmap: native multi-pack support via `--pack a,b`.

## Authoring custom packs

Start from the closest pack as a baseline, then customize:

```bash
fewwords init --pack browser-agent --write       # writes contracts/browser_agent.yml as .trajeval.yml
$EDITOR .trajeval.yml                             # tighten / loosen rules
```

The pack is just YAML, no Python, no DSL. Add your own banned tools, required tools, contracts, schemas, consent gates.

If your custom pack stabilizes and is reusable, contribute it back as a new pack file. Tests are in [tests/unit/test_contract_packs.py](../tests/unit/test_contract_packs.py); they parametrize over `_PACK_ALIASES`, so adding a pack auto-extends test coverage.

## When packs aren't enough

If a pack flags too much (false positives) or too little (false negatives), the next escalation is:

1. **Tune the pack**: drop banned tools that don't apply, loosen `max_retries`, etc.
2. **Author a custom contract**: `your_tool before any_other_tool` is one line.
3. **Synthesize from your trace corpus**: `fewwords init` (no `--pack`) scans your repo for framework signals + trace files and synthesizes a starter config. Better than starting from a pack if your agent is unusual.
4. **Compose primitives directly**: drop the pack and write your own `.trajeval.yml` from scratch using [src/trajeval/assertions/core.py](../src/trajeval/assertions/core.py) primitives.

## Pack quality bar

For a pack to ship to `contracts/*.yml`:

1. **Loads cleanly** via `load_config` (verified by `tests/unit/test_contract_packs.py`).
2. **Has at least one non-empty primitive**: banned_tools, required_tools, contracts, schemas, or require_user_consent_before. (Exception: `generic.yml` is intentionally near-empty.)
3. **Detects a synthetic violation**: calling a banned tool produces a failed check (parametrized test).
4. **Has a comment header**: what the pack is for, which incidents it prevents, an example invocation.
5. **Is reachable via at least one alias** in [`src/trajeval/cli.py:_PACK_ALIASES`](../src/trajeval/cli.py).
