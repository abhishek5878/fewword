# Contract reference

A contract is a YAML file read from `.trajeval.yml` (or from
`--config <path>`). Every key is optional; leaving a key out means
"no check of that kind."

## `allowed_tools`

List of tool names the agent may call. Anything not on the list
fails the `allowed_tools` check.

## `banned_tools`

List of tool names the agent must never call. A match fails the
`banned_tools` check regardless of context.

## `requires_prior_work`

Map from tool name to the tools that must have been called before
it. Useful for "payment requires verification", "reply requires
search", "delete requires user-confirm".

## `max_retries`

Upper bound on consecutive repeats of the same tool. Guards against
retry storms.

## `max_tool_repeat`

Upper bound on total calls of the same tool across the trajectory.

## `cost_budget_usd`

Dollar cap on total trace cost. Exceeding the cap fails the
`cost_budget` check.

## `schemas`

Per-tool JSON Schema for validating tool outputs. Useful for
catching schema-thin responses like `{"results": []}`.

## `dangerous_input_patterns`

Regex patterns that, if matched against tool inputs, fail the
check. Covers prompt-injection and SQL-injection surface area.

## `contracts`

Free-form English clauses. Each clause is compiled to a Python
expression, JSON Schema, or LTL formula at load time. Clauses are
always deterministic; there is no LLM in the check path.

## `tools.<name>.{requires, postcondition}` (R4.2b)

Per-tool typed-state contracts. Three pieces compose:

- `requires:` — list of `state.X op literal` predicates that must hold
  before the tool fires (e.g. `state.reservation_known == true`,
  `state.reservation_cabin != basic_economy`). Operators: `==, !=,
  <, <=, >, >=, in, exists`. Conjunction-only; compose multiple
  contracts for OR.
- `postcondition.returns:` — JSON-Schema-style validator over the
  observed `tool_output` (reuses the same schema engine as
  `schemas:`). Schema mismatch produces a failing
  `postcondition_schema[<tool>]` check and blocks state mutation.
- `postcondition.state_updates:` — flat dict of state writes applied
  on schema-passing output. Values are literals or `{tool_output_key}`
  template tokens (single-token only in v1).

Example:

```yaml
tools:
  get_reservation_details:
    postcondition:
      returns:
        reservation_id: str
        cabin: str
        status: str
      state_updates:
        reservation_cabin: "{cabin}"
        reservation_status: "{status}"
        reservation_known: true

  cancel_reservation:
    requires:
      - "state.reservation_known == true"
      - "state.reservation_cabin != basic_economy"
    postcondition:
      returns:
        success: bool
      state_updates:
        reservation_status: cancelled
```

State values are flat primitives: `str | int | float | bool | None`.
No nested dicts, no lists, no custom types — sub-millisecond p50 is a
hard budget. The check fires both offline (`run_checks`) and at
runtime (`guard.check_pre` / `check_post`). On-domain receipts at
[fewword-ai.fly.dev/benchmarks/postconditions](https://fewword-ai.fly.dev/benchmarks/postconditions) — 19 of 72 audited Sonnet-unique flags closed at 0.126 ms p50 on τ-bench airline + retail.
