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

## Notes

`fewwords init` and `fewwords discover` are documented in the
private source. They depend on the contract-suggestion engine
which stays invite-only — email for access if you need them.
