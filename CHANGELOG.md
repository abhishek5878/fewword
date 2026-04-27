# Changelog

All notable changes to fewwords are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-28

The initial public release.

### Added

- **LTL / Büchi runtime** over agent trajectories. Four formula shapes (`GloballyNever`, `Eventually`, `Precedes`, `Response`) compiled at load time, evaluated as deterministic explicit-state automata in microseconds.
- **`SymbolicState` typed-state engine** with parsed predicates (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `exists`), immutable state transitions, and well-defined missing-key semantics.
- **`run_checks` engine** running 16 categories of contracts: banned tools, allowed tools, ordering, schemas, retry storms, prior-work, dangerous-input regex patterns, user consent, intent inference, cascade root-cause analysis, fault resilience, stop-on-error, PII detection, tool-repeat caps, gate logic, cost-budget enforcement.
- **Pre-execution guard** with `check_pre` and `check_post`, tamper-evident state threading. The `would_enter_reject` probe is non-mutating, so guard mode can verify a proposed call without advancing the trace.
- **Cryptographic attestation** module (`fewwords.attestation`). HMAC-SHA256 signed receipts. Append-only JSONL ledger with chain integrity (each receipt's `prev_signature` points at the previous receipt; removing any record breaks the chain). Tamper detection via `verify_receipt` and `Ledger.verify_chain`.
- **13 vertical contract packs**: `code_agents`, `sales`, `customer_service`, `browser_agent`, `devops`, `data_pipeline`, `financial`, `healthcare`, `legal`, `rag`, `support`, `edit_safety`, `generic`.
- **6 trace adapters**: OpenAI tool-call format, OpenTelemetry GenAI spans (OTLP), LangGraph events, LiteLLM callback logs, Anthropic Claude Code session logs, native fewwords JSON. Auto-detection via `adapters/auto.py`.
- **CLI surface** (13 subcommands): `check`, `report`, `run`, `benchmark`, `suggest`, `dogfood`, `preflight`, `drift`, `scenario`, `discover`, `init`, `version`, `compose`. Both `fewwords` and `trajeval` console scripts work; both point at the same entry.
- **PocketOS reconstruction** at `examples/incidents/pocketos_drop_database.trace.json` + matching YAML rule. Visitors can run the trace and watch fewwords block the destructive Railway GraphQL call.
- **Quickstart example** at `examples/quickstart/`.
- **155 unit tests** passing (LTL semantics, `SymbolicState` evaluation, guard pre/post, attestation sign/verify/chain integrity, action engine, assertion library, auto-contract suggestion, auto-detect adapter).
- **τ-bench head-to-head benchmark** numbers documented: 80% audit agreement on enforced contracts vs Claude Sonnet 4.6's 57%. 0.14 ms median check vs 5.9 s for the LLM-judge baseline. Across 1,200 LLM-judge–trace pairs, zero violations missed on the contracts fewwords enforces.
- **Documentation**: `docs/cli.md`, `docs/contracts.md`, `docs/adapters.md`, `docs/prove-it.md` (14 reconstructed real-world incidents).
- **GitHub Actions CI**: lint + quickstart-passes-PASS-6/6 on every push.
- **Honest disclosures section**: 26% τ-bench coverage (formal-methods-expressible subset), founder-rated audit (external Fleiss' kappa publishes by 2026-05-15), no insurance LOI yet, hosted dashboard in early access, public bundle is a curated snapshot of an internal engine.

### Changed

- N/A (initial release)

### Known limitations

- `langchain-core` is currently a hard runtime dependency for the LangGraph auto-patch path. Moving it to `[langchain]` extras is on the v0.2 list.
- Public bundle does not include the contract YAML compiler (`trajeval.contract.compiler` and adjacent modules); ships only `state.py` for postcondition primitives. The full compiler is in the invite-only internal repo.
- 26% τ-bench coverage. The remaining 74% requires per-deployment contract authoring (the white-glove offer ships customers with a working YAML in 48 hours) plus structural-pattern expansion (target 60% coverage by EOY 2026).
- Founder-rated benchmark audit. External blind raters being commissioned this week; Fleiss' kappa across all three publishes 2026-05-15 before the public leaderboard launches.

### Notes

- The Python distribution name is `fewwords`; the internal Python module is `trajeval` (the original engine name from before the rebrand). Both `from fewwords import` and `from trajeval import` work. So do both `fewwords` and `trajeval` console scripts. Future versions will consolidate, but v0.1.0 keeps both for backward compatibility.

## [Unreleased]

See [ROADMAP.md](ROADMAP.md) for what's coming in v0.2 (Q2 2026), v0.3 (Q3 2026), and beyond.
