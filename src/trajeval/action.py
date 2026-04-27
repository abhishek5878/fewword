"""TrajEval Action — the "boring version" that ships.

Single entry point: read a ``.trajeval.yml`` config and a trace JSON file,
run the 5 core checks, return a structured result.  This is the engine behind
the GitHub Action, the ``POST /evaluate`` endpoint, and the ``fewwords run``
CLI command.

The 5 checks
-------------
1. **Retry storm** — ``no_retry_storm(max_consecutive)``
2. **Cost budget** — ``cost_within(p90)``
3. **Banned tools** — ``never_calls`` per tool
4. **Required tools** — ``must_visit``
5. **Schema validation** — ``validate_tool_outputs`` per tool

Usage::

    from trajeval.action import run_checks, load_config

    config = load_config(Path(".trajeval.yml"))
    result = run_checks(trace, config)
    print(result.summary)     # human-readable
    print(result.to_json())   # machine-readable
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from trajeval.contract.state import (
    SymbolicState,
    ToolContract,
    parse_tools_section,
    resolve_state_updates,
    validate_returns,
)
from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class ActionResult:
    """Aggregate result of all checks on one trace."""

    trace_id: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    @property
    def summary(self) -> str:
        status = "PASS" if self.all_passed else "FAIL"
        header = (
            f"fewwords: {status} — "
            f"{self.passed}/{self.total} checks passed"
        )
        if self.all_passed:
            return header
        lines = [header]
        for c in self.checks:
            if not c.passed:
                lines.append(f"  FAIL {c.name}: {c.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "all_passed": self.all_passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message}
                for c in self.checks
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class ActionConfig:
    """Parsed ``.trajeval.yml`` config."""

    banned_tools: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    cost_budget_usd: float | None = None
    max_retries: int = 3
    schemas: dict[str, dict[str, object]] = field(default_factory=dict)
    contracts: list[str] = field(default_factory=list)
    show_intent: bool = False
    show_cascade: bool = False
    fault_test: bool = False
    stop_on_error: bool = False
    check_pii: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    max_tool_repeat: int | None = None
    gates: list[dict[str, object]] = field(default_factory=list)
    requires_prior_work: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    dangerous_input_patterns: dict[str, list[str]] = field(
        default_factory=dict
    )
    require_user_consent_before: list[str] = field(default_factory=list)
    strict_consent_only: bool = False
    tools: dict[str, ToolContract] = field(default_factory=dict)


def load_config(path: Path) -> ActionConfig:
    """Load an :class:`ActionConfig` from a YAML file."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(raw).__name__}")
    return ActionConfig(
        banned_tools=raw.get("banned_tools", []),
        required_tools=raw.get("required_tools", []),
        cost_budget_usd=raw.get("cost_budget_usd"),
        max_retries=raw.get("max_retries", 3),
        schemas=raw.get("schemas", {}),
        contracts=raw.get("contracts", []),
        show_intent=raw.get("intent", False),
        show_cascade=raw.get("cascade", False),
        fault_test=raw.get("fault_test", False),
        stop_on_error=raw.get("stop_on_error", False),
        check_pii=raw.get("check_pii", False),
        allowed_tools=raw.get("allowed_tools", []),
        max_tool_repeat=raw.get("max_tool_repeat"),
        gates=raw.get("gates", []),
        requires_prior_work=raw.get("requires_prior_work", {}),
        dangerous_input_patterns=raw.get("dangerous_input_patterns", {}),
        require_user_consent_before=raw.get("require_user_consent_before", []),
        strict_consent_only=raw.get("strict_consent_only", False),
        tools=parse_tools_section(raw.get("tools")),
    )


def run_checks(trace: Trace, config: ActionConfig) -> ActionResult:
    """Run the 5 core checks against *trace* using *config*.

    Every check runs independently — failures don't short-circuit.
    """
    import functools

    from trajeval.assertions.core import (
        cost_within,
        must_visit,
        never_calls,
        no_retry_storm,
        require_user_consent_before,
        validate_tool_outputs,
    )

    checks: list[CheckResult] = []

    # 1. Retry storm
    try:
        no_retry_storm(trace, max_consecutive=config.max_retries)
        checks.append(CheckResult(
            name="retry_storm",
            passed=True,
            message=f"No tool retried >{config.max_retries}x consecutively",
        ))
    except AssertionError as exc:
        checks.append(CheckResult(
            name="retry_storm", passed=False, message=str(exc)
        ))

    # 2. Cost budget
    if config.cost_budget_usd is not None:
        try:
            cost_within(trace, p90=config.cost_budget_usd)
            checks.append(CheckResult(
                name="cost_budget",
                passed=True,
                message=(
                    f"${trace.total_cost_usd:.4f} within "
                    f"${config.cost_budget_usd:.2f} budget"
                ),
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="cost_budget", passed=False, message=str(exc)
            ))

    # 3. Banned tools
    for tool in config.banned_tools:
        try:
            never_calls(trace, tool=tool)
            checks.append(CheckResult(
                name=f"banned:{tool}",
                passed=True,
                message=f"'{tool}' never called",
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name=f"banned:{tool}",
                passed=False,
                message=str(exc),
            ))

    # 4. Required tools
    if config.required_tools:
        try:
            must_visit(trace, tools=config.required_tools)
            checks.append(CheckResult(
                name="required_tools",
                passed=True,
                message=f"All required tools called: {config.required_tools}",
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="required_tools",
                passed=False,
                message=str(exc),
            ))

    # 5. Schema validation
    if config.schemas:
        try:
            validate_tool_outputs(trace, schemas=config.schemas)
            checks.append(CheckResult(
                name="schema_validation",
                passed=True,
                message=(
                    f"All outputs match schemas for "
                    f"{sorted(config.schemas.keys())}"
                ),
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="schema_validation",
                passed=False,
                message=str(exc),
            ))

    # 6. NL contracts (compiled to LTL)
    if config.contracts:
        from trajeval.analysis.ltl import LTLRuntime
        from trajeval.analysis.ltl_compiler import (
            compile_contract,
            extract_ltl_formulas,
        )

        try:
            compiled = compile_contract(config.contracts)
            formulas = extract_ltl_formulas(compiled)
            if formulas:
                runtime = LTLRuntime(formulas)
                for node in trace.nodes:
                    runtime.advance(node)
                all_violations = runtime.violations + runtime.check_liveness()
                if all_violations:
                    checks.append(CheckResult(
                        name="contracts",
                        passed=False,
                        message="; ".join(all_violations),
                    ))
                else:
                    checks.append(CheckResult(
                        name="contracts",
                        passed=True,
                        message=(
                            f"{len(formulas)} contract(s) satisfied"
                        ),
                    ))
        except Exception as exc:
            checks.append(CheckResult(
                name="contracts",
                passed=False,
                message=f"Contract compilation error: {exc}",
            ))

    # 6b. Require user consent before writes (HITL-confirmation-text check)
    if config.require_user_consent_before:
        try:
            require_user_consent_before(
                trace,
                tools=config.require_user_consent_before,
                strict=config.strict_consent_only,
            )
            checks.append(CheckResult(
                name="user_consent",
                passed=True,
                message=(
                    f"All writes ({config.require_user_consent_before}) "
                    f"preceded by user consent"
                ),
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="user_consent",
                passed=False,
                message=str(exc),
            ))

    # 7. Intent extraction (informational — always passes)
    if config.show_intent:
        from trajeval.analysis.intent import extract_intent

        intent = extract_intent(trace)
        checks.append(CheckResult(
            name="intent",
            passed=True,
            message=(
                f"Intent: {intent.label} "
                f"(confidence={intent.confidence:.0%}, "
                f"source={intent.source})"
            ),
        ))

    # 8. Cascade root-cause (on failures only)
    if config.show_cascade:
        failed_checks = [c for c in checks if not c.passed]
        if failed_checks:
            from trajeval.analysis.cascade import find_root_cause

            rc = find_root_cause(
                trace,
                functools.partial(
                    no_retry_storm, max_consecutive=config.max_retries
                ),
            )
            if rc is not None:
                checks.append(CheckResult(
                    name="cascade",
                    passed=False,
                    message=(
                        f"Root cause: {rc.root_tool_name} "
                        f"at step {rc.step} — "
                        f"{rc.cascade_depth} nodes affected, "
                        f"${rc.wasted_cost_usd:.4f} wasted. "
                        f"{rc.explanation}"
                    ),
                ))

    # 9. Fault injection (structural resilience test)
    if config.fault_test:
        from trajeval.analysis.fault_injection import (
            FaultType,
            inject_all,
        )

        # Build assertion list from the checks that passed
        passing_assertions: list[
            tuple[str, Callable[[Trace], None]]
        ] = []
        passing_assertions.append((
            "retry_storm",
            functools.partial(
                no_retry_storm, max_consecutive=config.max_retries
            ),
        ))
        if config.banned_tools:
            from trajeval.assertions.core import never_calls

            for tool in config.banned_tools:
                passing_assertions.append((
                    f"banned:{tool}",
                    functools.partial(never_calls, tool=tool),
                ))

        report = inject_all(
            trace,
            assertions=passing_assertions,
            faults=[FaultType.ERROR, FaultType.TIMEOUT],
        )
        checks.append(CheckResult(
            name="fault_resilience",
            passed=report.resilience_rate >= 0.5,
            message=(
                f"Resilience: {report.resilience_rate:.0%} "
                f"({report.broken_count}/{report.total_scenarios} "
                f"scenarios broke under fault injection)"
            ),
        ))

    # 10. Stop on error
    if config.stop_on_error:
        from trajeval.assertions.core import stop_on_error as _stop_err

        try:
            _stop_err(trace)
            checks.append(CheckResult(
                name="stop_on_error",
                passed=True,
                message="Agent stops after error responses",
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="stop_on_error", passed=False, message=str(exc)
            ))

    # 11. PII detection
    if config.check_pii:
        from trajeval.assertions.core import no_pii_in_output

        try:
            no_pii_in_output(trace)
            checks.append(CheckResult(
                name="pii_check",
                passed=True,
                message="No PII patterns detected in tool outputs",
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="pii_check", passed=False, message=str(exc)
            ))

    # 12. Allowed tools only
    if config.allowed_tools:
        from trajeval.assertions.core import only_registered_tools

        try:
            only_registered_tools(
                trace, allowed_tools=config.allowed_tools
            )
            checks.append(CheckResult(
                name="allowed_tools",
                passed=True,
                message=(
                    f"All tools in allowed set: "
                    f"{config.allowed_tools}"
                ),
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="allowed_tools", passed=False, message=str(exc)
            ))

    # 13. Max tool repeat
    if config.max_tool_repeat is not None:
        from trajeval.assertions.core import no_tool_repeat

        tool_names = {
            n.tool_name
            for n in trace.nodes
            if n.node_type == "tool_call" and n.tool_name
        }
        repeat_violations = []
        for tool_name in tool_names:
            try:
                no_tool_repeat(
                    trace, tool_name,
                    max_calls=config.max_tool_repeat,
                )
            except AssertionError as exc:
                repeat_violations.append(str(exc))
        if repeat_violations:
            checks.append(CheckResult(
                name="tool_repeat",
                passed=False,
                message="; ".join(repeat_violations),
            ))
        else:
            checks.append(CheckResult(
                name="tool_repeat",
                passed=True,
                message=(
                    f"No tool called >{config.max_tool_repeat}x"
                ),
            ))

    # 14. Conditional gates
    if config.gates:
        from trajeval.assertions.core import conditional_block

        gate_violations: list[str] = []
        for gate in config.gates:
            try:
                conditional_block(
                    trace,
                    gate_tool=str(gate.get("tool", "")),
                    gate_key=str(gate.get("key", "")),
                    block_value=gate.get("block_value"),
                    blocked_tool=str(gate.get("blocked", "")),
                )
            except AssertionError as exc:
                gate_violations.append(str(exc))

        if gate_violations:
            checks.append(CheckResult(
                name="gates",
                passed=False,
                message="; ".join(gate_violations),
            ))
        else:
            checks.append(CheckResult(
                name="gates",
                passed=True,
                message=f"{len(config.gates)} gate(s) satisfied",
            ))

    # 15. Shortcut detection — completion tools must do work first.
    if config.requires_prior_work:
        from trajeval.assertions.core import requires_prior_work

        shortcut_violations: list[str] = []
        for tool_name, spec in config.requires_prior_work.items():
            raw_min = spec.get("min_distinct", 1)
            if isinstance(raw_min, bool):
                min_distinct = 1
            elif isinstance(raw_min, (int, str)):
                min_distinct = int(raw_min)
            else:
                min_distinct = 1
            required_raw = spec.get("required")
            required_tools = (
                [str(x) for x in required_raw]
                if isinstance(required_raw, list)
                else None
            )
            try:
                requires_prior_work(
                    trace,
                    completion_tool=tool_name,
                    min_prior_calls=min_distinct,
                    required_tools=required_tools,
                )
            except AssertionError as exc:
                shortcut_violations.append(str(exc))

        if shortcut_violations:
            checks.append(CheckResult(
                name="prior_work",
                passed=False,
                message="; ".join(shortcut_violations),
            ))
        else:
            checks.append(CheckResult(
                name="prior_work",
                passed=True,
                message=f"{len(config.requires_prior_work)} "
                f"completion tool(s) validated",
            ))

    # 16. Dangerous-input scanning — SQL injection, prompt-exfil
    # payloads, etc. Scans tool inputs (not outputs) for forbidden
    # regex patterns.
    if config.dangerous_input_patterns:
        from trajeval.assertions.core import no_dangerous_input

        try:
            no_dangerous_input(trace, config.dangerous_input_patterns)
            checks.append(CheckResult(
                name="dangerous_input",
                passed=True,
                message=(
                    f"{len(config.dangerous_input_patterns)} "
                    f"input-pattern rule(s) clean"
                ),
            ))
        except AssertionError as exc:
            checks.append(CheckResult(
                name="dangerous_input",
                passed=False,
                message=str(exc),
            ))

    # R4.2b — postcondition replay with typed symbolic state.
    # One CheckResult per (tool, kind) pair, matching the existing pattern.
    if config.tools:
        replay_state_checks(trace, config.tools, checks)

    return ActionResult(trace_id=trace.trace_id, checks=checks)


def replay_state_checks(
    trace: Trace,
    tools: dict[str, ToolContract],
    checks: list[CheckResult],
    *,
    starting_state: SymbolicState | None = None,
    skip_postcondition_node_ids: frozenset[str] = frozenset(),
) -> SymbolicState:
    """Replay *trace* through *tools*, threading SymbolicState top-to-bottom.

    Appends one CheckResult per (tool, precondition|postcondition_schema)
    pair encountered to *checks*. State mutations only commit when the
    postcondition schema passes — a bad ``tool_output`` cannot poison
    downstream state.

    Parameters
    ----------
    trace, tools, checks:
        See above.
    starting_state:
        Optional initial state. ``None`` means start fresh (offline replay).
        Guard mode threads the caller's state across calls via this kwarg.
    skip_postcondition_node_ids:
        Node IDs whose postcondition (schema + state mutation) is deferred
        — used by ``guard.check_pre`` because the proposed node has no
        ``tool_output`` yet.

    Returns
    -------
    The final :class:`SymbolicState` after replaying all nodes.
    """
    state = starting_state or SymbolicState()
    pre_violations: dict[str, list[str]] = {}
    post_violations: dict[str, list[str]] = {}
    pre_seen: set[str] = set()
    post_seen: set[str] = set()

    for node in trace.nodes:
        if node.node_type != "tool_call" or node.tool_name is None:
            continue
        tool_cfg = tools.get(node.tool_name)
        if tool_cfg is None:
            continue

        if tool_cfg.requires:
            pre_seen.add(node.tool_name)
            for pred in tool_cfg.requires:
                if not state.evaluate(pred):
                    pre_violations.setdefault(node.tool_name, []).append(
                        f"node={node.node_id} unsatisfied: {pred}"
                    )

        post = tool_cfg.postcondition
        if post is None:
            continue
        if node.node_id in skip_postcondition_node_ids:
            continue  # caller deferred postcondition for this node (PreToolUse)

        if post.returns:
            post_seen.add(node.tool_name)
            errs = validate_returns(node.tool_output, post.returns)
            if errs:
                post_violations.setdefault(node.tool_name, []).append(
                    f"node={node.node_id}: {'; '.join(errs)}"
                )
                continue  # do NOT apply state_updates on schema fail

        if post.state_updates:
            resolved = resolve_state_updates(post.state_updates, node.tool_output)
            state = state.apply(resolved)

    for tool_name in sorted(pre_seen):
        violations = pre_violations.get(tool_name, [])
        checks.append(
            CheckResult(
                name=f"precondition[{tool_name}]",
                passed=not violations,
                message="ok" if not violations else "; ".join(violations),
            )
        )
    for tool_name in sorted(post_seen):
        violations = post_violations.get(tool_name, [])
        checks.append(
            CheckResult(
                name=f"postcondition_schema[{tool_name}]",
                passed=not violations,
                message="ok" if not violations else "; ".join(violations),
            )
        )

    return state
