"""Assertion DSL for trajectory evaluation.

Each assertion receives a Trace and raises AssertionError with a clear message
on failure. All graph operations use NetworkX.

Severity levels
---------------
Every assertion is binary pass/fail, but not every failure is equally urgent.
A safety assertion firing (``never_calls``) is a P0 deploy-blocker; a latency
warning may be P2 (log, don't alert).

``ViolationError`` is an ``AssertionError`` subclass that carries a
:class:`Severity` tag.  Use the ``severity`` decorator to wrap any assertion
function and upgrade its plain ``AssertionError`` into a ``ViolationError``::

    from trajeval.assertions.core import severity, Severity, never_calls
    import functools

    # P0: block deploys if delete_user is called
    guarded = severity(
        functools.partial(never_calls, tool="delete_user"),
        level=Severity.P0,
        name="no-delete-user",
    )
    guarded(trace)   # raises ViolationError(severity=P0) on violation

All existing assertion functions continue to raise plain ``AssertionError``
unchanged — callers that do not use ``severity()`` see no behaviour change.
``ViolationError`` is a subclass of ``AssertionError`` so existing
``except AssertionError`` blocks still catch it.

Layer rule: pure Python, no FastAPI imports.
"""

from __future__ import annotations

import functools
import re
from collections import defaultdict
from collections.abc import Callable
from enum import IntEnum
from typing import Any

import networkx as nx
import numpy as np

from trajeval.analysis.graph import build_graph as _build_graph
from trajeval.sdk.models import Trace, TraceNode

# Public type alias: an assertion callable takes a Trace and either returns
# None (passed) or raises AssertionError (failed).
AssertionFn = Callable[[Trace], None]


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------


class Severity(IntEnum):
    """Violation severity level.

    P0 > P1 > P2 in urgency.  Comparisons work naturally::

        assert Severity.P0 > Severity.P1

    Typical mapping:
    - ``P0``: deploy blocker (safety rail violated, hard invariant broken)
    - ``P1``: alert / incident trigger (budget overrun, unexpected tool call)
    - ``P2``: warning / metric degradation (latency soft-limit, minor quality)
    """

    P0 = 0  # highest urgency
    P1 = 1
    P2 = 2  # lowest urgency


class ViolationError(AssertionError):
    """AssertionError subclass that carries a :class:`Severity` tag.

    Attributes
    ----------
    severity:
        Urgency level of this violation.
    assertion_name:
        Human-readable name of the assertion that fired.
    """

    def __init__(
        self,
        message: str,
        *,
        severity: Severity,
        assertion_name: str = "",
    ) -> None:
        super().__init__(message)
        self.severity = severity
        self.assertion_name = assertion_name

    def __str__(self) -> str:
        label = f"[{self.severity.name}]"
        name = f" ({self.assertion_name})" if self.assertion_name else ""
        return f"{label}{name} {super().__str__()}"


def severity(
    assertion_fn: AssertionFn,
    *,
    level: Severity,
    name: str = "",
) -> AssertionFn:
    """Wrap *assertion_fn* so its ``AssertionError`` is upgraded to ``ViolationError``.

    The wrapped function raises :class:`ViolationError` (with ``severity=level``)
    instead of plain ``AssertionError``.  Passes through silently on success.

    Parameters
    ----------
    assertion_fn:
        Any ``(Trace) -> None`` assertion callable.
    level:
        The severity to attach to violations.
    name:
        Human-readable label included in the error message.  Defaults to
        the function's ``__name__`` if available.

    Returns
    -------
    AssertionFn
        A wrapped callable with the same signature.

    Example
    -------
    ::

        import functools
        from trajeval.assertions.core import severity, Severity, never_calls

        safe_assertion = severity(
            functools.partial(never_calls, tool="drop_table"),
            level=Severity.P0,
            name="no-drop-table",
        )
        safe_assertion(trace)   # ViolationError(severity=P0) on failure
    """
    _name = name or getattr(assertion_fn, "__name__", "")

    @functools.wraps(assertion_fn)  # type: ignore[arg-type]
    def _wrapped(trace: Trace, *args: Any, **kwargs: Any) -> None:
        try:
            assertion_fn(trace, *args, **kwargs)  # type: ignore[call-arg]
        except ViolationError:
            raise  # already tagged — don't re-wrap
        except AssertionError as exc:
            raise ViolationError(
                str(exc), severity=level, assertion_name=_name
            ) from exc

    return _wrapped  # type: ignore[return-value]


def tool_must_precede(trace: Trace, tool: str, *, before: str) -> None:
    """Assert that every occurrence of *tool* call precedes every *before* call.

    Specifically: for every pair (t_tool, t_before) where t_tool is a node
    calling *tool* and t_before is a node calling *before*, there must be a
    directed path from t_tool → t_before in the trace graph.

    Raises AssertionError if any *before* node has no path from a *tool* node,
    or if *tool* never appears at all while *before* does.
    """
    tool_nodes = [n for n in trace.nodes if n.tool_name == tool]
    before_nodes = [n for n in trace.nodes if n.tool_name == before]

    if not before_nodes:
        return  # no `before` calls → constraint vacuously satisfied

    if not tool_nodes:
        raise AssertionError(
            f"tool_must_precede: '{tool}' was never called but '{before}' was "
            f"called {len(before_nodes)} time(s). "
            f"'{tool}' must precede '{before}'."
        )

    g = _build_graph(trace)

    for b_node in before_nodes:
        # There must exist at least one tool_node from which b_node is reachable.
        reachable_from_tool = any(
            nx.has_path(g, t_node.node_id, b_node.node_id) for t_node in tool_nodes
        )
        if not reachable_from_tool:
            raise AssertionError(
                f"tool_must_precede: node '{b_node.node_id}' (tool='{before}') "
                f"is not reachable from any '{tool}' node. "
                f"'{tool}' must precede '{before}'."
            )


def max_depth(trace: Trace, n: int) -> None:
    """Assert that no node in the trace exceeds depth *n*.

    Depth is the ``depth`` field on TraceNode — 0 is root level.

    Raises AssertionError listing every offending node.
    """
    if n < 0:
        raise ValueError(f"max_depth: n must be >= 0, got {n}")

    violations = [node for node in trace.nodes if node.depth > n]
    if violations:
        details = ", ".join(f"node '{v.node_id}' (depth={v.depth})" for v in violations)
        raise AssertionError(
            f"max_depth: {len(violations)} node(s) exceed max depth {n}: {details}"
        )


def no_cycles(trace: Trace) -> None:
    """Assert that the trace graph contains no directed cycles.

    A cycle in the execution graph indicates an infinite loop or retry storm
    that was not resolved.

    Raises AssertionError identifying one cycle that was found.
    """
    g = _build_graph(trace)
    try:
        cycle = nx.find_cycle(g, orientation="original")
    except nx.NetworkXNoCycle:
        return  # graph is acyclic — assertion passes

    edges = " → ".join(f"{u}→{v}" for u, v, *_ in cycle)
    raise AssertionError(
        f"no_cycles: directed cycle detected in trace '{trace.trace_id}': {edges}"
    )


def cost_within(trace: Trace, *, p90: float) -> None:
    """Assert that the trace's total cost does not exceed the *p90* budget.

    *p90* is the p90-level cost threshold for this agent (e.g. $1.50 means
    "this run should cost no more than the p90 of the reference distribution").
    The assertion checks ``total_cost_usd <= p90``.

    Raises AssertionError if the threshold is exceeded.
    """
    if p90 < 0:
        raise ValueError(f"cost_within: p90 must be >= 0, got {p90}")
    if trace.total_cost_usd > p90:
        raise AssertionError(
            f"cost_within: total cost {trace.total_cost_usd:.6f} USD "
            f"exceeds p90 budget {p90:.6f} USD "
            f"(trace_id='{trace.trace_id}')"
        )


def _tool_matches(tool_name: str | None, pattern: str) -> bool:
    """Check if *tool_name* matches *pattern*.

    Supports three modes:
    - Exact: ``"delete_user"`` matches only ``"delete_user"``
    - Wildcard: ``"*delete*"`` matches any tool containing ``"delete"``
    - Prefix: ``"admin_*"`` matches ``"admin_override"``, ``"admin_delete"``
    """
    if tool_name is None:
        return False
    if "*" not in pattern:
        return tool_name == pattern
    # Convert glob-style wildcards to simple containment
    parts = pattern.split("*")
    parts = [p for p in parts if p]
    if not parts:
        return True  # pattern is just "*"
    # All non-empty parts must appear in order
    pos = 0
    for part in parts:
        idx = tool_name.find(part, pos)
        if idx == -1:
            return False
        pos = idx + len(part)
    # Check anchoring
    if not pattern.startswith("*") and not tool_name.startswith(parts[0]):
        return False
    return pattern.endswith("*") or tool_name.endswith(parts[-1])


def never_calls(trace: Trace, tool: str) -> None:
    """Assert that *tool* is never invoked anywhere in the trace.

    Supports wildcard patterns: ``"*delete*"`` matches any tool
    containing "delete". ``"admin_*"`` matches any tool starting
    with "admin_". Exact match when no wildcards.

    Raises AssertionError listing every offending node.
    """
    violations = [
        n for n in trace.nodes if _tool_matches(n.tool_name, tool)
    ]
    if violations:
        ids = ", ".join(f"'{v.node_id}'" for v in violations)
        raise AssertionError(
            f"never_calls: tool pattern '{tool}' matched "
            f"{len(violations)} time(s). "
            f"Offending nodes: {ids}"
        )


def must_visit(trace: Trace, tools: list[str]) -> None:
    """Assert that every tool in *tools* is called at least once.

    Use this to enforce required steps in an agent workflow
    (e.g. validation must always precede booking).

    Raises AssertionError listing every tool that was skipped.
    """
    called = {n.tool_name for n in trace.nodes if n.tool_name is not None}
    missing = [t for t in tools if t not in called]
    if missing:
        raise AssertionError(
            f"must_visit: required tool(s) were never called: "
            f"{missing}. Called tools: {sorted(called)}"
        )


def tool_call_count(trace: Trace, tool: str, *, max: int) -> None:  # noqa: A002
    """Assert that *tool* is called no more than *max* times.

    Use this to enforce per-tool call budgets
    (e.g. LLM calls should not exceed 10 per run).

    Raises AssertionError with the actual count if exceeded.
    """
    if max < 0:
        raise ValueError(f"tool_call_count: max must be >= 0, got {max}")
    count = sum(1 for n in trace.nodes if n.tool_name == tool)
    if count > max:
        raise AssertionError(
            f"tool_call_count: tool '{tool}' was called {count} time(s), "
            f"exceeding the budget of {max} call(s)"
        )


def latency_within(trace: Trace, *, p95: int) -> None:
    """Assert that the p95 node duration does not exceed *p95* milliseconds.

    Computes the 95th-percentile of ``duration_ms`` across all nodes.
    An empty trace passes vacuously.

    Raises AssertionError if the p95 latency exceeds the threshold.
    """
    if p95 < 0:
        raise ValueError(f"latency_within: p95 must be >= 0, got {p95}")
    durations = [n.duration_ms for n in trace.nodes]
    if not durations:
        return
    p95_actual = int(np.percentile(durations, 95))
    if p95_actual > p95:
        raise AssertionError(
            f"latency_within: p95 node latency {p95_actual}ms "
            f"exceeds budget {p95}ms "
            f"(trace_id='{trace.trace_id}')"
        )


def total_tool_calls(trace: Trace, *, max: int) -> None:  # noqa: A002
    """Assert that the total number of tool_call nodes does not exceed *max*.

    Use this to enforce a per-run call budget across all tools combined.
    For per-tool limits use :func:`tool_call_count`.

    Raises AssertionError with the actual count if the limit is exceeded.
    """
    if max < 0:
        raise ValueError(f"total_tool_calls: max must be >= 0, got {max}")
    count = sum(1 for n in trace.nodes if n.node_type == "tool_call")
    if count > max:
        raise AssertionError(
            f"total_tool_calls: {count} tool call(s) exceed budget of {max} "
            f"(trace_id='{trace.trace_id}')"
        )


def tool_output_satisfies(
    trace: Trace,
    tool: str,
    *,
    key: str,
    predicate: Callable[[object], bool],
    description: str = "",
) -> None:
    """Assert that every call to *tool* has ``tool_output[key]`` satisfying *predicate*.

    Use this to enforce output content contracts — e.g. "search_flights must
    always return at least one result" or "book_flight must return a
    confirmation number".

    Nodes where *key* is absent from ``tool_output`` are treated as violations
    (the contract requires the key to be present and satisfy the predicate).

    Parameters
    ----------
    tool:
        Name of the tool whose output is being checked.
    key:
        Key to look up inside ``tool_output``.
    predicate:
        Callable ``(value) -> bool``.  Called with ``tool_output[key]`` for
        every matching node.  Should return ``True`` for values that satisfy
        the contract.
    description:
        Optional human-readable description of the predicate for error messages.
        E.g. ``"must be a non-empty list"``.

    Raises AssertionError listing every node where the predicate failed.
    """
    violations: list[str] = []
    for node in trace.nodes:
        if node.tool_name != tool:
            continue
        if key not in node.tool_output:
            violations.append(
                f"node '{node.node_id}': key '{key}' absent from tool_output "
                f"(got keys: {sorted(node.tool_output.keys())})"
            )
        else:
            value = node.tool_output[key]
            try:
                ok = predicate(value)
            except Exception as exc:
                violations.append(
                    f"node '{node.node_id}': predicate raised "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if not ok:
                label = f" ({description})" if description else ""
                violations.append(
                    f"node '{node.node_id}': tool_output['{key}'] = {value!r} "
                    f"did not satisfy predicate{label}"
                )
    if violations:
        detail = "; ".join(violations)
        raise AssertionError(
            f"tool_output_satisfies: tool '{tool}', key='{key}': {detail}"
        )


def tool_output_schema(trace: Trace, tool: str, *, required_keys: list[str]) -> None:
    """Assert that every call to *tool* has all *required_keys* in its output.

    Shorthand for a common ``tool_output_satisfies`` pattern: verifying that a
    tool returns a predictable schema.  Use this to catch silent API changes
    where a tool starts omitting fields your agent depends on.

    Raises AssertionError listing every node and every missing key.
    """
    if not required_keys:
        return
    violations: list[str] = []
    for node in trace.nodes:
        if node.tool_name != tool:
            continue
        missing = [k for k in required_keys if k not in node.tool_output]
        if missing:
            violations.append(
                f"node '{node.node_id}': missing keys {missing} "
                f"(got: {sorted(node.tool_output.keys())})"
            )
    if violations:
        detail = "; ".join(violations)
        raise AssertionError(
            f"tool_output_schema: tool '{tool}' output missing required keys. {detail}"
        )


def no_retry_storm(trace: Trace, *, max_consecutive: int = 3) -> None:
    """Assert no tool is called more than *max_consecutive* times in a row
    with identical arguments.

    The #1 empirical failure pattern in production agents: a tool call fails
    and the agent retries the identical malformed call up to 19 times without
    recovery ("How Do LLMs Fail", Dec 2025, 900 traces across 3 frontier
    models).

    Two consecutive calls are "identical" if they share the same
    ``tool_name`` **and** the same ``tool_input`` (compared by sorted-key
    JSON serialisation).  Tool output is deliberately ignored — a retry
    storm is defined by the agent sending the same request, regardless of
    whether the response differs.

    Parameters
    ----------
    max_consecutive:
        Maximum allowed consecutive identical calls.  Default 3.  Set to 1
        to forbid *any* immediate retry of the same call.

    Raises AssertionError listing the tool, the run length, and the node
    range where the storm was detected.
    """
    if max_consecutive < 1:
        raise ValueError(
            f"no_retry_storm: max_consecutive must be >= 1, "
            f"got {max_consecutive}"
        )

    tool_nodes = [n for n in trace.nodes if n.node_type == "tool_call"]
    if len(tool_nodes) < 2:
        return

    import json

    def _sig(node: object) -> str:
        n = node  # type: ignore[assignment]
        try:
            inp = json.dumps(n.tool_input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            inp = repr(n.tool_input)
        return f"{n.tool_name}::{inp}"

    violations: list[str] = []
    run_start = 0
    for i in range(1, len(tool_nodes)):
        if _sig(tool_nodes[i]) == _sig(tool_nodes[run_start]):
            run_len = i - run_start + 1
            if run_len > max_consecutive:
                start_id = tool_nodes[run_start].node_id
                end_id = tool_nodes[i].node_id
                violations.append(
                    f"tool '{tool_nodes[i].tool_name}' called "
                    f"{run_len} consecutive times with identical "
                    f"args (nodes {start_id}..{end_id})"
                )
        else:
            run_start = i

    if violations:
        detail = "; ".join(violations)
        raise AssertionError(
            f"no_retry_storm: {len(violations)} retry storm(s) "
            f"detected (max_consecutive={max_consecutive}): {detail}"
        )


def no_tool_repeat(
    trace: Trace,
    tool: str,
    *,
    max_calls: int = 3,
) -> None:
    """Assert that *tool* is not called more than *max_calls* times total,
    regardless of argument variation.

    Catches the varied-arg retry pattern: agent retries the same tool with
    slightly different args each time to avoid identical-input detection.
    ``no_retry_storm`` misses this because the inputs differ.

    Use this alongside ``no_retry_storm`` for defense in depth.
    """
    count = sum(
        1 for n in trace.nodes
        if _tool_matches(n.tool_name, tool)
    )
    if count > max_calls:
        raise AssertionError(
            f"no_tool_repeat: tool '{tool}' called {count} times, "
            f"exceeding limit of {max_calls}"
        )


_DEFAULT_ERROR_KEYS = ("error", "error_code", "fault")

_ERROR_MARKER_RE = re.compile(
    r'"(?:error|error_code|fault)"\s*:\s*'
    r'(?!null\b|false\b|0[,\s}]|""|\[\]|\{\})\S'
    r'|"success"\s*:\s*false'
    r'|"status"\s*:\s*"(?:error|failed|failure|fault)"'
    r'|status_code["\']?\s*[:=]\s*[45]\d{2}'
    r'|Traceback \(most recent call last\)'
    r'|\bHTTP/[0-9.]+\s+[45]\d{2}\b',
    re.IGNORECASE,
)


_BAD_STATUS_VALUES = frozenset({"error", "failed", "failure", "fault"})


def tool_output_has_error(
    tool_output: object,
    error_keys: tuple[str, ...] = _DEFAULT_ERROR_KEYS,
) -> bool:
    """Return True if a tool output carries a real error signal.

    Walks the output structure:
      * Dict: structural checks (``error``-like key with truthy value,
        ``status`` set to error/failed/failure/fault, ``success``
        explicitly False, ``status_code`` in 4xx/5xx range). Then
        recurses into all values.
      * List: recurses into every element.
      * String (leaf): scans for error markers (``"error": <truthy>``,
        ``"success": false``, ``"status": "failed"``, HTTP 4xx/5xx,
        Python traceback).

    Handles the real external payloads that silently passed before
    this change: ``{"raw": "Error: ... status_code: 500 ..."}`` and
    ``{"result": "{\\"error\\":\\"...\\"}"}``. Does not fire on
    ``{"error": null}``, ``{"errors": 0}``, ``{"success": true}``,
    ``{"status": "ok"}``, or prose mentioning "no errors".
    """
    if isinstance(tool_output, dict):
        for k, v in tool_output.items():
            if k in error_keys and v:
                return True
            if (
                k == "status"
                and isinstance(v, str)
                and v.lower() in _BAD_STATUS_VALUES
            ):
                return True
            if k == "success" and v is False:
                return True
            if (
                k == "status_code"
                and isinstance(v, int)
                and 400 <= v < 600
            ):
                return True
            if isinstance(v, (dict, list, str)) and tool_output_has_error(
                v, error_keys
            ):
                return True
        return False
    if isinstance(tool_output, list):
        return any(
            tool_output_has_error(v, error_keys) for v in tool_output
        )
    if isinstance(tool_output, str):
        return bool(_ERROR_MARKER_RE.search(tool_output))
    return False


def stop_on_error(
    trace: Trace,
    *,
    error_keys: list[str] | None = None,
    max_after_error: int = 0,
) -> None:
    """Assert that the agent handles tool errors before moving on.

    Fires when a tool returns an error and the agent calls a
    *different* tool before the failing tool recovers. A successful
    retry of the same tool (possibly with corrected arguments) counts
    as recovery and does not fire. Persistent same-tool retries are
    also allowed — they are a valid agent pattern.

    Parameters
    ----------
    error_keys:
        Keys in ``tool_output`` that indicate an error. Defaults to
        ``("error", "error_code", "fault")``.
    max_after_error:
        Slack for calls to other tools after an unrecovered error
        before this raises. Default 0 — the first different-tool call
        after an unrecovered error fires.
    """
    _keys = tuple(error_keys) if error_keys else _DEFAULT_ERROR_KEYS
    tool_nodes = [n for n in trace.nodes if n.node_type == "tool_call"]

    for i, node in enumerate(tool_nodes):
        if not tool_output_has_error(node.tool_output, _keys):
            continue
        non_retry_count = 0
        for follow in tool_nodes[i + 1:]:
            if follow.tool_name == node.tool_name:
                if not tool_output_has_error(follow.tool_output, _keys):
                    break  # recovered via successful retry
                continue  # same tool still erroring — still a retry attempt
            non_retry_count += 1
            if non_retry_count > max_after_error:
                raise AssertionError(
                    f"stop_on_error: agent called '{follow.tool_name}' "
                    f"after unrecovered error at node '{node.node_id}' "
                    f"(tool '{node.tool_name}'). Retry the failing tool "
                    f"or stop."
                )


def only_registered_tools(
    trace: Trace,
    *,
    allowed_tools: list[str],
) -> None:
    """Assert that only tools in *allowed_tools* were called.

    Catches hallucinated tool names — when the agent invents a tool
    that doesn't exist and calls it anyway.
    """
    allowed_set = set(allowed_tools)
    violations = [
        n for n in trace.nodes
        if n.node_type == "tool_call"
        and n.tool_name is not None
        and n.tool_name not in allowed_set
    ]
    if violations:
        unknown = {v.tool_name for v in violations}
        raise AssertionError(
            f"only_registered_tools: unregistered tool(s) called: "
            f"{sorted(unknown)}. "
            f"Allowed: {sorted(allowed_set)}"
        )


_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_CC_RE = re.compile(r"\b(\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}|\d{16})\b")

_PHONE_RE = re.compile(
    r"(?<!\d)("
    r"\+\d{1,2}[-. ]?\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}"
    r"|\(\d{3}\)[-. ]?\d{3}[-. ]?\d{4}"
    r"|\d{3}-\d{3}-\d{4}"
    r"|\d{3}\.\d{3}\.\d{4}"
    r")(?!\d)"
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def scan_for_pii(text: str) -> list[str]:
    """Return labels of PII patterns found in ``text`` (empty if none).

    Uses Luhn validation for credit-card candidates and shape-anchored
    phone detection (parens, separators, or explicit country code) to
    avoid matching Unix timestamps, URL IDs, and arbitrary digit runs.
    """
    found: list[str] = []
    if _SSN_RE.search(text):
        found.append("SSN")
    for m in _CC_RE.finditer(text):
        digits = re.sub(r"[- ]", "", m.group(1))
        if _luhn_ok(digits):
            found.append("credit_card")
            break
    if _PHONE_RE.search(text):
        found.append("phone")
    return found


def no_pii_in_output(trace: Trace) -> None:
    """Assert that no tool output contains obvious PII patterns.

    Checks for: SSN (XXX-XX-XXXX), Luhn-valid credit-card numbers,
    and formatted phone numbers. This is a heuristic — not a
    substitute for a proper DLP system, but catches common leaks
    without firing on every 10-digit timestamp.
    """
    violations: list[str] = []
    for node in trace.nodes:
        if node.node_type != "tool_call":
            continue
        output_str = str(node.tool_output)
        for label in scan_for_pii(output_str):
            violations.append(
                f"node '{node.node_id}' ({node.tool_name}): "
                f"{label} pattern detected"
            )

    if violations:
        detail = "; ".join(violations)
        raise AssertionError(
            f"no_pii_in_output: PII detected in tool outputs: {detail}"
        )


def conditional_block(
    trace: Trace,
    *,
    gate_tool: str,
    gate_key: str,
    block_value: object,
    blocked_tool: str,
) -> None:
    """Assert that *blocked_tool* is never called after *gate_tool*
    returns *block_value* for *gate_key*.

    Example: if ``wait_for_approval`` returns ``{"approved": false}``,
    then ``execute_transfer`` must not be called afterward.

    This is the output-conditional assertion that ``tool_must_precede``
    misses — it checks not just ordering but the result of the
    preceding tool.
    """
    gate_blocked = False
    for node in trace.nodes:
        if node.node_type != "tool_call":
            continue
        if node.tool_name == gate_tool:
            val = node.tool_output.get(gate_key)
            if val == block_value:
                gate_blocked = True
        elif node.tool_name == blocked_tool and gate_blocked:
            raise AssertionError(
                f"conditional_block: '{blocked_tool}' called after "
                f"'{gate_tool}' returned {gate_key}={block_value!r}. "
                f"Node: '{node.node_id}'"
            )


def final_output_contains(trace: Trace, *, key: str) -> None:
    """Assert that the last tool_call node's output contains *key*.

    "Final output" is the ``tool_output`` of the last ``tool_call`` node in
    the trace (by position, not by timestamp).  Use this to verify that the
    agent's terminal step produced the expected field — e.g. a
    ``confirmation_number``, ``booking_id``, or ``result``.

    An empty trace (no tool_call nodes) passes vacuously.

    Raises AssertionError if the last tool_call node's output does not contain *key*.
    """
    tool_call_nodes = [n for n in trace.nodes if n.node_type == "tool_call"]
    if not tool_call_nodes:
        return
    last = tool_call_nodes[-1]
    if key not in last.tool_output:
        raise AssertionError(
            f"final_output_contains: last tool_call node '{last.node_id}' "
            f"(tool='{last.tool_name}') does not contain key '{key}'. "
            f"Got keys: {sorted(last.tool_output.keys())}"
        )


def no_duplicate_task(
    trace: Trace,
    tool: str,
    *,
    task_key: str,
) -> None:
    """Assert that *tool* is never called with the same *task_key* value
    more than once — even if other arguments differ.

    Catches multi-agent coordination failures where the same logical task
    is assigned to multiple agents independently (different agent names,
    same task description). This is the semantic-level duplicate detector
    that ``no_retry_storm`` misses: retry storm checks full input
    equality, this checks one key.

    Parameters
    ----------
    tool:
        The tool name to check (e.g. ``"assign_task"``).
    task_key:
        The key in ``tool_input`` that identifies the logical task
        (e.g. ``"task"``). Nodes where this key is absent are skipped.

    Raises AssertionError listing every duplicated task value and the
    nodes that share it.
    """
    seen: dict[object, list[str]] = defaultdict(list)
    for node in trace.nodes:
        if node.tool_name != tool:
            continue
        val = node.tool_input.get(task_key)
        if val is not None:
            seen[val].append(node.node_id)

    duplicates = {v: ids for v, ids in seen.items() if len(ids) > 1}
    if duplicates:
        details = "; ".join(
            f"{task_key}={val!r} assigned {len(ids)} times "
            f"at nodes {ids}"
            for val, ids in duplicates.items()
        )
        raise AssertionError(
            f"no_duplicate_task: tool '{tool}' assigned the same "
            f"task multiple times: {details}"
        )


def validate_tool_outputs(
    trace: Trace,
    schemas: dict[str, dict[str, object]],
) -> None:
    """Assert that every tool_call node's output conforms to its JSON schema.

    Parameters
    ----------
    schemas:
        Mapping of ``{tool_name: json_schema_dict}``.  Tools not in the map
        are skipped (no schema = no constraint).  The schema subset supported
        is: ``type``, ``required``, ``properties`` (recursive), ``items``
        (for arrays).  This covers ~95% of real-world tool output contracts
        without requiring the ``jsonschema`` dependency.

    Raises AssertionError listing every node + every violation found.
    """
    violations: list[str] = []
    for node in trace.nodes:
        if node.node_type != "tool_call" or node.tool_name not in schemas:
            continue
        schema = schemas[node.tool_name]
        errors = _validate_value(node.tool_output, schema, path="")
        for err in errors:
            violations.append(f"node '{node.node_id}' ({node.tool_name}): {err}")
    if violations:
        detail = "; ".join(violations)
        raise AssertionError(
            f"validate_tool_outputs: {len(violations)} schema violation(s): "
            f"{detail}"
        )


def _validate_value(
    value: object,
    schema: dict[str, object],
    path: str,
) -> list[str]:
    """Validate *value* against a JSON Schema subset. Returns error strings."""
    errors: list[str] = []
    prefix = f"at '{path}'" if path else "at root"

    # Type check
    expected_type = schema.get("type")
    if expected_type is not None:
        type_map: dict[str, tuple[type, ...]] = {
            "object": (dict,),
            "array": (list,),
            "string": (str,),
            "number": (int, float),
            "integer": (int,),
            "boolean": (bool,),
            "null": (type(None),),
        }
        allowed = type_map.get(str(expected_type))
        if allowed is not None and not isinstance(value, allowed):
            actual = type(value).__name__
            errors.append(
                f"{prefix}: expected type '{expected_type}', got '{actual}'"
            )
            return errors  # type mismatch — skip deeper checks

    # Required keys (only for objects)
    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    errors.append(f"{prefix}: missing required key '{key}'")

        min_props = schema.get("minProperties")
        if isinstance(min_props, int) and len(value) < min_props:
            errors.append(
                f"{prefix}: object has {len(value)} properties, "
                f"minProperties is {min_props} "
                f"(lazy-agent signal: trivial output)"
            )
        max_props = schema.get("maxProperties")
        if isinstance(max_props, int) and len(value) > max_props:
            errors.append(
                f"{prefix}: object has {len(value)} properties, "
                f"maxProperties is {max_props}"
            )

        # Recurse into properties
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for prop_name, prop_schema in properties.items():
                if prop_name in value and isinstance(prop_schema, dict):
                    child_path = f"{path}.{prop_name}" if path else prop_name
                    errors.extend(
                        _validate_value(value[prop_name], prop_schema, child_path)
                    )

    # Array items and length
    if isinstance(value, list):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(
                f"{prefix}: array has {len(value)} items, "
                f"minItems is {min_items} "
                f"(lazy-agent signal: empty results)"
            )
        max_items = schema.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(
                f"{prefix}: array has {len(value)} items, "
                f"maxItems is {max_items}"
            )
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                child_path = f"{path}[{i}]" if path else f"[{i}]"
                errors.extend(_validate_value(item, items_schema, child_path))

    # String length
    if isinstance(value, str):
        min_len = schema.get("minLength")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append(
                f"{prefix}: string has {len(value)} chars, "
                f"minLength is {min_len} "
                f"(lazy-agent signal: trivial output)"
            )
        max_len = schema.get("maxLength")
        if isinstance(max_len, int) and len(value) > max_len:
            errors.append(
                f"{prefix}: string has {len(value)} chars, "
                f"maxLength is {max_len}"
            )

    return errors


def no_dangerous_input(
    trace: Trace,
    patterns: dict[str, list[str]],
) -> None:
    """Assert that no tool was called with a forbidden input pattern.

    Scans every tool's ``tool_input`` (stringified) against the regex
    patterns listed for that tool. Case-insensitive.

    Catches agents that compile user-supplied natural language into
    structured inputs carrying a dangerous payload — the canonical
    case is SQL injection in an analytics agent, where the user's
    question contains ``DROP TABLE`` and the agent emits a compound
    SQL statement.

    Parameters
    ----------
    patterns:
        Mapping of ``{tool_name: [regex_patterns]}``. Tools not in
        the mapping are not scanned.

    Raises AssertionError listing every (node, tool, matched_pattern).
    """
    violations: list[str] = []
    for node in trace.nodes:
        if (
            node.node_type != "tool_call"
            or node.tool_name is None
            or node.tool_name not in patterns
        ):
            continue
        input_str = str(node.tool_input)
        for pat in patterns[node.tool_name]:
            if re.search(pat, input_str, re.IGNORECASE):
                violations.append(
                    f"node '{node.node_id}' ({node.tool_name}): "
                    f"input matches forbidden pattern '{pat}'"
                )
    if violations:
        raise AssertionError(
            f"no_dangerous_input: {len(violations)} violation(s): "
            f"{'; '.join(violations)}"
        )


def get_ancestors(trace: Trace, node: TraceNode) -> list[TraceNode]:
    """Return the topological predecessors of *node* in *trace*.

    Preference order:
      1. If ``trace.edges`` is non-empty, walk the reverse adjacency
         graph (BFS) from ``node.node_id`` and return every reachable
         predecessor. This is the topology-true answer for any trace
         that carries explicit DAG edges — multi-agent fork/join,
         parallel subgraph calls, conditional branches.
      2. Else if any node has ``parent_node_id`` set, chain up parent
         links to produce a linearized ancestor set. This handles the
         subagent/tool-call-nested case where edges were not emitted.
      3. Else fall back to timestamp order: every node strictly before
         ``node.timestamp``.

    The result excludes ``node`` itself. Sorted by timestamp for
    deterministic consumer iteration.
    """
    # Mode 1: DAG via edges
    if trace.edges:
        reverse: dict[str, list[str]] = {}
        for edge in trace.edges:
            reverse.setdefault(edge.target, []).append(edge.source)
        by_id = {n.node_id: n for n in trace.nodes}
        seen: set[str] = set()
        queue = list(reverse.get(node.node_id, []))
        while queue:
            nid = queue.pop()
            if nid in seen:
                continue
            seen.add(nid)
            queue.extend(reverse.get(nid, []))
        # Exclude the node itself from its ancestor set (docstring promise).
        # Self-edges can reintroduce it via the reverse-adjacency walk.
        seen.discard(node.node_id)
        ancestors = [by_id[nid] for nid in seen if nid in by_id]
        return sorted(ancestors, key=lambda n: n.timestamp)

    # Mode 2: parent_node_id chain
    has_parents = any(n.parent_node_id for n in trace.nodes)
    if has_parents:
        by_id = {n.node_id: n for n in trace.nodes}
        ancestors = []
        current = by_id.get(node.parent_node_id or "")
        while current is not None:
            ancestors.append(current)
            current = by_id.get(current.parent_node_id or "")
        return sorted(ancestors, key=lambda n: n.timestamp)

    # Mode 3: timestamp fallback (linear-trace default)
    return sorted(
        (n for n in trace.nodes if n.timestamp < node.timestamp),
        key=lambda n: n.timestamp,
    )


def requires_prior_work(
    trace: Trace,
    *,
    completion_tool: str,
    min_prior_calls: int = 1,
    required_tools: list[str] | None = None,
) -> None:
    """Assert that a completion-style tool call was preceded by real work.

    Catches the "brilliant intern" shortcut: an agent calls a completion
    tool (``finish``, ``submit``, ``confirm``, ``mark_complete``) before
    actually doing the work it was asked to do. Today we have
    ``tool_must_precede`` for a single before-after pair; this assertion
    is coarser and more honest to the pattern.

    **Topology-aware.** "Prior" means *ancestor in the trace DAG* when
    the trace carries explicit edges or parent links. For linear traces
    (no edges, no parents) it means *strictly earlier by timestamp*.
    See :func:`get_ancestors` for the precedence order. This matters
    for multi-agent / fork-join traces where timestamp-only filtering
    misidentifies siblings as predecessors.

    Parameters
    ----------
    completion_tool:
        Tool name that signals the agent thinks it is done (supports
        wildcards via ``_tool_matches``).
    min_prior_calls:
        Minimum number of *distinct* prior tool calls (by tool name, not
        by node) required before ``completion_tool`` is invoked. Default
        1 — even one call is a low bar; real tasks usually warrant more.
    required_tools:
        Optional whitelist of tool names that count toward the prior-work
        total. If ``None``, any tool call counts. Pass a list like
        ``["search", "analyze"]`` to require that both appear before
        completion.

    Raises AssertionError listing the offending completion node and what
    was (or wasn't) called before it.
    """
    for node in trace.nodes:
        if node.node_type != "tool_call":
            continue
        if node.tool_name is None:
            continue
        if not _tool_matches(node.tool_name, completion_tool):
            continue

        prior_calls = [
            n for n in get_ancestors(trace, node)
            if n.node_type == "tool_call" and n.tool_name is not None
        ]
        prior_names = [n.tool_name for n in prior_calls]

        if required_tools is not None:
            missing = [t for t in required_tools if t not in prior_names]
            if missing:
                raise AssertionError(
                    f"requires_prior_work: '{node.tool_name}' called at "
                    f"node '{node.node_id}' but required prior tools were "
                    f"never called: {missing}. Called before: {prior_names}"
                )

        distinct = len(set(prior_names))
        if distinct < min_prior_calls:
            raise AssertionError(
                f"requires_prior_work: '{node.tool_name}' called at node "
                f"'{node.node_id}' after only {distinct} distinct prior "
                f"tool(s) ({prior_names}), minimum is {min_prior_calls}. "
                f"Looks like a premature-completion shortcut."
            )


_CONSENT_PATTERNS: tuple[str, ...] = (
    # Explicit affirmatives
    "yes",
    "yeah",
    "yep",
    "sure",
    "absolutely",
    "agreed",
    # Commands to act
    "proceed",
    "go ahead",
    "please do",
    "please proceed",
    "please process",
    "do it",
    "go for it",
    # Confirmations
    "confirm",
    "confirmed",
    "okay",
    "ok",
    "sounds good",
    "that works",
    "works for me",
    "correct",
    # Authorizations (KYC / financial tone)
    "approve",
    "approved",
    "authorize",
    "authorise",
    "authorized",
    "authorised",
)

# Narrower set for policies that require explicit affirmative or direct
# command, excluding permissive ambient phrasings ("sure", "ok", "absolutely",
# "agreed", "that works", "fine", "correct", "right"). Matches the strict
# consent tier used by the τ-bench rigorous_audit.py classifier. Activated by
# ``strict=True`` or the ``strict_consent_only: true`` YAML flag. Derivation:
# 2026-04-25 three-judge cross-reference showed 55/56 triple-agreement
# LLM-flagged violations cite missing explicit-affirmative consent (see
# benchmarks/results/three_judge_unique_overlap_2026-04-25.md); the strict
# tier matches the audit's policy-faithful reading, not the stricter
# individual-LLM reading.
_STRICT_CONSENT_PATTERNS: tuple[str, ...] = (
    # Explicit affirmatives
    "yes",
    "yeah",
    "yep",
    # Explicit confirmations / authorizations
    "confirm",
    "confirmed",
    "approve",
    "approved",
    "authorize",
    "authorise",
    "authorized",
    "authorised",
    # Direct commands to execute
    "proceed",
    "go ahead",
    "please do",
    "please proceed",
    "please process",
    "do it",
    "go for it",
    "sounds good",
    "works for me",
)


def _consent_regex(patterns: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a case-insensitive, word-boundary regex that matches any of
    *patterns* as a whole word/phrase. Multi-word patterns are supported.

    Using word boundaries prevents false positives like "book" matching "ok"
    or "look" matching "ok".
    """
    # Escape each pattern and normalize internal whitespace; allow any
    # whitespace between words (so "go ahead" matches "go  ahead" too).
    escaped = [re.escape(p).replace(r"\ ", r"\s+") for p in patterns]
    # \b is a word boundary. For phrases with spaces, the inner spaces are
    # already handled by the regex above; outer \b anchors start/end.
    return re.compile(
        r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE
    )


_DEFAULT_CONSENT_REGEX: re.Pattern[str] = _consent_regex(_CONSENT_PATTERNS)
_STRICT_CONSENT_REGEX: re.Pattern[str] = _consent_regex(_STRICT_CONSENT_PATTERNS)


def require_user_consent_before(
    trace: Trace,
    *,
    tools: list[str],
    consent_patterns: list[str] | None = None,
    strict: bool = False,
) -> None:
    """Assert that for every call to a tool in *tools*, the preceding user
    message (captured by the adapter as ``metadata["preceding_user_text"]``)
    contains at least one consent phrase as a whole word or phrase.

    The adapter layer is responsible for populating ``preceding_user_text`` on
    each ``tool_call`` TraceNode's ``metadata``.  Nodes that lack that
    metadata key are treated as failures (conservative default — no evidence
    of consent). Case-insensitive word-boundary match — "book" does NOT match
    "ok", "look" does NOT match "ok".

    Parameters
    ----------
    trace:
        The TrajEval ``Trace`` to inspect.
    tools:
        The tools that require explicit user consent in the preceding user
        turn — e.g. ``["book_reservation", "cancel_reservation"]``.
    consent_patterns:
        Optional override of the default consent phrase list
        (:data:`_CONSENT_PATTERNS`). Useful for non-English or domain-specific
        confirmation vocabulary. Explicit override wins over ``strict``.
    strict:
        When True (and ``consent_patterns`` is not provided), use the narrower
        :data:`_STRICT_CONSENT_PATTERNS` set (yes / yeah / yep / confirm /
        confirmed) instead of the default 27-entry list. Appropriate for
        policies that specify literal (yes) confirmation, e.g. τ-bench
        ("obtain explicit user confirmation (yes)"). Rejects lenient
        phrasings like "sure", "proceed", "okay".

    Raises
    ------
    AssertionError
        If any tool_call in *tools* fires without preceding consent. Message
        lists every offending node_id + tool_name + preceding_user_text
        prefix.
    """
    if consent_patterns is not None:
        regex = _consent_regex(tuple(consent_patterns))
    elif strict:
        regex = _STRICT_CONSENT_REGEX
    else:
        regex = _DEFAULT_CONSENT_REGEX
    targets = set(tools)
    violations: list[str] = []
    for node in trace.nodes:
        if node.node_type != "tool_call":
            continue
        if node.tool_name not in targets:
            continue
        text = str(node.metadata.get("preceding_user_text", ""))
        if not regex.search(text):
            prefix = text[:80].replace("\n", " ") if text else "<no prior user message>"
            violations.append(
                f"tool='{node.tool_name}' node='{node.node_id}' "
                f"preceding_user={prefix!r}"
            )
    if violations:
        raise AssertionError(
            "require_user_consent_before: "
            f"{len(violations)} write(s) without preceding user consent: "
            + "; ".join(violations)
        )


def no_duplicate_arg_call(trace: Trace, tool: str, *, arg_key: str) -> None:
    """Assert that *tool* is never called more than once with the same value
    for *arg_key* in its tool_input.

    Use this to enforce idempotency guards — e.g. "never call delete_booking
    twice on the same booking_id".

    Nodes where *arg_key* is absent from tool_input are excluded from duplicate
    detection (the argument is not being passed, so there is nothing to deduplicate on).

    Raises AssertionError listing every (arg_value, node_ids) pair that appears
    more than once.
    """
    calls: dict[object, list[str]] = defaultdict(list)
    for node in trace.nodes:
        if node.tool_name == tool and arg_key in node.tool_input:
            calls[node.tool_input[arg_key]].append(node.node_id)
    duplicates = {v: ids for v, ids in calls.items() if len(ids) > 1}
    if duplicates:
        details = "; ".join(
            f"{arg_key}={val!r} called {len(ids)} time(s) at nodes {ids}"
            for val, ids in duplicates.items()
        )
        raise AssertionError(
            f"no_duplicate_arg_call: tool '{tool}' called multiple times with "
            f"the same {arg_key}: {details}"
        )
