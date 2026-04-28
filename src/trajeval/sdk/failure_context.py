"""Failure context injection for self-correction loops.

Turns assertion failures into structured context that a well-designed
LangGraph agent (or any retry-aware orchestrator) can consume on its
next run.

Usage::

    from trajeval.sdk.failure_context import build_failure_context

    context = build_failure_context(trace, violations)
    result = app.invoke({
        "task": "book flight to SFO",
        "trajeval_context": context.to_state_dict(),
    })
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class FailureNode:
    """A node implicated in an assertion failure."""

    node_id: str
    tool_name: str | None
    depth: int
    issue: str


@dataclass
class FailureContext:
    """Structured failure report for injection into the next agent run.

    Attributes
    ----------
    previous_run_id:
        ``trace_id`` of the run that produced these violations.
    failed_assertions:
        Raw violation messages from the assertion runner.
    failure_nodes:
        Nodes in the trace that are directly implicated in failures.
    suggested_fixes:
        Human-readable, actionable suggestions for the agent developer
        (or the agent itself, if it has a self-correction loop).
    """

    previous_run_id: str
    failed_assertions: list[str]
    failure_nodes: list[FailureNode] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)

    def to_state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for injection into LangGraph state."""
        return {
            "previous_run_id": self.previous_run_id,
            "failed_assertions": self.failed_assertions,
            "failure_nodes": [
                {
                    "node_id": n.node_id,
                    "tool": n.tool_name,
                    "depth": n.depth,
                    "issue": n.issue,
                }
                for n in self.failure_nodes
            ],
            "suggested_fixes": self.suggested_fixes,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_failure_context(
    trace: Trace,
    violations: list[str],
) -> FailureContext:
    """Build a :class:`FailureContext` from *trace* and its *violations*.

    Parses each violation message to identify implicated tool names, finds the
    corresponding nodes in *trace*, and generates actionable fix suggestions.

    Parameters
    ----------
    trace:
        The trace that produced the violations.
    violations:
        Raw ``AssertionError`` messages from the assertion runner.

    Returns
    -------
    FailureContext
        Ready for injection via :meth:`FailureContext.to_state_dict`.
    """
    failure_nodes: list[FailureNode] = []
    suggested_fixes: list[str] = []

    seen_suggestions: set[str] = set()

    for violation in violations:
        nodes, fix = _parse_violation(violation, trace)
        failure_nodes.extend(nodes)
        if fix and fix not in seen_suggestions:
            suggested_fixes.append(fix)
            seen_suggestions.add(fix)

    return FailureContext(
        previous_run_id=trace.trace_id,
        failed_assertions=violations,
        failure_nodes=failure_nodes,
        suggested_fixes=suggested_fixes,
    )


# ---------------------------------------------------------------------------
# Violation parsers
# ---------------------------------------------------------------------------


def _parse_violation(
    violation: str,
    trace: Trace,
) -> tuple[list[FailureNode], str]:
    """Parse one violation string → (implicated nodes, suggested fix)."""
    # Dispatch on assertion prefix
    if violation.startswith("never_calls:"):
        return _parse_never_calls(violation, trace)
    if violation.startswith("tool_must_precede:"):
        return _parse_tool_must_precede(violation, trace)
    if violation.startswith("max_depth:"):
        return _parse_max_depth(violation, trace)
    if violation.startswith("no_cycles:"):
        return _parse_no_cycles(violation, trace)
    if violation.startswith("must_visit:"):
        return _parse_must_visit(violation, trace)
    if violation.startswith("tool_call_count:"):
        return _parse_tool_call_count(violation, trace)
    if violation.startswith("total_tool_calls:"):
        return _parse_total_tool_calls(violation, trace)
    if violation.startswith("cost_within:"):
        return _parse_cost_within(violation, trace)
    if violation.startswith("latency_within:"):
        return _parse_latency_within(violation, trace)
    if violation.startswith("no_duplicate_arg_call:"):
        return _parse_no_duplicate_arg_call(violation, trace)
    # Unknown assertion — no nodes, generic suggestion
    return [], f"Review assertion: {violation[:80]}"


# -- never_calls --------------------------------------------------------------
# "never_calls: tool '{tool}' was called {n} time(s) but must never be called.
#  Offending nodes: {ids}"

_RE_NEVER_CALLS = re.compile(r"never_calls: tool '([^']+)'")


def _parse_never_calls(violation: str, trace: Trace) -> tuple[list[FailureNode], str]:
    m = _RE_NEVER_CALLS.search(violation)
    if not m:
        return [], "Remove the banned tool call."
    tool = m.group(1)
    nodes = _nodes_for_tool(trace, tool, issue=f"Banned tool '{tool}' was called")
    fix = (
        f"Remove all calls to '{tool}'. "
        "This tool is explicitly forbidden by the contract."
    )
    return nodes, fix


# -- tool_must_precede --------------------------------------------------------
# "tool_must_precede: '{tool}' was never called but '{before}' was called..."
# "tool_must_precede: node '{node_id}' (tool='{before}') is not reachable from
#  any '{tool}' node. '{tool}' must precede '{before}'."

_RE_PRECEDE_MISSING = re.compile(
    r"tool_must_precede: '([^']+)' was never called but '([^']+)' was called"
)
_RE_PRECEDE_ORDER = re.compile(
    r"tool_must_precede: node '[^']+' \(tool='([^']+)'\) is not reachable from "
    r"any '([^']+)' node"
)


def _parse_tool_must_precede(
    violation: str, trace: Trace
) -> tuple[list[FailureNode], str]:
    m = _RE_PRECEDE_MISSING.search(violation)
    if m:
        tool, before = m.group(1), m.group(2)
        nodes = _nodes_for_tool(
            trace,
            before,
            issue=f"'{before}' called but required '{tool}' was never called",
        )
        fix = f"Ensure '{tool}' is called before '{before}'."
        return nodes, fix

    m = _RE_PRECEDE_ORDER.search(violation)
    if m:
        before, tool = m.group(1), m.group(2)
        nodes = _nodes_for_tool(
            trace,
            before,
            issue=f"'{before}' not reachable from '{tool}' — wrong order",
        )
        fix = f"Reorder execution: call '{tool}' before '{before}'."
        return nodes, fix

    return [], "Fix tool execution order per contract."


# -- max_depth ----------------------------------------------------------------
# "max_depth: {n} node(s) exceed max depth {limit}: node '{id}' (depth={d}), ..."

_RE_MAX_DEPTH = re.compile(r"max_depth: \d+ node\(s\) exceed max depth (\d+)")
_RE_DEPTH_NODE = re.compile(r"node '([^']+)' \(depth=(\d+)\)")


def _parse_max_depth(violation: str, trace: Trace) -> tuple[list[FailureNode], str]:
    limit_m = _RE_MAX_DEPTH.search(violation)
    limit = int(limit_m.group(1)) if limit_m else "?"

    offending: list[FailureNode] = []
    node_index = {n.node_id: n for n in trace.nodes}
    for m in _RE_DEPTH_NODE.finditer(violation):
        node_id, depth = m.group(1), int(m.group(2))
        node = node_index.get(node_id)
        tool = node.tool_name if node else None
        offending.append(
            FailureNode(
                node_id=node_id,
                tool_name=tool,
                depth=depth,
                issue=f"Node depth {depth} exceeds limit {limit}",
            )
        )
    fix = (
        f"Reduce nesting depth — max allowed is {limit}. "
        "Check for unbounded loops or deep recursion in the agent."
    )
    return offending, fix


# -- no_cycles ----------------------------------------------------------------
# "no_cycles: directed cycle detected in trace '{trace_id}': u→v → v→w ..."


def _parse_no_cycles(violation: str, trace: Trace) -> tuple[list[FailureNode], str]:
    fix = (
        "A cycle was detected in the execution graph. "
        "Add a termination condition or visited-set check to prevent infinite loops."
    )
    return [], fix


# -- must_visit ---------------------------------------------------------------
# "must_visit: required tool(s) were never called: ['a', 'b']. Called tools: ..."

_RE_MUST_VISIT = re.compile(
    r"must_visit: required tool\(s\) were never called: (\[[^\]]+\])"
)


def _parse_must_visit(violation: str, trace: Trace) -> tuple[list[FailureNode], str]:
    m = _RE_MUST_VISIT.search(violation)
    missing_str = m.group(1) if m else "unknown"
    fix = (
        f"The following required tools were never called: {missing_str}. "
        "Add these tool calls to the agent workflow."
    )
    return [], fix


# -- tool_call_count ----------------------------------------------------------
# "tool_call_count: tool '{tool}' was called {count} time(s), exceeding the
#  budget of {max} call(s)"

_RE_TOOL_COUNT = re.compile(
    r"tool_call_count: tool '([^']+)' was called (\d+) time\(s\), "
    r"exceeding the budget of (\d+)"
)


def _parse_tool_call_count(
    violation: str, trace: Trace
) -> tuple[list[FailureNode], str]:
    m = _RE_TOOL_COUNT.search(violation)
    if not m:
        return [], "Reduce tool call frequency to stay within budget."
    tool, count, budget = m.group(1), m.group(2), m.group(3)
    nodes = _nodes_for_tool(
        trace,
        tool,
        issue=f"Tool '{tool}' called {count}× — budget is {budget}",
    )
    fix = (
        f"Reduce calls to '{tool}' to at most {budget}. "
        f"It was called {count} time(s) this run."
    )
    return nodes, fix


# -- total_tool_calls ---------------------------------------------------------
# "total_tool_calls: {count} tool call(s) exceed budget of {max} ..."

_RE_TOTAL_CALLS = re.compile(
    r"total_tool_calls: (\d+) tool call\(s\) exceed budget of (\d+)"
)


def _parse_total_tool_calls(
    violation: str, trace: Trace
) -> tuple[list[FailureNode], str]:
    m = _RE_TOTAL_CALLS.search(violation)
    count = m.group(1) if m else "?"
    budget = m.group(2) if m else "?"
    fix = (
        f"Total tool calls ({count}) exceeded budget ({budget}). "
        "Reduce agent steps or increase the call budget in the contract."
    )
    return [], fix


# -- cost_within --------------------------------------------------------------
# "cost_within: total cost {actual} USD exceeds p90 budget {p90} USD ..."

_RE_COST = re.compile(
    r"cost_within: total cost ([\d.]+) USD exceeds p90 budget ([\d.]+) USD"
)


def _parse_cost_within(violation: str, trace: Trace) -> tuple[list[FailureNode], str]:
    m = _RE_COST.search(violation)
    actual = m.group(1) if m else "?"
    budget = m.group(2) if m else "?"
    fix = (
        f"Trace cost ${actual} exceeded the p90 budget ${budget}. "
        "Use cheaper model tiers or reduce the number of LLM calls."
    )
    return [], fix


# -- latency_within -----------------------------------------------------------
# "latency_within: p95 node latency {actual}ms exceeds budget {budget}ms ..."

_RE_LATENCY = re.compile(
    r"latency_within: p95 node latency (\d+)ms exceeds budget (\d+)ms"
)


def _parse_latency_within(
    violation: str, trace: Trace
) -> tuple[list[FailureNode], str]:
    m = _RE_LATENCY.search(violation)
    actual = m.group(1) if m else "?"
    budget = m.group(2) if m else "?"
    fix = (
        f"p95 node latency ({actual}ms) exceeded the budget ({budget}ms). "
        "Optimise the slowest tool calls or increase the latency budget."
    )
    return [], fix


# -- no_duplicate_arg_call ----------------------------------------------------
# "no_duplicate_arg_call: tool '{tool}' called multiple times with the same
#  {arg_key}: ..."

_RE_DUP_ARG = re.compile(
    r"no_duplicate_arg_call: tool '([^']+)' called multiple times with the same (\w+)"
)


def _parse_no_duplicate_arg_call(
    violation: str, trace: Trace
) -> tuple[list[FailureNode], str]:
    m = _RE_DUP_ARG.search(violation)
    if not m:
        return [], "Add an idempotency check to prevent duplicate tool calls."
    tool, arg_key = m.group(1), m.group(2)
    nodes = _nodes_for_tool(
        trace,
        tool,
        issue=f"Tool '{tool}' called multiple times with same '{arg_key}'",
    )
    fix = (
        f"Add an idempotency guard before calling '{tool}': "
        f"track seen values of '{arg_key}' and skip duplicates."
    )
    return nodes, fix


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _nodes_for_tool(
    trace: Trace,
    tool_name: str,
    *,
    issue: str,
) -> list[FailureNode]:
    return [
        FailureNode(
            node_id=n.node_id,
            tool_name=n.tool_name,
            depth=n.depth,
            issue=issue,
        )
        for n in trace.nodes
        if n.tool_name == tool_name
    ]
