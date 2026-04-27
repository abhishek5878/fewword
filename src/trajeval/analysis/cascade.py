"""Cascade analysis — root-cause attribution for failing traces.

Given a single failing trace and an assertion that fired, identifies
which node is the *root cause* — the earliest point where the agent's
behavior diverged into the cascade that ultimately caused the failure.

This differs from :mod:`trajeval.analysis.regret` which requires a
historical corpus and estimates counterfactual cost. Cascade analysis
is structural: it inspects the single trace and answers "which step
started the problem?"

Root-cause heuristics
---------------------
1. **First failing node**: run the assertion incrementally — on prefix
   [0..1], [0..2], ..., [0..N]. The first prefix where the assertion
   fails contains the root cause at its last node.
2. **Retry storm origin**: if a retry storm is detected, the root cause
   is the first node in the storm (the initial failed attempt).
3. **Error propagation**: if a tool_call node has an error in its output
   (configurable error keys), and later nodes consume that error,
   the error-producing node is the root cause.

Usage::

    from trajeval.analysis.cascade import find_root_cause

    result = find_root_cause(
        failing_trace,
        assertion=partial(must_visit, tools=["validate"]),
    )
    print(f"Root cause: node {result.root_node_id} at step {result.step}")
    print(f"Cascade depth: {result.cascade_depth} nodes affected")
    print(f"Wasted cost: ${result.wasted_cost_usd:.4f}")
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from trajeval.sdk.models import Trace, TraceNode


@dataclass(frozen=True)
class CascadeResult:
    """Root-cause attribution for a failing trace."""

    trace_id: str
    root_node_id: str
    root_tool_name: str | None
    step: int
    cascade_depth: int
    wasted_cost_usd: float
    wasted_nodes: int
    explanation: str


def find_root_cause(
    trace: Trace,
    assertion: Callable[[Trace], None],
    *,
    error_keys: list[str] | None = None,
) -> CascadeResult | None:
    """Find the root-cause node in a failing trace.

    Parameters
    ----------
    trace:
        A trace that fails the given assertion.
    assertion:
        The assertion that fires on this trace.
    error_keys:
        Keys in ``tool_output`` that indicate an error response
        (e.g. ``["error", "error_message", "status_code"]``).
        Nodes with these keys present are prioritized as root causes.

    Returns
    -------
    CascadeResult or None
        ``None`` if the trace actually passes the assertion (no failure
        to attribute).
    """
    # Verify the trace actually fails
    try:
        assertion(trace)
        return None  # trace passes — nothing to attribute
    except AssertionError:
        pass

    tool_nodes = [n for n in trace.nodes if n.node_type == "tool_call"]
    if not tool_nodes:
        return None

    _error_keys = error_keys or ["error", "error_message", "fault"]

    # --- Strategy 1: Incremental prefix scan ---
    # Run the assertion on increasingly longer prefixes of the trace.
    # The first prefix that fails tells us which node tipped it over.
    root_step = _incremental_scan(trace, tool_nodes, assertion)

    # --- Strategy 2: Error key detection ---
    # If a node's output contains an error key, and it appears before
    # or at the incremental root, prefer it as root cause.
    error_step = _first_error_node(tool_nodes, _error_keys)

    # --- Strategy 3: Retry storm origin ---
    storm_step = _retry_storm_origin(tool_nodes)

    # Pick the earliest signal
    candidates = [s for s in [root_step, error_step, storm_step] if s is not None]
    final_step = 0 if not candidates else min(candidates)

    root_node = tool_nodes[final_step]
    cascade_depth = len(tool_nodes) - final_step
    wasted_cost = sum(n.cost_usd for n in tool_nodes[final_step + 1 :])
    wasted_nodes = len(tool_nodes) - final_step - 1

    explanation = _build_explanation(
        final_step, root_node, root_step, error_step, storm_step
    )

    return CascadeResult(
        trace_id=trace.trace_id,
        root_node_id=root_node.node_id,
        root_tool_name=root_node.tool_name,
        step=final_step,
        cascade_depth=cascade_depth,
        wasted_cost_usd=wasted_cost,
        wasted_nodes=wasted_nodes,
        explanation=explanation,
    )


def _incremental_scan(
    trace: Trace,
    tool_nodes: list[TraceNode],
    assertion: Callable[[Trace], None],
) -> int | None:
    """Run the assertion on prefixes [0..1], [0..2], ... to find the tipping point."""
    for i in range(1, len(tool_nodes) + 1):
        prefix_nodes = list(trace.nodes[: _node_index(trace, tool_nodes[i - 1]) + 1])
        prefix_edges = [
            e
            for e in trace.edges
            if e.source in {n.node_id for n in prefix_nodes}
            and e.target in {n.node_id for n in prefix_nodes}
        ]
        prefix_trace = trace.model_copy(
            update={"nodes": prefix_nodes, "edges": prefix_edges}
        )
        try:
            assertion(prefix_trace)
        except AssertionError:
            return i - 1  # this node tipped it
    return None


def _node_index(trace: Trace, node: TraceNode) -> int:
    """Find the index of *node* in trace.nodes (by node_id)."""
    for i, n in enumerate(trace.nodes):
        if n.node_id == node.node_id:
            return i
    return len(trace.nodes) - 1


def _first_error_node(
    tool_nodes: list[TraceNode], error_keys: list[str]
) -> int | None:
    """Find the first tool node whose output contains an error key."""
    for i, node in enumerate(tool_nodes):
        if any(k in node.tool_output for k in error_keys):
            return i
    return None


def _retry_storm_origin(tool_nodes: list[TraceNode]) -> int | None:
    """Find the first node in a retry storm (3+ identical consecutive calls)."""
    if len(tool_nodes) < 3:
        return None

    def _sig(node: TraceNode) -> str:
        try:
            inp = json.dumps(node.tool_input, sort_keys=True, default=str)
        except (TypeError, ValueError):
            inp = repr(node.tool_input)
        return f"{node.tool_name}::{inp}"

    run_start = 0
    for i in range(1, len(tool_nodes)):
        if _sig(tool_nodes[i]) == _sig(tool_nodes[run_start]):
            if i - run_start + 1 >= 3:
                return run_start
        else:
            run_start = i
    return None


def _build_explanation(
    final_step: int,
    root_node: TraceNode,
    inc_step: int | None,
    err_step: int | None,
    storm_step: int | None,
) -> str:
    """Human-readable explanation of why this node was identified as root cause."""
    reasons: list[str] = []
    if storm_step == final_step:
        reasons.append(
            f"retry storm started at step {final_step} "
            f"(tool '{root_node.tool_name}')"
        )
    if err_step == final_step:
        reasons.append(
            f"error response detected in tool output at step {final_step}"
        )
    if inc_step == final_step:
        reasons.append(
            f"assertion first failed when step {final_step} was added"
        )
    if not reasons:
        reasons.append(
            f"earliest signal at step {final_step} "
            f"(tool '{root_node.tool_name}')"
        )
    return "; ".join(reasons)
