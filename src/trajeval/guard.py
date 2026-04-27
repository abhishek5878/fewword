"""Runtime guard — block a dangerous tool call before it executes.

This is the primitive the rest of TrajEval is built to support. Given:

  * a :class:`~trajeval.sdk.models.Trace` of what has already happened,
  * a proposed next tool call (name, input, and optionally a
    predicted output),
  * an :class:`~trajeval.action.ActionConfig` loaded from your
    ``.trajeval.yml``,

:func:`check` returns a :class:`GuardDecision` saying whether the
proposed call is safe or which assertions it would violate. The
check runs the same assertion stack as ``fewwords run`` — no new
detection logic — against a synthesized trace that includes the
proposed node appended.

The intended use is inside a tool dispatcher::

    from trajeval.action import load_config
    from trajeval.guard import check
    from trajeval.sdk.models import Trace

    config = load_config(Path(".trajeval.yml"))
    history = Trace(...)  # whatever the callback has captured so far

    decision = check(history, {"tool_name": "drop_database",
                               "tool_input": {"name": "prod"}}, config)
    if not decision.allow:
        raise RuntimeError(
            f"TrajEval blocked this call: {decision.messages[0]}"
        )
    # otherwise: execute the tool and record its output in ``history``
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

from trajeval.action import ActionConfig, CheckResult, replay_state_checks, run_checks
from trajeval.contract.state import SymbolicState
from trajeval.sdk.models import Trace, TraceEdge, TraceNode


@dataclass(frozen=True)
class GuardDecision:
    """Verdict from :func:`check`."""

    allow: bool
    violations: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def __str__(self) -> str:
        verdict = "ALLOW" if self.allow else "BLOCK"
        detail = "; ".join(self.violations) if self.violations else "-"
        return f"{verdict} ({self.latency_ms:.2f}ms) {detail}"


def check(
    history: Trace,
    proposed: dict[str, object],
    config: ActionConfig,
    *,
    node_id: str = "proposed",
) -> GuardDecision:
    """Return a :class:`GuardDecision` for ``proposed`` given ``history``.

    Parameters
    ----------
    history:
        Trace of what has already happened. Pass an empty Trace for the
        very first tool call.
    proposed:
        Dict with ``tool_name`` (str, required), ``tool_input`` (any,
        optional), ``tool_output`` (any, optional — pass if you have
        it for post-execution checks like ``stop_on_error`` or
        ``schema_validation`` on the just-run call; omit for
        pre-execution checks on name/input only).
    config:
        Parsed :class:`ActionConfig` (usually from ``load_config``).
    node_id:
        ID for the synthesized node. Default ``"proposed"``.

    The synthesized node is appended to ``history.nodes`` with a
    sequential edge from the last existing tool-call node. All
    assertions in ``config`` run against the synthesized trace — the
    same path as ``fewwords run`` — so behavior is identical.
    """
    tool_name = proposed.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("proposed['tool_name'] must be a non-empty str")
    tool_input = proposed.get("tool_input", {})
    tool_output = proposed.get("tool_output", {})

    new_node = TraceNode(
        node_id=node_id,
        node_type="tool_call",
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        cost_usd=0.0,
        duration_ms=0,
        depth=0,
        parent_node_id=None,
        timestamp=history.started_at or "1970-01-01T00:00:00Z",
    )

    new_nodes = [*history.nodes, new_node]
    last = next(
        (n for n in reversed(history.nodes) if n.node_type == "tool_call"),
        None,
    )
    extra_edge = (
        [TraceEdge(
            source=last.node_id, target=node_id, edge_type="sequential"
        )]
        if last is not None
        else [TraceEdge(
            source=node_id, target=node_id, edge_type="sequential"
        )]
    )
    new_edges = [*history.edges, *extra_edge]

    speculative = history.model_copy(
        update={"nodes": new_nodes, "edges": new_edges}
    )

    t0 = time.perf_counter()
    report = run_checks(speculative, config)
    latency_ms = (time.perf_counter() - t0) * 1000

    failing = [c for c in report.checks if not c.passed]
    return GuardDecision(
        allow=not failing,
        violations=[c.name for c in failing],
        messages=[c.message for c in failing],
        latency_ms=latency_ms,
    )


def check_pre(
    history: Trace,
    state: SymbolicState,
    proposed: dict[str, object],
    config: ActionConfig,
    *,
    node_id: str = "proposed",
) -> GuardDecision:
    """PreToolUse — block the proposed tool call before it executes.

    Evaluates the proposed tool's preconditions against *state* (the
    caller's threaded :class:`SymbolicState`) plus all the existing
    pre-execution checks (banned_tools, retry_storm, schemas, ...) over
    the speculative trace (``history`` + the proposed node).

    Postcondition schema validation for the proposed node is **deferred**
    — its ``tool_output`` isn't known yet. The caller invokes
    :func:`check_post` after the tool runs.

    Returns a :class:`GuardDecision`. State is not returned (PreToolUse
    is non-mutating); use :func:`check_post` to advance state.
    """
    speculative = _splice(history, proposed, node_id)
    return _evaluate(
        speculative,
        config,
        starting_state=state,
        skip_postcondition_node_ids=frozenset({node_id}),
    )


def check_post(
    history: Trace,
    state: SymbolicState,
    proposed: dict[str, object],
    observed_output: object,
    config: ActionConfig,
    *,
    node_id: str = "proposed",
) -> tuple[GuardDecision, SymbolicState]:
    """PostToolUse — validate the observed ``tool_output`` and advance state.

    Evaluates the proposed tool's postcondition (schema + state mutation)
    against *observed_output* plus the standard pre-execution checks.
    On schema pass, ``state_updates`` are applied and the new state is
    returned. On schema fail, the original *state* is returned unchanged
    — a bad output cannot poison downstream contracts.

    Returns ``(decision, new_state)``. The caller threads ``new_state``
    into the next :func:`check_pre` call.
    """
    proposed_with_output = {**proposed, "tool_output": observed_output}
    speculative = _splice(history, proposed_with_output, node_id)
    decision = _evaluate(speculative, config, starting_state=state)
    # Re-derive the post-replay state. Cheap (in-memory dict ops); avoids
    # threading state out of run_checks' ActionResult.
    discard: list[CheckResult] = []
    new_state = replay_state_checks(
        speculative, config.tools, discard, starting_state=state
    )
    return decision, new_state


def _splice(
    history: Trace, proposed: dict[str, object], node_id: str
) -> Trace:
    """Return ``history`` with a synthesized ``proposed`` node appended."""
    tool_name = proposed.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("proposed['tool_name'] must be a non-empty str")
    new_node = TraceNode(
        node_id=node_id,
        node_type="tool_call",
        tool_name=tool_name,
        tool_input=proposed.get("tool_input", {}),
        tool_output=proposed.get("tool_output", {}),
        cost_usd=0.0,
        duration_ms=0,
        depth=0,
        parent_node_id=None,
        timestamp=history.started_at or "1970-01-01T00:00:00Z",
    )
    last = next(
        (n for n in reversed(history.nodes) if n.node_type == "tool_call"),
        None,
    )
    extra_edge = (
        [TraceEdge(source=last.node_id, target=node_id, edge_type="sequential")]
        if last is not None
        else [TraceEdge(source=node_id, target=node_id, edge_type="sequential")]
    )
    return history.model_copy(
        update={
            "nodes": [*history.nodes, new_node],
            "edges": [*history.edges, *extra_edge],
        }
    )


def _evaluate(
    speculative: Trace,
    config: ActionConfig,
    *,
    starting_state: SymbolicState,
    skip_postcondition_node_ids: frozenset[str] = frozenset(),
) -> GuardDecision:
    """Run all checks over *speculative* and shape the GuardDecision.

    Splits the work: existing non-state checks come from ``run_checks``
    over a tools-stripped config (avoids double-replay); state-replay
    checks come from a controlled :func:`replay_state_checks` call.
    """
    t0 = time.perf_counter()
    config_no_tools = dataclasses.replace(config, tools={})
    non_state = run_checks(speculative, config_no_tools)
    state_checks: list[CheckResult] = []
    replay_state_checks(
        speculative,
        config.tools,
        state_checks,
        starting_state=starting_state,
        skip_postcondition_node_ids=skip_postcondition_node_ids,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    all_checks = [*non_state.checks, *state_checks]
    failing = [c for c in all_checks if not c.passed]
    return GuardDecision(
        allow=not failing,
        violations=[c.name for c in failing],
        messages=[c.message for c in failing],
        latency_ms=latency_ms,
    )


def empty_history(
    *,
    trace_id: str = "guard-session",
    agent_id: str = "agent",
) -> Trace:
    """Return an empty :class:`Trace` suitable for the first guard
    call of a session.

    A real integration should use the Trace the SDK callback has been
    populating. This helper is for tests, examples, and the common
    case where the guard is checked before any tool has run.
    """
    return Trace(
        trace_id=trace_id,
        agent_id=agent_id,
        version_hash="guard",
        started_at="1970-01-01T00:00:00Z",
        completed_at="1970-01-01T00:00:00Z",
        total_cost_usd=0.0,
        total_tokens=0,
        nodes=[],
        edges=[],
    )
