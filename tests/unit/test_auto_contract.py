"""Unit tests for automated contract synthesis."""

from __future__ import annotations

from trajeval.analysis.auto_contract import suggest_contracts
from trajeval.sdk.models import Trace, TraceNode

_TS = "2024-01-01T00:00:00Z"


def _trace(tools: list[str], tid: str = "t") -> Trace:
    nodes = [
        TraceNode(
            node_id=f"{tid}-n{i}", node_type="tool_call",
            tool_name=t, tool_input={}, tool_output={},
            timestamp=_TS,
        )
        for i, t in enumerate(tools)
    ]
    return Trace(
        trace_id=tid, agent_id="a", version_hash="v",
        started_at=_TS, completed_at=_TS,
        nodes=nodes, edges=[],
    )


def _retry_trace(tool: str, n: int, tid: str = "t") -> Trace:
    nodes = [
        TraceNode(
            node_id=f"{tid}-n{i}", node_type="tool_call",
            tool_name=tool, tool_input={"q": "same"},
            tool_output={}, timestamp=_TS,
        )
        for i in range(n)
    ]
    return Trace(
        trace_id=tid, agent_id="a", version_hash="v",
        started_at=_TS, completed_at=_TS,
        nodes=nodes, edges=[],
    )


# ---------------------------------------------------------------------------
# Banned tool discovery
# ---------------------------------------------------------------------------


def test_discovers_banned_tool() -> None:
    passing = [_trace(["search", "book"], f"p{i}") for i in range(5)]
    failing = [_trace(["search", "delete_user"], "f1")]
    suggestions = suggest_contracts(passing, failing)
    rules = [s.rule for s in suggestions]
    assert any("delete_user" in r for r in rules)


def test_no_banned_when_tool_in_both() -> None:
    passing = [_trace(["search"], f"p{i}") for i in range(5)]
    failing = [_trace(["search"], "f1")]
    suggestions = suggest_contracts(passing, failing)
    banned = [s for s in suggestions if s.strategy == "banned_tool"]
    assert banned == []


# ---------------------------------------------------------------------------
# Ordering patterns
# ---------------------------------------------------------------------------


def test_discovers_ordering_pattern() -> None:
    passing = [
        _trace(["search", "validate", "book"], f"p{i}")
        for i in range(5)
    ]
    suggestions = suggest_contracts(passing, [])
    ordering = [s for s in suggestions if s.strategy == "ordering"]
    rules = [s.rule for s in ordering]
    assert any("search before" in r for r in rules)


def test_no_ordering_with_too_few_traces() -> None:
    passing = [_trace(["search", "book"])]
    suggestions = suggest_contracts(passing, [])
    ordering = [s for s in suggestions if s.strategy == "ordering"]
    assert ordering == []


# ---------------------------------------------------------------------------
# Retry threshold
# ---------------------------------------------------------------------------


def test_discovers_retry_threshold() -> None:
    passing = [_trace(["search", "book"], f"p{i}") for i in range(3)]
    failing = [_retry_trace("search", 6, "f1")]
    suggestions = suggest_contracts(passing, failing)
    retry = [s for s in suggestions if s.strategy == "retry_threshold"]
    assert len(retry) >= 1
    assert "consecutive" in retry[0].rule.lower() or "retries" in retry[0].rule.lower()


# ---------------------------------------------------------------------------
# Confidence filter
# ---------------------------------------------------------------------------


def test_min_confidence_filters() -> None:
    passing = [_trace(["search", "book"], f"p{i}") for i in range(5)]
    failing = [_trace(["delete_user"], "f1")]
    high = suggest_contracts(passing, failing, min_confidence=0.99)
    low = suggest_contracts(passing, failing, min_confidence=0.1)
    assert len(high) <= len(low)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_traces() -> None:
    suggestions = suggest_contracts([], [])
    assert suggestions == []


def test_suggestion_has_rationale() -> None:
    passing = [_trace(["search", "book"], f"p{i}") for i in range(5)]
    failing = [_trace(["delete_user"], "f1")]
    suggestions = suggest_contracts(passing, failing)
    for s in suggestions:
        assert s.rationale
        assert s.strategy
