"""Unit tests for the assertion DSL.

Uses pytest for fixed cases and Hypothesis for property-based testing.
All tests are fast and in-memory — no I/O.
"""

from __future__ import annotations

import functools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trajeval.assertions.core import (
    Severity,
    ViolationError,
    cost_within,
    latency_within,
    max_depth,
    must_visit,
    never_calls,
    no_cycles,
    no_duplicate_arg_call,
    no_retry_storm,
    severity,
    tool_call_count,
    tool_must_precede,
    total_tool_calls,
    validate_tool_outputs,
)
from trajeval.sdk.models import Trace, TraceEdge, TraceNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STARTED = "2024-01-01T00:00:00Z"
_COMPLETED = "2024-01-01T00:01:00Z"


def _node(
    node_id: str,
    *,
    tool_name: str | None = None,
    depth: int = 0,
    parent_node_id: str | None = None,
    node_type: str = "tool_call",
) -> TraceNode:
    return TraceNode(
        node_id=node_id,
        node_type=node_type,  # type: ignore[arg-type]
        tool_name=tool_name,
        tool_input={},
        tool_output={},
        cost_usd=0.0,
        duration_ms=0,
        depth=depth,
        parent_node_id=parent_node_id,
        timestamp=_STARTED,
    )


def _edge(source: str, target: str, edge_type: str = "sequential") -> TraceEdge:
    return TraceEdge(source=source, target=target, edge_type=edge_type)  # type: ignore[arg-type]


def _trace(
    nodes: list[TraceNode],
    edges: list[TraceEdge],
    trace_id: str = "trace-1",
) -> Trace:
    return Trace(
        trace_id=trace_id,
        agent_id="agent-1",
        version_hash="abc1234",
        started_at=_STARTED,
        completed_at=_COMPLETED,
        total_cost_usd=0.0,
        total_tokens=0,
        nodes=nodes,
        edges=edges,
    )


# ===========================================================================
# tool_must_precede
# ===========================================================================


class TestToolMustPrecede:
    def test_passes_when_tool_precedes_before(self) -> None:
        """search → book: should pass."""
        nodes = [_node("n1", tool_name="search"), _node("n2", tool_name="book")]
        edges = [_edge("n1", "n2")]
        trace = _trace(nodes, edges)
        tool_must_precede(trace, "search", before="book")  # must not raise

    def test_passes_when_before_never_called(self) -> None:
        """If 'book' never appears, constraint is vacuously satisfied."""
        nodes = [_node("n1", tool_name="search")]
        trace = _trace(nodes, [])
        tool_must_precede(trace, "search", before="book")

    def test_fails_when_tool_never_called_but_before_is(self) -> None:
        """'book' is called but 'search' is never called."""
        nodes = [_node("n1", tool_name="book")]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="never called"):
            tool_must_precede(trace, "search", before="book")

    def test_fails_when_before_has_no_path_from_tool(self) -> None:
        """Two disconnected nodes: search and book with no edge between them."""
        nodes = [_node("n1", tool_name="search"), _node("n2", tool_name="book")]
        trace = _trace(nodes, [])  # no edges
        with pytest.raises(AssertionError, match="not reachable"):
            tool_must_precede(trace, "search", before="book")

    def test_fails_when_order_is_reversed(self) -> None:
        """book → search: 'search' does not precede 'book'."""
        nodes = [_node("n1", tool_name="book"), _node("n2", tool_name="search")]
        edges = [_edge("n1", "n2")]
        trace = _trace(nodes, edges)
        with pytest.raises(AssertionError, match="not reachable"):
            tool_must_precede(trace, "search", before="book")

    def test_passes_with_multiple_tool_nodes(self) -> None:
        """Multiple search nodes, all before book — should pass."""
        nodes = [
            _node("n1", tool_name="search"),
            _node("n2", tool_name="search"),
            _node("n3", tool_name="book"),
        ]
        edges = [_edge("n1", "n2"), _edge("n2", "n3")]
        trace = _trace(nodes, edges)
        tool_must_precede(trace, "search", before="book")

    def test_passes_with_branching_path(self) -> None:
        """search → validate → book: indirect path still satisfies constraint."""
        nodes = [
            _node("n1", tool_name="search"),
            _node("n2", tool_name="validate"),
            _node("n3", tool_name="book"),
        ]
        edges = [_edge("n1", "n2"), _edge("n2", "n3")]
        trace = _trace(nodes, edges)
        tool_must_precede(trace, "search", before="book")


# ===========================================================================
# max_depth
# ===========================================================================


class TestMaxDepth:
    def test_passes_when_all_nodes_within_depth(self) -> None:
        nodes = [_node("n1", depth=0), _node("n2", depth=2), _node("n3", depth=5)]
        trace = _trace(nodes, [])
        max_depth(trace, 5)  # must not raise

    def test_passes_on_empty_trace(self) -> None:
        trace = _trace([], [])
        max_depth(trace, 0)

    def test_passes_at_exact_limit(self) -> None:
        nodes = [_node("n1", depth=3)]
        trace = _trace(nodes, [])
        max_depth(trace, 3)

    def test_fails_when_one_node_exceeds_depth(self) -> None:
        nodes = [_node("n1", depth=0), _node("n2", depth=6)]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="exceed max depth 5"):
            max_depth(trace, 5)

    def test_fails_listing_all_violations(self) -> None:
        nodes = [_node("n1", depth=10), _node("n2", depth=20)]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="2 node"):
            max_depth(trace, 5)

    def test_raises_value_error_for_negative_n(self) -> None:
        trace = _trace([], [])
        with pytest.raises(ValueError, match="n must be >= 0"):
            max_depth(trace, -1)

    @given(
        depths=st.lists(
            st.integers(min_value=0, max_value=20), min_size=1, max_size=10
        ),
        limit=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=200)
    def test_property_max_depth_matches_python_max(
        self, depths: list[int], limit: int
    ) -> None:
        """max_depth passes iff all depths <= limit."""
        nodes = [_node(f"n{i}", depth=d) for i, d in enumerate(depths)]
        trace = _trace(nodes, [])
        if max(depths) <= limit:
            max_depth(trace, limit)  # must not raise
        else:
            with pytest.raises(AssertionError):
                max_depth(trace, limit)


# ===========================================================================
# no_cycles
# ===========================================================================


class TestNoCycles:
    def test_passes_on_empty_trace(self) -> None:
        trace = _trace([], [])
        no_cycles(trace)

    def test_passes_on_linear_chain(self) -> None:
        nodes = [_node("n1"), _node("n2"), _node("n3")]
        edges = [_edge("n1", "n2"), _edge("n2", "n3")]
        trace = _trace(nodes, edges)
        no_cycles(trace)

    def test_passes_on_dag(self) -> None:
        """Diamond DAG: n1 → n2, n1 → n3, n2 → n4, n3 → n4."""
        nodes = [_node("n1"), _node("n2"), _node("n3"), _node("n4")]
        edges = [
            _edge("n1", "n2"),
            _edge("n1", "n3"),
            _edge("n2", "n4"),
            _edge("n3", "n4"),
        ]
        trace = _trace(nodes, edges)
        no_cycles(trace)

    def test_fails_on_self_loop(self) -> None:
        nodes = [_node("n1")]
        edges = [_edge("n1", "n1")]
        trace = _trace(nodes, edges)
        with pytest.raises(AssertionError, match="cycle detected"):
            no_cycles(trace)

    def test_fails_on_two_node_cycle(self) -> None:
        nodes = [_node("n1"), _node("n2")]
        edges = [_edge("n1", "n2"), _edge("n2", "n1")]
        trace = _trace(nodes, edges)
        with pytest.raises(AssertionError, match="cycle detected"):
            no_cycles(trace)

    def test_fails_on_three_node_cycle(self) -> None:
        nodes = [_node("n1"), _node("n2"), _node("n3")]
        edges = [_edge("n1", "n2"), _edge("n2", "n3"), _edge("n3", "n1")]
        trace = _trace(nodes, edges)
        with pytest.raises(AssertionError, match="cycle detected"):
            no_cycles(trace)

    @given(
        n_nodes=st.integers(min_value=2, max_value=8),
    )
    @settings(max_examples=100)
    def test_property_linear_chain_never_has_cycles(self, n_nodes: int) -> None:
        """A simple linear chain n0→n1→…→n(k-1) is always acyclic."""
        nodes = [_node(f"n{i}") for i in range(n_nodes)]
        edges = [_edge(f"n{i}", f"n{i + 1}") for i in range(n_nodes - 1)]
        trace = _trace(nodes, edges)
        no_cycles(trace)  # must not raise


# ===========================================================================
# cost_within
# ===========================================================================


def _trace_with_cost(total: float) -> Trace:
    return Trace(
        trace_id="t1",
        agent_id="a1",
        version_hash="abc",
        started_at=_STARTED,
        completed_at=_COMPLETED,
        total_cost_usd=total,
    )


class TestCostWithin:
    def test_passes_when_cost_equals_budget(self) -> None:
        cost_within(_trace_with_cost(1.50), p90=1.50)

    def test_passes_when_cost_under_budget(self) -> None:
        cost_within(_trace_with_cost(0.50), p90=1.50)

    def test_passes_on_zero_cost(self) -> None:
        cost_within(_trace_with_cost(0.0), p90=1.50)

    def test_fails_when_cost_exceeds_budget(self) -> None:
        with pytest.raises(AssertionError, match="exceeds p90 budget"):
            cost_within(_trace_with_cost(2.00), p90=1.50)

    def test_raises_value_error_for_negative_budget(self) -> None:
        with pytest.raises(ValueError, match="p90 must be >= 0"):
            cost_within(_trace_with_cost(0.0), p90=-0.01)

    @given(
        total=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        budget=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_property_passes_iff_cost_le_budget(
        self, total: float, budget: float
    ) -> None:
        trace = _trace_with_cost(total)
        if total <= budget:
            cost_within(trace, p90=budget)
        else:
            with pytest.raises(AssertionError):
                cost_within(trace, p90=budget)


# ===========================================================================
# never_calls
# ===========================================================================


class TestNeverCalls:
    def test_passes_when_tool_absent(self) -> None:
        nodes = [_node("n1", tool_name="search")]
        trace = _trace(nodes, [])
        never_calls(trace, "delete_user")

    def test_passes_on_empty_trace(self) -> None:
        trace = _trace([], [])
        never_calls(trace, "delete_user")

    def test_fails_when_tool_called_once(self) -> None:
        nodes = [_node("n1", tool_name="delete_user")]
        trace = _trace(nodes, [])
        # never_calls error format: "never_calls: tool pattern '...'
        # matched N time(s). Offending nodes: '...'"
        pattern = "never_calls.*delete_user.*matched 1 time"
        with pytest.raises(AssertionError, match=pattern):
            never_calls(trace, "delete_user")

    def test_fails_listing_all_occurrences(self) -> None:
        nodes = [
            _node("n1", tool_name="delete_user"),
            _node("n2", tool_name="delete_user"),
        ]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="2 time"):
            never_calls(trace, "delete_user")


# ===========================================================================
# must_visit
# ===========================================================================


class TestMustVisit:
    def test_passes_when_all_tools_present(self) -> None:
        nodes = [
            _node("n1", tool_name="search"),
            _node("n2", tool_name="validate"),
        ]
        trace = _trace(nodes, [])
        must_visit(trace, ["search", "validate"])

    def test_passes_on_empty_required_list(self) -> None:
        trace = _trace([], [])
        must_visit(trace, [])

    def test_fails_when_one_tool_missing(self) -> None:
        nodes = [_node("n1", tool_name="search")]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="validate"):
            must_visit(trace, ["search", "validate"])

    def test_fails_when_all_tools_missing(self) -> None:
        trace = _trace([], [])
        with pytest.raises(AssertionError, match="required tool"):
            must_visit(trace, ["search", "validate"])

    def test_passes_with_extra_tools_in_trace(self) -> None:
        """Extra tool calls beyond the required set are fine."""
        nodes = [
            _node("n1", tool_name="search"),
            _node("n2", tool_name="validate"),
            _node("n3", tool_name="unrelated"),
        ]
        trace = _trace(nodes, [])
        must_visit(trace, ["search", "validate"])


# ===========================================================================
# tool_call_count
# ===========================================================================


class TestToolCallCount:
    def test_passes_when_count_under_limit(self) -> None:
        nodes = [_node("n1", tool_name="llm"), _node("n2", tool_name="llm")]
        trace = _trace(nodes, [])
        tool_call_count(trace, "llm", max=5)

    def test_passes_when_count_equals_limit(self) -> None:
        nodes = [_node(f"n{i}", tool_name="llm") for i in range(3)]
        trace = _trace(nodes, [])
        tool_call_count(trace, "llm", max=3)

    def test_passes_when_tool_absent(self) -> None:
        trace = _trace([], [])
        tool_call_count(trace, "llm", max=10)

    def test_fails_when_count_exceeds_limit(self) -> None:
        nodes = [_node(f"n{i}", tool_name="llm") for i in range(11)]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="exceeding the budget"):
            tool_call_count(trace, "llm", max=10)

    def test_raises_value_error_for_negative_max(self) -> None:
        trace = _trace([], [])
        with pytest.raises(ValueError, match="max must be >= 0"):
            tool_call_count(trace, "llm", max=-1)

    @given(
        n_calls=st.integers(min_value=0, max_value=20),
        budget=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=200)
    def test_property_passes_iff_count_le_budget(
        self, n_calls: int, budget: int
    ) -> None:
        nodes = [_node(f"n{i}", tool_name="llm") for i in range(n_calls)]
        trace = _trace(nodes, [])
        if n_calls <= budget:
            tool_call_count(trace, "llm", max=budget)
        else:
            with pytest.raises(AssertionError):
                tool_call_count(trace, "llm", max=budget)


# ===========================================================================
# latency_within
# ===========================================================================


def _trace_with_latencies(durations: list[int]) -> Trace:
    nodes = [_node(f"n{i}", depth=0) for i in range(len(durations))]
    # Replace frozen nodes with custom durations using model_copy
    new_nodes = [
        n.model_copy(update={"duration_ms": d})
        for n, d in zip(nodes, durations, strict=True)
    ]
    return Trace(
        trace_id="t-latency",
        agent_id="a1",
        version_hash="abc",
        started_at=_STARTED,
        completed_at=_COMPLETED,
        nodes=new_nodes,
    )


class TestLatencyWithin:
    def test_passes_on_empty_trace(self) -> None:
        trace = _trace([], [])
        latency_within(trace, p95=5000)

    def test_passes_when_all_durations_under_threshold(self) -> None:
        trace = _trace_with_latencies([100, 200, 300, 400, 500])
        latency_within(trace, p95=5000)

    def test_passes_at_exact_threshold(self) -> None:
        # With 20 values all equal, p95 == 500
        trace = _trace_with_latencies([500] * 20)
        latency_within(trace, p95=500)

    def test_fails_when_p95_exceeds_threshold(self) -> None:
        # 95% of nodes are slow
        durations = [6000] * 19 + [100]
        trace = _trace_with_latencies(durations)
        with pytest.raises(AssertionError, match="p95 node latency"):
            latency_within(trace, p95=5000)

    def test_raises_value_error_for_negative_p95(self) -> None:
        trace = _trace([], [])
        with pytest.raises(ValueError, match="p95 must be >= 0"):
            latency_within(trace, p95=-1)

    @given(
        durations=st.lists(
            st.integers(min_value=0, max_value=10000), min_size=1, max_size=20
        ),
        budget=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=200)
    def test_property_consistent_with_numpy_percentile(
        self, durations: list[int], budget: int
    ) -> None:
        """latency_within outcome matches numpy.percentile directly."""
        import numpy as np

        trace = _trace_with_latencies(durations)
        p95_actual = int(np.percentile(durations, 95))
        if p95_actual <= budget:
            latency_within(trace, p95=budget)
        else:
            with pytest.raises(AssertionError):
                latency_within(trace, p95=budget)


# ===========================================================================
# total_tool_calls
# ===========================================================================


def _tool_call_node(node_id: str, tool_name: str = "search") -> TraceNode:
    return _node(node_id, tool_name=tool_name, node_type="tool_call")


def _llm_node(node_id: str) -> TraceNode:
    return _node(node_id, tool_name=None, node_type="llm_call")


class TestTotalToolCalls:
    def test_passes_when_count_under_limit(self) -> None:
        nodes = [_tool_call_node("n1"), _tool_call_node("n2")]
        trace = _trace(nodes, [])
        total_tool_calls(trace, max=5)

    def test_passes_when_count_equals_limit(self) -> None:
        nodes = [_tool_call_node(f"n{i}") for i in range(3)]
        trace = _trace(nodes, [])
        total_tool_calls(trace, max=3)

    def test_passes_on_empty_trace(self) -> None:
        trace = _trace([], [])
        total_tool_calls(trace, max=0)

    def test_fails_when_count_exceeds_limit(self) -> None:
        nodes = [_tool_call_node(f"n{i}") for i in range(5)]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="exceed budget"):
            total_tool_calls(trace, max=4)

    def test_llm_nodes_not_counted(self) -> None:
        """Only tool_call nodes count — llm_call nodes are excluded."""
        nodes = [
            _tool_call_node("n1"),
            _llm_node("n2"),
            _llm_node("n3"),
        ]
        trace = _trace(nodes, [])
        total_tool_calls(trace, max=1)  # only 1 tool_call node

    def test_raises_value_error_for_negative_max(self) -> None:
        trace = _trace([], [])
        with pytest.raises(ValueError, match="max must be >= 0"):
            total_tool_calls(trace, max=-1)

    def test_error_includes_trace_id(self) -> None:
        nodes = [_tool_call_node(f"n{i}") for i in range(3)]
        trace = _trace(nodes, [], trace_id="my-trace")
        with pytest.raises(AssertionError, match="my-trace"):
            total_tool_calls(trace, max=2)


# ===========================================================================
# no_duplicate_arg_call
# ===========================================================================


def _booking_node(node_id: str, booking_id: str) -> TraceNode:
    return TraceNode(
        node_id=node_id,
        node_type="tool_call",  # type: ignore[arg-type]
        tool_name="delete_booking",
        tool_input={"booking_id": booking_id},
        tool_output={},
        cost_usd=0.0,
        duration_ms=0,
        depth=0,
        parent_node_id=None,
        timestamp=_STARTED,
    )


class TestNoDuplicateArgCall:
    def test_passes_when_all_args_unique(self) -> None:
        nodes = [
            _booking_node("n1", "abc"),
            _booking_node("n2", "xyz"),
        ]
        trace = _trace(nodes, [])
        no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")

    def test_passes_on_empty_trace(self) -> None:
        trace = _trace([], [])
        no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")

    def test_passes_when_tool_absent(self) -> None:
        nodes = [_node("n1", tool_name="search")]
        trace = _trace(nodes, [])
        no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")

    def test_passes_when_arg_key_absent_from_input(self) -> None:
        """Nodes missing the arg_key are excluded from duplicate detection."""
        node = TraceNode(
            node_id="n1",
            node_type="tool_call",  # type: ignore[arg-type]
            tool_name="delete_booking",
            tool_input={"other_key": "value"},
            tool_output={},
            cost_usd=0.0,
            duration_ms=0,
            depth=0,
            parent_node_id=None,
            timestamp=_STARTED,
        )
        trace = _trace([node, node.model_copy(update={"node_id": "n2"})], [])
        no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")

    def test_fails_when_same_arg_called_twice(self) -> None:
        nodes = [
            _booking_node("n1", "abc"),
            _booking_node("n2", "abc"),
        ]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="delete_booking"):
            no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")

    def test_error_includes_arg_value_and_nodes(self) -> None:
        nodes = [
            _booking_node("n1", "abc"),
            _booking_node("n2", "abc"),
        ]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError, match="booking_id"):
            no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")

    def test_multiple_duplicate_groups_all_reported(self) -> None:
        nodes = [
            _booking_node("n1", "abc"),
            _booking_node("n2", "abc"),
            _booking_node("n3", "xyz"),
            _booking_node("n4", "xyz"),
        ]
        trace = _trace(nodes, [])
        with pytest.raises(AssertionError) as exc_info:
            no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")
        msg = str(exc_info.value)
        assert "abc" in msg
        assert "xyz" in msg

    def test_passes_when_different_tools_share_arg_value(self) -> None:
        """Duplicate detection is per-tool — different tools sharing a value is fine."""
        n1 = TraceNode(
            node_id="n1",
            node_type="tool_call",  # type: ignore[arg-type]
            tool_name="delete_booking",
            tool_input={"booking_id": "abc"},
            tool_output={},
            cost_usd=0.0,
            duration_ms=0,
            depth=0,
            parent_node_id=None,
            timestamp=_STARTED,
        )
        n2 = TraceNode(
            node_id="n2",
            node_type="tool_call",  # type: ignore[arg-type]
            tool_name="update_booking",
            tool_input={"booking_id": "abc"},
            tool_output={},
            cost_usd=0.0,
            duration_ms=0,
            depth=0,
            parent_node_id=None,
            timestamp=_STARTED,
        )
        trace = _trace([n1, n2], [])
        no_duplicate_arg_call(trace, "delete_booking", arg_key="booking_id")


# ---------------------------------------------------------------------------
# Severity system — gap 7 fix
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_severity_ordering(self) -> None:
        assert Severity.P0 < Severity.P1 < Severity.P2

    def test_p0_is_highest_urgency(self) -> None:
        assert min(Severity) == Severity.P0

    def test_p2_is_lowest_urgency(self) -> None:
        assert max(Severity) == Severity.P2


class TestViolationError:
    def test_is_assertion_error_subclass(self) -> None:
        err = ViolationError("bad", severity=Severity.P0)
        assert isinstance(err, AssertionError)

    def test_carries_severity(self) -> None:
        err = ViolationError("bad", severity=Severity.P1)
        assert err.severity == Severity.P1

    def test_carries_assertion_name(self) -> None:
        err = ViolationError("bad", severity=Severity.P0, assertion_name="no-delete")
        assert err.assertion_name == "no-delete"

    def test_str_includes_severity_label(self) -> None:
        err = ViolationError("something broke", severity=Severity.P0)
        assert "[P0]" in str(err)

    def test_str_includes_assertion_name(self) -> None:
        err = ViolationError("msg", severity=Severity.P1, assertion_name="my-rule")
        assert "my-rule" in str(err)

    def test_caught_by_assertion_error_handler(self) -> None:
        """Existing except AssertionError blocks still catch ViolationError."""
        raised = False
        try:
            raise ViolationError("oops", severity=Severity.P2)
        except AssertionError:
            raised = True
        assert raised


class TestSeverityDecorator:
    def test_pass_through_on_success(self) -> None:
        fn = severity(lambda trace: None, level=Severity.P0)
        trace = _simple_trace()
        fn(trace)  # must not raise

    def test_upgrades_assertion_error_to_violation(self) -> None:
        def _always_fail(t: Trace) -> None:
            raise AssertionError("failed")

        fn = severity(_always_fail, level=Severity.P1)
        with pytest.raises(ViolationError) as exc_info:
            fn(_simple_trace())
        assert exc_info.value.severity == Severity.P1

    def test_preserves_original_message(self) -> None:
        def _fail(t: Trace) -> None:
            raise AssertionError("original message")

        fn = severity(_fail, level=Severity.P0)
        with pytest.raises(ViolationError) as exc_info:
            fn(_simple_trace())
        assert "original message" in str(exc_info.value)

    def test_name_appears_in_error(self) -> None:
        fn = severity(
            lambda t: (_ for _ in ()).throw(AssertionError("x")),
            level=Severity.P2,
            name="my-rule",
        )
        with pytest.raises(ViolationError) as exc_info:
            fn(_simple_trace())
        assert "my-rule" in str(exc_info.value)

    def test_wraps_never_calls_with_p0(self) -> None:
        """End-to-end: never_calls wrapped as P0 raises ViolationError."""
        from trajeval.assertions.core import never_calls

        guarded = severity(
            functools.partial(never_calls, tool="destroy_all"),
            level=Severity.P0,
            name="no-destroy",
        )
        bad_trace = _trace([_node("n1", tool_name="destroy_all")], [])
        with pytest.raises(ViolationError) as exc_info:
            guarded(bad_trace)
        assert exc_info.value.severity == Severity.P0

    def test_does_not_double_wrap_violation_error(self) -> None:
        """If inner raises ViolationError, outer severity must not re-wrap it."""
        inner = severity(
            lambda t: (_ for _ in ()).throw(AssertionError("inner")),
            level=Severity.P1,
            name="inner-rule",
        )
        outer = severity(inner, level=Severity.P0, name="outer-rule")
        with pytest.raises(ViolationError) as exc_info:
            outer(_simple_trace())
        # Should preserve the inner severity (P1), not re-wrap to P0
        assert exc_info.value.severity == Severity.P1

    def test_severity_p2_is_still_assertion_error(self) -> None:
        fn = severity(
            lambda t: (_ for _ in ()).throw(AssertionError("soft")),
            level=Severity.P2,
        )
        with pytest.raises(AssertionError):
            fn(_simple_trace())


def _simple_trace() -> Trace:
    return _trace([], [])


# ===========================================================================
# no_retry_storm
# ===========================================================================


def _retry_node(
    node_id: str, tool_name: str, tool_input: dict[str, object]
) -> TraceNode:
    return TraceNode(
        node_id=node_id,
        node_type="tool_call",
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output={},
        cost_usd=0.0,
        duration_ms=0,
        depth=0,
        parent_node_id=None,
        timestamp=_STARTED,
    )


class TestNoRetryStorm:
    def test_passes_on_empty_trace(self) -> None:
        trace = _trace([], [])
        no_retry_storm(trace)

    def test_passes_on_single_node(self) -> None:
        nodes = [_retry_node("n0", "search", {"q": "hello"})]
        no_retry_storm(_trace(nodes, []))

    def test_passes_when_consecutive_calls_differ_in_args(self) -> None:
        nodes = [
            _retry_node("n0", "search", {"q": "hello"}),
            _retry_node("n1", "search", {"q": "world"}),
            _retry_node("n2", "search", {"q": "foo"}),
            _retry_node("n3", "search", {"q": "bar"}),
        ]
        no_retry_storm(_trace(nodes, []))

    def test_passes_when_consecutive_identical_under_threshold(self) -> None:
        nodes = [
            _retry_node("n0", "search", {"q": "hello"}),
            _retry_node("n1", "search", {"q": "hello"}),
            _retry_node("n2", "search", {"q": "hello"}),
        ]
        no_retry_storm(_trace(nodes, []), max_consecutive=3)

    def test_fails_when_consecutive_exceeds_threshold(self) -> None:
        nodes = [
            _retry_node("n0", "search", {"q": "hello"}),
            _retry_node("n1", "search", {"q": "hello"}),
            _retry_node("n2", "search", {"q": "hello"}),
            _retry_node("n3", "search", {"q": "hello"}),
        ]
        with pytest.raises(AssertionError, match="retry storm"):
            no_retry_storm(_trace(nodes, []), max_consecutive=3)

    def test_error_includes_tool_name_and_run_length(self) -> None:
        nodes = [
            _retry_node(f"n{i}", "fetch_data", {"url": "http://x"}) for i in range(5)
        ]
        with pytest.raises(AssertionError, match="fetch_data") as exc:
            no_retry_storm(_trace(nodes, []), max_consecutive=2)
        assert "consecutive" in str(exc.value)

    def test_different_tools_same_args_resets_run(self) -> None:
        """search(q=x) then book(q=x) are not the same call."""
        nodes = [
            _retry_node("n0", "search", {"q": "hello"}),
            _retry_node("n1", "book", {"q": "hello"}),
            _retry_node("n2", "search", {"q": "hello"}),
        ]
        no_retry_storm(_trace(nodes, []), max_consecutive=1)

    def test_interleaved_different_call_resets_run(self) -> None:
        """3 identical, 1 different, 3 identical = no storm at threshold 3."""
        nodes = [
            _retry_node("n0", "search", {"q": "a"}),
            _retry_node("n1", "search", {"q": "a"}),
            _retry_node("n2", "search", {"q": "a"}),
            _retry_node("n3", "search", {"q": "b"}),
            _retry_node("n4", "search", {"q": "a"}),
            _retry_node("n5", "search", {"q": "a"}),
            _retry_node("n6", "search", {"q": "a"}),
        ]
        no_retry_storm(_trace(nodes, []), max_consecutive=3)

    def test_llm_nodes_are_ignored(self) -> None:
        """Only tool_call nodes participate — llm_call nodes are skipped."""
        nodes = [
            _retry_node("n0", "search", {"q": "hello"}),
            _node("n1", tool_name=None, node_type="llm_call"),
            _retry_node("n2", "search", {"q": "hello"}),
            _node("n3", tool_name=None, node_type="llm_call"),
            _retry_node("n4", "search", {"q": "hello"}),
        ]
        no_retry_storm(_trace(nodes, []), max_consecutive=3)

    def test_max_consecutive_1_catches_any_immediate_retry(self) -> None:
        nodes = [
            _retry_node("n0", "search", {"q": "hello"}),
            _retry_node("n1", "search", {"q": "hello"}),
        ]
        with pytest.raises(AssertionError, match="retry storm"):
            no_retry_storm(_trace(nodes, []), max_consecutive=1)

    def test_raises_value_error_for_zero_threshold(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive must be >= 1"):
            no_retry_storm(_trace([], []), max_consecutive=0)

    def test_default_threshold_is_3(self) -> None:
        """4 identical consecutive calls should fail with default threshold."""
        nodes = [_retry_node(f"n{i}", "search", {"q": "same"}) for i in range(4)]
        with pytest.raises(AssertionError):
            no_retry_storm(_trace(nodes, []))


# ===========================================================================
# validate_tool_outputs
# ===========================================================================


def _schema_node(
    node_id: str,
    tool_name: str,
    output: dict[str, object],
) -> TraceNode:
    return TraceNode(
        node_id=node_id,
        node_type="tool_call",
        tool_name=tool_name,
        tool_input={},
        tool_output=output,
        cost_usd=0.0,
        duration_ms=0,
        depth=0,
        parent_node_id=None,
        timestamp=_STARTED,
    )


class TestValidateToolOutputs:
    def test_passes_when_output_matches_schema(self) -> None:
        nodes = [_schema_node("n0", "search", {"results": ["a", "b"], "count": 2})]
        schemas = {
            "search": {
                "type": "object",
                "required": ["results", "count"],
                "properties": {
                    "results": {"type": "array", "items": {"type": "string"}},
                    "count": {"type": "integer"},
                },
            }
        }
        validate_tool_outputs(_trace(nodes, []), schemas)

    def test_passes_when_tool_not_in_schemas(self) -> None:
        """Tools without a schema entry are unconstrained."""
        nodes = [_schema_node("n0", "unknown_tool", {"anything": True})]
        validate_tool_outputs(_trace(nodes, []), {})

    def test_passes_on_empty_trace(self) -> None:
        validate_tool_outputs(_trace([], []), {"search": {"type": "object"}})

    def test_fails_on_missing_required_key(self) -> None:
        nodes = [_schema_node("n0", "search", {"count": 2})]
        schemas = {"search": {"type": "object", "required": ["results", "count"]}}
        with pytest.raises(AssertionError, match="missing required key 'results'"):
            validate_tool_outputs(_trace(nodes, []), schemas)

    def test_fails_on_wrong_type(self) -> None:
        nodes = [_schema_node("n0", "search", {"count": "not_a_number"})]
        schemas = {
            "search": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            }
        }
        with pytest.raises(AssertionError, match="expected type 'integer'"):
            validate_tool_outputs(_trace(nodes, []), schemas)

    def test_fails_on_wrong_root_type(self) -> None:
        nodes = [_schema_node("n0", "search", {})]
        # Cheat: output is a dict but schema expects array
        schemas = {"search": {"type": "array"}}
        with pytest.raises(AssertionError, match="expected type 'array'"):
            validate_tool_outputs(_trace(nodes, []), schemas)

    def test_validates_nested_properties(self) -> None:
        nodes = [
            _schema_node(
                "n0",
                "book",
                {
                    "booking": {"id": "abc", "status": 123},
                },
            )
        ]
        schemas = {
            "book": {
                "type": "object",
                "properties": {
                    "booking": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                        },
                    },
                },
            },
        }
        with pytest.raises(AssertionError, match="booking.status"):
            validate_tool_outputs(_trace(nodes, []), schemas)

    def test_validates_array_items(self) -> None:
        nodes = [_schema_node("n0", "search", {"results": ["ok", 42, "fine"]})]
        schemas = {
            "search": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        }
        with pytest.raises(AssertionError, match=r"results\[1\]"):
            validate_tool_outputs(_trace(nodes, []), schemas)

    def test_extra_keys_allowed_by_default(self) -> None:
        """Keys not in 'properties' are ignored — no additionalProperties check."""
        nodes = [_schema_node("n0", "search", {"results": [], "extra_field": True})]
        schemas = {
            "search": {
                "type": "object",
                "required": ["results"],
                "properties": {"results": {"type": "array"}},
            }
        }
        validate_tool_outputs(_trace(nodes, []), schemas)

    def test_multiple_violations_all_reported(self) -> None:
        nodes = [
            _schema_node("n0", "search", {"count": "bad"}),
            _schema_node("n1", "search", {}),
        ]
        schemas = {
            "search": {
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
            }
        }
        with pytest.raises(AssertionError, match="2 schema violation"):
            validate_tool_outputs(_trace(nodes, []), schemas)

    def test_error_includes_node_id_and_tool_name(self) -> None:
        nodes = [_schema_node("n42", "fetch", {"data": 123})]
        schemas = {
            "fetch": {"type": "object", "properties": {"data": {"type": "string"}}}
        }
        with pytest.raises(AssertionError, match="n42.*fetch"):
            validate_tool_outputs(_trace(nodes, []), schemas)

    def test_llm_call_nodes_are_skipped(self) -> None:
        """Only tool_call nodes are validated."""
        nodes = [_node("n0", tool_name=None, node_type="llm_call")]
        schemas = {"search": {"type": "object", "required": ["results"]}}
        validate_tool_outputs(_trace(nodes, []), schemas)
