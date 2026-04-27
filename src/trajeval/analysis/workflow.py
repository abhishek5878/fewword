"""Workflow graph coverage — FlowBench-inspired.

Defines an *expected* workflow as a directed graph of tool transitions, then
measures how much of that graph a trace actually covers.

``must_visit`` checks individual tools.  ``workflow_coverage`` checks the
*transitions* between them — edge coverage is what separates "called the right
tools" from "called them in the right order and connected them correctly".

Optional edges
--------------
Real agent workflows have branches.  A booking agent may legitimately take
either ``search → cache_hit → book`` or ``search → validate → book``.
Marking the ``cache_hit`` path as optional prevents it from being reported
as a missing edge when the trace takes the ``validate`` path instead.

``WorkflowGraph`` accepts an ``optional_edges`` set.  Optional edges are
excluded from the denominator when computing ``edge_coverage`` and are
reported separately as ``optional_missing_edges`` rather than
``missing_required_edges``.  Node coverage is unaffected — nodes that appear
exclusively in optional edges are still counted as expected nodes.

``CoverageReport`` now separates:
- ``missing_required_edges`` — edges that *must* be present but aren't
- ``missing_optional_edges`` — optional edges that weren't taken
- ``edge_coverage`` — ``covered_required_edges / total_required_edges``
  (optional edges do not affect this score)

Usage::

    from trajeval.analysis.workflow import WorkflowGraph, workflow_coverage

    wf = WorkflowGraph(
        edges=[
            ("search_flights", "validate_seat"),
            ("validate_seat", "book_flight"),
            ("book_flight", "send_confirmation"),
        ],
        optional_edges=[
            ("search_flights", "cache_hit"),
            ("cache_hit", "book_flight"),
        ],
    )

    report = workflow_coverage(trace, wf)
    print(report.edge_coverage)            # only counts required edges
    print(report.missing_required_edges)   # must-have transitions absent
    print(report.missing_optional_edges)   # optional paths not taken (info only)

Layer rule: pure Python, no FastAPI imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class WorkflowGraph:
    """Expected workflow defined as a set of directed tool transitions.

    Parameters
    ----------
    edges:
        List of (source_tool, target_tool) pairs defining the expected
        transitions.  The node set is inferred as the union of all
        source and target tool names across both *edges* and *optional_edges*.
    optional_edges:
        (source_tool, target_tool) pairs that represent valid-but-not-required
        transitions (e.g. alternative branches, retry loops).  A trace missing
        an optional edge is NOT penalised in ``edge_coverage``.
    name:
        Optional human-readable label for error messages.

    Example::

        wf = WorkflowGraph(
            edges=[
                ("search", "validate"),
                ("validate", "book"),
            ],
            optional_edges=[
                ("search", "cache_hit"),
                ("cache_hit", "book"),
            ],
        )
        # Required path:  search → validate → book
        # Optional path:  search → cache_hit → book (doesn't affect coverage score)
    """

    edges: list[tuple[str, str]]
    optional_edges: list[tuple[str, str]] = field(default_factory=list)
    name: str = ""

    @property
    def nodes(self) -> frozenset[str]:
        """Required tool names (nodes that appear in required edges only).

        These are the nodes that count toward ``node_coverage``.  Nodes that
        appear exclusively in optional edges are excluded so that a trace
        taking only the required path is not penalised for missing
        optional-only tools.

        Use :attr:`all_nodes` when you need the full union.
        """
        result: set[str] = set()
        for src, tgt in self.edges:
            result.add(src)
            result.add(tgt)
        return frozenset(result)

    @property
    def all_nodes(self) -> frozenset[str]:
        """All tool names referenced across both required and optional edges."""
        result: set[str] = set()
        for src, tgt in self.edges:
            result.add(src)
            result.add(tgt)
        for src, tgt in self.optional_edges:
            result.add(src)
            result.add(tgt)
        return frozenset(result)

    @property
    def edge_set(self) -> frozenset[tuple[str, str]]:
        """Required edges only."""
        return frozenset(self.edges)

    @property
    def optional_edge_set(self) -> frozenset[tuple[str, str]]:
        """Optional edges only."""
        return frozenset(self.optional_edges)


@dataclass(frozen=True)
class CoverageReport:
    """Coverage of an expected workflow by an observed trace.

    Attributes
    ----------
    node_coverage:
        Fraction of expected nodes (required + optional) that appeared in the
        trace, in [0.0, 1.0].  ``covered_nodes / total_nodes``.
    edge_coverage:
        Fraction of *required* edges that appeared in the trace, in [0.0, 1.0].
        Optional edges are excluded from this calculation.
        ``len(covered_required_edges) / total_required_edges``.
    covered_nodes:
        Tool names that were expected (required or optional) and appeared.
    missing_nodes:
        Tool names that were expected but never called.
    covered_edges:
        All (source, target) pairs (required or optional) that were observed.
    missing_required_edges:
        Required (source, target) pairs that were expected but never observed.
        These represent genuine gaps in the observed trajectory.
    missing_optional_edges:
        Optional (source, target) pairs that were not taken.  These are
        informational only — they do not affect ``edge_coverage``.
    total_nodes:
        Total distinct tool names in the expected workflow.
    total_required_edges:
        Number of required directed edges in the workflow.

    .. note::
        ``missing_edges`` is a deprecated alias for ``missing_required_edges``
        kept for backwards compatibility.
    """

    node_coverage: float
    edge_coverage: float
    covered_nodes: frozenset[str]
    missing_nodes: frozenset[str]
    covered_edges: frozenset[tuple[str, str]]
    missing_required_edges: frozenset[tuple[str, str]]
    missing_optional_edges: frozenset[tuple[str, str]]
    total_nodes: int
    total_required_edges: int

    @property
    def missing_edges(self) -> frozenset[tuple[str, str]]:
        """Backwards-compatible alias for ``missing_required_edges``."""
        return self.missing_required_edges

    @property
    def total_edges(self) -> int:
        """Backwards-compatible alias for ``total_required_edges``."""
        return self.total_required_edges

    @property
    def fully_covered(self) -> bool:
        """True if every required node and required edge was observed."""
        return self.node_coverage == 1.0 and self.edge_coverage == 1.0


def workflow_coverage(trace: Trace, workflow: WorkflowGraph) -> CoverageReport:
    """Measure how much of *workflow* is covered by *trace*.

    Node coverage: fraction of expected tool names (required + optional) that
    appear at least once in the trace's ``tool_call`` nodes.

    Edge coverage: fraction of *required* (source, target) transitions that
    appear as *consecutive* tool calls in the trace.  Optional edges are
    tallied separately and do not affect the coverage score.

    "Consecutive" means the target immediately follows the source in the
    ordered list of tool_call nodes — ``llm_call`` and ``state_transition``
    nodes between them are transparent and do not break a transition.

    Parameters
    ----------
    trace:
        The observed trajectory.
    workflow:
        The expected workflow graph to measure coverage against.

    Returns
    -------
    CoverageReport
        Coverage fractions and missing required/optional edges.
    """
    required_nodes = workflow.nodes          # required-only node set
    required_edges = workflow.edge_set
    optional_edges = workflow.optional_edge_set

    # Observed tool names (tool_call nodes only, in order)
    tool_sequence = [
        n.tool_name
        for n in trace.nodes
        if n.node_type == "tool_call" and n.tool_name is not None
    ]

    # Node coverage — required nodes only; optional-only nodes never penalise
    observed_tool_names = frozenset(tool_sequence)
    covered_nodes = required_nodes & observed_tool_names
    missing_nodes = required_nodes - observed_tool_names

    # Observed consecutive transitions
    observed_transitions: frozenset[tuple[str, str]] = frozenset(
        (tool_sequence[i], tool_sequence[i + 1])
        for i in range(len(tool_sequence) - 1)
    )

    # Required edge coverage
    covered_required = required_edges & observed_transitions
    missing_required = required_edges - observed_transitions

    # Optional edge coverage (informational)
    covered_optional = optional_edges & observed_transitions
    missing_optional = optional_edges - observed_transitions

    # All covered edges (required + optional)
    covered_edges = covered_required | covered_optional

    total_nodes = len(required_nodes)
    total_required = len(required_edges)

    node_coverage = len(covered_nodes) / total_nodes if total_nodes > 0 else 1.0
    edge_coverage = len(covered_required) / total_required if total_required > 0 else 1.0

    return CoverageReport(
        node_coverage=node_coverage,
        edge_coverage=edge_coverage,
        covered_nodes=covered_nodes,
        missing_nodes=missing_nodes,
        covered_edges=covered_edges,
        missing_required_edges=missing_required,
        missing_optional_edges=missing_optional,
        total_nodes=total_nodes,
        total_required_edges=total_required,
    )


def workflow_satisfies(
    trace: Trace,
    workflow: WorkflowGraph,
    *,
    min_node_coverage: float = 1.0,
    min_edge_coverage: float = 1.0,
) -> None:
    """Assert that *trace* covers *workflow* above the given thresholds.

    Only *required* edges are checked against ``min_edge_coverage``.  Optional
    edges never cause an assertion failure.

    Parameters
    ----------
    trace:
        The observed trajectory.
    workflow:
        The expected workflow graph.
    min_node_coverage:
        Minimum required node coverage fraction (default 1.0 = full coverage).
    min_edge_coverage:
        Minimum required *required-edge* coverage fraction (default 1.0).

    Raises AssertionError listing missing required nodes and edges when
    coverage is below either threshold.
    """
    if not 0.0 <= min_node_coverage <= 1.0:
        raise ValueError(
            f"min_node_coverage must be in [0, 1], got {min_node_coverage}"
        )
    if not 0.0 <= min_edge_coverage <= 1.0:
        raise ValueError(
            f"min_edge_coverage must be in [0, 1], got {min_edge_coverage}"
        )

    report = workflow_coverage(trace, workflow)
    failures: list[str] = []
    label = f" ({workflow.name})" if workflow.name else ""

    if report.node_coverage < min_node_coverage:
        failures.append(
            f"node coverage {report.node_coverage:.0%} < {min_node_coverage:.0%} "
            f"— missing: {sorted(report.missing_nodes)}"
        )
    if report.edge_coverage < min_edge_coverage:
        failures.append(
            f"edge coverage {report.edge_coverage:.0%} < {min_edge_coverage:.0%} "
            f"— missing required transitions: {sorted(report.missing_required_edges)}"
        )

    if failures:
        raise AssertionError(f"workflow_satisfies{label}: " + "; ".join(failures))
