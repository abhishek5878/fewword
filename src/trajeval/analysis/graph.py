"""Trajectory graph analysis using NetworkX.

Layer rule: pure Python, no FastAPI imports.

This module provides the canonical function for building a NetworkX DiGraph
from a Trace, plus higher-level analyses (critical path, parallel branches).
The assertions in ``trajeval.assertions.core`` delegate to :func:`build_graph`
rather than building their own graphs.

Usage::

    from trajeval.analysis.graph import build_graph, critical_path, parallel_branches

    g = build_graph(trace)
    path = critical_path(trace)           # node IDs along the longest path
    branches = parallel_branches(trace)   # groups of nodes at the same depth
"""

from __future__ import annotations

import networkx as nx

from trajeval.sdk.models import Trace


def build_graph(trace: Trace) -> nx.DiGraph[str]:
    """Return a directed graph whose nodes and edges mirror *trace*.

    Each graph node is labelled with the TraceNode's ``node_id`` and carries
    all TraceNode fields as node attributes.  Each graph edge carries the
    ``edge_type`` attribute from the corresponding TraceEdge.
    """
    g: nx.DiGraph[str] = nx.DiGraph()
    for node in trace.nodes:
        g.add_node(node.node_id, **node.model_dump())
    for edge in trace.edges:
        g.add_edge(edge.source, edge.target, edge_type=edge.edge_type)
    return g


def critical_path(trace: Trace) -> list[str]:
    """Return the node IDs along the longest (by node count) path in the trace.

    "Longest path" is computed on the DAG of trace nodes weighted by hop count
    (not by duration).  Use this to find the deepest execution chain.

    Returns an empty list for traces with no nodes.  Raises :exc:`ValueError`
    if the trace graph contains a directed cycle (use :func:`no_cycles` first).
    """
    if not trace.nodes:
        return []

    g = build_graph(trace)

    try:
        path: list[str] = nx.dag_longest_path(g)
    except nx.NetworkXUnfeasible as exc:
        raise ValueError(
            f"critical_path: trace '{trace.trace_id}' contains a cycle — "
            "run no_cycles assertion first"
        ) from exc

    return path


def parallel_branches(trace: Trace) -> list[list[str]]:
    """Return groups of node IDs that run at the same depth with no ordering.

    Two nodes are considered parallel when they share the same ``depth`` value
    and there is no directed path between them in either direction.

    Returns a list of groups (each group is a list of node IDs).  Groups with
    only one member are excluded.  Returns an empty list for traces with fewer
    than two nodes.
    """
    if len(trace.nodes) < 2:
        return []

    g = build_graph(trace)

    # Bucket nodes by depth
    depth_buckets: dict[int, list[str]] = {}
    for node in trace.nodes:
        depth_buckets.setdefault(node.depth, []).append(node.node_id)

    groups: list[list[str]] = []
    for node_ids in depth_buckets.values():
        if len(node_ids) < 2:
            continue
        # Within the same depth bucket, collect nodes with no path between them
        parallel: list[str] = []
        for nid in node_ids:
            is_ordered = any(
                nx.has_path(g, nid, other) or nx.has_path(g, other, nid)
                for other in parallel
            )
            if not is_ordered:
                parallel.append(nid)
        if len(parallel) >= 2:
            groups.append(parallel)

    return groups
