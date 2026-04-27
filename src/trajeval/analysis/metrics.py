"""TRACE framework metrics for agent trajectory evaluation.

Layer rule: pure Python, no FastAPI imports.

The TRACE framework evaluates agent quality along four dimensions that
enterprises care about for compliance and reliability:

  Evidence Grounding — does the agent cite its sources?
    Proxy: fraction of LLM calls that have at least one tool_call predecessor
    in the execution graph.  An LLM call with no grounding context is a
    hallucination risk.

  Cognitive Quality — does the agent show reasoning?
    Proxy: fraction of nodes that involve reasoning (llm_call or
    state_transition type).  A high fraction indicates the agent is
    thinking rather than just calling tools mechanically.

  Process Efficiency — does the agent reach its goal without redundancy?
    Proxy: unique tool names / total tool call count.  Score of 1.0 means
    every tool was called exactly once; repeated calls lower the score.

  Step Economy — is the total path length proportionate to the task?
    Proxy: 1 / node_count (normalized to [0,1] via a configurable max_steps
    reference).  A score near 1.0 means the agent used few steps.

Usage::

    from trajeval.analysis.metrics import compute_metrics

    metrics = compute_metrics(trace)
    print(metrics.evidence_grounding)   # 0.0 – 1.0
    print(metrics.cognitive_quality)    # 0.0 – 1.0
    print(metrics.process_efficiency)   # 0.0 – 1.0
    print(metrics.step_economy)         # 0.0 – 1.0
"""

from __future__ import annotations

from dataclasses import dataclass

from trajeval.analysis.graph import build_graph
from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class TraceMetrics:
    """TRACE framework metric scores for a single trajectory.

    All scores are floats in [0.0, 1.0] where 1.0 is optimal.
    """

    evidence_grounding: float
    """Fraction of LLM calls preceded by at least one tool call in the graph."""

    cognitive_quality: float
    """Fraction of nodes that involve reasoning (llm_call or state_transition)."""

    process_efficiency: float
    """Unique tool names / total tool calls (1.0 = no redundant tool use)."""

    step_economy: float
    """Normalised inverse of node count; higher means fewer steps taken."""


def compute_metrics(trace: Trace, *, max_steps: int = 50) -> TraceMetrics:
    """Compute TRACE framework metrics for *trace*.

    Parameters
    ----------
    trace:
        The trajectory to evaluate.
    max_steps:
        Reference maximum node count for step_economy normalisation.
        Traces with more nodes than *max_steps* receive a step_economy of 0.0.
    """
    return TraceMetrics(
        evidence_grounding=_evidence_grounding(trace),
        cognitive_quality=_cognitive_quality(trace),
        process_efficiency=_process_efficiency(trace),
        step_economy=_step_economy(trace, max_steps=max_steps),
    )


# ---------------------------------------------------------------------------
# Individual metric computations
# ---------------------------------------------------------------------------


def _evidence_grounding(trace: Trace) -> float:
    """Fraction of LLM calls that have at least one tool_call predecessor."""
    llm_nodes = [n for n in trace.nodes if n.node_type == "llm_call"]
    if not llm_nodes:
        return 1.0  # no LLM calls → vacuously grounded

    g = build_graph(trace)
    node_by_id = {n.node_id: n for n in trace.nodes}  # O(n) once, O(1) lookups
    grounded = 0
    for llm_node in llm_nodes:
        has_tool_predecessor = any(
            node_by_id[pred_id].node_type == "tool_call"
            for pred_id in g.predecessors(llm_node.node_id)
            if pred_id in node_by_id
        )
        if has_tool_predecessor:
            grounded += 1

    return grounded / len(llm_nodes)


def _cognitive_quality(trace: Trace) -> float:
    """Fraction of nodes that are llm_call or state_transition."""
    if not trace.nodes:
        return 0.0
    reasoning = sum(
        1 for n in trace.nodes if n.node_type in ("llm_call", "state_transition")
    )
    return reasoning / len(trace.nodes)


def _process_efficiency(trace: Trace) -> float:
    """Unique tool names / total tool call count."""
    tool_calls = [n for n in trace.nodes if n.node_type == "tool_call"]
    if not tool_calls:
        return 1.0  # no tool calls → no redundancy
    unique_tools = len({n.tool_name for n in tool_calls})
    return unique_tools / len(tool_calls)


def _step_economy(trace: Trace, *, max_steps: int) -> float:
    """Normalised inverse of node count relative to *max_steps*."""
    n = len(trace.nodes)
    if n == 0:
        return 1.0
    # Linear decay: 0 nodes → 1.0, max_steps nodes → 0.0, beyond → 0.0
    return max(0.0, 1.0 - (n / max_steps))
