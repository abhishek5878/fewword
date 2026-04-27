"""Fault injection — test agent resilience to tool failures.

Answers the question: "What happens when search returns a 500?
When the booking API times out? When the LLM returns garbage?"

ReliabilityBench (Jan 2026) showed that agents with 96.9% single-run
success drop to 88.1% under minor perturbations. This module lets you
quantify that drop for *your* agent, against *your* failure modes,
before production discovers them for you.

Two modes:

1. **Trace-level fault injection**: take a recorded trace, inject faults
   into specific tool outputs, re-run assertions to see what breaks.
   No LLM calls needed — pure structural analysis.

2. **Fault scenarios**: predefined failure patterns (error response,
   empty response, timeout marker, partial data) applied systematically
   across every tool call in a trace.

Usage::

    from trajeval.analysis.fault_injection import (
        inject_fault, inject_all, FaultType, FaultReport,
    )

    # Inject a single fault at node n3
    mutant = inject_fault(trace, node_id="n3", fault=FaultType.ERROR)

    # Inject faults at every tool call and report which assertions break
    report = inject_all(trace, assertions=[my_assertion], faults=[FaultType.ERROR])
    print(f"{report.broken_count}/{report.total_scenarios} scenarios broke")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from trajeval.sdk.models import Trace


class FaultType(Enum):
    """Predefined fault injection patterns."""

    ERROR = "error"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


_FAULT_OUTPUTS: dict[FaultType, dict[str, object]] = {
    FaultType.ERROR: {
        "error": "injected_fault",
        "error_code": 500,
        "message": "Internal server error (fault injection)",
    },
    FaultType.EMPTY: {},
    FaultType.TIMEOUT: {
        "error": "timeout",
        "error_code": 408,
        "message": "Request timed out (fault injection)",
    },
    FaultType.PARTIAL: {
        "partial": True,
        "message": "Incomplete response (fault injection)",
    },
}


def inject_fault(
    trace: Trace,
    *,
    node_id: str,
    fault: FaultType,
) -> Trace:
    """Return a copy of *trace* with the specified node's output replaced.

    The original trace is not modified (frozen Pydantic model).
    """
    fault_output = _FAULT_OUTPUTS[fault]
    new_nodes = []
    found = False
    for node in trace.nodes:
        if node.node_id == node_id:
            new_nodes.append(
                node.model_copy(update={"tool_output": dict(fault_output)})
            )
            found = True
        else:
            new_nodes.append(node)
    if not found:
        msg = f"inject_fault: node_id={node_id!r} not found in trace"
        raise ValueError(msg)
    return trace.model_copy(update={"nodes": new_nodes})


def inject_fault_at_tool(
    trace: Trace,
    *,
    tool_name: str,
    fault: FaultType,
) -> Trace:
    """Inject a fault into every call to *tool_name*."""
    fault_output = _FAULT_OUTPUTS[fault]
    new_nodes = [
        (
            node.model_copy(update={"tool_output": dict(fault_output)})
            if node.node_type == "tool_call" and node.tool_name == tool_name
            else node
        )
        for node in trace.nodes
    ]
    return trace.model_copy(update={"nodes": new_nodes})


# ---------------------------------------------------------------------------
# Systematic injection + assertion evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultScenario:
    """One injected-fault scenario and its result."""

    node_id: str
    tool_name: str | None
    fault: FaultType
    assertion_name: str
    broken: bool
    error_message: str | None


@dataclass(frozen=True)
class FaultReport:
    """Aggregate fault injection results across all scenarios."""

    trace_id: str
    scenarios: list[FaultScenario] = field(default_factory=list)

    @property
    def total_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def broken_count(self) -> int:
        return sum(1 for s in self.scenarios if s.broken)

    @property
    def resilient_count(self) -> int:
        return self.total_scenarios - self.broken_count

    @property
    def resilience_rate(self) -> float:
        if not self.scenarios:
            return 1.0
        return self.resilient_count / self.total_scenarios

    def summary(self) -> str:
        lines = [
            f"Fault injection: {self.broken_count}/{self.total_scenarios} "
            f"scenarios broke (resilience: {self.resilience_rate:.0%})"
        ]
        for s in self.scenarios:
            if s.broken:
                lines.append(
                    f"  BROKE {s.node_id} ({s.tool_name}) "
                    f"[{s.fault.value}] → {s.assertion_name}: "
                    f"{s.error_message}"
                )
        return "\n".join(lines)


def inject_all(
    trace: Trace,
    *,
    assertions: list[tuple[str, Callable[[Trace], None]]],
    faults: list[FaultType] | None = None,
) -> FaultReport:
    """Inject every fault type at every tool call and evaluate all assertions.

    Parameters
    ----------
    trace:
        The baseline (passing) trace to inject faults into.
    assertions:
        List of ``(name, fn)`` pairs — same shape as the CLI.
    faults:
        Fault types to inject. Defaults to all four.

    Returns
    -------
    FaultReport with one :class:`FaultScenario` per
    (node × fault × assertion) combination.
    """
    _faults = faults or list(FaultType)
    tool_nodes = [n for n in trace.nodes if n.node_type == "tool_call"]
    scenarios: list[FaultScenario] = []

    for node in tool_nodes:
        for fault in _faults:
            mutant = inject_fault(trace, node_id=node.node_id, fault=fault)
            for assertion_name, assertion_fn in assertions:
                broken = False
                error_msg: str | None = None
                try:
                    assertion_fn(mutant)
                except AssertionError as exc:
                    broken = True
                    error_msg = str(exc)
                scenarios.append(
                    FaultScenario(
                        node_id=node.node_id,
                        tool_name=node.tool_name,
                        fault=fault,
                        assertion_name=assertion_name,
                        broken=broken,
                        error_message=error_msg,
                    )
                )

    return FaultReport(trace_id=trace.trace_id, scenarios=scenarios)
