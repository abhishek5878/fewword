"""Schema-first pre-flight checks — validate before any trace exists.

The cold-start problem (Gap 17): when you deploy a new agent with zero
traces, TrajEval has nothing to evaluate.  Pre-flight checks fix this by
analyzing the agent's *declared* tool list against the active assertion
set and surfacing contradictions before the first run.

Examples of what pre-flight catches:
- Agent exposes ``delete_user`` but a contract says ``never_calls:delete_user``
- ``must_visit`` requires ``validate`` but the agent doesn't register it
- ``tool_must_precede`` references ``confirm`` which isn't in the tool list
- ``no_retry_storm`` is active but ``max_retries`` in config suggests retries
  are expected (advisory warning, not a conflict)

Pre-flight is advisory — warnings, not errors. The agent might still behave
correctly if a tool is registered dynamically at runtime. But if pre-flight
finds conflicts, the developer knows to investigate before deploying.

Usage::

    from trajeval.preflight import preflight_check, PreFlightWarning

    report = preflight_check(
        tools=["search", "book", "charge_card"],
        assertions=[
            ("never_calls:delete_user", partial(never_calls, tool="delete_user")),
            ("must_visit:validate", partial(must_visit, tools=["validate"])),
        ],
    )
    for w in report.warnings:
        print(f"[{w.severity}] {w.message}")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from trajeval.sdk.models import Trace, TraceNode


@dataclass(frozen=True)
class PreFlightWarning:
    """A single pre-flight finding."""

    severity: Literal["conflict", "advisory"]
    assertion_name: str
    message: str


@dataclass(frozen=True)
class PreFlightReport:
    """Result of a pre-flight check."""

    tools: list[str]
    warnings: list[PreFlightWarning] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return len(self.warnings) == 0

    @property
    def conflicts(self) -> list[PreFlightWarning]:
        return [w for w in self.warnings if w.severity == "conflict"]

    def summary(self) -> str:
        if self.clean:
            return f"Pre-flight: {len(self.tools)} tools, no warnings."
        lines = [
            f"Pre-flight: {len(self.tools)} tools, "
            f"{len(self.warnings)} warning(s):"
        ]
        for w in self.warnings:
            lines.append(f"  [{w.severity.upper()}] {w.assertion_name}: {w.message}")
        return "\n".join(lines)


def preflight_check(
    tools: list[str],
    assertions: list[tuple[str, Callable[[Trace], None]]],
) -> PreFlightReport:
    """Run pre-flight checks on a declared tool list against assertions.

    Builds a synthetic single-node trace per tool and runs each assertion
    to detect conflicts. Also inspects assertion names for ``must_visit``
    and ``tool_must_precede`` patterns that reference tools not in the list.

    Parameters
    ----------
    tools:
        The agent's declared/registered tool names.
    assertions:
        List of ``(name, fn)`` pairs — same shape as CLI's ``_parse_assertions``
        output.
    """
    tool_set = set(tools)
    warnings: list[PreFlightWarning] = []

    for name, _fn in assertions:
        parts = name.split(":")

        # --- never_calls: tool is in the agent's list ---
        if parts[0] == "never_calls" and len(parts) > 1:
            banned = parts[1]
            if banned in tool_set:
                warnings.append(PreFlightWarning(
                    severity="conflict",
                    assertion_name=name,
                    message=(
                        f"Agent registers tool '{banned}' but assertion "
                        f"bans it. The agent will always fail this check "
                        f"if it calls '{banned}'."
                    ),
                ))

        # --- must_visit: required tool not registered ---
        elif parts[0] == "must_visit" and len(parts) > 1:
            required = parts[1].split(",")
            missing = [t for t in required if t not in tool_set]
            if missing:
                warnings.append(PreFlightWarning(
                    severity="conflict",
                    assertion_name=name,
                    message=(
                        f"Required tool(s) {missing} not in agent's "
                        f"registered tools {sorted(tool_set)}. "
                        f"The agent cannot satisfy this assertion."
                    ),
                ))

        # --- tool_must_precede: referenced tools not registered ---
        elif parts[0] == "tool_must_precede" and len(parts) > 1:
            tool_a = parts[1]
            tool_b = (
                parts[2].split("=")[1]
                if len(parts) > 2 and "=" in parts[2]
                else (parts[2] if len(parts) > 2 else None)
            )
            for t in [tool_a, tool_b]:
                if t is not None and t not in tool_set:
                    warnings.append(PreFlightWarning(
                        severity="advisory",
                        assertion_name=name,
                        message=(
                            f"Tool '{t}' referenced in ordering constraint "
                            f"but not in agent's registered tools. "
                            f"Constraint may be vacuously satisfied."
                        ),
                    ))

        # --- no_retry_storm: advisory if agent config suggests retries ---
        elif parts[0] == "no_retry_storm":
            warnings.append(PreFlightWarning(
                severity="advisory",
                assertion_name=name,
                message=(
                    "Retry storm detection active. Ensure the agent's "
                    "retry/backoff logic varies arguments between retries."
                ),
            ))

    # --- Synthetic trace probe: run each assertion against a minimal trace
    #     containing all declared tools to catch unexpected failures ---
    if tools:
        probe_nodes = [
            TraceNode(
                node_id=f"preflight-{i}",
                node_type="tool_call",
                tool_name=t,
                tool_input={},
                tool_output={},
                timestamp="1970-01-01T00:00:00Z",
            )
            for i, t in enumerate(tools)
        ]
        probe_trace = Trace(
            trace_id="preflight-probe",
            agent_id="preflight",
            version_hash="preflight",
            started_at="1970-01-01T00:00:00Z",
            completed_at="1970-01-01T00:00:00Z",
            nodes=probe_nodes,
            edges=[],
        )
        for name, fn in assertions:
            try:
                fn(probe_trace)
            except AssertionError as exc:
                # Only add if not already caught by name-based checks above
                existing = {w.assertion_name for w in warnings}
                if name not in existing:
                    warnings.append(PreFlightWarning(
                        severity="conflict",
                        assertion_name=name,
                        message=f"Probe trace failed: {exc}",
                    ))
            except Exception:  # noqa: S110, S112
                continue  # non-assertion errors in probe are not pre-flight concerns

    return PreFlightReport(tools=list(tools), warnings=warnings)
