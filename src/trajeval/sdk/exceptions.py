"""SDK-layer exceptions for TrajEval.

These exceptions are raised by the SDK at runtime and must remain
zero-dependency — no backend, no contract, no assertions imports.

Layer rule: pure Python, stdlib only.
"""

from __future__ import annotations


class TrajectoryInterceptionError(RuntimeError):
    """Raised by :class:`~trajeval.sdk.callback.TrajEvalCallback` in guard mode
    when a pre-execution assertion detects a policy violation before a tool call
    executes.

    LangGraph receives this as a tool failure.  A well-designed agent handles
    it as a recoverable error and chooses an alternative action.

    Attributes
    ----------
    violations:
        Human-readable description of every assertion that fired.
        More than one entry means multiple rules were violated simultaneously.
    tool_name:
        The name of the tool that was blocked.  May be ``None`` if the tool
        name could not be determined.
    trace_id:
        The trace ID of the in-flight trajectory at intercept time.
    """

    def __init__(
        self,
        violations: list[str],
        *,
        tool_name: str | None = None,
        trace_id: str = "",
    ) -> None:
        n = len(violations)
        violation_list = "\n".join(f"  [{i + 1}] {v}" for i, v in enumerate(violations))
        tool_label = f"'{tool_name}'" if tool_name else "unknown tool"
        message = (
            f"TrajEval guard blocked {tool_label}: "
            f"{n} rule(s) violated (trace_id='{trace_id}'):\n{violation_list}"
        )
        super().__init__(message)
        self.violations: list[str] = violations
        self.tool_name: str | None = tool_name
        self.trace_id: str = trace_id
