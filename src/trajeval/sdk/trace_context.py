"""W3C Trace Context propagation for cross-process multi-agent tracing.

Implements the W3C traceparent header format so TrajEval traces can cross
process and machine boundaries — e.g., when an orchestrator calls a subagent
deployed as a separate HTTP service.

Header format::

    traceparent: 00-{trace_id}-{parent_id}-{flags}

Where:
- ``trace_id``  — 32 lowercase hex chars (128-bit), the TrajEval trace_id
  with hyphens stripped.
- ``parent_id`` — 16 lowercase hex chars (64-bit), the :attr:`Trace.span_id`
  of the agent that made the outbound call.
- ``flags``     — 2 hex chars; ``01`` = sampled.

Usage — orchestrator side::

    ctx = TraceContext.from_callback(orchestrator_callback)
    headers = {"traceparent": ctx.to_header()}
    response = httpx.post("https://search-agent.fly.dev/run",
                          json=payload, headers=headers)

Usage — subagent side::

    ctx = TraceContext.from_header(request.headers["traceparent"])
    callback = TrajEvalCallback(
        agent_id="search-specialist",
        parent_trace_context=ctx,
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# W3C spec requires exactly this pattern
_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def _normalize_trace_id(trace_id: str) -> str:
    """Strip hyphens from a UUID and lowercase — produces 32 hex chars."""
    return trace_id.replace("-", "").lower()


def _normalize_span_id(span_id: str) -> str:
    """Ensure the span_id is exactly 16 lowercase hex chars.

    Truncates longer values; left-pads shorter values with zeros.
    """
    clean = span_id.replace("-", "").lower()
    return clean[:16].ljust(16, "0")


@dataclass(frozen=True)
class TraceContext:
    """Parsed W3C ``traceparent`` header — carrier for cross-agent linking.

    Attributes
    ----------
    trace_id:
        32 lowercase hex chars.  Corresponds to the *parent* agent's
        ``Trace.trace_id`` (hyphens stripped).
    parent_id:
        16 lowercase hex chars.  Corresponds to the *parent* agent's
        ``Trace.span_id``.
    flags:
        2 hex chars; ``"01"`` = sampled (default).
    """

    trace_id: str
    parent_id: str
    flags: str = "01"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_header(cls, header: str) -> TraceContext:
        """Parse a W3C ``traceparent`` header string.

        Raises :exc:`ValueError` if the header is malformed.
        """
        m = _TRACEPARENT_RE.match(header.strip())
        if not m:
            raise ValueError(
                f"Invalid traceparent header: {header!r}. "
                "Expected format: 00-<32hex>-<16hex>-<2hex>"
            )
        return cls(trace_id=m.group(1), parent_id=m.group(2), flags=m.group(3))

    @classmethod
    def from_ids(
        cls,
        *,
        trace_id: str,
        span_id: str,
        flags: str = "01",
    ) -> TraceContext:
        """Build a :class:`TraceContext` from raw trace_id and span_id strings.

        Normalizes both IDs to meet W3C spec requirements.
        """
        return cls(
            trace_id=_normalize_trace_id(trace_id),
            parent_id=_normalize_span_id(span_id),
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_header(self) -> str:
        """Serialize to a W3C ``traceparent`` header value."""
        return f"00-{self.trace_id}-{self.parent_id}-{self.flags}"
