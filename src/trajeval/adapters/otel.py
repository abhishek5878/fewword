"""Adapter: OpenTelemetry GenAI spans → TrajEval Trace.

This is the *importer* side of TrajEval's OTel integration.
``sdk/otel.py`` exports a TrajEval Trace *to* OTel spans.
This module imports *from* OTel GenAI spans into a TrajEval Trace.

Together they close the round-trip: any framework that emits OTel spans
conforming to the GenAI semantic conventions can be evaluated by TrajEval
without any additional instrumentation.

OTel + TrajEval are complementary, not competing
-------------------------------------------------
OTel answers: *what happened operationally* (latency, errors, resource usage).
TrajEval answers: *was what happened correct* (contracts, reasoning, ordering).
A trace can be operationally clean (fast, no errors) and semantically broken
(wrong tool order, reasoning contradiction). This adapter is the bridge.

Targeting GenAI semantic conventions
-------------------------------------
This adapter targets the OpenTelemetry GenAI semantic conventions (draft):
https://opentelemetry.io/docs/specs/semconv/gen-ai/

By targeting the conventions rather than framework-specific attribute schemas,
any new framework that adopts the conventions works on day one. Supported
attribute keys:

  ``gen_ai.tool.name``             — identifies a tool call span
  ``gen_ai.operation.name``        — "execute_tool" also identifies tool spans
  ``gen_ai.request.model``         — identifies an LLM call span
  ``gen_ai.usage.input_tokens``    — input token count (for cost estimation)
  ``gen_ai.usage.output_tokens``   — output token count
  ``gen_ai.system``                — e.g. "openai", "anthropic"
  ``gen_ai.tool.call.id``          — tool call correlation ID
  ``gen_ai.tool.type``             — e.g. "function", "builtin"
  ``gen_ai.tool.call.arguments``   — JSON string of tool arguments (spec attr)
  ``gen_ai.tool.call.result``      — JSON string of tool result (spec attr)

Tool I/O recovery (in priority order):
1. Span events named ``gen_ai.tool.input`` / ``gen_ai.tool.output``
2. Span attributes ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result``
3. OpenLLMetry vendor attrs ``traceloop.entity.input`` / ``traceloop.entity.output``

This layered fallback means spans from any framework that follows the spec
(or openllmetry's widely-deployed implementation) will have structured I/O.

Input format
------------
Spans are represented as plain dicts — no OTel SDK dependency required::

    span = {
        "name": "tool search_flights",
        "span_id": "0000000000000001",
        "parent_span_id": None,
        "start_time_unix_nano": 1704067200_000_000_000,
        "end_time_unix_nano":   1704067200_120_000_000,
        "attributes": {
            "gen_ai.tool.name": "search_flights",
            "gen_ai.system": "openai",
        },
        "events": [
            {"name": "gen_ai.tool.input",
             "attributes": {"input": '{"origin": "NYC"}'}},
            {"name": "gen_ai.tool.output",
             "attributes": {"output": '{"flights": [...]}'}}
        ],
    }

Usage example::

    from trajeval.adapters.otel import from_otel_spans
    from trajeval.assertions.core import tool_must_precede

    result = from_otel_spans(spans, agent_id="booking-agent")
    print(result.capabilities.supported_analyses())
    tool_must_precede(result.trace, tool="validate_seat", before="book_flight")

Layer rule: SDK layer — zero backend imports.
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import Any

from trajeval.adapters.base import AdapterCapabilities, AdapterResult
from trajeval.sdk.models import Trace, TraceEdge, TraceNode

_TS_DEFAULT = "1970-01-01T00:00:00Z"
_NS_TO_MS = 1_000_000
_COST_PER_TOKEN_USD = 2e-6  # rough approximation; real cost depends on model


def from_otel_spans(
    spans: list[dict[str, Any]],
    *,
    agent_id: str = "otel-agent",
    version_hash: str = "unknown",
) -> AdapterResult:
    """Convert OTel GenAI spans to a TrajEval AdapterResult.

    Tool call spans (``gen_ai.tool.name`` present) and LLM call spans
    (``gen_ai.request.model`` present) are converted to TraceNodes.
    Spans with neither attribute are skipped with a warning.

    Parameters
    ----------
    spans:
        List of OTel span dicts. Must include at minimum ``attributes`` and
        either ``gen_ai.tool.name`` or ``gen_ai.request.model``.
        ``start_time_unix_nano`` / ``end_time_unix_nano`` are used for
        duration and timestamp if present.
    agent_id, version_hash:
        Trace metadata.
    """
    warnings: list[str] = []
    nodes: list[TraceNode] = []
    has_latency = False
    has_cost = False
    has_reasoning = False

    # Sort spans by start time to produce a deterministic ordering
    sorted_spans = sorted(
        spans, key=lambda s: s.get("start_time_unix_nano", 0)
    )

    # Build span_id → depth map from parent_span_id relationships.
    # Depth 0 = no parent; each hop adds 1.
    span_id_to_span: dict[str, dict[str, Any]] = {
        s.get("span_id", ""): s for s in spans if s.get("span_id")
    }
    _depth_cache: dict[str, int] = {}

    def _compute_depth(sid: str, visited: set[str] | None = None) -> int:
        if sid in _depth_cache:
            return _depth_cache[sid]
        if visited is None:
            visited = set()
        if sid in visited:  # cycle guard
            return 0
        visited.add(sid)
        s = span_id_to_span.get(sid)
        parent = s.get("parent_span_id") if s else None
        if parent and parent in span_id_to_span:
            d = 1 + _compute_depth(parent, visited)
        else:
            d = 0
        _depth_cache[sid] = d
        return d

    for span in sorted_spans:
        attrs = span.get("attributes", {})
        # Identify tool spans via gen_ai.tool.name OR operation.name="execute_tool"
        tool_name = (
            attrs.get("gen_ai.tool.name")
            or (
                attrs.get("gen_ai.tool.name")
                if attrs.get("gen_ai.operation.name") == "execute_tool"
                else None
            )
        )
        # If gen_ai.operation.name == "execute_tool" but gen_ai.tool.name is absent,
        # fall back to the span name (e.g. "execute_tool my_function" → "my_function")
        if not tool_name and attrs.get("gen_ai.operation.name") == "execute_tool":
            span_name_raw = span.get("name", "")
            tool_name = (
                span_name_raw.removeprefix("execute_tool ").strip() or "unknown_tool"
            )

        model_name = attrs.get("gen_ai.request.model")

        start_ns = span.get("start_time_unix_nano", 0)
        end_ns = span.get("end_time_unix_nano", 0)
        duration_ms = int((end_ns - start_ns) / _NS_TO_MS) if end_ns > start_ns else 0
        if duration_ms > 0:
            has_latency = True

        ts = _ns_to_iso(start_ns) if start_ns else _TS_DEFAULT

        input_tokens = int(attrs.get("gen_ai.usage.input_tokens", 0))
        output_tokens = int(attrs.get("gen_ai.usage.output_tokens", 0))
        cost_usd = (input_tokens + output_tokens) * _COST_PER_TOKEN_USD
        if cost_usd > 0:
            has_cost = True

        span_id = span.get("span_id", str(uuid.uuid4()))
        parent_span_id = span.get("parent_span_id")
        events = span.get("events", [])
        depth = _compute_depth(span_id)

        # Collect per-node metadata: all gen_ai.* and traceloop.* attributes
        # that don't map to a first-class TraceNode field.
        node_meta: dict[str, object] = {}
        for k, v in attrs.items():
            if k not in {
                "gen_ai.tool.name", "gen_ai.request.model",
                "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
                "gen_ai.operation.name",
            }:
                node_meta[k] = v

        if tool_name:
            inp, out = _extract_tool_io(events, attrs, warnings)
            nodes.append(
                TraceNode(
                    node_id=span_id,
                    node_type="tool_call",
                    tool_name=tool_name,
                    tool_input=inp,
                    tool_output=out,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                    depth=depth,
                    parent_node_id=parent_span_id,
                    timestamp=ts,
                    metadata=node_meta,
                )
            )
        elif model_name:
            reasoning = _extract_reasoning(events)
            if reasoning:
                has_reasoning = True
            nodes.append(
                TraceNode(
                    node_id=span_id,
                    node_type="llm_call",
                    tool_name=None,
                    tool_input={},
                    tool_output={"model": model_name},
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                    depth=depth,
                    parent_node_id=parent_span_id,
                    timestamp=ts,
                    reasoning_text=reasoning,
                    metadata=node_meta,
                )
            )
        else:
            span_name = span.get("name", "unknown")
            warnings.append(
                f"Skipped span '{span_name}': no gen_ai.tool.name or "
                "gen_ai.request.model attribute."
            )

    # Build sequential edges between consecutive tool_call nodes only
    tool_nodes = [n for n in nodes if n.node_type == "tool_call"]
    edges = [
        TraceEdge(
            source=tool_nodes[i].node_id,
            target=tool_nodes[i + 1].node_id,
            edge_type="sequential",
        )
        for i in range(len(tool_nodes) - 1)
    ]

    total_tokens = sum(
        int(s.get("attributes", {}).get("gen_ai.usage.input_tokens", 0))
        + int(s.get("attributes", {}).get("gen_ai.usage.output_tokens", 0))
        for s in spans
    )

    trace = Trace(
        trace_id=str(uuid.uuid4()),
        agent_id=agent_id,
        version_hash=version_hash,
        started_at=nodes[0].timestamp if nodes else _TS_DEFAULT,
        completed_at=nodes[-1].timestamp if nodes else _TS_DEFAULT,
        nodes=nodes,
        edges=edges,
        total_cost_usd=sum(n.cost_usd for n in nodes),
        total_tokens=total_tokens,
    )

    has_depth = any(n.depth > 0 for n in nodes)
    capabilities = AdapterCapabilities(
        has_cost=has_cost,
        has_reasoning=has_reasoning,
        has_structured_io=True,
        has_latency=has_latency,
        has_depth=has_depth,
    )

    return AdapterResult(trace=trace, capabilities=capabilities, warnings=warnings)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ns_to_iso(ns: int) -> str:
    """Convert nanoseconds-since-epoch to ISO 8601 string."""
    dt = datetime.datetime.fromtimestamp(ns / 1e9, tz=datetime.UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_tool_io(
    events: list[dict[str, Any]],
    attrs: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Extract tool input/output using a three-tier fallback strategy.

    Priority order:
    1. Span events ``gen_ai.tool.input`` / ``gen_ai.tool.output`` (OTel spec)
    2. Span attributes ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result``
       (OTel spec, alternative placement)
    3. OpenLLMetry vendor attributes ``traceloop.entity.input`` /
       ``traceloop.entity.output`` (widely deployed, not in spec)

    This ensures spans from any compliant framework or openllmetry-instrumented
    framework produce structured tool_input / tool_output.
    """
    inp: dict[str, object] = {}
    out: dict[str, object] = {"status": "ok"}

    # Tier 1: span events (gen_ai.tool.input / gen_ai.tool.output)
    for ev in events:
        ev_name = ev.get("name", "")
        ev_attrs = ev.get("attributes", {})
        if "input" in ev_name:
            raw = ev_attrs.get("input") or ev_attrs.get("gen_ai.event.content", "")
            if raw:
                inp = _try_parse_dict(raw, warnings)
        elif "output" in ev_name or "result" in ev_name:
            raw = ev_attrs.get("output") or ev_attrs.get("gen_ai.event.content", "")
            if raw:
                out = _try_parse_dict(raw, warnings) or {"status": "ok"}

    # Tier 2: span attributes (gen_ai.tool.call.arguments / .result)
    if not inp and attrs.get("gen_ai.tool.call.arguments"):
        inp = _try_parse_dict(attrs["gen_ai.tool.call.arguments"], warnings)
    if out == {"status": "ok"} and attrs.get("gen_ai.tool.call.result"):
        parsed = _try_parse_dict(attrs["gen_ai.tool.call.result"], warnings)
        out = parsed or {"status": "ok"}

    # Tier 3: openllmetry vendor attributes (traceloop.entity.input / .output)
    if not inp and attrs.get("traceloop.entity.input"):
        inp = _try_parse_dict(attrs["traceloop.entity.input"], warnings)
    if out == {"status": "ok"} and attrs.get("traceloop.entity.output"):
        parsed = _try_parse_dict(attrs["traceloop.entity.output"], warnings)
        out = parsed or {"status": "ok"}

    return inp, out


def _extract_reasoning(events: list[dict[str, Any]]) -> str | None:
    """Extract LLM reasoning text from span events if present."""
    for ev in events:
        ev_name = ev.get("name", "")
        if "completion" in ev_name or "output" in ev_name:
            content = ev.get("attributes", {}).get("content", "")
            if isinstance(content, str) and content:
                return content
    return None


def _try_parse_dict(
    raw: object, warnings: list[str]
) -> dict[str, object]:
    """Parse raw value to dict; return empty dict on failure."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except (json.JSONDecodeError, ValueError):
            warnings.append(f"Could not parse JSON from span event: {raw!r:.80}")
    return {}
