"""OpenTelemetry GenAI semantic conventions exporter for TrajEval traces.

Layer rule: SDK package.  Zero imports from trajeval.backend.

Converts a completed :class:`~trajeval.sdk.models.Trace` into OpenTelemetry
spans following the GenAI Semantic Conventions:
  https://opentelemetry.io/docs/specs/semconv/gen-ai/

Attribute usage:
  - ``gen_ai.system``, ``gen_ai.request.model``, ``gen_ai.operation.name``
    ``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens`` → span attrs
  - Prompt / completion text → span *events* (not attributes)
    Reason: span attributes are indexed and may be retained indefinitely by
    collectors.  Storing prompt text there creates a PII/compliance risk.
    Span events are ephemeral and can be filtered at the collector layer.

Usage::

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    from trajeval.sdk.otel import export_trace

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    export_trace(trace, tracer_provider=provider)
"""

from __future__ import annotations

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

from trajeval.sdk.models import Trace, TraceNode

# OTel instrumentation scope name
_TRACER_NAME = "trajeval"

# Attribute name constants drawn from GenAI semantic conventions
_ATTR_SYSTEM = gen_ai_attributes.GEN_AI_SYSTEM
_ATTR_REQUEST_MODEL = gen_ai_attributes.GEN_AI_REQUEST_MODEL
_ATTR_OPERATION = gen_ai_attributes.GEN_AI_OPERATION_NAME
_ATTR_INPUT_TOKENS = gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS
_ATTR_OUTPUT_TOKENS = gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS

# Span event names for prompt/completion content
_EVENT_PROMPT = "gen_ai.content.prompt"
_EVENT_COMPLETION = "gen_ai.content.completion"
_EVENT_TOOL_INPUT = "gen_ai.tool.input"
_EVENT_TOOL_OUTPUT = "gen_ai.tool.output"


def export_trace(
    trace: Trace,
    *,
    tracer_provider: TracerProvider | None = None,
) -> None:
    """Export *trace* as a hierarchy of OpenTelemetry spans.

    Creates one root span for the trace and one child span per node.
    LLM call nodes use GenAI semantic convention attributes.
    Tool inputs/outputs are stored in span *events*, not attributes,
    to avoid indexing PII.

    Parameters
    ----------
    trace:
        The trajectory to export.
    tracer_provider:
        An already-configured :class:`~opentelemetry.sdk.trace.TracerProvider`.
        If ``None``, the globally configured provider is used.
    """
    tracer = (tracer_provider or otel_trace.get_tracer_provider()).get_tracer(
        _TRACER_NAME
    )

    # Build a node_id → span mapping so child spans can reference parents
    with tracer.start_as_current_span(
        f"trajeval.trace/{trace.agent_id}",
        attributes={
            "trajeval.trace_id": trace.trace_id,
            "trajeval.agent_id": trace.agent_id,
            "trajeval.version_hash": trace.version_hash,
            "trajeval.total_cost_usd": trace.total_cost_usd,
            "trajeval.total_tokens": trace.total_tokens,
        },
    ) as root_span:
        _ = root_span  # root_span is the active context; child spans nest under it
        _id_to_span: dict[str, otel_trace.Span] = {}

        for node in trace.nodes:
            _emit_node_span(tracer, node, _id_to_span)


def _emit_node_span(
    tracer: otel_trace.Tracer,
    node: TraceNode,
    id_to_span: dict[str, otel_trace.Span],
) -> None:
    """Create and immediately end a span for *node*."""
    attrs: dict[str, str | int | float | bool] = {
        "trajeval.node_id": node.node_id,
        "trajeval.node_type": node.node_type,
        "trajeval.depth": node.depth,
        "trajeval.duration_ms": node.duration_ms,
        "trajeval.cost_usd": node.cost_usd,
    }

    if node.node_type == "llm_call":
        _add_llm_attributes(attrs, node)
    elif node.node_type == "tool_call" and node.tool_name:
        attrs["trajeval.tool_name"] = node.tool_name

    span_name = _span_name(node)

    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        id_to_span[node.node_id] = span

        if node.node_type == "tool_call":
            # Tool inputs/outputs → events (not attributes) to avoid PII indexing
            span.add_event(
                _EVENT_TOOL_INPUT,
                attributes={"input": str(node.tool_input)},
            )
            span.add_event(
                _EVENT_TOOL_OUTPUT,
                attributes={"output": str(node.tool_output)},
            )
        elif node.node_type == "llm_call":
            # Prompt/completion → events to avoid PII indexing
            if node.tool_input:
                span.add_event(
                    _EVENT_PROMPT,
                    attributes={"content": str(node.tool_input)},
                )
            if node.tool_output:
                span.add_event(
                    _EVENT_COMPLETION,
                    attributes={"content": str(node.tool_output)},
                )


def _add_llm_attributes(
    attrs: dict[str, str | int | float | bool], node: TraceNode
) -> None:
    """Populate GenAI semantic convention attributes for an LLM call node."""
    if node.tool_name:
        # tool_name for llm_call nodes stores the model identifier
        attrs[_ATTR_REQUEST_MODEL] = node.tool_name
        # Infer system from model name prefix
        if node.tool_name.startswith("gpt") or node.tool_name.startswith("o1"):
            attrs[_ATTR_SYSTEM] = "openai"
        elif node.tool_name.startswith("claude"):
            attrs[_ATTR_SYSTEM] = "anthropic"
        elif node.tool_name.startswith("gemini"):
            attrs[_ATTR_SYSTEM] = "google_vertexai"

    attrs[_ATTR_OPERATION] = "chat"

    # Token usage — extract from tool_output if present
    # Values are dict[str, object]; narrow to int/float before converting.
    output = node.tool_output
    if isinstance(output, dict):
        in_tok = output.get("input_tokens")
        if isinstance(in_tok, (int, float)):
            attrs[_ATTR_INPUT_TOKENS] = int(in_tok)
        out_tok = output.get("output_tokens")
        if isinstance(out_tok, (int, float)):
            attrs[_ATTR_OUTPUT_TOKENS] = int(out_tok)
        elif (total := output.get("tokens")) is not None and isinstance(
            total, (int, float)
        ):
            # Fallback: total tokens stored as single field
            attrs[_ATTR_OUTPUT_TOKENS] = int(total)


def _span_name(node: TraceNode) -> str:
    if node.node_type == "llm_call":
        model = node.tool_name or "unknown"
        return f"gen_ai chat {model}"
    if node.node_type == "tool_call":
        return f"tool {node.tool_name or 'unknown'}"
    return f"state_transition {node.node_id}"
