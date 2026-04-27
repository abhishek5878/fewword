"""Adapter: OpenAI messages array → TrajEval Trace.

Converts a standard OpenAI Chat Completions ``messages`` list into a
TrajEval Trace without requiring any changes to the user's agent code.

Input format
------------
The ``messages`` parameter is the list users pass to and receive from the
Chat Completions API::

    messages = [
        {"role": "user", "content": "Book me a flight to Tokyo"},
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "arguments": '{"origin": "NYC", "destination": "NRT"}'
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"flights": [{"id": "UA123", "price": 850}]}'
        },
        ...
    ]

Usage example::

    from trajeval.adapters.openai import from_openai_messages
    import functools
    from trajeval.assertions.core import tool_must_precede

    result = from_openai_messages(messages, agent_id="booking-agent")
    tool_must_precede(result.trace, tool="validate_seat", before="book_flight")

Capabilities
------------
- ``has_structured_io`` = True (arguments and tool results are parsed as JSON)
- ``has_cost`` = True only when ``usage`` dict is provided
- ``has_latency`` = False (timestamps not in message history)
- ``has_reasoning`` = False (reasoning_text not in standard tool_call messages)
- ``has_depth`` = False (flat message structure; all nodes at depth 0)

Layer rule: SDK layer — zero backend imports.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from trajeval.adapters.base import AdapterCapabilities, AdapterResult
from trajeval.sdk.models import Trace, TraceEdge, TraceNode

_TS_DEFAULT = "1970-01-01T00:00:00Z"
# Rough gpt-4o token cost; overridden if usage dict provides actuals
_COST_PER_TOKEN_USD = 2e-6


def from_openai_messages(
    messages: list[dict[str, Any]],
    *,
    agent_id: str = "openai-agent",
    version_hash: str = "unknown",
    usage: dict[str, int] | None = None,
) -> AdapterResult:
    """Convert an OpenAI messages array to a TrajEval AdapterResult.

    Extracts tool calls from ``role=assistant`` messages and matches each with
    its result from the corresponding ``role=tool`` message.  Non-tool messages
    (user, system) are ignored.

    Parameters
    ----------
    messages:
        Full conversation messages list (roles: user / assistant / tool).
    agent_id:
        Agent identifier for the resulting Trace.
    version_hash:
        Version identifier (e.g. git SHA or model name).
    usage:
        Optional usage object from the API response, e.g.
        ``{"prompt_tokens": 512, "completion_tokens": 128}``.
        When provided, cost_usd is estimated from total tokens and
        ``has_cost`` is set to True in the returned capabilities.
    """
    warnings: list[str] = []

    # Streaming detection: reject incomplete streamed messages early.
    # In streaming mode (stream=True), the API emits delta chunks — dicts with
    # a "delta" key inside choices[].  from_openai_messages needs assembled
    # messages (streaming=False), not raw chunks.  Passing streaming chunks
    # produces 0-node Traces with no warning, which is silently wrong.
    for i, msg in enumerate(messages):
        if "delta" in msg:
            raise ValueError(
                f"messages[{i}] looks like a streaming chunk (has 'delta' key). "
                "from_openai_messages requires assembled messages from a "
                "non-streaming response (stream=False). "
                "To use streaming, collect chunks and assemble them with "
                "openai.lib.streaming.AssistantStreamManager or equivalent "
                "before passing to from_openai_messages."
            )
        choices = msg.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict) and "delta" in choice:
                    raise ValueError(
                        f"messages[{i}] looks like a streaming response object "
                        "(choices[].delta present). from_openai_messages requires "
                        "assembled role/content/tool_calls messages from a "
                        "non-streaming call (stream=False), not raw stream chunks."
                    )

    # Map tool_call_id → list of parsed outputs (FIFO).
    #
    # Some agent runtimes (notably tau-bench's historical_trajectories)
    # reuse the same tool_call_id across multiple tool invocations in a
    # single conversation. A naive dict-by-id would silently let the
    # second response overwrite the first, leaving the first call's
    # node with whatever the duplicate id mapped to last (often an
    # error string from a different tool entirely). Storing per-id FIFO
    # queues and consuming positionally preserves correct shape on both
    # well-formed (unique-id) and id-reusing trajectories.
    tool_output_queues: dict[str, list[dict[str, object]]] = {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id", "")
        raw = msg.get("content", "{}")
        parsed = _parse_json_field(raw, warnings, context=call_id)
        tool_output_queues.setdefault(call_id, []).append(parsed)

    # Count total function-type tool calls for cost distribution
    all_tool_calls = [
        tc
        for msg in messages
        if msg.get("role") == "assistant"
        for tc in (msg.get("tool_calls") or [])
        if tc.get("type", "function") == "function"
    ]
    total_calls = max(len(all_tool_calls), 1)

    # Estimate per-call cost when usage object provided
    total_tokens = 0
    if usage:
        total_tokens = usage.get("prompt_tokens", 0) + usage.get(
            "completion_tokens", 0
        )
    cost_per_call = (
        (total_tokens * _COST_PER_TOKEN_USD / total_calls) if total_tokens else 0.0
    )

    # Build nodes from assistant tool_call entries.
    # Track the most recent user message so each tool_call node can carry the
    # preceding_user_text in its metadata — consumed by the
    # require_user_consent_before assertion (HITL-text check).
    nodes: list[TraceNode] = []
    last_user_text = ""
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, str):
                last_user_text = content
            elif isinstance(content, list):
                # Anthropic-style content blocks — collect text parts.
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                last_user_text = " ".join(parts)
            continue
        if role != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if tc.get("type", "function") != "function":
                continue
            fn = tc.get("function", {})
            name = fn.get("name", "unknown_tool")
            call_id = tc.get("id", str(uuid.uuid4()))
            inp = _parse_json_field(
                fn.get("arguments", "{}"), warnings, context=name
            )
            queue = tool_output_queues.get(call_id)
            out = queue.pop(0) if queue else {"status": "ok"}
            nodes.append(
                TraceNode(
                    node_id=str(uuid.uuid4()),
                    node_type="tool_call",
                    tool_name=name,
                    tool_input=inp,
                    tool_output=out,
                    cost_usd=cost_per_call,
                    duration_ms=0,
                    depth=0,
                    parent_node_id=None,
                    timestamp=_TS_DEFAULT,
                    metadata={"preceding_user_text": last_user_text},
                )
            )

    if not nodes:
        warnings.append(
            "No tool_call nodes found in message history. "
            "Check that assistant messages include 'tool_calls' entries."
        )

    edges = [
        TraceEdge(
            source=nodes[i].node_id,
            target=nodes[i + 1].node_id,
            edge_type="sequential",
        )
        for i in range(len(nodes) - 1)
    ]

    trace = Trace(
        trace_id=str(uuid.uuid4()),
        agent_id=agent_id,
        version_hash=version_hash,
        started_at=_TS_DEFAULT,
        completed_at=_TS_DEFAULT,
        nodes=nodes,
        edges=edges,
        total_cost_usd=sum(n.cost_usd for n in nodes),
        total_tokens=total_tokens,
    )

    capabilities = AdapterCapabilities(
        has_cost=bool(total_tokens),
        has_reasoning=False,
        has_structured_io=True,
        has_latency=False,
        has_depth=False,
    )

    return AdapterResult(trace=trace, capabilities=capabilities, warnings=warnings)


def _parse_json_field(
    raw: object,
    warnings: list[str],
    *,
    context: str,
) -> dict[str, object]:
    """Parse a JSON string into a dict; fall back to wrapping in 'raw' key."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except (json.JSONDecodeError, ValueError):
            warnings.append(
                f"Could not parse JSON for '{context}': {raw!r:.80}"
            )
            return {"raw": raw}
    return {"result": raw}
