"""Auto-detect trace format and route to the correct adapter.

Inspects the JSON structure and picks the right adapter. Users never
need to know whether their trace came from OpenAI, OTel, LangGraph,
or native TrajEval — they just pass the JSON.

Detection rules (checked in order):
1. Has ``trace_id`` + ``nodes`` → native TrajEval (passthrough)
2. Has ``resourceSpans`` or ``scopeSpans`` → OTel GenAI
3. Is a list where items have ``event`` with ``on_tool_start`` → LangGraph events
4. Is a list where items have ``role`` → OpenAI messages
5. Is a dict with ``messages`` key containing a list → OpenAI messages (wrapped)
6. Else → error with hint

Usage::

    from trajeval.adapters.auto import auto_detect

    result = auto_detect(payload, agent_id="my-agent")
    trace = result.trace
"""

from __future__ import annotations

from trajeval.adapters.base import AdapterResult
from trajeval.sdk.models import Trace


def auto_detect(
    payload: object,
    *,
    agent_id: str = "unknown",
    version_hash: str = "unknown",
) -> AdapterResult:
    """Detect the format of *payload* and return an :class:`AdapterResult`.

    Parameters
    ----------
    payload:
        Parsed JSON — a dict or list.
    agent_id, version_hash:
        Passed through to the adapter for Trace construction.

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    if isinstance(payload, dict):
        return _detect_dict(payload, agent_id=agent_id, version_hash=version_hash)
    if isinstance(payload, list):
        return _detect_list(payload, agent_id=agent_id, version_hash=version_hash)
    raise ValueError(
        f"auto_detect: expected dict or list, got {type(payload).__name__}. "
        f"Supported formats: TrajEval native, OpenAI messages, "
        f"OTel GenAI spans, LangGraph events."
    )


def _detect_dict(
    data: dict[str, object],
    *,
    agent_id: str,
    version_hash: str,
) -> AdapterResult:
    # 1. Native TrajEval
    if "trace_id" in data and "nodes" in data:
        trace = Trace.model_validate(data)
        from trajeval.adapters.base import AdapterCapabilities

        return AdapterResult(
            trace=trace,
            capabilities=AdapterCapabilities(
                has_cost=trace.total_cost_usd > 0,
                has_latency=any(n.duration_ms > 0 for n in trace.nodes),
                has_reasoning=any(n.reasoning_text is not None for n in trace.nodes),
                has_structured_io=True,
                has_depth=any(n.depth > 0 for n in trace.nodes),
            ),
            warnings=[],
        )

    # 2. Claude Code log (timestamp-keyed dict of message entries)
    first_val = next(iter(data.values()), None) if data else None
    if (
        isinstance(first_val, list)
        and first_val
        and isinstance(first_val[0], dict)
        and "message" in first_val[0]
        and "uuid" in first_val[0]
    ):
        msgs = _normalize_claude_code_log(data)
        from trajeval.adapters.openai import from_openai_messages

        return from_openai_messages(
            msgs,
            agent_id=agent_id,
            version_hash=version_hash,
        )

    # 3. OTel
    if "resourceSpans" in data or "scopeSpans" in data:
        from trajeval.adapters.otel import from_otel_spans

        spans = data.get("resourceSpans") or data.get("scopeSpans")
        if not isinstance(spans, list):
            raise ValueError(
                "auto_detect: OTel format detected but resourceSpans/scopeSpans "
                "is not a list."
            )
        return from_otel_spans(spans, agent_id=agent_id, version_hash=version_hash)

    # 3. LangGraph thread dump (values.messages with type=human/ai/tool)
    if "values" in data and isinstance(data.get("values"), dict):
        values = data["values"]
        if isinstance(values.get("messages"), list):
            msgs = _normalize_langgraph_thread(values["messages"])
            from trajeval.adapters.openai import from_openai_messages

            return from_openai_messages(
                msgs,
                agent_id=agent_id,
                version_hash=version_hash,
            )

    # 4. Wrapped OpenAI messages (or Anthropic Messages API with content blocks)
    if "messages" in data and isinstance(data["messages"], list):
        from trajeval.adapters.openai import from_openai_messages

        messages = data["messages"]
        # Anthropic sends assistant content as a list of content blocks
        # ({"type": "tool_use", ...} / {"type": "text", ...}). Normalize
        # to OpenAI's tool_calls shape before handing off.
        if _looks_like_anthropic_messages(messages):
            messages = _normalize_anthropic_messages(messages)
        return from_openai_messages(
            messages,
            agent_id=agent_id,
            version_hash=version_hash,
            usage=data.get("usage"),  # type: ignore[arg-type]
        )

    raise ValueError(
        "auto_detect: unrecognized dict format. Expected one of: "
        "TrajEval native (has 'trace_id' + 'nodes'), "
        "OTel (has 'resourceSpans'), "
        "OpenAI (has 'messages'). "
        f"Got keys: {sorted(data.keys())[:10]}"
    )


def _detect_list(
    data: list[object],
    *,
    agent_id: str,
    version_hash: str,
) -> AdapterResult:
    if not data:
        raise ValueError("auto_detect: empty list — cannot determine format.")

    first = data[0]
    if not isinstance(first, dict):
        raise ValueError(
            f"auto_detect: list items must be dicts, got {type(first).__name__}."
        )

    # 4. LangGraph events
    if "event" in first and any(
        isinstance(item, dict) and str(item.get("event", "")).startswith("on_")
        for item in data[:5]
    ):
        from trajeval.adapters.langgraph import from_langgraph_events

        return from_langgraph_events(
            data,  # type: ignore[arg-type]
            agent_id=agent_id,
            version_hash=version_hash,
        )

    # 5. OpenAI messages array
    if "role" in first:
        from trajeval.adapters.openai import from_openai_messages

        return from_openai_messages(
            data,  # type: ignore[arg-type]
            agent_id=agent_id,
            version_hash=version_hash,
        )

    # 6. LangGraph thread messages (type=human/ai/tool instead of role)
    if "type" in first and first.get("type") in ("human", "ai", "tool", "system"):
        msgs = _normalize_langgraph_thread(data)
        from trajeval.adapters.openai import from_openai_messages

        return from_openai_messages(
            msgs,
            agent_id=agent_id,
            version_hash=version_hash,
        )

    raise ValueError(
        "auto_detect: unrecognized list format. Expected one of: "
        "LangGraph events (items have 'event'), "
        "OpenAI messages (items have 'role'), "
        "LangGraph thread (items have 'type'). "
        f"First item keys: {sorted(first.keys())[:10]}"
    )


def _normalize_langgraph_thread(
    messages: list[object],
) -> list[dict[str, object]]:
    """Convert LangGraph thread messages to OpenAI format.

    LangGraph threads use ``type`` (human/ai/tool) instead of ``role``
    (user/assistant/tool), and tool_calls have ``name``+``args`` instead
    of ``function: {name, arguments}``.
    """
    import json as _json

    type_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    normalized: list[dict[str, object]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_type = str(msg.get("type", ""))
        role = type_map.get(msg_type, msg_type)

        out: dict[str, object] = {"role": role}

        content = msg.get("content", "")
        if isinstance(content, list):
            out["content"] = " ".join(
                str(c.get("text", "")) if isinstance(c, dict) else str(c)
                for c in content
            )
        else:
            out["content"] = content

        # Convert tool_calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls and isinstance(tool_calls, list):
            out["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            _json.dumps(tc["args"])
                            if isinstance(tc.get("args"), dict)
                            else str(tc.get("args", "{}"))
                        ),
                    },
                }
                for i, tc in enumerate(tool_calls)
                if isinstance(tc, dict)
            ]

        # Tool response: add tool_call_id
        if role == "tool" and "tool_call_id" in msg:
            out["tool_call_id"] = msg["tool_call_id"]
        elif role == "tool" and "name" in msg:
            out["tool_call_id"] = f"call_{msg['name']}"

        normalized.append(out)

    return normalized


def _looks_like_anthropic_messages(messages: list[object]) -> bool:
    """Heuristic: any assistant message whose content is a list of
    dicts containing a ``tool_use`` or ``tool_result`` block → treat
    the whole conversation as Anthropic-shaped.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("tool_use", "tool_result"):
                return True
    return False


def _normalize_anthropic_messages(
    messages: list[object],
) -> list[dict[str, object]]:
    """Convert an Anthropic Messages API history to OpenAI shape.

    Anthropic assistant messages carry a list of content blocks
    (``text`` / ``tool_use``). Anthropic user messages may carry
    ``tool_result`` blocks instead of free text. Both get flattened
    into the OpenAI pattern: assistant with ``tool_calls`` + a
    follow-up ``{"role": "tool", "tool_call_id": ...}`` message per
    result.
    """
    import json as _json

    out: list[dict[str, object]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            if not isinstance(content, list):
                continue
            # Anthropic user messages can contain tool_result blocks OR text
            tool_results = [
                b
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if tool_results:
                for block in tool_results:
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = " ".join(
                            c.get("text", "")
                            for c in result_content
                            if isinstance(c, dict)
                        )
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": str(result_content),
                        }
                    )
                continue
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_parts:
                out.append({"role": "user", "content": " ".join(text_parts)})

        elif role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue
            if not isinstance(content, list):
                continue
            asst_text_parts: list[str] = []
            tool_calls: list[dict[str, object]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    asst_text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", f"call_{block.get('name', '')}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": _json.dumps(block.get("input", {})),
                            },
                        }
                    )
            msg_out: dict[str, object] = {"role": "assistant"}
            if tool_calls:
                msg_out["content"] = (
                    " ".join(asst_text_parts) if asst_text_parts else None
                )
                msg_out["tool_calls"] = tool_calls
            else:
                msg_out["content"] = " ".join(asst_text_parts)
            out.append(msg_out)

        elif role == "system":
            if isinstance(content, str):
                out.append({"role": "system", "content": content})

    return out


def _normalize_claude_code_log(
    data: dict[str, object],
) -> list[dict[str, object]]:
    """Convert Claude Code log format to OpenAI messages.

    Claude Code logs are timestamp-keyed dicts of entries, each with
    ``message.content`` containing Anthropic-style content blocks:
    ``tool_use`` (with name + input) and ``tool_result`` (with content).
    """
    import json as _json

    normalized: list[dict[str, object]] = []

    # Sort by timestamp, flatten entries
    for _ts in sorted(data.keys()):
        entries = data[_ts]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            msg = entry.get("message", {})
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", [])

            if role == "user":
                text = ""
                if isinstance(content, list):
                    text = " ".join(
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                elif isinstance(content, str):
                    text = content
                if text:
                    normalized.append({"role": "user", "content": text})

            elif role == "assistant":
                if not isinstance(content, list):
                    continue
                # Extract text + tool_use blocks
                text_parts = []
                tool_calls = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", f"call_{block.get('name', '')}"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": _json.dumps(block.get("input", {})),
                                },
                            }
                        )

                out: dict[str, object] = {"role": "assistant"}
                if tool_calls:
                    out["content"] = " ".join(text_parts) if text_parts else None
                    out["tool_calls"] = tool_calls
                else:
                    out["content"] = " ".join(text_parts)
                normalized.append(out)

                # Find matching tool_result entries
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                c.get("text", "")
                                for c in result_content
                                if isinstance(c, dict)
                            )
                        normalized.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(result_content),
                            }
                        )

    return normalized
