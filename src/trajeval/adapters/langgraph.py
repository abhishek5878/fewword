"""Adapter: LangGraph astream_events v2 → TrajEval Trace.

Converts a LangGraph event stream — either as a JSONL string saved from a
production run, or as a list of in-memory event dicts — into a TrajEval
Trace for assertion evaluation.

The key value: users can save their LangGraph event streams to disk
(``async for event in graph.astream_events(..., version="v2"): log(event)``)
and replay them through TrajEval *without* re-running the agent.  The
``from_langgraph_jsonl`` path is the "I already have logs" use case.

Event format (LangGraph astream_events v2)
------------------------------------------
Each event is a JSON object. TrajEval processes four event types::

    # Tool start
    {"event": "on_tool_start", "name": "search_flights",
     "run_id": "uuid-1", "parent_ids": [],
     "data": {"input": {"origin": "NYC", "destination": "NRT"}}}

    # Tool end  (matched to start by run_id)
    {"event": "on_tool_end", "name": "search_flights",
     "run_id": "uuid-1",
     "data": {"output": {"flights": [{"id": "UA123", "price": 850}]}}}

    # LLM start
    {"event": "on_chat_model_start", "name": "gpt-4o",
     "run_id": "uuid-2", "data": {}}

    # LLM end  (reasoning extracted from generation content)
    {"event": "on_chat_model_end", "name": "gpt-4o",
     "run_id": "uuid-2",
     "data": {"output": {"generations": [[{"message": {"content": "..."}}]]}}}

``on_chain_stream`` events are inspected for ``__interrupt__`` data — when
present, a ``state_transition`` node with ``tool_name="__interrupt__"`` is
emitted so HITL pauses are visible to assertions.  All other event types
(``on_chain_start``, ``on_chain_end``, ``on_retriever_end``, etc.) are
silently ignored.

Usage example::

    from trajeval.adapters.langgraph import from_langgraph_jsonl
    from trajeval.assertions.core import tool_must_precede

    with open("agent_run.jsonl") as f:
        result = from_langgraph_jsonl(f.read(), agent_id="booking-agent")

    print(result.capabilities.supported_analyses())
    tool_must_precede(result.trace, tool="validate_seat", before="book_flight")

Capabilities
------------
- ``has_structured_io`` = True (tool inputs/outputs from event data)
- ``has_reasoning`` = True if any LLM end event has non-empty generation text
- ``has_cost`` = True if any AIMessage carries ``usage_metadata`` token counts
- ``has_latency`` = False (timestamps not in standard LangGraph events)
- ``has_depth`` = True when ``langgraph_step`` metadata is present (real graphs)

Field recovery details
----------------------
- **depth** — populated from ``event["metadata"]["langgraph_step"]`` (the
  LangGraph execution step number, present on all ToolNode events).
- **parent_node_id** — uses ``parent_ids[-1]`` (immediate parent), not
  ``parent_ids[0]`` (root).  LangGraph orders parent_ids root→immediate.
- **tool_output for errors** — ``on_tool_error`` events produce a
  ``{"error": "...", "status": "error"}`` output dict instead of ``on_tool_end``.
- **cost_usd** — estimated from ``AIMessage.usage_metadata`` token counts when
  present; uses a rough gpt-4o per-token rate as a fallback.

Layer rule: SDK layer — zero backend imports.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from trajeval.adapters.base import AdapterCapabilities, AdapterResult
from trajeval.sdk.models import Trace, TraceEdge, TraceNode

_TS_DEFAULT = "1970-01-01T00:00:00Z"


def from_langgraph_jsonl(
    jsonl: str,
    *,
    agent_id: str = "langgraph-agent",
    version_hash: str = "unknown",
) -> AdapterResult:
    """Convert a LangGraph JSONL event stream to a TrajEval AdapterResult.

    Parameters
    ----------
    jsonl:
        String containing one LangGraph event JSON object per line.
        Blank lines are skipped.  Parse errors emit warnings and skip the line.
    agent_id, version_hash:
        Trace metadata.
    """
    events: list[dict[str, Any]] = []
    warnings: list[str] = []

    for i, line in enumerate(jsonl.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            warnings.append(f"Line {i}: JSON parse error — {exc}")

    return _process_events(
        events, agent_id=agent_id, version_hash=version_hash, warnings=warnings
    )


def from_langgraph_events(
    events: list[dict[str, Any]],
    *,
    agent_id: str = "langgraph-agent",
    version_hash: str = "unknown",
) -> AdapterResult:
    """Convert in-memory LangGraph event dicts to a TrajEval AdapterResult.

    Accepts the list collected from
    ``async for e in graph.astream_events(..., version="v2"): events.append(e)``.

    Parameters
    ----------
    events:
        List of LangGraph event dicts.
    agent_id, version_hash:
        Trace metadata.
    """
    return _process_events(
        events, agent_id=agent_id, version_hash=version_hash, warnings=[]
    )


# ---------------------------------------------------------------------------
# Internal processing
# ---------------------------------------------------------------------------


def _process_events(
    events: list[dict[str, Any]],
    *,
    agent_id: str,
    version_hash: str,
    warnings: list[str],
) -> AdapterResult:
    nodes: list[TraceNode] = []
    has_reasoning = False
    has_cost = False

    # Pending starts keyed by run_id
    tool_starts: dict[str, dict[str, Any]] = {}
    model_starts: dict[str, dict[str, Any]] = {}

    for ev in events:
        event_type = ev.get("event", "")
        run_id = ev.get("run_id") or str(uuid.uuid4())
        name = ev.get("name", "unknown")
        data = ev.get("data", {})

        if event_type == "on_chain_stream":
            # Detect LangGraph human-in-the-loop interrupt events.
            #
            # When a node calls interrupt() the graph suspends and emits an
            # on_chain_stream event (name="LangGraph") with
            # data["chunk"]["__interrupt__"] set to a tuple/list of Interrupt
            # objects.  There is no dedicated on_interrupt event type in
            # astream_events v2 — this is the only signal.
            #
            # Real format (confirmed from astream_events):
            #   chunk = {"__interrupt__": (Interrupt(value=..., id=...), ...)}
            # where Interrupt is a LangGraph dataclass with .value and .id attrs.
            # Fixtures in tests may use plain dicts instead.
            #
            # We surface these as state_transition nodes so downstream
            # assertions can detect HITL pauses in the trajectory.
            chunk = data.get("chunk", {}) if isinstance(data, dict) else {}
            interrupt_items = (
                chunk.get("__interrupt__") if isinstance(chunk, dict) else None
            )
            if interrupt_items:
                for interrupt_item in interrupt_items:
                    # Support both real Interrupt objects and plain dicts (tests)
                    if hasattr(interrupt_item, "value"):
                        interrupt_value = interrupt_item.value
                        resumable = getattr(interrupt_item, "resumable", True)
                        ns = getattr(interrupt_item, "ns", [])
                    elif isinstance(interrupt_item, dict):
                        interrupt_value = interrupt_item.get("value")
                        resumable = interrupt_item.get("resumable", True)
                        ns = interrupt_item.get("ns", [])
                    else:
                        continue
                    nodes.append(
                        TraceNode(
                            node_id=str(uuid.uuid4()),
                            node_type="state_transition",
                            tool_name="__interrupt__",
                            tool_input={
                                "node": name,
                                "ns": ns,
                                "resumable": resumable,
                            },
                            tool_output=(
                                {"value": interrupt_value}
                                if not isinstance(interrupt_value, dict)
                                else interrupt_value
                            ),
                            cost_usd=0.0,
                            duration_ms=0,
                            depth=0,
                            parent_node_id=None,
                            timestamp=_TS_DEFAULT,
                        )
                    )

        elif event_type == "on_tool_start":
            tool_starts[run_id] = {
                "name": name, "input": data.get("input", {}), "ev": ev
            }

        elif event_type in ("on_tool_end", "on_tool_error"):
            # on_tool_error: {"error": <exception>, "input": ..., "tool_call_id": ...}
            # on_tool_end:   {"output": ..., "input": ...}
            start = tool_starts.pop(run_id, None)
            if start is None:
                warnings.append(
                    f"{event_type} run_id={run_id!r} has no matching on_tool_start."
                )
                continue

            if event_type == "on_tool_error":
                error = data.get("error")
                out: dict[str, object] = {
                    "error": str(error) if error is not None else "unknown error",
                    "status": "error",
                }
            else:
                raw_out = data.get("output", {})
                out = raw_out if isinstance(raw_out, dict) else {"result": raw_out}

            raw_inp = start["input"]
            inp: dict[str, object] = (
                raw_inp if isinstance(raw_inp, dict) else {"args": raw_inp}
            )
            # parent_ids is ordered root → immediate parent.
            # Use [-1] (immediate parent), not [0] (root).
            parent_ids: list[str] = (
                ev.get("parent_ids")
                or start.get("ev", {}).get("parent_ids")
                or []
            )
            # langgraph_step from metadata gives the actual execution step,
            # which maps cleanly to depth in a linear graph.
            meta = ev.get("metadata") or start.get("ev", {}).get("metadata") or {}
            depth = int(meta.get("langgraph_step", 0))
            # Preserve LangGraph execution metadata as per-node metadata.
            # Keys: langgraph_node (containing graph node), langgraph_step,
            # langgraph_triggers (channel writes that spawned this task),
            # langgraph_path, checkpoint_ns, langgraph_checkpoint_ns.
            _keep = {"checkpoint_ns", "ls_integration"}
            node_meta: dict[str, object] = {
                k: v for k, v in meta.items()
                if k.startswith("langgraph_") or k in _keep
            }
            nodes.append(
                TraceNode(
                    node_id=run_id,
                    node_type="tool_call",
                    tool_name=start["name"],
                    tool_input=inp,
                    tool_output=out,
                    cost_usd=0.0,
                    duration_ms=0,
                    depth=depth,
                    parent_node_id=parent_ids[-1] if parent_ids else None,
                    timestamp=_TS_DEFAULT,
                    metadata=node_meta,
                )
            )

        elif event_type == "on_chat_model_start":
            model_starts[run_id] = {"name": name}

        elif event_type == "on_chat_model_end":
            start = model_starts.pop(run_id, None)
            if start is None:
                continue
            output = data.get("output", {})
            reasoning = _extract_generation_text(output)
            if reasoning:
                has_reasoning = True
            # Extract token usage from AIMessage.usage_metadata when present.
            # This field is populated by models that return token counts
            # (e.g. gpt-4o, claude-3-5-sonnet-20241022).
            token_cost = _extract_token_cost(output)
            if token_cost > 0:
                has_cost = True
            nodes.append(
                TraceNode(
                    node_id=run_id,
                    node_type="llm_call",
                    tool_name=None,
                    tool_input={},
                    tool_output=(
                        output if isinstance(output, dict) else {}
                    ),
                    cost_usd=token_cost,
                    duration_ms=0,
                    depth=0,
                    parent_node_id=None,
                    timestamp=_TS_DEFAULT,
                    reasoning_text=reasoning,
                )
            )
        # All other event types (on_chain_start, on_chain_end, on_retriever_end,
        # etc.) are silently ignored.  on_chain_stream is handled above but
        # only when it carries __interrupt__ data.

    # Build sequential edges between consecutive tool_call nodes
    tool_nodes = [n for n in nodes if n.node_type == "tool_call"]
    edges = [
        TraceEdge(
            source=tool_nodes[i].node_id,
            target=tool_nodes[i + 1].node_id,
            edge_type="sequential",
        )
        for i in range(len(tool_nodes) - 1)
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
    )

    has_depth = any(n.depth > 0 for n in nodes if n.node_type == "tool_call")
    capabilities = AdapterCapabilities(
        has_cost=has_cost,
        has_reasoning=has_reasoning,
        has_structured_io=True,
        has_latency=False,
        has_depth=has_depth,
    )

    return AdapterResult(trace=trace, capabilities=capabilities, warnings=warnings)


def _extract_generation_text(output: object) -> str | None:
    """Extract LLM generation text from a LangGraph on_chat_model_end output.

    Handles three real formats:

    1. In-memory events: ``output`` is an ``AIMessage`` object with a
       ``.content`` attribute (``run_type == "chat_model"``).

    2. JSONL-serialized events: ``output`` is the ``AIMessage.model_dump()``
       dict — ``{"content": "...", "type": "ai", "tool_calls": [], ...}``.

    3. Old-style LLM completions (``run_type == "llm"``): ``output`` is
       ``{"generations": [[{"text": "..."}]]}``.

    The ``{"generations": [[{"message": {"content": "..."}}]]}`` format used
    in the test fixtures is also handled as a fourth path for compatibility.
    """
    # Case 1: in-memory AIMessage (or any object with .content)
    if hasattr(output, "content"):
        content = output.content
        return content if isinstance(content, str) and content else None

    if not isinstance(output, dict):
        return None

    # Case 2: AIMessage serialized to dict via .model_dump()
    # Keys: {"content": "...", "type": "ai", "tool_calls": [], ...}
    if "content" in output:
        content = output["content"]
        return content if isinstance(content, str) and content else None

    # Case 3 + 4: generations list (old LLM run type or test fixture format)
    try:
        generations = output.get("generations", [[]])
        if not generations or not generations[0]:
            return None
        first_item = generations[0][0]
        if not isinstance(first_item, dict):
            return None
        # Case 3: {"text": "..."} — old-style LLM completion
        if "text" in first_item:
            text = first_item["text"]
            return text if isinstance(text, str) and text else None
        # Case 4: {"message": {"content": "..."}} — test fixture format
        if "message" in first_item:
            msg = first_item["message"]
            content = (
                msg.get("content") if isinstance(msg, dict)
                else getattr(msg, "content", None)
            )
            return content if isinstance(content, str) and content else None
    except (IndexError, TypeError, AttributeError, KeyError):
        pass

    return None


# Cost-per-token rough estimate (gpt-4o pricing; used only when usage_metadata
# is present and no per-token price is known).  Not charged when 0.
_COST_PER_TOKEN_USD = 2e-6


def _extract_token_cost(output: object) -> float:
    """Extract a cost estimate from AIMessage usage_metadata.

    LangGraph's on_chat_model_end emits an AIMessage whose ``usage_metadata``
    field contains token counts when the underlying model returns them::

        AIMessage(usage_metadata={"input_tokens": 512, "output_tokens": 64, ...})

    For JSONL-serialized events the same data appears in the model_dump() dict.

    Returns estimated cost in USD (using a rough per-token rate), or 0.0 if no
    usage metadata is present.
    """
    usage: dict[str, object] | None = None

    # In-memory: AIMessage object
    usage_attr = getattr(output, "usage_metadata", None)
    if isinstance(usage_attr, dict):
        usage = usage_attr
    # Serialized: dict with usage_metadata key
    elif isinstance(output, dict):
        usage_val = output.get("usage_metadata")
        if isinstance(usage_val, dict):
            usage = usage_val

    if not usage:
        return 0.0

    total = (
        (usage.get("input_tokens") or 0)
        + (usage.get("output_tokens") or 0)
    )
    return float(total) * _COST_PER_TOKEN_USD
