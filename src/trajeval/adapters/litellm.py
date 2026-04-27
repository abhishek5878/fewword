"""Adapter: LiteLLM CustomLogger → TrajEval Trace.

LiteLLM is a proxy layer that routes LLM calls to 100+ providers behind a
single OpenAI-compatible API.  Because it sits between the caller and every
LLM provider, a LiteLLM callback receives every completion — making it the
highest-distribution integration point in the stack.

This module provides two things:

1. **``TrajEvalLiteLLMCallback``** — a LiteLLM ``CustomLogger`` subclass
   that accumulates ``llm_call`` nodes as completions arrive, then
   exposes a completed trace via ``get_result()``.

2. **``from_litellm_kwargs``** — a pure-function converter for offline replay:
   given a saved ``(kwargs, response_obj)`` pair from a LiteLLM callback, it
   produces an ``AdapterResult`` directly without re-running the LLM.

Both paths produce the same ``AdapterResult``; the class is for live
instrumentation, the function is for log replay.

LiteLLM CustomLogger interface
-------------------------------
Subclass ``litellm.integrations.custom_logger.CustomLogger`` and override::

    log_success_event(kwargs, response_obj, start_time, end_time)
    log_failure_event(kwargs, response_obj, start_time, end_time)
    async_log_success_event(kwargs, response_obj, start_time, end_time)
    async_log_failure_event(kwargs, response_obj, start_time, end_time)
    log_stream_event(kwargs, response_obj, start_time, end_time)

``kwargs`` is the dict passed to ``litellm.completion()`` and includes:
  ``model``, ``messages``, ``metadata``, ``call_type``, ``litellm_call_id``.

``response_obj`` is a ``ModelResponse`` with fields:
  ``id``, ``model``, ``created`` (Unix seconds), ``choices``, ``usage``.

``usage`` is a ``Usage`` object: ``prompt_tokens``, ``completion_tokens``,
``total_tokens``.

Usage example — live instrumentation::

    import litellm
    from trajeval.adapters.litellm import TrajEvalLiteLLMCallback

    cb = TrajEvalLiteLLMCallback(agent_id="my-agent")
    litellm.callbacks = [cb]

    litellm.completion(model="gpt-4o", messages=[...])
    litellm.completion(model="gpt-4o", messages=[...])

    result = cb.get_result()
    print(result.trace.nodes)

Usage example — offline replay::

    from trajeval.adapters.litellm import from_litellm_kwargs

    # Saved from a previous run:
    result = from_litellm_kwargs(
        kwargs={"model": "gpt-4o", "litellm_call_id": "abc"},
        response_obj=response,
        start_time=t0,
        end_time=t1,
        agent_id="my-agent",
    )

Layer rule: SDK layer — zero backend imports.
The ``litellm`` package is optional.  CustomLogger is subclassed lazily so
that importing this module does not require litellm to be installed.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from trajeval.adapters.base import AdapterCapabilities, AdapterResult
from trajeval.sdk.models import Trace, TraceEdge, TraceNode

_TS_DEFAULT = "1970-01-01T00:00:00Z"
_COST_PER_TOKEN_USD = 2e-6  # rough approximation; real cost depends on model


# ---------------------------------------------------------------------------
# Pure-function converter (offline replay)
# ---------------------------------------------------------------------------


def from_litellm_kwargs(
    kwargs: dict[str, Any],
    response_obj: object,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    *,
    agent_id: str = "litellm-agent",
    version_hash: str = "unknown",
    failed: bool = False,
) -> AdapterResult:
    """Convert a single LiteLLM (kwargs, response_obj) pair to an AdapterResult.

    Parameters
    ----------
    kwargs:
        The dict passed to ``litellm.completion()``.  Must include at least
        ``model`` or ``litellm_call_id``.
    response_obj:
        The ``ModelResponse`` returned by LiteLLM (or None for failures).
    start_time, end_time:
        ``datetime.datetime`` objects provided by the LiteLLM callback.
        Used to compute ``duration_ms`` and the ISO timestamp.
    agent_id, version_hash:
        Trace metadata.
    failed:
        True if called from ``log_failure_event`` — records an error node.
    """
    node = _build_node(kwargs, response_obj, start_time, end_time, failed=failed)
    nodes = [node]
    has_cost = node.cost_usd > 0
    has_latency = node.duration_ms > 0
    has_reasoning = bool(node.reasoning_text and node.reasoning_text.strip())

    trace = Trace(
        trace_id=str(uuid.uuid4()),
        agent_id=agent_id,
        version_hash=version_hash,
        started_at=node.timestamp,
        completed_at=node.timestamp,
        nodes=nodes,
        edges=[],
        total_cost_usd=node.cost_usd,
        total_tokens=_extract_total_tokens(response_obj),
    )
    capabilities = AdapterCapabilities(
        has_cost=has_cost,
        has_reasoning=has_reasoning,
        has_structured_io=False,
        has_latency=has_latency,
        has_depth=False,
    )
    return AdapterResult(trace=trace, capabilities=capabilities, warnings=[])


# ---------------------------------------------------------------------------
# Live callback class (optional litellm dependency)
# ---------------------------------------------------------------------------


def _make_callback_class() -> type:
    """Build TrajEvalLiteLLMCallback, importing litellm lazily."""
    try:
        from litellm.integrations.custom_logger import (
            CustomLogger,  # type: ignore[import-untyped]
        )
    except ImportError as exc:
        raise ImportError(
            "litellm is required for TrajEvalLiteLLMCallback. "
            "Install it with: pip install litellm"
        ) from exc

    class TrajEvalLiteLLMCallback(CustomLogger):  # type: ignore[misc]
        """LiteLLM callback that accumulates LLM call nodes into a TrajEval trace.

        Register with LiteLLM::

            import litellm
            cb = TrajEvalLiteLLMCallback(agent_id="my-agent")
            litellm.callbacks = [cb]

        After your completions::

            result = cb.get_result()
            tool_call_count(result.trace, "llm_call", max=10)
        """

        def __init__(
            self,
            *,
            agent_id: str = "litellm-agent",
            version_hash: str = "unknown",
            **kwargs: object,
        ) -> None:
            super().__init__(**kwargs)
            self._agent_id = agent_id
            self._version_hash = version_hash
            self._nodes: list[TraceNode] = []
            self._warnings: list[str] = []

        def log_success_event(
            self,
            kwargs: dict[str, Any],
            response_obj: object,
            start_time: datetime.datetime,
            end_time: datetime.datetime,
        ) -> None:
            self._append(kwargs, response_obj, start_time, end_time, failed=False)

        def log_failure_event(
            self,
            kwargs: dict[str, Any],
            response_obj: object,
            start_time: datetime.datetime,
            end_time: datetime.datetime,
        ) -> None:
            self._append(kwargs, response_obj, start_time, end_time, failed=True)

        async def async_log_success_event(
            self,
            kwargs: dict[str, Any],
            response_obj: object,
            start_time: datetime.datetime,
            end_time: datetime.datetime,
        ) -> None:
            self._append(kwargs, response_obj, start_time, end_time, failed=False)

        async def async_log_failure_event(
            self,
            kwargs: dict[str, Any],
            response_obj: object,
            start_time: datetime.datetime,
            end_time: datetime.datetime,
        ) -> None:
            self._append(kwargs, response_obj, start_time, end_time, failed=True)

        def _append(
            self,
            kwargs: dict[str, Any],
            response_obj: object,
            start_time: datetime.datetime,
            end_time: datetime.datetime,
            *,
            failed: bool,
        ) -> None:
            node = _build_node(
                kwargs, response_obj, start_time, end_time, failed=failed
            )
            self._nodes.append(node)

        def get_result(self) -> AdapterResult:
            """Return an AdapterResult from all accumulated completions."""
            nodes = list(self._nodes)
            llm_nodes = [n for n in nodes if n.node_type == "llm_call"]
            edges = [
                TraceEdge(
                    source=llm_nodes[i].node_id,
                    target=llm_nodes[i + 1].node_id,
                    edge_type="sequential",
                )
                for i in range(len(llm_nodes) - 1)
            ]
            has_cost = any(n.cost_usd > 0 for n in nodes)
            has_latency = any(n.duration_ms > 0 for n in nodes)
            has_reasoning = any(
                n.reasoning_text and n.reasoning_text.strip() for n in nodes
            )
            trace = Trace(
                trace_id=str(uuid.uuid4()),
                agent_id=self._agent_id,
                version_hash=self._version_hash,
                started_at=nodes[0].timestamp if nodes else _TS_DEFAULT,
                completed_at=nodes[-1].timestamp if nodes else _TS_DEFAULT,
                nodes=nodes,
                edges=edges,
                total_cost_usd=sum(n.cost_usd for n in nodes),
                total_tokens=sum(int(n.metadata.get("total_tokens", 0)) for n in nodes),
            )
            capabilities = AdapterCapabilities(
                has_cost=has_cost,
                has_reasoning=has_reasoning,
                has_structured_io=False,
                has_latency=has_latency,
                has_depth=False,
            )
            return AdapterResult(
                trace=trace,
                capabilities=capabilities,
                warnings=list(self._warnings),
            )

        def reset(self) -> None:
            """Clear accumulated nodes. Call between separate agent runs."""
            self._nodes = []
            self._warnings = []

    return TrajEvalLiteLLMCallback


class _LazyCallback:
    """Lazy proxy for TrajEvalLiteLLMCallback.

    Importing this module does not require litellm to be installed.
    The real class is constructed on first instantiation.
    """

    _cls: type | None = None

    def __new__(cls, **kwargs: object) -> object:  # type: ignore[misc]
        if cls._cls is None:
            cls._cls = _make_callback_class()
        return cls._cls(**kwargs)


TrajEvalLiteLLMCallback = _LazyCallback  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_node(
    kwargs: dict[str, Any],
    response_obj: object,
    start_time: datetime.datetime | None,
    end_time: datetime.datetime | None,
    *,
    failed: bool,
) -> TraceNode:
    """Build a TraceNode from a single LiteLLM completion callback."""
    model = kwargs.get("model", "unknown")
    call_id = kwargs.get("litellm_call_id", str(uuid.uuid4()))

    # Timestamps + duration
    ts = _TS_DEFAULT
    duration_ms = 0
    if start_time is not None:
        ts = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end_time is not None:
            delta = end_time - start_time
            duration_ms = max(0, int(delta.total_seconds() * 1000))

    # Token usage + cost
    total_tokens = _extract_total_tokens(response_obj)
    cost_usd = total_tokens * _COST_PER_TOKEN_USD

    # Reasoning text from first choice
    reasoning: str | None = None
    tool_output: dict[str, object] = {"model": model}
    if failed:
        tool_output = {"model": model, "status": "error"}
    elif response_obj is not None:
        reasoning = _extract_content(response_obj)
        tool_output = {"model": model, "status": "ok"}

    # Pass through safe kwargs metadata
    node_meta: dict[str, object] = {}
    for key in ("call_type", "metadata"):
        val = kwargs.get(key)
        if val is not None:
            node_meta[key] = val
    node_meta["total_tokens"] = total_tokens

    return TraceNode(
        node_id=str(call_id),
        node_type="llm_call",
        tool_name=None,
        tool_input={},
        tool_output=tool_output,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        depth=0,
        parent_node_id=None,
        timestamp=ts,
        reasoning_text=reasoning,
        metadata=node_meta,
    )


def _extract_total_tokens(response_obj: object) -> int:
    """Extract total token count from a ModelResponse or dict."""
    if response_obj is None:
        return 0
    # ModelResponse object: .usage.total_tokens
    usage = getattr(response_obj, "usage", None)
    if usage is not None:
        total = getattr(usage, "total_tokens", None)
        if isinstance(total, int):
            return total
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        return int(prompt) + int(completion)
    # dict form
    if isinstance(response_obj, dict):
        u = response_obj.get("usage", {})
        if isinstance(u, dict):
            return int(u.get("total_tokens", 0))
    return 0


def _extract_content(response_obj: object) -> str | None:
    """Extract assistant message content from a ModelResponse."""
    # ModelResponse.choices[0].message.content
    choices = getattr(response_obj, "choices", None)
    if choices and len(choices) > 0:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                return content
    # dict form
    if isinstance(response_obj, dict):
        try:
            content = response_obj["choices"][0]["message"]["content"]
            if isinstance(content, str) and content:
                return content
        except (KeyError, IndexError, TypeError):
            pass
    return None
