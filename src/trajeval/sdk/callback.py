"""LangGraph callback handler that builds a Trace in memory.

This module is part of the SDK — zero-dependency on the backend.

Usage::

    # Observe mode (default): record only
    callback = TrajEvalCallback(agent_id="my-agent", version_hash="abc123")
    graph.invoke(inputs, config={"callbacks": [callback]})
    trace = callback.get_trace()

    # Guard mode: intercept tool calls that violate rules before execution
    from trajeval.assertions.core import never_calls
    import functools

    callback = TrajEvalCallback(
        agent_id="my-agent",
        mode="guard",
        guard_assertions=[functools.partial(never_calls, tool="delete_user")],
    )

    # Convenience: pass a CompiledContract directly
    callback = TrajEvalCallback.from_contract(contract, mode="guard")

Both sync and async LangGraph invocations are supported.  The sync callback
methods are implemented here; BaseCallbackHandler's async stubs delegate to
the sync implementations automatically, so no a* overrides are needed.

Guard mode overhead:
    ~2 ms per tool call for assertion evaluation.  Disabled by default
    (mode="observe").  Each guard assertion runs against a prospective trace
    that includes the proposed tool call node before it executes.

    Assertions that work correctly as guard assertions (evaluate against a
    partial trace):
      - never_calls, no_duplicate_arg_call, total_tool_calls, tool_call_count,
        tool_must_precede, max_depth, no_cycles

    Assertions that do NOT work as guard assertions (produce false positives
    against a partial trace):
      - must_visit, cost_within, latency_within

Layer rule: this module may import from assertions/ but never from backend/.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from trajeval.analysis.ltl import LTLFormula, LTLRuntime
from trajeval.sdk.exceptions import TrajectoryInterceptionError
from trajeval.sdk.models import AgentConfig, NodeType, Trace, TraceEdge, TraceNode
from trajeval.sdk.trace_context import TraceContext

if TYPE_CHECKING:
    from trajeval.contract.compiled import CompiledContract

# Type alias for assertion callables accepted as guard assertions.
# Mirrors AssertionFn from assertions/core.py without the runtime import.
_GuardFn = Callable[[Trace], None]

# ---------------------------------------------------------------------------
# Model pricing — USD per 1 million tokens (input, output)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
}


# ---------------------------------------------------------------------------
# Internal state for an in-flight run
# ---------------------------------------------------------------------------


@dataclass
class _PendingNode:
    """Mutable state for a callback run that has started but not yet completed."""

    node_id: str
    node_type: NodeType
    tool_name: str | None
    tool_input: dict[str, object]
    start_time: float  # monotonic clock
    parent_run_id: str | None
    timestamp: str  # ISO 8601
    available_tools: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TrajEvalCallback
# ---------------------------------------------------------------------------


class TrajEvalCallback(BaseCallbackHandler):
    """Intercepts LangGraph events and builds a :class:`~trajeval.sdk.models.Trace`.

    Parameters
    ----------
    agent_id:
        Customer-defined identifier for the agent (stored on the Trace).
    version_hash:
        Git SHA or semver of the agent at the time of the run.
    model_pricing:
        Override the default token pricing dict.  Keys are model name strings;
        values are ``{"input": <usd_per_m>, "output": <usd_per_m>}`` dicts.
    metadata:
        Arbitrary key-value pairs forwarded to ``Trace.metadata``.
    mode:
        ``"observe"`` (default) — record only, zero overhead.
        ``"guard"`` — evaluate *guard_assertions* and *ltl_formulas* before
        each tool call executes.  Violations raise
        :exc:`TrajectoryInterceptionError`, which LangGraph sees as a tool
        failure.
    guard_assertions:
        Callables with signature ``(trace: Trace) -> None``.  Evaluated
        against a prospective trace (current nodes + the proposed call) in
        ``on_tool_start``.  Only used when ``mode="guard"``.
        See module docstring for which assertion types are safe to use here.
    ltl_formulas:
        LTL formulas compiled to Büchi automata at construction time.  In
        guard mode, :meth:`~trajeval.analysis.ltl.LTLRuntime.would_enter_reject`
        is called before each tool execution — O(1) per step regardless of
        trace depth.  In observe mode, violations are recorded and surfaced in
        ``Trace.metadata["ltl_violations"]`` when :meth:`get_trace` is called.
        Liveness violations (``Eventually``, ``Response``) are always checked
        at :meth:`get_trace` time.
    parent_trace_id:
        ``trace_id`` of the parent agent's :class:`~trajeval.sdk.models.Trace`.
        Set this when spawning a sub-agent from an orchestrator.
    parent_node_id:
        ``node_id`` of the node in the parent trace that spawned this agent.
        Combines with *parent_trace_id* to give the exact spawn point.
    parent_trace_context:
        W3C :class:`~trajeval.sdk.trace_context.TraceContext` extracted from an
        inbound ``traceparent`` HTTP header.  Takes lower priority than explicit
        *parent_trace_id* — use this for cross-process HTTP propagation.
    """

    raise_error: bool = False  # class default: swallow callback errors

    def __init__(
        self,
        agent_id: str = "default",
        version_hash: str = "unknown",
        model_pricing: dict[str, dict[str, float]] | None = None,
        metadata: dict[str, object] | None = None,
        mode: Literal["observe", "guard"] = "observe",
        guard_assertions: list[_GuardFn] | None = None,
        ltl_formulas: list[LTLFormula] | None = None,
        parent_trace_id: str | None = None,
        parent_node_id: str | None = None,
        parent_trace_context: TraceContext | None = None,
        registered_tools: list[str] | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        super().__init__()
        self._agent_id = agent_id
        self._version_hash = version_hash
        self._pricing = (
            model_pricing if model_pricing is not None else DEFAULT_MODEL_PRICING
        )
        self._metadata: dict[str, object] = metadata if metadata is not None else {}
        self._mode: Literal["observe", "guard"] = mode
        self._guard_assertions: list[_GuardFn] = (
            list(guard_assertions) if guard_assertions is not None else []
        )
        self._ltl_runtime: LTLRuntime | None = (
            LTLRuntime(ltl_formulas) if ltl_formulas is not None else None
        )
        self._registered_tools: list[str] = (
            list(registered_tools) if registered_tools is not None else []
        )
        self._config: AgentConfig | None = config
        # Override the class-level raise_error=False.  In guard mode, the
        # TrajectoryInterceptionError raised by on_tool_start must propagate
        # to the LangGraph agent rather than being swallowed by the callback
        # manager.  For observe mode, keep swallowing so callback bugs never
        # crash the agent.
        self.raise_error: bool = mode == "guard"

        self._trace_id: str = str(uuid.uuid4())
        self._span_id: str = uuid.uuid4().hex[:16]
        self._started_at: str | None = None
        self._completed_at: str | None = None

        # Multi-agent parent linking.  Explicit params take precedence over the
        # parsed TraceContext (which comes from an HTTP header and only carries
        # the parent's trace_id + span_id, not the full node_id).
        if parent_trace_id is not None:
            self._parent_trace_id: str | None = parent_trace_id
            self._parent_node_id: str | None = parent_node_id
        elif parent_trace_context is not None:
            # trace_id in the context is the parent's TrajEval trace_id
            # (32 hex = UUID with hyphens stripped).  parent_id is the span_id
            # that the parent callback exposed — we store it as parent_node_id
            # so SwarmTrace can attribute the spawn to that node even if the
            # full UUID can't be reconstructed.
            self._parent_trace_id = parent_trace_context.trace_id
            self._parent_node_id = parent_trace_context.parent_id
        else:
            self._parent_trace_id = None
            self._parent_node_id = None

        # run_id (str) → pending node (started but not committed)
        self._pending: dict[str, _PendingNode] = {}
        # run_id (str) → TraceNode.node_id for already-committed nodes
        self._run_to_node: dict[str, str] = {}

        self._nodes: list[TraceNode] = []
        self._edges: list[TraceEdge] = []

        self._total_cost_usd: float = 0.0
        self._total_tokens: int = 0

        # run_id → depth (populated at *_start time)
        self._depths: dict[str, int] = {}
        # node_id of the last committed node (for sequential edges)
        self._last_node_id: str | None = None

    # ------------------------------------------------------------------
    # Public read-only properties (for orchestrators spawning sub-agents)
    # ------------------------------------------------------------------

    @property
    def current_trace_id(self) -> str:
        """The ``trace_id`` that will appear on the completed :class:`Trace`."""
        return self._trace_id

    @property
    def current_node_id(self) -> str | None:
        """``node_id`` of the most recently committed node, or ``None``."""
        return self._last_node_id

    @property
    def span_id(self) -> str:
        """OTel-compatible 64-bit span ID (16 hex chars) for this trace.

        Propagate to child agents via
        :meth:`~trajeval.sdk.trace_context.TraceContext.from_ids`.
        """
        return self._span_id

    def make_child_context(self) -> TraceContext:
        """Create a :class:`~trajeval.sdk.trace_context.TraceContext` to pass
        to a sub-agent spawned from the *current* node.

        The context encodes this callback's ``trace_id`` and ``span_id``
        so the sub-agent can reconstruct the parent link via the W3C
        ``traceparent`` header or the *parent_trace_context* param.

        Example::

            ctx = orchestrator_callback.make_child_context()
            headers = {"traceparent": ctx.to_header()}
            # … HTTP call to sub-agent …

            # Sub-agent side:
            ctx = TraceContext.from_header(request.headers["traceparent"])
            cb = TrajEvalCallback(agent_id="sub", parent_trace_context=ctx)
        """
        return TraceContext.from_ids(
            trace_id=self._trace_id,
            span_id=self._span_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _rid(run_id: UUID) -> str:
        return str(run_id)

    @staticmethod
    def _pid(parent_run_id: UUID | None) -> str | None:
        return str(parent_run_id) if parent_run_id is not None else None

    def _depth_for(self, parent_run_id: str | None) -> int:
        if parent_run_id is None:
            return 0
        return self._depths.get(parent_run_id, 0) + 1

    def _node_id_for_run(self, run_id: str | None) -> str | None:
        """Return the TraceNode.node_id for *run_id*.

        Checks pending nodes first, then committed ones.
        """
        if run_id is None:
            return None
        pending = self._pending.get(run_id)
        if pending is not None:
            return pending.node_id
        return self._run_to_node.get(run_id)

    def _calc_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self._pricing.get(model)
        if p is None:
            return 0.0
        return (input_tokens / 1_000_000) * p["input"] + (
            output_tokens / 1_000_000
        ) * p["output"]

    def _commit(self, run_id: str, node: TraceNode) -> None:
        """Append *node* to the trace and wire a sequential edge from the previous."""
        self._run_to_node[run_id] = node.node_id
        if self._last_node_id is not None:
            self._edges.append(
                TraceEdge(
                    source=self._last_node_id,
                    target=node.node_id,
                    edge_type="sequential",
                )
            )
        self._nodes.append(node)
        self._last_node_id = node.node_id

        # Advance LTL automata (both modes).  In observe mode this records
        # safety violations for post-hoc inspection.  In guard mode, safety
        # violations were already caught pre-execution via would_enter_reject,
        # so this primarily handles non-tool_call nodes (llm_call, etc.) and
        # keeps automaton state consistent.
        if self._ltl_runtime is not None:
            self._ltl_runtime.advance(node)

    # ------------------------------------------------------------------
    # Guard mode helpers
    # ------------------------------------------------------------------

    def _build_prospective_trace(
        self, pending: _PendingNode, rid: str
    ) -> tuple[Trace, TraceNode]:
        """Build a prospective :class:`Trace` with *pending* appended as a node.

        Returns ``(prospective_trace, prospective_node)``.  The prospective
        node has ``tool_output={}`` and ``duration_ms=0`` because the tool has
        not yet executed.
        """
        prospective_node = TraceNode(
            node_id=pending.node_id,
            node_type="tool_call",
            tool_name=pending.tool_name,
            tool_input=pending.tool_input,
            tool_output={},
            cost_usd=0.0,
            duration_ms=0,
            depth=self._depths.get(rid, 0),
            parent_node_id=self._node_id_for_run(pending.parent_run_id),
            timestamp=pending.timestamp,
            available_tools=pending.available_tools,
        )

        prospective_edges = list(self._edges)
        if self._last_node_id is not None:
            prospective_edges.append(
                TraceEdge(
                    source=self._last_node_id,
                    target=prospective_node.node_id,
                    edge_type="sequential",
                )
            )

        prospective_trace = Trace(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            version_hash=self._version_hash,
            started_at=self._started_at or pending.timestamp,
            completed_at=pending.timestamp,
            total_cost_usd=self._total_cost_usd,
            total_tokens=self._total_tokens,
            nodes=list(self._nodes) + [prospective_node],
            edges=prospective_edges,
            assertion_results=[],
            anomaly_score=None,
            metadata=dict(self._metadata),
        )
        return prospective_trace, prospective_node

    def _run_guard_assertions(
        self, prospective_trace: Trace, tool_name: str | None
    ) -> None:
        """Evaluate guard assertions against *prospective_trace*.

        Collects ALL violations before raising so the agent sees the complete
        picture.  Raises :exc:`TrajectoryInterceptionError` if any fail.
        """
        violations: list[str] = []
        for assertion in self._guard_assertions:
            try:
                assertion(prospective_trace)
            except AssertionError as exc:
                violations.append(str(exc))

        if violations:
            raise TrajectoryInterceptionError(
                violations=violations,
                tool_name=tool_name,
                trace_id=self._trace_id,
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_contract(
        cls,
        contract: CompiledContract,
        *,
        mode: Literal["observe", "guard"] = "guard",
        agent_id: str = "default",
        version_hash: str = "unknown",
        model_pricing: dict[str, dict[str, float]] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TrajEvalCallback:
        """Create a callback pre-wired with a compiled contract's assertions.

        Convenience alternative to passing ``guard_assertions=contract.assertions``
        manually.  Defaults to ``mode="guard"`` since the typical use-case is
        enforcement, not just observation.

        Parameters
        ----------
        contract:
            A :class:`~trajeval.contract.compiled.CompiledContract` to enforce.
        mode:
            ``"guard"`` (default) or ``"observe"``.
        """
        return cls(
            agent_id=agent_id,
            version_hash=version_hash,
            model_pricing=model_pricing,
            metadata=metadata,
            mode=mode,
            guard_assertions=list(contract.assertions),
        )

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        pid = self._pid(parent_run_id)
        self._depths[rid] = self._depth_for(pid)

        tool_name: str = (
            str(serialized.get("name", "unknown")) if serialized else "unknown"
        )

        tool_input: dict[str, object] = (
            dict(inputs) if inputs is not None else {"input": input_str}
        )

        pending = _PendingNode(
            node_id=str(uuid.uuid4()),
            node_type="tool_call",
            tool_name=tool_name,
            tool_input=tool_input,
            start_time=time.monotonic(),
            parent_run_id=pid,
            timestamp=self._now_iso(),
            available_tools=list(self._registered_tools),
        )
        self._pending[rid] = pending

        # Guard mode: evaluate assertions against a prospective trace before
        # the tool executes.  If any assertion fires, remove the pending entry
        # (the tool won't execute), then raise TrajectoryInterceptionError so
        # LangGraph receives it as a tool failure.
        if self._mode == "guard" and self._guard_assertions:
            prospective_trace, _ = self._build_prospective_trace(pending, rid)
            try:
                self._run_guard_assertions(prospective_trace, tool_name)
            except TrajectoryInterceptionError:
                self._pending.pop(rid, None)  # clean up — tool is blocked
                raise

        # Guard mode: incremental LTL check — O(1) per tool call, no state mutation.
        # Fires BEFORE the tool executes so the tool can be blocked pre-emptively.
        if self._mode == "guard" and self._ltl_runtime is not None:
            ltl_violations = self._ltl_runtime.would_enter_reject(tool_name)
            if ltl_violations:
                self._pending.pop(rid, None)
                raise TrajectoryInterceptionError(
                    violations=ltl_violations,
                    tool_name=tool_name,
                    trace_id=self._trace_id,
                )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        pending = self._pending.pop(rid, None)
        if pending is None:
            return

        duration_ms = int((time.monotonic() - pending.start_time) * 1000)
        tool_output: dict[str, object] = (
            dict(output) if isinstance(output, dict) else {"output": str(output)}
        )

        node = TraceNode(
            node_id=pending.node_id,
            node_type="tool_call",
            tool_name=pending.tool_name,
            tool_input=pending.tool_input,
            tool_output=tool_output,
            cost_usd=0.0,
            duration_ms=duration_ms,
            depth=self._depths.get(rid, 0),
            parent_node_id=self._node_id_for_run(pending.parent_run_id),
            timestamp=pending.timestamp,
            available_tools=pending.available_tools,
        )
        self._commit(rid, node)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        pending = self._pending.pop(rid, None)
        if pending is None:
            return

        duration_ms = int((time.monotonic() - pending.start_time) * 1000)
        node = TraceNode(
            node_id=pending.node_id,
            node_type="tool_call",
            tool_name=pending.tool_name,
            tool_input=pending.tool_input,
            tool_output={"error": str(error)},
            cost_usd=0.0,
            duration_ms=duration_ms,
            depth=self._depths.get(rid, 0),
            parent_node_id=self._node_id_for_run(pending.parent_run_id),
            timestamp=pending.timestamp,
            available_tools=pending.available_tools,
        )
        self._commit(rid, node)

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        pid = self._pid(parent_run_id)
        self._depths[rid] = self._depth_for(pid)

        # Resolve model name: langchain-openai stores it under kwargs.model_name,
        # langchain-anthropic under kwargs.model.  Fall back to the class name.
        model_name: str = "unknown"
        if serialized:
            kw: dict[str, Any] = serialized.get("kwargs") or {}
            model_name = str(
                serialized.get("name")
                or kw.get("model_name")
                or kw.get("model")
                or "unknown"
            )

        self._pending[rid] = _PendingNode(
            node_id=str(uuid.uuid4()),
            node_type="llm_call",
            tool_name=model_name,
            tool_input={"prompts": list(prompts)},
            start_time=time.monotonic(),
            parent_run_id=pid,
            timestamp=self._now_iso(),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        pending = self._pending.pop(rid, None)
        if pending is None:
            return

        duration_ms = int((time.monotonic() - pending.start_time) * 1000)

        input_tokens = 0
        output_tokens = 0
        if response.llm_output:
            # OpenAI: token_usage.prompt_tokens / completion_tokens
            # Anthropic: usage.input_tokens / output_tokens
            raw_usage: object = (
                response.llm_output.get("token_usage")
                or response.llm_output.get("usage")
                or {}
            )
            usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
            input_tokens = int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            output_tokens = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )

        model = pending.tool_name or "unknown"
        cost = self._calc_cost(model, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        self._total_cost_usd += cost
        self._total_tokens += total_tokens

        # Capture the first generation's text as the reasoning scratchpad.
        # This is the chain-of-thought the LLM produced before the next tool call.
        reasoning_text: str | None = None
        if response.generations:
            first_gen = response.generations[0]
            if first_gen:
                reasoning_text = first_gen[0].text or None

        node = TraceNode(
            node_id=pending.node_id,
            node_type="llm_call",
            tool_name=model,
            tool_input=pending.tool_input,
            tool_output={
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            cost_usd=cost,
            duration_ms=duration_ms,
            depth=self._depths.get(rid, 0),
            parent_node_id=self._node_id_for_run(pending.parent_run_id),
            timestamp=pending.timestamp,
            reasoning_text=reasoning_text,
        )
        self._commit(rid, node)

    # ------------------------------------------------------------------
    # Chain callbacks  (state transitions / LangGraph node executions)
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        pid = self._pid(parent_run_id)
        self._depths[rid] = self._depth_for(pid)

        if self._started_at is None:
            self._started_at = self._now_iso()

        # Prefer "name"; fall back to last segment of the "id" list.
        node_name: str = "unknown"
        if serialized:
            ids: list[str] = serialized.get("id") or []
            node_name = str(
                serialized.get("name")
                or (ids[-1] if isinstance(ids, list) and ids else "unknown")
            )

        self._pending[rid] = _PendingNode(
            node_id=str(uuid.uuid4()),
            node_type="state_transition",
            tool_name=None,
            tool_input={"node_name": node_name},
            start_time=time.monotonic(),
            parent_run_id=pid,
            timestamp=self._now_iso(),
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        rid = self._rid(run_id)
        pending = self._pending.pop(rid, None)
        if pending is None:
            return

        duration_ms = int((time.monotonic() - pending.start_time) * 1000)

        output_dict: dict[str, object] = {}
        if isinstance(outputs, dict):
            output_dict = {k: str(v) for k, v in outputs.items()}

        # The root chain end (no parent) marks the run as complete.
        if parent_run_id is None and self._completed_at is None:
            self._completed_at = self._now_iso()

        node = TraceNode(
            node_id=pending.node_id,
            node_type="state_transition",
            tool_name=None,
            tool_input=pending.tool_input,
            tool_output=output_dict,
            cost_usd=0.0,
            duration_ms=duration_ms,
            depth=self._depths.get(rid, 0),
            parent_node_id=self._node_id_for_run(pending.parent_run_id),
            timestamp=pending.timestamp,
        )
        self._commit(rid, node)

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def get_trace(self) -> Trace:
        """Return the accumulated :class:`Trace`.  Call after the run completes."""
        now = self._now_iso()
        meta = dict(self._metadata)
        if self._ltl_runtime is not None:
            self._ltl_runtime.check_liveness()
            all_ltl = (
                self._ltl_runtime.violations
            )  # safety already appended + liveness above
            if all_ltl:
                meta["ltl_violations"] = list(all_ltl)
        return Trace(
            trace_id=self._trace_id,
            agent_id=self._agent_id,
            version_hash=self._version_hash,
            started_at=self._started_at or now,
            completed_at=self._completed_at or now,
            total_cost_usd=self._total_cost_usd,
            total_tokens=self._total_tokens,
            nodes=list(self._nodes),
            edges=list(self._edges),
            assertion_results=[],
            anomaly_score=None,
            metadata=meta,
            span_id=self._span_id,
            parent_trace_id=self._parent_trace_id,
            parent_node_id=self._parent_node_id,
            config=self._config,
        )
