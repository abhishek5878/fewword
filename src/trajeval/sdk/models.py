"""Trajectory trace data models — source of truth per CLAUDE.md.

These models are part of the SDK package and must remain zero-dependency on
the backend. Pydantic v2, strict types, no Any.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

NodeType = Literal["tool_call", "llm_call", "state_transition"]
EdgeType = Literal["sequential", "conditional", "retry"]


class AgentConfig(BaseModel):
    """Snapshot of the agent's configuration at trace-collection time.

    Capturing this alongside the trace enables sensitivity analysis: which
    configuration parameters most predict violation rates?  Pass this to
    :class:`~trajeval.sdk.callback.TrajEvalCallback` via the *config* param.

    All fields are optional so that partial configs can be stored.

    Attributes
    ----------
    model_id:
        LLM model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-6"``).
    temperature:
        Sampling temperature used for this run.
    system_prompt_hash:
        SHA-256 hex digest of the system prompt — store the hash, not the raw
        text, to avoid logging sensitive instructions.
    active_tools:
        Names of all tools registered with the agent for this run.
    memory_window:
        Context window size in tokens (if the agent manages a rolling window).
    extra:
        Arbitrary additional config dimensions (e.g. ``{"top_p": 0.9}``).
    """

    model_config = {"frozen": True}

    model_id: str = Field(default="unknown", description="LLM model identifier")
    temperature: float | None = Field(default=None, description="Sampling temperature")
    system_prompt_hash: str | None = Field(
        default=None, description="SHA-256 hex digest of the system prompt"
    )
    active_tools: list[str] = Field(
        default_factory=list, description="Registered tool names for this run"
    )
    memory_window: int | None = Field(
        default=None, description="Context window size in tokens"
    )
    extra: dict[str, object] = Field(
        default_factory=dict, description="Additional config dimensions"
    )


class TraceNode(BaseModel):
    """A single node in the agent trajectory graph."""

    model_config = {"frozen": True}

    node_id: str = Field(description="UUID identifying this node")
    node_type: NodeType = Field(description="Classification of the node")
    tool_name: str | None = Field(
        default=None,
        description="Name of the tool, only set when node_type == 'tool_call'",
    )
    tool_input: dict[str, object] = Field(
        default_factory=dict,
        description="Sanitized input passed to the tool",
    )
    tool_output: dict[str, object] = Field(
        default_factory=dict,
        description="Sanitized output returned by the tool",
    )
    cost_usd: float = Field(
        default=0.0, ge=0.0, description="Cost in USD for this node"
    )
    duration_ms: int = Field(
        default=0, ge=0, description="Wall-clock duration in milliseconds"
    )
    depth: int = Field(
        default=0, ge=0, description="Depth in the call graph (0 = root)"
    )
    parent_node_id: str | None = Field(
        default=None,
        description="node_id of the parent, None for root nodes",
    )
    timestamp: str = Field(description="ISO 8601 timestamp when this node was recorded")
    reasoning_text: str | None = Field(
        default=None,
        description=(
            "Raw chain-of-thought / scratchpad text emitted by the LLM for this node. "
            "Only set on node_type='llm_call' nodes; "
            "None for tool_call and state_transition."
        ),
    )
    available_tools: list[str] = Field(
        default_factory=list,
        description=(
            "All tools registered with the agent at this decision point. "
            "The unchosen actions are available_tools - {tool_name}. "
            "Populated only when registered_tools is passed to TrajEvalCallback."
        ),
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Arbitrary per-node metadata preserved from the source event. "
            "Adapters populate this with framework-specific fields that do not map "
            "to a first-class TraceNode field — e.g. langgraph_node, "
            "langgraph_triggers, checkpoint_ns (LangGraph); gen_ai.tool.type, "
            "gen_ai.system, gen_ai.tool.call.id (OTel); call_id (OpenAI). "
            "Analysis modules should use typed fields first and fall back to "
            "metadata only for framework-specific logic."
        ),
    )


class TraceEdge(BaseModel):
    """A directed edge in the trajectory graph."""

    model_config = {"frozen": True}

    source: str = Field(description="node_id of the source node")
    target: str = Field(description="node_id of the target node")
    edge_type: EdgeType = Field(description="Semantic type of this edge")


class Trace(BaseModel):
    """Complete trace of a single agent run."""

    model_config = {"frozen": True}

    trace_id: str = Field(description="UUID identifying this trace")
    agent_id: str = Field(description="Customer-defined identifier for the agent")
    version_hash: str = Field(description="Git SHA of the agent at time of run")
    started_at: str = Field(description="ISO 8601 timestamp when the run started")
    completed_at: str = Field(description="ISO 8601 timestamp when the run completed")
    total_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Aggregate cost for the run"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed")
    nodes: list[TraceNode] = Field(default_factory=list)
    edges: list[TraceEdge] = Field(default_factory=list)
    assertion_results: list[dict[str, object]] = Field(default_factory=list)
    anomaly_score: float | None = Field(
        default=None,
        description="Anomaly score from the analysis engine; None if not computed",
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Arbitrary customer-defined tags",
    )
    # Agent configuration snapshot (Gap 2: configuration sensitivity)
    config: AgentConfig | None = Field(
        default=None,
        description=(
            "Snapshot of agent configuration at trace-collection time. "
            "Used by analysis/sensitivity.py to correlate config parameters "
            "with violation rates across traces."
        ),
    )
    # Multi-agent stitching fields (Addition 4)
    span_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:16],
        description=(
            "OTel-compatible 64-bit span ID (16 lowercase hex chars). "
            "Propagated to child agents via the W3C traceparent header."
        ),
    )
    parent_trace_id: str | None = Field(
        default=None,
        description=(
            "trace_id of the parent agent's Trace. None for root (orchestrator) traces."
        ),
    )
    parent_node_id: str | None = Field(
        default=None,
        description=(
            "node_id of the node in the parent trace that spawned this agent. "
            "None for root traces."
        ),
    )
