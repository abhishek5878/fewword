"""Shared types for the TrajEval adapter layer.

Every adapter returns an :class:`AdapterResult` — a normalized Trace paired
with an :class:`AdapterCapabilities` manifest.

The capabilities manifest answers the question: "given the data that was
recoverable from this source format, which TrajEval analysis modules can
actually run?"

Two-tier schema
---------------
The rich Trace produced by the SDK callback (``sdk/callback.py``) has full
fidelity: cost_usd, reasoning_text, duration_ms, depth, structured I/O.
Adapter-produced Traces are sparse — fields that cannot be recovered from the
source format are set to their zero values and reflected in capabilities.

The analysis engine must check capabilities before dispatching to modules that
require specific fields (e.g. ``reasoning_consistency`` needs ``reasoning_text``
on at least one node; ``cost_risk`` needs non-zero ``cost_usd`` values).

60 / 40 analysis coverage split
---------------------------------
Adapters are the **zero-friction entry point**: parse existing logs and get
immediate structural correctness — no code changes required.  The SDK callback
is the **upgrade path** when you need semantic depth.

**Adapters deliver ~60% of analyses** (all structural checks):

- ``structural``  — tool ordering, topological constraints
- ``liveness``    — required tool visits, reachability
- ``safety``      — hard-stop / never-call rules
- ``ltl``         — linear temporal logic properties
- ``invariants``  — always-true and never-true assertions
- ``counterfactual`` — what-if trajectory mutations
- ``bisimulation`` — trace equivalence across runs
- ``synthesis``   — contract-driven trace generation
- ``cost_risk``   — when adapter recovers token counts (OTel / usage dict)
- ``latency_risk`` — when adapter recovers timestamps (OTel)

**SDK instrumentation unlocks the remaining ~40%** (semantic depth analyses):

- ``reasoning_consistency`` — LLM stated belief vs tool output agreement
- ``full_regret``            — per-step cost-opportunity analysis
- ``calibration``            — predicted vs realised outcome probability
- ``embedding_drift``        — semantic shift across trajectory steps
- ``probabilistic_safety``   — confidence-weighted safety violation scoring

See :meth:`AdapterCapabilities.analyses_requiring_sdk` for the explicit list.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class AdapterCapabilities:
    """Field-availability manifest for an adapter-produced Trace.

    Attributes
    ----------
    has_cost:
        At least one node has a non-zero cost_usd. Enables cost_risk analysis.
    has_reasoning:
        At least one node has reasoning_text. Enables reasoning_consistency.
    has_structured_io:
        tool_input / tool_output are structured dicts (not raw strings).
        Required for semantic assertion checks.
    has_latency:
        At least one node has a non-zero duration_ms. Enables latency_risk.
    has_depth:
        depth fields are populated. Enables max_depth assertions.
    """

    has_cost: bool = False
    has_reasoning: bool = False
    has_structured_io: bool = False
    has_latency: bool = False
    has_depth: bool = False

    def supported_analyses(self) -> list[str]:
        """Names of analysis modules that can run on this trace."""
        analyses = ["structural", "liveness", "safety"]
        if self.has_cost:
            analyses.append("cost_risk")
        if self.has_latency:
            analyses.append("latency_risk")
        if self.has_reasoning:
            analyses.append("reasoning_consistency")
        return analyses

    def unavailable_analyses(self) -> list[str]:
        """Names of modules that cannot run — missing required fields."""
        unavailable = []
        if not self.has_cost:
            unavailable.append("cost_risk")
        if not self.has_latency:
            unavailable.append("latency_risk")
        if not self.has_reasoning:
            unavailable.append("reasoning_consistency")
        return unavailable

    @staticmethod
    def analyses_requiring_sdk() -> list[str]:
        """Analyses that require SDK callback instrumentation, not adapters.

        These analyses need fields that no adapter can recover from logs:
        fine-grained per-step opportunity cost, calibrated probability
        metadata, semantic embeddings, and confidence scores.

        Note: ``reasoning_consistency`` is NOT in this list — the LangGraph
        and OTel adapters both recover ``reasoning_text`` from completion
        events and set ``has_reasoning=True``, so it is available via adapter
        (gated on ``has_reasoning`` in :meth:`supported_analyses`).

        The zero-friction adapter path delivers ~60% of TrajEval analyses
        (all structural checks).  Adding ``TrajEvalCallback`` to your agent
        unlocks the remaining ~40% of semantic depth analyses listed here.
        """
        return [
            "full_regret",  # needs per-step opportunity cost metadata
            "calibration",  # needs predicted probability metadata
            "embedding_drift",  # needs semantic embedding per step
            "probabilistic_safety",  # needs confidence score per tool call
        ]


@dataclass(frozen=True)
class AdapterResult:
    """Output of any TrajEval adapter.

    Attributes
    ----------
    trace:
        Normalized TrajEval Trace. Tool ordering, names, and structured I/O
        are preserved. Fields that could not be recovered are set to zero
        values (cost_usd=0.0, duration_ms=0, depth=0, reasoning_text=None).
    capabilities:
        Declares which fields are populated and which analysis modules can run.
    warnings:
        Non-fatal messages emitted during normalization (e.g. parse errors,
        skipped spans, unknown node types).
    """

    trace: Trace
    capabilities: AdapterCapabilities
    warnings: list[str] = field(default_factory=list)
