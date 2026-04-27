"""Discovery mode — zero-config observation that grows into enforcement.

The "no rules needed to start" path from the R2 sprint. A user records
traces to ``.trajeval/traces/`` (via this module's helpers or by
writing JSON files manually); once a threshold of traces accumulates,
``synthesize_discovered()`` mines contract candidates from them. The
intent: observe N real runs, then graduate the observations to
enforcement via ``.trajeval.yml``.

Contrast with :mod:`trajeval.initializer`: ``initializer`` is a one-shot "look at
what's already here and write a config." ``discover`` is the ongoing
loop that feeds ``init`` fresh signal as the agent runs in prod.

Layer: this module imports from SDK, adapters, and analysis only.
No FastAPI, no DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from trajeval.adapters.auto import auto_detect
from trajeval.analysis.auto_contract import ContractSuggestion, suggest_contracts
from trajeval.sdk.models import Trace

DEFAULT_DISCOVER_DIR = Path(".trajeval") / "traces"
DEFAULT_THRESHOLD = 10


@dataclass(frozen=True)
class DiscoveryStatus:
    """Snapshot of the current discovery corpus."""

    base_dir: Path
    total_traces: int
    parseable_traces: int
    agent_ids: list[str]
    tool_counts: dict[str, int]
    ready_for_synthesis: bool
    threshold: int


def record_trace(
    trace: Trace | dict[str, Any],
    *,
    base_dir: Path | str = DEFAULT_DISCOVER_DIR,
) -> Path:
    """Persist a single ``Trace`` (or raw dict) to the discovery corpus.

    Filename is ``{timestamp}_{trace_id}.json`` so listings sort by
    time naturally and collisions across concurrent runs are unlikely.
    Returns the written path.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    if isinstance(trace, Trace):
        payload = json.loads(trace.model_dump_json())
        trace_id = trace.trace_id
    else:
        payload = trace
        trace_id = str(trace.get("trace_id", "unknown"))
    # Microsecond resolution avoids filename collisions when the same
    # trace_id is recorded multiple times within a second (happens in
    # rapid replay / test scenarios).
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = base / f"{stamp}_{trace_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.debug(f"discover: recorded trace to {path}")
    return path


def _iter_traces(base: Path) -> list[Path]:
    if not base.exists() or not base.is_dir():
        return []
    return sorted(base.rglob("*.json"))


def _parse(path: Path) -> Trace | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(f"discover: unreadable {path}: {exc!r}")
        return None
    try:
        return auto_detect(data).trace
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"discover: unparseable {path}: {exc!r}")
        return None


def load_discovered(
    *,
    base_dir: Path | str = DEFAULT_DISCOVER_DIR,
    agent_id: str | None = None,
) -> list[Trace]:
    """Load all parseable traces from the discovery corpus.

    ``agent_id`` filters to traces whose ``trace.agent_id`` matches
    exactly. ``None`` returns every trace.
    """
    paths = _iter_traces(Path(base_dir))
    traces: list[Trace] = []
    for p in paths:
        t = _parse(p)
        if t is None:
            continue
        if agent_id is not None and t.agent_id != agent_id:
            continue
        traces.append(t)
    return traces


def discovery_status(
    *,
    base_dir: Path | str = DEFAULT_DISCOVER_DIR,
    agent_id: str | None = None,
    threshold: int = DEFAULT_THRESHOLD,
) -> DiscoveryStatus:
    """Summarize the discovery corpus — count, agent IDs, tool frequency."""
    base = Path(base_dir)
    paths = _iter_traces(base)
    traces = load_discovered(base_dir=base, agent_id=agent_id)
    agent_ids = sorted({t.agent_id for t in traces if t.agent_id})
    tool_counts: dict[str, int] = {}
    for t in traces:
        for n in t.nodes:
            if n.node_type == "tool_call" and n.tool_name:
                tool_counts[n.tool_name] = tool_counts.get(n.tool_name, 0) + 1
    return DiscoveryStatus(
        base_dir=base.resolve(),
        total_traces=len(paths),
        parseable_traces=len(traces),
        agent_ids=agent_ids,
        tool_counts=tool_counts,
        ready_for_synthesis=len(traces) >= threshold,
        threshold=threshold,
    )


def synthesize_discovered(
    *,
    base_dir: Path | str = DEFAULT_DISCOVER_DIR,
    agent_id: str | None = None,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[ContractSuggestion]:
    """Mine contract suggestions from the discovered corpus.

    Returns an empty list when the corpus is below ``threshold`` — the
    caller can use :class:`DiscoveryStatus` to tell the user how many
    more runs they need.
    """
    traces = load_discovered(base_dir=base_dir, agent_id=agent_id)
    if len(traces) < threshold:
        return []
    return suggest_contracts(traces, [])


class Discovery:
    """Context-manager helper for hands-off recording.

    ::

        from trajeval.discover import Discovery

        with Discovery(agent_id="my-agent") as d:
            trace = run_my_agent(...)
            d.record(trace)

    On ``__exit__`` we surface a one-line status summary to the log so
    users see when they cross the synthesis threshold without pulling
    up a dashboard.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        *,
        base_dir: Path | str = DEFAULT_DISCOVER_DIR,
        threshold: int = DEFAULT_THRESHOLD,
    ) -> None:
        self.agent_id = agent_id
        self.base_dir = Path(base_dir)
        self.threshold = threshold
        self._recorded_paths: list[Path] = []

    def __enter__(self) -> Discovery:
        return self

    def __exit__(self, *args: object) -> None:
        if not self._recorded_paths:
            return
        status = discovery_status(
            base_dir=self.base_dir,
            agent_id=self.agent_id,
            threshold=self.threshold,
        )
        if status.ready_for_synthesis:
            logger.info(
                f"discover: {status.parseable_traces} trace(s) available — "
                f"run `trajeval discover suggest` to mine contract rules."
            )
        else:
            need = max(0, self.threshold - status.parseable_traces)
            logger.info(
                f"discover: {status.parseable_traces}/{self.threshold} "
                f"trace(s) recorded — need {need} more before synthesis."
            )

    def record(self, trace: Trace | dict[str, Any]) -> Path:
        path = record_trace(trace, base_dir=self.base_dir)
        self._recorded_paths.append(path)
        return path
