"""TrajEvalClient — HTTP client for shipping traces to the TrajEval backend.

Layer rule: SDK package.  Zero imports from trajeval.backend.

Features
--------
- Retry with exponential back-off on transient errors (5xx, network failures).
- Dead-letter queue (DLQ): traces that exhaust all retries are written as
  JSON files under ``dlq_dir`` for later replay.

Usage::

    async with TrajEvalClient("http://localhost:8000") as client:
        await client.send_trace(trace)
        trace = await client.get_trace("trace-001")

    # With retry + DLQ:
    client = TrajEvalClient(
        "http://localhost:8000",
        retry_config=RetryConfig(max_retries=5),
        dlq_dir=Path("/var/lib/trajeval/dlq"),
        api_key="te_mysecret",
    )
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from trajeval.sdk.models import Trace

_DEFAULT_TIMEOUT = 10.0


class TrajEvalClientError(Exception):
    """Raised when the backend returns an unexpected HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class TraceConflictError(TrajEvalClientError):
    """Raised when a trace with the same trace_id already exists (409)."""


class DeadLetterError(Exception):
    """Raised when send_trace fails and DLQ write also fails."""


@dataclass
class RetryConfig:
    """Exponential back-off configuration for :class:`TrajEvalClient`.

    Attributes
    ----------
    max_retries:
        Maximum number of retry attempts after the initial failure.
        Total attempts = ``max_retries + 1``.
    initial_delay:
        Seconds to wait before the first retry.
    max_delay:
        Upper bound on the inter-retry delay (seconds).
    backoff_factor:
        Multiplier applied to the delay after each retry.
    """

    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 10.0
    backoff_factor: float = 2.0

    # Derived sequence lazily — not stored
    def delays(self) -> list[float]:
        """Return the list of per-retry sleep durations."""
        delays: list[float] = []
        delay = self.initial_delay
        for _ in range(self.max_retries):
            delays.append(delay)
            delay = min(delay * self.backoff_factor, self.max_delay)
        return delays


_DEFAULT_RETRY = RetryConfig()


class TrajEvalClient:
    """Async HTTP client for the TrajEval backend.

    Parameters
    ----------
    base_url:
        Root URL of the TrajEval backend, e.g. ``"http://localhost:8000"``.
    timeout:
        Per-request timeout in seconds.
    retry_config:
        Back-off settings for :meth:`send_trace`.  Defaults to 3 retries.
    dlq_dir:
        If set, traces that exhaust all retries are written to this directory
        as ``{trace_id}.json`` files.  The directory is created if absent.
    api_key:
        Optional bearer token sent as ``Authorization: Bearer <key>``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        retry_config: RetryConfig | None = None,
        dlq_dir: Path | None = None,
        api_key: str | None = None,
    ) -> None:
        headers: dict[str, str] = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
        )
        self._retry = retry_config or _DEFAULT_RETRY
        self._dlq_dir = dlq_dir

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    async def send_trace(self, trace: Trace) -> None:
        """POST *trace* to ``/traces`` with automatic retry.

        On 409 raises :exc:`TraceConflictError` immediately (no retry).
        On exhausted retries writes to DLQ if configured, then re-raises.
        Raises :exc:`DeadLetterError` if the DLQ write itself fails.
        """
        delays = self._retry.delays()
        last_exc: Exception = TrajEvalClientError(0, "no attempt made")

        for attempt in range(self._retry.max_retries + 1):
            try:
                await self._send_once(trace)
                return
            except TraceConflictError:
                raise  # 409 is not retryable
            except (httpx.TransportError, TrajEvalClientError) as exc:
                last_exc = exc
                if attempt < self._retry.max_retries:
                    await asyncio.sleep(delays[attempt])

        # All retries exhausted
        self._write_to_dlq(trace)
        raise last_exc

    async def get_trace(self, trace_id: str) -> Trace | None:
        """GET ``/traces/{trace_id}``.

        Returns the :class:`~trajeval.sdk.models.Trace` or ``None`` on 404.
        Raises :exc:`TrajEvalClientError` on any other non-2xx response.
        """
        response = await self._client.get(f"/traces/{trace_id}")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            raise TrajEvalClientError(response.status_code, detail)
        return Trace.model_validate(response.json())

    async def list_agent_traces(
        self,
        agent_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Trace]:
        """GET ``/agents/{agent_id}/traces`` with pagination.

        Returns a list of :class:`~trajeval.sdk.models.Trace` objects.
        """
        response = await self._client.get(
            f"/agents/{agent_id}/traces",
            params={"limit": limit, "offset": offset},
        )
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            raise TrajEvalClientError(response.status_code, detail)
        data = response.json()
        return [Trace.model_validate(t) for t in data.get("traces", [])]

    # ------------------------------------------------------------------
    # DLQ helpers
    # ------------------------------------------------------------------

    def _write_to_dlq(self, trace: Trace) -> None:
        """Write *trace* to the dead-letter directory.

        No-op when ``dlq_dir`` is not configured.
        Raises :exc:`DeadLetterError` if the write fails.
        """
        if self._dlq_dir is None:
            return
        try:
            self._dlq_dir.mkdir(parents=True, exist_ok=True)
            dest = self._dlq_dir / f"{trace.trace_id}.json"
            dest.write_text(trace.model_dump_json(), encoding="utf-8")
        except OSError as exc:
            raise DeadLetterError(
                f"Failed to write trace '{trace.trace_id}' to DLQ: {exc}"
            ) from exc

    def drain_dlq(self) -> list[Trace]:
        """Load all traces from the dead-letter directory.

        Returns an empty list when no DLQ is configured or the directory is
        empty.  Malformed files are silently skipped (logged at WARNING).
        """
        if self._dlq_dir is None or not self._dlq_dir.exists():
            return []
        traces: list[Trace] = []
        for path in sorted(self._dlq_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                traces.append(Trace.model_validate(data))
            except Exception:  # noqa: BLE001,S110  # reason: skip corrupt DLQ entries
                pass
        return traces

    def ack_dlq(self, trace_id: str) -> None:
        """Remove a trace from the dead-letter directory after successful replay."""
        if self._dlq_dir is None:
            return
        target = self._dlq_dir / f"{trace_id}.json"
        target.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send_once(self, trace: Trace) -> None:
        response = await self._client.post(
            "/traces",
            content=trace.model_dump_json(),
        )
        if response.status_code == 409:
            detail = response.json().get("detail", "conflict")
            raise TraceConflictError(409, detail)
        if response.status_code != 201:
            detail = response.json().get("detail", response.text)
            raise TrajEvalClientError(response.status_code, detail)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> TrajEvalClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
