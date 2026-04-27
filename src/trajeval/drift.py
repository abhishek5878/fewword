"""Drift monitoring — detect silent regressions across many traces.

Intern rule #3: *"Have monitoring systems in place to keep an eye on
silent failures. Spot-checking isn't enough."*

A single trace is a spot-check. Drift is the question "am I seeing
more of X than I used to?" — the thing you notice when the agent
worked fine yesterday and today 30% of responses time out but no
single trace is catastrophic.

Usage::

    from trajeval.drift import compute_drift

    report = compute_drift(
        baseline=load_traces("traces/last_week/"),
        recent=load_traces("traces/today/"),
        config=load_config(Path(".trajeval.yml")),
    )
    for entry in report.significant:
        print(f"{entry.check_name}: {entry.baseline_rate:.1%} -> "
              f"{entry.recent_rate:.1%}  ({entry.ratio:.1f}x)")

or from the CLI::

    trajeval drift traces/last_week/ traces/today/ --config .trajeval.yml

This module does no statistics beyond a ratio threshold — the point
is not to replace a proper SPC chart, but to surface obvious
regressions before any single one trips a per-trace assertion.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from trajeval.action import ActionConfig, run_checks
from trajeval.cli import _load_trace_auto
from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class CheckFireStats:
    """Fire rate for one named check across a batch of traces."""

    check_name: str
    fires: int
    total: int

    @property
    def rate(self) -> float:
        return self.fires / self.total if self.total else 0.0


@dataclass(frozen=True)
class DriftEntry:
    """One check's baseline vs recent fire rate."""

    check_name: str
    baseline_rate: float
    recent_rate: float
    baseline_n: int
    recent_n: int

    @property
    def ratio(self) -> float:
        """Recent / baseline, or ``float('inf')`` when baseline is 0."""
        if self.baseline_rate == 0.0:
            return float("inf") if self.recent_rate > 0 else 0.0
        return self.recent_rate / self.baseline_rate

    @property
    def delta(self) -> float:
        return self.recent_rate - self.baseline_rate


@dataclass(frozen=True)
class DriftReport:
    baseline_count: int
    recent_count: int
    entries: list[DriftEntry]
    ratio_threshold: float = 2.0
    min_delta: float = 0.05

    @property
    def significant(self) -> list[DriftEntry]:
        """Entries where recent rate is >= 2x baseline AND the absolute
        change is at least 5 percentage points.

        The 5pp floor keeps us from flagging "0.1% → 0.3%" as a 3x
        regression — technically true, practically noise.
        """
        return [
            e for e in self.entries
            if (
                e.ratio >= self.ratio_threshold
                and abs(e.delta) >= self.min_delta
            )
        ]


def _fire_stats(
    traces: list[Trace], config: ActionConfig
) -> dict[str, CheckFireStats]:
    """Count, per check name, how many traces had that check *fail*."""
    fires: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    for trace in traces:
        report = run_checks(trace, config)
        for check in report.checks:
            seen[check.name] += 1
            if not check.passed:
                fires[check.name] += 1
    return {
        name: CheckFireStats(
            check_name=name,
            fires=fires.get(name, 0),
            total=seen[name],
        )
        for name in seen
    }


def compute_drift(
    baseline: list[Trace],
    recent: list[Trace],
    config: ActionConfig,
    *,
    ratio_threshold: float = 2.0,
    min_delta: float = 0.05,
) -> DriftReport:
    """Compare per-check fire rates across two trace batches.

    A check is considered to have "drifted" when its recent fire rate
    is at least ``ratio_threshold`` times the baseline AND the absolute
    change is at least ``min_delta`` (default 0.05 = 5 percentage
    points).
    """
    base_stats = _fire_stats(baseline, config)
    recent_stats = _fire_stats(recent, config)
    all_names = sorted(set(base_stats) | set(recent_stats))

    entries = [
        DriftEntry(
            check_name=name,
            baseline_rate=base_stats[name].rate if name in base_stats else 0.0,
            recent_rate=recent_stats[name].rate
            if name in recent_stats
            else 0.0,
            baseline_n=base_stats[name].total if name in base_stats else 0,
            recent_n=recent_stats[name].total
            if name in recent_stats
            else 0,
        )
        for name in all_names
    ]
    return DriftReport(
        baseline_count=len(baseline),
        recent_count=len(recent),
        entries=entries,
        ratio_threshold=ratio_threshold,
        min_delta=min_delta,
    )


def load_traces_from_dir(directory: Path) -> list[Trace]:
    """Load every ``*.json`` under ``directory`` as a TrajEval Trace.

    Uses the same auto-detect path as the CLI, so mixed formats are OK.
    Skips files that fail to parse, printing a one-line note for each
    so missing traces don't silently shrink the sample.
    """
    traces: list[Trace] = []
    for path in sorted(directory.rglob("*.json")):
        trace = _load_trace_auto(str(path))
        if trace is not None:
            traces.append(trace)
    return traces


def report_to_dict(report: DriftReport) -> dict[str, object]:
    """Serialize a :class:`DriftReport` to a JSON-safe dict."""
    return {
        "baseline_count": report.baseline_count,
        "recent_count": report.recent_count,
        "ratio_threshold": report.ratio_threshold,
        "min_delta": report.min_delta,
        "entries": [
            {
                "check_name": e.check_name,
                "baseline_rate": e.baseline_rate,
                "recent_rate": e.recent_rate,
                "baseline_n": e.baseline_n,
                "recent_n": e.recent_n,
                "ratio": e.ratio if e.ratio != float("inf") else "inf",
                "delta": e.delta,
                "significant": e in report.significant,
            }
            for e in report.entries
        ],
    }


def format_report(report: DriftReport) -> str:
    """Human-readable single-screen summary. Used by the CLI."""
    lines = [
        f"Drift: baseline={report.baseline_count} trace(s) · "
        f"recent={report.recent_count} trace(s)",
        "",
        f"{'check':25s}{'base':>9s}{'recent':>9s}{'delta':>9s}"
        f"{'ratio':>8s}  note",
    ]
    for e in report.entries:
        ratio = f"{e.ratio:.1f}x" if e.ratio != float("inf") else "  inf"
        flag = (
            "REGRESSION" if e in report.significant and e.delta > 0
            else "improved" if e in report.significant
            else ""
        )
        lines.append(
            f"{e.check_name:25s}"
            f"{e.baseline_rate * 100:>7.1f}%  "
            f"{e.recent_rate * 100:>6.1f}%  "
            f"{e.delta * 100:>+6.1f}pp  "
            f"{ratio:>7s}  {flag}"
        )
    lines.append("")
    sig_count = len(report.significant)
    regressions = [e for e in report.significant if e.delta > 0]
    if regressions:
        lines.append(
            f"{len(regressions)} regression(s) detected "
            f"(>= {report.ratio_threshold}x baseline, "
            f">= {report.min_delta * 100:.0f}pp change)."
        )
    elif sig_count:
        lines.append(f"{sig_count} check(s) improved, none regressed.")
    else:
        lines.append("No significant drift.")
    return "\n".join(lines)


__all__ = [
    "CheckFireStats",
    "DriftEntry",
    "DriftReport",
    "compute_drift",
    "format_report",
    "load_traces_from_dir",
    "report_to_dict",
]


# keep json import used in report_to_dict explicit-imported for clarity
_ = json
