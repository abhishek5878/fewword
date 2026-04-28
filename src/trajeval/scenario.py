"""Scenario harness — reproducible pass/fail pre-release tests.

Inspired by LangWatch's "same agent, same task, different result"
framing. A scenario is a task description + a canned sequence of
tool calls (usually the ones a reference agent has been recorded
doing) + the contract pack that must pass. Loading and running is
deterministic: ``scenario.run(path)`` produces the same
``ScenarioResult`` every time because no LLM is called — the
"agent" is replaced by the recorded trace.

This is explicitly a skeletal primitive. A real scenario suite
would layer on:
  * mutation — swap one tool call to see if the contract still
    holds.
  * LLM-judged open-ended fields (we don't do these today; they
    belong in the soft-eval layer, not here).
  * multi-turn user simulation.

For now the harness is the minimum we need to point a (trace + config + expected-verdict) tuple at the runner and turn it into a pre-release regression suite runnable in CI.

Usage::

    from trajeval.scenario import run_scenario

    result = run_scenario(Path("my_scenario.yml"))
    assert result.passed, result.summary

YAML format::

    name: PocketOS shell mutation must be blocked
    trace: ../examples/incidents/pocketos_drop_database.trace.json
    config: ../examples/incidents/pocketos_drop_database.fewwords.yml
    must_fire: [dangerous_input, user_consent]   # required — names of checks that MUST fail
    must_not_fire: []                            # optional — names of checks that MUST pass
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from trajeval.action import load_config, run_checks
from trajeval.cli import _load_trace_auto


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    fired_checks: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    unexpected_fires: list[str] = field(default_factory=list)
    summary: str = ""


def run_scenario(scenario_path: Path) -> ScenarioResult:
    """Load a scenario YAML and run it. Deterministic and
    side-effect-free beyond reading from disk.
    """
    spec = yaml.safe_load(scenario_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(
            f"Expected a mapping in {scenario_path}, got {type(spec).__name__}"
        )

    name = str(spec.get("name", scenario_path.stem))
    trace_path = (scenario_path.parent / str(spec["trace"])).resolve()
    config_path = (scenario_path.parent / str(spec["config"])).resolve()
    must_fire = [str(x) for x in spec.get("must_fire", [])]
    must_not_fire = [str(x) for x in spec.get("must_not_fire", [])]

    trace = _load_trace_auto(str(trace_path))
    if trace is None:
        return ScenarioResult(
            name=name,
            passed=False,
            summary=f"failed to parse trace at {trace_path}",
        )

    report = run_checks(trace, load_config(config_path))
    fired = [c.name for c in report.checks if not c.passed]

    missing_required = [name for name in must_fire if name not in fired]
    unexpected_fires = [
        name for name in fired
        if name in must_not_fire
    ]

    passed = not missing_required and not unexpected_fires
    if passed:
        summary = (
            f"PASS · {len(fired)} check(s) fired "
            f"({', '.join(fired) or 'none'})"
        )
    else:
        parts: list[str] = []
        if missing_required:
            parts.append(f"missing required fires: {missing_required}")
        if unexpected_fires:
            parts.append(f"unexpected fires: {unexpected_fires}")
        summary = "FAIL · " + "; ".join(parts)

    return ScenarioResult(
        name=name,
        passed=passed,
        fired_checks=fired,
        missing_required=missing_required,
        unexpected_fires=unexpected_fires,
        summary=summary,
    )


def run_scenario_dir(directory: Path) -> list[ScenarioResult]:
    """Run every scenario YAML under ``directory``.

    A file counts as a scenario if its parsed YAML is a mapping that
    contains both ``trace`` and ``config`` keys. This lets you keep
    scenario files and contract-pack files in the same directory
    without confusion.
    """
    results: list[ScenarioResult] = []
    for path in sorted(directory.glob("*.yml")):
        try:
            spec = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if (
            isinstance(spec, dict)
            and "trace" in spec
            and "config" in spec
        ):
            results.append(run_scenario(path))
    return results


def format_results(results: list[ScenarioResult]) -> str:
    lines = []
    passed = sum(1 for r in results if r.passed)
    for r in results:
        lines.append(f"{'✓' if r.passed else '✗'} {r.name}: {r.summary}")
    lines.append(f"\n{passed}/{len(results)} scenarios passed.")
    return "\n".join(lines)


__all__ = [
    "ScenarioResult",
    "format_results",
    "run_scenario",
    "run_scenario_dir",
]
