"""Programmatic evaluation harness for LangGraph agents.

Run an agent across multiple scenarios and collect trajectory metrics.

Usage::

    from trajeval.sdk.evaluate import evaluate, EvalReport, Scenario

    async def my_agent(scenario: dict, cb: TrajEvalCallback) -> None:
        result = await graph.ainvoke(
            {"query": scenario["query"]},
            config={"callbacks": [cb]},
        )

    report = await evaluate(
        agent_fn=my_agent,
        scenarios=[
            Scenario(inputs={"query": "find flights"}, label="basic-search"),
            Scenario(inputs={"query": "book hotel"}, label="hotel-booking"),
        ],
        assertions=[
            functools.partial(never_calls, tool="delete_user"),
            functools.partial(max_depth, n=10),
        ],
        n_runs=3,
    )
    print(report.summary())
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from trajeval.sdk.callback import TrajEvalCallback
from trajeval.sdk.models import Trace

# Type alias: an assertion callable takes a Trace and raises AssertionError.
AssertionFn = Callable[[Trace], None]

# Signature the caller must provide: receives (scenario_inputs, callback) and
# must invoke the agent so the callback records the trajectory.
AgentFn = Callable[
    [dict[str, Any], TrajEvalCallback],
    Coroutine[Any, Any, Any],
]


@dataclass
class Scenario:
    """A single evaluation scenario."""

    inputs: dict[str, Any]
    label: str = ""


@dataclass
class RunResult:
    """Result for one (scenario, run) pair."""

    scenario: Scenario
    run: int
    trace: Trace
    violations: list[str]

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0


@dataclass
class EvalReport:
    """Aggregated report from :func:`evaluate`."""

    results: list[RunResult] = field(default_factory=list)

    @property
    def total_runs(self) -> int:
        return len(self.results)

    @property
    def passed_runs(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_runs / self.total_runs

    @property
    def all_violations(self) -> list[str]:
        """Flatten all violation messages across every run."""
        out: list[str] = []
        for r in self.results:
            out.extend(r.violations)
        return out

    def summary(self) -> str:
        lines = [
            f"EvalReport: {self.passed_runs}/{self.total_runs} runs passed "
            f"({self.pass_rate:.0%})"
        ]
        if self.all_violations:
            lines.append("Violations:")
            for v in self.all_violations:
                lines.append(f"  - {v}")
        return "\n".join(lines)


@dataclass
class MatrixCell:
    """Result for one (scenario_label, assertion_name) cell in the matrix."""

    scenario_label: str
    assertion_name: str
    passed: bool
    violation: str | None  # AssertionError message, or None if passed
    run: int = 0


@dataclass
class EvalMatrix:
    """Cell-level report: tasks × assertions.

    Each cell captures whether a specific scenario passed a specific assertion
    on a specific run. Replaces the scalar health score with a grid that shows
    exactly which (task, assertion) pairs are failing.

    Access patterns::

        matrix.pass_rate                   # overall fraction
        matrix.scenario_pass_rate("task")  # fraction for one scenario
        matrix.assertion_pass_rate("name") # fraction for one assertion
        matrix.table()                     # ASCII grid for CI logs
        matrix.failing_cells()             # list[MatrixCell] with passed=False
        matrix.statistical_warnings        # non-empty if n_runs was too low
        matrix.bonferroni_alpha            # corrected significance threshold
    """

    cells: list[MatrixCell] = field(default_factory=list)
    statistical_warnings: list[str] = field(default_factory=list)
    bonferroni_alpha: float = 0.05

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def passed_cells(self) -> int:
        return sum(1 for c in self.cells if c.passed)

    @property
    def pass_rate(self) -> float:
        if not self.cells:
            return 0.0
        return self.passed_cells / self.total_cells

    def scenario_pass_rate(self, label: str) -> float:
        relevant = [c for c in self.cells if c.scenario_label == label]
        if not relevant:
            return 0.0
        return sum(1 for c in relevant if c.passed) / len(relevant)

    def assertion_pass_rate(self, name: str) -> float:
        relevant = [c for c in self.cells if c.assertion_name == name]
        if not relevant:
            return 0.0
        return sum(1 for c in relevant if c.passed) / len(relevant)

    def failing_cells(self) -> list[MatrixCell]:
        return [c for c in self.cells if not c.passed]

    def table(self) -> str:
        """Return an ASCII table of scenario × assertion pass/fail results.

        PASS cells show ``PASS``; FAIL cells show ``FAIL``."""
        scenario_labels = list(dict.fromkeys(c.scenario_label for c in self.cells))
        assertion_names = list(dict.fromkeys(c.assertion_name for c in self.cells))

        if not scenario_labels or not assertion_names:
            return "(empty matrix)"

        # Build lookup: (scenario, assertion) → status
        lookup: dict[tuple[str, str], str] = {}
        for cell in self.cells:
            key = (cell.scenario_label, cell.assertion_name)
            # If any run failed, mark FAIL; else PASS
            if not cell.passed:
                lookup[key] = "FAIL"
            elif key not in lookup:
                lookup[key] = "PASS"

        col_w = max(len(n) for n in assertion_names)
        row_label_w = max(len(s) for s in scenario_labels)
        cols = "  ".join(n.ljust(col_w) for n in assertion_names)
        header = " " * (row_label_w + 2) + cols
        sep = "-" * len(header)
        rows = [header, sep]
        for sl in scenario_labels:
            cells_row = "  ".join(
                lookup.get((sl, an), "    ").ljust(col_w) for an in assertion_names
            )
            rows.append(f"{sl.ljust(row_label_w)}  {cells_row}")
        return "\n".join(rows)

    def summary(self) -> str:
        return (
            f"EvalMatrix: {self.passed_cells}/{self.total_cells} cells passed "
            f"({self.pass_rate:.0%}). "
            f"{len(self.failing_cells())} failing cell(s)."
        )


async def eval_matrix(
    agent_fn: AgentFn,
    scenarios: list[Scenario],
    assertions: list[AssertionFn],
    *,
    assertion_names: list[str] | None = None,
    n_runs: int = 1,
    agent_id: str = "eval",
    version_hash: str = "",
    concurrency: int = 1,
    min_runs_warning: int = 3,
) -> EvalMatrix:
    """Run agent against each scenario and build a tasks × assertions matrix.

    Each cell in the matrix represents one (scenario, assertion) pair.
    Unlike :func:`evaluate`, which aggregates everything into pass_rate,
    :func:`eval_matrix` keeps the cell-level breakdown so you can see exactly
    which task+assertion combinations are failing.

    Statistical validity
    --------------------
    With non-deterministic LLM agents, a single run per scenario produces noisy
    results.  This function enforces a minimum via *min_runs_warning*: if
    ``n_runs < min_runs_warning``, a statistical warning is attached to the
    matrix.  The matrix is still returned — the warning is not a hard error —
    but callers should inspect ``matrix.statistical_warnings`` before acting on
    cell-level pass rates.

    When more than one assertion is evaluated, the raw per-cell pass rate has
    an inflated false-positive probability (multiple testing problem).  The
    matrix exposes a Bonferroni-corrected significance threshold via
    ``matrix.bonferroni_alpha`` so callers can apply the correction themselves.

    Parameters
    ----------
    agent_fn:
        Async callable ``(inputs, callback) -> Any``.
    scenarios:
        Scenarios to evaluate.  Each scenario's ``label`` becomes the row key.
    assertions:
        Assertion functions.  Each function's ``__name__`` (or *assertion_names*
        override) becomes the column key.
    assertion_names:
        Optional explicit names for the assertion columns.  Must have the same
        length as *assertions* if provided.
    n_runs, agent_id, version_hash, concurrency:
        Same as :func:`evaluate`.
    min_runs_warning:
        Minimum recommended n_runs for statistical validity.  A warning is
        attached when ``n_runs < min_runs_warning``.  Default 3.

    Returns
    -------
    EvalMatrix
        Cell-level grid with statistical warnings and Bonferroni alpha.
    """
    if assertion_names is not None and len(assertion_names) != len(assertions):
        raise ValueError("assertion_names must have the same length as assertions")
    names = assertion_names or [
        getattr(fn, "__name__", f"assertion_{i}") for i, fn in enumerate(assertions)
    ]

    # Statistical validity warnings
    stat_warnings: list[str] = []
    if n_runs < min_runs_warning:
        stat_warnings.append(
            f"n_runs={n_runs} is below the recommended minimum of "
            f"{min_runs_warning} for statistical validity. "
            f"Pass rates computed from fewer than {min_runs_warning} runs "
            f"per scenario have wide confidence intervals and may not be "
            f"reproducible. Increase n_runs for reliable results."
        )

    # Bonferroni correction: if you run k assertions, the per-assertion
    # significance threshold is alpha/k to maintain family-wise error rate.
    # We expose this as informational metadata; callers apply it themselves.
    n_assertions = len(assertions)
    family_alpha = 0.05
    bonferroni_alpha = family_alpha / n_assertions if n_assertions > 0 else family_alpha

    if n_assertions > 1:
        stat_warnings.append(
            f"Multiple testing ({n_assertions} assertions): Bonferroni-corrected "
            f"per-assertion significance threshold is "
            f"alpha/{n_assertions} = {bonferroni_alpha:.4f} "
            f"(family-wise alpha={family_alpha}). "
            f"Consider this when interpreting per-assertion pass rates."
        )

    # Pass no assertions to evaluate() — we run them ourselves per-cell below
    # to get cell-level pass/fail. Passing them to evaluate() would run each
    # assertion twice per trace with no benefit.
    report = await evaluate(
        agent_fn,
        scenarios,
        assertions=None,
        n_runs=n_runs,
        agent_id=agent_id,
        version_hash=version_hash,
        concurrency=concurrency,
    )

    matrix = EvalMatrix(
        statistical_warnings=stat_warnings,
        bonferroni_alpha=bonferroni_alpha,
    )
    for run_result in report.results:
        for i, assert_fn in enumerate(assertions):
            violation: str | None = None
            passed = True
            try:
                assert_fn(run_result.trace)
            except AssertionError as exc:
                passed = False
                violation = str(exc)
            matrix.cells.append(
                MatrixCell(
                    scenario_label=run_result.scenario.label or f"scenario_{i}",
                    assertion_name=names[i],
                    passed=passed,
                    violation=violation,
                    run=run_result.run,
                )
            )

    return matrix


async def evaluate(
    agent_fn: AgentFn,
    scenarios: list[Scenario],
    *,
    assertions: list[AssertionFn] | None = None,
    n_runs: int = 1,
    agent_id: str = "eval",
    version_hash: str = "",
    concurrency: int = 1,
) -> EvalReport:
    """Run *agent_fn* across all *scenarios* × *n_runs* and evaluate trajectories.

    Parameters
    ----------
    agent_fn:
        Async callable with signature ``(inputs, callback) -> Any``.
        Must invoke the agent so the *callback* records the full trajectory.
    scenarios:
        List of :class:`Scenario` objects to evaluate.
    assertions:
        Assertion functions to run on each resulting trace.  Any assertion
        that raises ``AssertionError`` counts as a violation.
    n_runs:
        Number of times to run each scenario.  Use ``>1`` to measure variance.
    agent_id:
        Identifier embedded in every :class:`~trajeval.sdk.models.Trace`.
    version_hash:
        Git SHA or other version tag for the agent under test.
    concurrency:
        Maximum number of agent invocations running in parallel.  Defaults to
        1 (sequential) for deterministic ordering.

    Returns
    -------
    EvalReport
        Aggregated results; see :class:`EvalReport` for details.
    """
    active_assertions = assertions or []
    semaphore = asyncio.Semaphore(max(1, concurrency))
    report = EvalReport()

    async def _run_one(scenario: Scenario, run_idx: int) -> RunResult:
        cb = TrajEvalCallback(
            agent_id=agent_id,
            version_hash=version_hash,
            mode="observe",
        )
        async with semaphore:
            await agent_fn(scenario.inputs, cb)
        trace = cb.get_trace()
        violations: list[str] = []
        for assert_fn in active_assertions:
            try:
                assert_fn(trace)
            except AssertionError as exc:
                violations.append(str(exc))
        return RunResult(
            scenario=scenario,
            run=run_idx,
            trace=trace,
            violations=violations,
        )

    tasks = [
        _run_one(scenario, run_idx)
        for scenario in scenarios
        for run_idx in range(n_runs)
    ]
    run_results = await asyncio.gather(*tasks)
    report.results.extend(run_results)
    return report
