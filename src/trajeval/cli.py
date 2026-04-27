"""TrajEval command-line interface.

Provides a ``trajeval`` command with three subcommands:

``trajeval check <trace.json> [--assertion <name>] [--severity P0]``
    Load a JSON trace file and run assertions against it.  Exits non-zero
    on any violation.  Integrates directly into CI without Python glue code.

``trajeval report <trace.json> [--format text|json]``
    Compute TRACE framework metrics for a trace and print a summary report.
    Useful for debugging a single run or comparing two traces offline.

``trajeval version``
    Print the TrajEval package version and exit.

Installation
------------
After ``pip install trajeval`` the ``trajeval`` command is available.
For development, ``uv run trajeval check trace.json``.

Layer rule: CLI may import from SDK and analysis; never from backend.

Examples
--------
::

    # Block a CI step if any safety assertion fires
    trajeval check agent_run.json --assertion never_calls:delete_user

    # Print metrics for a trace
    trajeval report agent_run.json

    # Assert ordering in a trace file
    trajeval check agent_run.json \\
        --assertion tool_must_precede:search:before=book \\
        --assertion max_depth:10

    # JSON output for machine consumption
    trajeval report agent_run.json --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from trajeval.sdk.models import Trace

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* (or ``sys.argv[1:]``) and run the requested subcommand.

    Returns the exit code (0 = success, non-zero = failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)  # type: ignore[no-any-return]


def _build_parser() -> argparse.ArgumentParser:
    # Use the actual binary name the user invoked (``fewwords`` or
    # ``trajeval`` — they're aliases) so the usage line in --help
    # matches what they typed.
    import os
    import sys

    prog_name = os.path.basename(sys.argv[0]) or "fewwords"
    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="fewwords — deterministic trajectory contracts for production agents",
    )
    sub = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # check
    # ------------------------------------------------------------------
    check_p = sub.add_parser(
        "check",
        help="Run assertions against a trace JSON file.",
    )
    check_p.add_argument(
        "trace_file",
        metavar="TRACE.json",
        help="Path to a JSON file containing a serialised TrajEval Trace.",
    )
    check_p.add_argument(
        "--assertion",
        dest="assertions",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Assertion to run, specified as NAME or NAME:ARG or NAME:ARG:KEY=VAL. "
            "Supported: never_calls:TOOL, must_visit:TOOL[,TOOL...], "
            "tool_must_precede:TOOL:before=TOOL, max_depth:N, "
            "cost_within:AMOUNT, latency_within:MS, no_cycles. "
            "Repeat --assertion to run multiple checks."
        ),
    )
    check_p.add_argument(
        "--severity",
        choices=["P0", "P1", "P2"],
        default=None,
        help="Tag all assertions with this severity level (optional).",
    )
    check_p.set_defaults(func=_cmd_check)

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    report_p = sub.add_parser(
        "report",
        help="Print TRACE framework metrics for a trace file.",
    )
    report_p.add_argument(
        "trace_file",
        metavar="TRACE.json",
        help="Path to a JSON file containing a serialised TrajEval Trace.",
    )
    report_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (default) or 'json'.",
    )
    report_p.add_argument(
        "--max-steps",
        type=int,
        default=50,
        metavar="N",
        help="Reference max node count for step_economy normalisation (default 50).",
    )
    report_p.set_defaults(func=_cmd_report)

    # ------------------------------------------------------------------
    # run (action — the boring version)
    # ------------------------------------------------------------------
    run_p = sub.add_parser(
        "run",
        help="Run all checks from .trajeval.yml against a trace file.",
    )
    run_p.add_argument(
        "trace_file",
        metavar="TRACE.json",
        help="Path to a JSON trace file.",
    )
    run_p.add_argument(
        "--config",
        default=".trajeval.yml",
        metavar="PATH",
        help="Path to YAML config (default: .trajeval.yml).",
    )
    run_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    run_p.set_defaults(func=_cmd_run)

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------
    bench_p = sub.add_parser(
        "benchmark",
        help="Run the detection rate benchmark against a trace directory.",
    )
    bench_p.add_argument(
        "trace_dir",
        metavar="DIR",
        help="Directory containing benchmark trace JSON files.",
    )
    bench_p.add_argument(
        "--config",
        default="benchmarks/config.yml",
        metavar="PATH",
        help="Path to benchmark config YAML.",
    )
    bench_p.set_defaults(func=_cmd_benchmark)

    # ------------------------------------------------------------------
    # suggest
    # ------------------------------------------------------------------
    sug_p = sub.add_parser(
        "suggest",
        help="Suggest which contract pack fits a trace + map tool names.",
    )
    sug_p.add_argument(
        "trace_file",
        metavar="TRACE.json",
        help="Path to a trace file.",
    )
    sug_p.set_defaults(func=_cmd_suggest)

    # ------------------------------------------------------------------
    # dogfood
    # ------------------------------------------------------------------
    dog_p = sub.add_parser(
        "dogfood",
        help="Self-evaluate: run TrajEval on a trace, then evaluate the evaluation.",
    )
    dog_p.add_argument(
        "trace_file",
        metavar="TRACE.json",
        help="Path to a trace file to dogfood against.",
    )
    dog_p.add_argument(
        "--config",
        default=".trajeval.yml",
        metavar="PATH",
        help="Path to YAML config (default: .trajeval.yml).",
    )
    dog_p.set_defaults(func=_cmd_dogfood)

    # ------------------------------------------------------------------
    # preflight
    # ------------------------------------------------------------------
    pre_p = sub.add_parser(
        "preflight",
        help="Pre-flight checks: validate tools vs assertions before any trace.",
    )
    pre_p.add_argument(
        "--tools",
        required=True,
        help="Comma-separated list of the agent's registered tool names.",
    )
    pre_p.add_argument(
        "--assertion",
        dest="assertions",
        action="append",
        default=[],
        metavar="SPEC",
        help="Assertion spec (same format as 'check' subcommand).",
    )
    pre_p.set_defaults(func=_cmd_preflight)

    # ------------------------------------------------------------------
    # drift
    # ------------------------------------------------------------------
    drift_p = sub.add_parser(
        "drift",
        help="Compare assertion fire rates between a baseline and a "
        "recent batch of traces.",
    )
    drift_p.add_argument(
        "baseline_dir",
        metavar="BASELINE_DIR",
        help="Directory of trace JSON files representing the "
        "baseline window (e.g. last week).",
    )
    drift_p.add_argument(
        "recent_dir",
        metavar="RECENT_DIR",
        help="Directory of trace JSON files for the recent window (e.g. today).",
    )
    drift_p.add_argument(
        "--config",
        default=".trajeval.yml",
        metavar="PATH",
        help="Path to YAML config (default: .trajeval.yml).",
    )
    drift_p.add_argument(
        "--ratio",
        type=float,
        default=2.0,
        metavar="N",
        help="Regression threshold: recent fire rate must be >= N x "
        "baseline to flag (default: 2.0).",
    )
    drift_p.add_argument(
        "--min-delta",
        type=float,
        default=0.05,
        metavar="P",
        help="Minimum absolute change (as a fraction) required to "
        "flag drift (default: 0.05 = 5pp).",
    )
    drift_p.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit a JSON report instead of the human table.",
    )
    drift_p.set_defaults(func=_cmd_drift)

    # ------------------------------------------------------------------
    # scenario
    # ------------------------------------------------------------------
    scn_p = sub.add_parser(
        "scenario",
        help="Run reproducible scenario regressions (trace + config + "
        "must-fire assertions).",
    )
    scn_p.add_argument(
        "path",
        metavar="PATH",
        help="Scenario file (YAML) or directory of scenario files.",
    )
    scn_p.set_defaults(func=_cmd_scenario)

    # ------------------------------------------------------------------
    # discover — observation mode (R2 Phase C)
    # ------------------------------------------------------------------
    disc_p = sub.add_parser(
        "discover",
        help="Discovery mode: inspect the .trajeval/traces/ corpus and "
        "optionally synthesize contract suggestions.",
    )
    disc_sub = disc_p.add_subparsers(dest="subcommand")

    disc_status_p = disc_sub.add_parser(
        "status", help="Show how many traces are in the discovery corpus."
    )
    disc_status_p.add_argument(
        "--base-dir",
        default=".trajeval/traces",
        metavar="DIR",
        help="Discovery corpus directory (default: .trajeval/traces).",
    )
    disc_status_p.add_argument(
        "--agent-id",
        default=None,
        help="Filter to a single agent_id (default: include all).",
    )
    disc_status_p.add_argument(
        "--threshold",
        type=int,
        default=10,
        metavar="N",
        help="Trace count required before synthesis unlocks (default 10).",
    )
    disc_status_p.set_defaults(func=_cmd_discover_status)

    disc_sug_p = disc_sub.add_parser(
        "suggest",
        help="Mine contract candidates from the discovery corpus.",
    )
    disc_sug_p.add_argument(
        "--base-dir",
        default=".trajeval/traces",
        metavar="DIR",
        help="Discovery corpus directory (default: .trajeval/traces).",
    )
    disc_sug_p.add_argument(
        "--agent-id",
        default=None,
        help="Filter to a single agent_id (default: include all).",
    )
    disc_sug_p.add_argument(
        "--threshold",
        type=int,
        default=10,
        metavar="N",
        help="Trace count required before synthesis unlocks (default 10).",
    )
    disc_sug_p.set_defaults(func=_cmd_discover_suggest)

    # ------------------------------------------------------------------
    # init — one-command onboarding
    # ------------------------------------------------------------------
    init_p = sub.add_parser(
        "init",
        help="Scan the repo for frameworks + traces and write a starter .trajeval.yml",
    )
    init_p.add_argument(
        "--path",
        default=".",
        metavar="DIR",
        help="Directory to scan (default: current working directory).",
    )
    init_p.add_argument(
        "--write",
        action="store_true",
        help="Write the result to .trajeval.yml instead of printing.",
    )
    init_p.add_argument(
        "--max-files",
        type=int,
        default=50,
        metavar="N",
        help="Cap on trace files scanned (default: 50).",
    )
    init_p.add_argument(
        "--pack",
        default=None,
        metavar="NAME",
        help=(
            "Start from a pre-built contract pack instead of synthesizing "
            "from scratch. Available: sales, code, support, financial, "
            "rag, data-pipeline, healthcare, legal, generic. "
            "Pass --list-packs to enumerate."
        ),
    )
    init_p.add_argument(
        "--list-packs",
        action="store_true",
        help="List available contract packs and exit.",
    )
    init_p.set_defaults(func=_cmd_init)

    # ------------------------------------------------------------------
    # version
    # ------------------------------------------------------------------
    ver_p = sub.add_parser("version", help="Print TrajEval version and exit.")
    ver_p.set_defaults(func=_cmd_version)

    compose_p = sub.add_parser(
        "compose",
        help=(
            "Merge layered contract packs (base + vertical + regulatory) "
            "into a single YAML config."
        ),
    )
    compose_p.add_argument(
        "files",
        nargs="+",
        help="Pack files in merge order (later files win on scalar conflicts).",
    )
    compose_p.add_argument(
        "--out",
        default="-",
        help="Output path, or '-' for stdout (default).",
    )
    compose_p.set_defaults(func=_cmd_compose)

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    trace = _load_trace(args.trace_file)
    if trace is None:
        return 1

    assertions = _parse_assertions(args.assertions)
    if assertions is None:
        return 1

    if args.severity:
        from trajeval.assertions.core import Severity
        from trajeval.assertions.core import severity as tag

        level = Severity[args.severity]
        assertions = [
            (name, tag(fn, level=level, name=name)) for name, fn in assertions
        ]

    failures: list[tuple[str, str]] = []
    for name, fn in assertions:
        try:
            fn(trace)
        except AssertionError as exc:
            failures.append((name, str(exc)))

    if failures:
        print(
            f"FAIL — {len(failures)} violation(s) in {args.trace_file}:",
            file=sys.stderr,
        )
        for name, msg in failures:
            print(f"  [{name}] {msg}", file=sys.stderr)
        return 1

    n = len(assertions)
    print(f"OK — {n} assertion(s) passed for {args.trace_file}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    trace = _load_trace(args.trace_file)
    if trace is None:
        return 1

    from trajeval.analysis.metrics import compute_metrics

    m = compute_metrics(trace, max_steps=args.max_steps)

    if args.format == "json":
        output: dict[str, Any] = {
            "trace_id": trace.trace_id,
            "agent_id": trace.agent_id,
            "version_hash": trace.version_hash,
            "nodes": len(trace.nodes),
            "total_cost_usd": trace.total_cost_usd,
            "total_tokens": trace.total_tokens,
            "metrics": {
                "evidence_grounding": round(m.evidence_grounding, 4),
                "cognitive_quality": round(m.cognitive_quality, 4),
                "process_efficiency": round(m.process_efficiency, 4),
                "step_economy": round(m.step_economy, 4),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"=== TrajEval Report: {args.trace_file} ===")
        print(f"Trace ID    : {trace.trace_id}")
        print(f"Agent       : {trace.agent_id}  version={trace.version_hash}")
        print(f"Nodes       : {len(trace.nodes)}")
        print(f"Cost (USD)  : ${trace.total_cost_usd:.6f}")
        print(f"Tokens      : {trace.total_tokens}")
        print()
        print("TRACE Metrics:")
        print(f"  Evidence Grounding  : {m.evidence_grounding:.2%}")
        print(f"  Cognitive Quality   : {m.cognitive_quality:.2%}")
        print(f"  Process Efficiency  : {m.process_efficiency:.2%}")
        print(f"  Step Economy        : {m.step_economy:.2%}")

    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import time as _time

    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_dir():
        print(f"Error: not a directory: {trace_dir}", file=sys.stderr)
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    from trajeval.action import load_config, run_checks

    config = load_config(config_path)
    tp = fp = tn = fn = 0
    start = _time.perf_counter()

    for trace_file in sorted(trace_dir.glob("*.json")):
        trace = _load_trace_auto(str(trace_file))
        if trace is None:
            continue
        report = run_checks(trace, config)
        is_clean = trace_file.name.startswith("clean_")
        caught = not report.all_passed

        if is_clean and not caught:
            tn += 1
        elif is_clean and caught:
            fp += 1
            fails = [c.name for c in report.checks if not c.passed]
            print(f"  FP {trace_file.name}: {fails}")
        elif not is_clean and caught:
            tp += 1
        else:
            fn += 1
            print(f"  MISS {trace_file.name}")

    elapsed = _time.perf_counter() - start
    total = tp + tn + fp + fn
    detection = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * detection / max(precision + detection, 0.001)

    print(f"\nBenchmark: {total} traces")
    print(
        f"  Detection: {detection:.0%} | FPR: {fpr:.0%} | "
        f"Precision: {precision:.0%} | F1: {f1:.2f}"
    )
    print(
        f"  Latency: {elapsed * 1000:.0f}ms "
        f"({elapsed * 1000 / max(total, 1):.1f}ms/trace)"
    )

    return 0 if fn == 0 and fp == 0 else 1


def _cmd_suggest(args: argparse.Namespace) -> int:
    trace = _load_trace_auto(args.trace_file)
    if trace is None:
        return 1

    from trajeval.analysis.contract_suggest import suggest_pack

    suggestion = suggest_pack(trace)
    tools = sorted(
        {n.tool_name for n in trace.nodes if n.node_type == "tool_call" and n.tool_name}
    )

    print(f"Trace tools: {tools}")

    if not suggestion.is_confident:
        print(
            f"No confident vertical match "
            f"(best: {suggestion.pack_name} at "
            f"{suggestion.confidence:.0%}, "
            f"{len(suggestion.unmapped_tools)} of "
            f"{len(tools)} tools unmapped)."
        )
        print("Recommending generic pack: contracts/generic.yml")
        print(
            f"\nTo use: fewwords run {args.trace_file} --config contracts/generic.yml"
        )
        return 0

    print(
        f"Best pack: {suggestion.pack_name} (confidence: {suggestion.confidence:.0%})"
    )
    print(f"Config: {suggestion.pack_path}")

    if suggestion.tool_mapping:
        print("\nTool mapping (pack → your agent):")
        for generic, actual in sorted(suggestion.tool_mapping.items()):
            print(f"  {generic:30s} → {actual}")

    if suggestion.unmapped_tools:
        print(f"\nUnmapped tools: {suggestion.unmapped_tools}")
        print("  Add these to your .trajeval.yml manually or they'll be unchecked.")

    print(f"\nTo use: fewwords run {args.trace_file} --config {suggestion.pack_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    trace = _load_trace_auto(args.trace_file)
    if trace is None:
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(
            f"Error: config not found: {config_path}",
            file=sys.stderr,
        )
        return 1

    from trajeval.action import load_config, run_checks

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Error: invalid config: {exc}", file=sys.stderr)
        return 1

    result = run_checks(trace, config)
    if args.format == "json":
        print(result.to_json())
    else:
        print(result.summary)
    return 0 if result.all_passed else 1


def _cmd_dogfood(args: argparse.Namespace) -> int:
    trace = _load_trace_auto(args.trace_file)
    if trace is None:
        return 1

    config_path = Path(args.config)
    config = None
    if config_path.exists():
        from trajeval.action import load_config

        try:
            config = load_config(config_path)
        except Exception:
            pass

    from trajeval.action import ActionConfig, run_checks
    from trajeval.analysis.fault_injection import FaultType, inject_all
    from trajeval.analysis.intent import extract_intent
    from trajeval.analysis.self_eval import run_self_eval

    if config is None:
        config = ActionConfig(max_retries=3)

    # Step 1: Run checks
    result = run_checks(trace, config)
    print(f"Checks: {result.passed}/{result.total} passed")
    for c in result.checks:
        if not c.passed:
            print(f"  FAIL {c.name}: {c.message}")

    # Step 2: Intent
    intent = extract_intent(trace)
    print(f"Intent: {intent.label} ({intent.confidence:.0%})")

    # Step 3: Fault resilience
    import functools

    from trajeval.assertions.core import no_retry_storm

    fault_report = inject_all(
        trace,
        assertions=[
            (
                "retry",
                functools.partial(
                    no_retry_storm,
                    max_consecutive=config.max_retries,
                ),
            ),
        ],
        faults=[FaultType.ERROR, FaultType.TIMEOUT],
    )
    print(f"Fault resilience: {fault_report.resilience_rate:.0%}")

    # Step 4: Self-eval
    self_report = run_self_eval([trace], n_consistency_runs=3)
    if self_report.gaps:
        print(f"Self-eval gaps: {len(self_report.gaps)}")
        for g in self_report.gaps[:3]:
            print(f"  - {g}")
    else:
        print("Self-eval: no gaps in TrajEval's own pipeline")

    # Step 5: Meta-trace check
    if self_report.meta_traces:
        meta = run_checks(self_report.meta_traces[0], ActionConfig(max_retries=5))
        print(f"Meta-trace: {meta.passed}/{meta.total} passed")

    return 0 if result.all_passed else 1


def _cmd_preflight(args: argparse.Namespace) -> int:
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    assertions = _parse_assertions(args.assertions)
    if assertions is None:
        return 1

    from trajeval.preflight import preflight_check

    report = preflight_check(tools, assertions)
    print(report.summary())
    return 1 if report.conflicts else 0


def _cmd_drift(args: argparse.Namespace) -> int:
    from trajeval.action import load_config
    from trajeval.drift import (
        compute_drift,
        format_report,
        load_traces_from_dir,
        report_to_dict,
    )

    baseline_dir = Path(args.baseline_dir)
    recent_dir = Path(args.recent_dir)
    cfg_path = Path(args.config)
    for label, p in [
        ("baseline", baseline_dir),
        ("recent", recent_dir),
        ("config", cfg_path),
    ]:
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            return 1

    baseline_traces = load_traces_from_dir(baseline_dir)
    recent_traces = load_traces_from_dir(recent_dir)
    if not baseline_traces:
        print(
            f"Error: no parseable traces in {baseline_dir}",
            file=sys.stderr,
        )
        return 1
    if not recent_traces:
        print(
            f"Error: no parseable traces in {recent_dir}",
            file=sys.stderr,
        )
        return 1

    config = load_config(cfg_path)
    report = compute_drift(
        baseline_traces,
        recent_traces,
        config,
        ratio_threshold=args.ratio,
        min_delta=args.min_delta,
    )

    if args.emit_json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(format_report(report))

    # Exit non-zero on any regression — useful for CI.
    regressions = [e for e in report.significant if e.delta > 0]
    return 1 if regressions else 0


def _cmd_scenario(args: argparse.Namespace) -> int:
    from trajeval.scenario import (
        format_results,
        run_scenario,
        run_scenario_dir,
    )

    p = Path(args.path)
    if not p.exists():
        print(f"Error: scenario path not found: {p}", file=sys.stderr)
        return 1
    results = run_scenario_dir(p) if p.is_dir() else [run_scenario(p)]
    print(format_results(results))
    return 0 if all(r.passed for r in results) else 1


def _cmd_discover_status(args: argparse.Namespace) -> int:
    """Print a one-screen status of the discovery corpus."""
    from trajeval.discover import discovery_status

    status = discovery_status(
        base_dir=args.base_dir,
        agent_id=args.agent_id,
        threshold=args.threshold,
    )
    print(f"discovery corpus: {status.base_dir}")
    print(
        f"  trace files       : {status.total_traces}"
        f" ({status.parseable_traces} parseable)"
    )
    if args.agent_id:
        print(f"  agent_id filter   : {args.agent_id}")
    print(
        f"  distinct agent_ids: "
        f"{', '.join(status.agent_ids) if status.agent_ids else 'none'}"
    )
    if status.tool_counts:
        top_tools = sorted(status.tool_counts.items(), key=lambda kv: (-kv[1], kv[0]))[
            :8
        ]
        rendered = ", ".join(f"{k}×{v}" for k, v in top_tools)
        print(f"  top tools (top 8) : {rendered}")
    if status.ready_for_synthesis:
        print(f"  synthesis         : ready — run `trajeval discover suggest`")
    else:
        need = max(0, status.threshold - status.parseable_traces)
        print(
            f"  synthesis         : need {need} more trace(s) "
            f"(threshold {status.threshold})"
        )
    return 0


def _cmd_discover_suggest(args: argparse.Namespace) -> int:
    """Mine contract candidates from the discovery corpus."""
    from trajeval.discover import discovery_status, synthesize_discovered

    status = discovery_status(
        base_dir=args.base_dir,
        agent_id=args.agent_id,
        threshold=args.threshold,
    )
    if not status.ready_for_synthesis:
        need = max(0, status.threshold - status.parseable_traces)
        print(
            f"Not enough traces yet: "
            f"{status.parseable_traces}/{status.threshold}. "
            f"Need {need} more before synthesis unlocks.",
            file=sys.stderr,
        )
        return 1
    suggestions = synthesize_discovered(
        base_dir=args.base_dir,
        agent_id=args.agent_id,
        threshold=args.threshold,
    )
    if not suggestions:
        print(
            f"Mined 0 rules from {status.parseable_traces} trace(s). "
            "The corpus may be too heterogeneous (different workflows "
            "mixed together) or lack the patterns our heuristics look "
            "for. Try narrowing with --agent-id.",
            file=sys.stderr,
        )
        return 0
    print(
        f"Discovered {len(suggestions)} rule(s) from "
        f"{status.parseable_traces} trace(s):"
    )
    for s in suggestions:
        conf = int(100 * s.confidence)
        print(f"  [{conf}% {s.strategy}] {s.rule}")
        print(f"    └ {s.rationale}")
    print()
    print(
        "Promote any you trust into .trajeval.yml and run "
        "`fewwords run <trace>` to enforce."
    )
    return 0


_PACK_ALIASES = {
    "sales": "sales.yml",
    "sales-agent": "sales.yml",
    "gtm": "sales.yml",
    "code": "code_agents.yml",
    "code-agent": "code_agents.yml",
    "support": "support.yml",
    "support-agent": "support.yml",
    "financial": "financial.yml",
    "financial-agent": "financial.yml",
    "rag": "rag.yml",
    "rag-agent": "rag.yml",
    "data": "data_pipeline.yml",
    "data-pipeline": "data_pipeline.yml",
    "etl": "data_pipeline.yml",
    "healthcare": "healthcare.yml",
    "legal": "legal.yml",
    "generic": "generic.yml",
    # Added 2026-04-25 (R5.1):
    "browser": "browser_agent.yml",
    "browser-agent": "browser_agent.yml",
    "computer-use": "browser_agent.yml",
    "devops": "devops.yml",
    "infra": "devops.yml",
    "iac": "devops.yml",
    "customer-service": "customer_service.yml",
    "cs": "customer_service.yml",
    "support-rich": "customer_service.yml",  # heavier than minimal "support"
    "edit-safety": "edit_safety.yml",         # was on disk but unreachable
    "edits": "edit_safety.yml",
}


def _resolve_pack(name: str) -> Path | None:
    """Resolve a pack name / alias to the packaged YAML file.

    Pack files live at ``contracts/`` in the repo root (parents[3] of
    this module).
    """
    fname = _PACK_ALIASES.get(name.lower())
    if not fname:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "contracts" / fname
    return candidate if candidate.exists() else None


def _cmd_init(args: argparse.Namespace) -> int:
    """Scan the repo, synthesize a starter .trajeval.yml, print or write.

    Defaults to stdout so users can pipe / inspect before committing to
    a file. ``--write`` lands the file at ``<path>/.trajeval.yml`` and
    refuses to overwrite an existing one — no silent destruction.

    With ``--pack <name>`` the chosen vertical contract pack is used as
    the starter instead of the synthesized skeleton. Packs are listed
    with ``--list-packs``.
    """
    if getattr(args, "list_packs", False):
        repo_root = Path(__file__).resolve().parents[2]
        seen: set[str] = set()
        print("Available contract packs:\n", file=sys.stderr)
        for fname in _PACK_ALIASES.values():
            canonical = fname.replace(".yml", "")
            if canonical in seen:
                continue
            seen.add(canonical)
            path = repo_root / "contracts" / fname
            size = path.stat().st_size if path.exists() else 0
            print(
                f"  {canonical:16} {size:>5} bytes  contracts/{fname}",
                file=sys.stderr,
            )
        print(file=sys.stderr)
        print("Aliases:", file=sys.stderr)
        for alias, fname in sorted(_PACK_ALIASES.items()):
            canonical = fname.replace(".yml", "")
            if alias != canonical:
                print(f"  {alias:16} -> {canonical}", file=sys.stderr)
        return 0

    from trajeval.initializer import run_init

    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    # If a pack is chosen, skip synthesis and use the pack verbatim.
    pack_name: str | None = getattr(args, "pack", None)
    if pack_name:
        pack_path = _resolve_pack(pack_name)
        if pack_path is None:
            aliases = ", ".join(sorted(set(_PACK_ALIASES)))
            print(
                f"Error: unknown pack '{pack_name}'. Available: {aliases}. "
                "Run `fewwords init --list-packs` for details.",
                file=sys.stderr,
            )
            return 1
        pack_yaml = pack_path.read_text()
        rel = pack_path.relative_to(Path(__file__).resolve().parents[2])
        print(
            f"fewwords init --pack {pack_name}: using {rel}",
            file=sys.stderr,
        )
        if args.write:
            dest = root / ".trajeval.yml"
            if dest.exists():
                print(
                    f"Refusing to overwrite existing {dest}. "
                    "Remove it or diff manually.",
                    file=sys.stderr,
                )
                return 1
            dest.write_text(pack_yaml)
            print(f"Wrote {dest}", file=sys.stderr)
            return 0
        print(pack_yaml, end="")
        return 0

    report = run_init(root, max_files=args.max_files)

    # Summary goes to stderr so `--write` users can still pipe-inspect
    # a stdout-generated yaml without summary noise.
    print(f"fewwords init — scanned {report.root}", file=sys.stderr)
    frameworks = ", ".join(report.frameworks) if report.frameworks else "none"
    print(f"  frameworks detected : {frameworks}", file=sys.stderr)
    print(
        f"  trace files found   : {len(report.trace_paths)}"
        f" ({report.traces_parsed} parsed)",
        file=sys.stderr,
    )
    print(f"  unique tools        : {len(report.tools_seen)}", file=sys.stderr)
    if report.banned_tools_suggested:
        print(
            f"  banned (destructive): {', '.join(report.banned_tools_suggested)}",
            file=sys.stderr,
        )
    if report.suggestions:
        print(f"  mined rules         : {len(report.suggestions)}", file=sys.stderr)
    print(file=sys.stderr)

    if args.write:
        dest = root / ".trajeval.yml"
        if dest.exists():
            print(
                f"Refusing to overwrite existing {dest}. "
                "Diff manually or remove the file and re-run.",
                file=sys.stderr,
            )
            return 1
        dest.write_text(report.yaml)
        print(f"Wrote {dest}", file=sys.stderr)
        return 0

    print(report.yaml, end="")
    return 0


def _cmd_version(args: argparse.Namespace) -> int:  # noqa: ARG001
    try:
        from importlib.metadata import version

        v = version("trajeval")
    except Exception:
        v = "0.1.0 (development)"
    print(f"trajeval {v}")
    return 0


def _merge_pack(base: object, overlay: object) -> object:
    """Deep-merge two YAML-loaded structures.

    Semantics:
      - dict + dict: deep merge, recursing on shared keys.
      - list + list: concat, de-duplicate preserving order (scalars
        compared by equality, dicts by JSON-stable repr).
      - scalar + anything: overlay wins.
    """
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged: dict[str, object] = dict(base)
        for k, v in overlay.items():
            if k in merged:
                merged[k] = _merge_pack(merged[k], v)
            else:
                merged[k] = v
        return merged
    if isinstance(base, list) and isinstance(overlay, list):
        out: list[object] = []
        seen: set[str] = set()
        for item in list(base) + list(overlay):
            key = json.dumps(item, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out
    return overlay


def _cmd_compose(args: argparse.Namespace) -> int:
    import yaml

    if not args.files:
        print("Error: provide at least one pack file.", file=sys.stderr)
        return 1

    merged: object = {}
    provenance: list[str] = []
    for path_str in args.files:
        p = Path(path_str)
        if not p.exists():
            print(f"Error: pack not found: {path_str}", file=sys.stderr)
            return 1
        data = yaml.safe_load(p.read_text())
        if data is None:
            continue
        if not isinstance(data, dict):
            print(
                f"Error: {path_str} root must be a mapping, got "
                f"{type(data).__name__}.",
                file=sys.stderr,
            )
            return 1
        merged = _merge_pack(merged, data)
        provenance.append(str(p))

    header = "# Composed by `trajeval compose` from:\n" + "\n".join(
        f"#   - {p}" for p in provenance
    ) + "\n#\n# Edit the output below to override any value per customer.\n\n"
    body = yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)
    output = header + body

    if args.out == "-":
        print(output, end="")
    else:
        dest = Path(args.out)
        if dest.exists():
            print(
                f"Refusing to overwrite existing {dest}. "
                "Remove the file or pick a different --out.",
                file=sys.stderr,
            )
            return 1
        dest.write_text(output)
        print(f"Wrote {dest}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_trace(path: str) -> Trace | None:
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return None
    try:
        data = json.loads(p.read_text())
        return Trace.model_validate(data)
    except Exception as exc:
        print(
            f"Error: could not parse trace from {path}: {exc}",
            file=sys.stderr,
        )
        return None


def _load_trace_auto(path: str) -> Trace | None:
    """Load a trace with auto-format detection.

    Tries native TrajEval first. On failure, falls back to
    auto-detect (OpenAI, OTel, LangGraph).
    """
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"Error: invalid JSON in {path}: {exc}",
            file=sys.stderr,
        )
        return None

    # Try native first (fast path)
    try:
        return Trace.model_validate(data)
    except Exception:
        pass

    # Fall back to auto-detect
    try:
        from trajeval.adapters.auto import auto_detect

        result = auto_detect(data)
        return result.trace
    except Exception as exc:
        print(
            f"Error: could not parse trace from {path}: {exc}",
            file=sys.stderr,
        )
        return None


def _parse_assertions(
    specs: list[str],
) -> list[tuple[str, Any]] | None:
    """Parse assertion specs like 'never_calls:delete_user' into (name, fn) pairs."""
    import functools

    from trajeval.assertions.core import (
        cost_within,
        latency_within,
        max_depth,
        must_visit,
        never_calls,
        no_cycles,
        no_retry_storm,
        tool_must_precede,
        validate_tool_outputs,
    )

    result: list[tuple[str, Any]] = []
    for spec in specs:
        parts = spec.split(":")
        name = parts[0]
        try:
            if name == "never_calls":
                tool = parts[1]
                result.append((spec, functools.partial(never_calls, tool=tool)))
            elif name == "must_visit":
                tools = parts[1].split(",")
                result.append((spec, functools.partial(must_visit, tools=tools)))
            elif name == "tool_must_precede":
                tool = parts[1]
                before = parts[2].split("=")[1] if "=" in parts[2] else parts[2]
                result.append(
                    (
                        spec,
                        functools.partial(tool_must_precede, tool=tool, before=before),
                    )
                )
            elif name == "max_depth":
                n = int(parts[1])
                result.append((spec, functools.partial(max_depth, n=n)))
            elif name == "cost_within":
                p90 = float(parts[1])
                result.append((spec, functools.partial(cost_within, p90=p90)))
            elif name == "latency_within":
                ms = int(parts[1])
                result.append((spec, functools.partial(latency_within, p95=ms)))
            elif name == "no_cycles":
                result.append((spec, no_cycles))
            elif name == "no_retry_storm":
                n = int(parts[1]) if len(parts) > 1 else 3
                result.append(
                    (spec, functools.partial(no_retry_storm, max_consecutive=n))
                )
            elif name == "validate_schemas":
                schema_path = parts[1]
                schemas = json.loads(Path(schema_path).read_text())
                result.append(
                    (spec, functools.partial(validate_tool_outputs, schemas=schemas))
                )
            else:
                print(
                    f"Error: unknown assertion '{name}'. "
                    f"Supported: never_calls, must_visit, tool_must_precede, "
                    f"max_depth, cost_within, latency_within, no_cycles, "
                    f"no_retry_storm, validate_schemas.",
                    file=sys.stderr,
                )
                return None
        except (IndexError, ValueError) as exc:
            print(f"Error: malformed assertion spec '{spec}': {exc}", file=sys.stderr)
            return None

    return result


if __name__ == "__main__":
    sys.exit(main())
