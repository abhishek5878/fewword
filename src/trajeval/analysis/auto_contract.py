"""Automated contract synthesis from production traces — Gap 12.

Assertions require developer foresight. This module mines production
failure traces for recurring patterns and proposes assertion candidates.

Three synthesis strategies:

1. **Banned tool discovery**: tools that appear only in failing traces
   but never in passing ones → candidate ``never_calls`` rule.
2. **Ordering patterns**: tool sequences that always appear in passing
   traces → candidate ``precedes`` rules.
3. **Retry storm thresholds**: observe the max consecutive identical
   calls across passing traces → suggest ``no_retry_storm`` threshold.

Usage::

    from trajeval.analysis.auto_contract import suggest_contracts

    suggestions = suggest_contracts(
        passing_traces=good_runs,
        failing_traces=bad_runs,
    )
    for s in suggestions:
        print(f"[{s.confidence:.0%}] {s.rule}  ({s.rationale})")
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class ContractSuggestion:
    """A proposed contract rule mined from trace data."""

    rule: str
    confidence: float
    rationale: str
    strategy: str


def suggest_contracts(
    passing_traces: list[Trace],
    failing_traces: list[Trace],
    *,
    min_confidence: float = 0.5,
) -> list[ContractSuggestion]:
    """Mine passing + failing traces for contract candidates."""
    suggestions: list[ContractSuggestion] = []

    suggestions.extend(
        _banned_tool_discovery(passing_traces, failing_traces)
    )
    suggestions.extend(
        _ordering_patterns(passing_traces)
    )
    suggestions.extend(
        _retry_threshold(passing_traces, failing_traces)
    )

    return [s for s in suggestions if s.confidence >= min_confidence]


def _tool_set(traces: list[Trace]) -> set[str]:
    return {
        n.tool_name
        for t in traces
        for n in t.nodes
        if n.node_type == "tool_call" and n.tool_name
    }


_DESTRUCTIVE_KEYWORDS = {
    "delete", "drop", "destroy", "remove", "truncate",
    "purge", "wipe", "override", "bypass", "disable",
    "margin", "prescribe", "file_court", "waive",
    "rm_rf", "format",
}

_SAFE_KEYWORDS = {
    "search", "read", "get", "list", "fetch", "query",
    "check", "verify", "validate", "lint", "test", "log",
    "send", "notify", "schedule", "lookup", "find",
    "analyze", "summarize", "report", "export", "import",
    "run_linter", "run_tests", "grep", "git_log",
}


def _looks_destructive(tool_name: str) -> bool:
    """Heuristic: does the tool name suggest a destructive action?"""
    lower = tool_name.lower()
    if any(kw in lower for kw in _SAFE_KEYWORDS):
        return False
    return any(kw in lower for kw in _DESTRUCTIVE_KEYWORDS)


def _banned_tool_discovery(
    passing: list[Trace], failing: list[Trace]
) -> list[ContractSuggestion]:
    """Tools in failing traces but never in passing → never_calls candidates.

    Reduces false positives by:
    1. Requiring the tool appears in 2+ failing TRACES (not just nodes)
    2. Boosting confidence for tools with destructive-sounding names
    3. Penalizing tools that look like normal operations
    """
    pass_tools = _tool_set(passing)
    fail_tools = _tool_set(failing)
    suspicious = fail_tools - pass_tools

    results: list[ContractSuggestion] = []
    for tool in sorted(suspicious):
        # Count unique failing traces containing this tool
        fail_trace_count = sum(
            1
            for t in failing
            if any(n.tool_name == tool for n in t.nodes)
        )
        fail_node_count = sum(
            1
            for t in failing
            for n in t.nodes
            if n.tool_name == tool
        )

        # Heuristic scoring
        is_destructive = _looks_destructive(tool)

        # Hard skip: tools with safe-sounding names
        lower = tool.lower()
        if any(kw in lower for kw in _SAFE_KEYWORDS):
            continue

        if fail_trace_count < 2 and not is_destructive:
            # Low support + non-destructive name → likely false positive
            continue

        base = 0.4 if not is_destructive else 0.6
        confidence = min(
            0.95,
            base + 0.1 * fail_trace_count + (0.1 if is_destructive else 0),
        )

        results.append(ContractSuggestion(
            rule=f"never call {tool}",
            confidence=confidence,
            rationale=(
                f"'{tool}' appeared in {fail_trace_count} failing trace(s) "
                f"({fail_node_count} nodes) but never in passing. "
                f"{'Destructive name.' if is_destructive else 'Non-destructive name.'}"
            ),
            strategy="banned_tool",
        ))
    return results


def _ordering_patterns(
    passing: list[Trace],
) -> list[ContractSuggestion]:
    """Tool pairs that always appear in order in passing traces → precedes."""
    if len(passing) < 3:
        return []

    pair_counts: Counter[tuple[str, str]] = Counter()
    trace_count = 0

    for trace in passing:
        tools = [
            n.tool_name
            for n in trace.nodes
            if n.node_type == "tool_call" and n.tool_name
        ]
        if len(tools) < 2:
            continue
        trace_count += 1
        seen_pairs: set[tuple[str, str]] = set()
        for i, a in enumerate(tools):
            for b in tools[i + 1 :]:
                if a != b and (a, b) not in seen_pairs:
                    seen_pairs.add((a, b))
                    pair_counts[(a, b)] += 1

    if trace_count == 0:
        return []

    results: list[ContractSuggestion] = []
    for (a, b), count in pair_counts.most_common():
        if count < trace_count * 0.8:
            continue
        confidence = count / trace_count
        results.append(ContractSuggestion(
            rule=f"{a} before {b}",
            confidence=min(0.9, confidence),
            rationale=(
                f"'{a}' preceded '{b}' in "
                f"{count}/{trace_count} passing traces"
            ),
            strategy="ordering",
        ))

    return results[:10]


def _retry_threshold(
    passing: list[Trace], failing: list[Trace]
) -> list[ContractSuggestion]:
    """Compare max consecutive identical calls between pass/fail."""
    pass_max = _max_consecutive(passing)
    fail_max = _max_consecutive(failing)

    if fail_max > pass_max and fail_max >= 3:
        threshold = max(pass_max, 2)
        return [ContractSuggestion(
            rule=f"max {threshold} consecutive retries",
            confidence=0.7,
            rationale=(
                f"Passing traces max {pass_max} consecutive identical "
                f"calls; failing traces max {fail_max}"
            ),
            strategy="retry_threshold",
        )]
    return []


def _max_consecutive(traces: list[Trace]) -> int:
    max_run = 0
    for trace in traces:
        tool_nodes = [n for n in trace.nodes if n.node_type == "tool_call"]
        if len(tool_nodes) < 2:
            continue
        run = 1
        for i in range(1, len(tool_nodes)):
            if _sig(tool_nodes[i]) == _sig(tool_nodes[i - 1]):
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
    return max_run


def _sig(node: object) -> str:
    n = node  # type: ignore[assignment]
    try:
        inp = json.dumps(n.tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        inp = repr(n.tool_input)
    return f"{n.tool_name}::{inp}"
