"""``fewwords init`` — one-command onboarding.

Scans the target directory for framework signals (langchain / langgraph /
openai / anthropic / otel / llama_index / autogen / litellm) and trace
files in conventional paths, then synthesizes a starter ``.trajeval.yml``
with commented rationale — the "1 minute from zero to first config"
path the R2 sprint depends on.

Layer rule: this module lives alongside the CLI; it imports from SDK,
adapters, and analysis only — never from backend or FastAPI.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from trajeval.adapters.auto import auto_detect
from trajeval.analysis.auto_contract import ContractSuggestion, suggest_contracts
from trajeval.sdk.models import Trace

_FRAMEWORK_SIGNALS: dict[str, tuple[str, ...]] = {
    "langchain": ("langchain", "langchain-core", "langchain_core"),
    "langgraph": ("langgraph",),
    "openai": ("openai",),
    "anthropic": ("anthropic",),
    "opentelemetry": ("opentelemetry-api", "opentelemetry-sdk", "opentelemetry_"),
    "llama_index": ("llama-index", "llama_index"),
    "autogen": ("pyautogen", "autogen-agentchat", "ag2"),
    "litellm": ("litellm",),
}

_TRACE_DIRS: tuple[str, ...] = (
    "traces",
    "logs",
    "otel-output",
    ".trajeval/traces",
    ".trajeval",
    "tmp/traces",
    "tmp/logs",
)

_DESTRUCTIVE_KEYWORDS: frozenset[str] = frozenset(
    {
        "delete",
        "drop",
        "destroy",
        "truncate",
        "purge",
        "wipe",
        "nuke",
        "unlink",
        "rm_",
        "remove_all",
    }
)


@dataclass(frozen=True)
class InitReport:
    """Summary of an init run. The YAML field is the user-facing artifact."""

    root: Path
    frameworks: list[str]
    trace_paths: list[Path]
    traces_parsed: int
    tools_seen: list[str]
    banned_tools_suggested: list[str]
    suggestions: list[ContractSuggestion]
    yaml: str


def detect_frameworks(root: Path) -> list[str]:
    """Scan common manifest files for framework signals.

    Inspects: pyproject.toml, requirements.txt, Pipfile, uv.lock,
    package.json. Case-insensitive substring match — coarse but
    reliable enough for the "tell me what you're using" read.
    """
    haystack_parts: list[str] = []
    for name in (
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "Pipfile",
        "Pipfile.lock",
        "uv.lock",
        "poetry.lock",
        "package.json",
    ):
        p = root / name
        if p.exists() and p.is_file():
            try:
                haystack_parts.append(p.read_text(errors="ignore").lower())
            except OSError:
                continue
    haystack = "\n".join(haystack_parts)
    if not haystack:
        return []
    detected: list[str] = []
    for fw, needles in _FRAMEWORK_SIGNALS.items():
        if any(n in haystack for n in needles):
            detected.append(fw)
    return sorted(detected)


def discover_traces(root: Path, *, max_files: int = 50) -> list[Path]:
    """Find trace-shaped JSON/JSONL files in conventional paths.

    Caps at ``max_files`` to bound IO on large repos. Returns paths
    without parsing — parsing is caller's job.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for rel in _TRACE_DIRS:
        base = root / rel
        if not base.exists() or not base.is_dir():
            continue
        for ext in ("json", "jsonl"):
            for p in sorted(base.rglob(f"*.{ext}")):
                if not p.is_file() or p in seen:
                    continue
                seen.add(p)
                found.append(p)
                if len(found) >= max_files:
                    return found
    return found


def _parse_one(path: Path) -> list[Trace]:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    out: list[Trace] = []
    # Heuristic: if .jsonl, split on newlines
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                out.append(auto_detect(data).trace)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"init: skip line in {path}: {exc!r}")
                continue
        return out
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    try:
        out.append(auto_detect(data).trace)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"init: skip {path}: {exc!r}")
        return []
    return out


def parse_traces(paths: list[Path]) -> list[Trace]:
    traces: list[Trace] = []
    for p in paths:
        traces.extend(_parse_one(p))
    return traces


def _tools_in(traces: list[Trace]) -> list[str]:
    counts: Counter[str] = Counter()
    for t in traces:
        for n in t.nodes:
            if n.node_type == "tool_call" and n.tool_name:
                counts[n.tool_name] += 1
    return sorted(counts.keys(), key=lambda name: (-counts[name], name))


def _destructive(tools: list[str]) -> list[str]:
    return sorted(
        t for t in tools if any(kw in t.lower() for kw in _DESTRUCTIVE_KEYWORDS)
    )


def _mine_suggestions(traces: list[Trace]) -> list[ContractSuggestion]:
    if len(traces) < 3:
        return []
    return suggest_contracts(traces, [])


def _render_yaml(
    *,
    root: Path,
    frameworks: list[str],
    trace_paths: list[Path],
    traces_parsed: int,
    tools_seen: list[str],
    banned: list[str],
    suggestions: list[ContractSuggestion],
) -> str:
    lines: list[str] = []
    lines.append("# .trajeval.yml — generated by `fewwords init`")
    lines.append(f"# scanned: {root}")
    lines.append(f"# found {len(trace_paths)} trace file(s), parsed {traces_parsed}")
    if frameworks:
        lines.append(f"# frameworks detected: {', '.join(frameworks)}")
    else:
        lines.append("# frameworks detected: none (couldn't find a manifest)")
    lines.append("#")
    lines.append("# Edit below to reflect your agent's intent. Re-run `fewwords run")
    lines.append("# <trace.json>` after edits to validate. Remove this comment block")
    lines.append("# once you're happy with the contract.")
    lines.append("")

    if banned:
        lines.append(
            "# Tools whose names match destructive keywords (delete / drop / wipe …)."
        )
        lines.append(
            "# Hard-block these unless you explicitly want the agent to run them."
        )
        lines.append("banned_tools:")
        for t in banned:
            lines.append(f"  - {t}")
    else:
        lines.append("# No destructive-keyword tools found. Add any you want")
        lines.append("# to block here (e.g. delete_user, drop_table).")
        lines.append("banned_tools: []")
    lines.append("")

    if tools_seen:
        lines.append(
            "# Tools observed in your traces. Narrowing allowed_tools shrinks the"
        )
        lines.append(
            "# attack surface — anything off-list gets blocked at the dispatcher."
        )
        lines.append("allowed_tools:")
        for t in tools_seen[:30]:
            lines.append(f"  - {t}")
        if len(tools_seen) > 30:
            lines.append(f"  # … + {len(tools_seen) - 30} more seen")
    else:
        lines.append("# No traces parsed — populate allowed_tools once your")
        lines.append("# agent starts producing logs we can read.")
        lines.append("allowed_tools: []")
    lines.append("")

    lines.append("# Cost + retry guardrails. Conservative defaults.")
    lines.append("max_retries: 3")
    lines.append("max_tool_repeat: 10")
    lines.append("stop_on_error: true")
    lines.append("")

    if suggestions:
        lines.append("# Rules mined from your traces by `suggest_contracts`.")
        lines.append("# Each line shows the rule + why we suggested it.")
        lines.append("# Promote any you trust into a proper assertion.")
        lines.append("contracts:")
        for s in suggestions[:10]:
            conf_pct = int(100 * s.confidence)
            lines.append(f"  # [{conf_pct}% {s.strategy}] {s.rationale}")
            lines.append(f"  - {s.rule}")
    elif traces_parsed >= 3:
        lines.append(
            "# Mined no cross-run patterns from "
            f"{traces_parsed} traces. Either the runs"
        )
        lines.append(
            "# vary too much or the heuristics need more data. Add rules manually:"
        )
        lines.append("contracts: []")
    else:
        lines.append(
            f"# Need ≥3 parseable traces to mine rules (have {traces_parsed})."
        )
        lines.append(
            "# Drop more trace JSONs into ./traces/ or ./logs/ "
            "and re-run `fewwords init`."
        )
        lines.append("contracts: []")
    lines.append("")

    return "\n".join(lines) + "\n"


def run_init(root: Path, *, max_files: int = 50) -> InitReport:
    """Scan the root, synthesize a starter config, return the report.

    Pure function: no IO beyond reading manifest / trace files. Caller
    decides whether to print or write.
    """
    root = root.resolve()
    frameworks = detect_frameworks(root)
    trace_paths = discover_traces(root, max_files=max_files)
    traces = parse_traces(trace_paths)
    tools_seen = _tools_in(traces)
    banned = _destructive(tools_seen)
    suggestions = _mine_suggestions(traces)
    yaml_text = _render_yaml(
        root=root,
        frameworks=frameworks,
        trace_paths=trace_paths,
        traces_parsed=len(traces),
        tools_seen=tools_seen,
        banned=banned,
        suggestions=suggestions,
    )
    return InitReport(
        root=root,
        frameworks=frameworks,
        trace_paths=trace_paths,
        traces_parsed=len(traces),
        tools_seen=tools_seen,
        banned_tools_suggested=banned,
        suggestions=suggestions,
        yaml=yaml_text,
    )
