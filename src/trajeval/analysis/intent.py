"""Intent extraction — classify what the agent was trying to do.

Without intent, metrics are context-free. ``evidence_grounding=0.87``
on a "just check prices" trace that ends in a booking is wrong, not
good. Intent conditioning makes every downstream metric interpretable.

Three extraction strategies (tried in order):

1. **Explicit metadata**: ``trace.metadata["intent"]`` — set by the
   caller who knows what the user asked for.
2. **First user message**: if the trace has an ``llm_call`` node whose
   ``tool_input`` contains a ``messages`` array, extract the first
   user message and classify via keyword matching.
3. **Tool sequence heuristic**: infer intent from the tool call pattern
   (e.g. search+book = "booking", search-only = "lookup").

Usage::

    from trajeval.analysis.intent import extract_intent

    intent = extract_intent(trace)
    print(intent.label)       # "booking", "lookup", "modification", ...
    print(intent.confidence)  # 0.0–1.0
    print(intent.source)      # "metadata", "user_message", "tool_heuristic"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from trajeval.sdk.models import Trace

IntentSource = Literal["metadata", "user_message", "tool_heuristic", "unknown"]


@dataclass(frozen=True)
class IntentResult:
    """Extracted intent for a trace."""

    label: str
    confidence: float
    source: IntentSource
    raw_signal: str


# ---------------------------------------------------------------------------
# Keyword patterns for user-message classification
# ---------------------------------------------------------------------------

_MESSAGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("booking", re.compile(
        r"\b(book|reserve|schedule|purchase|buy|order)\b", re.IGNORECASE
    )),
    ("lookup", re.compile(
        r"\b(search|find|look\s*up|check|show|list|browse)\b", re.IGNORECASE
    )),
    ("modification", re.compile(
        r"\b(change|update|modify|edit|cancel|reschedule)\b", re.IGNORECASE
    )),
    ("deletion", re.compile(
        r"\b(delete|remove|drop|destroy|purge)\b", re.IGNORECASE
    )),
    ("analysis", re.compile(
        r"\b(analy[sz]e|compare|evaluate|assess|review|audit)\b", re.IGNORECASE
    )),
    ("creation", re.compile(
        r"\b(create|make|build|generate|compose|write)\b", re.IGNORECASE
    )),
]

# ---------------------------------------------------------------------------
# Tool-sequence heuristics
# ---------------------------------------------------------------------------

_TOOL_PATTERNS: list[tuple[str, set[str]]] = [
    ("booking", {"book", "reserve", "purchase", "charge", "pay", "confirm"}),
    ("research", {
        "search", "find", "fetch", "web_search",
        "web_fetch", "scrape", "crawl",
    }),
    ("lookup", {"list", "get", "query", "read", "retrieve", "describe"}),
    ("modification", {"update", "edit", "modify", "change", "patch"}),
    ("deletion", {"delete", "remove", "drop", "destroy", "cancel"}),
    ("creation", {"create", "make", "build", "generate", "write", "post", "compose"}),
    ("code", {
        "python_repl", "execute", "run_code", "run_tests",
        "compile", "edit_file", "read_file",
    }),
    ("analysis", {"analyze", "compare", "evaluate", "summarize", "classify"}),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_intent(trace: Trace) -> IntentResult:
    """Extract the task intent from a trace.

    Tries metadata → user message → tool heuristic, returns the first
    match with confidence > 0.
    """
    # Strategy 1: Explicit metadata
    if trace.metadata:
        intent = trace.metadata.get("intent")
        if isinstance(intent, str) and intent.strip():
            return IntentResult(
                label=intent.strip(),
                confidence=1.0,
                source="metadata",
                raw_signal=f"metadata.intent={intent}",
            )

    # Strategy 2: First user message
    msg_result = _from_user_message(trace)
    if msg_result is not None:
        return msg_result

    # Strategy 3: Tool sequence heuristic
    tool_result = _from_tool_sequence(trace)
    if tool_result is not None:
        return tool_result

    return IntentResult(
        label="unknown",
        confidence=0.0,
        source="unknown",
        raw_signal="no signal found",
    )


def _from_user_message(trace: Trace) -> IntentResult | None:
    """Try to extract intent from LLM call inputs containing user messages."""
    for node in trace.nodes:
        if node.node_type != "llm_call":
            continue
        messages = node.tool_input.get("messages")
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                if not content:
                    continue
                for label, pattern in _MESSAGE_PATTERNS:
                    if pattern.search(content):
                        return IntentResult(
                            label=label,
                            confidence=0.8,
                            source="user_message",
                            raw_signal=content[:200],
                        )
    return None


def _from_tool_sequence(trace: Trace) -> IntentResult | None:
    """Infer intent from tool names called in the trace."""
    tool_names = {
        n.tool_name.lower()
        for n in trace.nodes
        if n.node_type == "tool_call" and n.tool_name
    }
    if not tool_names:
        return None

    best_label = None
    best_overlap = 0
    for label, keywords in _TOOL_PATTERNS:
        overlap = len(tool_names & keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    # Also check partial matches (tool name contains keyword)
    if best_overlap == 0:
        partial_scores: list[tuple[str, int]] = []
        for label, keywords in _TOOL_PATTERNS:
            hits = sum(
                1
                for tool in tool_names
                for kw in keywords
                if kw in tool
            )
            if hits > 0:
                partial_scores.append((label, hits))
        if partial_scores:
            best_partial = max(partial_scores, key=lambda x: x[1])
            confidence = min(0.7, 0.4 + 0.1 * best_partial[1])
            return IntentResult(
                label=best_partial[0],
                confidence=confidence,
                source="tool_heuristic",
                raw_signal=f"partial match: {sorted(tool_names)}",
            )

    if best_label is not None and best_overlap > 0:
        confidence = min(0.8, 0.4 + 0.15 * best_overlap)
        return IntentResult(
            label=best_label,
            confidence=confidence,
            source="tool_heuristic",
            raw_signal=f"tools={sorted(tool_names)}",
        )

    return None
