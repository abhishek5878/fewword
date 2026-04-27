"""Natural-language contract compiler — NL rules → LTL formulas.

Translates plain-English contract rules into the LTL formula types
that the Büchi automata runtime (:mod:`trajeval.analysis.ltl`) executes.

The compiler uses pattern matching, not an LLM. This keeps it
deterministic, <1ms, and zero-dependency. Three pattern families
cover ~80% of real-world contracts:

1. **Never**: ``"never call delete_user"`` → ``GloballyNever("delete_user")``
2. **Eventually / must**: ``"must call validate"`` → ``Eventually("validate")``
3. **Before / precedes**: ``"search before book"`` → ``Precedes("search", "book")``
4. **Whenever / response**: ``"whenever charge then confirm"``
   → ``Response("charge", "confirm")``

Unrecognized rules raise ``CompileError`` with a hint showing the
supported patterns.

Usage::

    from trajeval.analysis.ltl_compiler import compile_contract

    formulas = compile_contract([
        "never call delete_user",
        "search before book",
        "must call validate",
        "whenever charge_card then send_confirmation",
    ])
    # [GloballyNever("delete_user"), Precedes("search", "book"),
    #  Eventually("validate"), Response("charge_card", "send_confirmation")]

YAML integration::

    # .trajeval.yml
    contracts:
      - never call delete_user
      - search before book
      - must call validate
      - whenever charge_card then send_confirmation
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from trajeval.analysis.ltl import (
    Eventually,
    GloballyNever,
    LTLFormula,
    Precedes,
    Response,
)


class CompileError(ValueError):
    """Raised when a natural-language rule cannot be compiled."""


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each pattern: (regex, factory function)
# Regexes are case-insensitive and match the full rule string.
# Named groups capture tool names.

_PATTERNS: list[tuple[re.Pattern[str], str]] = []


def _register(
    pattern: str, description: str
) -> re.Pattern[str]:
    compiled = re.compile(pattern, re.IGNORECASE)
    _PATTERNS.append((compiled, description))
    return compiled


# 1. Never
_RE_NEVER = _register(
    r"^never\s+(?:call\s+)?(?P<tool>\w+)$",
    'never [call] <tool>  →  "never call delete_user"',
)

# 2. Eventually / must
_RE_MUST = _register(
    r"^(?:must|eventually)\s+(?:call\s+)?(?P<tool>\w+)$",
    'must [call] <tool>  →  "must call validate"',
)

# 3. Before / precedes
_RE_BEFORE = _register(
    r"^(?P<before>\w+)\s+(?:before|precedes|must\s+precede)\s+(?P<after>\w+)$",
    '<tool> before <tool>  →  "search before book"',
)

# 4. Whenever / response
_RE_RESPONSE = _register(
    r"^(?:whenever|if|after)\s+(?P<trigger>\w+)\s+(?:then|must\s+follow\s+with)\s+(?P<goal>\w+)$",
    'whenever <tool> then <tool>  →  "whenever charge then confirm"',
)

# 5. Max N tool calls (bonus — common contract)
_RE_MAX_CALLS = _register(
    r"^max\s+(?P<n>\d+)\s+(?:tool\s+)?calls?(?:\s+per\s+run)?$",
    'max <N> [tool] calls  →  "max 15 tool calls"',
)

# 6. Cost under (bonus)
_RE_COST = _register(
    r"^cost\s+(?:under|within|below)\s+\$?(?P<amount>[\d.]+)(?:\s+at\s+p\d+)?$",
    'cost under $<amount>  →  "cost under $2.00"',
)


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledRule:
    """A compiled rule: the original text + the resulting formula or check."""

    text: str
    formula: LTLFormula | None  # None for non-LTL rules (max_calls, cost)
    check_type: str  # "ltl", "max_calls", "cost"
    params: dict[str, object]


def compile_rule(text: str) -> CompiledRule:
    """Compile a single NL rule into a :class:`CompiledRule`.

    Raises :class:`CompileError` if no pattern matches.
    """
    text = text.strip()

    # Never
    m = _RE_NEVER.match(text)
    if m:
        tool = m.group("tool")
        return CompiledRule(
            text=text,
            formula=GloballyNever(tool),
            check_type="ltl",
            params={"tool": tool},
        )

    # Must / eventually
    m = _RE_MUST.match(text)
    if m:
        tool = m.group("tool")
        return CompiledRule(
            text=text,
            formula=Eventually(tool),
            check_type="ltl",
            params={"tool": tool},
        )

    # Before / precedes
    m = _RE_BEFORE.match(text)
    if m:
        before = m.group("before")
        after = m.group("after")
        return CompiledRule(
            text=text,
            formula=Precedes(before_tool=before, after_tool=after),
            check_type="ltl",
            params={"before": before, "after": after},
        )

    # Whenever / response
    m = _RE_RESPONSE.match(text)
    if m:
        trigger = m.group("trigger")
        goal = m.group("goal")
        return CompiledRule(
            text=text,
            formula=Response(trigger=trigger, goal=goal),
            check_type="ltl",
            params={"trigger": trigger, "goal": goal},
        )

    # Max calls
    m = _RE_MAX_CALLS.match(text)
    if m:
        n = int(m.group("n"))
        return CompiledRule(
            text=text, formula=None, check_type="max_calls", params={"n": n}
        )

    # Cost
    m = _RE_COST.match(text)
    if m:
        amount = float(m.group("amount"))
        return CompiledRule(
            text=text,
            formula=None,
            check_type="cost",
            params={"amount": amount},
        )

    # No match
    hints = "\n".join(f"  - {desc}" for _, desc in _PATTERNS)
    raise CompileError(
        f"Cannot compile rule: {text!r}\n"
        f"Supported patterns:\n{hints}"
    )


def compile_contract(rules: list[str]) -> list[CompiledRule]:
    """Compile a list of NL rules into :class:`CompiledRule` objects.

    Raises :class:`CompileError` on the first unrecognized rule.
    """
    return [compile_rule(r) for r in rules]


def extract_ltl_formulas(compiled: list[CompiledRule]) -> list[LTLFormula]:
    """Extract just the LTL formulas from compiled rules (for the runtime)."""
    return [r.formula for r in compiled if r.formula is not None]
