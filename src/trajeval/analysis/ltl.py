"""Online incremental LTL checking via Büchi automata.

Instead of re-running a full LTL checker after every tool call — O(n) per step,
O(n²) total — this module compiles each formula *once* to a Büchi automaton at
contract-load time, then advances the automaton state O(1) per tool call.

Supported formula types
-----------------------
``GloballyNever(tool)``       ``G(¬calls(tool))``                   safety — never call this tool
``Eventually(tool)``          ``F(calls(tool))``                    liveness — must call this tool
``Precedes(before, after)``   ``(¬after) U before``                 ordering — before must precede after
``Response(trigger, goal)``   ``G(calls(trigger)→F(calls(goal)))``  response — whenever trigger, goal must follow

Safety properties (``GloballyNever``, ``Precedes``) fire the moment the automaton
enters a *reject sink* — before the end of the trace.

Liveness properties (``Eventually``, ``Response``) are checked by
:meth:`LTLRuntime.check_liveness` when the trace completes.  In guard mode,
:meth:`LTLRuntime.would_enter_reject` allows pre-execution checks without
mutating automaton state.

Usage::

    from trajeval.analysis.ltl import (
        Eventually, GloballyNever, LTLRuntime, Precedes, Response,
    )

    runtime = LTLRuntime([
        GloballyNever("delete_user"),
        Precedes("search_flights", "book_flight"),
        Response("charge_card", "send_confirmation"),
        Eventually("validate"),
    ])

    for node in trace.nodes:
        violations = runtime.advance(node)   # O(k) per call
        if violations:
            print("Safety violation:", violations)

    liveness_violations = runtime.check_liveness()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, assert_never

from trajeval.sdk.models import TraceNode

# ---------------------------------------------------------------------------
# Formula types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GloballyNever:
    """``G(¬calls(tool))`` — the named tool must never be called."""

    tool: str


@dataclass(frozen=True)
class Eventually:
    """``F(calls(tool))`` — the named tool must be called at some point (liveness)."""

    tool: str


@dataclass(frozen=True)
class Precedes:
    """``(¬after_tool) U before_tool`` — *before_tool* must be called before any *after_tool*."""

    before_tool: str
    after_tool: str


@dataclass(frozen=True)
class Response:
    """``G(calls(trigger) → F(calls(goal)))`` — whenever *trigger* is called, *goal* must follow."""

    trigger: str
    goal: str


#: Union of all supported LTL formula types.
LTLFormula = Union[GloballyNever, Eventually, Precedes, Response]


# ---------------------------------------------------------------------------
# Internal Büchi automaton
# ---------------------------------------------------------------------------


@dataclass
class _BuchiAutomaton:
    """Explicit-state Büchi automaton compiled from a single LTL formula.

    States are small integers.  Transitions are resolved by a dict lookup:
    first check ``tool_transitions[(state, tool_name)]``; if missing, fall back
    to ``default_transitions[state]``; if still missing, stay in current state.
    """

    formula_name: str
    initial_state: int
    accepting_states: frozenset[int]
    #: Entering any state in this set is an immediate (safety) violation.
    reject_sinks: frozenset[int]
    #: ``(current_state, tool_name) -> next_state``
    tool_transitions: dict[tuple[int, str], int]
    #: ``current_state -> next_state`` for events that match no tool transition.
    default_transitions: dict[int, int]

    def step(self, state: int, tool_name: str | None) -> int:
        """Advance one automaton step.  O(1) per call (single dict lookup)."""
        if tool_name is not None:
            key = (state, tool_name)
            if key in self.tool_transitions:
                return self.tool_transitions[key]
        return self.default_transitions.get(state, state)


# ---------------------------------------------------------------------------
# Formula → automaton compiler
# ---------------------------------------------------------------------------


def _compile(formula: LTLFormula) -> _BuchiAutomaton:
    """Compile *formula* into a :class:`_BuchiAutomaton`.

    Each formula type produces a small explicit automaton:

    ``GloballyNever(t)``
        States 0=ok, 1=violated.
        0 --[t]-→ 1 (reject sink).  Once in 1 the trace can never recover.

    ``Eventually(t)``
        States 0=waiting, 1=satisfied.
        0 --[t]-→ 1 (accepting).  Liveness: violation if trace ends in 0.

    ``Precedes(b, a)``  — ``(¬a) U b``
        States 0=before_b, 1=b_seen, 2=violated (reject sink).
        0 --[b]-→ 1; 0 --[a]-→ 2.  Once b is seen (state 1) all is safe.

    ``Response(t, g)``  — ``G(t → F(g))``
        States 0=free, 1=waiting_for_goal.
        0 --[t]-→ 1; 1 --[g]-→ 0.  Liveness: violation if trace ends in 1.
    """
    if isinstance(formula, GloballyNever):
        t = formula.tool
        return _BuchiAutomaton(
            formula_name=f"G(¬{t})",
            initial_state=0,
            accepting_states=frozenset({0}),
            reject_sinks=frozenset({1}),
            tool_transitions={(0, t): 1, (1, t): 1},
            default_transitions={0: 0, 1: 1},
        )

    elif isinstance(formula, Eventually):
        t = formula.tool
        return _BuchiAutomaton(
            formula_name=f"F({t})",
            initial_state=0,
            accepting_states=frozenset({1}),
            reject_sinks=frozenset(),
            tool_transitions={(0, t): 1, (1, t): 1},
            default_transitions={0: 0, 1: 1},
        )

    elif isinstance(formula, Precedes):
        b, a = formula.before_tool, formula.after_tool
        # If before == after, seeing the tool transitions to state 1 (satisfied)
        # because the "before" occurrence is encountered first.
        if b == a:
            after_to = 1
        else:
            after_to = 2
        tool_trans: dict[tuple[int, str], int] = {
            (0, b): 1,
            (0, a): after_to,
            (1, b): 1,
            (1, a): 1,
            (2, b): 2,
            (2, a): 2,
        }
        return _BuchiAutomaton(
            formula_name=f"(¬{a}) U {b}",
            initial_state=0,
            accepting_states=frozenset({0, 1}),
            reject_sinks=frozenset({2}),
            tool_transitions=tool_trans,
            default_transitions={0: 0, 1: 1, 2: 2},
        )

    elif isinstance(formula, Response):
        t, g = formula.trigger, formula.goal
        # If trigger == goal, calling the tool both triggers and satisfies in one step.
        if t == g:
            tool_trans_r: dict[tuple[int, str], int] = {
                (0, t): 0,  # trigger+goal at once → remains free
                (1, g): 0,
            }
        else:
            tool_trans_r = {
                (0, t): 1,  # trigger seen → waiting
                (0, g): 0,  # goal seen while free → still free
                (1, g): 0,  # goal satisfies the pending response
                (1, t): 1,  # new trigger while still waiting → still waiting
            }
        return _BuchiAutomaton(
            formula_name=f"G({t}→F({g}))",
            initial_state=0,
            accepting_states=frozenset({0}),
            reject_sinks=frozenset(),
            tool_transitions=tool_trans_r,
            default_transitions={0: 0, 1: 1},
        )

    else:
        assert_never(formula)


# ---------------------------------------------------------------------------
# LTL Runtime
# ---------------------------------------------------------------------------


class LTLRuntime:
    """Incremental LTL monitor — advance O(k) per tool call, not O(n).

    Each formula is compiled once to a :class:`_BuchiAutomaton` at construction.
    Subsequent calls to :meth:`advance` tick every automaton forward by one step
    in O(k) total (k = number of formulas), regardless of trace depth.

    Parameters
    ----------
    formulas:
        LTL formulas to monitor.  May be heterogeneous (safety + liveness).

    Example
    -------
    ::

        runtime = LTLRuntime([
            GloballyNever("delete_user"),
            Eventually("validate"),
        ])
        for node in trace.nodes:
            violations = runtime.advance(node)
        liveness = runtime.check_liveness()
    """

    def __init__(self, formulas: list[LTLFormula]) -> None:
        self._formulas: list[LTLFormula] = list(formulas)
        self._automata: list[_BuchiAutomaton] = [_compile(f) for f in self._formulas]
        self._states: list[int] = [a.initial_state for a in self._automata]
        self._violations: list[str] = []
        self._liveness_checked: bool = False

    # ------------------------------------------------------------------
    # Core advance — O(k) per call
    # ------------------------------------------------------------------

    def advance(self, node: TraceNode) -> list[str]:
        """Advance all automata by one step and return new safety violations.

        Non-tool-call nodes (``llm_call``, ``state_transition``) are passed with
        ``tool_name=None`` and only trigger *default* transitions — they never
        cause safety violations on their own.

        Parameters
        ----------
        node:
            The committed :class:`~trajeval.sdk.models.TraceNode`.

        Returns
        -------
        list[str]
            Violation messages for automata that just entered a reject sink for
            the first time.  Empty when no new violations occurred.
        """
        tool = node.tool_name if node.node_type == "tool_call" else None
        new_violations: list[str] = []

        for i, automaton in enumerate(self._automata):
            prev = self._states[i]
            nxt = automaton.step(prev, tool)
            self._states[i] = nxt
            if nxt in automaton.reject_sinks and prev not in automaton.reject_sinks:
                msg = (
                    f"LTL safety violation [{automaton.formula_name}]: "
                    f"node {node.node_id!r} (tool={node.tool_name!r})"
                )
                new_violations.append(msg)
                self._violations.append(msg)

        return new_violations

    # ------------------------------------------------------------------
    # Pre-execution guard check — does NOT mutate state
    # ------------------------------------------------------------------

    def would_enter_reject(self, tool_name: str | None) -> list[str]:
        """Return violation messages for automata that *would* enter a reject sink.

        Simulates one step without committing the state change.  Used by
        :class:`~trajeval.sdk.callback.TrajEvalCallback` in guard mode to
        intercept a tool call *before* it executes.

        Parameters
        ----------
        tool_name:
            The tool name being proposed; ``None`` for non-tool events.

        Returns
        -------
        list[str]
            Violation messages.  Empty when the proposed step is safe.
        """
        violations: list[str] = []
        for i, automaton in enumerate(self._automata):
            prev = self._states[i]
            nxt = automaton.step(prev, tool_name)
            if nxt in automaton.reject_sinks and prev not in automaton.reject_sinks:
                violations.append(
                    f"LTL safety violation [{automaton.formula_name}]: "
                    f"proposed tool {tool_name!r} would enter rejecting state"
                )
        return violations

    # ------------------------------------------------------------------
    # Liveness check — call once at trace end
    # ------------------------------------------------------------------

    def check_liveness(self) -> list[str]:
        """Check liveness properties after the trace completes.

        Liveness automata (``Eventually``, ``Response``) have no reject sinks —
        they can only be violated when the trace ends in a non-accepting state.
        This method fires those violations and appends them to
        :attr:`violations`.

        Idempotent: subsequent calls return an empty list.

        Returns
        -------
        list[str]
            Liveness violation messages.  Empty when all liveness properties
            were satisfied or if already called.
        """
        if self._liveness_checked:
            return []
        self._liveness_checked = True

        new_violations: list[str] = []
        for automaton, state in zip(self._automata, self._states):
            # Only check pure liveness automata (no reject sinks).
            # Safety automata have already fired their violations via advance().
            if automaton.reject_sinks:
                continue
            if state not in automaton.accepting_states:
                msg = (
                    f"LTL liveness violation [{automaton.formula_name}]: "
                    "trace ended without satisfying this property"
                )
                new_violations.append(msg)
                self._violations.append(msg)

        return new_violations

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def violations(self) -> list[str]:
        """All violations collected so far (safety + any liveness from :meth:`check_liveness`)."""
        return list(self._violations)

    @property
    def current_states(self) -> list[int]:
        """Snapshot of current automaton state indices (one per formula)."""
        return list(self._states)

    @property
    def formulas(self) -> list[LTLFormula]:
        """The formulas this runtime was constructed with."""
        return list(self._formulas)
