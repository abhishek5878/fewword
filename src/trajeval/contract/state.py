"""Typed symbolic state for trajectory contracts (postconditions primitive).

A :class:`SymbolicState` is the runtime object threaded through
``run_checks`` / ``guard.check`` to express contracts of the form

    "tool X may only fire when state.Y is true"
    "tool X's output must match this schema; on success, set state.Z"

State is **runtime**, not part of the frozen :class:`~trajeval.sdk.models.Trace`.
Mutations go through :meth:`SymbolicState.apply`, which returns a new
instance — the original is never modified in place.

Predicate language (v1, intentionally narrow)
---------------------------------------------
``state.<var> <op> <literal>`` or ``state.<var> exists`` where
``op`` ∈ ``{==, !=, <, <=, >, >=, in, exists}``. Literal values are
flat (str | int | float | bool | None). Lists are only valid as the
right-hand side of ``in``. No OR, no arithmetic, no nested expressions.
Compose multiple predicates as a list (treated as AND).

Missing-key semantics: a predicate over an unset ``state.X`` evaluates
to ``False`` for every op except ``exists``. This is deliberate — it
prevents rules from firing before the relevant state has been set by
an upstream postcondition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

StateValue = str | int | float | bool | None
StateOp = Literal["==", "!=", "<", "<=", ">", ">=", "in", "exists"]

_VAR_RE = re.compile(r"^state\.([A-Za-z_][A-Za-z0-9_]*)\s+(.*)$")
_OP_RE = re.compile(r"^(==|!=|<=|>=|<|>|in|exists)(?=\s|$)\s*(.*)$")
_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass(frozen=True)
class ParsedPredicate:
    """A single ``state.X op literal`` predicate, parsed once at config-load.

    Runtime evaluation is a dict lookup + comparison — no per-call parsing.
    """

    var: str
    op: StateOp
    literal: StateValue | list[StateValue] = None
    raw: str = ""

    def __str__(self) -> str:
        return self.raw or f"state.{self.var} {self.op} {self.literal!r}"


@dataclass
class SymbolicState:
    """Mutable-by-replacement runtime state.

    Internally a flat dict of ``str -> StateValue``. All mutation goes
    through :meth:`apply`, which returns a new instance. Treat instances
    as immutable in user code.
    """

    _store: dict[str, StateValue] = field(default_factory=dict)

    def get(self, key: str) -> StateValue:
        return self._store.get(key)

    def exists(self, key: str) -> bool:
        return key in self._store

    def as_dict(self) -> dict[str, StateValue]:
        """Return a shallow copy of the underlying store (for debug / dumps)."""
        return dict(self._store)

    def apply(self, updates: dict[str, StateValue]) -> SymbolicState:
        """Return a new :class:`SymbolicState` with *updates* merged on top."""
        return SymbolicState({**self._store, **updates})

    def evaluate(self, predicate: ParsedPredicate) -> bool:
        """Evaluate *predicate* against the current store.

        Missing keys evaluate to ``False`` for every op except ``exists``.
        """
        if predicate.op == "exists":
            return predicate.var in self._store

        if predicate.var not in self._store:
            return False

        value = self._store[predicate.var]
        op = predicate.op
        literal = predicate.literal

        if op == "==":
            return value == literal
        if op == "!=":
            return value != literal
        if op == "in":
            if not isinstance(literal, list):
                return False
            return value in literal
        # ordered comparisons require both sides to be numeric or both str
        if not _comparable(value, literal):
            return False
        if op == "<":
            return value < literal  # type: ignore[operator]  # reason: _comparable narrows to comparable types
        if op == "<=":
            return value <= literal  # type: ignore[operator]  # reason: _comparable narrows to comparable types
        if op == ">":
            return value > literal  # type: ignore[operator]  # reason: _comparable narrows to comparable types
        if op == ">=":
            return value >= literal  # type: ignore[operator]  # reason: _comparable narrows to comparable types
        return False


def _comparable(a: StateValue, b: StateValue | list[StateValue]) -> bool:
    if isinstance(b, list):
        return False
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        # bool comparisons are confusing in mixed types; require both bool
        return isinstance(a, bool) and isinstance(b, bool)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return True
    return isinstance(a, str) and isinstance(b, str)


def parse_predicate(raw: str) -> ParsedPredicate:
    """Parse a predicate string into a :class:`ParsedPredicate`.

    Format::

        state.<var> <op> <literal>
        state.<var> exists

    Raises :class:`ValueError` on any malformed input.
    """
    text = raw.strip()
    head = _VAR_RE.match(text)
    if head is None:
        raise ValueError(f"predicate must start with 'state.<var>', got {raw!r}")
    var = head.group(1)
    tail = head.group(2).strip()

    op_match = _OP_RE.match(tail)
    if op_match is None:
        raise ValueError(
            f"predicate {raw!r} has no recognized operator "
            f"(==, !=, <, <=, >, >=, in, exists)"
        )
    op_str = op_match.group(1)
    rest = op_match.group(2).strip()

    if op_str == "exists":
        if rest:
            raise ValueError(
                f"'exists' takes no operand, got trailing {rest!r} in {raw!r}"
            )
        return ParsedPredicate(var=var, op="exists", literal=None, raw=raw)

    if not rest:
        raise ValueError(f"predicate {raw!r} is missing a right-hand value")

    op: StateOp = op_str  # type: ignore[assignment]

    if op == "in":
        return ParsedPredicate(
            var=var, op=op, literal=_parse_list(rest, raw=raw), raw=raw
        )
    return ParsedPredicate(var=var, op=op, literal=_parse_literal(rest), raw=raw)


def _parse_literal(text: str) -> StateValue:
    s = text.strip()
    if not s:
        raise ValueError("empty literal")
    lower = s.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    if _NUMERIC_RE.match(s):
        return float(s) if "." in s else int(s)
    return s


def _parse_list(text: str, *, raw: str) -> list[StateValue]:
    s = text.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"'in' requires a [a, b, c] literal, got {raw!r}")
    inner = s[1:-1].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [_parse_literal(p) for p in parts]


# ---------------------------------------------------------------------------
# Tool config (typed-state YAML extension)
# ---------------------------------------------------------------------------


_TOOL_KEYS = frozenset({"requires", "postcondition"})
_POSTCOND_KEYS = frozenset({"returns", "state_updates"})


@dataclass(frozen=True)
class ToolPostcondition:
    """Schema + state-mutation contract for a tool's observed output.

    ``returns`` mirrors the existing top-level ``schemas:`` shape — a flat
    mapping of ``field_name -> type_string`` consumed by the validator in
    ``trajeval.assertions.core.validate_tool_outputs``.

    ``state_updates`` values may be literals (bool/int/float/None) or string
    template tokens of the form ``"{key}"`` — single-token only in v1, no
    nested paths. Templates resolve against ``tool_output`` at run time;
    Phase 2 stores them verbatim.
    """

    returns: dict[str, str] = field(default_factory=dict)
    state_updates: dict[str, StateValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContract:
    """Parsed ``tools.<name>`` block from a ``.trajeval.yml`` config."""

    name: str
    requires: tuple[ParsedPredicate, ...] = ()
    postcondition: ToolPostcondition | None = None


def parse_tools_section(raw: object) -> dict[str, ToolContract]:
    """Parse the top-level ``tools:`` mapping from a YAML config.

    Returns ``{tool_name: ToolContract}``. An absent / ``None`` value yields
    an empty dict. Raises :class:`ValueError` on:

    * non-mapping at any level
    * unknown keys under a tool or under ``postcondition``
    * malformed predicate strings in ``requires``
    * malformed ``returns`` (must be ``dict[str, str]``)
    * malformed ``state_updates`` (values must be flat StateValue)
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"`tools:` must be a mapping, got {type(raw).__name__}"
        )

    parsed: dict[str, ToolContract] = {}
    for tool_name, tool_cfg in raw.items():
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(
                f"tool name must be a non-empty string, got {tool_name!r}"
            )
        if not isinstance(tool_cfg, dict):
            raise ValueError(
                f"tools.{tool_name} must be a mapping, "
                f"got {type(tool_cfg).__name__}"
            )
        unknown = set(tool_cfg) - _TOOL_KEYS
        if unknown:
            raise ValueError(
                f"tools.{tool_name} has unknown key(s): {sorted(unknown)!r}; "
                f"allowed: {sorted(_TOOL_KEYS)!r}"
            )

        requires = _parse_requires(tool_cfg.get("requires"), tool_name=tool_name)
        post = _parse_postcondition(
            tool_cfg.get("postcondition"), tool_name=tool_name
        )
        parsed[tool_name] = ToolContract(
            name=tool_name, requires=requires, postcondition=post
        )

    return parsed


def _parse_requires(
    raw: object, *, tool_name: str
) -> tuple[ParsedPredicate, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"tools.{tool_name}.requires must be a list of predicate strings, "
            f"got {type(raw).__name__}"
        )
    out: list[ParsedPredicate] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError(
                f"tools.{tool_name}.requires entries must be strings, "
                f"got {type(entry).__name__}"
            )
        out.append(parse_predicate(entry))
    return tuple(out)


def _parse_postcondition(
    raw: object, *, tool_name: str
) -> ToolPostcondition | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"tools.{tool_name}.postcondition must be a mapping, "
            f"got {type(raw).__name__}"
        )
    unknown = set(raw) - _POSTCOND_KEYS
    if unknown:
        raise ValueError(
            f"tools.{tool_name}.postcondition has unknown key(s): "
            f"{sorted(unknown)!r}; allowed: {sorted(_POSTCOND_KEYS)!r}"
        )
    returns = _parse_returns(raw.get("returns"), tool_name=tool_name)
    state_updates = _parse_state_updates(
        raw.get("state_updates"), tool_name=tool_name
    )
    return ToolPostcondition(returns=returns, state_updates=state_updates)


def _parse_returns(raw: object, *, tool_name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"tools.{tool_name}.postcondition.returns must be a mapping, "
            f"got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ValueError(
                f"tools.{tool_name}.postcondition.returns key must be str, "
                f"got {type(k).__name__}"
            )
        if not isinstance(v, str):
            raise ValueError(
                f"tools.{tool_name}.postcondition.returns[{k!r}] type must "
                f"be a string (e.g. 'str', 'int'), got {type(v).__name__}"
            )
        out[k] = v
    return out


_RETURN_TYPE_MAP: dict[str, type | None] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "list": list,
    "array": list,
    "dict": dict,
    "object": dict,
    "null": type(None),
    "any": None,
}


def validate_returns(observed: object, schema: dict[str, str]) -> list[str]:
    """Validate *observed* tool output against a flat ``returns`` schema.

    Returns a list of human-readable error messages. Empty list = pass.
    Unknown type names produce an error (typo guard).
    """
    errors: list[str] = []
    if not schema:
        return errors
    if not isinstance(observed, dict):
        errors.append(
            f"expected dict tool_output, got {type(observed).__name__}"
        )
        return errors
    for key, type_str in schema.items():
        if key not in observed:
            errors.append(f"missing key {key!r}")
            continue
        type_str_low = type_str.lower()
        if type_str_low not in _RETURN_TYPE_MAP:
            errors.append(
                f"unknown type {type_str!r} for key {key!r}"
            )
            continue
        expected = _RETURN_TYPE_MAP[type_str_low]
        if expected is None:  # 'any' — no constraint
            continue
        value = observed[key]
        # bool subclasses int — disambiguate.
        if expected is int and isinstance(value, bool):
            errors.append(f"key {key!r} expected int, got bool")
            continue
        if expected is float and isinstance(value, bool):
            errors.append(f"key {key!r} expected float, got bool")
            continue
        if expected is float and isinstance(value, int):
            continue  # accept ints where float is expected
        if not isinstance(value, expected):
            errors.append(
                f"key {key!r} expected {type_str}, "
                f"got {type(value).__name__}"
            )
    return errors


def resolve_state_updates(
    updates: dict[str, StateValue], output: object
) -> dict[str, StateValue]:
    """Resolve ``{key}`` template tokens in *updates* against *output*.

    Single-token only — values like ``"{cabin}"`` look up ``output['cabin']``.
    Non-template values pass through. Resolved values that are not flat
    (str/int/float/bool/None) are coerced to ``None`` rather than crashing.
    """
    resolved: dict[str, StateValue] = {}
    out_dict = output if isinstance(output, dict) else None
    for k, v in updates.items():
        if (
            isinstance(v, str)
            and len(v) >= 3
            and v.startswith("{")
            and v.endswith("}")
        ):
            ref_key = v[1:-1]
            ref_value = out_dict.get(ref_key) if out_dict is not None else None
            if isinstance(ref_value, (str, int, float, bool)) or ref_value is None:
                resolved[k] = ref_value
            else:
                resolved[k] = None
        else:
            resolved[k] = v
    return resolved


def _parse_state_updates(
    raw: object, *, tool_name: str
) -> dict[str, StateValue]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"tools.{tool_name}.postcondition.state_updates must be a "
            f"mapping, got {type(raw).__name__}"
        )
    out: dict[str, StateValue] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            raise ValueError(
                f"tools.{tool_name}.postcondition.state_updates key must be "
                f"str, got {type(k).__name__}"
            )
        if v is None or isinstance(v, (str, bool, int, float)):
            out[k] = v
        else:
            raise ValueError(
                f"tools.{tool_name}.postcondition.state_updates[{k!r}] must "
                f"be a flat value (str/int/float/bool/None), "
                f"got {type(v).__name__}"
            )
    return out
