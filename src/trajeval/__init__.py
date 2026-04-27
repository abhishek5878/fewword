"""TrajEval — trajectory evaluation for production AI agents.

Two-line quickstart::

    import trajeval
    trajeval.init()

That's it. TrajEval auto-detects installed frameworks (LangGraph,
OpenAI), instruments them, and accumulates traces. After your agent
runs, call ``trajeval.get_traces()`` to retrieve them, or use
``trajeval.end_session()`` to finalize + run checks from
``.trajeval.yml`` if present.

For explicit control, use the SDK directly::

    from trajeval.sdk.callback import TrajEvalCallback
    cb = TrajEvalCallback(agent_id="my-agent", mode="guard", ...)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from trajeval.sdk.models import Trace

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_sessions: list[Trace] = []
_active_callback: object | None = None  # TrajEvalCallback or None
_initialized: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init(
    *,
    agent_id: str = "default",
    version_hash: str = "unknown",
    mode: Literal["observe", "guard"] = "observe",
    config_path: str | None = None,
) -> object:
    """Initialize TrajEval with 1 line. Auto-instruments installed frameworks.

    Parameters
    ----------
    agent_id:
        Identifier for the agent being traced.
    version_hash:
        Git SHA or version string for the agent.
    mode:
        ``"observe"`` (default) records traces silently.
        ``"guard"`` blocks tool calls that violate contracts from
        ``.trajeval.yml``.
    config_path:
        Path to a ``.trajeval.yml`` config file. If ``None``, looks
        for ``.trajeval.yml`` in the current directory.

    Returns
    -------
    The :class:`~trajeval.sdk.callback.TrajEvalCallback` instance
    (useful for passing to ``graph.invoke(config={"callbacks": [cb]})``
    if auto-patching doesn't cover your framework).
    """
    global _active_callback, _initialized  # noqa: PLW0603

    from trajeval.sdk.callback import TrajEvalCallback

    guard_assertions = None
    ltl_formulas = None
    if mode == "guard":
        guard_assertions, ltl_formulas = _load_guard_config(config_path)

    cb = TrajEvalCallback(
        agent_id=agent_id,
        version_hash=version_hash,
        mode=mode,
        guard_assertions=guard_assertions,
        ltl_formulas=ltl_formulas,
    )
    _active_callback = cb
    _initialized = True

    _auto_patch(cb)

    return cb


def end_session(label: str = "Success") -> Trace | None:
    """Finalize the current session and store the trace.

    Parameters
    ----------
    label:
        Human-readable label for the session outcome.

    Returns
    -------
    The completed :class:`Trace`, or ``None`` if no session is active.
    """
    global _active_callback  # noqa: PLW0603

    if _active_callback is None:
        return None

    from trajeval.sdk.callback import TrajEvalCallback

    if not isinstance(_active_callback, TrajEvalCallback):
        return None

    trace = _active_callback.get_trace()
    _sessions.append(trace)
    _active_callback = None
    return trace


def get_traces() -> list[Trace]:
    """Return all completed session traces since ``init()``."""
    return list(_sessions)


def get_callback() -> object | None:
    """Return the active callback (for manual wiring)."""
    return _active_callback


def reset() -> None:
    """Clear all state. Useful between test runs."""
    global _active_callback, _initialized  # noqa: PLW0603
    _sessions.clear()
    _active_callback = None
    _initialized = False


# ---------------------------------------------------------------------------
# Auto-patching
# ---------------------------------------------------------------------------


def _auto_patch(cb: object) -> None:
    """Auto-instrument installed frameworks.

    Currently supports:
    - LangGraph / LangChain (via langchain_core callback injection)
    """
    _try_patch_langchain(cb)


def _try_patch_langchain(cb: object) -> None:
    """Inject callback into langchain_core's default callbacks if available."""
    try:
        from langchain_core.callbacks.manager import (
            CallbackManager,
        )

        # Patch the default callback manager to include our callback
        _original_configure = CallbackManager.configure

        @staticmethod  # type: ignore[misc]
        def _patched_configure(
            inheritable_callbacks: object = None,
            local_callbacks: object = None,
            verbose: bool = False,
            inheritable_tags: object = None,
            local_tags: object = None,
            inheritable_metadata: object = None,
            local_metadata: object = None,
        ) -> object:
            manager = _original_configure(
                inheritable_callbacks,
                local_callbacks,
                verbose,
                inheritable_tags,
                local_tags,
                inheritable_metadata,
                local_metadata,
            )
            # Inject our callback if not already present
            if (
                hasattr(manager, "inheritable_handlers")
                and cb not in manager.inheritable_handlers
            ):
                manager.inheritable_handlers.append(cb)
            return manager

        CallbackManager.configure = _patched_configure  # type: ignore[assignment]
    except ImportError:
        pass  # langchain_core not installed — skip


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_guard_config(
    config_path: str | None,
) -> tuple[list[object] | None, list[object] | None]:
    """Load guard assertions + LTL formulas from .trajeval.yml.

    Returns (guard_assertions, ltl_formulas) — either may be None.
    """
    if config_path:
        path = Path(config_path)
    elif Path(".fewwords.yml").exists():
        path = Path(".fewwords.yml")
    elif Path(".trajeval.yml").exists():
        path = Path(".trajeval.yml")
    else:
        return None, None
    if not path.exists():
        return None, None

    try:
        from trajeval.action import load_config

        config = load_config(path)
    except Exception:
        return None, None

    from trajeval.assertions.core import (
        must_visit,
        never_calls,
        no_retry_storm,
    )

    guards: list[object] = []
    for tool in config.banned_tools:
        guards.append(functools.partial(never_calls, tool=tool))
    if config.required_tools:
        guards.append(
            functools.partial(must_visit, tools=config.required_tools)
        )
    guards.append(
        functools.partial(no_retry_storm, max_consecutive=config.max_retries)
    )

    # Compile NL contracts to LTL formulas
    ltl_formulas: list[object] | None = None
    if config.contracts:
        try:
            from trajeval.analysis.ltl_compiler import (
                compile_contract,
                extract_ltl_formulas,
            )

            compiled = compile_contract(config.contracts)
            formulas = extract_ltl_formulas(compiled)
            if formulas:
                ltl_formulas = formulas  # type: ignore[assignment]
        except Exception:  # noqa: S110
            pass  # compilation failure — guard still works without LTL

    return (guards if guards else None, ltl_formulas)
