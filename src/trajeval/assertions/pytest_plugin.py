"""TrajEval pytest plugin — @trajectory_test decorator.

Marks a test function as a trajectory test.  The decorated function must
return a :class:`~trajeval.sdk.models.Trace`.  After the function returns,
every provided assertion callable is run against that trace.  Any assertion
failures are collected and reported together.

Usage::

    from trajeval.assertions.pytest_plugin import trajectory_test
    from trajeval.assertions.core import cost_within, max_depth, no_cycles

    @trajectory_test(
        no_cycles,
        max_depth(5),
        cost_within(p90=1.50),
    )
    async def test_booking_agent(agent_graph) -> Trace:
        cb = TrajEvalCallback()
        await agent_graph.ainvoke({"query": "find flights"}, config={"callbacks": [cb]})
        return cb.get_trace()

The ``@trajectory_test`` decorator is compatible with both sync and async
test functions, and cooperates with ``pytest-asyncio`` auto mode.

Layer rule: assertions/pytest_plugin.py may import from assertions/core.py
and sdk/, but never from trajeval.backend.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable
from typing import Any

from trajeval.assertions.core import AssertionFn
from trajeval.sdk.models import Trace


def trajectory_test(
    *assertions: AssertionFn,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory: run *assertions* against the Trace returned by the test.

    Parameters
    ----------
    *assertions:
        Zero or more callables with signature ``(trace: Trace) -> None``.
        Each is called after the test function returns.  All failures are
        collected; the test fails with a combined message.

    Returns
    -------
    Callable
        A pytest-compatible decorator (sync or async, matching the original).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:  # noqa: ANN202
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> None:
                trace: Trace = await fn(*args, **kwargs)
                _run_assertions(trace, assertions)

            return async_wrapper

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> None:
                result = fn(*args, **kwargs)
                # Support functions that return a coroutine (edge case)
                if asyncio.iscoroutine(result):
                    trace = asyncio.get_event_loop().run_until_complete(result)
                else:
                    trace = result
                _run_assertions(trace, assertions)

            return sync_wrapper

    return decorator


def _run_assertions(trace: Trace, assertions: tuple[AssertionFn, ...]) -> None:
    """Run every assertion against *trace*, collecting all failures."""
    if not assertions:
        return

    failures: list[str] = []
    for assertion in assertions:
        try:
            assertion(trace)
        except AssertionError as exc:
            failures.append(str(exc))

    if failures:
        count = len(failures)
        joined = "\n".join(f"  [{i + 1}] {msg}" for i, msg in enumerate(failures))
        raise AssertionError(
            f"{count} trajectory assertion(s) failed for trace "
            f"'{trace.trace_id}':\n{joined}"
        )
