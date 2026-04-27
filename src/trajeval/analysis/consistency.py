"""pass^k consistency metric — measures agent reliability across repeated runs.

Two distinct estimators live here — they answer different questions:

pass^k (consistency threshold)
-------------------------------
``pass_k(traces, assertion, k)`` counts how many of N runs passed and checks
whether that count meets the threshold k.  ``score = pass_count / n`` is the
raw pass rate.  Use this to enforce "at least 8/10 runs must honour the
ordering contract".

pass_at_k_unbiased (combinatorial estimator)
---------------------------------------------
``pass_at_k_unbiased(n, c, k)`` answers a different question: *if you sampled
k runs at random from the N you observed, what is the probability that at
least one of them passes?*  This is the unbiased estimator from Chen et al.
2021 (the Codex / HumanEval paper):

    pass@k = 1 − C(n − c, k) / C(n, k)

It is more statistically sound than ``c/n`` when n is small relative to k,
because ``c/n`` is a biased estimator of pass@k.  For large n (n >> k) both
converge, but for n=10, k=5 the combinatorial estimator avoids the upward
bias in the naïve fraction.

Which one to use
----------------
- **Threshold checking** (does this agent meet our reliability bar?): ``pass_k``
- **Estimating capability** (what's the probability a new k-run batch succeeds?):
  ``pass_at_k_unbiased``
- **Confidence intervals** on either: ``wilson_ci`` wraps the underlying
  proportion with a 95 % Wilson score interval, which is preferred over the
  normal approximation for small n and extreme proportions.

Layer rule: pure Python, no FastAPI imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trajeval.sdk.models import Trace

# Any callable (Trace) -> None that raises AssertionError on failure.


@dataclass(frozen=True)
class ConsistencyResult:
    """Result of a pass^k consistency evaluation.

    Attributes
    ----------
    score:
        Raw pass rate: ``pass_count / total``.  In [0.0, 1.0].
    pass_at_k:
        Unbiased combinatorial estimate of the probability that at least one
        of *k* randomly sampled runs passes.  Uses ``1 − C(n−c, k) / C(n, k)``
        from Chen et al. 2021.  Equal to *score* when k == 1.
    ci_lower:
        Lower bound of the 95 % Wilson score confidence interval on *score*.
    ci_upper:
        Upper bound of the 95 % Wilson score confidence interval on *score*.
    pass_count:
        Number of traces where the assertion passed (no AssertionError raised).
    fail_count:
        Number of traces where the assertion failed (AssertionError raised).
    total:
        Total number of traces evaluated (``pass_count + fail_count``).
    meets_k:
        True if ``pass_count >= k``, i.e. the task passed in at least k runs.
    k:
        The required minimum pass count used to determine ``meets_k``.
    failures:
        List of (trace_index, error_message) for each failing trace.
        Useful for debugging which runs are inconsistent.
    """

    score: float
    pass_at_k: float
    ci_lower: float
    ci_upper: float
    pass_count: int
    fail_count: int
    total: int
    meets_k: bool
    k: int
    failures: list[tuple[int, str]]


# ---------------------------------------------------------------------------
# Public estimators
# ---------------------------------------------------------------------------


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased combinatorial estimator: P(at least 1 of k sampled runs passes).

    From Chen et al. 2021 (Codex / HumanEval):

        pass@k = 1 − C(n − c, k) / C(n, k)

    This avoids the upward bias in the naïve ``c/n`` estimate when n is small
    relative to k.  For n >> k both converge.

    Parameters
    ----------
    n:
        Total number of runs observed.
    c:
        Number of runs that passed (``c <= n``).
    k:
        Number of runs to sample.  Must satisfy ``k <= n``.

    Returns
    -------
    float
        Probability in [0.0, 1.0].

    Raises
    ------
    ValueError
        If ``n < 1``, ``c < 0``, ``c > n``, or ``k < 1`` or ``k > n``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0 <= c <= n):
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, n], got k={k}, n={n}")

    # If more runs pass than there are failing slots to fill k draws from,
    # it's impossible to draw k failures — so at least one pass is guaranteed.
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson_ci(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Preferred over the normal approximation for small *total* and extreme
    proportions (near 0 or 1).  Returns (lower, upper) in [0.0, 1.0].

    Parameters
    ----------
    successes:
        Number of successes (pass_count).
    total:
        Total number of trials.
    confidence:
        Confidence level.  Default 0.95 (95 %).

    Raises
    ------
    ValueError
        If ``total < 1`` or ``successes > total``.
    """
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    if successes > total:
        raise ValueError(f"successes ({successes}) > total ({total})")

    # z for confidence level (two-tailed)
    # Using pre-computed values for common levels to avoid scipy dependency.
    _Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    z = _Z.get(confidence)
    if z is None:
        # Fallback: normal quantile approximation for other levels
        p = 1.0 - (1.0 - confidence) / 2.0
        # Rational approximation (Abramowitz & Stegun 26.2.17)
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
            1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
        )

    n = total
    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def pass_k(
    traces: list[Trace],
    assertion: object,
    k: int,
) -> ConsistencyResult:
    """Evaluate assertion consistency across N repeated runs of the same task.

    Parameters
    ----------
    traces:
        List of traces from N repeated runs of the same task.
        All traces should represent the same task/prompt with the same agent —
        variation between them reflects agent non-determinism.
    assertion:
        A callable ``(Trace) -> None`` that raises ``AssertionError`` on failure.
        Compatible with TrajEval assertion DSL functions (e.g. ``tool_must_precede``).
    k:
        Required minimum number of passing runs.  ``meets_k`` is True iff
        ``pass_count >= k``.  Must be >= 1 and <= len(traces).

    Returns
    -------
    ConsistencyResult
        ``score = pass_count / total`` (raw pass rate).
        ``pass_at_k`` = unbiased combinatorial estimate of P(≥1 pass in k draws).
        ``ci_lower`` / ``ci_upper`` = 95 % Wilson score confidence interval.
        ``meets_k = pass_count >= k``.

    Raises
    ------
    ValueError
        If traces is empty, k < 1, or k > len(traces).
    TypeError
        If assertion is not callable.
    """
    if not traces:
        raise ValueError("traces must be non-empty")
    n = len(traces)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed len(traces) ({n})")
    if not callable(assertion):
        raise TypeError(f"assertion must be callable, got {type(assertion).__name__}")

    pass_count = 0
    failures: list[tuple[int, str]] = []

    for i, trace in enumerate(traces):
        try:
            assertion(trace)  # type: ignore[call-arg]
            pass_count += 1
        except AssertionError as exc:
            failures.append((i, str(exc)))

    fail_count = n - pass_count
    score = pass_count / n
    ci_lower, ci_upper = wilson_ci(pass_count, n)
    p_at_k = pass_at_k_unbiased(n, pass_count, k)

    return ConsistencyResult(
        score=score,
        pass_at_k=p_at_k,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pass_count=pass_count,
        fail_count=fail_count,
        total=n,
        meets_k=pass_count >= k,
        k=k,
        failures=failures,
    )


def consistency_score(
    traces: list[Trace],
    assertion: object,
) -> float:
    """Shorthand: return the raw consistency fraction for an assertion.

    Equivalent to ``pass_k(traces, assertion, k=1).score``.
    """
    return pass_k(traces, assertion, k=1).score
