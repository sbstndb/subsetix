"""
CuPy-aware micro-benchmark harness for subsetix.

Each benchmark case returns a callable suitable for :func:`cupyx.profiler.benchmark`.
Use ``python -m subsetix_cupy.benchmarks`` to list or execute the suites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import cupyx.profiler

from ..expressions import _require_cupy


@dataclass
class BenchmarkTarget:
    """
    Concrete benchmark callable returned by a setup function.

    Attributes
    ----------
    func:
        Zero-argument callable performing the work to benchmark.
    repeat:
        How many timed iterations to perform (passed to ``cupyx.profiler.benchmark``).
    warmup:
        Number of warmup iterations to discard before timing.
    metadata:
        Arbitrary information (problem size, etc.) displayed alongside timings.
    """

    func: Callable[[], Any]
    repeat: int = 50
    warmup: int = 5
    metadata: Dict[str, Any] | None = None


@dataclass
class BenchmarkCase:
    """
    Describes a benchmark scenario.

    Attributes
    ----------
    name:
        Identifier used for CLI filtering.
    description:
        Human readable summary.
    setup:
        Callable receiving the CuPy module and returning a :class:`BenchmarkTarget`.
    """

    name: str
    description: str
    setup: Callable[[Any], BenchmarkTarget]


def run_case(case: BenchmarkCase, *, repeat: Optional[int] = None, warmup: Optional[int] = None):
    """
    Execute a benchmark case and return the Cupy benchmark result.
    """

    cp = _require_cupy()
    target = case.setup(cp)
    repetitions = repeat if repeat is not None else target.repeat
    warmups = warmup if warmup is not None else target.warmup
    if repetitions <= 0:
        raise ValueError("repeat must be positive")
    if warmups < 0:
        raise ValueError("warmup must be >= 0")

    cp.cuda.runtime.deviceSynchronize()
    result = cupyx.profiler.benchmark(
        target.func, n_repeat=repetitions, n_warmup=warmups
    )
    cp.cuda.runtime.deviceSynchronize()
    return result, target.metadata or {}, repetitions, warmups


def load_cases() -> List[BenchmarkCase]:
    """
    Discover all registered benchmark cases.
    """

    from .cases import CASES

    return list(CASES)


__all__ = [
    "BenchmarkCase",
    "BenchmarkTarget",
    "load_cases",
    "run_case",
]
