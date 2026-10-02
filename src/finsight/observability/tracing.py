"""Lightweight observability: structured step timing + a metrics collector.

Dependency-free on purpose. In production this is where you would emit
OpenTelemetry spans (or ship traces to LangSmith); the stdlib implementation
keeps the instrumentation behaviour obvious, deterministic, and testable.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepMetrics:
    name: str
    duration_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Accumulates per-step timing and metadata for a single run."""

    def __init__(self) -> None:
        self.steps: list[StepMetrics] = []

    def record(self, name: str, duration_ms: float, **extra: Any) -> None:
        self.steps.append(StepMetrics(name=name, duration_ms=duration_ms, extra=extra))

    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    def summary(self) -> str:
        return "\n".join(f"  {s.name:<20} {s.duration_ms:8.1f} ms" for s in self.steps)


@contextmanager
def timed(collector: MetricsCollector, name: str, **extra: Any) -> Iterator[None]:
    """Context manager that records the wall-clock duration of a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        collector.record(name, (time.perf_counter() - start) * 1000.0, **extra)
