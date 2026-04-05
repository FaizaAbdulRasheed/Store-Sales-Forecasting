"""Lightweight stage timer for pipeline profiling."""

import time
import logging
from contextlib import contextmanager
from typing import Dict, List

logger = logging.getLogger(__name__)


class PipelineTimer:
    """Records wall-clock time for each named pipeline stage."""

    def __init__(self):
        self._times: Dict[str, float] = {}
        self._order: List[str] = []

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._times[name] = elapsed
            if name not in self._order:
                self._order.append(name)
            logger.info(f"⏱  {name}: {elapsed:.2f}s")

    def summary(self) -> str:
        lines = ["Pipeline Timing Summary", "─" * 40]
        total = 0.0
        for stage in self._order:
            t = self._times[stage]
            total += t
            lines.append(f"  {stage:<30} {t:>6.2f}s")
        lines.append("─" * 40)
        lines.append(f"  {'TOTAL':<30} {total:>6.2f}s")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, float]:
        return dict(self._times)
