from __future__ import annotations

import time
from typing import Callable


class RateLimiter:
    """Simple token-interval rate limiter. Clock/sleep are injectable for tests."""

    def __init__(
        self,
        max_per_second: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last_call = now
