from __future__ import annotations

from medrag.ratelimit import RateLimiter


def test_waits_when_calls_too_fast():
    clock_values = [0.0]
    sleeps = []

    def fake_clock():
        return clock_values[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock_values[0] += seconds

    limiter = RateLimiter(max_per_second=3.0, clock=fake_clock, sleep=fake_sleep)

    limiter.wait()  # first call, no wait
    clock_values[0] += 0.05  # simulate 50ms elapsed, less than 1/3s interval
    limiter.wait()  # should sleep ~1/3 - 0.05

    assert len(sleeps) == 1
    assert abs(sleeps[0] - (1.0 / 3.0 - 0.05)) < 1e-9


def test_no_wait_when_calls_already_spaced_out():
    clock_values = [0.0]
    sleeps = []

    def fake_clock():
        return clock_values[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)

    limiter = RateLimiter(max_per_second=3.0, clock=fake_clock, sleep=fake_sleep)

    limiter.wait()
    clock_values[0] += 10.0  # plenty of time has passed
    limiter.wait()

    assert sleeps == []
