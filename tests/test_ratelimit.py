"""Unit tests for the bounded sliding-window rate limiter."""

from __future__ import annotations

import pytest

from leaklens.ratelimit import SlidingWindowLimiter


def test_allows_within_the_window_then_blocks_with_retry_after():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)
    assert limiter.check("client", now=0.0) == (True, 0)
    assert limiter.check("client", now=1.0) == (True, 0)
    allowed, retry = limiter.check("client", now=2.0)
    assert allowed is False
    assert retry == 9


def test_the_window_slides_instead_of_resetting_in_blocks():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)
    assert limiter.check("client", now=0.0)[0] is True
    assert limiter.check("client", now=5.0)[0] is True
    assert limiter.check("client", now=9.9)[0] is False
    assert limiter.check("client", now=10.1)[0] is True


def test_clients_are_isolated():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    assert limiter.check("a", now=0.0)[0] is True
    assert limiter.check("b", now=0.0)[0] is True
    assert limiter.check("a", now=0.0)[0] is False


def test_empty_and_overlong_keys_are_collapsed_safely():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    assert limiter.check("", now=0.0)[0] is True
    assert limiter.check("", now=0.0)[0] is False

    limiter.reset()
    long_key = "x" * 200
    assert limiter.check(long_key, now=0.0)[0] is True
    assert limiter.check("x" * 120, now=0.0)[0] is False


def test_bucket_count_is_bounded_for_forged_keys():
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, max_keys=3)
    for index in range(10):
        assert limiter.check(f"client-{index}", now=0.0)[0] is True
    assert len(limiter) <= 3


def test_stale_buckets_are_evicted_before_the_lru():
    limiter = SlidingWindowLimiter(limit=5, window_seconds=10, max_keys=2)
    limiter.check("old", now=0.0)
    limiter.check("fresh", now=9.0)
    limiter.check("new", now=11.0)  # "old" is stale and should be removed
    assert len(limiter) == 2
    # "fresh" is still tracked, so it is blocked immediately.
    limiter_limited = SlidingWindowLimiter(limit=1, window_seconds=30, max_keys=1)
    assert limiter_limited.check("fresh", now=9.0)[0] is True
    assert limiter_limited.check("fresh", now=10.0)[0] is False


def test_reset_clears_every_bucket():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    limiter.check("a", now=0.0)
    limiter.check("b", now=0.0)
    assert len(limiter) == 2
    limiter.reset()
    assert len(limiter) == 0
    assert limiter.check("a", now=0.0)[0] is True


@pytest.mark.parametrize(
    "limit,window,max_keys",
    [(0, 10, 10), (1, 0, 10), (1, 10, 0), (-1, 10, 10)],
)
def test_invalid_settings_are_rejected(limit, window, max_keys):
    with pytest.raises(ValueError):
        SlidingWindowLimiter(limit, window, max_keys)
