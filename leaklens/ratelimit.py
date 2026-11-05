"""A small, bounded sliding-window rate limiter.

The public scanner has no user accounts, so the unit of fairness is the client
address.  Buckets are capped: a flood of forged keys evicts old buckets instead
of growing the process without limit.  The limiter is thread-safe because
FastAPI may run synchronous endpoints in a worker thread.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque


class SlidingWindowLimiter:
    """Allow at most ``limit`` events per ``window_seconds`` for each key."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        max_keys: int = 2_048,
    ) -> None:
        if limit < 1 or window_seconds <= 0 or max_keys < 1:
            raise ValueError("rate-limit settings must be positive")
        self.limit = int(limit)
        self.window_seconds = float(window_seconds)
        self.max_keys = int(max_keys)
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Record one event and return ``(allowed, retry_after_seconds)``."""
        moment = time.monotonic() if now is None else float(now)
        bucket_key = (key or "unknown")[:96]
        with self._lock:
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    self._evict_locked(moment)
                bucket = deque()
                self._buckets[bucket_key] = bucket
            else:
                self._buckets.move_to_end(bucket_key)
            cutoff = moment - self.window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry = max(1, int(bucket[0] + self.window_seconds - moment) + 1)
                return False, retry
            bucket.append(moment)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def _evict_locked(self, now: float) -> None:
        """Drop stale buckets first, then the least-recently-used key."""
        cutoff = now - self.window_seconds
        for key in list(self._buckets):
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                del self._buckets[key]
        while len(self._buckets) >= self.max_keys:
            self._buckets.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
