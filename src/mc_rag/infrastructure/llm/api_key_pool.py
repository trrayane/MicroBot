"""Thread-safe round-robin API-key pool with per-key cooldowns.

Used by the Gemini client to rotate to the next key when one hits its
quota (HTTP 429 / ``RESOURCE_EXHAUSTED``). A penalised key is skipped
until its cooldown elapses; once every key is cooling down the caller
receives a rate-limit error and can retry later.
"""

from __future__ import annotations

import threading
import time


class ApiKeyPool:
    """Round-robin pool of API keys with per-key cooldowns.

    ``get()`` returns the next usable key (skipping keys still cooling
    down) or ``None`` when every key is unavailable. ``penalize()`` puts
    a key on cooldown for ``seconds`` (defaults to ``cooldown_seconds``).
    """

    def __init__(self, keys: list[str], *, cooldown_seconds: float = 300.0) -> None:
        seen: set[str] = set()
        self._keys: list[str] = [k for k in keys if k and not (k in seen or seen.add(k))]
        self._lock = threading.Lock()
        self._cursor = 0
        self._blocked_until: dict[str, float] = {}
        self.cooldown_seconds = cooldown_seconds
        self._stats: dict[str, dict[str, float | int | None]] = {
            key: {
                "calls": 0,
                "rate_limits": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "last_used_at": None,
            }
            for key in self._keys
        }

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def size(self) -> int:
        """Number of distinct keys in the pool."""
        return len(self._keys)

    def get(self) -> str | None:
        """Return the next usable key in round-robin order, or ``None``."""
        with self._lock:
            if not self._keys:
                return None
            now = time.monotonic()
            for _ in range(len(self._keys)):
                key = self._keys[self._cursor % len(self._keys)]
                self._cursor += 1
                if self._blocked_until.get(key, 0.0) <= now:
                    return key
            return None

    def penalize(self, key: str, seconds: float | None = None) -> None:
        """Cool *key* down for *seconds* (default: the pool cooldown)."""
        duration = seconds if seconds and seconds > 0 else self.cooldown_seconds
        with self._lock:
            until = time.monotonic() + duration
            if until > self._blocked_until.get(key, 0.0):
                self._blocked_until[key] = until
            stats = self._stats.setdefault(key, {})
            stats["rate_limits"] = int(stats.get("rate_limits", 0)) + 1

    def record_usage(self, key: str, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Record a successful call on *key* (call count, tokens, last use)."""
        with self._lock:
            stats = self._stats.setdefault(key, {})
            stats["calls"] = int(stats.get("calls", 0)) + 1
            stats["prompt_tokens"] = int(stats.get("prompt_tokens", 0)) + max(prompt_tokens, 0)
            stats["completion_tokens"] = int(stats.get("completion_tokens", 0)) + max(
                completion_tokens, 0
            )
            stats["last_used_at"] = time.time()

    def snapshot(self) -> list[dict[str, float | int | str | None]]:
        """Per-key status for the admin dashboard (masked keys only)."""
        with self._lock:
            now = time.monotonic()
            now_wall = time.time()
            rows: list[dict[str, float | int | str | None]] = []
            for key in self._keys:
                stats = self._stats.get(key, {})
                blocked_until = self._blocked_until.get(key, 0.0)
                cooldown_remaining = max(0.0, blocked_until - now)
                last_used = stats.get("last_used_at")
                rows.append(
                    {
                        "key": (
                            f"{key[:4]}…{key[-4:]}" if len(key) >= 9 else "****"
                        ),
                        "status": "cooling" if cooldown_remaining > 0 else "available",
                        "cooldown_remaining_s": round(cooldown_remaining, 1),
                        "calls": int(stats.get("calls", 0)),
                        "rate_limits": int(stats.get("rate_limits", 0)),
                        "prompt_tokens": int(stats.get("prompt_tokens", 0)),
                        "completion_tokens": int(stats.get("completion_tokens", 0)),
                        "tokens": int(stats.get("prompt_tokens", 0))
                        + int(stats.get("completion_tokens", 0)),
                        "last_used_ago_s": round(now_wall - last_used, 1) if last_used else None,
                    }
                )
            return rows

    def reset(self, key: str | None = None) -> None:
        """Clear the cooldown of *key*, or of every key when ``None``."""
        with self._lock:
            if key is None:
                self._blocked_until.clear()
            else:
                self._blocked_until.pop(key, None)

    def available(self) -> int:
        """How many keys are currently usable."""
        with self._lock:
            now = time.monotonic()
            return sum(1 for until in self._blocked_until.values() if until <= now) + sum(
                1 for key in self._keys if key not in self._blocked_until
            )

    def masked_keys(self) -> list[str]:
        """Short masked identities of every key (safe to log)."""
        return [f"{key[:6]}****" if len(key) >= 6 else "****" for key in self._keys]

    def raw_keys(self) -> list[str]:
        """The raw keys (for redaction of error bodies — never log these)."""
        return list(self._keys)
