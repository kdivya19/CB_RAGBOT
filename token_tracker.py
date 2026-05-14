"""
Token Usage Tracker
───────────────────
Tracks daily token consumption and triggers LLM switching
when the primary LLM's free tier limit is reached.

Usage is persisted to a JSON file so it survives server restarts.
The counter resets automatically at midnight (UTC).
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


# ─── Configuration ───────────────────────────────────────────
# Set your daily token budget here.
# Gemini 2.5 Flash free tier is dynamic, but commonly ~1,000,000 TPD.
# Set a conservative limit so you switch BEFORE hitting a 429 error.
DEFAULT_DAILY_TOKEN_LIMIT = 800_000  # tokens per day (conservative)

USAGE_FILE = Path(__file__).parent / "token_usage.json"


class TokenTracker:
    """Thread-safe daily token usage tracker with file persistence."""

    def __init__(self, daily_limit: int = DEFAULT_DAILY_TOKEN_LIMIT):
        self.daily_limit = daily_limit
        self._lock = threading.RLock()
        self._usage = self._load_usage()

    # ─── Core API ────────────────────────────────────────────

    def add_tokens(self, input_tokens: int, output_tokens: int) -> dict:
        """Record token usage. Returns updated stats."""
        with self._lock:
            self._maybe_reset()
            self._usage["input_tokens"] += input_tokens
            self._usage["output_tokens"] += output_tokens
            self._usage["total_tokens"] += (input_tokens + output_tokens)
            self._usage["request_count"] += 1
            self._save_usage()
            return self.get_stats()

    def is_limit_reached(self) -> bool:
        """Check if daily token limit has been exceeded."""
        with self._lock:
            self._maybe_reset()
            return self._usage["total_tokens"] >= self.daily_limit

    def get_stats(self) -> dict:
        """Get current usage statistics."""
        with self._lock:
            self._maybe_reset()
            total = self._usage["total_tokens"]
            return {
                "date": self._usage["date"],
                "input_tokens": self._usage["input_tokens"],
                "output_tokens": self._usage["output_tokens"],
                "total_tokens": total,
                "request_count": self._usage["request_count"],
                "daily_limit": self.daily_limit,
                "remaining_tokens": max(0, self.daily_limit - total),
                "usage_percent": round((total / self.daily_limit) * 100, 1),
                "limit_reached": total >= self.daily_limit,
            }

    def get_active_llm_name(self) -> str:
        """Returns which LLM should be used right now."""
        return "fallback" if self.is_limit_reached() else "primary"

    # ─── Persistence ─────────────────────────────────────────

    def _today_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _empty_usage(self) -> dict:
        return {
            "date": self._today_str(),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        }

    def _maybe_reset(self):
        """Reset counters if the date has rolled over (new day)."""
        if self._usage["date"] != self._today_str():
            self._usage = self._empty_usage()
            self._save_usage()

    def _load_usage(self) -> dict:
        """Load usage from disk, or start fresh."""
        if USAGE_FILE.exists():
            try:
                with open(USAGE_FILE, "r") as f:
                    data = json.load(f)
                # If it's a stale date, reset
                if data.get("date") != self._today_str():
                    return self._empty_usage()
                return data
            except (json.JSONDecodeError, KeyError):
                return self._empty_usage()
        return self._empty_usage()

    def _save_usage(self):
        """Persist usage to disk."""
        with open(USAGE_FILE, "w") as f:
            json.dump(self._usage, f, indent=2)


# ─── Singleton instance ─────────────────────────────────────
tracker = TokenTracker()
