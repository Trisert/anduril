"""Cumulative token usage metrics + display formatters."""

from __future__ import annotations

import time
from typing import Any, Optional

from anduril.pricing import pricing_for


def _abbr(n: int) -> str:
    """Compact token count for the stats footer: 832 → '832', 1500 → '1.5K', 78K, 1.2M."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}K"
    if n < 1_000_000:
        return f"{n // 1000}K"
    if n < 10_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n // 1_000_000}M"


def _precise_abbr(n: int) -> str:
    """Two-decimal abbr for live counters: 25152 → '25.15K'."""
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.2f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n // 1_000_000_000}B"


def _normalize_usage(usage: Any, timings: Optional[dict]) -> Optional[dict]:
    """Extract {input, output, cache_read, reasoning} from a finished call.

    Accepts either the standard streaming `usage` object (OpenAI / Z.ai) or
    the llama.cpp `timings` extra (where `usage` is null).
    """
    if (not usage) and timings and timings.get("predicted_n"):
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        gen_n = int(timings.get("predicted_n") or 0)
        return {
            "input_tokens": max(0, prompt_n - cache_n),
            "output_tokens": gen_n,
            "cache_read_tokens": cache_n,
            "reasoning_tokens": 0,
        }
    if not usage:
        return None
    prompt_total = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_total = int(getattr(usage, "completion_tokens", 0) or 0)
    ptd = getattr(usage, "prompt_tokens_details", None)
    cache_n = int(getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
    return {
        "input_tokens": max(0, prompt_total - cache_n),
        "output_tokens": max(0, completion_total),
        "cache_read_tokens": cache_n,
        "reasoning_tokens": 0,
    }


class _Metrics:
    """Cumulative token usage and cost for the current session."""

    def __init__(self, session_id: str, model: str | None = None,
                 started_at: float | None = None) -> None:
        self.session_id = session_id
        self.model = model
        self.started_at = started_at or time.time()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.reasoning_tokens = 0
        self.api_calls = 0
        # Total USD cost across the whole session. We track
        # this separately from per-turn costs so a single
        # turn can show what it just cost while the session
        # total accumulates independently.
        self.total_cost: float = 0.0
        # Per-model breakdown. A user who switches models
        # mid-session (via /model) gets a row per model
        # they've used, not a single amalgamated total.
        self.cost_by_model: dict[str, float] = {}
        # Cost cap (USD). ``None`` means no cap. When set,
        # the agent refuses to make further model calls
        # once ``total_cost >= budget``. Set via
        # ``/budget <usd>`` in the REPL.
        self.budget: float | None = None

    def add(self, delta: dict | None, model: str | None = None) -> None:
        """Add a usage delta. Optionally tag it with a model name.

        The model argument is what the agent was configured
        with at the time of the call. If the user has switched
        since then (``self.model`` may be different from
        ``model``), we record the cost under the model that
        actually produced it. ``self.model`` is only used as
        a fallback when the caller doesn't pass a model.
        """
        if not delta:
            return
        in_t = int(delta.get("input_tokens") or 0)
        out_t = int(delta.get("output_tokens") or 0)
        cache_t = int(delta.get("cache_read_tokens") or 0)
        reason_t = int(delta.get("reasoning_tokens") or 0)
        self.input_tokens += in_t
        self.output_tokens += out_t
        self.cache_read_tokens += cache_t
        self.reasoning_tokens += reason_t
        self.api_calls += 1
        # Cost: look up the pricing for the model that was
        # used. If we don't have pricing (e.g. an unknown
        # local model name) we silently skip the cost
        # update — the token counts still go up, the
        # cost-display formatter will show "—" for this
        # call.
        used_model = model or self.model
        p = pricing_for(used_model)
        if p is not None:
            cost = p.cost(
                input_tokens=in_t,
                cached_tokens=cache_t,
                output_tokens=out_t,
                reasoning_tokens=reason_t,
            )
            self.total_cost += cost
            key = used_model or "unknown"
            self.cost_by_model[key] = self.cost_by_model.get(key, 0.0) + cost

    def last_turn_cost(self) -> float:
        """Cost of the most recent call. 0.0 if no call yet.

        We re-derive this from the most recent call's
        delta; the agent's status bar reads it after the
        API's ``usage`` chunk lands. A simpler design would
        be a per-call log, but that's overkill — we just
        need the last one.
        """
        # We don't actually store per-turn costs separately
        # right now (only the cumulative). The TUI's
        # ``/cost`` command can show the cumulative; the
        # status bar shows the cumulative too (which is
        # what the user wants to see ticking up).
        return self.total_cost

    def as_meta(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "api_calls": self.api_calls,
            "started_at": self.started_at,
            "total_cost": self.total_cost,
            "cost_by_model": dict(self.cost_by_model),
        }

    def load(self, saved: dict) -> None:
        """Reload cumulative totals from a saved session JSON."""
        self.input_tokens = int(saved.get("input_tokens") or 0)
        self.output_tokens = int(saved.get("output_tokens") or 0)
        self.cache_read_tokens = int(saved.get("cache_read_tokens") or 0)
        self.reasoning_tokens = int(saved.get("reasoning_tokens") or 0)
        self.api_calls = int(saved.get("api_calls") or 0)
        if saved.get("started_at"):
            self.started_at = float(saved["started_at"])
        # Cost is best-effort: if the saved session was on
        # a different model whose price has since changed,
        # the loaded total is what was paid at the time.
        self.total_cost = float(saved.get("total_cost") or 0.0)
        for k, v in (saved.get("cost_by_model") or {}).items():
            self.cost_by_model[str(k)] = float(v)
