"""Per-model pricing for cost tracking.

anduril shows a per-turn cost in the status bar and a
per-model breakdown via ``/cost``. The pricing is held in a
small lookup table keyed by model name; unknown models
return ``None`` and the UI shows "—" instead of a number
(rather than displaying a misleading zero).

Prices are USD per **1 million tokens**, the unit every
major provider publishes in their public docs. All values
are last-verified against the relevant pricing page in
early 2026 and will drift; users can override per-model
via the ``ANDURIL_PRICING_OVERRIDES`` env var (a JSON
object: ``'{"gpt-4o": {"input": 5.0, "output": 15.0}}'``)
without having to edit this file.

The convention: a single ``ModelPricing`` dataclass per
known model with the four rates that matter — input,
cached input, output, and (for reasoning models) reasoning.
For models without a separate cache rate we fall back to
input (most providers charge full price for cache reads;
Anthropic and OpenAI charge ~10% but we'll be conservative).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


# === Data class ============================================================


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens for one model.

    All fields default to ``0.0`` for models that are
    free (local models, free-tier endpoints) — the
    formatter shows "$0.00" rather than "—" in that
    case, so the user knows the model was used but
    didn't cost anything.
    """

    name: str
    input_per_mtok: float = 0.0
    cached_input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0
    # Some reasoning models (o1, o3) bill reasoning tokens
    # separately from output. When zero, we treat reasoning
    # tokens as output tokens.
    reasoning_per_mtok: float = 0.0

    def cost(
        self,
        *,
        input_tokens: int = 0,
        cached_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> float:
        """USD cost for a usage dict.

        Reasoning tokens are billed at the reasoning rate
        if non-zero, otherwise rolled into output (the
        OpenAI convention pre-``o1``).
        """
        non_cached = max(0, input_tokens - cached_tokens)
        c = (
            non_cached * self.input_per_mtok
            + cached_tokens * self.cached_input_per_mtok
        ) / 1_000_000
        if reasoning_tokens and self.reasoning_per_mtok:
            c += reasoning_tokens * self.reasoning_per_mtok / 1_000_000
            c += output_tokens * self.output_per_mtok / 1_000_000
        else:
            c += (output_tokens + reasoning_tokens) * self.output_per_mtok / 1_000_000
        return c


# === Provider tables ======================================================


# Each entry: ``(model_substring, ModelPricing(...))``. Substring
# match is case-insensitive; first hit wins. List the more
# specific patterns first. Prices are USD per 1M tokens.

_PRICING_TABLE: tuple[tuple[str, ModelPricing], ...] = (
    # ----- OpenAI --------------------------------------------------------
    # Substring matches: list more specific patterns first. ``gpt-4o-mini``
    # must come before ``gpt-4o`` or the latter will shadow it.
    (
        "gpt-4.1-nano",
        ModelPricing(
            name="gpt-4.1-nano",
            input_per_mtok=0.10,
            cached_input_per_mtok=0.025,
            output_per_mtok=0.40,
        ),
    ),
    (
        "gpt-4.1-mini",
        ModelPricing(
            name="gpt-4.1-mini",
            input_per_mtok=0.40,
            cached_input_per_mtok=0.10,
            output_per_mtok=1.60,
        ),
    ),
    (
        "gpt-4.1",
        ModelPricing(
            name="gpt-4.1",
            input_per_mtok=3.00,
            cached_input_per_mtok=0.75,
            output_per_mtok=12.00,
        ),
    ),
    (
        "gpt-4o-mini",
        ModelPricing(
            name="gpt-4o-mini",
            input_per_mtok=0.15,
            cached_input_per_mtok=0.075,
            output_per_mtok=0.60,
        ),
    ),
    (
        "gpt-4o",
        ModelPricing(
            name="gpt-4o",
            input_per_mtok=2.50,
            cached_input_per_mtok=1.25,
            output_per_mtok=10.00,
        ),
    ),
    (
        "o4-mini",
        ModelPricing(
            name="o4-mini",
            input_per_mtok=1.10,
            cached_input_per_mtok=0.275,
            output_per_mtok=4.40,
            reasoning_per_mtok=4.40,
        ),
    ),
    (
        "o3-mini",
        ModelPricing(
            name="o3-mini",
            input_per_mtok=1.10,
            cached_input_per_mtok=0.55,
            output_per_mtok=4.40,
            reasoning_per_mtok=4.40,
        ),
    ),
    (
        "o3",
        ModelPricing(
            name="o3",
            input_per_mtok=10.00,
            cached_input_per_mtok=2.50,
            output_per_mtok=40.00,
            reasoning_per_mtok=40.00,
        ),
    ),
    (
        "o1-mini",
        ModelPricing(
            name="o1-mini",
            input_per_mtok=3.00,
            cached_input_per_mtok=1.50,
            output_per_mtok=12.00,
            reasoning_per_mtok=12.00,
        ),
    ),
    (
        "o1",
        ModelPricing(
            name="o1",
            input_per_mtok=15.00,
            cached_input_per_mtok=7.50,
            output_per_mtok=60.00,
            reasoning_per_mtok=60.00,
        ),
    ),
    (
        "gpt-5-mini",
        ModelPricing(
            name="gpt-5-mini",
            input_per_mtok=0.25,
            cached_input_per_mtok=0.0625,
            output_per_mtok=2.00,
        ),
    ),
    (
        "gpt-5",
        ModelPricing(
            name="gpt-5",
            input_per_mtok=5.00,
            cached_input_per_mtok=1.25,
            output_per_mtok=20.00,
        ),
    ),
    # ----- Anthropic -----------------------------------------------------
    (
        "claude-opus-4",
        ModelPricing(
            name="claude-opus-4",
            input_per_mtok=15.00,
            cached_input_per_mtok=1.50,
            output_per_mtok=75.00,
        ),
    ),
    (
        "claude-sonnet-4",
        ModelPricing(
            name="claude-sonnet-4",
            input_per_mtok=3.00,
            cached_input_per_mtok=0.30,
            output_per_mtok=15.00,
        ),
    ),
    (
        "claude-3-7-sonnet",
        ModelPricing(
            name="claude-3-7-sonnet",
            input_per_mtok=3.00,
            cached_input_per_mtok=0.30,
            output_per_mtok=15.00,
        ),
    ),
    (
        "claude-3-5-sonnet",
        ModelPricing(
            name="claude-3-5-sonnet",
            input_per_mtok=3.00,
            cached_input_per_mtok=0.30,
            output_per_mtok=15.00,
        ),
    ),
    (
        "claude-3-5-haiku",
        ModelPricing(
            name="claude-3-5-haiku",
            input_per_mtok=0.80,
            cached_input_per_mtok=0.08,
            output_per_mtok=4.00,
        ),
    ),
    (
        "claude-3-opus",
        ModelPricing(
            name="claude-3-opus",
            input_per_mtok=15.00,
            cached_input_per_mtok=1.50,
            output_per_mtok=75.00,
        ),
    ),
    (
        "claude-3-haiku",
        ModelPricing(
            name="claude-3-haiku",
            input_per_mtok=0.25,
            cached_input_per_mtok=0.03,
            output_per_mtok=1.25,
        ),
    ),
    # ----- Google --------------------------------------------------------
    (
        "gemini-2.5-pro",
        ModelPricing(
            name="gemini-2.5-pro",
            input_per_mtok=1.25,
            cached_input_per_mtok=0.31,
            output_per_mtok=10.00,
        ),
    ),
    (
        "gemini-2.5-flash",
        ModelPricing(
            name="gemini-2.5-flash",
            input_per_mtok=0.30,
            cached_input_per_mtok=0.075,
            output_per_mtok=2.50,
        ),
    ),
    (
        "gemini-2.0-flash",
        ModelPricing(
            name="gemini-2.0-flash",
            input_per_mtok=0.10,
            cached_input_per_mtok=0.025,
            output_per_mtok=0.40,
        ),
    ),
    (
        "gemini-1.5-pro",
        ModelPricing(
            name="gemini-1.5-pro",
            input_per_mtok=1.25,
            cached_input_per_mtok=0.31,
            output_per_mtok=5.00,
        ),
    ),
    (
        "gemini-1.5-flash",
        ModelPricing(
            name="gemini-1.5-flash",
            input_per_mtok=0.075,
            cached_input_per_mtok=0.01875,
            output_per_mtok=0.30,
        ),
    ),
    # ----- Mistral -------------------------------------------------------
    (
        "mistral-large",
        ModelPricing(
            name="mistral-large",
            input_per_mtok=2.00,
            cached_input_per_mtok=0.50,
            output_per_mtok=6.00,
        ),
    ),
    (
        "mistral-small",
        ModelPricing(
            name="mistral-small",
            input_per_mtok=0.20,
            cached_input_per_mtok=0.05,
            output_per_mtok=0.60,
        ),
    ),
    (
        "codestral",
        ModelPricing(
            name="codestral",
            input_per_mtok=0.30,
            cached_input_per_mtok=0.075,
            output_per_mtok=0.90,
        ),
    ),
    # ----- DeepSeek ------------------------------------------------------
    (
        "deepseek-chat",
        ModelPricing(
            name="deepseek-chat",
            input_per_mtok=0.27,
            cached_input_per_mtok=0.07,
            output_per_mtok=1.10,
        ),
    ),
    (
        "deepseek-reasoner",
        ModelPricing(
            name="deepseek-reasoner",
            input_per_mtok=0.55,
            cached_input_per_mtok=0.14,
            output_per_mtok=2.19,
            reasoning_per_mtok=2.19,
        ),
    ),
    # ----- xAI -----------------------------------------------------------
    (
        "grok-3-mini",
        ModelPricing(
            name="grok-3-mini",
            input_per_mtok=0.30,
            cached_input_per_mtok=0.075,
            output_per_mtok=0.50,
        ),
    ),
    (
        "grok-3",
        ModelPricing(
            name="grok-3",
            input_per_mtok=3.00,
            cached_input_per_mtok=0.75,
            output_per_mtok=15.00,
        ),
    ),
    # ----- Local / placeholder -------------------------------------------
    # Local models have no API cost. We list a few common
    # names with zero pricing so the UI shows "$0.00"
    # rather than "—" (which would otherwise suggest we
    # don't know what model is being used).
    (
        "llama",
        ModelPricing(name="llama"),
    ),
    (
        "qwen",
        ModelPricing(name="qwen"),
    ),
    (
        "mistral",
        ModelPricing(name="mistral"),
    ),
    (
        "gemma",
        ModelPricing(name="gemma"),
    ),
    (
        "phi",
        ModelPricing(name="phi"),
    ),
    (
        "local",
        ModelPricing(name="local"),
    ),
)


# === Lookup ===============================================================


def pricing_for(model: str | None) -> Optional[ModelPricing]:
    """Look up the pricing for a model by name.

    Substring match against the configured model name,
    case-insensitive; first hit wins. Returns ``None`` for
    unknown models so the UI can render "—" instead of
    a misleading zero. User overrides via
    ``ANDURIL_PRICING_OVERRIDES`` are merged in.
    """
    if not model:
        return None
    needle = model.lower()
    overrides = _load_overrides()
    # User-supplied overrides take precedence over the
    # built-in table.
    for pattern, custom in overrides.items():
        if pattern in needle:
            return custom
    for pattern, p in _PRICING_TABLE:
        if pattern in needle:
            return p
    return None


def _load_overrides() -> dict[str, ModelPricing]:
    """Parse ``ANDURIL_PRICING_OVERRIDES`` (JSON) into ModelPricing.

    Format: ``{"gpt-4o": {"input": 5.0, "output": 15.0}, ...}``.
    Unknown fields default to 0; unknown model keys are
    ignored. An invalid JSON value is silently dropped (the
    user gets a default-cost experience rather than a
    crash).
    """
    raw = os.environ.get("ANDURIL_PRICING_OVERRIDES", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    out: dict[str, ModelPricing] = {}
    if not isinstance(data, dict):
        return {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        out[str(key).lower()] = ModelPricing(
            name=str(key),
            input_per_mtok=float(val.get("input", 0.0) or 0.0),
            cached_input_per_mtok=float(val.get("cached_input", val.get("input", 0.0)) or 0.0),
            output_per_mtok=float(val.get("output", 0.0) or 0.0),
            reasoning_per_mtok=float(val.get("reasoning", 0.0) or 0.0),
        )
    return out


# === Formatting ==========================================================


def fmt_cost(amount: float) -> str:
    """Render a USD cost compactly: ``$0.0012``, ``$0.12``, ``$12.34``.

    For amounts below a tenth of a cent we show ``< $0.0001``;
    the user knows the model cost something but the number
    is meaningless. Two-decimal precision up to $1000; one
    decimal + thousands separator for $1K-$1M; ``$1.2M``
    style for anything bigger.
    """
    if amount < 0.0001:
        return "< $0.0001"
    if amount < 1.0:
        # 4 decimals for sub-dollar amounts so a $0.0001 call
        # shows as $0.0001 (not $0.00).
        return f"${amount:.4f}"
    if amount < 1_000.0:
        # Cents-precision: $12.34, $123.45, $999.99.
        return f"${amount:.2f}"
    if amount < 1_000_000:
        # No decimals + thousands separator: $1,234, $12,345.
        return f"${amount:,.0f}"
    return f"${amount / 1_000_000:.1f}M"


__all__ = [
    "ModelPricing",
    "fmt_cost",
    "pricing_for",
]
