"""Automatic context-compression: model context windows and threshold checks.

The model has a finite context window. Once the conversation history
plus system prompt plus tool schemas plus the new user message exceed
that window, the model starts to silently truncate (some backends) or
return an error (others). Both are bad — the model either forgets
what the user told it two turns ago, or the turn fails outright.

The fix is to call :func:`anduril.agent.compress` *before* the window
is exceeded, summarising the oldest turns into a single condensed
user turn. ``compress`` is non-trivial (it costs a model call of its
own), so we don't want to run it on every turn — only when the
estimated prompt size crosses a configurable threshold (default 80%
of the model's context window).

This module owns the data (model → context window) and the
arithmetic (chars → tokens → "should we compress?"). The agent's
run loop reads it once per turn and acts on the verdict.
"""

from __future__ import annotations

import json
from typing import Any

from anduril.env import _env_bool, _env_float


# === Tunables =============================================================

#: Fraction of the model's context window at which we trigger auto-
#: compression. 0.8 means "compress when the prompt reaches 80% of
#: the window". Leave headroom for the model's response (max_tokens),
#: the tool schemas, and the next user message.
#:
#: Set ``ANDURIL_CONTEXT_FRACTION=0.5`` for an aggressive policy,
#: ``ANDURIL_CONTEXT_FRACTION=0.95`` for a permissive one.
DEFAULT_CONTEXT_FRACTION: float = _env_float("ANDURIL_CONTEXT_FRACTION", 0.8)

#: Master switch. ``ANDURIL_AUTO_COMPRESS=0`` disables the trigger
#: entirely (manual ``/compress`` still works).
DEFAULT_AUTO_COMPRESS: bool = _env_bool("ANDURIL_AUTO_COMPRESS", True)

#: Conservative chars-to-tokens ratio for the local estimator. OpenAI's
#: cl100k_base averages ~3.5 chars/token for English, ~3 for code,
#: ~2 for CJK. We pick 3.0 as a *low-ball* (more tokens per char →
#: we trip the threshold a bit earlier, which is safer than
#: under-counting and missing the window). 3.0 is close to the code
#: ratio and errs on the side of caution.
CHARS_PER_TOKEN: float = 3.0

#: Per-message envelope cost. Chat templates typically add a few
#: tokens of role/format markers per message (e.g.
#: ``<|im_start|>role\\n<|im_end|>\\n`` for ChatML). 4 is a
#: conservative estimate; the real value varies by backend.
ENVELOPE_TOKENS_PER_MESSAGE: int = 4

#: Minimum number of body turns (after the system prompt) we require
#: before auto-compress is allowed to run. With fewer than this many
#: turns there's nothing useful to summarise — we'd just be
#: spending a model call to compress a 2-message conversation.
MIN_BODY_TURNS_TO_COMPRESS: int = 4

#: Default context window for unknown models. Conservative enough
#: to cover the most common local models (8K-32K) without being so
#: small that the trigger fires prematurely on long single
#: documents. 32K is a sensible middle for "I have no idea what
#: this is, be safe".
FALLBACK_CONTEXT_WINDOW: int = 32_768


# === Model → context window registry =====================================

#: Known model → context window mapping. Substring match against
#: the configured model name (case-insensitive). The first match
#: wins, so list the more specific patterns first.
#:
#: Numbers are in *tokens*, not chars. Sources: each vendor's docs
#: as of early 2026. Update as windows change.
MODEL_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    # OpenAI
    ("o3", 200_000),
    ("o4-mini", 200_000),
    ("o1", 200_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("gpt-3.5-turbo", 16_385),
    # Anthropic
    ("claude-3-7", 200_000),
    ("claude-3-5", 200_000),
    ("claude-3-opus", 200_000),
    ("claude-3-sonnet", 200_000),
    ("claude-3-haiku", 200_000),
    # Mistral
    ("mistral-large", 128_000),
    ("mistral-small", 32_000),
    ("mixtral", 32_000),
    # Google
    ("gemini-1.5-pro", 2_000_000),
    ("gemini-1.5-flash", 1_000_000),
    ("gemini-2", 1_000_000),
    # Qwen
    ("qwen2.5", 128_000),
    ("qwen2", 32_000),
    ("qwen", 32_000),
    # Llama
    ("llama-3.1", 128_000),
    ("llama-3", 8_192),
    ("llama-2", 4_096),
    # DeepSeek
    ("deepseek", 64_000),
    # Generic
    ("128k", 128_000),
    ("200k", 200_000),
    ("32k", 32_000),
    ("16k", 16_000),
    ("8k", 8_192),
    ("4k", 4_096),
)


def context_window_for(model: str | None) -> int:
    """Return the context window (in tokens) for ``model``.

    The match is a case-insensitive substring scan over
    :data:`MODEL_CONTEXT_WINDOWS`. The first hit wins. Unknown
    models fall back to :data:`FALLBACK_CONTEXT_WINDOW`.
    """
    if not model:
        return FALLBACK_CONTEXT_WINDOW
    needle = model.lower()
    for pattern, window in MODEL_CONTEXT_WINDOWS:
        if pattern in needle:
            return window
    return FALLBACK_CONTEXT_WINDOW


# === Estimator ============================================================


def _count_chars(value: Any) -> int:
    """Recursively count the "text-equivalent" character length of a value.

    Strings contribute their length. Lists/dicts are walked
    recursively (OpenAI multimodal content is a list of parts).
    Tool-call arguments contribute the JSON-serialised length of
    name + arguments. Everything else contributes 0 (we can't
    estimate image bytes, but a flat per-image bump is added by
    the caller).
    """
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_count_chars(v) for v in value)
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            return 0  # caller adds a flat per-image bump
        return sum(_count_chars(v) for v in value.values())
    return 0


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
    tool_schemas: list[dict[str, Any]] | None = None,
    image_count: int = 0,
    image_tokens: int = 1000,
) -> int:
    """Estimate the number of tokens a request would consume.

    Walks the message list and counts characters, then converts to
    tokens using :data:`CHARS_PER_TOKEN`. Adds a per-message
    envelope. Adds a flat per-image bump (we can't tell from
    message content alone how many tokens an image will cost —
    OpenAI's published numbers vary by detail level; 1000 is a
    safe middle for unanalysed ``detail: auto`` uploads).

    The agent's per-turn usage (from the API's ``usage`` chunk)
    replaces this estimate at end of turn; this function is
    useful for the *pre-flight* size check that decides whether
    to compress, before we've sent the request.
    """
    chars = len(system or "")
    schemas = tool_schemas or []
    if schemas:
        try:
            chars += len(json.dumps(schemas))
        except Exception:
            # Serialisation failure is rare; the worst case is
            # under-counting by a few hundred tokens, which
            # pushes the threshold a little later. Acceptable.
            chars += 200 * len(schemas)
    msg_count = 0
    for m in messages:
        if m.get("role") == "system":
            # Already counted from system= above.
            continue
        msg_count += 1
        content = m.get("content")
        chars += _count_chars(content)
        for tc in (m.get("tool_calls") or []):
            fn = (tc or {}).get("function") or {}
            chars += len(fn.get("name", "") or "")
            args = fn.get("arguments", "") or ""
            if not isinstance(args, str):
                try:
                    args = json.dumps(args)
                except Exception:
                    args = ""
            chars += len(args)
    content_tokens = int(round(chars / max(0.1, CHARS_PER_TOKEN)))
    envelope = ENVELOPE_TOKENS_PER_MESSAGE * (msg_count + 1)  # +1 for next user msg
    image_cost = image_count * max(0, image_tokens)
    return max(1, content_tokens + envelope + image_cost)


def should_auto_compress(
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    system: str = "",
    tool_schemas: list[dict[str, Any]] | None = None,
    fraction: float = DEFAULT_CONTEXT_FRACTION,
    enabled: bool = DEFAULT_AUTO_COMPRESS,
) -> tuple[bool, int, int, int]:
    """Decide whether the agent should call :func:`compress` now.

    Returns a 4-tuple ``(should_compress, est_tokens, window, threshold)``
    so callers can log the reasoning ("prompt at 28K of 32K window,
    threshold 25.6K — not yet"). The 4-tuple is also what the
    TUI shows in its auto-compress log line.

    The decision is:

    * Disabled? False.
    * Fewer than :data:`MIN_BODY_TURNS_TO_COMPRESS` non-system
      turns? False (nothing useful to summarise yet).
    * Estimated tokens below ``window * fraction``? False.
    * Otherwise: True.
    """
    if not enabled:
        return (False, 0, 0, 0)
    body_n = sum(1 for m in messages if m.get("role") != "system")
    if body_n < MIN_BODY_TURNS_TO_COMPRESS:
        return (False, 0, 0, 0)
    image_count = sum(
        1 for m in messages
        if isinstance(m.get("content"), list)
        and any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in m["content"]
        )
    )
    est = estimate_prompt_tokens(
        messages, system=system, tool_schemas=tool_schemas,
        image_count=image_count,
    )
    window = context_window_for(model)
    threshold = int(window * max(0.0, min(1.0, fraction)))
    return (est >= threshold, est, window, threshold)


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_AUTO_COMPRESS",
    "DEFAULT_CONTEXT_FRACTION",
    "FALLBACK_CONTEXT_WINDOW",
    "MIN_BODY_TURNS_TO_COMPRESS",
    "MODEL_CONTEXT_WINDOWS",
    "context_window_for",
    "estimate_prompt_tokens",
    "should_auto_compress",
]
