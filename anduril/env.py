"""Environment helpers and ANSI codes.

Pure stdlib. Imported by every other anduril submodule so the helpers
are kept in one place.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a bool-ish env var. ``1``, ``true``, ``yes``, ``on`` ⇒ True.

    Used for feature flags like ``ANDURIL_AUTO_COMPRESS``. Case
    insensitive; whitespace tolerated. Anything not in the truthy
    set returns ``default`` (we deliberately don't treat
    ``"false"`` as falsy when ``default`` is True, so an
    incorrectly set env var doesn't silently disable a feature).
    """
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ANSI codes used in the non-TUI fallback path (and for any printf-style
# messages that sneak into stderr).
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"
