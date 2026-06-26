"""Curses TUI: multi-line editor, log viewer, command dispatch, and input loop.

The TUI is the biggest single file. It owns the editor, the scrollable
log of past events, the model-streaming callbacks, and the per-key
input loop. It does NOT own the agent — that's :class:`anduril_agent.Agent`.
It does NOT own session persistence — that's :mod:`anduril_sessions`.

Performance
-----------

The biggest cost in the TUI is per-token re-rendering during streaming.
The hot path is wrapping the entire ``log`` list to terminal width on
every render. We cache the wrapped form per log entry, keyed on
``(entry, max_w)``. When the terminal resizes, the cache is invalidated
and rebuilt lazily on the next render.
"""

from __future__ import annotations

import base64
import curses
import json
import os
import pathlib
import re
import signal
import sys
import time
from dataclasses import dataclass
import textwrap
from typing import Callable

from rich.markdown import Markdown as _RichMarkdown

from anduril.env import _env_int, _env_str
from anduril.agent import (
    COMPRESS_KEEP,
    Agent,
    compress,
)
from anduril.highlight import (
    highlight_code as _highlight_code,
)
from anduril.metrics import _abbr, _Metrics
from anduril.pricing import fmt_cost as _fmt_cost, pricing_for as _pricing_for
from anduril.tools import RISK_RANK
from anduril.sessions import (
    _load_session,
    _new_session_id,
    _safe_title,
    _write_session,
)
from anduril.files import (
    list_files as _list_files,
    fuzzy_match as _fuzzy_match,
    expand_mentions as _expand_mentions,
    is_image as _is_image,
    find_active_mention as _find_active_mention,
    mention_query as _mention_query,
    save_pasted_image as _save_pasted_image,
    read_clipboard_image as _read_clipboard_image,
    clipboard_tools_status as _clipboard_tools_status,
    IMAGE_EXTS as _IMAGE_EXTS,
    MAX_IMAGE_BYTES as _MAX_IMAGE_BYTES,
    MAX_TEXT_CHARS as _MAX_TEXT_CHARS,
)


# Max wrapped lines from a single tool result shown in the log. The model
# always receives the full (sanitized) output; this only trims the on-screen
# view so a huge grep result doesn't push everything else off-screen.
MAX_TOOL_LINES = _env_int("ANDURIL_MAX_TOOL_LINES", 30)
# Max height (rows) the input editor is allowed to take from the screen.
MAX_EDITOR_LINES = _env_int("ANDURIL_MAX_EDITOR_LINES", 7)
MIN_EDITOR_LINES = 3

# How many rows the ``@``-file menu is allowed to occupy. The fuzzy
# matcher returns up to ``ANDURIL_FILE_MENU_LIMIT`` candidates; the
# rest are reachable by typing more characters, not by scrolling
# the menu (we keep the picker single-screen on purpose).
FILE_MENU_ROWS = _env_int("ANDURIL_FILE_MENU_ROWS", 10)
FILE_MENU_CANDIDATE_LIMIT = _env_int("ANDURIL_FILE_MENU_LIMIT", 200)
# Maximum files we'll walk in a single scan. The fuzzy picker is a
# search affordance, not a full file browser — capping the working
# set keeps the TUI snappy on large repos and prevents a runaway
# scan (e.g. ``/``) from blocking the input loop.
FILE_MENU_MAX_FILES = _env_int("ANDURIL_FILE_MENU_MAX_FILES", 2000)
# Rough per-image token cost used by the prompt-size estimator.
# OpenAI's published numbers vary by detail level (low: 85, high:
# 170 × tile count); 1000 is a defensible middle for an unanalysed
# ``detail: auto`` upload. The API's usage chunk overwrites this
# estimate at end of turn.
APPROX_IMAGE_TOKENS = _env_int("ANDURIL_IMAGE_TOKENS", 1000)


# --- multi-line input editor -----------------------------------------------


class _Editor:
    """Multi-line chatbox buffer.

    A list-of-lines + (row, col) cursor. No length cap, so long pastes
    survive intact. Bracketed paste (\\x1b[200~ ... \\x1b[201~) is handled
    in the input loop; the editor itself just gets insert_text() calls.

    Submit semantics:
      • Enter on a non-empty buffer → submit, push to history, reset.
      • Enter on an empty buffer → no-op.
      • Alt+Enter (\\x1b\\r) or Ctrl+J (\\x0a) → insert newline at cursor.

    History navigation:
      • Up at first line → recall older history (minion convention).
      • Down at last line → advance to newer history / restore draft.
    """

    def __init__(self, history: list[str] | None = None) -> None:
        self.buf: list[str] = [""]
        self.row = 0
        self.col = 0
        self.history: list[str] = list(history or [])
        self.h_idx = len(self.history)
        self._saved_draft: list[str] | None = None

    # --- buffer mutation ----------------------------------------------------

    def insert_char(self, c: str) -> None:
        line = self.buf[self.row]
        self.buf[self.row] = line[: self.col] + c + line[self.col :]
        self.col += 1

    def insert_text(self, s: str) -> None:
        """Insert a (possibly multi-line) string at the cursor."""
        if not s:
            return
        parts = s.split("\n")
        cur = self.buf[self.row]
        tail = cur[self.col :]
        if len(parts) == 1:
            self.buf[self.row] = cur[: self.col] + parts[0] + tail
            self.col += len(parts[0])
            return
        self.buf[self.row] = cur[: self.col] + parts[0]
        new_lines = list(parts[1:-1]) + [parts[-1] + tail]
        self.buf[self.row + 1 : self.row + 1] = new_lines
        self.row += len(new_lines)
        self.col = len(parts[-1])

    def newline(self) -> None:
        cur = self.buf[self.row]
        head, tail = cur[: self.col], cur[self.col :]
        self.buf[self.row] = head
        self.buf.insert(self.row + 1, tail)
        self.row += 1
        self.col = 0

    def backspace(self) -> None:
        if self.col > 0:
            line = self.buf[self.row]
            self.buf[self.row] = line[: self.col - 1] + line[self.col :]
            self.col -= 1
        elif self.row > 0:
            prev = self.buf[self.row - 1]
            self.col = len(prev)
            self.buf[self.row - 1] = prev + self.buf[self.row]
            del self.buf[self.row]
            self.row -= 1

    def replace_range(self, start: int, end: int, text: str,
                      row: int | None = None) -> None:
        """Replace ``buf[row][start:end]`` with ``text``.

        Used by the ``@``-file menu to overwrite a half-typed mention
        (``@src`` → ``@src/main.py``) without disturbing the rest of
        the line. The cursor lands at ``start + len(text)`` so the
        user can keep typing right after the inserted path.

        Multi-line ``text`` (containing ``\\n``) is rejected: the
        file menu always inserts single-line paths. The caller is
        expected to pre-validate.
        """
        if "\n" in text:
            raise ValueError("replace_range only supports single-line text")
        r = self.row if row is None else row
        line = self.buf[r]
        # Clamp to the line length — a cursor past the end of the
        # line is treated as "at the end".
        start = max(0, min(start, len(line)))
        end = max(start, min(end, len(line)))
        self.buf[r] = line[:start] + text + line[end:]
        self.row = r
        self.col = start + len(text)

    def delete_forward(self) -> None:
        line = self.buf[self.row]
        if self.col < len(line):
            self.buf[self.row] = line[: self.col] + line[self.col + 1 :]
        elif self.row < len(self.buf) - 1:
            self.buf[self.row] = line + self.buf[self.row + 1]
            del self.buf[self.row + 1]

    def clear_line(self) -> None:
        self.buf[self.row] = ""
        self.col = 0

    # --- navigation ---------------------------------------------------------

    def move_left(self) -> None:
        if self.col > 0:
            self.col -= 1
        elif self.row > 0:
            self.row -= 1
            self.col = len(self.buf[self.row])

    def move_right(self) -> None:
        if self.col < len(self.buf[self.row]):
            self.col += 1
        elif self.row < len(self.buf) - 1:
            self.row += 1
            self.col = 0

    def move_home(self) -> None:
        self.col = 0

    def move_end(self) -> None:
        self.col = len(self.buf[self.row])

    def move_up(self) -> bool:
        """Move up one line; at the top, recall older history. Returns True
        if history was triggered (so the TUI can flash a hint)."""
        if self.row > 0:
            self.row -= 1
            self.col = min(self.col, len(self.buf[self.row]))
            return False
        if self.history and self.h_idx > 0:
            self.h_idx -= 1
            self._load_history()
            return True
        return False

    def move_down(self) -> bool:
        if self.row < len(self.buf) - 1:
            self.row += 1
            self.col = min(self.col, len(self.buf[self.row]))
            return False
        if self.h_idx < len(self.history):
            self.h_idx += 1
            self._load_history()
            return True
        return False

    def _load_history(self) -> None:
        if self.h_idx == len(self.history):
            if self._saved_draft is not None:
                self.buf = list(self._saved_draft)
                self._saved_draft = None
            else:
                self.buf = [""]
        else:
            if self._saved_draft is None:
                self._saved_draft = list(self.buf)
            self.buf = self.history[self.h_idx].split("\n") or [""]
        self.row = len(self.buf) - 1
        self.col = len(self.buf[-1])

    # --- submit -------------------------------------------------------------

    def submit(self) -> str:
        text = "\n".join(self.buf)
        if text.strip():
            if not self.history or self.history[-1] != text:
                self.history.append(text)
            self.buf = [""]
            self.row = 0
            self.col = 0
            self.h_idx = len(self.history)
            self._saved_draft = None
        return text

    def is_empty(self) -> bool:
        return not any(self.buf)

    def char_count(self) -> int:
        return sum(len(line) for line in self.buf)


# --- TUI state + command dispatch -----------------------------------------


_INDENT = {
    "user": "› ",
    "user_attachment": "  + ",
    "assistant": "  ",
    "reasoning": "  · ",
    "tool": "    ↳ ",
    "tool_call": "    ↪ ",
    "stats": "  └ ",
    "note": "! ",
    "blank": "",
}


def _short_args(args: dict, n: int = 60) -> str:
    """Render an args dict as a short, single-line, terminal-safe string."""
    text = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= n:
        return text
    return text[:n - 1] + "…"


def _common_prefix(strs: list[str]) -> str:
    """Longest common prefix of a non-empty list of strings."""
    if not strs:
        return ""
    if len(strs) == 1:
        return strs[0]
    prefix = strs[0]
    for s in strs[1:]:
        i = 0
        limit = min(len(prefix), len(s))
        while i < limit and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


# --- Syntax highlighting --------------------------------------------------

#: Map a token category (either a pygments Token.* name or a regex
#: highlighter category like "keyword", "string") to a curses
#: attribute. The highlighter's "default" attr is the entry-point attr
#: passed in — here we use ``0`` to mean "uncoloured" and reserve
#: any positive int as "matched". The highlighter walks the token
#: tree, so a single "Keyword" entry covers Keyword.* automatically.
#:
#: Categories are added to the mapping in priority order: the first
#: match wins, so put the more specific categories first.
_HIGHLIGHT_TOKEN_MAP: dict[str, int] = {
    # Pygments-style names (used when pygments is installed).
    "Comment": 1,
    "Comment.Single": 1,
    "Comment.Multiline": 1,
    "String": 2,
    "String.Doc": 2,
    "String.Escape": 2,
    "Number": 3,
    "Number.Integer": 3,
    "Number.Float": 3,
    "Number.Bin": 3,
    "Number.Hex": 3,
    "Number.Oct": 3,
    "Keyword": 4,
    "Keyword.Constant": 4,
    "Keyword.Declaration": 4,
    "Keyword.Namespace": 4,
    "Keyword.Pseudo": 4,
    "Keyword.Reserved": 4,
    "Keyword.Type": 4,
    "Name.Builtin": 5,
    "Name.Builtin.Pseudo": 5,
    "Operator": 6,
    "Operator.Word": 6,
    "Punctuation": 7,
    # Regex-fallback category names (strings the regex highlighter emits).
    "comment": 1,
    "string": 2,
    "number": 3,
    "keyword": 4,
    "builtin": 5,
    "type": 5,
    "operator": 6,
    "variable": 8,
    "literal": 9,
    "tag": 10,
    "section": 11,
    "diff_meta": 12,
    "diff_added": 13,
    "diff_removed": 14,
}

#: Sentinel attr returned by ``_token_to_attr`` for tokens the
#: mapping doesn't know about. ``0`` would be ambiguous with the
#: "default" attr; we use ``-1`` so the highlighter can tell the
#: difference between "this token is uncoloured" and "we don't have
#: a colour for this token". The highlighter falls back to the
#: default attr when ``-1`` is returned.
_HL_UNMAPPED = -1


def _token_to_attr(token_name: str) -> int:
    """Map a token name to a curses attribute. Returns ``_HL_UNMAPPED``
    for unmapped tokens. The highlighter resolves ``_HL_UNMAPPED`` to
    the default attr of the surrounding log entry, so this can be
    permissive without losing info.
    """
    if not token_name:
        return _HL_UNMAPPED
    # Pygments tokens look like "Token.Keyword.Namespace". We strip
    # the "Token" prefix and walk from the most specific to the
    # most general, so a single "Keyword" entry covers Keyword.*.
    parts = token_name.split(".")
    if parts[0] == "Token":
        parts = parts[1:]
    for i in range(len(parts), 0, -1):
        key = ".".join(parts[:i])
        if key in _HIGHLIGHT_TOKEN_MAP:
            return _HIGHLIGHT_TOKEN_MAP[key]
    return _HL_UNMAPPED


#: Match a fenced code block opener or closer. Both ``\`\`\`python``
#: and ``\`\`\``` (no language) are accepted; the closer is bare.
_FENCE_RE = re.compile(
    r"^\s*(?P<close>```\s*$)|^\s*```(?P<lang>[A-Za-z0-9_+\-#]*)\s*$"
)


def _split_code_fences(text: str) -> list[tuple[str, str, str]]:
    """Split ``text`` into ``[(kind, text, lang), ...]`` segments.

    ``kind`` is one of:

    * ``"text"``  — prose (rendered with the entry's default attr)
    * ``"code"``  — the inner body of a fenced code block (the
      fence delimiters themselves are stripped; the body is
      passed to the highlighter verbatim). The trailing element
      ``lang`` carries the language tag from the opening fence.

    The state machine walks the input line by line. The opening
    fence carries the language tag (three backticks + ``python``);
    the closing fence is bare (three backticks). Unterminated
    fences (an opening fence with no closer before EOF) are
    treated as plain text — the highlighter isn't worth the
    surprise of a runaway span.
    """
    out: list[tuple[str, str, str]] = []
    lines = text.split("\n")
    in_code = False
    current_lang = ""
    buf: list[str] = []
    text_buf: list[str] = []

    def _flush_text() -> None:
        if text_buf:
            out.append(("text", "\n".join(text_buf), ""))
            text_buf.clear()

    def _flush_code() -> None:
        if buf:
            out.append(("code", "\n".join(buf), current_lang))
            buf.clear()

    for raw in lines:
        m = _FENCE_RE.match(raw)
        if m is not None and m.group("close"):
            # Closing fence — flush the code buffer.
            if in_code:
                _flush_code()
                in_code = False
                current_lang = ""
            else:
                # Bare ``` not inside a block; treat as text.
                text_buf.append(raw)
            continue
        if m is not None and not in_code:
            # Opening fence — flush any pending text.
            _flush_text()
            in_code = True
            current_lang = m.group("lang") or ""
            continue
        if in_code:
            buf.append(raw)
        else:
            text_buf.append(raw)
    if in_code:
        # Unterminated fence. Don't drop the partial code — just
        # emit it as text so the user sees the line.
        text_buf.append("```" + current_lang)
        text_buf.extend(buf)
    _flush_text()
    return out


def _line_to_spans(
    wrapped_line: str,
    char_attrs: list[tuple[str, int]],
    indent: str,
) -> list[tuple[str, int]]:
    """Slice the per-char attr list to match a wrapped line.

    ``wrapped_line`` is the result of ``textwrap.wrap`` for a
    single logical line — it includes the leading indent. The
    indent is rendered in its own span (attr = the entry's
    default, resolved by the caller); the body is sliced from
    ``char_attrs`` (one ``(char, attr)`` per source char).

    Adjacent same-attr chars in the body are merged back into a
    single span so the renderer's per-span overhead is small
    even for long, mostly-default lines. A line with a single
    attr comes out as a single span — the fast path for
    non-highlighted text.
    """
    # Strip the indent off the front of the wrapped line; the
    # remainder corresponds 1:1 to ``char_attrs``.
    if indent and wrapped_line.startswith(indent):
        body = wrapped_line[len(indent):]
    else:
        body = wrapped_line
        indent = ""
    body_chars = char_attrs[:len(body)]
    # Consume the chars we just used. The caller iterates
    # left-to-right so the next call resumes at the right
    # place.
    del char_attrs[:len(body)]
    # Build spans: the indent (own span), then merged body.
    # The indent has no "own" attr here; the caller passes
    # ``_HL_UNMAPPED`` and we resolve it to the body's first
    # attr at the end. This way the indent renders in the
    # same colour as the body, which is what the eye expects
    # when a highlighted line is wrapped.
    spans: list[tuple[str, int]] = []
    if indent:
        spans.append((indent, _HL_UNMAPPED))
    for ch, attr in body_chars:
        if spans and spans[-1][1] == attr:
            spans[-1] = (spans[-1][0] + ch, attr)
        else:
            spans.append((ch, attr))
    # Resolve the indent's attr. If the body is non-empty, use
    # its first non-unmapped attr; otherwise drop the indent
    # span (no body to inherit from).
    if indent and spans:
        body_attr = _HL_UNMAPPED
        for text, attr in spans[1:]:
            if attr != _HL_UNMAPPED:
                body_attr = attr
                break
        if body_attr == _HL_UNMAPPED:
            # Body is empty or all-unmapped; just drop the
            # indent span (it would render nothing useful).
            spans = spans[1:]
        else:
            spans[0] = (indent, body_attr)
    return [(t, a) for t, a in spans if t]


def _latex_to_unicode(text: str) -> str:
    """Convert LaTeX math expressions (``$...$`` and ``$$...$$``) to Unicode.

    Falls back to the original text if the ``pylatexenc`` library is not
    available or if conversion fails for a specific expression.
    """
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        _converter = LatexNodes2Text()
    except ImportError:
        return text

    def _convert(m: re.Match) -> str:
        try:
            return _converter.latex_to_text(m.group(1))
        except Exception:
            return m.group(0)

    # Display math $$...$$ (block-level, may span multiple lines).
    text = re.sub(r'\$\$(.*?)\$\$', _convert, text, flags=re.DOTALL)
    # Inline math $...$ — must contain at least one letter (avoids matching
    # prices like $10.99) and not be preceded by a backslash.
    text = re.sub(r'(?<!\$)\$(?=[^$]*[A-Za-z])([^$]+?)\$(?!\$)', _convert, text)
    return text


def _normalize_approval(s: str) -> tuple[str | None, str | None]:
    """Parse an approval-level string into a (kind, level) pair.

    Returns ``(None, None)`` on unknown input. The shape mirrors what
    the TUI's ``approval_level`` state expects:

      * ``("yolo", None)``            — skip every prompt
      * ``("prompt_all", None)``      — prompt for every dangerous tool
      * ``("threshold", "low")``      — prompt for low-risk + above
      * ``("threshold", "medium")``
      * ``("threshold", "high")``
    """
    s = (s or "").strip().lower()
    if s in ("yolo",):
        return ("yolo", None)
    if s in ("all", "prompt", "none", "strict"):
        return ("prompt_all", None)
    if s in ("low", "lo"):
        return ("threshold", "low")
    if s in ("high", "hi"):
        return ("threshold", "high")
    if s in ("medium", "med", "mid"):
        return ("threshold", "medium")
    return (None, None)


# --- TUI entry point -------------------------------------------------------

# Forward declaration for type-checking. The real class is defined further
# down to keep the helpers above (which it uses) visible at the top.
class _TUIState:
    pass


# --- TUI entry point -------------------------------------------------------


def tui(agent: Agent) -> None:
    """Main TUI loop. Blocks until the user quits."""

    def main(stdscr: "curses._CursesWindow") -> None:
        curses.use_default_colors()
        # Color pair ids — paired with -1 (default fg/bg) so the TUI works
        # on light terminals too. (No actual colors are forced.)
        for i in range(8):
            try:
                curses.init_pair(i + 1, i, -1)
            except curses.error:
                pass
        # Additional color pairs for markdown rendering.
        try:
            curses.init_pair(9, curses.COLOR_YELLOW, -1)   # headings
        except curses.error:
            pass
        try:
            curses.init_pair(10, curses.COLOR_CYAN, -1)    # inline code
        except curses.error:
            pass
        try:
            curses.init_pair(11, curses.COLOR_MAGENTA, -1) # links
        except curses.error:
            pass
        try:
            curses.init_pair(12, curses.COLOR_WHITE, -1)   # blockquote
        except curses.error:
            pass
        # Enable mouse reporting so the wheel can scroll the log. We ask
        # for button-press events (which is what wheel scrolls come through
        # as on most terminals: button 4 = up, button 5 = down). The
        # terminal needs to support SGR mouse mode (1006) for modern
        # terminals; older terminals use the legacy 1003 mode. Either
        # way, curses translates both into BUTTON4/BUTTON5_PRESSED.
        try:
            curses.mousemask(
                curses.BUTTON1_PRESSED | curses.BUTTON4_PRESSED
                | curses.BUTTON5_PRESSED | curses.BUTTON2_PRESSED
            )
        except Exception:
            pass
        A_DIM = curses.color_pair(0) | curses.A_DIM
        A_BOLD = curses.A_BOLD
        A_GREEN = curses.color_pair(2) | curses.A_BOLD
        A_YELLOW = curses.color_pair(3) | curses.A_BOLD
        A_RED = curses.color_pair(1) | curses.A_BOLD
        A_MAGENTA = curses.color_pair(5) | curses.A_BOLD
        A_CYAN = curses.color_pair(4) | curses.A_BOLD
        A_NORMAL = curses.A_NORMAL
        # Per-level heading attributes — no color-pair dependency so they
        # work on any terminal.  Higher levels (smaller number) = more
        # visual weight: H1–H4 bold, H5/H6 plain (the ### prefix alone
        # provides the depth cue).
        MD_HEADING = 0  # fallback (not used when per-level attrs are set)
        MD_H1 = curses.A_BOLD
        MD_H2 = curses.A_BOLD
        MD_H3 = curses.A_BOLD
        MD_H4 = curses.A_BOLD
        MD_H5 = 0
        MD_H6 = 0
        MD_CODE_INLINE = curses.color_pair(10)
        MD_LINK = curses.color_pair(11) | curses.A_UNDERLINE
        MD_BLOCKQUOTE = curses.color_pair(12) | curses.A_DIM

        # ----- build state ----------------------------------------------
        state = _TUIState(agent, stdscr, A_NORMAL, A_DIM, A_BOLD,
                          A_GREEN, A_YELLOW, A_RED, A_MAGENTA, A_CYAN,
                          MD_HEADING, MD_CODE_INLINE, MD_LINK, MD_BLOCKQUOTE,
                          MD_H1, MD_H2, MD_H3, MD_H4, MD_H5, MD_H6)
        state.bootstrap()

        # ----- main key loop --------------------------------------------
        # Enable bracketed paste mode so terminals send \x1b[200~ ... \x1b[201~
        # around pasted text. Without this, each newline in a paste is
        # indistinguishable from a user pressing Enter and submits the
        # buffer prematurely (the first line goes, the rest get submitted
        # one by one — leaving the user staring at a one-line editor).
        # Write the enable sequence via curses (more reliable than
        # sys.stdout once curses has taken over the terminal), then refresh.
        try:
            stdscr.addstr("\x1b[?2004h")
            stdscr.refresh()
        except Exception:
            try:
                sys.stdout.write("\x1b[?2004h")
                sys.stdout.flush()
            except Exception:
                pass

        # No paste-burst detector. Most terminals send \r (and many send
        # \n) for the Enter key, so we treat BOTH as submit. Newlines are
        # only inserted via Shift+Enter (Alt+Enter, \x1b\r) or via a
        # bracketed paste (\x1b[200~...\x1b[201~). If you want multi-line
        # text without bracketed paste, send Shift+Enter per line.
        #
        # SIGWINCH handler: when the terminal is resized, set a flag on
        # the state. render() will pick it up, call curses.resizeterm with
        # the new dimensions, and redraw at the new size. We avoid
        # calling curses from inside the signal handler itself (not
        # signal-safe); the flag is checked on the next render tick —
        # which happens per-token during streaming, so resize latency
        # during a model call is at most one chunk.
        old_winch = signal.getsignal(signal.SIGWINCH)
        try:
            signal.signal(
                signal.SIGWINCH,
                lambda *_: setattr(state, "resize_pending", True),
            )
        except (ValueError, OSError):
            # SIGWINCH isn't available on this platform (e.g. Windows).
            old_winch = None
        try:
            while True:
                state.render()
                try:
                    ch = stdscr.get_wch()
                except KeyboardInterrupt:
                    # Ctrl-C: clear the current input if non-empty, else quit.
                    if not state.editor.is_empty():
                        state.editor.buf = [""]
                        state.editor.row = 0
                        state.editor.col = 0
                        state.render()
                        continue
                    else:
                        break
                except curses.error:
                    # get_wch can be interrupted by a signal — most
                    # often SIGWINCH when the user resizes or splits a
                    # pane. Python surfaces that as "no input". Just
                    # retry: the next render() at the top of the
                    # loop will pick up any pending resize via the
                    # flag set by the SIGWINCH handler.
                    continue

                state._handle_key(ch)
        finally:
            if old_winch is not None:
                signal.signal(signal.SIGWINCH, old_winch)

        # On exit, disable bracketed paste + save the session.
        try:
            sys.stdout.write("\x1b[?2004l")
            sys.stdout.flush()
        except Exception:
            pass
        state.save_session()

    curses.wrapper(main)


# --- TUIState class -------------------------------------------------------


def _make_confirm_callback(agent: Agent, stdscr, level: str) -> Callable[[str, dict], bool]:
    """Build a callback that gates ``dangerous=True`` tools behind a prompt.

    The ``level`` is one of:

    * ``"yolo"``   — never prompt, run everything.
    * ``"all"``    — prompt for every dangerous tool (the default; the
                     classic behaviour). Note: this also prompts for
                     tools that are not dangerous — it's a paranoid mode
                     useful for first-time use of an unknown tool.
    * ``"high"``   — prompt only for tools with ``risk="high"``.
    * ``"medium"`` — prompt for tools with ``risk="medium"`` or higher.
    * ``"low"``    — prompt for tools with ``risk="low"`` or higher
                     (i.e. effectively every dangerous tool).
    """
    # Map the level to a numeric threshold.
    #   * ``yolo``     -> never prompt; dangerous tools run silently.
    #   * ``all``      -> prompt for every tool (dangerous or not).
    #   * ``high``     -> prompt only for ``risk="high"`` tools.
    #   * ``medium``   -> prompt for ``risk >= "medium"``.
    #   * ``low``      -> prompt for any dangerous tool.
    # Tools with risk below the threshold run silently. ``threshold``
    # is the minimum RISK_RANK value that triggers a prompt for
    # dangerous tools; it is unused when ``level`` is "yolo" or "all".
    never_prompt = level == "yolo"
    prompt_for_safe = level in ("all",) or level not in ("yolo", "low", "medium", "high")
    if level in ("low", "medium", "high"):
        threshold: int = RISK_RANK[level]
    else:
        threshold = 0  # never used: never_prompt short-circuits or all-tools-match

    def _confirm(name: str, args: dict) -> bool:
        if never_prompt:
            return True
        tool = agent.tools.get(name)
        if tool is None:
            return True
        if not tool.dangerous:
            if not prompt_for_safe:
                return True
            return _confirm_key(
                stdscr, f"  Allow tool '{name}({_short_args(args)})'? [y/N/esc] "
            )
        if not prompt_for_safe and RISK_RANK.get(tool.risk, 1) < threshold:
            return True
        prompt = f"  Allow {tool.risk}-risk tool '{name}({_short_args(args)})'? [y/N/esc] "
        return _confirm_key(stdscr, prompt)
    return _confirm


class _TUIState:
    """All mutable state for the TUI main loop, in one place.

    Kept as a class (not a dataclass) so the methods (``push``,
    ``render``, ``run_agent_turn``) live next to the state they
    operate on. Cached per-entry log wrap lives here too.
    """

    # ----- construction ------------------------------------------------

    def __init__(self, agent, stdscr,
                 A_NORMAL, A_DIM, A_BOLD, A_GREEN, A_YELLOW, A_RED,
                 A_MAGENTA, A_CYAN,
                 MD_HEADING=0, MD_CODE_INLINE=0, MD_LINK=0, MD_BLOCKQUOTE=0,
                 MD_H1=0, MD_H2=0, MD_H3=0, MD_H4=0, MD_H5=0, MD_H6=0) -> None:
        self.agent = agent
        self.stdscr = stdscr
        self.A_NORMAL = A_NORMAL
        self.A_DIM = A_DIM
        self.A_BOLD = A_BOLD
        self.A_GREEN = A_GREEN
        self.A_YELLOW = A_YELLOW
        self.A_RED = A_RED
        self.A_MAGENTA = A_MAGENTA
        self.A_CYAN = A_CYAN
        self.MD_HEADING = MD_HEADING
        self.MD_CODE_INLINE = MD_CODE_INLINE
        self.MD_LINK = MD_LINK
        self.MD_BLOCKQUOTE = MD_BLOCKQUOTE
        self.MD_H1 = MD_H1
        self.MD_H2 = MD_H2
        self.MD_H3 = MD_H3
        self.MD_H4 = MD_H4
        self.MD_H5 = MD_H5
        self.MD_H6 = MD_H6

        # Session / goal
        self._goal: str | None = None

        # Session / metrics
        self.session_id = _new_session_id()
        self.metrics = _Metrics(self.session_id, model=agent.model)
        agent.set_metrics(self.metrics)
        self.messages: list = list(agent.messages)
        # Reload metrics from disk if this session already has a file.
        prior = _load_session(self.session_id)
        if prior:
            self.metrics.load(prior)
            if prior.get("goal"):
                self._goal = prior["goal"]
                self.agent.goal = prior["goal"]

        # Scrollable log: (kind, text, attr)
        # Scrollable log: (kind, text, attr)
        self.log: list[tuple[str, str, int]] = []
        self.scroll = 0
        self.editor = _Editor()
        self.approval_level = _init_approval_level()

        # Streaming accumulators.
        self.streaming_assistant: list[str] = []
        self.streaming_reasoning: list[str] = []
        self.streaming_tool_calls: dict[int, dict] = {}
        self.streaming_kind: str | None = None

        # Wrap cache (per-entry wrapped form, plus the width it was wrapped
        # at). The cache is invalidated on push() or terminal resize.
        self._log_wrapped: list[list[tuple[str, list[tuple[str, int]]]] | None] = []
        self._wrap_width: int = -1

        # Slash-command autocomplete menu. ``menu_selected`` is the
        # index into the filtered command list. The menu is "active"
        # iff the current line starts with "/" and the cursor is on
        # the first line — derived on demand in ``_menu_active()``.
        self.menu_selected: int = 0

        # ``@``-file picker menu. The menu is "active" iff the cursor
        # is currently inside an ``@-mention`` (see
        # :func:`anduril.files.find_active_mention`). The candidate
        # list is computed from ``file_menu_candidates`` + the
        # current query, then sorted by fuzzy score.
        self.file_menu_selected: int = 0
        # Cached file list. Keyed on (cwd, mtime-of-root) so a fresh
        # ``cd`` or a new file in the tree invalidates it. ``None``
        # means "not yet scanned" — the first menu activation pays
        # the cost, subsequent activations hit the cache.
        self.file_menu_candidates: list[str] | None = None
        self.file_menu_cwd: pathlib.Path = pathlib.Path.cwd()
        self.file_menu_cache_key: tuple | None = None
        # Last fuzzy-match result for the current query. Keyed on the
        # query string so re-rendering (or moving the cursor without
        # changing the query) is O(1).
        self.file_menu_last_query: str | None = None
        self.file_menu_last_matches: list[tuple[int, str]] = []

        # Short-reference attachments. When a pasted image (or a
        # future "attach" command) saves a file to disk, the
        # buffer shows a short ID like ``@image-1`` and the
        # actual path lives in this dict. On submit, the
        # expand_mentions function looks each ID up here and
        # attaches the real file. This keeps the editor line
        # short even when a paste produces a long auto-generated
        # filename like ``image-2026-06-24-145848-001.png``.
        self.attachments: dict[str, str] = {}
        # Counter for the next short ID. Reset to 0 when the
        # session is cleared (so a /clear doesn't continue the
        # numbering from the previous session).
        self._next_attachment_id: int = 1

        # Live-resize support: a SIGWINCH handler installed by tui() sets
        # this flag. The next render() picks it up, calls curses.resizeterm
        # with the new dimensions, and redraws at the new size.
        self.resize_pending: bool = False

        # Maps streaming tool-call id → log index, so we can update an
        # in-progress call in place as more deltas arrive. Reset on /clear.
        self._tool_call_log_idx: dict[str, int] = {}

        # Log length snapshot taken at the start of every user turn.
        # ``/undo`` truncates ``self.log`` back to this length so
        # the user sees the assistant turn (and any tool results)
        # vanish from the screen, not just from the message list.
        # Starts at ``len(self.log)`` (a no-op) so the first call
        # is safe before any turn has run.
        self._pre_turn_log_len: int = len(self.log)

        # Set by ``/edit``. While true, the next ``_submit_editor``
        # call does an implicit ``/undo`` first so the submitted
        # text *replaces* the previous user message instead of
        # appending a new turn. Reset on submit (handled in
        # ``_submit_editor``) and on Esc.
        self._edit_in_progress: bool = False

        # Set by ``/goal``. While true, the next ``_submit_editor``
        # call sets the goal text instead of sending a user message.
        # Reset on submit (handled in ``_submit_editor``) and on Esc.
        self._goal_edit_in_progress: bool = False

        # Per-turn live stats, so the status bar tracks data during
        # streaming instead of staying frozen until the API reports
        # usage at the end of the response. Updated as deltas arrive
        # in on_event(); reset to idle in run_agent_turn()'s finally.
        #
        # `turn_active` flips True when a turn starts and False when
        # it ends. `turn_prompt_tokens` is the rough prompt size for
        # the current request (chars/4 — OpenAI's published ratio for
        # English text is closer to 1 token per 3-4 chars, so /4 is a
        # conservative low-ball). `turn_cached_tokens` is the cached
        # portion of the prompt, populated from the API's usage at
        # turn end. `turn_output_tokens` is the running output chars
        # streamed so far. `turn_t0`/`turn_t_first` are for the live
        # tok/s readout.
        self.turn_active: bool = False
        self.turn_prompt_tokens: int = 0
        self.turn_cached_tokens: int = 0
        self.turn_output_tokens: int = 0
        self.turn_t0: float = 0.0
        self.turn_t_first: float | None = None
        self._spinner_frame: int = 0

    def bootstrap(self) -> None:
        """Wire up the agent's callbacks now that state exists."""
        self.agent.confirm_callback = _make_confirm_callback(
            self.agent, self.stdscr, self.approval_level
        )
        self.agent.user_input_callback = self._prompt_user

    # ----- log / streaming ---------------------------------------------

    def push(self, kind: str, text: str, attr: int | None = None) -> None:
        """Append or merge a log entry, invalidating the wrap cache."""
        if attr is None:
            attr = self.A_NORMAL
        if (self.log
                and self.log[-1][0] == kind == "assistant"
                and self.streaming_kind == "assistant"):
            self.log[-1] = (kind, text, attr)
            if self._log_wrapped:
                self._log_wrapped[-1] = None
        elif (self.log
                and self.log[-1][0] == kind == "reasoning"
                and self.streaming_kind == "reasoning"):
            self.log[-1] = (kind, text, attr)
            if self._log_wrapped:
                self._log_wrapped[-1] = None
        else:
            if self.log and self.log[-1][0] != "blank":
                self.log.append(("blank", "", self.A_NORMAL))
                self._log_wrapped.append(None)
            self.log.append((kind, text, attr))
            self._log_wrapped.append(None)
            self.streaming_kind = kind if kind in ("assistant", "reasoning") else None

    def stop_streaming(self) -> None:
        self.streaming_kind = None
        self.streaming_assistant = []
        self.streaming_reasoning = []

    def _estimate_prompt_tokens(self, user_text: str) -> int:
        """Rough prompt-size estimate for the current request.

        Used by the status bar mid-stream when the API hasn't yet
        reported usage. The estimate walks the TUI's mirror of the
        conversation history (which lags the agent's by one turn in
        the other direction — see the comment in ``run_agent_turn``)
        plus the system prompt, the tool schemas, and the user text
        we're about to send.

        The conversion is chars/3.5 (OpenAI's cl100k_base averages
        ~3.5 chars/token for English, ~3 for code, ~2 for CJK — 3.5
        is a reasonable middle). We also add a small per-message
        envelope (4 tokens: ``<|im_start|>{role}\\n<|im_end|>\\n``).
        None of this is exact — the API will overwrite it at end of
        turn with the real tokenizer count.
        """
        import json as _json
        chars = 0
        # System prompt (if any). The agent's _messages also includes
        # it, so we count it once here and skip the system entry
        # below.
        if getattr(self.agent, "system", None):
            chars += len(self.agent.system)
        # Tool schemas. They take up a meaningful chunk of the prompt
        # for tool-using agents (a single tool with a few params is
        # typically 100-200 tokens; multi-tool agents can be 1-2K).
        schemas = getattr(self.agent, "_tool_schemas", None) or []
        if schemas:
            try:
                chars += len(_json.dumps(schemas))
            except Exception:
                # Worst-case fallback if serialization fails.
                chars += 200 * len(schemas)
        # Mirror of the conversation history.
        for msg in self.messages:
            if msg.get("role") == "system":
                # Already counted from agent.system above.
                continue
            content = msg.get("content")
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        chars += len(part.get("text", "") or "")
            # Tool-call arguments also count toward prompt.
            for tc in (msg.get("tool_calls") or []):
                fn = (tc or {}).get("function") or {}
                chars += len(fn.get("name", "") or "")
                chars += len(fn.get("arguments", "") or "")
        # The user message we're about to send.
        chars += len(user_text)
        # Message-envelope overhead. OpenAI's chat format adds about
        # 4 tokens per message for the role markers. The +2 covers
        # the system message and the user message we're sending now
        # (which is in self.messages but the agent will append again
        # when it sends the request — so we double-count the new
        # user message as both content and envelope).
        envelope_tokens = 4 * (len(self.messages) + 2)
        # Convert to tokens. 3.5 chars/token is the rough average;
        # we round to nearest int to avoid fractional display.
        content_tokens = int(round(chars / 3.5))
        return max(1, content_tokens + envelope_tokens)

    def _tick_turn_output(self) -> None:
        """Refresh ``turn_output_tokens`` and ``turn_t_first`` from the
        current streaming buffers. Called per assistant/reasoning
        delta in :meth:`on_event`."""
        out_chars = (
            sum(len(s) for s in self.streaming_assistant)
            + sum(len(s) for s in self.streaming_reasoning)
        )
        # 3.5 chars/token is the rough average for cl100k_base on
        # English text; a single token can decode to multiple chars
        # (e.g. " the" is 1 token = 4 chars), so the streamed text
        # is usually a bit shorter in tokens than chars/4 suggests.
        # The API's reported completion_tokens replaces this estimate
        # at end of turn, so the live number is a moving target.
        est_tokens = int(round(out_chars / 3.5))
        self.turn_output_tokens = max(self.turn_output_tokens, est_tokens)
        if self.turn_t_first is None and out_chars > 0 and self.turn_t0:
            self.turn_t_first = time.time() - self.turn_t0

    def _update_or_push_tool_call(self, ev: dict) -> None:
        """Update or append a streaming tool-call header.

        Called per tool-call delta from the agent. Replaces the existing
        log entry in place if we've already seen this id, otherwise
        appends a new "tool_call" entry. The header is shown as
        ``name(args)`` and the args text grows in place as the model
        streams the JSON.
        """
        tc_id = ev.get("id", "") or ""
        name = ev.get("name", "?")
        args_text = ev.get("arguments", "") or ""
        short = args_text if len(args_text) <= 80 else args_text[:77] + "…"
        text = f"{name}({short})"
        if tc_id and tc_id in self._tool_call_log_idx:
            idx = self._tool_call_log_idx[tc_id]
            # Update in place. Invalidate the wrap cache for this row
            # so the new text is re-wrapped on the next render.
            self.log[idx] = ("tool_call", text, self.A_CYAN)
            if idx < len(self._log_wrapped):
                self._log_wrapped[idx] = None
        else:
            self.push("tool_call", text, self.A_CYAN)
            if tc_id:
                self._tool_call_log_idx[tc_id] = len(self.log) - 1

    # ----- session I/O -------------------------------------------------

    def session_meta(self) -> dict:
        m = {"model": self.agent.model}
        first_user = ""
        for msg in self.messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                first_user = msg["content"]
                break
        t = _safe_title(first_user)
        if t:
            m["title"] = t
        if self._goal:
            m["goal"] = self._goal
        m.update(self.metrics.as_meta())
        return m

    def save_session(self) -> None:
        try:
            _write_session(
                self.session_id, self.messages, self.session_meta(),
                created_at=self.metrics.started_at,
            )
        except OSError as e:
            self.push("note", f"save failed: {type(e).__name__}: {e}", self.A_RED)

    # ----- agent event hook -------------------------------------------

    def on_event(self, ev: dict) -> None:
        # Apply any pending resize before processing this event so the
        # streamed content (or new tool-call line) draws at the right
        # size. render() also calls this, but doing it here first means
        # a resize that arrives between two on_event calls is reflected
        # on the very next event rather than only at render time.
        self._apply_resize()
        role = ev.get("role") or ev.get("type")
        if role == "reasoning":
            self.streaming_reasoning.append(ev.get("delta", ""))
            self._tick_turn_output()
            self.push("reasoning", "".join(self.streaming_reasoning), self.A_DIM)
            self.render()
        elif role == "assistant":
            if "delta" in ev:
                self.streaming_assistant.append(ev["delta"])
            self._tick_turn_output()
            self.push(
                "assistant",
                ev.get("content", "".join(self.streaming_assistant)),
                self.A_GREEN if not self.streaming_assistant else self.A_NORMAL,
            )
            self.render()
        elif role == "tool_call":
            # Streaming tool-call header. Emitted by the agent as the
            # model emits the call (per delta) so the user sees the
            # call appear in the log as soon as the model decides to
            # make it — not after the tool finishes executing.
            self._update_or_push_tool_call(ev)
            self.render()
        elif role == "tool":
            # Tool finished executing. The header was already pushed
            # during streaming (or, for non-streaming, just before this
            # event), so we just add the result below it.
            self.stop_streaming()
            result = ev.get("result", "")
            result_short = result if len(result) <= 2000 else (
                result[:2000] + f"\n… [{len(result) - 2000} more chars in log]"
            )
            self.push("tool", result_short, self.A_DIM)
            self.render()
        elif role == "stats":
            # Handled by the inner runner; the TUI just redraws.
            self.render()
        elif role == "error":
            self.push("note", ev.get("message", "error"), self.A_RED)
            self.render()
        elif role == "auto_compress":
            # The agent decided the conversation is too long and is
            # about to summarise older turns. Surface the trigger in
            # the log so the user can see why the next model call
            # is "wasting" tokens on a summary.
            est = ev.get("est_tokens", 0)
            window = ev.get("window", 0)
            threshold = ev.get("threshold", 0)
            self.push(
                "note",
                f"auto-compressing: ~{_abbr(est)} of "
                f"{_abbr(window)} ctx window (threshold {_abbr(threshold)})",
                self.A_DIM,
            )
            self.render()
        elif role == "auto_compress_done":
            # Compression finished. The summarise step is one model
            # call of its own; show the win so the user knows the
            # context was actually trimmed.
            kept = ev.get("kept", 0)
            summarized = ev.get("summarized", 0)
            chars = ev.get("summary_chars", 0)
            self.push(
                "note",
                f"compressed {summarized} older turns → "
                f"{_abbr(chars)}-char summary, kept last {kept} verbatim",
                self.A_DIM,
            )
            self.render()

    def run_agent_turn(self, text: str) -> None:
        """Run one user turn. ``text`` is the (possibly multi-line) input."""
        # Snapshot the log length so /undo can roll back to
        # the state right before this user message.
        self._pre_turn_log_len = len(self.log)
        self.push("user", text)
        self.push("note", "thinking…", self.A_DIM)
        # Per-turn stats: estimate the prompt size right now (the user
        # message was just pushed; the agent will append it again to
        # self._messages, so we approximate by counting the local
        # mirror). The output counter ticks up via on_event() as the
        # model streams deltas.
        self.turn_active = True
        self.turn_prompt_tokens = self._estimate_prompt_tokens(text)
        self.turn_cached_tokens = 0
        self.turn_output_tokens = 0
        self.turn_t0 = time.time()
        self.turn_t_first = None
        self.render()
        try:
            # Wire the agent's interrupt_check to a TUI-scoped Esc poller.
            self.agent.interrupt_check = lambda: _poll_esc(self.stdscr)

            def _tick() -> None:
                self._spinner_frame += 1
                self.render()

            self.agent.run(text, on_event=self.on_event, stream=True,
                           tick_callback=_tick)
        except KeyboardInterrupt:
            self.push("note", "(interrupted)", self.A_DIM)
            # Drop the user turn if it's still at the end.
            if (self.agent._messages
                    and self.agent._messages[-1].get("role") == "user"
                    and self.agent._messages[-1].get("content") == text):
                self.agent.pop_last()
            self.stop_streaming()
        except Exception as e:
            self.push("note", f"error: {type(e).__name__}: {e}", self.A_RED)
            if (self.agent._messages
                    and self.agent._messages[-1].get("role") == "user"
                    and self.agent._messages[-1].get("content") == text):
                self.agent.pop_last()
            self.stop_streaming()
        finally:
            self.agent.interrupt_check = None
            # Pull the API-reported per-turn numbers out of the agent
            # so the status bar can show the final ground-truth values
            # (rather than the rough char-based estimate we used
            # mid-stream). The agent tracks `last_usage` and exposes
            # the input/output/cache deltas for the most recent turn.
            last = getattr(self.agent, "last_turn_usage", None)
            if last:
                self.turn_prompt_tokens = (
                    last.get("input_tokens", 0) + last.get("cache_read_tokens", 0)
                )
                self.turn_cached_tokens = last.get("cache_read_tokens", 0)
                self.turn_output_tokens = last.get("output_tokens", 0)
            self.turn_active = False
            # Refresh the local view from the agent (in case messages
            # grew via resume or were dropped on error).
            self.messages.clear()
            self.messages.extend(self.agent.messages)
            self.stop_streaming()
            self.save_session()
            self.render()

    def run_agent_turn_with_parts(self, parts: list[dict]) -> None:
        """Submit a multimodal user message (text + image_url parts).

        Same as :meth:`run_agent_turn` but takes a pre-built content
        list instead of a string. The agent's prompt-size estimator
        doesn't know how to count image tokens exactly, so we add a
        rough per-image bump (``APPROX_IMAGE_TOKENS``) to the
        live estimate and let the API's usage chunk overwrite it
        at end of turn.

        The user/attachment log lines were already pushed by
        :meth:`_submit_editor` before we get here, so we don't
        re-push them.
        """
        # Snapshot the log length so /undo can roll back to
        # the state right before this user message.
        self._pre_turn_log_len = len(self.log)
        # Cheap description for the log/estimator: total chars of
        # the text parts, plus a flat per-image bump.
        text_chars = sum(
            len(p.get("text", "") or "")
            for p in parts
            if p.get("type") == "text"
        )
        n_images = sum(1 for p in parts if p.get("type") == "image_url")
        self.push("note", "thinking…", self.A_DIM)
        self.turn_active = True
        # Use the existing estimator (text-only heuristic) plus a
        # flat per-image bump. The total is replaced by the API's
        # reported usage at end of turn anyway.
        base = self._estimate_prompt_tokens("")  # system + tools + history
        # Account for the new user message's text length.
        content_tokens = int(round(text_chars / 3.5)) + 4  # +4 envelope
        self.turn_prompt_tokens = base + content_tokens + (
            APPROX_IMAGE_TOKENS * n_images
        )
        self.turn_cached_tokens = 0
        self.turn_output_tokens = 0
        self.turn_t0 = time.time()
        self.turn_t_first = None
        self.render()
        try:
            self.agent.interrupt_check = lambda: _poll_esc(self.stdscr)

            def _tick() -> None:
                self._spinner_frame += 1
                self.render()

            self.agent.run(parts, on_event=self.on_event, stream=True,
                           tick_callback=_tick)
        except KeyboardInterrupt:
            self.push("note", "(interrupted)", self.A_DIM)
            if (self.agent._messages
                    and self.agent._messages[-1].get("role") == "user"
                    and isinstance(self.agent._messages[-1].get("content"), list)):
                self.agent.pop_last()
            self.stop_streaming()
        except Exception as e:
            self.push("note", f"error: {type(e).__name__}: {e}", self.A_RED)
            if (self.agent._messages
                    and self.agent._messages[-1].get("role") == "user"
                    and isinstance(self.agent._messages[-1].get("content"), list)):
                self.agent.pop_last()
            self.stop_streaming()
        finally:
            self.agent.interrupt_check = None
            last = getattr(self.agent, "last_turn_usage", None)
            if last:
                self.turn_prompt_tokens = (
                    last.get("input_tokens", 0) + last.get("cache_read_tokens", 0)
                )
                self.turn_cached_tokens = last.get("cache_read_tokens", 0)
                self.turn_output_tokens = last.get("output_tokens", 0)
            self.turn_active = False
            self.messages.clear()
            self.messages.extend(self.agent.messages)
            self.stop_streaming()
            self.save_session()
            self.render()

    # ----- render ------------------------------------------------------

    def _get_wrapped_log(self, max_w: int) -> list[tuple[str, list[tuple[str, int]]]]:
        """Per-entry wrapped log with width-aware caching.

        Each entry is wrapped to ``max_w`` columns; the wrapped
        form is a list of ``(kind, spans)`` where ``spans`` is a
        list of ``(text, attr)`` pairs ready to be drawn with
        ``addnstr`` (one ``addnstr`` per span). The cache is
        keyed on the entry's text (changed when the model
        streams new content) and the wrap width (changed on
        terminal resize).
        """
        if max_w != self._wrap_width:
            for i in range(len(self._log_wrapped)):
                self._log_wrapped[i] = None
            self._wrap_width = max_w

        # Re-wrap any entry that has been invalidated.
        for i, entry in enumerate(self.log):
            if i >= len(self._log_wrapped):
                self._log_wrapped.append(None)
            if self._log_wrapped[i] is None:
                kind, text, attr = entry
                self._log_wrapped[i] = self._wrap_entry(kind, text, attr, max_w)

        # Trim if log shrank (e.g. via /clear).
        if len(self._log_wrapped) > len(self.log):
            del self._log_wrapped[len(self.log):]

        result: list[tuple[str, list[tuple[str, int]]]] = []
        for entry_wrapped in self._log_wrapped:
            if entry_wrapped:
                result.extend(entry_wrapped)
        return result

    def _wrap_entry(self, kind: str, text: str, attr: int,
                    max_w: int) -> list[tuple[str, list[tuple[str, int]]]]:
        """Wrap a single log entry to ``max_w`` columns.

        The returned list has one entry per visual line. Each
        visual line is ``(kind, spans)`` where ``spans`` is a
        list of ``(text, attr)`` pairs that concatenate to the
        line's text. Most lines have a single span; lines that
        contain highlighted code have several.
        """
        if kind == "blank":
            return [("blank", [("", self.A_NORMAL)])]
        prefix = _INDENT.get(kind, "")
        # First: split the entry on code fences. Each segment is
        # either "text" (rendered with the entry's default attr)
        # or "code" (passed to the highlighter for token spans).
        # The highlighter preserves the round-trip: spans concatenate
        # back to the segment's text.
        segments: list[tuple[str, str, str]] = (
            _split_code_fences(text) if text else [("text", text, "")]
        )
        # Flatten each segment into a list of (char, attr) so we
        # can word-wrap while respecting span boundaries. A code
        # segment is highlighted; a text segment uses the entry's
        # default attr throughout.
        # For assistant/reasoning entries, use rich markdown rendering.
        if kind in ("assistant", "reasoning") and text:
            char_attrs = self._render_markdown(_latex_to_unicode(text), attr)
        else:
            char_attrs = self._build_char_attrs(segments, attr)
        if not char_attrs:
            return [("blank", [("", self.A_NORMAL)], self.A_NORMAL)]
        # Indent-aware wrap. textwrap only handles whole strings, so
        # we wrap each line of the input separately, then carry
        # the per-char attrs through.
        result: list[tuple[str, list[tuple[str, int]]]] = []
        first = True
        pos = 0
        flat = "".join(c for c, _ in char_attrs)
        for raw in flat.split("\n"):
            indent = prefix if first else (" " * len(prefix))
            # Slice the char_attrs window for this logical line.
            line_chars = char_attrs[pos:pos + len(raw)]
            pos += len(raw) + 1  # +1 for the \n
            if not raw:
                # Empty line: one empty span.
                result.append((kind, [("", self.A_NORMAL)]))
                first = False
                continue
            try:
                # Wrap by character count (indent + remainder), then
                # walk the result and slice ``line_chars`` accordingly.
                wrapped = textwrap.wrap(
                    raw,
                    width=max(1, max_w),
                    initial_indent=indent,
                    subsequent_indent=" " * len(prefix),
                    break_long_words=True,
                    break_on_hyphens=False,
                    drop_whitespace=False,
                )
            except ValueError:
                wrapped = [indent + raw]
            if not wrapped:
                wrapped = [indent]
            for wline in wrapped:
                spans = _line_to_spans(wline, line_chars, indent)
                # Defensive: ensure no visual line exceeds max_w.  Span
                # concatenation should never exceed max_w when textwrap
                # honours the indent-less width, but edge cases (multi-
                # column chars, markdown rendering that changes line
                # length) can push it over.  Truncate the last span if
                # needed — the entry is still correct, just one char
                # shorter.
                total = sum(len(t) for t, _ in spans)
                if total > max_w:
                    excess = total - max_w
                    last_text, last_attr = spans[-1]
                    keep = max(0, len(last_text) - excess)
                    spans[-1] = (last_text[:keep], last_attr)
                result.append((kind, spans))
            first = False
        return result

    def _render_markdown(self, text: str,
                         default_attr: int) -> list[tuple[str, int]]:
        """Parse ``text`` as markdown and return per-character attrs.

        Falls back to plain text (``default_attr`` for every char) if
        Rich's Markdown parser raises.
        """
        try:
            md = _RichMarkdown(text)
            tokens = md.parsed
        except Exception:
            return [(ch, default_attr) for ch in text]

        out: list[tuple[str, int]] = []
        attr_stack = [default_attr]
        list_stack: list[str | list] = []

        def _cur() -> int:
            return attr_stack[-1]

        for token in tokens:
            if token.type == "heading_open":
                level = int(token.tag[1])
                level_attr = getattr(self, f'MD_H{level}', 0)
                attr_stack.append(level_attr or self.MD_HEADING or default_attr)
                prefix = "#" * level + " "
                for ch in prefix:
                    out.append((ch, _cur()))

            elif token.type == "heading_close":
                out.append(("\n", _cur()))
                attr_stack.pop()

            elif token.type == "paragraph_open":
                pass

            elif token.type == "paragraph_close":
                if out and out[-1][0] != "\n":
                    out.append(("\n", _cur()))

            elif token.type == "bullet_list_open":
                list_stack.append("bullet")

            elif token.type == "ordered_list_open":
                list_stack.append(["ordered", 0])

            elif token.type in ("bullet_list_close", "ordered_list_close"):
                if list_stack:
                    list_stack.pop()

            elif token.type == "list_item_open":
                indent = len(list_stack) - 1
                if list_stack and list_stack[-1] == "bullet":
                    prefix = "  " * indent + " • "
                elif list_stack:
                    list_stack[-1][1] += 1
                    prefix = "  " * indent + f" {list_stack[-1][1]}. "
                else:
                    prefix = ""
                for ch in prefix:
                    out.append((ch, _cur()))

            elif token.type == "blockquote_open":
                attr_stack.append(self.MD_BLOCKQUOTE or default_attr)
                for ch in "> ":
                    out.append((ch, _cur()))

            elif token.type == "blockquote_close":
                attr_stack.pop()

            elif token.type == "fence":
                try:
                    spans = _highlight_code(
                        token.content, token.info, default_attr,
                        _token_to_attr,
                    )
                except Exception:
                    spans = [(token.content, default_attr)]
                for span_text, span_attr in spans:
                    for ch in span_text:
                        out.append((ch, span_attr))
                out.append(("\n", _cur()))

            elif token.type == "hr":
                out.append(("\n", _cur()))
                out.append(("─" * 40, self.A_DIM))
                out.append(("\n", _cur()))

            elif token.type == "inline":
                if token.children:
                    for child in token.children:
                        self._process_inline_token(child, out, attr_stack)
                else:
                    for ch in token.content:
                        out.append((ch, _cur()))

        # Strip trailing newline so the wrap pipeline doesn't get an
        # extra blank visual line.
        while out and out[-1][0] == "\n":
            out.pop()
        return out

    def _process_inline_token(
        self,
        token,
        out: list[tuple[str, int]],
        attr_stack: list[int],
    ) -> None:
        """Walk a single inline token (child of an ``inline`` block-level
        token) and append styled ``(char, attr)`` pairs to ``out``."""
        if token.type == "text":
            cur = attr_stack[-1]
            for ch in token.content:
                out.append((ch, cur))

        elif token.type == "strong_open":
            cur = attr_stack[-1]
            attr_stack.append(cur | self.A_BOLD)

        elif token.type == "strong_close":
            if len(attr_stack) > 1:
                attr_stack.pop()

        elif token.type == "em_open":
            cur = attr_stack[-1]
            attr_stack.append(cur | curses.A_UNDERLINE)

        elif token.type == "em_close":
            if len(attr_stack) > 1:
                attr_stack.pop()

        elif token.type == "code_inline":
            code_attr = self.MD_CODE_INLINE or attr_stack[-1]
            for ch in token.content:
                out.append((ch, code_attr))

        elif token.type == "s_open":
            cur = attr_stack[-1]
            attr_stack.append(cur | self.A_DIM)

        elif token.type == "s_close":
            if len(attr_stack) > 1:
                attr_stack.pop()

        elif token.type == "link_open":
            attr_stack.append(self.MD_LINK or attr_stack[-1])

        elif token.type == "link_close":
            if len(attr_stack) > 1:
                attr_stack.pop()

        elif token.type == "softbreak":
            out.append(("\n", attr_stack[-1]))

        elif token.type == "hardbreak":
            out.append(("\n", attr_stack[-1]))

    def _build_char_attrs(
        self,
        segments: list[tuple[str, str, str]],
        default_attr: int,
    ) -> list[tuple[str, int]]:
        """Convert a list of ``(kind, text, lang)`` segments into a
        flat list of ``(char, attr)`` pairs, applying the
        highlighter to code segments.

        The flat list is the input to the line-wrap pass: it
        knows the attr of every character, so when the wrap cuts
        a line at a word boundary the spans align with the
        rendered text.
        """
        out: list[tuple[str, int]] = []
        for seg_kind, seg_text, seg_lang in segments:
            if seg_kind == "text":
                # Prose segment: every char gets the default attr.
                for ch in seg_text:
                    out.append((ch, default_attr))
                continue
            # Code segment: highlight. The highlighter returns
            # spans; we flatten to one char per attr pair.
            try:
                spans = _highlight_code(
                    seg_text, seg_lang, default_attr, _token_to_attr,
                )
            except Exception:
                spans = [(seg_text, default_attr)]
            for span_text, span_attr in spans:
                for ch in span_text:
                    out.append((ch, span_attr))
        return out

    def _truncate_tool_blocks(
        self,
        wrapped: list[tuple[str, list[tuple[str, int]]]],
    ) -> list[list[tuple[str, int]]]:
        """Collapse long tool blocks for display.

        Each entry in ``wrapped`` is ``(kind, spans)``; ``spans``
        is the list of ``(text, attr)`` pairs for one visual
        line. We return a list of span-lists, ready for the
        render loop to draw.
        """
        final_lines: list[list[tuple[str, int]]] = []
        i = 0
        while i < len(wrapped):
            kind, spans = wrapped[i]
            if kind == "tool":
                block_start = i
                while i < len(wrapped) and wrapped[i][0] == "tool":
                    i += 1
                block = wrapped[block_start:i]
                if len(block) > MAX_TOOL_LINES:
                    omitted = len(block) - MAX_TOOL_LINES
                    final_lines.extend(spans for _, spans in block[:MAX_TOOL_LINES])
                    final_lines.append([
                        (f"    ↳ … output truncated ({omitted} more lines)", self.A_DIM),
                    ])
                else:
                    final_lines.extend(spans for _, spans in block)
                continue
            final_lines.append(spans)
            i += 1
        return final_lines

    # Minimum dimensions below which the TUI cannot render anything
    # useful. Below this we just clear the screen and wait for a
    # resize — better a blank terminal than a crash.
    _MIN_ROWS = 3
    _MIN_COLS = 20

    def _apply_resize(self) -> None:
        """Apply any pending terminal resize. Idempotent and crash-safe.

        Called from both :meth:`render` and :meth:`on_event` so a
        SIGWINCH received during streaming is applied on the next tick.
        We can't call ``curses.resizeterm`` from the signal handler
        itself (not async-signal-safe); the SIGWINCH handler just sets
        :attr:`resize_pending`, and we do the actual work here on the
        main thread.

        The flag is re-armed only for the transient "terminal too
        small" case (a window split that briefly reports a tiny pane,
        or the race during the resize itself). Hard failures
        (``OSError`` from ``os.get_terminal_size``, an exception from
        ``curses.resizeterm``) clear the flag — a retry wouldn't help
        and would just spin.
        """
        if not self.resize_pending:
            return
        self.resize_pending = False
        try:
            size = os.get_terminal_size()
        except Exception:
            # get_terminal_size can fail with OSError on non-TTY
            # streams, or with weirdness during a resize race. The
            # terminal is probably gone — clear the flag and bail.
            return
        new_rows, new_cols = size.lines, size.columns
        if new_rows < self._MIN_ROWS or new_cols < self._MIN_COLS:
            # Either a transient race during the resize itself, or
            # the user split their terminal into a too-small pane.
            # Re-arm the flag so we retry next tick, and let render
            # bail to a cleared screen.
            self.resize_pending = True
            return
        try:
            cur_rows, cur_cols = self.stdscr.getmaxyx()
            if new_rows == cur_rows and new_cols == cur_cols:
                return
            curses.resizeterm(new_rows, new_cols)
            # Some terminals (notably over SSH) drop color pairs across
            # a resize. Re-initialize so the A_CYAN / A_RED / etc.
            # attributes still point at valid pairs. The loop must
            # swallow *every* failure mode — in test environments
            # curses.init_pair can raise something other than
            # curses.error, and that would otherwise propagate up and
            # tear down the render.
            for i in range(8):
                try:
                    curses.init_pair(i + 1, i, -1)
                except Exception:
                    pass
        except Exception:
            # resizeterm itself failed. Clear the flag — another
            # retry in the same render tick won't help.
            pass

    def render(self) -> None:
        stdscr = self.stdscr
        # Apply any pending resize before drawing. Doing this at the top
        # of render (rather than mid-draw) keeps the screen state
        # consistent: stdscr.erase() below then draws on a clean canvas
        # at the new size.
        self._apply_resize()
        try:
            h, w = stdscr.getmaxyx()
            if h < self._MIN_ROWS or w < self._MIN_COLS:
                # Too small to fit the status bar, log, and editor.
                # Just clear and refresh; the next render after the
                # user resizes back will draw the real layout.
                try:
                    stdscr.erase()
                    stdscr.refresh()
                except Exception:
                    pass
                return
            self._render_inner()
        except Exception:
            # Curses errors during render are usually recoverable (a
            # coordinate slipped past the new screen size, etc.). The
            # next render — triggered by the next event or key — will
            # redraw cleanly. Don't propagate; that would tear down the
            # TUI for a transient screen-size issue. We deliberately
            # catch Exception (not just curses.error) because some
            # failures during a SIGWINCH race present as TypeError or
            # IndexError when a cached (rows, cols) tuple becomes stale.
            try:
                stdscr.refresh()
            except Exception:
                pass

    def _render_inner(self) -> None:
        stdscr = self.stdscr
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        max_w = w

        # Status bar.
        appr = "yolo" if self.approval_level == "yolo" else (
            "all" if self.approval_level == "all"
            else ("auto:" + self.approval_level)
        )
        # Decide whether to show live per-turn data, last-turn data,
        # or pure session cumulative. The previous version of the
        # bar always showed the cumulative session metrics, which
        # made it look frozen during a streaming response (the API
        # only reports usage at end of stream, so the per-delta
        # renders between deltas all read the same pre-turn values).
        # Now we also keep a "live" view that ticks up per delta,
        # plus a "last" view of the most recent turn (ground-truth
        # from the API if it reported usage, else the live estimate
        # frozen at the last token).
        cached_str = (
            f"+{_abbr(self.turn_cached_tokens)}c"
            if self.turn_cached_tokens else ""
        )
        out_session = _abbr(self.metrics.output_tokens)
        # Live tok/s readout: generation rate AFTER the first token
        # (i.e. excluding TTFT). This is the conventional "tok/s" in
        # LLM UIs — the rate at which the model is producing
        # completion tokens, not the round-trip latency to first
        # token. Suppressed for the first 0.2s of generation so a
        # single token in 1ms doesn't read as "1000 tok/s" (the
        # result is meaningless until we have a non-trivial
        # sample).
        tok_per_sec = ""
        if (self.turn_t_first is not None
                and self.turn_output_tokens > 0
                and time.time() - self.turn_t0 - self.turn_t_first > 0.2):
            elapsed_gen = max(0.001, time.time() - self.turn_t0 - self.turn_t_first)
            tps = self.turn_output_tokens / elapsed_gen
            tok_per_sec = f" · {tps:.1f} tok/s"
        # Cost readout. We show the session-total USD cost
        # (omitted entirely if the model is unpriced or the
        # total is exactly zero, so local models don't add
        # noise). Per-model numbers live in the /cost command.
        cost_str = ""
        if self.metrics.total_cost > 0 and _pricing_for(self.agent.model) is not None:
            cost_str = f" · {_fmt_cost(self.metrics.total_cost)}"
        # Budget cap (if set). Shown as a separate suffix so
        # the user can see how much headroom they have.
        if self.metrics.budget is not None:
            remaining = max(0.0, self.metrics.budget - self.metrics.total_cost)
            cost_str += f" / {_fmt_cost(self.metrics.budget)}"
        if self.turn_active:
            # Mid-stream: live per-turn view. The prompt is the size
            # of the request we just sent; the output ticks up per
            # delta; the cache portion is unknown until the API
            # reports it.
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self._spinner_frame % 10]
            self._spinner_frame += 1
            status = (
                f"{spinner} anduril · {self.agent.model} · {appr} · "
                f"turn ctx {_abbr(self.turn_prompt_tokens)}{cached_str} · "
                f"out {_abbr(self.turn_output_tokens)} · "
                f"ses {out_session}{tok_per_sec}{cost_str}"
            )
        elif self.turn_prompt_tokens or self.turn_output_tokens:
            # Between turns: show the last turn's per-turn numbers
            # (the live estimate, replaced with API-reported values
            # at end-of-turn when the model returned usage info).
            # The session cumulative still rides along on the right
            # so the running total is visible.
            tok_per_sec = ""  # tok/s is meaningless for a past turn
            status = (
                f"anduril · {self.agent.model} · {appr} · "
                f"last ctx {_abbr(self.turn_prompt_tokens)}{cached_str} · "
                f"out {_abbr(self.turn_output_tokens)} · "
                f"ses {out_session}{cost_str}"
            )
        else:
            # Brand-new session with no data yet: show the session
            # cumulative (which is 0 right now).
            ctx = _abbr(
                self.metrics.input_tokens + self.metrics.cache_read_tokens
            )
            cached = (
                f"+{_abbr(self.metrics.cache_read_tokens)} cached"
                if self.metrics.cache_read_tokens else ""
            )
            status = (
                f"anduril · {self.agent.model} · {appr} · "
                f"ctx {ctx}{cached} · in {_abbr(self.metrics.input_tokens)}/"
                f"out {_abbr(self.metrics.output_tokens)}{cost_str}"
            )
        stdscr.addnstr(0, 0, status, max_w, self.A_DIM)

        # Pre-compute visual lines (after word-wrap) for the editor.
        # We need this before sizing the log area, because a long
        # single-line buffer (e.g. a 300-char prompt) should grow the
        # editor to multiple rows, which shrinks the log accordingly.
        #
        # Each visual row stores the actual character range in the
        # original logical line — textwrap.wrap breaks at word
        # boundaries, not at fixed character widths, so col // inner_w
        # is NOT a reliable mapping. We use buf.find() to locate each
        # chunk's start in the source, and remember (start, end) to
        # map a logical (row, col) to a (visual_row, shown_col).
        inner_w_preview = max(1, max_w - 2)
        visual: list[tuple[int, int, int, int, str]] = []
        # (li, wi, char_start, char_end, chunk)
        for _li, _line in enumerate(self.editor.buf):
            if not _line:
                visual.append((_li, 0, 0, 0, ""))
                continue
            _chunks = textwrap.wrap(
                _line, width=inner_w_preview,
                break_long_words=True, break_on_hyphens=False,
                drop_whitespace=False,
            ) or [""]
            _pos = 0
            for _wi, _chunk in enumerate(_chunks):
                _start = _line.find(_chunk, _pos)
                if _start == -1:
                    _start = _pos
                _end = _start + len(_chunk)
                # Strip leading whitespace on continuation rows for
                # visual tidiness. The buffer still holds the
                # original char at _start; the cursor mapping uses
                # the chunk's actual start in the source line.
                shown = _chunk.lstrip() if _wi > 0 else _chunk
                visual.append((_li, _wi, _start, _end, shown))
                _pos = _end

        # Editor height: min(MAX, max(MIN, visual line count)).
        ed_h = min(MAX_EDITOR_LINES, max(MIN_EDITOR_LINES, len(visual)))
        # Log area: h - 1 (status) - 1 (editor top line) - ed_h rows.
        # No bottom border on the editor anymore, so we reclaim that row.
        log_top = 1
        log_h = max(1, h - 1 - 1 - ed_h)

        wrapped = self._get_wrapped_log(max_w)
        final_lines = self._truncate_tool_blocks(wrapped)

        # Clamp scroll to the actual visible-line count. self.scroll is
        # measured in wrapped visible lines (not log entries), so the
        # bound is len(final_lines) - log_h. The wheel/page-up input
        # handlers can push self.scroll past this — the render is the
        # single source of truth for clamping.
        max_scroll = max(0, len(final_lines) - log_h)
        if self.scroll > max_scroll:
            self.scroll = max_scroll

        if self.scroll == 0:
            visible = final_lines[-log_h:]
        else:
            end = len(final_lines) - self.scroll
            start = max(0, end - log_h)
            visible = final_lines[start:end]
        top_pad = log_h - len(visible)
        for i, spans in enumerate(visible):
            row = log_top + top_pad + i
            # Don't draw into the editor's top line.
            if 0 <= row < h - ed_h - 1:
                col = 0
                for text, attr in spans:
                    remaining = max_w - col
                    if remaining <= 0:
                        break
                    try:
                        stdscr.addnstr(row, col, text, remaining, attr)
                    except curses.error:
                        pass
                    col += min(len(text), remaining)

        # Slash-command autocomplete menu overlay. Drawn just above the
        # editor's separator line. We re-clamp the selection here so
        # the displayed highlight always points to a valid match.
        self._menu_keep_selection()
        menu_lines = self._menu_lines(max_w)
        if menu_lines:
            box_top = h - ed_h - 1
            menu_bottom = max(log_top, box_top - 1)
            menu_top = max(log_top, menu_bottom - len(menu_lines) + 1)
            # Clear the background behind the menu first so log text
            # doesn't bleed through.
            for r in range(menu_top, menu_bottom + 1):
                try:
                    stdscr.addnstr(r, 0, " " * max_w, max_w, self.A_NORMAL)
                except curses.error:
                    pass
            for i, spans in enumerate(menu_lines):
                row = menu_top + i
                if row > menu_bottom:
                    break
                col = 0
                for text, attr in spans:
                    if col >= max_w:
                        break
                    try:
                        stdscr.addnstr(row, col, text, max_w - col, attr)
                    except curses.error:
                        pass
                    col += len(text)
                # Pad the rest of the row with spaces in the same attr
                # so selected (REVERSE) rows have a solid highlight bar.
                if col < max_w:
                    attr = spans[-1][1] if spans else self.A_NORMAL
                    try:
                        stdscr.addnstr(row, col, " " * (max_w - col),
                                       max_w - col, attr)
                    except curses.error:
                        pass

        # @-file picker menu overlay. Drawn just above the editor's
        # separator line, same as the slash menu. We keep the two
        # menus independent — if both are active (rare: e.g. the
        # user is typing a slash command that takes a file
        # argument), the file menu wins because it's the more
        # recent interaction.
        self._file_menu_keep_selection()
        file_menu_lines = self._file_menu_lines(max_w)
        if file_menu_lines:
            box_top = h - ed_h - 1
            menu_bottom = max(log_top, box_top - 1)
            menu_top = max(log_top, menu_bottom - len(file_menu_lines) + 1)
            # Clear the background behind the file menu as well.
            for r in range(menu_top, menu_bottom + 1):
                try:
                    stdscr.addnstr(r, 0, " " * max_w, max_w, self.A_NORMAL)
                except curses.error:
                    pass
            for i, spans in enumerate(file_menu_lines):
                row = menu_top + i
                if row > menu_bottom:
                    break
                col = 0
                for text, attr in spans:
                    if col >= max_w:
                        break
                    try:
                        stdscr.addnstr(row, col, text, max_w - col, attr)
                    except curses.error:
                        pass
                    col += len(text)
                if col < max_w:
                    attr = spans[-1][1] if spans else self.A_NORMAL
                    try:
                        stdscr.addnstr(row, col, " " * (max_w - col),
                                       max_w - col, attr)
                    except curses.error:
                        pass
        # visual row of the first logical line gets a `> ` prompt prefix
        # to mark where the user types.
        box_top = h - ed_h - 1
        inner_w = max(1, max_w)
        # Top line: just a straight horizontal line.
        top_line = "─" * inner_w
        try:
            stdscr.addnstr(box_top, 0, top_line, max_w, self.A_DIM)
        except curses.error:
            pass

        # Determine which slice of `visual` to show. We want the
        # cursor's wrap row visible — center it if possible.
        cur_visual = 0
        for i, (li, _, _, _, _) in enumerate(visual):
            if li == self.editor.row:
                cur_visual = i
                break
        if len(visual) <= ed_h:
            vis_start = 0
        else:
            vis_start = max(0, cur_visual - ed_h // 2)
            vis_start = min(vis_start, len(visual) - ed_h)
        shown_visual = visual[vis_start : vis_start + ed_h]

        # Content rows: plain text, no side borders. The first visual
        # chunk of the first logical line gets a `> ` prefix.
        for vi, (_li, _wi, _cs, _ce, chunk) in enumerate(shown_visual):
            row = box_top + 1 + vi
            if row >= h:
                break
            if _li == 0 and _wi == 0:
                display = "> " + chunk
            else:
                display = chunk
            try:
                stdscr.addnstr(row, 0, display.ljust(inner_w), max_w, self.A_NORMAL)
            except curses.error:
                pass

        # Position the cursor inside the box. The cursor's logical
        # (row, col) maps to a (visual row, shown col) using the
        # actual char ranges we stored in `visual` — textwrap.wrap
        # breaks at word boundaries, so col // inner_w is wrong.
        cur_visual_row = 0
        cur_shown_col = self.editor.col
        for i, (_li, _wi, _cs, _ce, _chunk) in enumerate(visual):
            if _li == self.editor.row and _cs <= self.editor.col <= _ce:
                cur_visual_row = i
                # The first visual chunk of the first logical line has
                # a "> " prefix in the display; offset the cursor by 2
                # so it lands right after the prompt.
                prefix = 2 if (_li == 0 and _wi == 0) else 0
                cur_shown_col = self.editor.col - _cs + prefix
                # Clamp to the visible window.
                cur_visual_row = max(vis_start,
                                     min(cur_visual_row, vis_start + ed_h - 1))
                break
        if 0 <= cur_visual_row - vis_start < ed_h:
            shown_col = min(max(0, cur_shown_col), max(0, inner_w - 1))
            cur_y = box_top + 1 + (cur_visual_row - vis_start)
            cur_x = shown_col
            if 0 <= cur_y < h - 1 and 0 <= cur_x < w:
                try:
                    stdscr.move(cur_y, cur_x)
                except curses.error:
                    pass
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.refresh()

    def _prompt_user(self, question: str, options: list[str] | None = None) -> str:
        """Blocking text prompt shown in the TUI. Returns the user's answer."""
        self.push("note", f"  ┌─ Ask from agent: {question}", self.A_BOLD)
        if options:
            for i, opt in enumerate(options, 1):
                self.push("note", f"  │   {i}. {opt}", self.A_NORMAL)
            self.push("note", f"  │   {len(options) + 1}. Type your answer...", self.A_NORMAL)

        saved_buf = list(self.editor.buf)
        saved_row = self.editor.row
        saved_col = self.editor.col
        self.editor.buf = [""]
        self.editor.row = 0
        self.editor.col = 0
        self.render()

        buf: list[str] = []
        prompt_text = "  └─ " if not options else f"  └─ Select [1-{len(options) + 1}]: "

        h, w = self.stdscr.getmaxyx()
        row = max(0, h - 3)
        try:
            while True:
                try:
                    self.stdscr.addnstr(row, 0, " " * max(0, w - 1), max(0, w - 1), curses.A_BOLD)
                    display = prompt_text + "".join(buf) + "▊"
                    self.stdscr.addnstr(row, 0, display, max(0, w - 1), curses.A_BOLD)
                    self.stdscr.refresh()
                except curses.error:
                    pass
                try:
                    ch = self.stdscr.get_wch()
                except KeyboardInterrupt:
                    raise
                except curses.error:
                    continue
                if isinstance(ch, str):
                    if ch == "\n" or ch == "\r":
                        answer = "".join(buf)
                        if options:
                            try:
                                n = int(answer)
                                if 1 <= n <= len(options):
                                    answer = options[n - 1]
                                elif n == len(options) + 1:
                                    answer = ""
                            except ValueError:
                                pass
                        self.push("note", f"  └─ {answer}" if answer else "  └─ (cancelled)", self.A_CYAN)
                        return answer
                    if ch == "\x1b":
                        return ""
                    if ch in ("\x7f", "\b"):
                        if buf:
                            buf.pop()
                    elif ch == "\x03":
                        raise KeyboardInterrupt
                    elif ch.isprintable():
                        buf.append(ch)
        finally:
            self.editor.buf = saved_buf
            self.editor.row = saved_row
            self.editor.col = saved_col
            self.render()

    # ----- command dispatch -------------------------------------------

    def _cmd_quit(self, _arg: str) -> None:
        raise SystemExit(0)

    def _cmd_clear(self, _arg: str) -> None:
        self.agent.clear()
        self.messages.clear()
        self.messages.extend(self.agent.messages)
        self.log.clear()
        self._log_wrapped.clear()
        self._tool_call_log_idx.clear()
        self.scroll = 0  # reset scroll on /clear — the log is empty now
        self.streaming_assistant = []
        self.streaming_reasoning = []
        self.stop_streaming()
        self.session_id = _new_session_id()
        self.metrics = _Metrics(self.session_id, model=self.agent.model)
        self.agent.set_metrics(self.metrics)
        # New session → reset per-turn tracking too, so the status
        # bar doesn't briefly show stale numbers from the old
        # session's last turn.
        self.turn_active = False
        self.turn_prompt_tokens = 0
        self.turn_cached_tokens = 0
        self.turn_output_tokens = 0
        self.turn_t0 = 0.0
        self.turn_t_first = None
        self.agent.last_turn_usage = None
        # Reset goal for the new session.
        self._goal = None
        self.agent.goal = None
        # Re-wire user input callback for the new session.
        self.agent.user_input_callback = self._prompt_user
        # Reset short-reference attachments. The pasted image
        # files themselves are kept on disk (under
        # ~/.local/state/anduril/images/) in case the user
        # wants to re-attach them across sessions, but the
        # session-scoped ID counter restarts at 1 so the
        # numbering stays intuitive.
        self.attachments.clear()
        self._next_attachment_id = 1
        self.push("note",
                  f"  context cleared, new session {self.session_id}",
                  self.A_DIM)

    def _cmd_model(self, arg: str) -> str:
        if not arg:
            return f"  current model: {self.agent.model}"
        self.agent.model = arg
        self.metrics.model = arg
        return f"  model → {arg}"

    def _cmd_system(self, arg: str) -> str:
        """Show / set the system prompt.

        Subcommands:

        * ``/system`` — show the current resolved prompt and
          any per-model overrides.
        * ``/system <text>`` — set the global default prompt.
        * ``/system <model> <text>`` — set a per-model
          override. The most specific (longest-substring)
          match wins when the agent picks a prompt for a
          given model.
        """
        if not arg:
            overrides = self.agent.system_overrides
            lines = [f"  default: {self.agent.system or '(empty)'}"]
            if overrides:
                lines.append("  per-model overrides:")
                # Show by descending key length (most specific
                # first) so the user can see the precedence.
                for key in sorted(overrides, key=len, reverse=True):
                    lines.append(f"    {key:<32s}  {overrides[key]!r}")
            else:
                lines.append("  (no per-model overrides)")
            return "\n".join(lines)
        # Two-arg form: ``/system <model> <text>``. Heuristic:
        # if the first token looks like a real model name
        # (typically contains a digit, dot, or dash — local
        # model paths often do too), treat it as a per-model
        # override. Otherwise treat the whole arg as the
        # global default prompt.
        parts = arg.split(None, 1)
        if len(parts) == 2 and len(parts[0]) >= 3 and len(parts[0]) <= 60:
            first, rest = parts
            # A "real" model name has a non-alpha (digit,
            # dot, dash, slash) or starts with a slash. Pure
            # alpha tokens like "a", "foo" are treated as
            # the start of a prompt.
            is_model_like = (
                any(c in "0123456789-./:_" for c in first)
                and all(c.isalnum() or c in "-._:/" for c in first)
            )
            if is_model_like and rest:
                # Per-model form. Register the override.
                self.agent.set_system(rest, for_model=first)
                return (
                    f"  system override for {first!r} → "
                    f"{len(rest)} chars"
                )
        # Global default.
        self.agent.set_system(arg)
        return f"  system prompt updated ({len(arg)} chars)"

    def _cmd_yolo(self, _arg: str) -> str:
        self.approval_level = "yolo" if self.approval_level != "yolo" else "all"
        self.agent.confirm_callback = _make_confirm_callback(
            self.agent, self.stdscr, self.approval_level
        )
        return f"  yolo={'on' if self.approval_level == 'yolo' else 'off'}"

    def _cmd_approval(self, arg: str) -> str:
        if not arg:
            return (f"  approval={self.approval_level} "
                    f"(all|low|medium|high|yolo)")
        kind, level = _normalize_approval(arg)
        if not kind:
            return f"  unknown level {arg!r} (all|low|medium|high|yolo)"
        if kind == "yolo":
            self.approval_level = "yolo"
        elif kind == "prompt_all":
            self.approval_level = "all"
        else:
            self.approval_level = level
        self.agent.confirm_callback = _make_confirm_callback(
            self.agent, self.stdscr, self.approval_level
        )
        return f"  approval → {self.approval_level}"

    def _cmd_write(self, arg: str) -> str:
        last_text = None
        for msg in reversed(self.agent._messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_text = msg["content"]
                break
        if last_text is None:
            return "  no assistant turn to write yet"
        path = arg.strip() or "anduril-output.txt"
        try:
            p = pathlib.Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(last_text, encoding="utf-8")
        except OSError as e:
            return f"  write failed: {type(e).__name__}: {e}"
        return f"  wrote {len(last_text)} chars to {p}"

    def _cmd_compress(self, _arg: str) -> str:
        body_len = len(self.agent._messages) - (
            1 if self.agent._messages
            and self.agent._messages[0].get("role") == "system"
            else 0
        )
        if body_len <= COMPRESS_KEEP:
            return (f"  nothing to compress "
                    f"({body_len} turn{'s' if body_len != 1 else ''})")
        result = compress(
            self.agent._messages,
            model=self.agent.model,
            client=self.agent.client,
        )
        if result is None:
            return "  compress failed (see error above)"
        kept_n, summarized_n, summary_chars = result
        return (f"  compressed {summarized_n} turns → 1 summary "
                f"({summary_chars} chars), kept last {kept_n} verbatim")

    def _cmd_undo(self, _arg: str) -> str:
        """Drop the most recent assistant turn (and any tool chain).

        Pops the agent's message list back to the last user
        message, and truncates the visible log so the screen
        matches. The user can then re-submit a different
        message (``/edit``) or re-run the same question
        (``/retry``).
        """
        if not self.agent.undo_last_turn():
            return "  nothing to undo"
        # Truncate the log back to the pre-turn snapshot. We
        # also have to truncate the wrap cache and the
        # streaming accumulators (the user might undo mid-
        # stream if they hit Esc and then /undo).
        target = self._pre_turn_log_len
        del self.log[target:]
        del self._log_wrapped[target:]
        self.streaming_assistant = []
        self.streaming_reasoning = []
        self.streaming_kind = None
        self.turn_active = False
        self._tool_call_log_idx.clear()
        # A pending ``/edit`` is no longer meaningful — the
        # user already undid the turn it would have replaced.
        self._edit_in_progress = False
        # Re-sync the local message mirror.
        self.messages.clear()
        self.messages.extend(self.agent.messages)
        # Reset scroll so the user sees the post-undo state
        # at the bottom (where they're typing).
        self.scroll = 0
        return "  undone"

    def _cmd_retry(self, _arg: str) -> str:
        """Re-run the most recent user message.

        Equivalent to ``/undo`` followed by re-submitting the
        same question, but atomic. The model gets a fresh
        attempt at the same prompt — useful when the
        previous answer was bad, the model timed out, or
        the tool chain produced an error.
        """
        if self.turn_active:
            return "  can't /retry mid-turn — wait for the current turn to finish"
        user_msg = self.agent.last_user_message()
        if user_msg is None:
            return "  nothing to retry"
        # Undo the previous turn (so the model sees a clean
        # slate) and re-run. The agent's ``run()`` will
        # re-append the user message, so we don't need to
        # pass it explicitly.
        self._cmd_undo("")
        # Replay. We use the same event hook so the new
        # turn's events show up in the log just like a
        # normal turn would.
        new_content = self.agent.replay_last_user(
            on_event=self.on_event, stream=True,
        )
        # The replay already pushed events to the log, so
        # there's nothing to print here. We return a
        # one-liner so the user knows /retry did something
        # (vs. silently doing nothing on a no-op history).
        if new_content is None:
            return "  retry: no user message to replay"
        return ""

    def _cmd_edit(self, _arg: str) -> str:
        """Load the most recent user message into the editor.

        On submit, the previous turn is dropped (``/undo``) and
        the edited message is submitted. Lets the user
        correct a typo or add context without re-typing the
        whole question.
        """
        user_msg = self.agent.last_user_message()
        if user_msg is None:
            return "  nothing to edit"
        content = user_msg.get("content", "")
        # Multimodal messages can't be edited in the line
        # editor (the image_url parts wouldn't survive).
        # Refuse gracefully.
        if not isinstance(content, str):
            return "  can't edit: most recent user message has non-text content"
        # Pre-fill the editor with the previous text. The
        # user is now expected to edit it and press Enter to
        # submit; that submit will *replace* the previous
        # turn (we set a flag that ``_submit_editor`` checks).
        self.editor.buf = content.split("\n")
        self.editor.row = len(self.editor.buf) - 1
        self.editor.col = len(self.editor.buf[-1])
        # History nav: don't push this onto the editor
        # history (the user typed it before). Cursor at the
        # end so the user can edit and press Enter.
        self._edit_in_progress = True
        return "  edit the message above and press Enter to submit (Esc to cancel)"

    def _set_goal(self, text: str) -> None:
        self._goal = text
        self.agent.goal = text
        # Inject the goal as a permanent system message in _messages
        goal_msg = {"role": "system", "content": f"Goal: {text}"}
        if self.agent._messages and self.agent._messages[0].get("role") == "system":
            for i, msg in enumerate(self.agent._messages):
                if isinstance(msg.get("content"), str) and msg["content"].startswith("Goal: "):
                    self.agent._messages[i] = goal_msg
                    break
            else:
                self.agent._messages.insert(1, goal_msg)
        else:
            self.agent._messages.insert(0, goal_msg)
        # Push a note and fire a standalone turn so the model sees the goal immediately.
        self.push("note", f"  goal set to: {text}", self.A_DIM)
        self.render()
        self.run_agent_turn(text)

    def _clear_goal(self) -> str:
        self._goal = None
        self.agent.goal = None
        # Remove the goal system message from _messages
        if self.agent._messages:
            self.agent._messages[:] = [
                m for m in self.agent._messages
                if not (isinstance(m.get("content"), str) and m["content"].startswith("Goal: "))
            ]
        return "  goal cleared"

    def _cmd_goal(self, arg: str) -> str | None:
        arg = (arg or "").strip()

        if not arg:
            current = self._goal or ""
            self.editor.buf = current.split("\n")
            self.editor.row = len(self.editor.buf) - 1
            self.editor.col = len(self.editor.buf[-1])
            self._goal_edit_in_progress = True
            return "  edit the goal above and press Enter to confirm (Esc to cancel)"

        if arg.lower() in ("clear", "off"):
            return self._clear_goal()

        self._set_goal(arg)
        return None

    def _cmd_autocompress(self, arg: str) -> str:
        """Toggle the per-turn auto-compression trigger.

        ``/autocompress``           — toggle on/off.
        ``/autocompress 0.5``       — set the context fraction (0–1).
        ``/autocompress status``    — show the current setting.
        """
        from anduril.context import context_window_for, estimate_prompt_tokens
        arg = (arg or "").strip().lower()
        if not arg:
            self.agent.auto_compress = not self.agent.auto_compress
            return (f"  auto-compress → "
                    f"{'on' if self.agent.auto_compress else 'off'}")
        if arg in ("status", "show"):
            window = context_window_for(self.agent.model)
            threshold = int(window * self.agent.context_fraction)
            est = estimate_prompt_tokens(
                self.agent._messages,
                system=self.agent.system or "",
                tool_schemas=self.agent._tool_schemas,
            )
            return (f"  auto-compress: "
                    f"{'on' if self.agent.auto_compress else 'off'}  "
                    f"fraction={self.agent.context_fraction:.2f}  "
                    f"window={window}  "
                    f"threshold={threshold}  "
                    f"current=~{est}")
        try:
            f = float(arg)
        except ValueError:
            return (f"  unknown argument {arg!r} (no arg = toggle, "
                    f"or a fraction 0–1, or 'status')")
        if f < 0.0 or f > 1.0:
            return f"  fraction must be between 0 and 1, got {f}"
        self.agent.context_fraction = f
        return f"  auto-compress fraction → {f:.2f}"

    def _cmd_budget(self, arg: str) -> str:
        """Show or set a session cost cap.

        ``/budget`` — show the current cap (or "no cap").
        ``/budget 5.00`` — cap the session at $5.00. The
        next model call that would push the cumulative cost
        past the cap is refused and the run() returns a
        short status message instead of a continuation.
        ``/budget 0`` (or ``/budget off``) — remove the cap.
        ``/budget +1.50`` — raise the existing cap by $1.50.
        ``/budget -0.50`` — lower the cap by $0.50.
        """
        arg = (arg or "").strip()
        m = self.metrics
        if not arg:
            if m.budget is None:
                return "  no budget set (use /budget <usd> to set one)"
            remaining = max(0.0, m.budget - m.total_cost)
            return (
                f"  budget: {_fmt_cost(m.budget)}  "
                f"({_fmt_cost(m.total_cost)} spent, "
                f"~{_fmt_cost(remaining)} remaining)"
            )
        # Relative adjustment.
        if arg.startswith(("+", "-")):
            try:
                delta = float(arg)
            except ValueError:
                return f"  bad number: {arg!r}"
            base = m.budget if m.budget is not None else 0.0
            new_budget = max(0.0, base + delta)
            m.budget = new_budget
            return (
                f"  budget → {_fmt_cost(new_budget)}"
                if m.budget is not None else "  budget cleared"
            )
        # "off" / "0" → clear.
        if arg.lower() in ("off", "none", "0"):
            m.budget = None
            return "  budget cleared (no cap)"
        # Absolute value.
        try:
            value = float(arg)
        except ValueError:
            return f"  bad number: {arg!r}"
        if value < 0:
            return f"  budget must be >= 0, got {value}"
        m.budget = value
        return f"  budget → {_fmt_cost(value)}"

    @dataclass(frozen=True)
    class _Command:
        """A slash command: human-readable description and handler.

        ``fn`` is called as ``fn(self, arg)`` where ``self`` is the
        :class:`_TUIState` and ``arg`` is everything after the command
        name (already stripped). It returns either ``None`` (no
        feedback to print) or a status string that the dispatcher
        pushes to the log as a note.
        """
        description: str
        fn: Callable[["_TUIState", str], str | None]

    _COMMANDS: dict[str, "_TUIState._Command"] = {
        "quit":     _Command("exit the TUI",                          lambda s, a: s._cmd_quit(a)),
        "exit":     _Command("exit the TUI",                          lambda s, a: s._cmd_quit(a)),
        "clear":    _Command("clear context, start a new session",    lambda s, a: s._cmd_clear(a)),
        "model":    _Command("show or set the model (e.g. /model X)", lambda s, a: s._cmd_model(a)),
        "system":   _Command("show or set the system prompt",         lambda s, a: s._cmd_system(a)),
        "yolo":     _Command("toggle approval prompts",               lambda s, a: s._cmd_yolo(a)),
        "approval": _Command("set approval threshold",                lambda s, a: s._cmd_approval(a)),
        "write":    _Command("write last assistant turn to a file",   lambda s, a: s._cmd_write(a)),
        "compress": _Command("summarize older turns to bound context",lambda s, a: s._cmd_compress(a)),
        "autocompress": _Command(
            "toggle auto-compression (/autocompress [0–1|status])",
            lambda s, a: s._cmd_autocompress(a),
        ),
        "skills":   _Command("list installed skills",                 lambda s, a: s._cmd_skills(a)),
        "skill":    _Command("show details for one skill (/skill web)", lambda s, a: s._cmd_skills(a)),
        "paste":    _Command("attach an image from the system clipboard", lambda s, a: s._cmd_paste(a)),
        "attachments": _Command("list short-id attachments (pasted images)",
                                lambda s, a: s._cmd_attachments(a)),
        "mcp":       _Command("list connected MCP servers and their tools",
                                lambda s, a: s._cmd_mcp(a)),
        "goal":     _Command("show, set, or clear the session goal (/goal [clear])",
                                lambda s, a: s._cmd_goal(a)),
        "cost":      _Command("show per-model cost breakdown for the session",
                                lambda s, a: s._cmd_cost(a)),
        "budget":    _Command("show or set a session cost cap (/budget [usd|off|+/-])",
                                lambda s, a: s._cmd_budget(a)),
        "undo":     _Command("drop the last assistant turn (and any tool chain)",
                                lambda s, a: s._cmd_undo(a)),
        "retry":    _Command("re-run the most recent user message",
                                lambda s, a: s._cmd_retry(a)),
        "edit":     _Command("load the most recent user message into the editor",
                                lambda s, a: s._cmd_edit(a)),
    }

    def _cmd_paste(self, _arg: str) -> str:
        """Attach an image from the system clipboard.

        This is a fallback for terminals that don't pass image
        data through Ctrl+Shift+V (or where the user has the
        paste shortcut rebound). It calls the platform's native
        clipboard tool: ``osascript`` on macOS, ``wl-paste`` or
        ``xclip`` on Linux (whichever is installed), PowerShell
        ``Get-Clipboard`` on Windows. If no image is in the
        clipboard (or no platform tool is installed), the call
        returns a one-line status and the editor is left alone.

        On success, we don't push a log line — the @-mention
        appearing in the editor is the feedback. The previous
        behaviour of pushing ``image attached: XXX.png
        (as @image-1)`` was clutter: the line lingered after
        the user deleted the @-mention, looking like a ghost
        attachment. Use ``/attachments`` to see the full
        filename → short-ID mapping if needed.
        """
        result = _read_clipboard_image()
        if result is None:
            tools = _clipboard_tools_status()
            if tools.startswith("none"):
                return "  no clipboard tool available " \
                       f"({tools}); install wl-clipboard or xclip"
            return ("  no image in clipboard "
                    f"(tools available: {tools})")
        data, ext = result
        try:
            path = _save_pasted_image(data, ext)
        except (OSError, ValueError) as e:
            return f"  paste failed: {type(e).__name__}: {e}"
        short_id = self._register_attachment(path)
        self.editor.insert_text(f"@{short_id} ")
        return ""  # silent success — the @-mention is the feedback

    def _cmd_attachments(self, _arg: str) -> str:
        """List all short-id attachments and their file paths.

        Companion to ``/paste``: shows the user which
        ``@image-N`` references are bound to which actual
        files. The output also marks each entry with whether
        the corresponding ``@image-N`` is currently in the
        editor buffer — useful for finding "ghost" attachments
        (paste-then-delete) and confirming the live state
        before submitting.
        """
        if not self.attachments:
            return "  no attachments"
        # Mark each short ID as either in-use (referenced in
        # the current buffer) or stale (the user deleted the
        # @-mention since the paste).
        buf = "\n".join(self.editor.buf)
        referenced = set()
        for m in re.finditer(r"@image-(\d+)", buf):
            try:
                referenced.add(f"image-{int(m.group(1))}")
            except ValueError:
                pass
        lines = ["  attachments:"]
        # Sort by short ID number for stable output.
        for short_id in sorted(self.attachments,
                               key=lambda s: int(s.split("-", 1)[1])):
            path = pathlib.Path(self.attachments[short_id])
            mark = "in use" if short_id in referenced else "stale"
            lines.append(f"    @{short_id}  {path.name}  ({mark})")
            lines.append(f"      {path}")
        return "\n".join(lines)

    def _cmd_cost(self, _arg: str) -> str:
        """Show a per-model cost breakdown for the current session.

        Reads the cumulative ``cost_by_model`` dict from the
        metrics object. If no pricing data is available for
        the current model, we surface that explicitly rather
        than silently showing $0.00 (the user should know
        they need to set ANDURIL_PRICING_OVERRIDES or
        upgrade the model name).
        """
        m = self.metrics
        if m.api_calls == 0:
            return "  no API calls yet"
        lines = [
            f"  cost: {_fmt_cost(m.total_cost)}  "
            f"({m.api_calls} call{'s' if m.api_calls != 1 else ''}, "
            f"{_abbr(m.input_tokens + m.cache_read_tokens)} in, "
            f"{_abbr(m.output_tokens)} out)",
        ]
        if m.cost_by_model:
            lines.append("  per model:")
            # Sort by cost descending so the expensive model
            # is at the top.
            for model, cost in sorted(
                m.cost_by_model.items(), key=lambda kv: kv[1], reverse=True,
            ):
                lines.append(f"    {model:<32s}  {_fmt_cost(cost)}")
        # Show pricing-lookup status for the current model so
        # the user knows whether the live status-bar readout
        # is "real" or suppressed.
        live = _pricing_for(self.agent.model)
        if live is None:
            lines.append(
                f"  pricing for {self.agent.model!r}: not in table "
                f"(set ANDURIL_PRICING_OVERRIDES to add it)"
            )
        return "\n".join(lines)

    def _cmd_mcp(self, _arg: str) -> str:
        """List the MCP servers and their tools (if connected).

        We have to peek at the agent's tool list to find the
        ones whose ``fn`` is an MCP-backed closure. The
        closure has a ``__name__`` we set, but no other
        accessible metadata; the cleanest filter is by
        name prefix. If the user has both native and MCP
        tools with the same name, the MCP one wins (we look
        for ``__`` in the name, which is the conventional MCP
        separator produced by ``_tool_name_for``).
        """
        mcp_tools = [t for t in self.agent.tools.values() if "__" in t.name]
        if not mcp_tools:
            return ("  no MCP tools registered  "
                    "(configure servers in pyproject.toml under "
                    "[tool.anduril.mcp_servers] or via ANDURIL_MCP_SERVERS)")
        # Group by server prefix.
        groups: dict[str, list[str]] = {}
        for t in mcp_tools:
            server, _, _tool = t.name.partition("__")
            groups.setdefault(server, []).append(_tool)
        lines = [f"  MCP tools ({len(mcp_tools)} across "
                 f"{len(groups)} server{'s' if len(groups) != 1 else ''}):"]
        for server in sorted(groups):
            lines.append(f"    {server}:")
            for tool in sorted(groups[server]):
                lines.append(f"      - {tool}")
        return "\n".join(lines)

    def _cmd_skills(self, arg: str) -> str:
        """List installed skills, or show details for a single skill.

        ``/skills`` — list all installed skills.
        ``/skill <name>`` — show details for one skill.
        """
        # Lazy import to avoid pulling skills.py at module load time.
        from anduril.skills import list_skills
        skills = list_skills()
        if not skills:
            return ("  no skills installed  "
                    "(drop a Python module exposing `tools = [...]` into "
                    "~/.local/share/anduril/skills/ or any $ANDURIL_SKILLS_PATH dir)")
        name = arg.strip()
        if name:
            # /skill <name> — show one
            for s in skills:
                if s["name"] == name:
                    tool_list = ", ".join(s["tools"]) if s["tools"] else "(none)"
                    return (
                        f"  {s['name']}  ({len(s['tools'])} tool{'s' if len(s['tools']) != 1 else ''})\n"
                        f"    {s['description'] or '(no description)'}\n"
                        f"    path: {s['path']}\n"
                        f"    tools: {tool_list}"
                    )
            available = ", ".join(s["name"] for s in skills)
            return f"  no skill named {name!r}  (available: {available})"
        # /skills — list all
        lines = [f"  skills ({len(skills)} installed):"]
        for s in skills:
            n_tools = len(s["tools"])
            tool_summary = (
                f"  ({n_tools} tool{'s' if n_tools != 1 else ''})"
                if n_tools else "  (no tools loaded — check deps)"
            )
            lines.append(f"  • {s['name']}{tool_summary}")
            if s["description"]:
                lines.append(f"      {s['description']}")
            if s["tools"]:
                lines.append(f"      tools: {', '.join(s['tools'])}")
        lines.append("")
        lines.append("  use /skill <name> for details on one skill")
        return "\n".join(lines)

    def _handle_command(self, text: str) -> None:
        parts = text.split(None, 1)
        cmd = parts[0][1:].lower()  # strip leading "/"
        arg = parts[1] if len(parts) > 1 else ""
        entry = self._COMMANDS.get(cmd)
        if entry is None:
            names = " /".join(sorted(self._COMMANDS))
            self.push(
                "note",
                f"  unknown command: /{cmd}  (try /{names})",
                self.A_DIM,
            )
            self.render()
            return
        try:
            result = entry.fn(self, arg)
        except SystemExit:
            raise
        except Exception as e:
            self.push("note",
                      f"  {cmd} failed: {type(e).__name__}: {e}", self.A_DIM)
            self.render()
            return
        if result is not None:
            self.push("note", result, self.A_DIM)
            self.render()

    # ----- slash command menu -----------------------------------------

    def _menu_active(self) -> bool:
        """True if the slash-command autocomplete menu should be shown.

        Active when the current (first) line starts with ``/`` and the
        cursor has not yet moved past the command name (no space typed).
        Multi-line input disables the menu since none of the existing
        commands span lines.
        """
        if not self.editor.buf:
            return False
        if self.editor.row != 0:
            return False
        line = self.editor.buf[0]
        if not line.startswith("/"):
            return False
        # If there's a space, the user is on the argument part — no menu.
        if " " in line:
            return False
        return True

    def _menu_matches(self) -> list[str]:
        """Return command names that match the current ``/prefix``."""
        if not self._menu_active():
            return []
        line = self.editor.buf[0]
        prefix = line[1:].lower()
        return sorted(n for n in self._COMMANDS if n.startswith(prefix))

    def _menu_keep_selection(self) -> None:
        """Clamp :attr:`menu_selected` to the current match list."""
        n = len(self._menu_matches())
        if n == 0:
            self.menu_selected = 0
        elif self.menu_selected >= n:
            self.menu_selected = n - 1

    def _menu_move(self, delta: int) -> None:
        matches = self._menu_matches()
        if not matches:
            return
        self.menu_selected = (self.menu_selected + delta) % len(matches)

    def _menu_complete(self) -> bool:
        """Apply the menu's selection to the editor buffer. Returns True
        if the buffer was modified (so the caller can re-render)."""
        matches = self._menu_matches()
        if not matches:
            return False
        # If there's exactly one match, take it. Otherwise extend the
        # buffer to the longest common prefix of all matches.
        if len(matches) == 1:
            target = matches[0]
        else:
            target = _common_prefix(matches)
        new_line = "/" + target
        if self.editor.buf[0] == new_line:
            return False
        self.editor.buf[0] = new_line
        self.editor.col = len(new_line)
        self.menu_selected = 0
        return True

    def _menu_lines(self, max_w: int) -> list[list[tuple[str, int]]]:
        """Build the menu as a list of visual lines, each a list of
        ``(text, attr)`` spans ready to be drawn with ``addnstr`` per
        span. Returns an empty list if the menu is inactive.

        Layout per row: ``▶ name  description`` (or two spaces if not
        selected). Rows are truncated to ``max_w`` columns.
        """
        if not self._menu_active():
            return []
        matches = self._menu_matches()
        if not matches:
            return [[("  (no matching commands)", self.A_DIM)]]
        # Width budget: cap at 60% of screen, but at least 30 chars.
        budget = min(max_w, max(30, max_w * 3 // 5))
        rows: list[list[tuple[str, int]]] = []
        # Top border: a faint horizontal rule.
        rows.append([("─" * min(budget, max_w), self.A_DIM)])
        name_w = max(len(n) for n in matches)
        for i, name in enumerate(matches):
            marker = "▶ " if i == self.menu_selected else "  "
            desc = self._COMMANDS[name].description
            row_attr = curses.A_REVERSE if i == self.menu_selected else self.A_NORMAL
            line = f"{marker}{name:<{name_w}}  {desc}"
            if len(line) > max_w:
                line = line[: max_w - 1] + "…"
            rows.append([(line, row_attr)])
        return rows

    # ----- @ file menu -----------------------------------------------

    def _file_menu_candidates(self) -> list[str]:
        """Return the candidate path list, refreshing the cache if needed.

        The cache key is ``(cwd, mtime-of-cwd)`` — both cheap to read
        and sufficient for typical usage. A new file in a subdirectory
        won't bump the root mtime, but typing a few more characters
        of the path will narrow the result anyway, and the user can
        always :kbd:`Ctrl-L` to force a redraw that refreshes the
        cache (we expose this on the menu itself via the
        ``[refresh]`` hint, but it's not necessary to act on it
        because stale cache results are a UX nit, not a bug).
        """
        try:
            cwd = pathlib.Path.cwd()
        except OSError:
            cwd = pathlib.Path.home()
        try:
            mtime = cwd.stat().st_mtime
        except OSError:
            mtime = 0.0
        key = (str(cwd), mtime)
        if (self.file_menu_cache_key == key
                and self.file_menu_candidates is not None):
            return self.file_menu_candidates
        try:
            paths = _list_files(
                cwd, max_count=FILE_MENU_MAX_FILES,
            )
        except OSError:
            paths = []
        # Sort for deterministic ordering before fuzzy_match re-sorts
        # by score — short paths first, then alphabetical.
        paths.sort(key=lambda p: (len(p.parts), str(p).lower()))
        self.file_menu_candidates = [str(p) for p in paths]
        self.file_menu_cwd = cwd
        self.file_menu_cache_key = key
        # Invalidate the last-query cache so a fresh scan re-ranks.
        self.file_menu_last_query = None
        return self.file_menu_candidates

    def _file_menu_active(self) -> bool:
        """True if the ``@``-file menu should be shown.

        The menu is active when the cursor is currently inside a
        mention — see :func:`anduril.files.find_active_mention`.
        Concretely: there's an ``@`` somewhere on the current
        line that is preceded by a non-identifier char, and the
        cursor is somewhere after it without a terminator in
        between.
        """
        if not self.editor.buf:
            return False
        if self.editor.row >= len(self.editor.buf):
            return False
        # Compose the buffer up to the cursor (multi-line friendly).
        # find_active_mention needs an absolute offset, so we walk
        # the rows to build one.
        offset = 0
        for r in range(self.editor.row):
            offset += len(self.editor.buf[r]) + 1  # +1 for \n
        offset += self.editor.col
        full = "\n".join(self.editor.buf)
        return _find_active_mention(full, offset) is not None

    def _file_menu_query(self) -> str:
        """Return the current search query (text after ``@`` up to cursor)."""
        if not self.editor.buf:
            return ""
        if self.editor.row >= len(self.editor.buf):
            return ""
        offset = 0
        for r in range(self.editor.row):
            offset += len(self.editor.buf[r]) + 1
        offset += self.editor.col
        full = "\n".join(self.editor.buf)
        q, _, _ = _mention_query(full, offset)
        return q

    def _file_menu_matches(self) -> list[str]:
        """Return fuzzy-ranked candidates for the current query.

        Memoizes on the query string so each render tick (or each
        cursor move that doesn't change the query) is O(1). When
        the query changes, the full fuzzy match runs once against
        the cached candidate list.
        """
        q = self._file_menu_query()
        if q == self.file_menu_last_query and self.file_menu_last_matches:
            return [name for _, name in self.file_menu_last_matches]
        cands = self._file_menu_candidates()
        ranked = _fuzzy_match(
            q, cands, limit=FILE_MENU_CANDIDATE_LIMIT,
        )
        self.file_menu_last_query = q
        self.file_menu_last_matches = ranked
        return [name for _, name in ranked]

    def _file_menu_move(self, delta: int) -> None:
        """Move the menu selection. Called on Up/Down arrow keys."""
        n = len(self._file_menu_matches())
        if n == 0:
            self.file_menu_selected = 0
            return
        self.file_menu_selected = (self.file_menu_selected + delta) % n

    def _file_menu_complete(self) -> bool:
        """Replace the active mention with the selected file path.

        Returns True if the buffer was modified. The inserted text
        is ``@<path>`` — the ``@`` is preserved so the buffer is
        a valid mention when the user submits.
        """
        matches = self._file_menu_matches()
        if not matches:
            return False
        if self.file_menu_selected >= len(matches):
            self.file_menu_selected = 0
        path = matches[self.file_menu_selected]
        # Build the absolute position of the cursor and the @-span.
        if not self.editor.buf:
            return False
        offset = 0
        for r in range(self.editor.row):
            offset += len(self.editor.buf[r]) + 1
        offset += self.editor.col
        full = "\n".join(self.editor.buf)
        span = _find_active_mention(full, offset)
        if span is None:
            return False
        at_pos, cursor_pos = span
        # Convert absolute offsets to (row, col) for the editor.
        # We only support replacements on the current row (a
        # mention that spans rows is a pathological case we don't
        # expect; bail out cleanly if it somehow happens).
        row = self.editor.row
        line = self.editor.buf[row]
        if at_pos < offset - self.editor.col or cursor_pos > offset - self.editor.col + len(line):
            return False
        local_start = at_pos - (offset - self.editor.col)
        local_end = cursor_pos - (offset - self.editor.col)
        # local_end is the cursor's local col (the mention ends at the
        # cursor). insert the full path text, then move cursor to the
        # end of it.
        new_text = f"@{path}"
        self.editor.replace_range(local_start, local_end, new_text, row=row)
        self.file_menu_selected = 0
        # Force a re-rank on the next render (the query just changed).
        self.file_menu_last_query = None
        return True

    def _file_menu_lines(self, max_w: int) -> list[list[tuple[str, int]]]:
        """Build the file menu as visual spans, mirroring the slash menu.

        The layout per row is ``  [file] path/to/file.txt`` (or
        ``  [img]  path/to/image.png`` for images), with the
        selected row highlighted via ``A_REVERSE``. We cap at
        ``FILE_MENU_ROWS`` rows; the rest are reachable by typing
        more characters of the query, not by scrolling the menu.

        If no candidates are found, the menu shows a hint line
        (``(no matching files)``).
        """
        if not self._file_menu_active():
            return []
        matches = self._file_menu_matches()
        q = self._file_menu_query()
        # Top border: a faint horizontal rule, like the slash menu.
        rows: list[list[tuple[str, int]]] = []
        rows.append([("-" * min(max_w, max_w), self.A_DIM)])
        # Hint line — the current query and a count of matches.
        # Suppressed when the menu is empty (to keep the layout tight).
        if matches:
            n = len(matches)
            n_more = max(0, n - FILE_MENU_ROWS)
            label = f"  @ {q}  {n} match{'es' if n != 1 else ''}"
            if n_more:
                label += f"  ({n_more} more - keep typing)"
            if len(label) > max_w:
                label = label[: max_w - 1] + "..."
            rows.append([(label, self.A_DIM)])
        visible = matches[:FILE_MENU_ROWS]
        if not visible:
            rows.append([("  (no matching files)", self.A_DIM)])
            return rows
        # Width budget: cap at 60% of screen, but at least 30 chars.
        # The "[file] " or "[img]  " tag is 7 columns.
        for i, path in enumerate(visible):
            tag = "[img]  " if _is_image(path) else "[file] "
            marker = "> " if i == self.file_menu_selected else "  "
            attr = curses.A_REVERSE if i == self.file_menu_selected else self.A_NORMAL
            # Truncate long paths with a trailing ellipsis. The path
            # itself is preserved in the buffer; the display just
            # has to fit in the terminal.
            budget = max(8, max_w - len(marker) - len(tag))
            if len(path) > budget:
                # Show the tail of the path (the part the user is
                # most likely to be searching for) and elide the
                # head.
                tail = path
                head = ""
                while len(tail) + 3 > budget and tail:
                    head = tail[0] + head
                    tail = tail[1:]
                shown = "..." + tail if tail else path[:budget]
            else:
                shown = path
            line = f"{marker}{tag}{shown}"
            if len(line) > max_w:
                line = line[: max_w - 1] + "..."
            rows.append([(line, attr)])
        return rows

    def _file_menu_keep_selection(self) -> None:
        """Clamp :attr:`file_menu_selected` to the current match list."""
        n = len(self._file_menu_matches())
        if n == 0:
            self.file_menu_selected = 0
        elif self.file_menu_selected >= n:
            self.file_menu_selected = n - 1

    # ----- attachment short IDs --------------------------------------

    def _register_attachment(self, path: pathlib.Path) -> str:
        """Add a new short-reference attachment and return its ID.

        The ID looks like ``image-1``, ``image-2``, ... and is
        what the user sees in the editor (``@image-1``). The
        actual filesystem path is stored in
        :attr:`attachments`; :func:`anduril.files.expand_mentions`
        looks each ID up there at submit time. This keeps the
        visible buffer line short even for auto-generated names
        like ``image-2026-06-24-145848-001.png``.

        ID allocation scans the current buffer for any
        ``@image-N`` reference and picks the smallest N NOT
        already in use. This means that pasting, deleting, and
        pasting again reuses ``image-1`` rather than bumping
        to ``image-2`` — the user always sees the smallest
        number that fits the current state of the editor.
        Sessions that genuinely have many simultaneous images
        (e.g. comparing two screenshots side-by-side) get
        ``image-1`` and ``image-2`` as expected.

        A previously-pasted image whose ``@image-N`` was
        deleted from the buffer is still on disk and still
        in :attr:`attachments`, but the ID is fair game for
        reuse. If the user later types ``@image-1`` themselves,
        they get whatever file is currently mapped to that ID
        (which is the most recent paste that wasn't followed
        by a delete). This is the simplest, least-surprising
        model: short IDs are pointers, and a freed pointer can
        be reassigned.
        """
        # Collect the set of IDs currently referenced anywhere
        # in the editor buffer. We look at the full buffer
        # (not just the current line) so a paste on a different
        # row from an existing reference still gets a unique
        # ID.
        used: set[int] = set()
        buf = "\n".join(self.editor.buf)
        for m in re.finditer(r"@image-(\d+)", buf):
            try:
                used.add(int(m.group(1)))
            except ValueError:
                pass
        # Find the smallest unused ID. We track the high-water
        # mark separately for the diagnostics (the counter
        # shown in /skills / debug) but it does NOT influence
        # allocation — that would conflict with the reuse
        # behaviour.
        n = 1
        while n in used:
            n += 1
        self._next_attachment_id = max(self._next_attachment_id, n + 1)
        short_id = f"image-{n}"
        self.attachments[short_id] = str(path)
        return short_id

    def _attachment_label(self, short_id: str) -> str:
        """Return a human-readable label for an attachment.

        Used in log lines so the user can see which file the
        short ID refers to. Falls back to the short ID itself
        if the attachment has been removed.
        """
        path = self.attachments.get(short_id)
        if not path:
            return short_id
        return pathlib.Path(path).name

    def _accumulate_image_chunk(self, data: bytes, ext: str) -> None:
        """Append one chunk to the multi-chunk image-paste accumulator.

        The Kitty protocol can split a single image across many
        ``\\x1b_G`` sequences. We detect "more chunks coming" via
        the ``_read_image_paste._more_chunks`` flag (set by the
        parser when the chunk's params include ``m=1``). When
        the flag is false, this is the LAST chunk — we
        concatenate all accumulated chunks, base64-decode the
        whole thing once (decoding per-chunk would fail on
        partial base64 strings), and save the assembled image.

        Per-chunk ``data`` may be either raw base64 text (for
        continuation chunks and the first chunk of a multi-
        chunk paste) or actual decoded bytes (for single-chunk
        pastes, which the parser decodes eagerly). We detect
        the difference by looking at the format: raw b64 is
        ASCII-only, decoded bytes are typically binary.
        """
        global _ImageChunk
        # Append the chunk. If we have an existing
        # accumulator, the extension must match — otherwise
        # the chunks are from different images and we drop
        # the stale buffer.
        if _ImageChunk is None:
            _ImageChunk = ([data], ext)
        else:
            chunks, existing_ext = _ImageChunk
            if ext != existing_ext:
                _ImageChunk = ([data], ext)
            else:
                chunks.append(data)
        # If more chunks are coming, just wait.
        if getattr(_read_image_paste, "_more_chunks", False):
            return
        # Last chunk — assemble and decode.
        chunks, ext = _ImageChunk
        _ImageChunk = None
        # Try to decode the concatenated data as a single
        # base64 blob. If the per-chunk parser already
        # produced raw bytes (single-chunk path), this is a
        # no-op (the bytes are the actual image content).
        joined = b"".join(chunks)
        try:
            text = joined.decode("ascii", errors="strict")
            # Looks like base64 text — decode it.
            decoded = base64.b64decode(text, validate=False)
        except (UnicodeDecodeError, ValueError, Exception):
            # Not ASCII / not b64 — assume it's already
            # decoded bytes (single-chunk fast path).
            decoded = joined
        if not decoded:
            # The image_paste_failed path stashes ("", "png")
            # on the reader; this catches that case.
            self.push("note",
                      "image paste produced no data "
                      "(try /paste if your terminal doesn't pass images)",
                      self.A_DIM)
            self.render()
            return
        try:
            path = _save_pasted_image(decoded, ext)
        except (OSError, ValueError) as e:
            self.push("note",
                      f"image paste failed: {type(e).__name__}: {e}",
                      self.A_RED)
            self.render()
            return
        # Register a short attachment ID and insert it at the
        # cursor. The @-mention appearing in the editor is
        # the feedback — we don't push a separate "image
        # attached" log line because that lingers after the
        # user deletes the @-mention, looking like a ghost
        # attachment. Use /attachments for the filename.
        short_id = self._register_attachment(path)
        self.editor.insert_text(f"@{short_id} ")
        self.render()

    def _dismiss_file_menu(self) -> None:
        """Drop the active ``@-mention`` from the buffer and reset menu state.

        Called on bare-Esc while the file menu is open. We remove the
        partial mention (the ``@`` and the query text up to the cursor)
        so the next keypress doesn't immediately re-open the menu;
        otherwise the user would have to backspace through the same
        characters they were trying to dismiss.

        If the active mention is in the middle of a larger word
        (e.g. the user typed ``user@host``), we leave the buffer
        alone and just reset the menu state — the cursor isn't
        "inside" a real mention, so there's nothing clean to
        delete.
        """
        if not self.editor.buf or self.editor.row >= len(self.editor.buf):
            return
        # Compute the cursor's absolute offset.
        offset = 0
        for r in range(self.editor.row):
            offset += len(self.editor.buf[r]) + 1
        offset += self.editor.col
        full = "\n".join(self.editor.buf)
        span = _find_active_mention(full, offset)
        if span is None:
            self.file_menu_selected = 0
            self.file_menu_last_query = None
            return
        at_pos, cursor_pos = span
        # Convert to (row, col). Only handle single-row mentions.
        row = self.editor.row
        line_start = offset - self.editor.col
        if at_pos < line_start or cursor_pos > line_start + len(self.editor.buf[row]):
            self.file_menu_selected = 0
            self.file_menu_last_query = None
            return
        local_start = at_pos - line_start
        local_end = cursor_pos - line_start
        self.editor.replace_range(local_start, local_end, "", row=row)
        self.file_menu_selected = 0
        self.file_menu_last_query = None

    # ----- key dispatch ------------------------------------------------

    def _submit_editor(self) -> None:
        if self.editor.is_empty():
            return
        text = self.editor.submit()
        if text.startswith("/"):
            self._handle_command(text)
            return

        if self._goal_edit_in_progress:
            self._goal_edit_in_progress = False
            self.editor.buf = [""]
            self.editor.row = 0
            self.editor.col = 0
            if text:
                self._set_goal(text)
            else:
                self.render()
            return

        # ``/edit`` pre-fills the editor with the previous user
        # message and sets this flag. The submission is a
        # *replacement*, not a new turn — undo the previous
        # turn first so the model doesn't see both the old and
        # the new message.
        if self._edit_in_progress:
            self._edit_in_progress = False
            # Undo the previous turn. The ``run_agent_turn`` we
            # call below will re-append the user message, so
            # the agent's history ends up correct.
            self.agent.undo_last_turn()
            # Also truncate the log so the screen doesn't
            # briefly show the old turn's "thinking..." /
            # tool lines when the new turn starts streaming.
            target = self._pre_turn_log_len
            del self.log[target:]
            del self._log_wrapped[target:]
            self._tool_call_log_idx.clear()
            self.messages.clear()
            self.messages.extend(self.agent.messages)
            self.scroll = 0
        # Expand ``@path`` mentions into a multimodal content list.
        # If the expansion yields more than one part, or any
        # non-text part, send the list; otherwise the original
        # string is the cheapest payload. The attachments dict
        # maps short IDs (``image-1``) to absolute paths so
        # pasted images keep a short visible line in the
        # editor while the actual file lookup happens here.
        try:
            parts = _expand_mentions(
                text,
                cwd=pathlib.Path.cwd(),
                max_text_chars=_MAX_TEXT_CHARS,
                max_image_bytes=_MAX_IMAGE_BYTES,
                attachments=self.attachments,
            )
        except Exception as e:
            # Never let a file-read error block the user from
            # submitting. Fall back to plain text and surface a note.
            self.push("note", f"@ expansion failed: {type(e).__name__}: {e}",
                      self.A_DIM)
            self.run_agent_turn(text)
            return
        is_multimodal = (
            len(parts) > 1
            or any(p.get("type") != "text" for p in parts)
        )
        # Also collect the list of attached files for the log, so
        # the user can see what was sent.
        attachments: list[tuple[str, str]] = []  # (kind, label)
        if is_multimodal:
            for p in parts:
                if p.get("type") == "image_url":
                    # The data URL is too long to display; show the
                    # marker so the user can see "yes, the image went".
                    attachments.append(("image", "(attached image)"))
                elif p.get("type") == "text":
                    text_body = p.get("text", "")
                    # Heuristic: text parts that start with "[file:"
                    # are inlined file bodies — list their path in
                    # the attachment summary. The path may be a
                    # short attachment ID (``image-1``) in which
                    # case we resolve it to the actual filename
                    # via the attachments dict, so the log shows
                    # something the user can recognise.
                    s = text_body.lstrip()
                    if s.startswith("[file:"):
                        end = s.find("]")
                        if end > 0:
                            label = s[6:end].strip()
                            if label in self.attachments:
                                label = pathlib.Path(
                                    self.attachments[label]
                                ).name
                            attachments.append(("text", label))
            # Show a short log line for each attached file (one
            # combined line is fine — the model still gets the
            # full body).
            self.push("user", text)
            if attachments:
                lines = ["  + " + (f"[image] {lbl}" if k == "image" else f"[file] {lbl}")
                         for k, lbl in attachments]
                self.push("user_attachment", "\n".join(lines),
                          self.A_DIM)
            # Coalesce the text parts back into a single string
            # for the user log (the model still gets the
            # interleaved list). We don't want to dump the file
            # bodies in the log — that's what the attachments
            # summary is for.
            self.run_agent_turn_with_parts(parts)
            return
        # Plain text path: nothing to expand, send as-is.
        self.run_agent_turn(text)

    def _handle_key(self, ch) -> None:
        if isinstance(ch, str):
            # Both \r and \n submit (Enter). The terminal decides
            # which one it sends for the Enter key; we don't try to
            # distinguish a "literal newline" from "Enter". Newlines
            # are inserted via Shift+Enter (\x1b\r or \x1b\n) or via
            # a bracketed paste (handled below).
            #
            # Exception: when the @-file menu is open, Enter completes
            # the highlighted file (just like Tab) instead of
            # submitting. The user explicitly navigates to a file and
            # presses Enter to insert it; pressing Enter again on the
            # next line submits. This matches Cursor / Zed / VS Code.
            if ch == "\r" or ch == "\n":
                if self._file_menu_active():
                    if self._file_menu_complete():
                        self._file_menu_keep_selection()
                    return
                self._submit_editor()
                return
            if ch == "\x1b":
                # Bare Esc (no following sequence) closes any open
                # picker. Priority: file menu first (it's the more
                # transient one — the user is mid-mention), then
                # slash menu, then a pending ``/edit`` (cancel
                # the edit and restore the previous message).
                seq = _read_escape_seq(self.stdscr)
                if seq == "":
                    if self._file_menu_active():
                        # Drop the active mention from the buffer
                        # so the user starts fresh. Without this
                        # the next keypress would re-activate the
                        # menu immediately, which is confusing.
                        self._dismiss_file_menu()
                        return
                    if self._menu_active():
                        self.menu_selected = 0
                        return
                    if self._goal_edit_in_progress:
                        self._goal_edit_in_progress = False
                        self.editor.buf = [""]
                        self.editor.row = 0
                        self.editor.col = 0
                        self.push("note", "  goal unchanged", self.A_DIM)
                        self.render()
                        return
                    if self._edit_in_progress:
                        # Cancel the edit. The previous turn is
                        # still in the history; we just need to
                        # clear the editor and the flag.
                        self._edit_in_progress = False
                        self.editor.buf = [""]
                        self.editor.row = 0
                        self.editor.col = 0
                        self.push(
                            "note",
                            "  edit cancelled",
                            self.A_DIM,
                        )
                        self.render()
                        return
                self._handle_escape_seq(seq)
                return
            if ch == "\x03":  # Ctrl-C
                if not self.editor.is_empty():
                    self.editor.buf = [""]
                    self.editor.row = 0
                    self.editor.col = 0
                else:
                    raise KeyboardInterrupt
                return
            if ch == "\x04":  # Ctrl-D
                if self.editor.is_empty():
                    raise KeyboardInterrupt
                return
            if ch in ("\x7f", "\x08"):  # Backspace / DEL
                self.editor.backspace()
                # Selection may now point past the end of the match list.
                self._menu_keep_selection()
                self._file_menu_keep_selection()
                return
            if ch == "\x15":  # Ctrl+U
                self.editor.clear_line()
                return
            if ch == "\x0b":  # Ctrl+K
                line = self.editor.buf[self.editor.row]
                self.editor.buf[self.editor.row] = line[: self.editor.col]
                return
            if ch == "\x17":  # Ctrl+W
                line = self.editor.buf[self.editor.row]
                i = self.editor.col
                while i > 0 and line[i - 1].isspace():
                    i -= 1
                while i > 0 and not line[i - 1].isspace():
                    i -= 1
                self.editor.buf[self.editor.row] = line[:i] + line[self.editor.col:]
                self.editor.col = i
                return
            if ch == "\x01":  # Ctrl+A
                self.editor.move_home()
                return
            if ch == "\x05":  # Ctrl+E
                self.editor.move_end()
                return
            if ch == "\x12":  # Ctrl+R — cyclic history prev
                if self.editor.history:
                    if self.editor.h_idx > 0:
                        self.editor.h_idx -= 1
                        self.editor._load_history()
                return
            if ch == "\x0e":  # Ctrl+N — history next
                if self.editor.h_idx < len(self.editor.history):
                    self.editor.h_idx += 1
                    self.editor._load_history()
                return
            if ch == "\x0c":  # Ctrl+L (redraw)
                self.render()
                return
            if ch == "\x07":  # Ctrl+G — dismiss any open picker
                if self._file_menu_active():
                    self._dismiss_file_menu()
                elif self._menu_active():
                    self.menu_selected = 0
                return
            if ch == "\t":  # Tab — complete the active picker
                if self._file_menu_active():
                    if self._file_menu_complete():
                        self._file_menu_keep_selection()
                    return
                if self._menu_active():
                    self._menu_complete()
                    return
            if ch and ch.isprintable():
                self.editor.insert_char(ch)
                self._menu_keep_selection()
                self._file_menu_keep_selection()
                return
        else:
            # int key code from curses
            if ch == curses.KEY_UP:
                if self._file_menu_active():
                    self._file_menu_move(-1)
                elif self._menu_active():
                    self._menu_move(-1)
                else:
                    self.editor.move_up()
            elif ch == curses.KEY_DOWN:
                if self._file_menu_active():
                    self._file_menu_move(1)
                elif self._menu_active():
                    self._menu_move(1)
                else:
                    self.editor.move_down()
            elif ch == curses.KEY_LEFT:
                self.editor.move_left()
            elif ch == curses.KEY_RIGHT:
                self.editor.move_right()
            elif ch == curses.KEY_HOME:
                self.editor.move_home()
            elif ch == curses.KEY_END:
                self.editor.move_end()
            elif ch == curses.KEY_DC:
                self.editor.delete_forward()
            elif ch == curses.KEY_BACKSPACE or ch == 127:
                self.editor.backspace()
            elif ch == curses.KEY_NPAGE:
                self.scroll = max(0, self.scroll - 1)
            elif ch == curses.KEY_PPAGE:
                # Render clamps the upper bound (the wrap cache means
                # final_lines can be much longer than len(self.log)).
                self.scroll += 1
            elif ch == curses.KEY_RESIZE:
                # Curses re-reads getmaxyx on the next render automatically.
                pass
            elif ch == curses.KEY_MOUSE:
                try:
                    _id, _x, _y, _z, _bstate = curses.getmouse()
                except Exception:
                    return
                # BUTTON4 = wheel up (back in history); BUTTON5 = wheel
                # down (toward latest). The upper bound is enforced by
                # render() because it depends on wrapped-line count, not
                # log-entry count.
                if _bstate & curses.BUTTON4_PRESSED:
                    self.scroll += 3
                elif _bstate & curses.BUTTON5_PRESSED:
                    self.scroll = max(0, self.scroll - 3)
                return
            elif ch == curses.KEY_ENTER or ch == 10 or ch == 13:
                if self._file_menu_active():
                    if self._file_menu_complete():
                        self._file_menu_keep_selection()
                    return
                self._submit_editor()
            # Unknown KEY_* — ignore.

    def _handle_escape_seq(self, seq: str) -> None:
        if not seq:
            return
        if seq == "paste_start":
            pasted = _read_paste(self.stdscr)
            if pasted:
                if pasted.endswith("\n"):
                    pasted = pasted[:-1]
                if pasted.endswith("\r"):
                    pasted = pasted[:-1]
                self.editor.insert_text(pasted)
            return
        if seq == "image_paste":
            # A Kitty / iTerm2 image sequence was just consumed
            # by _read_escape_seq. The decoded bytes + extension
            # are stashed on the reader function as a function
            # attribute; we pick them up here. Multi-chunk
            # pastes arrive as several consecutive "image_paste"
            # tokens, so we accumulate the chunks in the
            # module-level _ImageChunk state. A chunk is the
            # "last" one when the parse function reports the
            # absence of an "m=1" continuation flag.
            data, ext = _read_image_paste._last  # type: ignore[attr-defined]
            _read_image_paste._last = None  # type: ignore[attr-defined]
            self._accumulate_image_chunk(data, ext)
            return
        if seq == "image_paste_failed":
            # The escape sequence looked like an image paste
            # (ESC _ or ESC ]) but the parser couldn't decode
            # it. Most likely causes: empty paste, terminal
            # stripped the data, or a non-image sequence that
            # happens to start with the same byte. The user
            # gets a hint so they know to try /paste instead.
            self.push("note",
                      "image paste not recognised "
                      "(try /paste or check your terminal config)",
                      self.A_DIM)
            self.render()
            return
        if seq == "alt_enter":
            self.editor.newline()
        elif seq == "alt_v":
            # Alt+V: same as the /paste slash command. Useful
            # when the user is mid-message and doesn't want to
            # break the typing flow with a slash command. We
            # call the same handler so the behaviour stays
            # identical (push a note, insert a short ID).
            result = self._cmd_paste("")
            if result:
                self.push("note", result, self.A_DIM)
                self.render()
        elif seq == "up":
            self.editor.move_up()
        elif seq == "down":
            self.editor.move_down()
        elif seq == "left":
            self.editor.move_left()
        elif seq == "right":
            self.editor.move_right()
        elif seq == "home":
            self.editor.move_home()
        elif seq == "end":
            self.editor.move_end()
        elif seq == "delete":
            self.editor.delete_forward()
        elif seq == "ctrl_up":
            self.editor.move_up()
        elif seq == "ctrl_down":
            self.editor.move_down()
        elif seq == "ctrl_left":
            self.editor.move_left()
        elif seq == "ctrl_right":
            self.editor.move_right()
        elif seq == "page_up":
            # render() clamps the upper bound against final_lines.
            self.scroll += 1
        elif seq == "page_down":
            self.scroll = max(0, self.scroll - 1)
        # Unknown sequences (F1-F12, bare Esc, etc.) are ignored.


# --- helpers used by the TUI ------------------------------------------------


def _poll_esc(stdscr) -> bool:
    """Non-blocking check for a bare Esc key. Returns True once, then arms
    for the next bare Esc (consuming any escape sequence lead bytes that
    happen to be sitting in the buffer)."""
    stdscr.nodelay(True)
    try:
        ch = stdscr.get_wch()
    except Exception:
        ch = -1
    finally:
        stdscr.nodelay(False)
    if ch == -1 or ch == "":
        return False
    if ch == "\x1b":
        # Could be bare Esc OR start of an escape sequence (arrows, etc.).
        # Read the next byte with a tiny timeout to disambiguate.
        stdscr.nodelay(True)
        try:
            n2 = stdscr.get_wch()
        except Exception:
            n2 = -1
        finally:
            stdscr.nodelay(False)
        if n2 == -1 or n2 == "":
            return True  # bare Esc
        # It's a sequence — consume the rest (best-effort) and return False.
        stdscr.nodelay(True)
        try:
            while True:
                more = stdscr.get_wch()
                if more == -1 or more == "":
                    break
        except Exception:
            pass
        finally:
            stdscr.nodelay(False)
        return False
    return False


def _read_escape_seq(stdscr) -> str:
    """Read a CSI/SS3 sequence after a leading \\x1b and return a token name.

    Handles common sequences: arrows, Home/End, Delete, Page Up/Down,
    Alt+Enter, and the bracketed-paste start marker (\\x1b[200~). Returns
    '' for unknown sequences.

    Also peeks for the Kitty Graphics Protocol (``\\x1b_G``) and iTerm2
    image sequence (``\\x1b]1337;File=``) starts. When it sees one of
    those, it consumes the entire sequence (possibly many chunks) and
    returns the special token ``"image_paste"`` so the caller can
    route the buffered image to the file picker.

    Bracketed-paste mode wraps the entire image sequence, so a normal
    Kitty paste looks like ``\\x1b[200~\\x1b_G...\\x1b\\\\\\x1b[201~``
    on the wire; in that case the ``paste_start`` token takes
    precedence and the image is read as part of the paste.
    """
    stdscr.nodelay(True)
    try:
        n2 = stdscr.get_wch()
    except Exception:
        n2 = -1
    if n2 == -1 or n2 == "":
        return "esc"  # bare Esc
    if n2 == "\r" or n2 == "\n":
        return "alt_enter"
    if n2 in ("[", "O"):
        # CSI / SS3 sequence.
        buf = []
        while True:
            try:
                c = stdscr.get_wch()
            except Exception:
                c = -1
            if c == -1 or c == "":
                break
            buf.append(c if isinstance(c, str) else chr(c))
            ch = buf[-1]
            if ch.isalpha() or ch in ("~", "^"):
                break
        s = "".join(buf)
        # Bracketed paste start — the caller switches to paste mode and
        # calls _read_paste() to slurp the rest.
        if s == "[200~":
            return "paste_start"
        return {
            "[A": "up", "[B": "down", "[C": "right", "[D": "left",
            "[H": "home", "[F": "end",
            "[1~": "home", "[7~": "home",
            "[4~": "end", "[8~": "end",
            "[3~": "delete", "[2~": "insert",
            "[5~": "page_up", "[6~": "page_down",
        }.get(s, "")
    # Kitty Graphics Protocol: ESC _G <params> ; <base64> ESC \\
    # The first byte after ESC is "_" (literal underscore). Read
    # everything up to the next ESC and let the dedicated reader
    # decode it.
    if n2 == "_" or n2 == "]":
        # Two flavors:
        #   _G ... (Kitty Graphics)
        #   ] ...  (OSC, used by iTerm2 for images)
        # Both are dispatched to the same reader which recognises
        # the start byte we already consumed.
        stdscr.nodelay(False)
        result = _read_image_paste(stdscr, n2)
        if result is not None:
            # Stash the most recently received image on the
            # function attribute so _handle_escape_seq can pick
            # it up. We use a function attribute rather than
            # threading a return value through the input loop
            # to keep _read_escape_seq's signature stable.
            _read_image_paste._last = result  # type: ignore[attr-defined]
            return "image_paste"
        # The reader returned None — either the sequence was
        # malformed or it wasn't actually an image. Surface a
        # brief note so the user knows we got *something* but
        # couldn't make sense of it (the alternative is silent
        # failure, which is what was happening before).
        _read_image_paste._last = ("", "png")  # type: ignore[attr-defined]
        return "image_paste_failed"
    # Alt+letter: \x1b<letter>
    if isinstance(n2, str) and n2.isprintable():
        return f"alt_{n2}"
    return ""


# Multi-chunk image-paste state. The Kitty protocol can split a
# single image across many ``\\x1b_G`` sequences (each ``m=1``
# except the last ``m=0``), so we accumulate chunks here. The
# (data, ext) tuple is None when no image is being received; the
# ``pending`` flag tells the input loop whether the next
# "image_paste" token should be appended to the accumulator or
# replace it (i.e. a fresh image).
_ImageChunk: tuple[list[str], str] | None = None

# Format from the first chunk of a multi-chunk Kitty paste. The
# subsequent chunks only carry the m= flag and the base64 data;
# the action / format / transmission-mode are implied from the
# first chunk. Stashing them here lets :func:`_parse_kitty_graphics`
# recognise a continuation chunk even though its params dict is
# almost empty.
_KittyContinuationState: dict | None = None


def _read_image_paste(stdscr, start_byte: str) -> tuple[bytes, str] | None:
    """Read a Kitty / iTerm2 image paste sequence and decode it.

    Returns ``(raw_bytes, file_extension)`` on success, ``None`` if
    the sequence is malformed or not actually an image. ``start_byte``
    is the byte AFTER the leading ``\\x1b`` — ``"_"`` for Kitty
    (``\\x1b_G``) or ``"]"`` for OSC (iTerm2 ``\\x1b]1337;File=...``).

    The reader is permissive: it ignores chunks it doesn't
    understand, returns ``None`` on any parse failure, and bounds
    the total payload size to avoid being DoS'd by a misbehaving
    terminal. A real image paste is bounded by
    :data:`MAX_IMAGE_BYTES`; anything bigger is rejected (the
    terminal can be configured to cap the size, but the model
    can only digest so many pixels anyway).

    Multi-chunk pastes are reassembled at the caller level — this
    function returns a single chunk's worth of bytes + the
    extension, and :meth:`_TUIState._handle_escape_seq` stitches
    them together across consecutive ``"image_paste"`` tokens
    using the :data:`_ImageChunk` module-level state.
    """
    # Stay in blocking mode — a paste can be many KB and we
    # don't want a tiny timeout aborting us partway through.
    stdscr.nodelay(False)
    buf: list[str] = []
    while True:
        try:
            c = stdscr.get_wch()
        except KeyboardInterrupt:
            raise
        except Exception:
            return None
        if c == -1 or c == "":
            return None
        ch = c if isinstance(c, str) else chr(c)
        # The terminator is ESC followed by '\\' (which we read
        # as two chars). We do NOT try to handle the case where
        # the terminal sends ESC alone as a terminator — that's
        # only used for very short control messages, not image
        # data.
        if ch == "\x1b":
            try:
                c2 = stdscr.get_wch()
            except Exception:
                return None
            n = c2 if isinstance(c2, str) else chr(c2) if c2 != -1 else ""
            if n == "\\":
                break
            # Anything else after ESC inside a payload is malformed.
            return None
        buf.append(ch)
        # Safety net: cap the raw buffer at 8MB. A real PNG is
        # usually well under 1MB, so this is generous. Anything
        # larger is almost certainly a misbehaving terminal.
        if sum(len(s) for s in buf) > _MAX_IMAGE_BYTES * 4:
            return None
    raw = "".join(buf)
    if start_byte == "_":
        return _parse_kitty_graphics(raw)
    if start_byte == "]":
        return _parse_iterm2_image(raw)
    return None


# Format codes from the Kitty Graphics Protocol. The spec is at
# https://sw.kovidgoyal.net/kitty/graphics-protocol/ — we only
# handle the subset commonly seen in clipboard pastes.
_KITTY_FORMAT_TO_EXT: dict[str, str] = {
    "24": "rgb",
    "32": "rgba",
    "100": "png",
}


def _parse_kitty_graphics(payload: str) -> tuple[bytes, str] | None:
    """Parse a Kitty ``\\x1b_G...`` payload (already-stripped).

    The format is ``<key>=<val>,<key>=<val>;<base64-chunk>``. The
    base64 chunk is the part after the first ``;``. For multi-
    chunk pastes, the first message has ``m=1`` (more) and
    subsequent messages have only ``m=0`` or no ``m`` key plus a
    payload — we don't see those here because each chunk arrives
    as its own ``\\x1b_G`` sequence. This function therefore only
    handles the first chunk, but :func:`_read_image_paste` is
    called per chunk, so a multi-chunk paste is reassembled by
    the caller in :meth:`_handle_escape_seq`.

    Continuation chunks (``m=0;B64DATA``) reuse the format
    recorded from the first chunk via the module-level
    :data:`_KittyContinuationState` — the per-chunk params dict
    doesn't carry the format / action so we'd otherwise reject
    them.

    For multi-chunk pastes, the per-chunk base64 is incomplete
    and ``base64.b64decode`` would fail on it (the byte count
    must be a multiple of 4). We therefore return the raw b64
    text on intermediate chunks, and only the FINAL chunk is
    decoded (with the previously-accumulated b64 stitched in).
    The decoder chain in :meth:`_TUIState._accumulate_image_chunk`
    does the stitching.
    """
    global _KittyContinuationState
    if ";" not in payload:
        return None
    params, b64_chunk = payload.split(";", 1)
    kv: dict[str, str] = {}
    for part in params.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    # Continuation chunk: no action key, but the global
    # continuation state is set from a previous m=1 chunk.
    if "a" not in kv and _KittyContinuationState is not None:
        ext = _KittyContinuationState["ext"]
        if not b64_chunk:
            return None
        # Per-chunk flag: m=1 means MORE chunks follow, m=0
        # (or absent) means this is the LAST one. Reset the
        # global state on the last chunk.
        more = kv.get("m") == "1"
        if not more:
            _KittyContinuationState = None
            try:
                del _read_image_paste._more_chunks
            except AttributeError:
                pass
        else:
            _read_image_paste._more_chunks = True  # type: ignore[attr-defined]
        # Return the raw b64 text (not decoded) — the
        # accumulator concatenates the chunks and only the
        # last one triggers a base64 decode. We return
        # ``b64_chunk.encode()`` as the "bytes" so the
        # accumulator's b"".join() works regardless of
        # whether the chunk is raw b64 text or actual bytes.
        return b64_chunk.encode("ascii"), ext
    # First chunk (or single-chunk) — must have a=T and t=d.
    action = kv.get("a", "")
    if action != "T":
        # If a stray sequence arrives while we're NOT in a
        # continuation, drop the stale state so a fresh
        # paste doesn't inherit a wrong format.
        _KittyContinuationState = None
        return None
    if kv.get("t", "d") != "d":
        _KittyContinuationState = None
        return None
    fmt = kv.get("f", "100")
    ext = _KITTY_FORMAT_TO_EXT.get(fmt, "png")
    more = kv.get("m") == "1"
    if not b64_chunk:
        return None
    if more:
        # Record the format so the next continuation chunk
        # can match it.
        _KittyContinuationState = {"ext": ext}
        _read_image_paste._more_chunks = True  # type: ignore[attr-defined]
        # Return raw b64 text — the accumulator will
        # concatenate the chunks.
        return b64_chunk.encode("ascii"), ext
    # Single chunk — decode it now.
    _KittyContinuationState = None
    try:
        chunk = base64.b64decode(b64_chunk, validate=False)
    except Exception:
        return None
    if not chunk:
        return None
    # Make sure the flag is reset so a subsequent
    # single-chunk paste after a multi-chunk one isn't
    # held open waiting for a non-existent next chunk.
    try:
        del _read_image_paste._more_chunks
    except AttributeError:
        pass
    return chunk, ext


def _parse_iterm2_image(payload: str) -> tuple[bytes, str] | None:
    """Parse an iTerm2 ``\\x1b]1337;File=...`` payload.

    Format: ``File=name=<b64>;size=<n>;<flags>:<b64-data>``. The
    ``size`` field is in bytes of the DECODED file; the b64
    payload is what we want. We only handle inline (i.e. we
    receive the bytes directly) — the ``preserveAspectRatio``
    and similar display hints are ignored.
    """
    # Strip the "File=" prefix and the trailing ":" separator.
    if not payload.startswith("File="):
        return None
    body = payload[len("File="):]
    if ":" not in body:
        return None
    params_str, b64_data = body.rsplit(":", 1)
    kv: dict[str, str] = {}
    for part in params_str.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    if "size" not in kv:
        return None
    try:
        size = int(kv["size"])
    except ValueError:
        return None
    # Reject anything larger than MAX_IMAGE_BYTES up front — the
    # b64 below will also be that big.
    if size > _MAX_IMAGE_BYTES:
        return None
    try:
        decoded_name = base64.b64decode(kv.get("name", "")).decode(
            "ascii", errors="replace"
        ) if kv.get("name") else ""
    except Exception:
        decoded_name = ""
    # Pick a sensible extension from the (decoded) name. Default
    # to png since that's what iTerm2 uses when no name is
    # supplied.
    ext = "png"
    if "." in decoded_name:
        candidate = decoded_name.rsplit(".", 1)[-1].lower().lstrip(".")
        if candidate in {e.lstrip(".") for e in _IMAGE_EXTS}:
            ext = "jpg" if candidate == "jpeg" else candidate
    try:
        data = base64.b64decode(b64_data, validate=False)
    except Exception:
        return None
    if not data:
        return None
    return data, ext


def _read_paste(stdscr) -> str:
    """Read characters verbatim until the bracketed-paste terminator \\x1b[201~.

    Called after _read_escape_seq has returned "paste_start" (i.e. the
    leading \\x1b[200~ has already been consumed). Returns the pasted
    text, exclusive of the terminator. Empty string if the terminator
    never arrives (shouldn't happen with a well-behaved terminal).
    """
    # Switch back to blocking mode — a paste can be many MB and we don't
    # want to time out partway through.
    stdscr.nodelay(False)
    out: list[str] = []
    while True:
        try:
            c = stdscr.get_wch()
        except KeyboardInterrupt:
            raise
        except Exception:
            break
        if c == -1 or c == "":
            break
        ch = c if isinstance(c, str) else chr(c)
        # Watch for the terminator \x1b[201~. We have to match it across
        # multiple reads; the leading \x1b is unambiguous (paste content
        # almost never contains a bare \x1b), and the trailing [201~ is
        # unlikely in real text. If the match fails, fall through and
        # append the bytes to the buffer (so we don't lose data).
        if ch == "\x1b":
            tail = _try_match_terminator(stdscr)
            if tail is not None:
                # Terminator consumed. We're done.
                break
            # Not the terminator. _try_match_terminator returns the
            # bytes it consumed so we can re-insert them.
            out.append("\x1b")
            out.extend(tail or [])
            continue
        # Normalize line endings as we go: CR or CRLF both become LF.
        # This keeps pasted text consistent regardless of whether the
        # terminal/shell converts to \n or \r\n at the boundary.
        if ch == "\r":
            out.append("\n")
        elif ch == "\n":
            out.append("\n")
        else:
            out.append(ch)
    return "".join(out)


def _try_match_terminator(stdscr) -> list[str] | None:
    """After a leading \\x1b, try to read '[201~'. Returns None on match,
    or the list of bytes consumed (to be re-inserted) on mismatch.

    Best-effort: there's no real pushback in curses, so we have to
    consume the bytes either way. The terminator is short and unlikely
    to appear in real pasted text.
    """
    expected = "[201~"
    got: list[str] = []
    for want in expected:
        try:
            c = stdscr.get_wch()
        except Exception:
            return got or None  # nothing more to read — assume match? be safe
        if c == -1 or c == "":
            return got or None
        ch = c if isinstance(c, str) else chr(c)
        got.append(ch)
        if ch != want:
            return got  # mismatch — caller re-inserts
    return None  # full match


def _confirm_key(stdscr, prompt: str) -> bool:
    """Single-keypress y/N/esc prompt drawn into the log area."""
    h, w = stdscr.getmaxyx()
    row = max(0, h - 3)
    stdscr.addnstr(row, 0, " " * max(0, w - 1), max(0, w - 1), curses.A_BOLD)
    stdscr.addnstr(row, 0, prompt, max(0, w - 1), curses.A_BOLD)
    stdscr.refresh()
    while True:
        try:
            ch = stdscr.get_wch()
        except KeyboardInterrupt:
            return False
        except curses.error:
            # Signal (SIGWINCH) interrupted the read — redraw and
            # retry so the prompt doesn't vanish when the user resizes.
            try:
                stdscr.addnstr(row, 0, prompt, max(0, w - 1), curses.A_BOLD)
                stdscr.refresh()
            except Exception:
                pass
            continue
        if isinstance(ch, str):
            cl = ch.lower()
            if cl == "y":
                return True
            if cl in ("n", "d", "\x03", "\x04"):
                return False
            if ch == "\x1b":
                # Bare Esc → cancel the prompt + return False (the agent
                # already routes Esc at approval into TURN_ESC).
                return False
        # Ignore other keys.


def _init_approval_level() -> str:
    """Resolve the default approval level from env + CLI args.

    Precedence (highest first):
      1. --yolo / --approval on sys.argv
      2. ANDURIL_APPROVAL / ANDURIL_YOLO env var
      3. safe default: prompt for dangerous tools
    """
    for i, a in enumerate(sys.argv):
        if a == "--approval" and i + 1 < len(sys.argv):
            kind, level = _normalize_approval(sys.argv[i + 1])
            if kind == "yolo":
                return "yolo"
            if kind == "prompt_all":
                return "all"
            if kind == "threshold":
                return level
        if a == "--yolo":
            return "yolo"
    if _env_str("ANDURIL_YOLO").lower() in ("1", "true", "yes"):
        return "yolo"
    kind, level = _normalize_approval(_env_str("ANDURIL_APPROVAL"))
    if kind == "yolo":
        return "yolo"
    if kind == "prompt_all":
        return "all"
    if kind == "threshold":
        return level
    # Safe default: prompt for dangerous tools (level "all").
    return "all"
