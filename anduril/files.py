"""File scanning, fuzzy matching, MIME detection, and `@`-mention expansion.

Two features live here:

* :func:`list_files` walks a directory tree, honoring a sensible default
  ignore-list (``.git``, ``__pycache__``, ``node_modules``, …) and a
  depth / count cap so a runaway filesystem (e.g. ``/``) can't blow up
  the TUI's file-picker.

* :func:`fuzzy_match` ranks candidate paths against a query. The
  scoring is deliberately simple: an in-order subsequence match with
  bonuses for prefix / word-boundary / consecutive runs. No external
  dependency — the project ships with a "one runtime dep: openai"
  rule, and we want to keep it that way.

* :func:`expand_mentions` parses a user-typed buffer for ``@path``
  references. Text files get inlined as ``[file: …]`` fenced blocks;
  image files become base64 ``image_url`` parts. The result is the
  OpenAI multimodal ``content`` list that the agent passes straight
  to the chat-completions endpoint.

Path resolution
---------------

A mention like ``@src/main.py`` is resolved relative to the current
working directory by default. An absolute path (``@/etc/hosts``) is
used as-is. The parser ignores ``@`` characters that look like
they're part of an email or identifier (the preceding char must not
be alphanumeric or ``_``), so ``user@example.com`` is left alone.
"""

from __future__ import annotations

import base64
import os
import pathlib
import platform
import re
import shutil
import subprocess
import time
from typing import Any, Iterable


# === Path collection ======================================================


# Directories we always skip on a walk. Configurable per call, but most
# callers want the default — a vendored ``node_modules`` or ``.git``
# directory will make every fuzzy-match latency awful and adds nothing
# the model can usefully reference.
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".ruff_cache",
    ".venv", "venv", "env", ".env", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target", "build", "dist", ".next", ".nuxt", "out",
    ".idea", ".vscode", ".DS_Store", "*.egg-info",
    ".cache", ".parcel-cache", ".turbo", ".svelte-kit",
})

# Default file-extension allow-list for "interesting" files. We do NOT
# use this to filter ``list_files`` (the model often wants to read
# weird extensions like ``.proto`` or ``.toml``), only for the rough
# text/binary heuristic in :func:`is_text_file`.
TEXT_EXT_HINT: frozenset[str] = frozenset({
    ".py", ".pyi", ".pyx", ".pyw", ".ipynb",
    ".md", ".markdown", ".rst", ".txt", ".adoc",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".htm", ".xml", ".css", ".scss", ".less",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
    ".rs", ".go", ".java", ".kt", ".scala", ".clj",
    ".rb", ".php", ".pl", ".lua", ".r", ".jl",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql", ".proto", ".thrift",
    ".diff", ".patch", ".log", ".csv", ".tsv",
    ".tex", ".bib", ".org",
    ".vim", ".el", ".lisp", ".cl", ".scm",
    ".dockerfile", ".gitignore", ".gitattributes", ".editorconfig",
    ".envrc", ".env",
})

# File extensions we treat as images for multimodal uploads. Anything
# not in this set is left as a literal mention (the user can still
# reference it in text, but the model won't see the file).
IMAGE_EXTS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
})

# Maximum image size we'll inline as a base64 data URL. 8MB is enough
# for any reasonable screenshot / diagram and well under the per-image
# 20MB ceiling most providers enforce. Files larger than this are
# left as a literal mention — the user can shrink them externally.
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

# Maximum characters we read from a text file when inlining it as a
# mention. The chat template typically has a 200K-token context window;
# 200K characters ≈ 50-60K tokens which is well within reason, and
# the truncation marker tells the model "there's more, ask if needed".
MAX_TEXT_CHARS: int = 200_000


def _ignored_dir(name: str) -> bool:
    """Match a directory name against the default ignore patterns.

    The patterns are exact directory names (``__pycache__``) or glob
    prefixes (``*.egg-info``). A path with the prefix is enough — the
    walk never recurses into ``foo.egg-info`` because it doesn't
    actually match the glob (a real egg-info dir is named
    ``Foo-1.0.egg-info``).
    """
    if name in DEFAULT_IGNORE_DIRS:
        return True
    for pat in DEFAULT_IGNORE_DIRS:
        if "*" in pat and pat.startswith("*"):
            if name.endswith(pat[1:]):
                return True
    return False


def list_files(
    root: str | pathlib.Path | None = None,
    max_depth: int = 8,
    max_count: int = 2000,
    ignore: Iterable[str] | None = None,
) -> list[pathlib.Path]:
    """Walk ``root`` and return files as paths relative to it.

    Stops the walk as soon as ``max_count`` files are collected, or
    when the current recursion depth exceeds ``max_depth`` (the depth
    is measured from ``root``; ``max_depth=8`` covers
    ``root/a/b/c/d/e/f/g/``).

    Symlinks are not followed — a recursive symlink would otherwise
    loop forever. Symlinks to single files are reported as their
    target (so ``link`` and ``target`` both appear in the picker).

    The order is filesystem-defined; callers that want a stable order
    should sort the result.
    """
    base = pathlib.Path(root) if root else pathlib.Path.cwd()
    base = base.resolve()
    ignore_set = set(ignore) if ignore is not None else set(DEFAULT_IGNORE_DIRS)

    out: list[pathlib.Path] = []
    # The walk is iterative (manual stack) rather than recursive so
    # we can short-circuit as soon as we hit max_count, and so a
    # very deep tree doesn't blow Python's recursion limit.
    stack: list[tuple[pathlib.Path, int]] = [(base, 0)]
    while stack and len(out) < max_count:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if entry.name in ignore_set or _ignored_dir(entry.name):
                continue
            try:
                if entry.is_dir():
                    if entry.is_symlink():
                        continue  # don't follow recursive symlinks
                    stack.append((entry, depth + 1))
                elif entry.is_file():
                    try:
                        rel = entry.relative_to(base)
                    except ValueError:
                        rel = entry  # shouldn't happen after resolve, but…
                    out.append(rel)
            except OSError:
                continue
            if len(out) >= max_count:
                break
    return out


def is_image(path: str | pathlib.Path) -> bool:
    """True if the file's extension is a known image format."""
    suffix = pathlib.Path(path).suffix.lower()
    return suffix in IMAGE_EXTS


def is_text_file(path: str | pathlib.Path, sample_bytes: int = 4096) -> bool:
    """Heuristic: True if the file looks like text (UTF-8 / ASCII / …).

    Strategy: read the first ``sample_bytes`` and try to decode as
    UTF-8. If it decodes AND contains no NUL bytes (a strong
    "binary" signal that catches most compiled / packed formats),
    it's text. This isn't perfect — a binary file with no NULs in
    the first 4K would slip through — but combined with the
    extension hint it's good enough for the mention-expander.
    """
    if is_image(path):
        return False
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in TEXT_EXT_HINT:
        return True
    if suffix:  # unknown extension with a dot: assume binary
        return False
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_bytes)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def read_image_data_url(
    path: str | pathlib.Path,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> tuple[str, int]:
    """Read an image file and return ``(data_url, byte_count)``.

    The data URL is in the format ``data:image/<ext>;base64,<...>``,
    which is what the OpenAI multimodal endpoint accepts in the
    ``image_url.url`` field.

    Raises :class:`FileNotFoundError`, :class:`PermissionError`, or
    :class:`ValueError` (when the file is larger than ``max_bytes``).
    """
    p = pathlib.Path(path)
    size = p.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"image {p} is {size} bytes; refusing to inline (limit {max_bytes})"
        )
    suffix = p.suffix.lower().lstrip(".")
    # Map jpeg -> jpeg, jpg -> jpeg, etc.
    if suffix == "jpg":
        suffix = "jpeg"
    mime = f"image/{suffix}"
    with open(p, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}", size


def read_text_file(
    path: str | pathlib.Path,
    max_chars: int = MAX_TEXT_CHARS,
) -> str:
    """Read a text file with a hard cap, returning the content as a string.

    Truncates with a clear marker so the model can tell the file was
    cut off. Raises :class:`UnicodeDecodeError` if the file isn't
    decodable as UTF-8 — the caller's :func:`is_text_file` check
    should have caught that, but we don't want to silently return
    garbage.
    """
    p = pathlib.Path(path)
    with open(p, "r", encoding="utf-8", errors="strict") as f:
        text = f.read(max_chars + 1)
    if len(text) > max_chars:
        text = text[:max_chars] + (
            f"\n[... file truncated at {max_chars} chars; "
            f"total {p.stat().st_size} bytes on disk]"
        )
    return text


# === Pasted-image storage =================================================


# Where pasted images are stored. The directory is created lazily on
# the first paste. Each file is named ``image-YYYYMMDD-HHMMSS-NNN.<ext>``
# so multiple pastes within the same second don't clobber each other
# and the user can see when each image was added. The directory is
# shared across sessions, so a user who restarts the TUI mid-task
# won't lose the references to previously-pasted images.
PASTED_IMAGE_DIR: pathlib.Path = (
    pathlib.Path.home() / ".local" / "state" / "anduril" / "images"
)


def _ensure_pasted_dir() -> pathlib.Path:
    """Create the image directory if it doesn't exist yet.

    Returns the path. Cached on the function attribute so a long
    session doesn't hit ``stat()`` on every paste.
    """
    cached = getattr(_ensure_pasted_dir, "_cached", None)
    if cached is not None:
        return cached
    PASTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_pasted_dir._cached = PASTED_IMAGE_DIR  # type: ignore[attr-defined]
    return PASTED_IMAGE_DIR


# === Clipboard reading ====================================================


# Path lookup for the platform-specific clipboard tools. The
# /paste command probes the available tool at call time so an
# uninstalled binary doesn't break the rest of the TUI.
#
# Linux is special: a single user can be on Wayland OR X11 (or
# have both display servers wired up), and the relevant env
# var (``XDG_SESSION_TYPE``) isn't always set correctly inside
# child processes — for example, when anduril is launched from
# a terminal that was itself launched from a different DE. So
# we keep a LIST of tools to try, in preference order, and let
# the available-binary check at call time decide which one
# actually runs. The Wayland tool is tried first when
# ``WAYLAND_DISPLAY`` is set, then X11's ``xclip``; if neither
# env var is set, we default to ``wl-paste`` (it's the more
# common modern choice on current distros).
_CLIPBOARD_TOOLS: dict[str, list[list[str]]] = {
    "darwin": [[
        "osascript", "-e",
        'set theFile to POSIX file "/tmp/anduril_clip.png"\n'
        'try\n'
        '  set theImage to (the clipboard as «class PNGf»)\n'
        '  set theFileRef to open for access theFile with write permission\n'
        '  set eof of theFileRef to 0\n'
        '  write theImage to theFileRef\n'
        '  close access theFileRef\n'
        '  return "ok"\n'
        'on error errMsg\n'
        '  return "no-image: " & errMsg\n'
        'end try',
    ]],
    # Linux: we try BOTH wl-paste and xclip, preferring whichever
    # matches the active display server. The selection is
    # deferred to _linux_tools_in_order() so we can re-evaluate
    # on every call (env vars can change between calls if the
    # user SSH's into a different session, etc.).
    "linux": [],  # populated dynamically
    "win32": [[
        "powershell", "-NoProfile", "-Command",
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
        "if ($img) { $img.Save('C:\\Windows\\Temp\\anduril_clip.png', "
        "[System.Drawing.Imaging.ImageFormat]::Png); 'ok' } "
        "else { 'no-image' }",
    ]],
}

# Canonical tool definitions for Linux. Order doesn't matter at
# the dict level — :func:`_linux_tools_in_order` decides the
# actual call order at runtime.
_LINUX_TOOL_CMDS: dict[str, list[str]] = {
    "wl-paste": ["wl-paste", "-t", "image/png"],
    "xclip":    ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
}


def _linux_tools_in_order() -> list[list[str]]:
    """Return the Linux clipboard commands in preference order.

    The first command that's on PATH wins. Preference order:

    1. The tool that matches the active display server
       (Wayland → ``wl-paste``, X11 → ``xclip``).
    2. The other one, in case the env var is wrong but the
       binary is installed.

    This is the most common case that broke the previous
    version: a user on Wayland whose ``XDG_SESSION_TYPE`` was
    not set, so we picked the X11 tool, which then wasn't
    installed, so /paste silently said "no image". Now we
    just look at what's actually on PATH.
    """
    if os.environ.get("WAYLAND_DISPLAY"):
        preferred, fallback = "wl-paste", "xclip"
    else:
        preferred, fallback = "xclip", "wl-paste"
    out = []
    if shutil.which(preferred):
        out.append(_LINUX_TOOL_CMDS[preferred])
    if shutil.which(fallback):
        out.append(_LINUX_TOOL_CMDS[fallback])
    return out


def _detect_platform_key() -> str | None:
    """Map platform.system() to a key in :data:`_CLIPBOARD_TOOLS`.

    Linux gets the special ``"linux"`` key, whose command list
    is built dynamically. macOS and Windows each have one
    tool. Anything else (FreeBSD, OpenBSD, …) returns None
    and ``/paste`` reports "unsupported platform".
    """
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    if system == "linux":
        return "linux"
    if system in ("windows", "win32"):
        return "win32"
    return None


def read_clipboard_image() -> tuple[bytes, str] | None:
    """Read an image from the system clipboard, if one is present.

    Returns ``(bytes, ext)`` on success, ``None`` if the clipboard
    has no image OR the platform tool isn't available. The
    /paste command turns ``None`` into a one-line status message
    so the user gets feedback either way.

    The platform tools we use:

    * macOS:    ``osascript`` (built-in)
    * Linux:    ``wl-paste`` or ``xclip`` (whichever is on PATH)
    * Windows:  PowerShell with System.Windows.Forms

    Each command is best-effort: a missing binary or a clipboard
    that contains no image both produce ``None`` rather than
    raising. Errors are caught and turned into ``None`` so the
    caller can show a friendly message.
    """
    key = _detect_platform_key()
    if key is None:
        return None
    if key == "linux":
        commands = _linux_tools_in_order()
    else:
        commands = _CLIPBOARD_TOOLS.get(key, [])
    for cmd in commands:
        # Quick check: is the binary on PATH? Saves us from
        # the slow failure mode of an OSError on every paste.
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            continue
        # macOS writes to a known file path; everything else
        # streams the image bytes to stdout.
        if key == "darwin":
            tmp = pathlib.Path("/tmp/anduril_clip.png")
            try:
                data = tmp.read_bytes()
            except OSError:
                continue
            try:
                tmp.unlink()
            except OSError:
                pass
        elif key == "win32":
            tmp = pathlib.Path("C:/Windows/Temp/anduril_clip.png")
            try:
                data = tmp.read_bytes()
            except OSError:
                continue
            try:
                tmp.unlink()
            except OSError:
                pass
        else:
            data = proc.stdout or b""
        if not data:
            continue
        # Pick the extension. We default to PNG because every
        # platform's clipboard image path produces PNG, and
        # that's what the multimodal endpoint accepts.
        ext = "png"
        if data[:3] == b"\xff\xd8\xff":
            ext = "jpg"
        elif data[:4] == b"GIF8":
            ext = "gif"
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            ext = "webp"
        return data, ext
    return None


def clipboard_tools_status() -> str:
    """Human-readable summary of which clipboard tools are available.

    Used by the TUI when ``/paste`` fails so the user can see
    exactly what we tried and what we expected. Returns a
    one-line string like ``"wl-paste"`` (only Wayland tool
    installed) or ``"wl-paste, xclip"`` (both) or ``"none"``
    (no tool, the user needs to install one).
    """
    if _detect_platform_key() == "linux":
        found = [name for name in _LINUX_TOOL_CMDS
                 if shutil.which(name)]
        if not found:
            return "none (install wl-clipboard or xclip)"
        return ", ".join(found)
    return "default"


def save_pasted_image(
    data: bytes,
    ext: str = "png",
) -> pathlib.Path:
    """Save a pasted image to the anduril images directory.

    ``data`` is the raw image bytes (already decoded from whatever
    the terminal sent, base64 or otherwise). ``ext`` is the file
    extension WITHOUT the dot — defaults to ``"png"`` because that's
    what most terminals send (Kitty/iTerm2 both default to PNG for
    "copy image to clipboard"). The extension is normalised to
    lowercase and validated against :data:`IMAGE_EXTS` so we don't
    end up with ``.exe`` files in the images directory.

    The filename is ``image-YYYYMMDD-HHMMSS-NNN.<ext>`` where
    ``NNN`` is a 3-digit counter that resets every second. The
    counter exists because a user can paste multiple images
    quickly (e.g. dragging three files into the terminal at
    once) and the timestamp alone would collide.

    Returns the saved path. Raises :class:`ValueError` if ``data``
    is empty, :class:`OSError` on filesystem failure.
    """
    if not data:
        raise ValueError("save_pasted_image: empty data")
    ext = ext.lower().lstrip(".")
    # Map any unknown extension to .png so we always produce a
    # valid image file. PNG is the safest default because every
    # multimodal endpoint accepts it.
    if ext not in {e.lstrip(".") for e in IMAGE_EXTS}:
        ext = "png"
    if ext == "jpg":
        ext = "jpeg"
    d = _ensure_pasted_dir()
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    # Find the next free counter for this second.
    n = 1
    while True:
        candidate = d / f"image-{stamp}-{n:03d}.{ext}"
        if not candidate.exists():
            break
        n += 1
        if n > 999:
            # Pathological case: 1000+ pastes in one second. Fall
            # through to using a millisecond suffix.
            candidate = d / f"image-{stamp}-{int(now*1000) % 1000000}.{ext}"
            break
    candidate.write_bytes(data)
    return candidate


# === Fuzzy matching =======================================================


def fuzzy_match(
    query: str,
    candidates: Iterable[str],
    limit: int = 50,
) -> list[tuple[int, str]]:
    """Rank ``candidates`` against ``query`` by fuzzy match.

    Returns ``(score, candidate)`` pairs sorted by ascending score.
    Lower score = better match. Candidates that don't contain the
    query as a subsequence (in order) are excluded from the
    result — the picker should only show real matches, not the
    whole filesystem in a random order. (Callers that need the
    "everything is a candidate" behaviour can pass an empty
    query and get the original order back.)

    Scoring rules (per query character, summed):

    * Match in the right order:        0
    * Consecutive to the previous one: −10 (big bonus)
    * Match at a path-separator / word boundary (``/``, ``_``,
      ``-``, space):                   −5
    * First query char matches the candidate's first char:  −20

    A length penalty of ``len/20`` keeps long paths from outscoring
    a tight prefix match on a short file. Cap the result at
    ``limit`` to keep the picker menu sane.
    """
    q = query.strip().lower()
    if not q:
        # Empty query → preserve order, score is the original index.
        return [(i, c) for i, c in enumerate(list(candidates)[:limit])]

    scored: list[tuple[int, str]] = []
    for cand in candidates:
        cl = cand.lower()
        # All query characters must appear in the candidate, in order.
        last = -1
        score = 0
        ok = True
        # Pre-compute word-boundary offsets once per candidate.
        boundaries = {0}
        for i, ch in enumerate(cl):
            if ch in "/_-" or ch.isspace():
                boundaries.add(i + 1)
        for k, ch in enumerate(q):
            idx = cl.find(ch, last + 1)
            if idx < 0:
                ok = False
                break
            if k == 0 and idx == 0:
                score -= 20
            if k == 0 and idx in boundaries:
                score -= 5
            if idx == last + 1 and last >= 0:
                score -= 10
            elif idx in boundaries:
                score -= 5
            last = idx
        if not ok:
            continue
        # Tiebreaker: shorter paths rank higher (more "specific").
        score += len(cl) // 20
        # And exact-prefix matches beat anything else.
        if cl.startswith(q):
            score -= 50
        scored.append((score, cand))

    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[:limit]


# === @-mention parsing ====================================================


# An @-mention. ``@`` is the trigger; the path runs until the next
# whitespace, ``@`` (so ``@@`` doesn't match), or one of the
# terminators below. The character before the ``@`` must NOT be an
# identifier-continuation char (so ``user@host.com`` is left alone).
# Note: ``.`` is intentionally NOT a terminator — file extensions
# like ``.py`` / ``.txt`` are part of the path, and stripping them
# at the parser level would force the user to escape the dot when
# typing a real filename.
_MENTION_RE = re.compile(
    r"(?:^|(?<=[\s(\[\{,;:!?]))@(?!@)([^\s@,;:!?)\}]+)"
)


def _resolve_mention_path(raw: str, cwd: str | pathlib.Path) -> pathlib.Path:
    """Resolve a mention's path.

    Resolution order:

    1. ``~`` or ``~/...`` → expand to the current user's home dir.
       This is what makes the ``@~/.../image-XXX.png`` form work
       when the TUI inserts the full path of a pasted image.
    2. Absolute path (``/...``) → used as-is.
    3. Anything else → resolved relative to ``cwd``.
    """
    p = pathlib.Path(raw)
    if raw.startswith("~"):
        return p.expanduser()
    if p.is_absolute():
        return p
    return pathlib.Path(cwd) / p


def expand_mentions(
    text: str,
    cwd: str | pathlib.Path | None = None,
    max_text_chars: int = MAX_TEXT_CHARS,
    max_image_bytes: int = MAX_IMAGE_BYTES,
    attachments: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Expand ``@path`` mentions in ``text`` into a multimodal content list.

    Returns a list of OpenAI-style content parts. The structure is:

    * Every text segment (the runs between / around mentions) becomes
      a ``{"type": "text", "text": "..."}`` part.
    * Every image mention becomes
      ``{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}``.
    * Every text-file mention is inlined as a fenced
      ``[file: <path>]\n```\n<contents>\n```` block. The path itself
      is preserved as a marker so the model can reference it back.

    If a mention is invalid (path doesn't exist, isn't readable, is
    too large, or the extension isn't supported), the original
    ``@path`` text is preserved verbatim — the user sees the mention
    in the prompt and the model can decide what to do.

    The optional ``attachments`` dict maps short reference IDs
    (e.g. ``"image-1"``) to absolute file paths on disk. This is
    how the TUI keeps the editor buffer short — pasted images are
    inserted as ``@image-1`` and the actual filesystem path is
    looked up here at submit time. If a mention's raw text matches
    a key in ``attachments``, that path is used INSTEAD of trying
    to resolve the raw text as a relative / absolute path. This
    also means a user can later type ``@image-1`` themselves and
    have the previously-pasted image attached.
    """
    if not text:
        return [{"type": "text", "text": text}]

    base = pathlib.Path(cwd) if cwd else pathlib.Path.cwd()
    parts: list[dict[str, Any]] = []
    last_end = 0

    for m in _MENTION_RE.finditer(text):
        before = text[last_end:m.start()]
        if before:
            parts.append({"type": "text", "text": before})
        raw = m.group(1)
        # Attachment short ID takes priority over file-path
        # resolution. This is what makes `@image-1` resolve to
        # the actual image at submit time, even though the
        # buffer only contains the short ID.
        if attachments and raw in attachments:
            replacement = _expand_resolved(
                pathlib.Path(attachments[raw]), raw,
                max_text_chars, max_image_bytes,
            )
        else:
            replacement = _expand_one(raw, base, max_text_chars, max_image_bytes)
        parts.extend(replacement)
        last_end = m.end()

    rest = text[last_end:]
    if rest:
        parts.append({"type": "text", "text": rest})

    # If the whole string was empty (no text, no mentions) or we
    # somehow produced no parts, fall back to a single text part so
    # the chat-completions endpoint gets a valid ``content`` value.
    if not parts:
        return [{"type": "text", "text": text}]
    # Same fallback for the "only one text part" case — it can be
    # sent as a string instead of a list, but we don't try to
    # optimize that here; the OpenAI client accepts both forms.
    return parts


def _expand_one(
    raw: str,
    cwd: pathlib.Path,
    max_text_chars: int,
    max_image_bytes: int,
) -> list[dict[str, Any]]:
    """Resolve a single ``@path`` mention into content parts.

    Returns either a list of one+ parts (image, text-file, or
    fallback literal) or an empty list (which the caller treats as
    "no expansion happened, just leave the original text in").
    """
    target = _resolve_mention_path(raw, cwd)
    return _expand_resolved(target, raw, max_text_chars, max_image_bytes)


def _expand_resolved(
    target: pathlib.Path,
    raw: str,
    max_text_chars: int,
    max_image_bytes: int,
) -> list[dict[str, Any]]:
    """Expand a mention whose path has already been resolved.

    Used by :func:`_expand_one` for ordinary mentions and by
    :func:`expand_mentions` for the ``attachments`` short-ID
    path (which is already absolute, so no cwd resolution is
    needed). Symlink resolution, size check, and the
    image-or-text-or-literal decision happen here in one place.
    """
    try:
        # Follow symlinks for the size check, but only for files
        # (a symlink to a directory is unusual; bail).
        resolved = target.resolve()
        if not resolved.is_file():
            return [{"type": "text", "text": f"@{raw}"}]
    except (OSError, RuntimeError):
        return [{"type": "text", "text": f"@{raw}"}]

    if is_image(resolved):
        try:
            data_url, _ = read_image_data_url(resolved, max_bytes=max_image_bytes)
        except (OSError, ValueError):
            return [{"type": "text", "text": f"@{raw}"}]
        return [{"type": "image_url", "image_url": {"url": data_url}}]

    if is_text_file(resolved):
        try:
            content = read_text_file(resolved, max_chars=max_text_chars)
        except (OSError, UnicodeDecodeError):
            return [{"type": "text", "text": f"@{raw}"}]
        body = f"\n[file: {raw}]\n```\n{content}\n```\n"
        return [{"type": "text", "text": body}]

    # Unknown extension: leave the mention literal so the model
    # can decide whether to call a tool to fetch it.
    return [{"type": "text", "text": f"@{raw}"}]


def find_active_mention(buffer: str, cursor: int) -> tuple[int, int] | None:
    """Locate the ``@``-mention that the cursor is currently inside.

    Returns ``(start, end)`` where ``start`` is the index of the
    ``@`` and ``end`` is one past the last character of the
    mention. Both indices are absolute positions in ``buffer``
    (not row/col).

    The rules mirror :data:`_MENTION_RE` so the editor and the
    expander agree on what counts:

    * The ``@`` is preceded by start-of-string or a non-identifier
      character (anything not ``[A-Za-z0-9_]``).
    * The mention extends to the next whitespace / terminator or
      to the cursor, whichever comes first.

    Returns ``None`` if there is no active mention (no ``@`` on
    the current line in the relevant range, or the cursor is
    before the ``@``).
    """
    if cursor <= 0 or not buffer:
        return None
    # Walk back from the cursor, looking for the most recent @ that
    # has no whitespace / terminator between it and the cursor.
    # The set of terminators matches :data:`_MENTION_RE` so the
    # editor and the expander agree on what counts. The dot is NOT
    # a terminator (file extensions like .py / .txt are part of
    # the path).
    end = cursor
    i = cursor - 1
    while i >= 0:
        ch = buffer[i]
        if ch.isspace() or ch in ",;:!?)]}" or ch == "@":
            # Cursor is past the end of the mention (in a terminator
            # or whitespace gap, or right after an @ that bounds it).
            # The mention itself is to the right of this position;
            # we need to find the @ in the remaining window.
            if ch == "@" and (i == 0 or not (buffer[i - 1].isalnum()
                                             or buffer[i - 1] == "_")):
                # Edge case: cursor is right after an @ with no
                # characters typed yet (@ by itself).
                return (i, end)
            break
        if ch == "@":
            # Check the character before is not an identifier-continuation.
            if i == 0 or not (buffer[i - 1].isalnum() or buffer[i - 1] == "_"):
                return (i, end)
            return None
        i -= 1
    # If we hit whitespace/terminator without finding an @, no
    # active mention.
    return None


def mention_query(buffer: str, cursor: int) -> tuple[str, int, int]:
    """Return the search query + ``(start, end)`` indices of the mention.

    Empty string + ``(0, 0)`` if no active mention. ``start`` is the
    position of the ``@``; ``end`` is the cursor position. The
    caller can replace ``buffer[start:end]`` with the resolved path
    to insert the file.
    """
    span = find_active_mention(buffer, cursor)
    if span is None:
        return "", 0, 0
    s, _ = span
    return buffer[s + 1:cursor], s, cursor
