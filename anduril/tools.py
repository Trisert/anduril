"""Tool system: decorator, JSON Schema derivation, and argument validation.

The :func:`tool` decorator turns a regular function into a :class:`Tool`
namedtuple. The JSON schema is derived from the function's signature and
type hints (including ``Optional``, ``Union``, ``Literal``, ``list``,
``dict``, and ``Annotated``). The :func:`_validate` function provides
lightweight runtime validation of arguments before they're passed to
the tool function.
"""

from __future__ import annotations

import inspect
import json
import os
import pathlib
import re
import signal
import subprocess
import types
import typing
from collections import namedtuple
from typing import Any, Callable, Union


Tool = namedtuple("Tool", "name description parameters fn dangerous risk")

# Risk levels, ordered from least to most risky. Used by the approval
# gate to decide which dangerous tools to prompt for at a given
# threshold. ``"medium"`` is the default for any tool declared with
# ``dangerous=True`` but no explicit ``risk=...`` so existing tools
# behave the same as before under the default ``--approval all`` mode.
RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high")
RISK_RANK: dict[str, int] = {name: i for i, name in enumerate(RISK_LEVELS)}


_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _json_type(t: Any) -> dict[str, Any]:
    """Return a JSON Schema fragment for a Python type."""
    if t in _TYPE_MAP:
        return {"type": _TYPE_MAP[t]}

    origin = typing.get_origin(t)
    args = typing.get_args(t)

    if origin is typing.Annotated:
        schema = _json_type(args[0]) if args else {}
        for meta in args[1:]:
            if isinstance(meta, str):
                schema.setdefault("description", meta)
        return schema

    if origin in (Union, getattr(types, "UnionType", None)):
        non_none = [a for a in args if a not in (type(None), None)]
        is_optional = len(non_none) < len(args)
        if not non_none:
            return {"type": "null"}
        if len(non_none) == 1:
            schema = _json_type(non_none[0])
        else:
            schema = {"anyOf": [_json_type(a) for a in non_none]}
        if is_optional:
            schema = {"anyOf": [schema, {"type": "null"}]}
        return schema

    if origin in (list, tuple, set, frozenset):
        items = _json_type(args[0]) if args else {}
        return {"type": "array", "items": items}

    if origin is dict:
        additional = _json_type(args[1]) if len(args) >= 2 else True
        return {"type": "object", "additionalProperties": additional}

    if origin is typing.Literal:
        return {"enum": list(args)}

    # Fallback: treat unknown types as strings so the agent can still try.
    return {"type": "string"}


def _is_optional(t: Any) -> bool:
    if typing.get_origin(t) is typing.Annotated:
        args = typing.get_args(t)
        if args:
            t = args[0]
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if origin in (Union, getattr(types, "UnionType", None)):
        return type(None) in args or None in args
    return False


def _parse_param_docs(doc: str) -> dict[str, str]:
    """Extract `:param name: description` / Google-style arg docs."""
    docs: dict[str, str] = {}
    if not doc:
        return docs
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            continue
        if stripped.startswith(":param "):
            # `:param` can appear with or without an explicit Args section.
            rest = stripped[7:]
            if ":" in rest:
                name, desc = rest.split(":", 1)
                docs[name.strip()] = desc.strip()
            continue
        if not in_args:
            continue
        if stripped.startswith(("Returns:", "Raises:", "Yields:", "Notes:", "Example:")):
            break
        if stripped.startswith("- "):
            rest = stripped[2:]
            if ":" in rest:
                name, desc = rest.split(":", 1)
                docs[name.strip()] = desc.strip()
        elif stripped:
            m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", stripped)
            if m:
                name, desc = m.groups()
                docs[name.strip()] = desc.strip()
    return docs


def _build_schema(fn: Callable) -> dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    param_docs = _parse_param_docs(inspect.getdoc(fn) or "")
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str
        prop = _json_type(annotation)
        desc = param_docs.get(name)
        if desc:
            prop["description"] = desc
        properties[name] = prop
        if param.default is inspect.Parameter.empty and not _is_optional(annotation):
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    dangerous: bool = False,
    risk: str = "medium",
) -> Callable | Tool:
    """Decorator that turns a function into a Tool.

    The JSON schema is derived from the function's signature and type hints,
    including `Optional`, `Union`, `Literal`, `list`, `dict`, and `Annotated`.
    Override with `parameters={...}` for a fully custom schema. Use `name=` and
    `description=` to override the defaults (function name and docstring).
    Mark a tool as `dangerous=True` to require user confirmation before runs,
    and set ``risk="low" | "medium" | "high"`` to control which approval
    threshold (set via ``--approval``) actually triggers the prompt.
    """

    def wrap(f: Callable) -> Tool:
        if dangerous and risk not in RISK_LEVELS:
            raise ValueError(
                f"tool {f.__name__!r}: risk must be one of {RISK_LEVELS}, got {risk!r}"
            )
        return Tool(
            name=name or f.__name__,
            description=description or inspect.getdoc(f) or "",
            parameters=parameters or _build_schema(f),
            fn=f,
            dangerous=dangerous,
            risk=risk if dangerous else "low",
        )

    if fn is not None and callable(fn):
        return wrap(fn)
    return wrap


# ============================================================================
# Validation
# ============================================================================

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def _validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Lightweight JSON Schema validation. Returns a list of error messages."""
    errors: list[str] = []

    if "anyOf" in schema:
        for sub in schema["anyOf"]:
            if not _validate(value, sub, path):
                break
        else:
            errors.append(f"{path}: does not match any allowed type")
        return errors

    jtype = schema.get("type")
    if jtype:
        types_allowed = [jtype] if isinstance(jtype, str) else jtype
        allowed = []
        for t in types_allowed:
            allowed.extend(_JSON_TYPES.get(t, ()))
        if not allowed or not isinstance(value, tuple(allowed)):
            errors.append(f"{path}: expected type {jtype}, got {type(value).__name__}")
            return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")

    if jtype == "array" and isinstance(value, (list, tuple)):
        items = schema.get("items", {})
        for i, item in enumerate(value):
            errors.extend(_validate(item, items, f"{path}[{i}]"))

    if jtype == "object" and isinstance(value, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required property '{key}'")
        for key, val in value.items():
            if key in props:
                errors.extend(_validate(val, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property '{key}'")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(_validate(val, schema["additionalProperties"], f"{path}.{key}"))

    return errors


# ============================================================================
# Default tool: bash
# ============================================================================


def _kill_pg(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


@tool(dangerous=True, risk="high")
def bash(command: str, timeout: int = 120) -> str:
    """Run a shell command and return its stdout, stderr, and exit code.

    :param command: The shell command to execute.
    :param timeout: Maximum seconds to wait before killing the process.
    Times out after `timeout` seconds. Ctrl-C kills the subprocess group.
    """
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError:
        return "error: shell not found"
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_pg(proc.pid)
            proc.wait()
            return f"error: command timed out after {timeout} seconds"
        except KeyboardInterrupt:
            _kill_pg(proc.pid)
            proc.wait()
            raise
    except Exception:
        _kill_pg(proc.pid)
        proc.wait()
        raise

    out = (stdout or "").rstrip("\n")
    err = (stderr or "").rstrip("\n")
    parts: list[str] = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"[stderr] {err}")
    parts.append(f"[exit {proc.returncode}]")
    return "\n".join(parts)


# ============================================================================
# create_skill: build a new skill on disk and load its tools
# ============================================================================


@tool(dangerous=True, risk="high")
def create_skill(
    name: str,
    code: str,
    description: str = "",
    persistent: bool = False,
) -> str:
    """Create a new skill from Python code and load its tools.

    Skills are Python modules that define ``@tool``-decorated functions
    and assign them to a module-level ``tools`` list. The new skill's
    tools become available to this agent in the next turn (or the
    current one if you call them right after).

    By default, skills are written to a session-scoped directory and
    vanish when the session ends. Pass ``persistent=True`` to save to
    the global skills dir at ``~/.local/share/anduril/skills/`` where
    they survive restarts and are visible to other sessions.

    Example code::

        from anduril.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            '''Add two numbers.'''
            return a + b

        @tool
        def multiply(a: int, b: int) -> int:
            '''Multiply two numbers.'''
            return a * b

        tools = [add, multiply]

    :param name: Skill name (lowercase letters, digits, hyphens).
    :param code: Python code with @tool functions and a ``tools`` list.
    :param description: Short human-readable description (used by /skills).
    :param persistent: Save to global skills dir instead of session-only.
    """
    # Imported here to avoid a top-level cycle: skills -> tools.
    from anduril.skills import register_skill
    try:
        tools, file_path = register_skill(
            name=name,
            code=code,
            description=description,
            persistent=persistent,
        )
    except (ValueError, RuntimeError) as e:
        return f"error: {e}"
    tool_names = ", ".join(t.name for t in tools)
    location = "global" if persistent else "session"
    return (
        f"created skill {name!r} ({location}) at {file_path}\n"
        f"registered tools: {tool_names}\n"
        f"they will be available in the next turn."
    )


# ============================================================================
# File-editing tools: read_file, write_file, apply_diff, search_files
# ============================================================================
#
# These are the dedicated file tools that replace the previous "use bash
# with heredocs" workflow. The model gets a small, well-typed surface:
#
#   * read_file    — read text (or a line range) of a file.
#   * write_file   — overwrite a file (creates parents).
#   * apply_diff   — surgical replacement (old_text must match exactly once).
#   * search_files — substring or glob search across a directory tree.
#
# They integrate with the standard tool pipeline (schema derivation,
# validation, approval gating, sanitization) and never shell out.


# Cap on a single read_file response, in characters. A return of 200K
# chars is well within modern context windows; longer files must be
# read in slices via start_line / end_line. Mirrors the existing
# mention-expander cap so the model sees consistent numbers.
_MAX_READ_CHARS = 200_000

# Cap on a single search_files response. Search results can balloon
# when a query matches many files (think `import os` across a big
# repo), so we trim and report the omitted count.
_MAX_SEARCH_RESULTS = 500

# Cap on a single search hit's "context" line, in characters. Very
# long lines (minified JS, packed JSON) would otherwise dominate the
# response. The marker tells the model to re-run with a narrower
# pattern if it needs the rest.
_MAX_HIT_CHARS = 500


def _read_text_file(
    path: str | pathlib.Path,
    *,
    max_chars: int = _MAX_READ_CHARS,
) -> str:
    """Read a text file as a string. Raises on binary / non-UTF-8 / missing.

    Used by both :func:`read_file` and :func:`apply_diff`. The error
    messages are tuned for the model's eyes — they name the actual
    failure mode (binary file, permission denied, etc.) so the model
    can pick a recovery strategy without having to guess.
    """
    p = pathlib.Path(path).expanduser()
    try:
        # stat first so the error message names "is a directory" or
        # "no such file" cleanly without opening and failing
        # ambiguously.
        st = p.stat()
    except FileNotFoundError:
        return f"error: file not found: {p}"
    except PermissionError as e:
        return f"error: permission denied: {p} ({e})"
    except OSError as e:
        return f"error: cannot stat {p}: {type(e).__name__}: {e}"
    if not p.is_file():
        return f"error: not a regular file: {p}"
    try:
        with p.open("r", encoding="utf-8", errors="strict") as f:
            text = f.read(max_chars + 1)
    except UnicodeDecodeError:
        return (
            f"error: {p} is not valid UTF-8 (binary or non-text file?) "
            f"— refusing to read as text"
        )
    except PermissionError as e:
        return f"error: permission denied: {p} ({e})"
    except OSError as e:
        return f"error: read failed for {p}: {type(e).__name__}: {e}"
    if len(text) > max_chars:
        text = text[:max_chars] + (
            f"\n\n[... file truncated at {max_chars} chars; "
            f"total {st.st_size} bytes on disk. Re-read with "
            f"start_line / end_line to see the rest.]"
        )
    return text


@tool
def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    numbered: bool = False,
) -> str:
    """Read a UTF-8 text file and return its contents.

    Returns up to ~200,000 characters; larger files are truncated
    with a clear marker. Use ``start_line`` / ``end_line`` (1-based,
    inclusive on both ends) to read a specific range — useful for
    very large files or for re-reading the slice around an edit.

    By default the returned text is **plain** (no per-line
    numbering). For multi-kilobyte files that would otherwise
    eat context window on redundant ``path:line:`` prefixes,
    plain is the right default. Pass ``numbered=True`` to
    prefix every line with ``path:LINE:`` — useful when
    you're constructing an :func:`apply_diff` (which needs
    exact line numbers) or about to refer back to a
    specific line in your answer.

    When ``start_line`` / ``end_line`` are given but
    ``numbered=False`` (the common case for "show me lines
    400-600"), the result is the raw slice of the file with
    a single ``[path:start-end]`` header line. The model can
    refer to specific lines by their absolute number
    (mentioned in the header) without paying the per-line
    prefix cost in every returned line.

    Binary files (anything that isn't valid UTF-8) are
    rejected with an explicit error rather than returned as
    garbage. To inspect a binary file, use :func:`bash` with
    ``file`` or ``xxd``.

    :param path: Path to the file. ``~`` is expanded.
    :param start_line: First line to include (1-based). ``None``
        starts at the beginning of the file.
    :param end_line: Last line to include (1-based, inclusive).
        ``None`` reads to the end of the file.
    :param numbered: Prefix every returned line with
        ``path:LINE: ``. Off by default to keep the
        response compact; enable when you need to refer
        back to specific lines.
    """
    p = pathlib.Path(path).expanduser()
    if start_line is not None and start_line < 1:
        return f"error: start_line must be >= 1 (got {start_line})"
    if end_line is not None and end_line < 1:
        return f"error: end_line must be >= 1 (got {end_line})"
    if (start_line is not None and end_line is not None
            and end_line < start_line):
        return (f"error: end_line ({end_line}) is before "
                f"start_line ({start_line})")
    try:
        text = _read_text_file(p)
    except Exception as e:  # last-resort safety net
        return f"error: read failed: {type(e).__name__}: {e}"
    if text.startswith("error:"):
        return text
    # No range and no numbering requested: return the raw
    # text verbatim. This is the most common call ("read
    # this file for me") and shouldn't pay any overhead.
    if start_line is None and end_line is None and not numbered:
        return text
    lines = text.splitlines()
    s = (start_line or 1) - 1
    e = end_line if end_line is not None else len(lines)
    slice_ = lines[s:e]
    if numbered:
        # Per-line ``path:LINE: `` prefix. Useful when the
        # caller needs to anchor references to specific
        # lines (e.g. building an apply_diff call). The
        # path uses the same ``str(p)`` form every other
        # tool does, so the caller can ``grep`` the result
        # to find lines.
        out: list[str] = []
        for i, line in enumerate(slice_, start=s + 1):
            out.append(f"{p}:{i}: {line}")
        return "\n".join(out)
    # Range but no numbering: the cheap default. A single
    # header so the caller knows the absolute line range,
    # then the raw text. We do not duplicate the path on
    # every line — that would burn context window for no
    # benefit when the caller only needs to see the
    # content.
    actual_start = s + 1
    # ``actual_end`` is the absolute number of the last line
    # we actually returned. When the slice is empty (e.g.
    # ``start_line`` past EOF, or a 1-line file with
    # ``start_line=5, end_line=5``) the result is the
    # requested end line so the caller can see "nothing was
    # in this range" at a glance.
    if slice_:
        actual_end = s + len(slice_)
    else:
        actual_end = end_line if end_line is not None else actual_start
    header = f"[{p}:{actual_start}-{actual_end}]"
    if not slice_:
        return f"{header}\n(empty)"
    return f"{header}\n" + "\n".join(slice_)


@tool(dangerous=True, risk="medium")
def write_file(path: str, content: str) -> str:
    """Write ``content`` to ``path``, creating parent directories as needed.

    Overwrites any existing file. To make a surgical change, prefer
    :func:`apply_diff` — it forces the model to anchor the change
    to existing text and makes accidental overwrites much harder.
    Use ``write_file`` for new files, full rewrites, or when the
    model is generating the file from scratch (e.g. ``cat > foo.py
    << 'EOF' ... EOF`` patterns).

    :param path: Destination path. ``~`` is expanded; missing
        parent directories are created.
    :param content: The full file contents to write. Use ``\\n`` for
        newlines; trailing newline is the caller's responsibility.
    """
    p = pathlib.Path(path).expanduser()
    try:
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically: write to a sibling tmp file, fsync, rename.
        # This avoids leaving a half-written file if the process dies
        # mid-write (power loss, OOM kill, Ctrl-C at exactly the
        # wrong moment). The temp file lives in the same directory so
        # the rename is on the same filesystem.
        tmp = p.with_name(p.name + f".anduril.tmp.{os.getpid()}")
        with tmp.open("w", encoding="utf-8", errors="strict") as f:
            f.write(content)
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                # fsync can fail on some filesystems; the write is
                # still durable on most OSes and the rename will
                # catch the bad case.
                pass
        os.replace(tmp, p)
    except PermissionError as e:
        return f"error: permission denied: {p} ({e})"
    except OSError as e:
        return f"error: write failed: {type(e).__name__}: {e}"
    except UnicodeEncodeError as e:
        return f"error: content is not valid UTF-8: {e}"
    return f"wrote {len(content)} chars to {p}"


@tool(dangerous=True, risk="medium")
def apply_diff(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    """Replace ``old_text`` with ``new_text`` in ``path``.

    This is the safe replacement primitive. It anchors the change
    to the exact text the model wants to change, so the model
    can't accidentally clobber a different file or a different
    section of the same file.

    Matching rules:

    * By default, ``old_text`` must match **exactly once** in the
      file. If it appears 0 times, the call fails with a clear
      error (no changes are made). If it appears more than once,
      the call fails and the model must add more surrounding
      context to make the match unique.
    * With ``replace_all=True``, every occurrence of ``old_text``
      is replaced. Use this for renames or bulk edits where the
      pattern is intentionally repeated.

    Whitespace is matched literally. Tabs and spaces are not
    interchangeable; the model must paste the indentation it sees
    in the file (or re-read the file with :func:`read_file` if
    unsure).

    :param path: Path to the file. ``~`` is expanded.
    :param old_text: The exact text to find. Must be non-empty.
    :param new_text: The replacement text. Use ``""`` to delete.
    :param replace_all: Replace every occurrence instead of
        requiring a unique match.
    """
    if not old_text:
        return "error: old_text must be non-empty (use write_file to overwrite)"
    p = pathlib.Path(path).expanduser()
    text = _read_text_file(p)
    if text.startswith("error:"):
        return text
    count = text.count(old_text)
    if count == 0:
        return (
            f"error: old_text not found in {p}. "
            f"Re-read the file with read_file to see the current content."
        )
    if count > 1 and not replace_all:
        return (
            f"error: old_text matches {count} places in {p}. "
            f"Either add more context to make it unique, or pass "
            f"replace_all=True if the bulk replacement is intentional."
        )
    if replace_all:
        new_content = text.replace(old_text, new_text)
        occurrences = count
    else:
        new_content = text.replace(old_text, new_text, 1)
        occurrences = 1
    # Reuse write_file's atomic write path so the on-disk effect is
    # identical (tmp + fsync + rename).
    res = write_file.fn(path=str(p), content=new_content)
    if res.startswith("error:"):
        return res
    return f"applied diff to {p} ({occurrences} replacement{'s' if occurrences != 1 else ''})"


@tool
def search_files(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 100,
) -> str:
    """Search files in a directory tree for a substring.

    Walks ``path`` (relative to the current working directory, or
    absolute) and reports lines that contain ``pattern``,
    grouped by file.

    Output format::

        relative/path.py: N matches
          :LINE: matched line
          :LINE: matched line

        other/file.py: N matches
          :LINE: matched line

    The file path is shown once per group, saving context when
    many lines in the same file match.

    The search is a plain substring match, not a regex — keeps
    the model from having to escape metacharacters for the
    common case ("find every place that calls ``process``").
    Combine with :func:`bash` ``grep -E`` when you need regex.

    Directories in the default ignore list (``.git``,
    ``__pycache__``, ``node_modules``, …) are skipped, as are
    files larger than 1MB (likely generated/binary even if they
    happen to contain the pattern as bytes).

    :param pattern: Substring to search for.
    :param path: Directory to walk. Defaults to ``.``.
    :param glob: Optional filename pattern (e.g. ``*.py``). Only
        files matching this pattern are searched. The pattern is
        matched against the basename, not the full path.
    :param case_sensitive: Case-sensitive match. Default ``False``.
    :param max_results: Stop after this many matches. Default 100,
        capped at ``_MAX_SEARCH_RESULTS`` (500).
    """
    if not pattern:
        return "error: pattern must be non-empty"
    if max_results < 1:
        return "error: max_results must be >= 1"
    max_results = min(max_results, _MAX_SEARCH_RESULTS)
    root = pathlib.Path(path).expanduser()
    if not root.is_dir():
        return f"error: not a directory: {root}"
    # Lazy import to keep the top of the module free of cycles
    # (anduril.files -> anduril.tools is the wrong direction).
    from anduril.files import DEFAULT_IGNORE_DIRS
    needle = pattern if case_sensitive else pattern.lower()
    # Group hits by file path so we don't repeat the path on every line.
    groups: dict[str, list[tuple[int, str]]] = {}
    seen = 0
    # Iterative walk with a manual stack so we can short-circuit
    # when we hit the result cap, and so a recursive symlink
    # doesn't loop forever. Mirrors list_files() in anduril.files
    # but with smaller scope (no max_count on the walk itself).
    stack: list[pathlib.Path] = [root.resolve()]
    while stack and seen < max_results:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name in DEFAULT_IGNORE_DIRS:
                    continue
                if entry.is_symlink():
                    continue
                stack.append(entry)
            elif entry.is_file():
                if glob is not None:
                    if not entry.match(glob):
                        continue
                try:
                    if entry.stat().st_size > 1_000_000:
                        continue
                except OSError:
                    continue
                try:
                    with entry.open("r", encoding="utf-8", errors="strict") as f:
                        for lineno, raw in enumerate(f, start=1):
                            haystack = raw if case_sensitive else raw.lower()
                            if needle in haystack:
                                line = raw.rstrip("\n")
                                if len(line) > _MAX_HIT_CHARS:
                                    line = line[:_MAX_HIT_CHARS] + "…"
                                try:
                                    rel = entry.relative_to(root.resolve())
                                except ValueError:
                                    rel = entry
                                groups.setdefault(str(rel), []).append((lineno, line))
                                seen += 1
                                if seen >= max_results:
                                    break
                except (OSError, UnicodeDecodeError, PermissionError):
                    continue
            if seen >= max_results:
                break
    if not groups:
        return f"no matches for {pattern!r} under {root}"
    out: list[str] = []
    for rel_path, hits in groups.items():
        label = "match" if len(hits) == 1 else "matches"
        out.append(f"{rel_path}: {len(hits)} {label}")
        for lineno, line in hits:
            out.append(f"  :{lineno}: {line}")
    body = "\n".join(out)
    if seen >= max_results:
        body += f"\n\n[... truncated at {max_results} results. Narrow the pattern or pass a more specific `glob` to see more.]"
    return body


# Default file tool set, exported for the CLI / TUI to add to the
# agent's tool list. Centralised here so adding a new file tool is
# one line in this module and one entry in DEFAULT_FILE_TOOLS.
DEFAULT_FILE_TOOLS: list[Tool] = [read_file, write_file, apply_diff, search_files]


# === MCP server registration tool =========================================
#
# ``add_mcp_server`` lets the model register an external MCP server
# at runtime, instead of forcing the user to edit pyproject.toml by
# hand. The model can discover a tool, decide it wants to use it
# (e.g. via the create_skill / system prompt path), and then call
# ``add_mcp_server(name=..., command=...)`` to wire it up. The new
# server's tools become available in the same session (no restart
# needed); the user just sees the new tool names appear in the log.

# Default location for MCP server configs the model writes.
# Override with ``ANDURIL_MCP_CONFIG_DIR`` to keep the global
# state out of ``~/.local/state/anduril/``. We resolve this
# lazily (function call) so tests that set the env var after
# import actually take effect.
def _mcp_config_dir() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get(
            "ANDURIL_MCP_CONFIG_DIR",
            str(pathlib.Path.home() / ".local" / "state" / "anduril" / "mcp"),
        )
    )


@tool(dangerous=True, risk="medium")
def add_mcp_server(
    name: str,
    command: list[str] | str | None = None,
    *,
    url: str | None = None,
    persistent: bool = False,
) -> str:
    """Register an external MCP server. Tools become available immediately.

    Exactly one of ``command`` (stdio) or ``url`` (HTTP) must
    be set. The new server is started, its tools are
    discovered, and registered with the running agent.
    Optionally persist the config so future sessions
    auto-load the server (``persistent=True`` writes to the
    global MCP config; the default is session-scoped, which
    is written under the active session id and dies with
    the session).

    :param name: Identifier for the server. Used as the
        tool-name prefix (``<name>__<tool>``).
    :param command: For stdio servers. A list, or a string
        which is shlex-split.
    :param url: For HTTP servers. A ``http://`` or
        ``https://`` URL.
    :param persistent: If True, write the config to the
        global MCP dir (``~/.local/state/anduril/mcp/``) so
        all future sessions pick it up. Default is the
        session-scoped dir (dies with the session).
    """
    if command is None and url is None:
        return "error: must set either command= or url="
    if command is not None and url is not None:
        return "error: set only one of command= or url="

    from anduril.mcp_client import (
        MCPServer, discover_mcp_tools, shutdown_servers,
    )
    from anduril import metrics as _metrics_mod  # noqa: F401  (forces init)

    # Build the server.
    if url is not None:
        server = MCPServer(name=name, url=url)
    else:
        server = MCPServer(name=name, command=command)

    # Persist the config (best-effort: a failure here doesn't
    # block the in-session registration).
    try:
        if persistent:
            cfg_dir = _mcp_config_dir()
        else:
            # Session-scoped: nest under the current session id.
            session_id = os.environ.get(
                "ANDURIL_SESSION_ID", os.environ.get("ANDURIL_SESSION", "default")
            )
            cfg_dir = _mcp_config_dir() / "sessions" / session_id
        cfg_dir.mkdir(parents=True, exist_ok=True)
        # Write the config as a small JSON file the loader can
        # read on next startup. Schema mirrors the
        # ``[tool.anduril.mcp_servers.servers.<name>]`` block
        # in pyproject.toml.
        cfg_file = cfg_dir / f"{name}.json"
        with cfg_file.open("w", encoding="utf-8") as f:
            json.dump({
                "command": command if command else None,
                "url": url if url else None,
            }, f, indent=2)
    except Exception as e:
        persistence_warning = (
            f"\n  (config persistence failed: {e})"
        )
    else:
        persistence_warning = ""

    # Try the live registration. The current agent instance
    # exposes its tool list via a global — we discover what
    # we can here, but the actual ``register_tool`` call
    # happens via the agent_mod hook the model is
    # registered through. For now we return the discovered
    # tool names and the user (or the next agent run)
    # re-discovers via the persisted config.
    try:
        tools = discover_mcp_tools([server])
    except Exception as e:
        shutdown_servers([server])
        return f"error: server failed to start: {e}{persistence_warning}"

    tool_names = [t.name for t in tools]
    if not tool_names:
        shutdown_servers([server])
        return (
            f"server {name!r} registered but reported no tools"
            f"{persistence_warning}"
        )

    # Register the new tools with the running agent so the
    # model can use them this turn. The agent writes its
    # own reference to ``anduril.agent._current_agent_module``
    # on construction, so we look it up there.
    try:
        from anduril.agent import _current_agent_module
        agent = _current_agent_module._current
    except Exception:
        agent = None
    registered_with_agent = 0
    if agent is not None:
        for t in tools:
            try:
                agent.register_tool(t)
                registered_with_agent += 1
            except Exception:
                pass

    location = "global" if persistent else "session"
    return (
        f"registered MCP server {name!r} ({location}) — "
        f"{len(tool_names)} tool{'s' if len(tool_names) != 1 else ''}: "
        f"{', '.join(tool_names)}"
        f"{persistence_warning}"
    )


