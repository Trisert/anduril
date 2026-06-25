"""Skill discovery, registration, and lifecycle.

A *skill* is a Python file or package that defines ``@tool``-decorated
functions and assigns them to a module-level ``tools`` list. The agent
loads skills on startup from a discovery path, and the
:func:`anduril.tools.create_skill` tool can create new skills at runtime.

Discovery path
--------------

Skills are looked up in the following locations, in order:

1. ``$ANDURIL_SKILLS_PATH`` — a colon-separated list of directories.
2. ``~/.local/share/anduril/skills/`` — the global skills directory.
3. ``<session>/skills/`` if ``$ANDURIL_SESSION_ID`` is set — the
   per-session skills directory (auto-created by :func:`register_skill`).

Each directory is scanned for either a single ``<name>.py`` file or a
``<name>/__init__.py`` package. Modules starting with ``_`` or ``.`` are
ignored. Skills must expose a ``tools`` list attribute; if absent, the
loader skips them silently.

Session vs persistent
---------------------

:func:`register_skill` writes to the session directory by default, so
skills the agent creates mid-session disappear with the session (kept
under ``~/.local/state/anduril/skills/<session_id>/`` alongside
sessions). Pass ``persistent=True`` to write to the global directory,
where they survive restarts and are visible to other sessions.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from typing import Any

from anduril.tools import Tool


# === Paths ===============================================================


def _global_skills_dir() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("ANDURIL_SKILLS_DIR")
        or (pathlib.Path.home() / ".local" / "share" / "anduril" / "skills")
    )


def _session_skills_dir(session_id: str | None = None) -> pathlib.Path | None:
    sid = session_id or os.environ.get("ANDURIL_SESSION_ID")
    if not sid:
        return None
    return pathlib.Path.home() / ".local" / "state" / "anduril" / "skills" / sid


# === Loader ==============================================================


def _load_module(path: pathlib.Path, name: str) -> Any | None:
    """Load a Python file or package and return the module, or None on failure."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"anduril_skill_{name}", str(path)
        )
    except (ValueError, ImportError):
        return None
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        _warn_skill_load(name, e)
        return None
    return mod


def _warn_skill_load(name: str, e: BaseException) -> None:
    """Emit a one-line skill-load warning to stderr, unless suppressed.

    Honors two env vars:

    * ``ANDURIL_SKILLS_QUIET=1`` — silence all skill-load warnings.
    * ``ANDURIL_SKILL_DEBUG=1`` — also append the full exception chain.

    For the very common case of a missing optional dependency
    (``ImportError`` caused by ``ModuleNotFoundError``), prints a single
    short line with the install hint. Other failures get the standard
    ``type: message`` form.
    """
    if os.environ.get("ANDURIL_SKILLS_QUIET"):
        return
    debug = bool(os.environ.get("ANDURIL_SKILL_DEBUG"))
    # Most skill load failures are import problems. Surface them cleanly.
    cause = getattr(e, "__cause__", None) or e
    if isinstance(cause, ModuleNotFoundError) or (
        isinstance(e, ImportError) and getattr(e, "name", None)
    ):
        missing = getattr(cause, "name", None) or "<unknown>"
        msg = f"skill {name!r} missing dep {missing!r} — install it (e.g. pip install {missing})"
        if debug:
            msg += f"\n  original: {type(e).__name__}: {e}"
        print(f"  {msg}", file=sys.stderr)
        return
    msg = f"skill {name!r} failed to load: {type(e).__name__}: {e}"
    if debug:
        import traceback
        msg += "\n" + traceback.format_exc().rstrip()
    print(f"  {msg}", file=sys.stderr)


def _load_skill_dir(entry: pathlib.Path) -> list[Tool]:
    """Load a single skill (file or package directory)."""
    if entry.name.startswith(("_", ".")):
        return []
    if entry.is_file() and entry.suffix == ".py":
        target = entry
        name = entry.stem
    elif entry.is_dir():
        init = entry / "__init__.py"
        if not init.is_file():
            return []
        target = init
        name = entry.name
    else:
        return []
    mod = _load_module(target, name)
    if mod is None:
        return []
    tools_obj = getattr(mod, "tools", None)
    if not isinstance(tools_obj, (list, tuple)):
        return []
    out: list[Tool] = []
    for t in tools_obj:
        if isinstance(t, Tool) and t.name not in (existing.name for existing in out):
            out.append(t)
    return out


def discover_skills(paths: list[str] | None = None) -> list[Tool]:
    """Return all tools from skills in the given directories.

    If ``paths`` is None, uses ``$ANDURIL_SKILLS_PATH`` (colon-separated),
    then the global skills dir, then the session skills dir.
    """
    dirs: list[pathlib.Path] = []
    if paths:
        for p in paths:
            if p:
                dirs.append(pathlib.Path(p))
    else:
        env = os.environ.get("ANDURIL_SKILLS_PATH", "")
        for p in env.split(":"):
            if p:
                dirs.append(pathlib.Path(p))
        dirs.append(_global_skills_dir())
        sess = _session_skills_dir()
        if sess:
            dirs.append(sess)

    seen: set[str] = set()
    tools: list[Tool] = []
    for d in dirs:
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            for t in _load_skill_dir(entry):
                if t.name in seen:
                    continue
                seen.add(t.name)
                tools.append(t)
    return tools


# === Runtime registration =================================================


def register_skill(
    name: str,
    code: str,
    *,
    description: str = "",
    persistent: bool = False,
) -> tuple[list[Tool], pathlib.Path]:
    """Create a new skill on disk, load it, and return its tools + file path.

    By default, the skill is written to the session-scoped directory and
    vanishes with the session. Pass ``persistent=True`` to write to the
    global skills dir, where it survives restarts.

    The returned tools are also pushed onto the global
    :data:`_pending_registrations` queue so any running :class:`Agent`
    picks them up on the next tool call. The caller does not need to
    drain the queue manually.
    """
    if not name or not name.replace("-", "_").replace("_", "").isalnum():
        raise ValueError(
            f"invalid skill name {name!r} (use lowercase letters, digits, hyphens)"
        )
    if persistent:
        target_dir = _global_skills_dir()
    else:
        sess = _session_skills_dir()
        if sess is None:
            # No session context — fall back to global so the skill still works.
            target_dir = _global_skills_dir()
        else:
            target_dir = sess
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{name}.py"
    header = f'"""{description or name} — anduril skill."""\n\n' if description else ""
    file_path.write_text(header + code, encoding="utf-8")

    mod = _load_module(file_path, name)
    if mod is None:
        raise RuntimeError(f"skill {name!r} failed to import — check the code")
    tools_obj = getattr(mod, "tools", None)
    if not isinstance(tools_obj, (list, tuple)) or not tools_obj:
        raise RuntimeError(
            f"skill {name!r} did not define a `tools` list with at least one Tool"
        )
    tools: list[Tool] = [t for t in tools_obj if isinstance(t, Tool)]
    for t in tools:
        register_tool(t)
    return tools, file_path


# Global queue of tools waiting to be added to a running Agent. The agent
# drains this after each tool call so that `create_skill` (and any other
# tool that wants to extend the agent) takes effect within the same
# session, no restart required.
_pending_registrations: list[Tool] = []


def register_tool(tool: Tool) -> None:
    """Queue a tool to be added to the next agent that drains the queue."""
    if not isinstance(tool, Tool):
        raise TypeError(f"register_tool expected Tool, got {type(tool).__name__}")
    for existing in _pending_registrations:
        if existing.name == tool.name:
            return  # de-dupe silently
    _pending_registrations.append(tool)


def drain_pending_registrations() -> list[Tool]:
    """Return and clear all pending tool registrations."""
    out = _pending_registrations[:]
    _pending_registrations.clear()
    return out


# === Introspection / management ==========================================


def list_skills(paths: list[str] | None = None) -> list[dict[str, Any]]:
    """Return metadata for every skill on the discovery path."""
    dirs: list[pathlib.Path] = []
    if paths:
        for p in paths:
            if p:
                dirs.append(pathlib.Path(p))
    else:
        env = os.environ.get("ANDURIL_SKILLS_PATH", "")
        for p in env.split(":"):
            if p:
                dirs.append(pathlib.Path(p))
        dirs.append(_global_skills_dir())
        sess = _session_skills_dir()
        if sess:
            dirs.append(sess)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(("_", ".")):
                continue
            if entry.is_file() and entry.suffix == ".py":
                name = entry.stem
                file_path = entry
            elif entry.is_dir() and (entry / "__init__.py").is_file():
                name = entry.name
                file_path = entry / "__init__.py"
            else:
                continue
            if name in seen:
                continue
            seen.add(name)
            docstring = ""
            try:
                first_line = file_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[0:3]
                for line in first_line:
                    line = line.strip().strip('"').strip("'")
                    if line and not line.startswith("#"):
                        docstring = line
                        break
            except OSError:
                pass
            try:
                tools = _load_skill_dir(entry)
            except Exception:
                tools = []
            out.append({
                "name": name,
                "path": str(file_path),
                "dir": str(d),
                "description": docstring,
                "tools": [t.name for t in tools],
            })
    return out


def delete_skill(name: str, persistent: bool = False) -> bool:
    """Remove a skill file. Returns True if a file was removed."""
    if persistent:
        target = _global_skills_dir() / f"{name}.py"
    else:
        sess = _session_skills_dir()
        target = (sess / f"{name}.py") if sess else None
    if target is None:
        return False
    if target.is_file():
        target.unlink()
        return True
    pkg = target.with_suffix("")
    if pkg.is_dir() and (pkg / "__init__.py").is_file():
        import shutil
        shutil.rmtree(pkg)
        return True
    return False
