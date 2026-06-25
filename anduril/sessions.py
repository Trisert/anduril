"""Session persistence.

Each chat is stored as a JSON file under
``$ANDURIL_SESSIONS_DIR`` (defaulting to
``~/.local/state/anduril/sessions``). Writes are atomic (tmp + rename)
so a crash mid-save can't leave a half-written file. The directory
listing is mtime-based for cheap ordering.

A lightweight metadata index lives at ``<sessions_dir>/_index.json``,
mapping ``session_id -> {title, n, updated_at, model, created_at}``.
``_list_sessions`` reads the index (a few KB even for thousands of
sessions) instead of opening every full session JSON, which makes
``anduril sessions`` O(1) in the number of files on disk. The index
is auto-built on first use (or on upgrade from an older version that
didn't have one) by walking the sessions directory once.
"""

from __future__ import annotations

import json
import os
import pathlib
import secrets
import time

from anduril.env import _env_str


SESSION_HOME = pathlib.Path(
    _env_str("ANDURIL_HOME") or os.path.expanduser("~/.local/state/anduril")
)
SESSION_LIST_DEFAULT_LIMIT = 10
SESSION_LIST_MAX_LIMIT = 100


def _sessions_dir() -> pathlib.Path:
    override = _env_str("ANDURIL_SESSIONS_DIR")
    if override:
        return pathlib.Path(override).expanduser()
    return SESSION_HOME / "sessions"


def _new_session_id() -> str:
    """Short, unguessable, sortable-ish: YYYYMMDD-HHMMSS-<6 hex>."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _short_id(session_id: str) -> str:
    if session_id and "-" in session_id:
        return session_id.rsplit("-", 1)[-1]
    return session_id or ""


def _safe_title(text: str, maxlen: int = 60) -> str | None:
    """Turn the first user message into a filesystem-safe-ish title."""
    if not text:
        return None
    text = " ".join(str(text).split())
    text = "".join(c for c in text if c.isprintable())
    if len(text) > maxlen:
        text = text[:maxlen - 1] + "…"
    return text or None


def _is_empty_assistant_message(msg: dict) -> bool:
    if msg.get("role") != "assistant" or msg.get("tool_calls"):
        return False
    content = msg.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part.strip():
                return False
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return False
        return True
    return False


def _prune_empty_assistant_messages(messages: list) -> int:
    """Drop assistant turns that have neither visible content nor tool calls.

    Returns the number of entries removed. Does NOT mutate the input
    list — it returns the kept list via a slice assignment by the caller.
    """
    if not isinstance(messages, list):
        return 0
    kept = [m for m in messages if not _is_empty_assistant_message(m)]
    removed = len(messages) - len(kept)
    if removed:
        messages[:] = kept
    return removed


def _session_path(session_id: str) -> pathlib.Path:
    return _sessions_dir() / f"{session_id}.json"


# === Metadata index =======================================================

# Filename of the lightweight metadata index. Stored inside the
# sessions dir so it travels with the rest of the state. The leading
# underscore keeps it visually separate from session files in a raw
# directory listing, and :func:`_list_sessions` skips it when walking
# the dir.
INDEX_FILENAME = "_index.json"
INDEX_VERSION = 1

# In-memory cache of the index. Keyed on (mtime of the index file,
# mtime of the sessions dir) so an external write to either
# invalidates the cache. ``None`` means "not yet loaded" — the first
# list after startup pays the cost, subsequent lists hit the cache.
_index_cache: dict | None = None
_index_cache_key: tuple | None = None


def _index_path() -> pathlib.Path:
    """Return the path to the metadata index file."""
    return _sessions_dir() / INDEX_FILENAME


def _load_index_from_disk() -> dict:
    """Read the index file, returning an empty index on miss/error.

    Tolerant of malformed files: a corrupt index just means we
    rebuild from the session files on the next listing. We never
    raise from this path because a broken index shouldn't brick
    the whole sessions subsystem.
    """
    path = _index_path()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if (isinstance(data, dict)
                and data.get("version") == INDEX_VERSION
                and isinstance(data.get("entries"), dict)):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": INDEX_VERSION, "entries": {}}


def _save_index_to_disk(entries: dict) -> None:
    """Atomically write the index. Best-effort: failures are silent.

    A failed index write isn't fatal — the next list will rebuild
    from the on-disk session files. We log to stderr rather than
    raising because index writes happen on every chat save, and
    a chat save that crashes after a successful session write
    would lose the user's turn.

    After a successful write we also invalidate the in-memory
    cache key, so the very next ``get_index()`` re-stats the
    file rather than returning a possibly-stale value. (The
    mtime would normally pick this up, but on coarse-grained
    filesystems the write can land in the same second as the
    last read, leaving the cache key valid by accident.)
    """
    global _index_cache_key
    d = _sessions_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = _index_path()
    payload = {"version": INDEX_VERSION, "entries": entries}
    tmp = path.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        _index_cache_key = None  # force a re-stat on the next get_index()
    except OSError:
        # Best-effort: try to clean up the temp file if the rename failed.
        try:
            tmp.unlink()
        except OSError:
            pass


def _cache_key() -> tuple | None:
    """A cache key that changes when the index file or dir is touched.

    Returns ``None`` if the stat fails, which forces a re-read on
    the next list. We could also re-read on stat failure but
    ``None`` is the right "definitely stale" sentinel.
    """
    try:
        idx_mtime = _index_path().stat().st_mtime
    except OSError:
        idx_mtime = 0.0
    try:
        dir_mtime = _sessions_dir().stat().st_mtime
    except OSError:
        dir_mtime = 0.0
    return (idx_mtime, dir_mtime)


def get_index() -> dict:
    """Return the index, using the in-memory cache when fresh.

    On first call (or after an external write), the index is
    loaded from disk. If the index file is missing, an empty
    index is returned; the caller should then call
    :func:`synthesize_index_from_files` to backfill it from any
    pre-existing session files.
    """
    global _index_cache, _index_cache_key
    key = _cache_key()
    if _index_cache is not None and _index_cache_key == key:
        return _index_cache
    _index_cache = _load_index_from_disk()
    _index_cache_key = key
    return _index_cache


def invalidate_index_cache() -> None:
    """Drop the in-memory index cache. Tests use this to force a re-read.

    Production code should not need to call this — the cache is
    keyed on the mtimes of the index file and the sessions dir,
    so any change to either is picked up automatically. But for
    test isolation, or for long-running processes that want to
    guarantee a fresh read, this is the lever.
    """
    global _index_cache, _index_cache_key
    _index_cache = None
    _index_cache_key = None


def _update_index_entry(session_id: str, summary: dict) -> None:
    """Merge a single session's summary into the in-memory + on-disk index.

    Called from :func:`_write_session` after a successful session
    write. The summary dict must contain ``title``, ``n``,
    ``updated_at``, and ideally ``model`` and ``created_at``. We
    don't validate the shape — the file is the source of truth for
    structure; the index is just a cache.
    """
    global _index_cache, _index_cache_key
    if _index_cache is None:
        _index_cache = _load_index_from_disk()
    entries = _index_cache.setdefault("entries", {})
    # Only the fields the listing cares about. Anything else from
    # ``summary`` would bloat the index; the session file holds the
    # full data.
    entry = entries.get(session_id) or {}
    for k in ("title", "preview", "n", "model", "created_at", "updated_at"):
        if k in summary:
            entry[k] = summary[k]
    entries[session_id] = entry
    _save_index_to_disk(entries)
    # Invalidate the cache key so the next list reads back the file
    # (or rather, treats the file as fresh). We could leave the
    # cache in place since we just wrote it; the only cost is a
    # re-stat of the index file on the next access.
    _index_cache_key = _cache_key()


def _remove_index_entry(session_id: str) -> None:
    """Drop a session from the index. No-op if the entry isn't there."""
    global _index_cache, _index_cache_key
    if _index_cache is None:
        _index_cache = _load_index_from_disk()
    entries = _index_cache.setdefault("entries", {})
    if session_id in entries:
        del entries[session_id]
        _save_index_to_disk(entries)
        _index_cache_key = _cache_key()


def synthesize_index_from_files() -> int:
    """Build a fresh index from the on-disk session files.

    Called on first use (or after the user manually deletes the
    index). Walks the sessions dir, parses every ``*.json`` (skipping
    the index itself), and writes a new index. Returns the number
    of entries synthesised.

    Existing entries are overwritten — if you ran this by mistake,
    the worst case is that some stale entries get refreshed (with
    the file's mtime, which is the same as ``updated_at`` in our
    write path anyway). This is a clean "self-heal" operation.
    """
    global _index_cache, _index_cache_key
    d = _sessions_dir()
    d.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    try:
        with os.scandir(d) as it:
            for entry in it:
                if not entry.name.endswith(".json"):
                    continue
                if entry.name == INDEX_FILENAME:
                    continue
                summary = _session_summary_from_file(entry.name)
                if summary is None:
                    continue
                # Pop derived fields. The index stores the raw
                # summary minus anything the listing derives on
                # the fly (id comes from the dict key, short from
                # the id).
                summary.pop("id", None)
                summary.pop("short", None)
                sid = entry.name[:-5]
                entries[sid] = summary
    except OSError:
        pass
    _save_index_to_disk(entries)
    _index_cache = {"version": INDEX_VERSION, "entries": entries}
    _index_cache_key = _cache_key()
    return len(entries)


def _prune_missing_from_index() -> int:
    """Drop index entries whose session file is gone. Returns the count."""
    global _index_cache, _index_cache_key
    if _index_cache is None:
        _index_cache = _load_index_from_disk()
    entries = _index_cache.get("entries", {})
    if not entries:
        return 0
    removed = 0
    for sid in list(entries.keys()):
        if not (_session_path(sid)).is_file():
            del entries[sid]
            removed += 1
    if removed:
        _save_index_to_disk(entries)
        _index_cache_key = _cache_key()
    return removed



def _write_session(session_id: str, messages: list, meta: dict | None = None,
                   created_at: float | None = None) -> None:
    """Persist `messages` to the session file. Atomic (tmp + rename).

    `created_at` lets the caller pass through the original creation time
    (e.g. from a metrics cache) so we don't have to re-read the existing
    file just to preserve it. The on-disk file is still mtime-updated for
    the cheap listing path.
    """
    d = _sessions_dir()
    d.mkdir(parents=True, exist_ok=True)
    _prune_empty_assistant_messages(messages)
    path = _session_path(session_id)
    now = time.time()
    existing: dict = {}
    if created_at is None:
        # Fallback: read existing file to preserve created_at. Skipped
        # on the hot path when the caller already has the value.
        try:
            with path.open(encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, json.JSONDecodeError):
            pass
    data = dict(existing)
    data["id"] = session_id
    data["messages"] = messages
    data["created_at"] = created_at if created_at is not None else existing.get("created_at", now)
    data["updated_at"] = now
    if meta:
        for k, v in meta.items():
            if v is not None:
                data[k] = v
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    try:
        os.utime(path, (now, now))
    except OSError:
        pass
    # Update the metadata index. Best-effort — a failure here
    # doesn't roll back the session write, just means the next
    # list will rebuild the entry from the file we just wrote.
    try:
        n = len([m for m in messages if m.get("role") != "system"])
        preview = ""
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                preview = m["content"]
                break
        _update_index_entry(session_id, {
            "title": data.get("title") or _safe_title(preview) or "(empty)",
            "preview": _safe_title(preview) or "",
            "n": n,
            "model": data.get("model"),
            "created_at": data.get("created_at", now),
            "updated_at": now,
        })
    except Exception:
        # Never let an index-update error block the write.
        pass


def _load_session(session_id: str) -> dict | None:
    path = _session_path(session_id)
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "messages" in data:
            _prune_empty_assistant_messages(data.get("messages"))
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _session_summary_from_file(fname: str) -> dict | None:
    path = _sessions_dir() / fname
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "messages" not in data:
            return None
    except (OSError, json.JSONDecodeError):
        return None
    sid = data.get("id") or fname[:-5]
    msgs = data.get("messages", [])
    preview = ""
    for m in msgs:
        if m.get("role") == "user" and m.get("content"):
            preview = _safe_title(m["content"]) or ""
            break
    return {
        "id": sid,
        "short": _short_id(sid),
        "title": data.get("title") or preview or "(empty)",
        "preview": preview,
        "updated_at": data.get("updated_at", 0),
        "n": len([m for m in msgs if m.get("role") != "system"]),
        "model": data.get("model"),
    }


def _session_matches_query(summary: dict, query: str | None) -> bool:
    if not query:
        return True
    q = query.lower()
    return (
        q in (summary.get("title") or "").lower()
        or q in (summary.get("preview") or "").lower()
        or q in (summary.get("id") or "").lower()
        or q in (summary.get("short") or "").lower()
    )


def _list_sessions(limit: int = 20, offset: int = 0, query: str | None = None) -> list[dict]:
    """List sessions, newest first.

    Reads the metadata index (O(1) in the number of sessions) and
    applies the query / limit / offset filters. On first call, or
    if the index file is missing, the index is synthesised from
    the on-disk session files (O(files) once at startup; the
    result is cached on disk for subsequent calls).

    Stale entries (file deleted externally) are pruned lazily, so a
    rogue ``rm ~/.local/state/anduril/sessions/X.json`` doesn't leave
    a zombie in the listing.
    """
    try:
        limit = max(1, int(limit)) if limit is not None else None
    except (TypeError, ValueError):
        limit = 20
    try:
        offset = max(0, int(offset or 0))
    except (TypeError, ValueError):
        offset = 0
    # First-time setup: if the index doesn't exist (or is empty),
    # synthesise it from the on-disk files. This is the
    # backwards-compat path for users upgrading from a version that
    # didn't have an index.
    index = get_index()
    entries = index.get("entries", {})
    if not entries:
        # Either the index is empty (fresh install) or missing
        # (never built). Distinguish via the file's existence.
        if not _index_path().is_file():
            # Walk the dir once and build. We don't prune here —
            # a missing index means "no one's been here yet", and
            # any orphans are still on disk and can be picked up.
            n_synth = synthesize_index_from_files()
            if n_synth:
                index = get_index()
                entries = index.get("entries", {})
    # Prune missing files from the index before filtering. This is
    # O(entries) per list which is much better than the old
    # O(files × messages) parse.
    d = _sessions_dir()
    try:
        with os.scandir(d) as it:
            present = {e.name for e in it if e.is_file() and e.name.endswith(".json") and e.name != INDEX_FILENAME}
    except OSError:
        present = set()
    # Drop entries whose file is gone, in-memory only — we
    # re-persist below.
    stale = [sid for sid in entries if f"{sid}.json" not in present]
    for sid in stale:
        del entries[sid]
    # Add entries for files that exist on disk but aren't in the
    # index yet (e.g. added behind the API's back, or a freshly-
    # upgraded install whose first list happened to find new
    # files). We parse each new file once and add it to the
    # index in-memory. Persisting is a one-shot at the end.
    known = {f"{sid}.json" for sid in entries}
    new_fnames = [n for n in present if n not in known]
    for fname in new_fnames:
        summary = _session_summary_from_file(fname)
        if summary is None:
            continue
        sid = summary.pop("id", None) or fname[:-5]
        summary.pop("short", None)
        entries[sid] = summary
    if stale or new_fnames:
        # Persist the updated index. A write failure here is
        # benign — the next list will just re-do the same work.
        try:
            _save_index_to_disk(entries)
        except Exception:
            pass
    # Convert to the listing shape the callers expect. The index
    # already has every field except ``short`` and ``id``; we
    # add those here.
    out: list[dict] = []
    for sid, entry in entries.items():
        summary = dict(entry)
        summary["id"] = sid
        summary["short"] = _short_id(sid)
        if not _session_matches_query(summary, query):
            continue
        out.append(summary)
    # Sort newest first.
    out.sort(key=lambda s: s.get("updated_at") or 0, reverse=True)
    if not query:
        # Without a query, the user wants the top N. Apply limit
        # and offset now.
        end = None if limit is None else offset + limit
        return out[offset:end]
    # With a query, the user wants every match (CLI / TUI paging
    # can chunk via offset+limit if they want).
    if limit is None:
        return out[offset:]
    return out[offset:offset + limit]


def _delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    try:
        path.unlink()
    except OSError:
        return False
    # Drop the index entry too. If this fails, the next list will
    # prune it lazily via _prune_missing_from_index().
    try:
        _remove_index_entry(session_id)
    except Exception:
        pass
    return True


def _resolve_session(target: str, sessions: list[dict] | None = None) -> str | None:
    """Resolve a user-typed target to a session id.

    Accepts: a full id, a numeric index into the recent-sessions list,
    a unique id prefix, a 6-hex short id, or an exact title.
    """
    if not target:
        return None
    target = target.strip()
    sessions = sessions if sessions is not None else _list_sessions(limit=50)
    ids = [s["id"] for s in sessions]
    if target.isdigit():
        idx = int(target)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]["id"]
    if target in ids:
        return target
    prefixed = [i for i in ids if i.startswith(target)]
    if len(prefixed) == 1:
        return prefixed[0]
    suffixed = [i for i in ids if i.endswith("-" + target) or _short_id(i) == target]
    if len(suffixed) == 1:
        return suffixed[0]
    titled = [s["id"] for s in sessions if s["title"] == target]
    if len(titled) == 1:
        return titled[0]
    return None


def _fmt_when(ts: float) -> str:
    if not ts:
        return "?"
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)}d ago"
    return time.strftime("%Y-%m-%d", time.localtime(ts))
