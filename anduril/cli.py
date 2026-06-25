"""CLI entry point and the default system prompt.

The default system prompt is the longest piece of text in the CLI
module. It's exposed as ``_DEFAULT_SYSTEM`` so the TUI/CLI build
agents with it as a fallback when neither ``--system`` nor
``$ANDURIL_SYSTEM`` is set.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys

from anduril.agent import Agent
from anduril.sessions import (
    SESSION_LIST_DEFAULT_LIMIT,
    SESSION_LIST_MAX_LIMIT,
    _fmt_when,
    _list_sessions,
    _resolve_session,
    _sessions_dir,
)
from anduril.tools import Tool, bash, create_skill
from anduril.tui import tui
from anduril.skills import discover_skills


# Default system prompt. The TUI / CLI build agents with this as a
# fallback when neither --system nor $ANDURIL_SYSTEM is set. It nudges
# the model toward concise, deliverable-focused responses — the failure
# mode we keep seeing is models spending the first 200 tokens on "I
# would typically need... let me think about this... I'm unable to..."
# meta-commentary before producing the actual answer.
DEFAULT_SYSTEM = """\
You are anduril, a coding and writing agent that runs in a terminal TUI.

Behavior:
- Be concise. Skip preamble, meta-commentary, and apologies for limitations.
- Just do the task. When asked for code, produce the code directly.
- When asked for a file (HTML, script, config), produce the full file contents in your reply.
- When uncertain between approaches, pick one and proceed; explain the choice briefly only if non-obvious.
- Do not narrate your reasoning or list what you "would typically need". Start with the answer.

Tools:
- File tools (preferred for any file work):
  - `read_file(path, start_line=None, end_line=None)` reads a UTF-8 text file.
    Use the start_line/end_line form for very large files.
  - `write_file(path, content)` overwrites a file (creates parent dirs as
    needed). Use for new files and full rewrites.
  - `apply_diff(path, old_text, new_text, replace_all=False)` is the safe
    way to edit an existing file. `old_text` must match exactly once
    (unless replace_all=True). Whitespace is matched literally.
  - `search_files(pattern, path=".", glob=None, case_sensitive=False, max_results=100)`
    finds every line containing `pattern` under `path`. Output is
    `relative/path:LINE: matched line`.
  These are the dedicated file tools — prefer them over `bash` heredocs.
- `bash` runs shell commands. Approval is required for each call unless the user has set --yolo.
  To run multiple commands without re-prompting, prefix with --yolo or chain with &&.
  Use bash for things the file tools can't do: process management, shell
  pipelines, git, package managers, etc.
- `create_skill` (name, code, description='', persistent=False) writes a new Python skill to
  disk and registers its tools in this session. Use it when you need a tool that doesn't
  exist yet. Skills are Python modules with `@tool`-decorated functions and a `tools = [...]`
  list. By default they are session-scoped (live in
  ~/.local/state/anduril/skills/$ANDURIL_SESSION_ID/); pass `persistent=True` to write to
  ~/.local/share/anduril/skills/ where they survive restarts.
- To list available tools, run `compgen -c` (bash completion) or `type -a` for builtins.

Attaching files to a turn:
- Type `@` to open a fuzzy file picker (TAB completes, Enter picks + inserts).
- Image files (`.png .jpg .jpeg .gif .webp .bmp`) are sent inline as image attachments.
- Text files are inlined as `[file: <path>]` fenced blocks so the content is in context.
- Pasting an image from the clipboard (Kitty / WezTerm / iTerm2) is also
  supported: the terminal sends the image bytes inline and anduril saves
  them to `~/.local/state/anduril/images/` and attaches them.
- `@` is NOT triggered inside an identifier (so `user@host.com` is left alone).

Output format:
- Code blocks should be marked with the language (` ```python `, ` ```html `) so the TUI can syntax-highlight.
- Keep prose responses under a few sentences unless the user asked for explanation.
"""


def _default_tools() -> list[Tool]:
    from anduril.tools import DEFAULT_FILE_TOOLS, add_mcp_server
    tools: list[Tool] = [
        bash, create_skill, add_mcp_server,
        *DEFAULT_FILE_TOOLS, *discover_skills(),
    ]
    # MCP servers from pyproject.toml + ANDURIL_MCP_SERVERS env. Failures
    # are logged to stderr by discover_mcp_tools; the agent still
    # starts with the native tool set so a broken MCP server doesn't
    # brick the rest of the session.
    try:
        from anduril.mcp_client import (
            MCPServer,
            discover_mcp_tools,
            load_mcp_servers_from_pyproject,
        )
        servers = load_mcp_servers_from_pyproject()
        # ANDURIL_MCP_SERVERS is a colon-separated list of
        # ``name=command`` pairs (e.g.
        # ``fs=npx -y @mcp/server-fs /tmp``) for stdio
        # servers, or ``name@url`` pairs (e.g.
        # ``remote@https://my-mcp.example/mcp``) for HTTP
        # servers. The two separators are deliberately
        # distinct so a single env var can mix both kinds
        # without ambiguity.
        extra = os.environ.get("ANDURIL_MCP_SERVERS", "").strip()
        if extra:
            for spec in extra.split(":"):
                spec = spec.strip()
                if not spec:
                    continue
                if "@" in spec and "=" not in spec.split("@", 1)[0]:
                    # HTTP: ``name@url``
                    name, _, url = spec.partition("@")
                    servers.append(MCPServer(name=name.strip(), url=url.strip()))
                elif "=" in spec:
                    # stdio: ``name=command``
                    name, _, cmd = spec.partition("=")
                    servers.append(
                        MCPServer(name=name.strip(), command=cmd.strip())
                    )
        if servers:
            tools.extend(discover_mcp_tools(servers))
    except Exception as e:
        # Never let MCP setup break the agent.
        print(f"MCP setup failed: {e}", file=sys.stderr)
    return tools


def _build_agent(args: argparse.Namespace) -> Agent:
    system = args.system or os.environ.get("ANDURIL_SYSTEM")
    if not system:
        system = DEFAULT_SYSTEM
    return Agent(
        model=args.model or os.environ.get("ANDURIL_MODEL", "local"),
        system=system,
        base_url=args.base_url or os.environ.get("ANDURIL_BASE_URL"),
        tools=tuple(_default_tools()),
        history_path=args.history or os.environ.get("ANDURIL_HISTORY"),
    )


def _print_session_list(sessions: list[dict], start_index: int = 1,
                        current_id: str | None = None) -> None:
    for offset, s in enumerate(sessions):
        i = start_index + offset
        title = s["title"] or "(empty)"
        if current_id is not None:
            mark = "●" if s["id"] == current_id else f"{i:>3}"
            prefix = f"  {mark} "
        else:
            prefix = f"  {i:>3}  "
        sid = s.get("id") or ""
        print(f"{prefix}{sid}  {title}")
        meta = [f"{s['n']} msg", _fmt_when(s["updated_at"])]
        if s.get("model"):
            meta.append(f"model {s['model']}")
        print(f"       {' · '.join(meta)}")


def _session_list_page_options(args: list, default_limit: int = SESSION_LIST_DEFAULT_LIMIT):
    """Parse paging flags and leave everything else as the search query."""
    page = 1
    limit = default_limit
    query_parts: list[str] = []
    warnings: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--page", "-p"):
            if i + 1 >= len(args):
                warnings.append(f"ignored missing value for {a}")
                i += 1
                continue
            try:
                page = int(args[i + 1])
                if page < 1:
                    raise ValueError
            except (TypeError, ValueError):
                warnings.append(f"ignored invalid page: {args[i + 1]!r}")
            i += 2
            continue
        if a.startswith("--page="):
            try:
                page = int(a.split("=", 1)[1])
                if page < 1:
                    raise ValueError
            except (TypeError, ValueError):
                warnings.append(f"ignored invalid page: {a}")
            i += 1
            continue
        if a in ("--limit", "-n"):
            if i + 1 >= len(args):
                warnings.append(f"ignored missing value for {a}")
                i += 1
                continue
            try:
                limit = int(args[i + 1])
                if limit < 1:
                    raise ValueError
            except (TypeError, ValueError):
                warnings.append(f"ignored invalid limit: {args[i + 1]!r}")
            i += 2
            continue
        if a.startswith("--limit="):
            try:
                limit = int(a.split("=", 1)[1])
                if limit < 1:
                    raise ValueError
            except (TypeError, ValueError):
                warnings.append(f"ignored invalid limit: {a}")
            i += 1
            continue
        query_parts.append(a)
        i += 1
    limit = max(1, min(limit, SESSION_LIST_MAX_LIMIT))
    query = " ".join(query_parts).strip() or None
    return query, page, limit, warnings


def _print_transcript(messages: list, max_chars: int = 120) -> None:
    """Render a session's message history as a one-line-per-message recap."""
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue
        content = m.get("content")
        if content is None and m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}(...)" for tc in m["tool_calls"]
            )
            content = f"→ {calls}"
        content = (content or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        print(f"  {role:>9}  {content}")


def _cli_sessions(args: list) -> bool:
    """`anduril sessions [query] [--page N] [--limit N]` — list and exit."""
    if not args:
        return False
    if args[0] not in ("sessions", "--sessions", "ls", "list"):
        return False
    query, page, limit, warnings = _session_list_page_options(args[1:])
    offset = (page - 1) * limit
    fetched = _list_sessions(limit=limit + 1, offset=offset, query=query)
    sessions = fetched[:limit]
    has_next = len(fetched) > limit
    for w in warnings:
        print(f"  {w}")
    if not sessions:
        if query:
            print(f"  no sessions matching {query!r} on page {page}")
        elif page > 1:
            print(f"  no sessions on page {page}  ({_sessions_dir()})")
        else:
            print(f"  no saved sessions yet  ({_sessions_dir()})")
        return True
    where = f" matching {query!r}" if query else ""
    print(f"  sessions{where} · page {page} · {_sessions_dir()}")
    _print_session_list(sessions, start_index=offset + 1)
    print("  resume with: anduril --resume <n|short-id|title>")
    if has_next:
        q = f" {shlex.quote(query)}" if query else ""
        print(f"  next page: anduril sessions{q} --page {page + 1} --limit {limit}")
    return True


def _session_id_from_args() -> str | None:
    """Resolve a starting session id from --resume/-r and --session flags."""
    args = sys.argv[1:]
    out: str | None = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--resume", "-r"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                target = args[i + 1]
                sessions = _list_sessions(limit=50)
                out = _resolve_session(target, sessions) or target
                i += 2
            else:
                sessions = _list_sessions(limit=1)
                out = sessions[0]["id"] if sessions else None
                i += 1
            continue
        if a == "--session" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
            continue
        i += 1
    return out


def main() -> None:
    # `anduril sessions [query]` — list saved sessions and exit. Checked
    # before argparse so the verb doesn't get rejected as a positional arg.
    if _cli_sessions(sys.argv[1:]):
        return

    parser = argparse.ArgumentParser(
        description="anduril — minimal agent for OpenAI-compatible endpoints",
    )
    parser.add_argument("--model", help="model name (or ANDURIL_MODEL)")
    parser.add_argument("--base-url", help="base URL (or ANDURIL_BASE_URL)")
    parser.add_argument("--system", help="system prompt (or ANDURIL_SYSTEM)")
    parser.add_argument("--history", help="history file (or ANDURIL_HISTORY)")
    parser.add_argument(
        "--resume", "-r", nargs="?", const="__bare__", default=None,
        help="resume a session: --resume <n|id|short-id|title>, or --resume "
             "(alone) for the most recent",
    )
    parser.add_argument("--session", help="force a specific session id")
    parser.add_argument("--no-stream", action="store_true", help="disable response streaming")
    parser.add_argument(
        "--yolo", action="store_true",
        help="skip all tool approval prompts (same as --approval yolo)",
    )
    parser.add_argument(
        "--approval", default=None,
        help="approval threshold: all | low | medium | high | yolo",
    )
    parser.add_argument(
        "--bash-confirm", action="store_true",
        help="(legacy) mark bash as dangerous — the default is now True",
    )
    parser.add_argument(
        "--prompt", "-p", default=None,
        help="one-shot mode: run this single prompt, print the response, and exit "
             "(useful for scripting and for the agent to call itself: "
             "`anduril -p '...'`).",
    )
    args = parser.parse_args()

    agent = _build_agent(args)

    # Optional resume.
    if args.resume is not None:
        target = args.resume
        if target == "__bare__":
            sessions = _list_sessions(limit=1)
            if sessions:
                agent.load_session(sessions[0]["id"])
            else:
                print("  no saved sessions to resume — starting fresh")
        else:
            sessions = _list_sessions(limit=50)
            sid = _resolve_session(target, sessions)
            if sid:
                agent.load_session(sid)
            else:
                print(f"  no session matching {target!r} — starting fresh")

    if args.bash_confirm:
        # Legacy flag — bash is now dangerous by default; this is a no-op
        # kept for backwards compatibility.
        pass

    # One-shot mode: run a single prompt, print the response, exit.
    # Works in any environment (TTY or not) and never enters the REPL.
    if args.prompt is not None:
        _oneshot(agent, args.prompt, stream=not args.no_stream)
        return

    # If a non-tty environment, fall back to a plain line-based REPL.
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        _plain_repl(agent)
        return

    tui(agent)


def _plain_repl(agent: Agent) -> None:
    """Fallback REPL for non-TTY environments (piped, scripts, CI)."""
    print(f"anduril · {agent.model} · {agent.client.base_url}  (non-TTY mode; Ctrl-D to quit)")
    while True:
        try:
            line = input("› ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        try:
            agent.run(line, stream=True)
        except KeyboardInterrupt:
            print("(interrupted)")
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")


def _oneshot(agent: Agent, prompt: str, stream: bool = True) -> None:
    """Run a single prompt, print the response, exit. No REPL.

    Designed for scripting and for the agent to invoke itself via::

        anduril -p "summarize the README"

    Approval prompts for dangerous tools are honored unless --yolo was
    passed. With stream=False the response is printed in one block.
    """
    try:
        response = agent.run(prompt, stream=stream)
    except KeyboardInterrupt:
        print("(interrupted)")
        return
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    if response:
        print(response)
