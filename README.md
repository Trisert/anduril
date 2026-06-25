# anduril

[![CI](https://github.com/USER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/USER/REPO/actions/workflows/ci.yml)

A minimal coding agent for OpenAI-compatible endpoints. Talks to **llama.cpp**, **vLLM**, **SGLang**, **OpenAI**, or any other server that speaks the chat-completions API. Single runtime dependency: `openai`.

```
$ anduril
anduril · local · prompt…
› write a hello world python script
```

## Features

* **Native tool calling** with a `@tool` decorator that derives JSON Schema from type hints (`Optional`, `Union`, `Literal`, `Annotated`, `list[T]`, `dict[K, V]` all supported) and validates arguments before invoking.
* **Built-in file tools** — `read_file`, `write_file`, `apply_diff`, `search_files` for reliable file editing without `bash` heredocs.
* **Multi-session persistence**. Every chat is auto-saved to `~/.local/state/anduril/sessions/<id>.json` (atomic writes, lightweight metadata index for sub-millisecond listing, resume by id / prefix / short-id / title / index).
* **Syntax highlighting in the log**. Fenced code blocks (``\`\`\`python``, ``\`\`\`js``, …) are tokenized and rendered in colour. Uses [Pygments](https://pygments.org/) when available, falls back to a stdlib regex-based highlighter otherwise.
* **Per-tool risk gating**. Tools marked `dangerous=True` require confirmation; `--approval all|low|medium|high|yolo` controls the threshold.
* **Reasoning-aware streaming**. Dim "reasoning" block separate from the visible answer; recovery for reasoning-only stalls and malformed streams / tool calls.
* **Tool-result sanitization**. Head/tail cap and dedup of ≥3 identical consecutive lines, to prevent context-copying collapse on local models.
* **Esc-to-interrupt** mid-stream and **Esc-to-skip** a tool approval prompt.
* **Stats footer** per turn (TTFT, tok/s, prompt + cached context size, USD cost).
* **Cost tracking** — the status bar shows the session total; `/cost` gives a per-model breakdown. Prices for OpenAI, Anthropic, Google, Mistral, DeepSeek, xAI, and a few common local model names are built in; override any model via `ANDURIL_PRICING_OVERRIDES` (JSON).
* **Cost cap** — `/budget 5.00` sets a session cap; the agent refuses to make further model calls once the cumulative cost reaches it. Relative form (`/budget +1.00`) raises/lowers an existing cap.
* **MCP servers** — any [Model Context Protocol](https://modelcontextprotocol.io) server exposes its tools to anduril via stdio (subprocess) or HTTP (Streamable HTTP transport). Configure in `pyproject.toml` or via the env var.
* **Agent skills** — `add_mcp_server` lets the model register a new MCP server at runtime; the new tools become available in the same session, no restart needed.
* **Per-model system prompts** — `/system <model-with-dash> <text>` sets a system-prompt override that only applies to a specific model. Most-specific (longest-substring) match wins.
* **Multi-line chatbox** with bracketed-paste, history navigation, word-wrap, and a live line + char counter.
* **Image attachments** via `@path`, paste from clipboard, or `Ctrl+Shift+V` in supporting terminals.

## Install

```bash
# With uv (recommended)
uv venv && uv pip install -e .

# With the optional web skill (free search & fetch, no API keys)
uv pip install -e ".[web]"

# With dev tools (ruff, pytest)
uv pip install -e ".[dev,web]"

# Or use make (uses uv under the hood)
make install
```

Nix users get everything via `nix develop` (direnv activates automatically with `use flake`).

Optional syntax-highlighting support (Pygments makes the colours nicer
than the built-in regex highlighter):

```bash
uv pip install pygments
```

## Run

```bash
# Default: looks for a server at http://localhost:8080/v1
anduril

# Custom endpoint and model
anduril --base-url http://localhost:8080/v1 --model qwen2.5-coder-7b

# One-shot mode for scripting
anduril -p "summarize the README in one sentence"

# Resume the most recent session
anduril --resume

# Or pick a specific session
anduril --resume 3
anduril --resume 20250624-145030-a1b2c3
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `ANDURIL_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible base URL |
| `ANDURIL_MODEL` | `local` | Model name |
| `ANDURIL_SYSTEM` | (default prompt) | System prompt override |
| `ANDURIL_HISTORY` | `~/.local/state/anduril/history.jsonl` | History file path |
| `ANDURIL_APPROVAL` | `all` | `all`/`low`/`medium`/`high`/`yolo` |
| `ANDURIL_YOLO` | `0` | Set to `1` to skip all prompts |
| `ANDURIL_SKILLS_PATH` | (empty) | Colon-separated extra skill dirs |
| `ANDURIL_SKILLS_DIR` | `~/.local/share/anduril/skills` | Global skills dir |
| `ANDURIL_HOME` | `~/.local/state/anduril` | Sessions & state root |
| `ANDURIL_SESSIONS_DIR` | `$ANDURIL_HOME/sessions` | Session files dir |
| `ANDURIL_MAX_TOKENS` | `16000` | Per-response token cap |
| `ANDURIL_TOOL_RESULT_CHARS` | `20000` | Sanitization cap for tool outputs |
| `ANDURIL_AUTO_COMPRESS` | `1` | When `1`, automatically summarise older turns once the prompt reaches the model's context window threshold. |
| `ANDURIL_CONTEXT_FRACTION` | `0.8` | Fraction of the model's context window that triggers auto-compression (0.0–1.0). |
| `ANDURIL_PRICING_OVERRIDES` | (none) | JSON object overriding per-model pricing: `'{"my-model": {"input": 5, "output": 15, "cached_input": 1.25}}'` (USD per 1M tokens). |
| `ANDURIL_MCP_CONFIG_DIR` | `~/.local/state/anduril/mcp` | Where the `add_mcp_server` tool writes per-session and global MCP server configs. |

## In-REPL commands

| Command | Action |
|---|---|
| `/quit`, `/exit` | Exit the TUI |
| `/clear` | Drop context, start a new session |
| `/model [name]` | Show or set the model |
| `/system [text]` | Show or set the system prompt |
| `/compress` | Summarize older turns to bound context |
| `/autocompress` | Toggle / configure auto-compression (no arg = toggle, `0.5` = set fraction, `status` = show) |
| `/skills` | List installed skills |
| `/skill <name>` | Show one skill's details |
| `/paste` | Attach an image from the system clipboard |
| `/mcp` | List connected MCP servers and their tools |
| `/cost` | Show per-model cost breakdown for the session |
| `/budget` | Show or set a session cost cap (`/budget [usd|off|+/-]`) |
| `/undo` | Drop the last assistant turn (and any tool chain) |
| `/retry` | Re-run the most recent user message |
| `/edit` | Load the most recent user message into the editor for re-submission |
| `/attachments` | List pasted image attachments |
| `/write [path]` | Write the last assistant turn to a file |
| `/yolo` | Toggle approval prompts |
| `/approval <level>` | Set approval threshold |

## Skills

Skills are Python modules with `@tool`-decorated functions and a `tools = [...]` list. Drop one into `~/.local/share/anduril/skills/`, or use the in-REPL `create_skill` tool to write one mid-session. The bundled `web` skill provides free, no-API-key search and page fetching.

## MCP servers

anduril can talk to any [MCP](https://modelcontextprotocol.io) server via two transports:

* **stdio** (local subprocess): the canonical MCP transport. Configure in `pyproject.toml`:
  ```toml
  [tool.anduril.mcp_servers.servers.fs]
  command = "npx -y @modelcontextprotocol/server-filesystem /tmp"
  ```
* **HTTP** (remote, Streamable HTTP): talk to a remote MCP server over HTTPS. Configure:
  ```toml
  [tool.anduril.mcp_servers.servers.remote]
  url = "https://my-mcp-server.example/mcp"
  ```
  Or pass `Authorization` headers via the API (the pyproject parser doesn't support header dicts; use `MCPServer(name, url, headers={"Authorization": "Bearer ..."})` programmatically).

…or via the `ANDURIL_MCP_SERVERS` env var. For stdio, `name=command` (colon-separated); for HTTP, `name@url`:
```bash
export ANDURIL_MCP_SERVERS="fs=npx -y @mcp/server-fs /tmp:remote@https://my-mcp.example/mcp"
```

At startup, anduril connects to each server, lists its tools, and registers them as native tools (names get the `<server>__<tool>` prefix to avoid collisions). Use `/mcp` in the REPL to see what's connected. The MCP client is a minimal stdlib implementation — no extra runtime dependencies. The model can also register a new server mid-session via the `add_mcp_server` tool.

The HTTP transport uses the [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) protocol from the 2025-03-26 MCP spec: one POST endpoint that responds with either a single JSON body or an SSE stream. The client handles both modes transparently.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

A `Makefile` is provided for the full local CI loop (lint + test + build):

```bash
make install   # one-time: venv + dev/web extras
make lint      # ruff check + ruff format --check
make test      # pytest
make ci        # everything CI runs
```

CI is configured under `.github/workflows/ci.yml`. It runs `ruff check`, `ruff format --check`, the full `pytest` matrix across Python 3.11–3.13, and a build verification (`python -m build` + `twine check`).

## License

MIT.
