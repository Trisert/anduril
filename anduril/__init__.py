"""anduril — minimal agent for OpenAI-compatible endpoints (default: llama.cpp server).

A coding agent that talks to any OpenAI-compatible server (llama.cpp,
vLLM, SGLang, OpenAI itself, …). One runtime dep: ``openai``.

Features
--------

* Native tool calling with a ``@tool`` decorator that derives the JSON
  Schema from the function's type hints (Optional, Union, Literal,
  Annotated, list[T], dict[K, V] all supported) and validates args
  before invoking.
* Multi-session persistence. Every chat is auto-saved to
  ``~/.local/state/anduril/sessions/<id>.json`` (atomic writes, cheap
  mtime-based listing, resume by id / prefix / short-id / title / index,
  ``/sessions`` listing with paging, ``/delete``).
* Per-tool risk gating. ``dangerous=True`` tools require confirmation;
  ``--approval all|low|medium|high|yolo`` controls the threshold
  (CLI / env / ``/approval`` / ``/yolo`` all supported).
* Reasoning-aware streaming. Dim "reasoning" block separate from the
  visible answer; recovery for reasoning-only stalls (force a final
  answer) and for malformed streams / tool calls (retry cleanly).
* Tool-result sanitization. Head/tail cap and dedup of ≥3 identical
  consecutive lines, to prevent context-copying collapse on local models.
* Esc-to-interrupt mid-stream and Esc-to-skip a tool approval prompt.
* Stats footer per turn (TTFT, tok/s, prompt + cached context size).
* Multi-line chatbox with bracketed-paste, history navigation, word-wrap,
  and a live line + char counter.

Run it with::

    python anduril.py

…or::

    python -m anduril

…after setting ``ANDURIL_BASE_URL`` / ``ANDURIL_MODEL`` (or passing
``--base-url`` / ``--model``). One runtime dep: ``openai``.
"""

from __future__ import annotations

# Public re-exports — kept here so ``from anduril import Agent`` etc.
# continues to work after the file split.

# Agent
from anduril.agent import (
    Agent,
    COMPRESS_KEEP,
    UserMessage,
    _ToolCallAggregator,
    _sanitize_tool_result,
    compress,
    parse_text_calls,
    # Undo / retry / re-edit support
    # (Agent.last_user_message, Agent.undo_last_turn, Agent.replay_last_user
    # are methods; nothing extra to re-export here).
)

# Files: `@`-mention picker and multimodal expansion.
from anduril.files import (
    DEFAULT_IGNORE_DIRS,
    IMAGE_EXTS,
    MAX_IMAGE_BYTES,
    MAX_TEXT_CHARS,
    PASTED_IMAGE_DIR,
    expand_mentions,
    find_active_mention,
    fuzzy_match,
    is_image,
    is_text_file,
    list_files,
    mention_query,
    read_image_data_url,
    read_text_file,
    save_pasted_image,
)

# CLI bits (default system prompt, the main() entry point)
from anduril.cli import (
    DEFAULT_SYSTEM as _DEFAULT_SYSTEM,
    _build_agent,
    main as _main,
)

# Context-window sizing & auto-compression trigger.
from anduril.context import (
    DEFAULT_AUTO_COMPRESS,
    DEFAULT_CONTEXT_FRACTION,
    FALLBACK_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    context_window_for,
    estimate_prompt_tokens,
    should_auto_compress,
)

# Syntax highlighting
from anduril.highlight import (
    LANG_ALIASES,
    highlight_code,
    normalize_lang,
)

# MCP (Model Context Protocol) client
from anduril.mcp_client import (
    CLIENT_INFO,
    DEFAULT_TIMEOUT_S,
    MCPError,
    MCPServer,
    MCPProtocolError,
    MCPServerError,
    MCPTimeoutError,
    PROTOCOL_VERSION,
    _HTTPMCPTransport,
    _StdioMCPTransport,
    discover_mcp_tools,
    load_mcp_servers_from_pyproject,
    shutdown_servers,
)

# Pricing / cost tracking
from anduril.pricing import (
    ModelPricing,
    fmt_cost,
    pricing_for,
)

# Metrics
from anduril.metrics import (
    _abbr,
    _precise_abbr,
    _Metrics,
    _normalize_usage,
)

# Sessions
from anduril.sessions import (
    INDEX_FILENAME,
    INDEX_VERSION,
    SESSION_HOME,
    SESSION_LIST_DEFAULT_LIMIT,
    SESSION_LIST_MAX_LIMIT,
    _delete_session,
    _fmt_when,
    _index_path,
    _list_sessions,
    _load_session,
    _new_session_id,
    _prune_empty_assistant_messages,
    _prune_missing_from_index,
    _remove_index_entry,
    _resolve_session,
    invalidate_index_cache,
    _safe_title,
    _session_matches_query,
    _session_path,
    _session_summary_from_file,
    _short_id,
    _update_index_entry,
    _write_session,
    get_index,
    synthesize_index_from_files,
)

# Tool system + default bash tool
from anduril.tools import (
    DEFAULT_FILE_TOOLS,
    RISK_LEVELS,
    RISK_RANK,
    Tool,
    _validate,
    add_mcp_server,
    apply_diff,
    bash,
    create_skill,
    read_file,
    search_files,
    tool,
    write_file,
)

# Skill system
from anduril.skills import (
    discover_skills,
    drain_pending_registrations,
    list_skills,
    register_skill,
    register_tool,
)

# TUI
from anduril.tui import (
    _Editor,
    _init_approval_level,
    _normalize_approval,
    _short_args,
    tui,
)

# Env helpers
from anduril.env import _env_int, _env_str, _env_float


__all__ = [
    # Agent
    "Agent", "COMPRESS_KEEP", "UserMessage", "_ToolCallAggregator",
    "_sanitize_tool_result", "compress", "parse_text_calls",
    # CLI
    "_DEFAULT_SYSTEM", "_build_agent", "_main",
    # Context / auto-compression
    "DEFAULT_AUTO_COMPRESS", "DEFAULT_CONTEXT_FRACTION",
    "FALLBACK_CONTEXT_WINDOW", "MODEL_CONTEXT_WINDOWS",
    "context_window_for", "estimate_prompt_tokens", "should_auto_compress",
    # Syntax highlighting
    "LANG_ALIASES", "highlight_code", "normalize_lang",
    # MCP client
    "CLIENT_INFO", "DEFAULT_TIMEOUT_S", "MCPError", "MCPServer",
    "MCPProtocolError", "MCPServerError", "MCPTimeoutError",
    "PROTOCOL_VERSION", "_HTTPMCPTransport", "_StdioMCPTransport",
    "discover_mcp_tools",
    "load_mcp_servers_from_pyproject", "shutdown_servers",
    # Pricing
    "ModelPricing", "fmt_cost", "pricing_for",
    # Files
    "DEFAULT_IGNORE_DIRS", "IMAGE_EXTS", "MAX_IMAGE_BYTES", "MAX_TEXT_CHARS",
    "PASTED_IMAGE_DIR", "expand_mentions", "find_active_mention",
    "fuzzy_match", "is_image", "is_text_file", "list_files",
    "mention_query", "read_image_data_url", "read_text_file",
    "save_pasted_image",
    # Metrics
    "_abbr", "_precise_abbr", "_Metrics", "_normalize_usage",
    # Sessions
    "SESSION_HOME", "SESSION_LIST_DEFAULT_LIMIT", "SESSION_LIST_MAX_LIMIT",
    "INDEX_FILENAME", "INDEX_VERSION",
    "_delete_session", "_fmt_when", "_index_path", "_list_sessions",
    "_load_session", "_new_session_id", "_prune_empty_assistant_messages",
    "_prune_missing_from_index", "_remove_index_entry", "_resolve_session",
    "invalidate_index_cache",
    "_safe_title", "_session_matches_query", "_session_path",
    "_session_summary_from_file", "_short_id", "_update_index_entry",
    "_write_session", "get_index", "synthesize_index_from_files",
    # Tools
    "Tool", "_validate", "bash", "create_skill", "tool",
    "read_file", "write_file", "apply_diff", "search_files",
    "add_mcp_server",
    "DEFAULT_FILE_TOOLS",
    "RISK_LEVELS", "RISK_RANK",
    # Skills
    "discover_skills", "drain_pending_registrations", "list_skills",
    "register_skill", "register_tool",
    # TUI
    "_Editor", "_init_approval_level", "_normalize_approval", "_short_args", "tui",
    # Env
    "_env_int", "_env_str", "_env_float",
]
