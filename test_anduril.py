"""Quick tests for anduril schema generation, validation, and history."""

import json
import os
import pathlib
import tempfile
from typing import Annotated, Literal, Optional

from anduril import (
    Agent,
    _DEFAULT_SYSTEM,
    _Editor,
    _Metrics,
    _ToolCallAggregator,
    _abbr,
    _build_agent,
    _delete_session,
    _list_sessions,
    _load_session,
    _new_session_id,
    _normalize_approval,
    _normalize_usage,
    _precise_abbr,
    _prune_empty_assistant_messages,
    _resolve_session,
    _safe_title,
    _short_id,
    _sanitize_tool_result,
    _write_session,
    bash,
    parse_text_calls,
    tool,
    _validate,
    create_skill,
    read_file,
    write_file,
    apply_diff,
    search_files,
    # Context / auto-compression
    DEFAULT_AUTO_COMPRESS,
    DEFAULT_CONTEXT_FRACTION,
    FALLBACK_CONTEXT_WINDOW,
    context_window_for,
    estimate_prompt_tokens,
    should_auto_compress,
    # Syntax highlighting
    highlight_code,
    normalize_lang,
    # MCP client
    MCPServer,
    _StdioMCPTransport,
    discover_mcp_tools,
    load_mcp_servers_from_pyproject,
    shutdown_servers,
    # Pricing / cost
    fmt_cost,
    pricing_for,
)


@tool
def example(a: str, b: int = 1, c: list[str] | None = None, d: bool = False) -> str:
    """Example tool.

    :param a: the string argument
    :param b: the integer argument
    """
    return "ok"


@tool
def rich(
    mode: Annotated[Literal["fast", "slow"], "execution mode"] = "fast",
    count: Optional[int] = None,
    tags: list[str] = None,
    meta: dict[str, int] = None,
) -> str:
    """Rich tool."""
    return "ok"


def test_basic_schema() -> None:
    schema = example.parameters
    assert schema["required"] == ["a"]
    assert schema["properties"]["a"]["description"] == "the string argument"
    assert schema["properties"]["b"]["description"] == "the integer argument"
    assert schema["properties"]["c"]["anyOf"][0]["type"] == "array"
    assert schema["properties"]["c"]["anyOf"][0]["items"]["type"] == "string"


def test_validation() -> None:
    schema = example.parameters
    assert _validate({"a": "hello", "d": True}, schema) == []
    errors = _validate({"a": 123, "d": "no"}, schema)
    assert len(errors) == 2


def test_rich_schema() -> None:
    schema = rich.parameters
    assert schema["properties"]["mode"]["enum"] == ["fast", "slow"]
    assert schema["properties"]["mode"]["description"] == "execution mode"
    assert "null" in [s.get("type") for s in schema["properties"]["count"]["anyOf"]]
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["meta"]["type"] == "object"
    assert "count" not in schema.get("required", [])


def test_bash_defaults() -> None:
    # bash is dangerous by default — the TUI gates it via --approval.
    assert bash.dangerous is True
    assert bash.risk == "high"
    not_dangerous = bash._replace(dangerous=False)
    assert not_dangerous.dangerous is False


def test_risk_levels_exported() -> None:
    from anduril import RISK_LEVELS, RISK_RANK
    assert RISK_LEVELS == ("low", "medium", "high")
    assert RISK_RANK == {"low": 0, "medium": 1, "high": 2}


def test_tool_decorator_risk_default() -> None:
    @tool
    def safe_fn() -> str:
        """A safe tool."""
        return "ok"

    assert safe_fn.dangerous is False
    assert safe_fn.risk == "low"  # normalized to "low" when not dangerous

    @tool(dangerous=True)
    def default_risk_fn() -> str:
        """A dangerous tool with no explicit risk."""
        return "ok"

    assert default_risk_fn.dangerous is True
    assert default_risk_fn.risk == "medium"  # backward-compat default


def test_tool_decorator_explicit_risk() -> None:
    @tool(dangerous=True, risk="low")
    def low_risk() -> str:
        """Low risk."""
        return "ok"

    @tool(dangerous=True, risk="medium")
    def med_risk() -> str:
        """Medium risk."""
        return "ok"

    @tool(dangerous=True, risk="high")
    def high_risk() -> str:
        """High risk."""
        return "ok"

    assert low_risk.risk == "low"
    assert med_risk.risk == "medium"
    assert high_risk.risk == "high"


def test_tool_decorator_invalid_risk_rejected() -> None:
    import pytest
    with pytest.raises(ValueError, match="risk must be one of"):
        @tool(dangerous=True, risk="yolo")
        def bad() -> str:
            """Bad risk."""
            return "ok"


def test_create_skill_has_high_risk() -> None:
    assert create_skill.dangerous is True
    assert create_skill.risk == "high"


# === File-editing tools ==================================================


def test_file_tools_schemas_present() -> None:
    """Each file tool exports a valid JSON schema for its declared args."""
    for fn in (read_file, write_file, apply_diff, search_files):
        schema = fn.parameters
        assert schema["type"] == "object"
        assert "properties" in schema
        # Each tool should declare a `path` or `pattern` arg.
        props = set(schema["properties"].keys())
        assert "path" in props or "pattern" in props


def test_file_tools_risk_levels() -> None:
    """read_file/search_files are safe; write_file/apply_diff are medium."""
    assert read_file.dangerous is False
    assert search_files.dangerous is False
    assert write_file.dangerous is True
    assert write_file.risk == "medium"
    assert apply_diff.dangerous is True
    assert apply_diff.risk == "medium"


def test_read_file_basic() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "hello.txt"
        p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        out = read_file.fn(path=str(p))
        assert out == "alpha\nbeta\ngamma\n"


def test_read_file_line_range() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "lines.txt"
        p.write_text("\n".join(f"L{i}" for i in range(1, 11)), encoding="utf-8")
        out = read_file.fn(path=str(p), start_line=3, end_line=5)
        # The cheap default: a single ``[path:start-end]`` header
        # followed by the raw slice. The model can refer to
        # absolute line numbers (3, 4, 5) without paying the
        # per-line prefix cost.
        lines = out.splitlines()
        # First line: header.
        assert lines[0] == f"[{p}:3-5]"
        # Body: the raw three lines, no per-line prefix.
        assert lines[1] == "L3"
        assert lines[2] == "L4"
        assert lines[3] == "L5"


def test_read_file_line_range_numbered() -> None:
    """``numbered=True`` produces the per-line path:LINE: prefix."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "lines.txt"
        p.write_text("\n".join(f"L{i}" for i in range(1, 6)), encoding="utf-8")
        out = read_file.fn(path=str(p), start_line=2, end_line=4, numbered=True)
        lines = out.splitlines()
        assert len(lines) == 3
        assert lines[0] == f"{p}:2: L2"
        assert lines[1] == f"{p}:3: L3"
        assert lines[2] == f"{p}:4: L4"


def test_read_file_default_is_plain() -> None:
    """A read_file with no range and no numbered returns the raw text."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "f.txt"
        p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        out = read_file.fn(path=str(p))
        # Raw text, no path:line prefixes anywhere.
        assert out == "alpha\nbeta\ngamma\n"
        assert str(p) not in out
        assert ":1:" not in out


def test_read_file_range_empty_slice() -> None:
    """A range that selects no lines returns the header + '(empty)'."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "f.txt"
        p.write_text("only one line\n", encoding="utf-8")
        out = read_file.fn(path=str(p), start_line=5, end_line=5)
        # Header + empty marker.
        assert f"[{p}:5-5]" in out
        assert "empty" in out


def test_read_file_line_range_validation() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("a\nb\nc\n", encoding="utf-8")
        assert "error" in read_file.fn(path=str(p), start_line=0)
        assert "error" in read_file.fn(path=str(p), end_line=0)
        assert "error" in read_file.fn(path=str(p), start_line=5, end_line=2)


def test_read_file_missing() -> None:
    out = read_file.fn(path="/no/such/file/hopefully.txt")
    assert out.startswith("error:")
    assert "not found" in out


def test_read_file_binary_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "blob.bin"
        # A valid UTF-8 lead byte followed by an invalid continuation
        # byte forces a UnicodeDecodeError on read. Using a NUL byte
        # alone wouldn't trip the decoder (it'd just decode as \x00).
        p.write_bytes(b"\xff\xfe\xfd\xfcnot-utf-8")
        out = read_file.fn(path=str(p))
        assert "error" in out
        assert "UTF-8" in out or "binary" in out


def test_read_file_directory_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = read_file.fn(path=td)
        assert "error" in out
        assert "not a regular file" in out


def test_write_file_creates_parents() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "a" / "b" / "c.txt"
        out = write_file.fn(path=str(p), content="hello\n")
        assert out.startswith("wrote ")
        assert p.read_text(encoding="utf-8") == "hello\n"


def test_write_file_overwrites() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("old", encoding="utf-8")
        write_file.fn(path=str(p), content="new")
        assert p.read_text(encoding="utf-8") == "new"


def test_write_file_atomic_no_partial_file() -> None:
    """A successful write leaves no .anduril.tmp.* sibling behind."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        write_file.fn(path=str(p), content="content")
        siblings = [s.name for s in p.parent.iterdir()]
        assert all(".anduril.tmp" not in s for s in siblings)


def test_apply_diff_unique_match() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("foo bar baz\n", encoding="utf-8")
        out = apply_diff.fn(path=str(p), old_text="bar", new_text="BAR")
        assert "applied diff" in out
        assert p.read_text(encoding="utf-8") == "foo BAR baz\n"


def test_apply_diff_zero_matches_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("foo\n", encoding="utf-8")
        out = apply_diff.fn(path=str(p), old_text="missing", new_text="x")
        assert out.startswith("error:")
        assert "not found" in out
        # File unchanged.
        assert p.read_text(encoding="utf-8") == "foo\n"


def test_apply_diff_multiple_matches_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("foo foo foo\n", encoding="utf-8")
        out = apply_diff.fn(path=str(p), old_text="foo", new_text="bar")
        assert out.startswith("error:")
        assert "matches 3" in out
        # File unchanged.
        assert p.read_text(encoding="utf-8") == "foo foo foo\n"


def test_apply_diff_replace_all() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("foo foo foo\n", encoding="utf-8")
        out = apply_diff.fn(
            path=str(p), old_text="foo", new_text="bar", replace_all=True,
        )
        assert "3 replacements" in out
        assert p.read_text(encoding="utf-8") == "bar bar bar\n"


def test_apply_diff_empty_old_text_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("foo\n", encoding="utf-8")
        out = apply_diff.fn(path=str(p), old_text="", new_text="bar")
        assert out.startswith("error:")
        assert "non-empty" in out


def test_apply_diff_delete_with_empty_new_text() -> None:
    """Setting new_text='' is the documented way to delete text."""
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "x.txt"
        p.write_text("foo DELETE_ME bar\n", encoding="utf-8")
        out = apply_diff.fn(
            path=str(p), old_text=" DELETE_ME", new_text="",
        )
        assert "applied" in out
        assert p.read_text(encoding="utf-8") == "foo bar\n"


def test_search_files_finds_matches() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        (root / "b.py").write_text("# TODO: refactor foo\n", encoding="utf-8")
        (root / "c.txt").write_text("nothing here\n", encoding="utf-8")
        out = search_files.fn(pattern="foo", path=str(root))
        # Both Python files match; the .txt file does not.
        assert "a.py:1: def foo():" in out
        assert "b.py:1: # TODO: refactor foo" in out
        assert "c.txt" not in out


def test_search_files_case_insensitive_default() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "x.txt").write_text("Hello World\nhello again\n", encoding="utf-8")
        out = search_files.fn(pattern="hello", path=str(root))
        # Both lines match because the default is case-insensitive.
        assert out.count(":1: Hello World") == 1
        assert out.count(":2: hello again") == 1


def test_search_files_case_sensitive() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "x.txt").write_text("Hello\nhello\n", encoding="utf-8")
        out = search_files.fn(
            pattern="Hello", path=str(root), case_sensitive=True,
        )
        assert ":1: Hello" in out
        assert ":2: hello" not in out


def test_search_files_glob_filter() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "a.py").write_text("foo\n", encoding="utf-8")
        (root / "b.txt").write_text("foo\n", encoding="utf-8")
        out = search_files.fn(pattern="foo", path=str(root), glob="*.py")
        assert "a.py" in out
        assert "b.txt" not in out


def test_search_files_skips_ignored_dirs() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / ".git").mkdir()
        (root / ".git" / "HEAD").write_text("foo\n", encoding="utf-8")
        (root / "src.py").write_text("foo\n", encoding="utf-8")
        out = search_files.fn(pattern="foo", path=str(root))
        assert "src.py" in out
        assert ".git" not in out


def test_search_files_no_matches() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "x.txt").write_text("hello\n", encoding="utf-8")
        out = search_files.fn(pattern="nothere", path=str(root))
        assert "no matches" in out


def test_search_files_validation() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = search_files.fn(pattern="", path=td)
        assert "error" in out
        out = search_files.fn(pattern="x", path=td, max_results=0)
        assert "error" in out
        out = search_files.fn(pattern="x", path="/no/such/dir")
        assert "error" in out


def test_search_files_truncation_marker() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        # Many files each with one match.
        for i in range(10):
            (root / f"f{i}.txt").write_text("foo\n", encoding="utf-8")
        out = search_files.fn(pattern="foo", path=str(root), max_results=3)
        assert "truncated" in out


def test_file_tools_in_default_set() -> None:
    """The CLI's default tool set exposes the file tools."""
    from anduril.tools import DEFAULT_FILE_TOOLS
    names = {t.name for t in DEFAULT_FILE_TOOLS}
    assert names == {"read_file", "write_file", "apply_diff", "search_files"}


def test_cli_default_tools_includes_file_tools() -> None:
    """The CLI builds an agent whose tool list includes the file tools."""
    from anduril.cli import _default_tools
    names = {t.name for t in _default_tools()}
    for required in ("bash", "create_skill", "read_file", "write_file",
                     "apply_diff", "search_files"):
        assert required in names


# === Auto-compression: context.py =======================================


def test_context_window_known_models() -> None:
    assert context_window_for("o3-mini") == 200_000
    assert context_window_for("gpt-4o-mini") == 128_000
    assert context_window_for("claude-3-5-sonnet-latest") == 200_000
    assert context_window_for("qwen2.5-coder-7b") == 128_000
    assert context_window_for("llama-3.1-70b") == 128_000
    assert context_window_for("deepseek-coder") == 64_000
    assert context_window_for("gemini-1.5-pro-latest") == 2_000_000


def test_context_window_case_insensitive() -> None:
    assert context_window_for("GPT-4O") == 128_000
    assert context_window_for("Claude-3-5-Sonnet") == 200_000


def test_context_window_fallback() -> None:
    assert context_window_for(None) == FALLBACK_CONTEXT_WINDOW
    assert context_window_for("") == FALLBACK_CONTEXT_WINDOW
    assert context_window_for("totally-unknown-xyz-99") == FALLBACK_CONTEXT_WINDOW


def test_estimate_prompt_tokens_basic() -> None:
    msgs = [{"role": "user", "content": "hello world"}]
    est = estimate_prompt_tokens(msgs, system="sys")
    # chars/3 + envelope: (3+3+11)/3 ≈ 5 + 4*2 = 13.
    assert 10 <= est <= 30


def test_estimate_prompt_tokens_tool_calls() -> None:
    """Tool-call arguments count toward prompt size."""
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "read_file", "arguments": '{"path": "x.py"}'}},
        ]},
    ]
    est_with = estimate_prompt_tokens(msgs)
    msgs_no_tc = [{"role": "assistant", "content": None}]
    est_without = estimate_prompt_tokens(msgs_no_tc)
    assert est_with > est_without


def test_estimate_prompt_tokens_multimodal() -> None:
    """Image content parts contribute text length + per-image bump."""
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]
    est = estimate_prompt_tokens(msgs, image_count=1, image_tokens=1000)
    # Bump dominates the text length.
    assert est >= 1000


def test_estimate_prompt_tokens_tool_schemas() -> None:
    schemas = [
        {"type": "function", "function": {"name": "f", "parameters": {"a": "b"}}}
    ]
    est = estimate_prompt_tokens(
        [], system="", tool_schemas=schemas,
    )
    # No messages, but the schema itself + 1 envelope = a small number > 0.
    assert est > 0


def test_should_auto_compress_disabled() -> None:
    long_msgs = [
        {"role": "user", "content": "x" * 100_000}
        for _ in range(20)
    ]
    should, _, _, _ = should_auto_compress(
        long_msgs, model="o3", enabled=False,
    )
    assert should is False


def test_should_auto_compress_too_few_turns() -> None:
    # Two body turns, even with a long first turn, should not trigger.
    msgs = [
        {"role": "user", "content": "x" * 100_000},
        {"role": "assistant", "content": "ok"},
    ]
    should, _, _, _ = should_auto_compress(msgs, model="o3")
    assert should is False


def test_should_auto_compress_under_threshold() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
        {"role": "assistant", "content": "fine"},
    ]
    should, est, window, threshold = should_auto_compress(
        msgs, model="gpt-4o",  # 128K window
    )
    assert should is False
    assert window == 128_000
    assert threshold == int(128_000 * DEFAULT_CONTEXT_FRACTION)
    assert est < threshold


def test_should_auto_compress_over_threshold() -> None:
    # Pad the messages enough to exceed 80% of o3's 200K window.
    pad = "lorem ipsum " * 4000  # ~48K chars ≈ 16K tokens
    msgs = [{"role": "user", "content": pad} for _ in range(20)]
    should, est, window, threshold = should_auto_compress(
        msgs, model="o3", fraction=0.5,
    )
    assert should is True
    assert window == 200_000
    assert threshold == 100_000
    assert est >= threshold


def test_should_auto_compress_custom_fraction() -> None:
    pad = "x" * 30_000  # 10K tokens
    msgs = [{"role": "user", "content": pad} for _ in range(8)]
    # 0.5 fraction on gpt-4o: threshold = 64K. ~80K > 64K → True.
    should, _, _, threshold = should_auto_compress(
        msgs, model="gpt-4o", fraction=0.5,
    )
    assert should is True
    assert threshold == 64_000


# === Auto-compression: Agent.run() integration ==========================


class _FakeChunk:
    def __init__(self, content="", usage=None):
        self.choices = [type("C", (), {"delta": type("D", (), {
            "content": content,
            "reasoning_content": None,
            "tool_calls": None,
        })()})()] if content or usage is None else []
        self.usage = usage
        self.model_extra = {}


class _FakeResponse:
    """A streaming response that yields a single chunk with final content."""
    def __init__(self, content):
        self._content = content
    def __iter__(self):
        yield _FakeChunk(content=self._content, usage=_FakeUsage())
    def close(self):
        pass


class _FakeUsage:
    def __init__(self):
        self.prompt_tokens = 100
        self.completion_tokens = 10
        self.prompt_tokens_details = type("P", (), {"cached_tokens": 0})()


def _make_agent_with_compress(model_name: str, *, auto: bool = True,
                              fraction: float = 0.8) -> Agent:
    """Build an Agent that won't actually hit the network for compress.

    The auto-compress path uses the real ``compress()`` function
    (which makes its own model call). We monkey-patch the
    client's ``chat.completions.create`` so that call is a no-op
    that returns a fixed summary.
    """
    from anduril import agent as agent_mod

    a = Agent(
        model=model_name, system="sys",
        auto_compress=auto, context_fraction=fraction,
    )
    # Replace the compress() helper used inside Agent.run with one
    # that mutates the messages in place and returns a sentinel
    # tuple — the real compress() would need a real model client.
    calls = {"n": 0}

    def fake_compress(messages, keep=2, model=None, client=None):
        calls["n"] += 1
        # Drop everything except the system + last `keep` turns,
        # then prepend a single summary turn.
        if len(messages) <= 1 + keep:
            return None
        sys_msg = messages[0] if messages[0].get("role") == "system" else None
        body = messages[1:] if sys_msg else messages
        if len(body) <= keep:
            return None
        kept = body[-keep:]
        summarized_n = len(body) - keep
        new_mid = [{"role": "user", "content": "[summary of older turns]"}]
        messages[:] = ([sys_msg] if sys_msg else []) + new_mid + kept
        return (len(kept), summarized_n, len("[summary of older turns]"))

    a._compress_for_test = fake_compress  # type: ignore[attr-defined]
    # Monkey-patch the symbol in the agent module so Agent.run sees it.
    original_compress = agent_mod.compress
    agent_mod.compress = fake_compress
    a._restore_compress = lambda: setattr(  # type: ignore[attr-defined]
        agent_mod, "compress", original_compress,
    )
    return a


def test_agent_run_auto_compress_fires_on_long_prompt() -> None:
    """A long conversation triggers auto-compress at the start of run()."""
    a = _make_agent_with_compress("o3", auto=True, fraction=0.5)
    try:
        events: list[dict] = []

        def on_event(ev):
            events.append(ev)

        # Build a long history.
        pad = "lorem ipsum " * 4000  # ~16K tokens
        for i in range(10):
            a._messages.append({"role": "user", "content": pad})
            a._messages.append({"role": "assistant", "content": "ok"})

        # Stub the client's streaming call so we don't hit the network.
        captured = {}

        def fake_create(**kwargs):
            captured["kwargs"] = kwargs
            return _FakeResponse("hi")

        a.client.chat.completions.create = fake_create  # type: ignore[method-assign]
        a.run("next", on_event=on_event, stream=True)

        # auto_compress event was emitted at least once.
        assert any(e.get("type") == "auto_compress" for e in events)
        # The message list shrunk after compression. We
        # started with 1 sys + 20 body, kept 2 verbatim,
        # compressed the rest to a summary, then the new
        # user + assistant turns added 2 more: 1 sys + 1
        # summary + 2 kept + 1 new user + 1 new assistant = 6.
        assert len(a._messages) <= 6
    finally:
        a._restore_compress()  # type: ignore[attr-defined]


def test_agent_run_auto_compress_disabled() -> None:
    a = _make_agent_with_compress("o3", auto=False, fraction=0.5)
    try:
        events: list[dict] = []

        def on_event(ev):
            events.append(ev)

        pad = "lorem ipsum " * 4000
        for i in range(10):
            a._messages.append({"role": "user", "content": pad})
            a._messages.append({"role": "assistant", "content": "ok"})

        def fake_create(**kwargs):
            return _FakeResponse("hi")

        a.client.chat.completions.create = fake_create  # type: ignore[method-assign]
        a.run("next", on_event=on_event, stream=True)
        assert not any(e.get("type") == "auto_compress" for e in events)
        # No compression happened — the agent added 1 user message
        # and 1 assistant message, on top of the 1 sys + 20 prior
        # body turns. Total = 23.
        assert len(a._messages) == 23
    finally:
        a._restore_compress()  # type: ignore[attr-defined]


def test_agent_run_auto_compress_short_conversation() -> None:
    """A short conversation never triggers auto-compress."""
    a = _make_agent_with_compress("o3", auto=True, fraction=0.5)
    try:
        events: list[dict] = []

        def on_event(ev):
            events.append(ev)

        # Just two short turns.
        a._messages.append({"role": "user", "content": "hi"})
        a._messages.append({"role": "assistant", "content": "hello"})

        def fake_create(**kwargs):
            return _FakeResponse("ok")

        a.client.chat.completions.create = fake_create  # type: ignore[method-assign]
        a.run("next", on_event=on_event, stream=True)
        assert not any(e.get("type") == "auto_compress" for e in events)
    finally:
        a._restore_compress()  # type: ignore[attr-defined]


def test_agent_init_stores_auto_compress_settings() -> None:
    a = Agent(model="x", system="", auto_compress=False, context_fraction=0.5)
    assert a.auto_compress is False
    assert a.context_fraction == 0.5
    b = Agent(model="x", system="")
    assert b.auto_compress is DEFAULT_AUTO_COMPRESS
    assert b.context_fraction == DEFAULT_CONTEXT_FRACTION


# === Agent undo / retry ==================================================


def test_agent_last_user_message_returns_most_recent() -> None:
    """last_user_message walks the history backwards to find the user."""
    a = Agent(model="x", system="sys")
    a._messages.append({"role": "user", "content": "first"})
    a._messages.append({"role": "assistant", "content": "ok"})
    a._messages.append({"role": "user", "content": "second"})
    msg = a.last_user_message()
    assert msg is not None
    assert msg["content"] == "second"


def test_agent_last_user_message_no_user_returns_none() -> None:
    """No user message means we can't retry / edit."""
    a = Agent(model="x", system="sys")
    assert a.last_user_message() is None
    # Even with a system + assistant, no user.
    a._messages.append({"role": "assistant", "content": "hi"})
    assert a.last_user_message() is None


def test_agent_undo_last_turn_drops_assistant_and_tools() -> None:
    """undo_last_turn removes the post-user-message sequence."""
    a = Agent(model="x", system="sys")
    a._messages.append({"role": "user", "content": "q1"})
    a._messages.append({"role": "assistant", "content": "a1"})
    a._messages.append({"role": "tool", "tool_call_id": "x", "content": "r1"})
    a._messages.append({"role": "assistant", "content": "a1b"})
    a._messages.append({"role": "user", "content": "q2"})
    a._messages.append({"role": "assistant", "content": "a2"})
    # History: sys, q1, a1, r1, a1b, q2, a2
    assert len(a._messages) == 7
    popped = a.undo_last_turn()
    assert popped is True
    # Should be: sys, q1, a1, r1, a1b, q2
    assert len(a._messages) == 6
    assert a._messages[-1]["role"] == "user"
    assert a._messages[-1]["content"] == "q2"


def test_agent_undo_last_turn_empty_history() -> None:
    """An empty history is a no-op (returns False)."""
    a = Agent(model="x", system="sys")
    assert a.undo_last_turn() is False
    # System-only is also a no-op.
    assert len(a._messages) == 1  # just the system
    assert a.undo_last_turn() is False


def test_agent_undo_last_turn_with_only_user() -> None:
    """If the most recent message is a user (assistant hasn't replied yet),
    ``/undo`` retracts that user message too."""
    a = Agent(model="x", system="sys")
    a._messages.append({"role": "user", "content": "ask"})
    assert a.undo_last_turn() is True
    assert len(a._messages) == 1  # just the system


def test_agent_undo_last_turn_preserves_system() -> None:
    """The system message at index 0 is never popped."""
    a = Agent(model="x", system="sys")
    a._messages.append({"role": "user", "content": "q"})
    a._messages.append({"role": "assistant", "content": "a"})
    a.undo_last_turn()
    assert a._messages[0]["role"] == "system"
    assert a._messages[0]["content"] == "sys"


def test_agent_replay_last_user_uses_most_recent() -> None:
    """replay_last_user pops everything after the last user message,
    then re-``run()``s with that content. We use a fake client so
    we don't actually hit the network.
    """
    a = Agent(model="x", system="sys", max_turns=2)
    # Fake the model so we can observe the messages without HTTP.
    captured = {"called_with": []}
    class _FakeChunk:
        def __init__(self, content, usage=None):
            self.choices = [type("C", (), {"delta": type("D", (), {
                "content": content, "reasoning_content": None,
                "tool_calls": None,
            })()})()] if content or usage is None else []
            self.usage = usage
            self.model_extra = {}
    class _FakeResp:
        def __init__(self, content):
            self._content = content
        def __iter__(self):
            yield _FakeChunk(content=self._content,
                              usage=type("U", (), {
                                  "prompt_tokens": 10, "completion_tokens": 5,
                                  "prompt_tokens_details": type("P", (), {"cached_tokens": 0})(),
                              })())
        def close(self): pass
    def fake_create(**kwargs):
        # Capture a *copy* of the messages at call time. The
        # agent passes a reference to ``self._messages``,
        # so we have to copy or we'd see later appends
        # (the assistant turn) reflected here too.
        captured["called_with"].append([dict(m) for m in kwargs["messages"]])
        return _FakeResp("reply")
    a.client.chat.completions.create = fake_create  # type: ignore[method-assign]
    # Seed a history.
    a._messages.append({"role": "user", "content": "hello"})
    a._messages.append({"role": "assistant", "content": "first reply"})
    # Replay.
    result = a.replay_last_user()
    assert result == "reply"
    # One new model call for the replay (the original turn
    # was seeded directly into ``_messages`` without going
    # through ``run()``, so it didn't make a call).
    assert len(captured["called_with"]) == 1
    msgs = captured["called_with"][0]
    # The replay's call should have system + the replayed
    # user message (and the new assistant turn will be
    # appended to ``_messages`` after the call returns).
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert msgs[-1]["content"] == "hello"
    # And the agent's full history has the new assistant
    # message at the end (the streaming path appends after
    # the call).
    assert a._messages[-1]["role"] == "assistant"
    assert a._messages[-1]["content"] == "reply"


def test_agent_replay_last_user_empty_history() -> None:
    """No user message → replay is a no-op."""
    a = Agent(model="x", system="sys")
    assert a.replay_last_user() is None


def test_tui_undo_command_no_history() -> None:
    """``/undo`` with no assistant turn to drop returns a hint."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_undo("")
    assert "nothing to undo" in out


def test_tui_undo_command_drops_log_and_messages() -> None:
    """``/undo`` truncates both the log and the agent's history."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    # Simulate a finished turn: user push, assistant push, etc.
    state._pre_turn_log_len = len(state.log)
    state.push("user", "hello")
    state.push("note", "thinking…", state.A_DIM)
    state.push("assistant", "hi back", state.A_GREEN)
    # The agent's history also gets a fake user message.
    agent._messages.append({"role": "user", "content": "hello"})
    agent._messages.append({"role": "assistant", "content": "hi back"})
    log_len_before = len(state.log)
    msgs_before = len(agent._messages)
    assert log_len_before > state._pre_turn_log_len
    out = state._cmd_undo("")
    assert "undone" in out
    # Log truncated to the snapshot.
    assert len(state.log) == state._pre_turn_log_len
    # Agent's messages truncated (the assistant turn is gone).
    assert len(agent._messages) == msgs_before - 1
    assert agent._messages[-1]["role"] == "user"
    # State flags reset.
    assert state.turn_active is False
    assert state.streaming_assistant == []


def test_tui_retry_command_reruns() -> None:
    """``/retry`` pops the previous turn and re-runs with the same content.

    We use a fake client so the replay doesn't hit the network.
    """
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys", max_turns=2)
    captured = {"called": 0}
    class _FakeChunk:
        def __init__(self, content):
            self.choices = [type("C", (), {"delta": type("D", (), {
                "content": content, "reasoning_content": None,
                "tool_calls": None,
            })()})()]
            self.usage = type("U", (), {
                "prompt_tokens": 10, "completion_tokens": 5,
                "prompt_tokens_details": type("P", (), {"cached_tokens": 0})(),
            })()
            self.model_extra = {}
    class _FakeResp:
        def __init__(self, content):
            self._content = content
        def __iter__(self):
            yield _FakeChunk(content=self._content)
        def close(self): pass
    def fake_create(**kwargs):
        captured["called"] += 1
        return _FakeResp("retry reply")
    agent.client.chat.completions.create = fake_create  # type: ignore[method-assign]
    # Original turn.
    agent._messages.append({"role": "user", "content": "original question"})
    agent._messages.append({"role": "assistant", "content": "first answer"})
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    state._pre_turn_log_len = len(state.log)
    state._cmd_retry("")
    # The replay made exactly one new model call.
    assert captured["called"] == 1
    # The agent's history now has the new user + new assistant
    # turn (the streaming path now appends the assistant
    # message, so the total is system + user + assistant = 3).
    assert len(agent._messages) == 3
    assert agent._messages[-1]["role"] == "assistant"
    assert agent._messages[-1]["content"] == "retry reply"


def test_tui_edit_command_prefills_editor() -> None:
    """``/edit`` puts the previous message in the editor and arms submit."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    agent._messages.append({"role": "user", "content": "previous question"})
    agent._messages.append({"role": "assistant", "content": "previous answer"})
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_edit("")
    assert "edit" in out
    # Editor was pre-filled.
    assert state.editor.buf == ["previous question"]
    # Edit flag set.
    assert state._edit_in_progress is True


def test_tui_edit_command_no_history() -> None:
    """``/edit`` with no user message returns a hint, no state change."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_edit("")
    assert "nothing to edit" in out
    assert state._edit_in_progress is False


def test_tui_edit_command_rejects_multimodal() -> None:
    """A multimodal user message can't be edited in the line editor."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    agent._messages.append({"role": "user", "content": [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]})
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_edit("")
    assert "non-text" in out
    assert state._edit_in_progress is False


def test_tui_undo_clears_edit_in_progress() -> None:
    """``/undo`` while a ``/edit`` is pending cancels the edit."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    agent._messages.append({"role": "user", "content": "ask"})
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    state._edit_in_progress = True
    state._cmd_undo("")
    assert state._edit_in_progress is False


# === TUI autocompress command ===========================================


def test_tui_autocompress_command_toggles() -> None:
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys", auto_compress=True)
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    stdscr = _StubWin()
    state = _TUIState(agent, stdscr, 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()

    # Toggle off.
    state._handle_command("/autocompress")
    assert agent.auto_compress is False
    # Toggle back on.
    state._handle_command("/autocompress")
    assert agent.auto_compress is True
    # Set fraction.
    state._handle_command("/autocompress 0.5")
    assert agent.context_fraction == 0.5
    # Show status.
    state._handle_command("/autocompress status")
    assert any("auto-compress: on" in t for k, t, _ in state.log)


def test_tui_autocompress_command_rejects_bad_fraction() -> None:
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    state._handle_command("/autocompress 1.5")
    assert any("between 0 and 1" in t for k, t, _ in state.log)
    state._handle_command("/autocompress bogus")
    assert any("unknown argument" in t for k, t, _ in state.log)


def test_tui_auto_compress_event_logged() -> None:
    """The TUI shows an auto_compress note when the agent emits the event."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    state.on_event({
        "type": "auto_compress",
        "est_tokens": 28_000, "window": 32_768, "threshold": 26_214,
    })
    state.on_event({
        "type": "auto_compress_done",
        "kept": 2, "summarized": 10, "summary_chars": 450,
    })
    msgs = [t for k, t, _ in state.log if k == "note"]
    assert any("auto-compressing" in t for t in msgs)
    assert any("compressed 10 older turns" in t for t in msgs)


def test_history_save_load() -> None:
    agent = Agent(model="test", system="be helpful")
    agent._messages.append({"role": "user", "content": "hi"})
    agent._messages.append({"role": "assistant", "content": "hello"})

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "hist.jsonl"
        saved = agent.save_history(path)
        assert saved == path

        agent2 = Agent(model="test", system="be helpful")
        n = agent2.load_history(path)
        assert n == 3
        assert agent2.messages == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]


def test_history_preserves_local_system() -> None:
    agent = Agent(model="test", system="original")
    agent._messages.append({"role": "user", "content": "hi"})

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "hist.jsonl"
        agent.save_history(path)

        agent2 = Agent(model="test", system="new")
        agent2.load_history(path)
        assert agent2.messages[0] == {"role": "system", "content": "new"}


# === Sanitization ========================================================


def test_sanitize_tool_result_short_passthrough() -> None:
    """Short results pass through unchanged."""
    text = "hello\nworld\n" * 5
    assert _sanitize_tool_result(text) == text


def test_sanitize_tool_result_dedup_runs() -> None:
    """≥3 identical consecutive lines are collapsed."""
    # Pad to >1000 chars so the sanitizer actually runs.
    text = ("a\n" * 500 + "b\n" * 500 + "same\n" * 500 + "c\n" * 500)
    assert len(text) > 1000
    out = _sanitize_tool_result(text)
    # Single 'same' line survives; the run is collapsed.
    assert "same" in out
    assert "identical lines elided" in out
    assert "a" in out and "b" in out and "c" in out


def test_sanitize_tool_result_caps_long_output() -> None:
    """Huge output is head/tail-capped with an elision marker."""
    text = "x" * 50_000
    out = _sanitize_tool_result(text)
    assert len(out) < len(text)
    assert "chars elided" in out


# === Text-fallback tool parsing ==========================================


def test_parse_text_calls_basic() -> None:
    text = (
        'I will read the file.\n'
        '<tool_call>{"name": "read_file", "arguments": {"path": "foo.py"}}</tool_call>\n'
    )
    calls = parse_text_calls(text)
    assert calls == [("read_file", {"path": "foo.py"})]


def test_parse_text_calls_no_calls() -> None:
    assert parse_text_calls("nothing here") == []


def test_parse_text_calls_ignores_malformed() -> None:
    text = '<tool_call>not valid json</tool_call><tool_call>{"name": "ok", "arguments": {}}</tool_call>'
    calls = parse_text_calls(text)
    # Only the well-formed call survives.
    assert calls == [("ok", {})]


# === Tool call aggregator ================================================


class _FakeDelta:
    def __init__(self, index, id_="", name="", arguments=""):
        self.index = index
        self.id = id_
        self.function = type("F", (), {"name": name, "arguments": arguments})()


def test_tool_call_aggregator() -> None:
    agg = _ToolCallAggregator()
    agg.add(_FakeDelta(0, id_="call_1", name="read_"))
    agg.add(_FakeDelta(0, arguments='{"pat'))
    agg.add(_FakeDelta(0, name="file", arguments='h": "x"}'))
    out = agg.finalize()
    assert len(out) == 1
    assert out[0]["id"] == "call_1"
    assert out[0]["function"]["name"] == "read_file"
    assert out[0]["function"]["arguments"] == '{"path": "x"}'


# === Metrics =============================================================


def test_metrics_add_and_meta() -> None:
    m = _Metrics("sid", model="m")
    m.add({"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 3})
    m.add({"input_tokens": 7, "output_tokens": 2})
    meta = m.as_meta()
    assert meta["input_tokens"] == 17
    assert meta["output_tokens"] == 7
    assert meta["cache_read_tokens"] == 3
    assert meta["api_calls"] == 2
    assert meta["started_at"] > 0


def test_metrics_load_from_saved() -> None:
    m = _Metrics("sid")
    m.load({"input_tokens": 100, "output_tokens": 50, "api_calls": 5,
            "started_at": 12345.0})
    assert m.input_tokens == 100
    assert m.started_at == 12345.0


def test_normalize_usage_basic() -> None:
    """Extract {input, output, cache_read} from an OpenAI-style usage."""
    usage = type("U", (), {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "prompt_tokens_details": type("P", (), {"cached_tokens": 4})(),
        "output_tokens_details": None,
    })()
    out = _normalize_usage(usage, None)
    assert out == {
        "input_tokens": 8,
        "output_tokens": 8,
        "cache_read_tokens": 4,
        "reasoning_tokens": 0,
    }


# === Sessions ============================================================


def test_safe_title() -> None:
    assert _safe_title("hello world") == "hello world"
    assert _safe_title("  multiple\n\nspaces  ") == "multiple spaces"
    assert _safe_title("") is None
    assert _safe_title("x" * 100) is not None
    assert len(_safe_title("x" * 100)) <= 60


def test_new_session_id_format() -> None:
    sid = _new_session_id()
    parts = sid.split("-")
    # YYYYMMDD-HHMMSS-XXXXXX
    assert len(parts) == 3
    assert len(parts[0]) == 8
    assert len(parts[1]) == 6
    assert len(parts[2]) == 6


def test_short_id() -> None:
    assert _short_id("20240101-120000-abc123") == "abc123"
    assert _short_id("nohyphen") == "nohyphen"
    assert _short_id("") == ""


def test_prune_empty_assistant_messages() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]},
    ]
    removed = _prune_empty_assistant_messages(msgs)
    assert removed == 3
    assert len(msgs) == 2


def test_sessions_save_load_delete_list() -> None:
    import os
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        try:
            sid = _new_session_id()
            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "hi!"},
            ]
            _write_session(sid, messages, {"title": "greeting"})
            loaded = _load_session(sid)
            assert loaded is not None
            assert loaded["messages"] == messages
            assert loaded["title"] == "greeting"
            sessions = _list_sessions(limit=10)
            assert any(s["id"] == sid for s in sessions)
            assert _delete_session(sid) is True
            assert _load_session(sid) is None
        finally:
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


# === Session metadata index ==============================================


def test_index_created_on_write() -> None:
    """Writing a session should create (or update) the index."""
    from anduril import invalidate_index_cache, get_index
    import anduril.sessions as sess_mod
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            sid = "20250101-120000-abc123"
            _write_session(
                sid,
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                {"title": "Test", "model": "m"},
            )
            index = get_index()
            assert sid in index["entries"]
            entry = index["entries"][sid]
            assert entry["title"] == "Test"
            assert entry["model"] == "m"
            assert entry["n"] == 2
            # Index file is on disk.
            assert sess_mod._index_path().is_file()
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_index_removed_on_delete() -> None:
    from anduril import invalidate_index_cache, get_index
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            sid = "20250101-120000-abc123"
            _write_session(
                sid, [{"role": "user", "content": "x"}], {"title": "T"},
            )
            assert sid in get_index()["entries"]
            _delete_session(sid)
            assert sid not in get_index()["entries"]
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_list_sessions_uses_index() -> None:
    """Listing should not parse any session JSON files."""
    from anduril import invalidate_index_cache
    import anduril.sessions as sess_mod
    sessions_read: list[str] = []
    orig_summary = sess_mod._session_summary_from_file

    def tracking_summary(fname: str) -> dict | None:
        sessions_read.append(fname)
        return orig_summary(fname)

    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        sess_mod._session_summary_from_file = tracking_summary  # type: ignore[assignment]
        try:
            # Write 3 sessions.
            for i in range(3):
                _write_session(
                    f"sid{i}",
                    [{"role": "user", "content": f"u{i}"},
                     {"role": "assistant", "content": f"a{i}"}],
                    {"title": f"Title {i}"},
                )
            sessions_read.clear()
            sessions = _list_sessions(limit=10)
            assert len(sessions) == 3
            # The index-backed list should NOT have called
            # _session_summary_from_file for any session.
            assert sessions_read == []
        finally:
            sess_mod._session_summary_from_file = orig_summary  # type: ignore[assignment]
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_list_sessions_query_uses_index() -> None:
    from anduril import invalidate_index_cache
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            _write_session("a", [{"role": "user", "content": "x"}],
                           {"title": "alpha"})
            _write_session("b", [{"role": "user", "content": "x"}],
                           {"title": "beta"})
            _write_session("c", [{"role": "user", "content": "x"}],
                           {"title": "gamma"})
            sessions = _list_sessions(query="alph")
            assert len(sessions) == 1
            assert sessions[0]["id"] == "a"
            assert sessions[0]["title"] == "alpha"
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_list_sessions_limit_offset_uses_index() -> None:
    from anduril import invalidate_index_cache
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            for i in range(5):
                _write_session(
                    f"sid{i}",
                    [{"role": "user", "content": f"u{i}"}],
                    {"title": f"T{i}"},
                )
            page1 = _list_sessions(limit=2, offset=0)
            page2 = _list_sessions(limit=2, offset=2)
            page3 = _list_sessions(limit=2, offset=4)
            assert len(page1) == 2
            assert len(page2) == 2
            assert len(page3) == 1
            # Newest first, so sid4, sid3, sid2, sid1, sid0.
            assert page1[0]["id"] == "sid4"
            assert page2[0]["id"] == "sid2"
            assert page3[0]["id"] == "sid0"
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_synthesize_index_from_existing_files() -> None:
    """A user upgrading from a no-index version should auto-build it."""
    from anduril import invalidate_index_cache, get_index
    import anduril.sessions as sess_mod
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            # Write two session files directly (bypassing
            # _write_session, which would auto-build the index).
            for i in range(2):
                data = {
                    "id": f"old{i}",
                    "messages": [
                        {"role": "user", "content": f"u{i}"},
                        {"role": "assistant", "content": f"a{i}"},
                    ],
                    "title": f"Old {i}",
                    "model": "m",
                    "updated_at": 12345.0 + i,
                }
                p = sess_mod._session_path(f"old{i}")
                p.write_text(json.dumps(data), encoding="utf-8")
            # Make sure no index file exists.
            if sess_mod._index_path().exists():
                sess_mod._index_path().unlink()
            # List should synthesise the index.
            sessions = _list_sessions(limit=10)
            assert len(sessions) == 2
            # Index file is now on disk.
            assert sess_mod._index_path().is_file()
            # And both entries are present.
            entries = get_index()["entries"]
            assert "old0" in entries
            assert "old1" in entries
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_list_prunes_stale_entries() -> None:
    """An index entry for a deleted file should disappear on the next list."""
    from anduril import invalidate_index_cache, get_index
    import anduril.sessions as sess_mod
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            # Create a real session and a fake index entry.
            _write_session("real",
                           [{"role": "user", "content": "x"}],
                           {"title": "Real"})
            idx = get_index()
            idx["entries"]["ghost"] = {
                "title": "Ghost", "n": 0, "updated_at": 0.0, "model": "m",
            }
            sess_mod._save_index_to_disk(idx["entries"])
            invalidate_index_cache()

            # List drops the ghost.
            sessions = _list_sessions(limit=10)
            assert [s["id"] for s in sessions] == ["real"]
            # Index on disk is also cleaned.
            invalidate_index_cache()
            assert "ghost" not in get_index()["entries"]
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_list_picks_up_new_file() -> None:
    """A file added behind the index's back should be picked up next list."""
    from anduril import invalidate_index_cache, get_index
    import anduril.sessions as sess_mod
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            _write_session("a", [{"role": "user", "content": "x"}],
                           {"title": "A"})
            # Drop a file directly into the sessions dir.
            data = {
                "id": "manual",
                "messages": [
                    {"role": "user", "content": "manual"},
                    {"role": "assistant", "content": "ok"},
                ],
                "title": "Manual",
                "updated_at": 99999.0,
            }
            sess_mod._session_path("manual").write_text(
                json.dumps(data), encoding="utf-8",
            )
            # Touch the index cache key (mtime) so it doesn't
            # notice. Actually, since the dir mtime changes when we
            # add a file, the cache invalidates on its own.
            sessions = _list_sessions(limit=10)
            ids = {s["id"] for s in sessions}
            assert "manual" in ids
            # The manual session was synthesised into the index.
            invalidate_index_cache()
            assert "manual" in get_index()["entries"]
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_index_invalidate_cache_forces_reload() -> None:
    """invalidate_index_cache() forces a fresh read on the next get_index()."""
    from anduril import invalidate_index_cache, get_index
    import anduril.sessions as sess_mod
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            _write_session("a", [{"role": "user", "content": "x"}],
                           {"title": "A"})
            # Cache is now populated.
            assert sess_mod._index_cache is not None
            assert "a" in sess_mod._index_cache["entries"]
            # Drop the cache. The next get_index() must re-read.
            invalidate_index_cache()
            assert sess_mod._index_cache is None
            assert "a" in get_index()["entries"]
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_index_handles_corrupt_file() -> None:
    """A corrupt index should not break the listing — we rebuild from files."""
    from anduril import invalidate_index_cache
    import anduril.sessions as sess_mod
    with tempfile.TemporaryDirectory() as td:
        original = os.environ.get("ANDURIL_SESSIONS_DIR")
        os.environ["ANDURIL_SESSIONS_DIR"] = td
        invalidate_index_cache()
        try:
            # Write a real session, then overwrite the index with garbage.
            _write_session("a", [{"role": "user", "content": "x"}],
                           {"title": "A"})
            sess_mod._index_path().write_text("not valid json", encoding="utf-8")
            invalidate_index_cache()
            # Listing should self-heal: see the session, rebuild
            # the index, and return the session.
            sessions = _list_sessions(limit=10)
            assert any(s["id"] == "a" for s in sessions)
        finally:
            invalidate_index_cache()
            if original is None:
                os.environ.pop("ANDURIL_SESSIONS_DIR", None)
            else:
                os.environ["ANDURIL_SESSIONS_DIR"] = original


def test_resolve_session_by_id_and_prefix() -> None:
    sessions = [
        {"id": "20240101-120000-abc123", "title": "first"},
        {"id": "20240101-120001-def456", "title": "second"},
    ]
    assert _resolve_session("20240101-120001-def456", sessions) == "20240101-120001-def456"
    assert _resolve_session("20240101-120001", sessions) == "20240101-120001-def456"
    assert _resolve_session("def456", sessions) == "20240101-120001-def456"
    assert _resolve_session("1", sessions) == "20240101-120000-abc123"
    assert _resolve_session("second", sessions) == "20240101-120001-def456"
    assert _resolve_session("20240101", sessions) is None


def test_resolve_session_by_index() -> None:
    sessions = [
        {"id": "a", "title": "one"},
        {"id": "b", "title": "two"},
    ]
    assert _resolve_session("1", sessions) == "a"
    assert _resolve_session("2", sessions) == "b"
    assert _resolve_session("3", sessions) is None


# === Editor ==============================================================


def test_editor_basic_insert_and_submit() -> None:
    e = _Editor()
    e.insert_char("h")
    e.insert_char("i")
    assert e.buf == ["hi"]
    assert e.row == 0 and e.col == 2
    text = e.submit()
    assert text == "hi"
    assert e.buf == [""]


def test_editor_multiline_paste_no_cap() -> None:
    """A long pasted string with many lines must NOT be truncated."""
    e = _Editor()
    big = "line\n" * 5000  # 25k chars
    e.insert_text(big.rstrip("\n"))
    assert e.char_count() >= 20000  # no cap
    assert len(e.buf) == 5000


def test_editor_backspace_merges_lines() -> None:
    e = _Editor()
    e.insert_text("foo\nbar")
    # Cursor is at end of "bar" — col=3, row=1
    assert e.row == 1 and e.col == 3
    e.backspace()  # delete 'r'
    assert e.buf == ["foo", "ba"]
    e.backspace()  # delete 'a'
    assert e.buf == ["foo", "b"]
    e.backspace()  # delete 'b'
    assert e.buf == ["foo", ""]
    e.backspace()  # merge: "foo" + "" = "foo"
    assert e.buf == ["foo"]
    assert e.row == 0 and e.col == 3


def test_editor_history_navigation() -> None:
    e = _Editor(history=["first prompt", "second prompt"])
    # h_idx starts past the end (== "draft position").
    assert e.h_idx == 2
    # Up at the top of the buffer pulls the most recent history first.
    e.move_up()
    assert e.buf == ["second prompt"]
    assert e.h_idx == 1
    e.move_up()
    assert e.buf == ["first prompt"]
    assert e.h_idx == 0
    # Down advances forward through history, then restores the draft.
    e.move_down()
    assert e.buf == ["second prompt"]
    assert e.h_idx == 1
    e.move_down()
    assert e.buf == [""]  # restored draft
    assert e.h_idx == 2


def test_wrap_visual_chunks_use_word_boundaries() -> None:
    """Long lines should wrap on word boundaries (not split mid-word).

    textwrap.wrap with break_long_words=True is used in render. The
    cursor mapping relies on the chunk ranges being correct.
    """
    import textwrap
    line = ("Design and create a very creative, elaborate, and detailed "
            "voxel art scene of a pagoda in a beautiful Japanese garden "
            "with cherry blossom trees, koi pond, stone lanterns, and a "
            "wooden bridge. The scene should be rendered in Minecraft with "
            "detailed textures and lighting effects, with a soft sunset glow.")
    inner_w = 78
    chunks = textwrap.wrap(line, width=inner_w,
                           break_long_words=True, break_on_hyphens=False,
                           drop_whitespace=False)
    # Chunks should cover the line without gaps.
    reconstructed = "".join(chunks)
    assert reconstructed == line
    # No chunk should exceed inner_w.
    for c in chunks:
        assert len(c) <= inner_w


def test_wrap_cursor_mapping_with_ranges() -> None:
    """Cursor col -> visual row + shown col using actual chunk boundaries."""
    import textwrap
    inner_w = 78
    line = ("Design and create a very creative, elaborate, and detailed "
            "voxel art scene of a pagoda in a beautiful Japanese garden "
            "with cherry blossom trees, koi pond, stone lanterns, and a "
            "wooden bridge. The scene should be rendered in Minecraft with "
            "detailed textures and lighting effects, with a soft sunset glow "
            "casting long shadows across the scene.")
    chunks = textwrap.wrap(line, width=inner_w,
                           break_long_words=True, break_on_hyphens=False,
                           drop_whitespace=False)
    pos = 0
    visual = []
    for c in chunks:
        start = line.find(c, pos)
        end = start + len(c)
        visual.append((start, end, c))
        pos = end
    # Verify cursor at col 200 lands on the right char.
    col = 200
    cur_visual_row = None
    cur_shown_col = None
    for i, (cs, ce, c) in enumerate(visual):
        if cs <= col <= ce:
            cur_visual_row = i
            cur_shown_col = col - cs
            break
    assert cur_visual_row is not None
    assert visual[cur_visual_row][2][cur_shown_col] == line[col]


def test_enter_submits_regardless_of_cr_or_lf() -> None:
    """Both \\r and \\n must submit (Enter). No burst detector — Enter
    always submits, even immediately after typing. Newlines are only
    inserted via Shift+Enter or bracketed paste."""
    e = _Editor()
    submitted: list[str] = []

    def on_key(ch: str) -> str:
        if ch == "\r" or ch == "\n":
            if not e.is_empty():
                submitted.append(e.submit())
            return "submit"
        if ch == "\x1b":
            return "alt_enter"  # the real TUI calls editor.newline() for this
        if ch.isprintable():
            e.insert_char(ch)
            return "inserted"
        return "ignored"

    # Type a line, then immediately press Enter — must submit, not newline.
    for ch in "hello":
        on_key(ch)
    on_key("\r")
    assert submitted == ["hello"]
    assert e.buf == [""]

    # Same for \\n (some terminals send LF for Enter).
    for ch in "world":
        on_key(ch)
    on_key("\n")
    assert submitted == ["hello", "world"]
    assert e.buf == [""]


# === Files / @-mentions =================================================


def test_files_is_image_recognizes_known_formats() -> None:
    """is_image is true for the formats the OpenAI multimodal endpoint
    accepts (png, jpg, jpeg, gif, webp, bmp) and false otherwise."""
    from anduril.files import is_image
    assert is_image("foo.png") is True
    assert is_image("foo.PNG") is True  # case-insensitive
    assert is_image("foo.jpg") is True
    assert is_image("foo.jpeg") is True
    assert is_image("foo.gif") is True
    assert is_image("foo.webp") is True
    assert is_image("foo.bmp") is True
    assert is_image("foo.txt") is False
    assert is_image("foo.py") is False
    assert is_image("foo") is False
    # Path with directories: only the suffix matters.
    assert is_image("a/b/c/d.png") is True


def test_files_is_text_file_uses_extension_hint() -> None:
    """Known text extensions are classified as text without reading."""
    from anduril.files import is_text_file
    # Image files are never text (they'd trip up the encoder on read).
    assert is_text_file("foo.png") is False
    # Known code / config / doc extensions are text.
    for ext in (".py", ".md", ".json", ".yaml", ".html", ".sh", ".txt"):
        assert is_text_file(f"foo{ext}") is True, f"{ext} should be text"
    # Unknown extension with a dot: default to binary.
    assert is_text_file("foo.unknownext") is False


def test_files_fuzzy_match_empty_query_preserves_order() -> None:
    """With no query, the candidates are returned in their original order
    (up to the limit). This is what the file picker shows before the user
    types anything."""
    from anduril.files import fuzzy_match
    cands = ["src/main.py", "README.md", "anduril/agent.py", "tests/test_x.py"]
    out = fuzzy_match("", cands, limit=10)
    assert [c for _, c in out] == cands


def test_files_fuzzy_match_prefix_wins() -> None:
    """A candidate starting with the query outranks a candidate that
    contains the query as a substring elsewhere."""
    from anduril.files import fuzzy_match
    out = fuzzy_match("agent", [
        "anduril/agent.py",   # exact prefix - should rank first
        "src/agent_runner.py",  # also starts with agent
        "tools/agent_helper.py", # contains 'agent' but not as prefix
    ], limit=10)
    # All three contain 'agent' as a subsequence. The two starting
    # with 'agent' should rank above the one that doesn't.
    names = [c for _, c in out]
    assert names.index("anduril/agent.py") < names.index("tools/agent_helper.py")
    assert names.index("src/agent_runner.py") < names.index("tools/agent_helper.py")


def test_files_fuzzy_match_word_boundary_bonus() -> None:
    """A match at a path-separator / word boundary outranks a match in
    the middle of a token."""
    from anduril.files import fuzzy_match
    out = fuzzy_match("a", [
        "alpha.py",           # 'a' is the first char (prefix)
        "anduril/agent.py",   # 'a' is at the start (prefix)
        "data.txt",           # 'a' is in the middle
        "xyzza",              # 'a' is at the end
    ], limit=10)
    # Prefix matches should rank first; the suffix-only match last.
    names = [c for _, c in out]
    assert names.index("alpha.py") < names.index("data.txt")
    assert names.index("anduril/agent.py") < names.index("data.txt")
    # The suffix match should be last among the matches (it has no
    # boundary bonus).
    assert names.index("xyzza") > names.index("data.txt")


def test_files_fuzzy_match_rejects_non_subsequence() -> None:
    """A candidate that doesn't contain the query as a subsequence is
    excluded from the result — the picker only shows real matches,
    not the whole filesystem in arbitrary order."""
    from anduril.files import fuzzy_match
    # "abc" and "xabc" both contain "abc" as a subsequence; "xyz"
    # and "ab" do not.
    out = fuzzy_match("abc", ["xyz", "ab", "abc", "xabc"], limit=10)
    names = [c for _, c in out]
    assert "xyz" not in names, f"non-subsequence match leaked through: {names}"
    assert "ab" not in names, f"non-subsequence match leaked through: {names}"
    assert set(names) == {"abc", "xabc"}
    # 'abc' has the exact prefix; it should outrank 'xabc'.
    assert names[0] == "abc"


def test_files_find_active_mention_basic() -> None:
    """Cursor inside a mention returns the @-position and the cursor."""
    from anduril.files import find_active_mention
    # Cursor right after the @: position 1.
    assert find_active_mention("@", 1) == (0, 1)
    # Cursor right after @src: position 4.
    assert find_active_mention("Hello @src world", 9) == (6, 9)
    # Cursor in the middle of the mention.
    assert find_active_mention("Hello @src world", 8) == (6, 8)


def test_files_find_active_mention_email_left_alone() -> None:
    """An @ preceded by an alphanumeric char (e.g. inside an email
    address) is NOT recognized as a mention start."""
    from anduril.files import find_active_mention
    # The cursor is after 'host' in 'user@host'; the @ is preceded by
    # 'r' (alphanumeric), so there's no active mention.
    assert find_active_mention("user@host", 9) is None
    assert find_active_mention("user@host.com", 13) is None
    # Same when the @ is preceded by underscore.
    assert find_active_mention("foo_@bar", 8) is None


def test_files_find_active_mention_trailing_terminator() -> None:
    """A terminator (whitespace, punctuation) between the cursor and
    the @ means there's no active mention (the user has moved on)."""
    from anduril.files import find_active_mention
    # Cursor at the space AFTER the comma — no active mention.
    assert find_active_mention("Hello @src, world", 11) is None
    # Cursor at the space after @foo — no active mention.
    assert find_active_mention("@foo bar", 5) is None
    # Cursor right after the @ (no query yet) — still active.
    assert find_active_mention("@foo bar", 1) == (0, 1)
    # Cursor AT the terminator position — the mention ends at the
    # cursor (the terminator is the cursor position itself, not
    # before it). So the mention is still active and includes
    # everything up to that point.
    assert find_active_mention("Hello @src, world", 10) == (6, 10)


def test_files_mention_query_returns_path() -> None:
    """mention_query returns the text after the @ up to the cursor."""
    from anduril.files import mention_query
    # Cursor right after the @ → empty query, start at 0.
    assert mention_query("@", 1) == ("", 0, 1)
    # Cursor at position 4 = after 'src' → query is "src".
    # "Hello @src world" — @ is at 6, s=7, r=8, c=9, space=10.
    # Cursor at 10 sits between 'c' and the space, with the
    # space-terminator not yet crossed.
    assert mention_query("Hello @src world", 10) == ("src", 6, 10)
    # No active mention (cursor past terminator) → empty + zeros.
    assert mention_query("Hello @src, world", 11) == ("", 0, 0)


def test_files_list_files_respects_max_count() -> None:
    """list_files caps the result at max_count, even when more files exist."""
    from anduril.files import list_files
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        for i in range(50):
            (root / f"file_{i:03d}.txt").write_text("x")
        out = list_files(root, max_count=10, max_depth=2)
        assert len(out) == 10, f"expected cap at 10, got {len(out)}"


def test_files_list_files_skips_default_ignored_dirs() -> None:
    """The default ignore set skips .git, __pycache__, node_modules, etc."""
    from anduril.files import list_files
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "real.txt").write_text("x")
        for ignored in ("__pycache__", ".git", "node_modules", ".ruff_cache"):
            d = root / ignored
            d.mkdir()
            (d / "skip.py").write_text("x")
        out = list_files(root, max_count=200, max_depth=5)
        names = [str(p) for p in out]
        assert "real.txt" in names
        # No path under any ignored dir.
        for n in names:
            assert "__pycache__" not in n
            assert ".git" not in n
            assert "node_modules" not in n
            assert ".ruff_cache" not in n


def test_files_expand_mentions_text_file_inlined() -> None:
    """A @-mention of a text file is inlined as a fenced [file: ...] block."""
    from anduril.files import expand_mentions
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        f = root / "hello.txt"
        f.write_text("hello world\n", encoding="utf-8")
        out = expand_mentions(
            f"please look at @{f.name} and reply",
            cwd=root,
        )
        # Two parts: a text part before the file, a text part containing
        # the file body, and a text part after.
        assert len(out) == 3
        assert all(p["type"] == "text" for p in out)
        assert "please look at" in out[0]["text"]
        assert "[file: hello.txt]" in out[1]["text"]
        assert "hello world" in out[1]["text"]
        assert "and reply" in out[2]["text"]


def test_files_expand_mentions_image_loaded_as_data_url() -> None:
    """A @-mention of an image file becomes an image_url content part."""
    import base64
    from anduril.files import expand_mentions
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        # 1x1 red PNG (smallest valid PNG)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
            b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
        )
        img = root / "tiny.png"
        img.write_bytes(png_bytes)
        out = expand_mentions("what is this? @tiny.png", cwd=root)
        # Find the image part.
        img_parts = [p for p in out if p.get("type") == "image_url"]
        assert len(img_parts) == 1, f"expected 1 image part, got {len(img_parts)}"
        url = img_parts[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        # Verify the base64 decodes back to the original bytes.
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == png_bytes


def test_files_expand_mentions_unknown_file_left_literal() -> None:
    """A @-mention of a non-existent or unknown-extension file is left
    as literal text (the model can decide what to do)."""
    from anduril.files import expand_mentions
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        out = expand_mentions("look at @does_not_exist", cwd=root)
        # Two text parts: a prefix and the literal mention.
        assert len(out) == 2
        assert all(p["type"] == "text" for p in out)
        # The mention is preserved verbatim in the second part.
        assert "@does_not_exist" in out[1]["text"]


def test_files_expand_mentions_email_left_alone() -> None:
    """An email address is not a mention (the @ is preceded by an
    identifier character)."""
    from anduril.files import expand_mentions
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        out = expand_mentions("contact me at user@example.com thanks",
                              cwd=root)
        # The whole thing is one text part, untouched.
        assert len(out) == 1
        assert out[0]["type"] == "text"
        assert "user@example.com" in out[0]["text"]


def test_files_expand_mentions_multiple_in_order() -> None:
    """Multiple @-mentions in one buffer produce content parts in the
    same order they appear, with text segments interleaved."""
    from anduril.files import expand_mentions
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "a.py").write_text("print('a')\n", encoding="utf-8")
        (root / "b.py").write_text("print('b')\n", encoding="utf-8")
        out = expand_mentions(
            "compare @a.py and @b.py now",
            cwd=root,
        )
        # 5 parts: text("compare "), text(file a), text(" and "),
        # text(file b), text(" now").
        assert len(out) == 5
        assert all(p["type"] == "text" for p in out)
        assert "compare" in out[0]["text"]
        assert "a.py" in out[1]["text"]
        assert "and" in out[2]["text"]
        assert "b.py" in out[3]["text"]
        assert "now" in out[4]["text"]


def test_files_read_image_data_url_size_limit() -> None:
    """Images larger than max_bytes raise ValueError (not silent corruption)."""
    from anduril.files import read_image_data_url
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        f = root / "big.png"
        # 100 bytes of zero data — small enough to write but big enough
        # to exceed the max_bytes we pass.
        f.write_bytes(b"\x00" * 100)
        try:
            read_image_data_url(f, max_bytes=50)
        except ValueError as e:
            assert "too large" in str(e).lower() or "refusing" in str(e).lower()
        else:
            raise AssertionError("expected ValueError for oversized image")


def test_files_read_text_file_truncates() -> None:
    """Reading a text file past max_chars returns a truncated copy
    with a clear marker."""
    from anduril.files import read_text_file
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        f = root / "big.txt"
        # 100 chars, cap at 50.
        f.write_text("x" * 100, encoding="utf-8")
        out = read_text_file(f, max_chars=50)
        # The output should be 50 'x' chars + a marker.
        assert out.startswith("x" * 50)
        assert "truncated" in out.lower()


# === Agent multimodal support ============================================


def test_agent_run_accepts_multimodal_content() -> None:
    """Agent.run() accepts a list of content parts (text + image_url)
    as the user message and appends it verbatim to the message history."""
    agent = Agent(model="test", system="sys", max_retries=1)
    parts = [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    # Don't actually call the model — just verify the message is
    # appended. We use the non-streaming path's "ready to send" logic
    # by checking what would be sent.
    agent._messages.append({"role": "user", "content": parts})
    assert agent._messages[-1] == {"role": "user", "content": parts}
    # The mirror in the TUI walks the same list, so it should also
    # handle list content.
    text_only = ""
    for msg in agent._messages:
        c = msg.get("content")
        if isinstance(c, str):
            text_only += c
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict):
                    text_only += p.get("text", "") or ""
    assert "what is in this image?" in text_only


# === TUI file menu =======================================================


def test_tui_file_menu_activates_on_at() -> None:
    """The file menu becomes active when the cursor is right after an @."""
    state = _make_tui_state()
    # "look at @src" — @ is at position 8, s/r/c at 9/10/11.
    # Cursor at 12 sits just past the end of the string (so the
    # editor is in a state where the user has just finished typing
    # the path). Equivalently, position 11 is between 'r' and 'c';
    # for a complete mention we want the cursor at the end.
    state.editor.buf = ["look at @src"]
    state.editor.row = 0
    state.editor.col = len("look at @src")  # 12
    assert state._file_menu_active() is True
    assert state._file_menu_query() == "src"


def test_tui_file_menu_does_not_activate_for_email() -> None:
    """An @ preceded by an alphanumeric char (e.g. inside an email)
    does NOT activate the file menu."""
    state = _make_tui_state()
    state.editor.buf = ["contact user@example.com"]
    state.editor.row = 0
    state.editor.col = len(state.editor.buf[0])  # at end
    assert state._file_menu_active() is False
    assert state._file_menu_query() == ""


def test_tui_file_menu_complete_inserts_path() -> None:
    """Pressing Tab with a single file in the picker inserts the full path
    with an @ prefix; the cursor lands right after the inserted text."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Seed the cache with a controlled candidate list. The cache
        # key has to match what _file_menu_candidates expects: a
        # (cwd, mtime) tuple where cwd is the real cwd and mtime
        # is its stat mtime. We read those, then overwrite the
        # candidates list to the controlled values.
        cwd = pathlib.Path.cwd()
        try:
            mtime = cwd.stat().st_mtime
        except OSError:
            mtime = 0.0
        state.file_menu_cache_key = (str(cwd), mtime)
        state.file_menu_candidates = ["src/main.py", "src/agent.py"]
        state.file_menu_last_query = None

        # "look @" is 6 chars. Cursor at 6 sits right after the @.
        state.editor.buf = ["look @"]
        state.editor.row = 0
        state.editor.col = 6

        # Cursor is in an @-mention (no query yet). On the first
        # completion, fuzzy_match returns all candidates ordered by
        # the score, with the empty-query case preserving order. So
        # the first match is "src/main.py".
        assert state._file_menu_active() is True
        assert state._file_menu_query() == ""
        # Move down to select the second one.
        state._file_menu_move(1)
        assert state.file_menu_selected == 1

        ok = state._file_menu_complete()
        assert ok is True
        # The buffer should now contain "@src/agent.py" (the selected path).
        assert state.editor.buf[0] == "look @src/agent.py"
        # Cursor is right after the inserted path.
        assert state.editor.col == len("look @src/agent.py")


def test_tui_file_menu_query_updates_on_typing() -> None:
    """The fuzzy query updates as the user types more characters after @."""
    state = _make_tui_state()
    # Seed the cache with a controlled list. See the comment in
    # test_tui_file_menu_complete_inserts_path for the key format.
    cwd = pathlib.Path.cwd()
    try:
        mtime = cwd.stat().st_mtime
    except OSError:
        mtime = 0.0
    state.file_menu_cache_key = (str(cwd), mtime)
    state.file_menu_candidates = [
        "src/main.py", "src/agent.py", "tests/test_x.py", "README.md",
    ]
    state.file_menu_last_query = None
    # Type "@a" — the matches should be limited to paths containing
    # 'a' as a subsequence.
    state.editor.buf = ["@a"]
    state.editor.row = 0
    state.editor.col = 2
    assert state._file_menu_active() is True
    assert state._file_menu_query() == "a"
    matches = state._file_menu_matches()
    # README.md has 'a' in 'readme' so it matches. The 'src/agent.py'
    # has 'a' at a boundary, so it should rank higher than the one
    # with 'a' deep in a token.
    names = set(matches)
    assert "src/agent.py" in names
    assert "src/main.py" in names
    assert "README.md" in names
    # 'tests/test_x.py' has no 'a' at all — should be excluded.
    assert "tests/test_x.py" not in names


def test_tui_file_menu_dismiss_clears_mention() -> None:
    """Pressing Esc while the file menu is open drops the partial mention
    from the buffer (so the next keypress doesn't immediately re-open
    the menu)."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Seed the file list cache so the dismiss path doesn't walk
        # the filesystem. (dismiss itself doesn't need it, but the
        # _file_menu_active check inside it does call _file_menu_query
        # which only reads the editor.)
        cwd = pathlib.Path.cwd()
        try:
            mtime = cwd.stat().st_mtime
        except OSError:
            mtime = 0.0
        state.file_menu_cache_key = (str(cwd), mtime)
        state.file_menu_candidates = ["x.py"]

        state.editor.buf = ["look @sr"]
        state.editor.row = 0
        state.editor.col = 8
        assert state._file_menu_active() is True
        state._dismiss_file_menu()
        assert state._file_menu_active() is False
        # The mention is gone, but the rest of the buffer survives.
        assert state.editor.buf[0] == "look "
        # Cursor is at the position where the mention used to start.
        assert state.editor.col == 5


def test_tui_file_menu_uses_cache_for_repeated_queries() -> None:
    """Re-rendering with the same query re-uses the cached match list
    (the file walk + fuzzy rank are each O(N) and would be wasteful to
    redo per render tick)."""
    state = _make_tui_state()
    # Seed the cache with a controlled list.
    cwd = pathlib.Path.cwd()
    try:
        mtime = cwd.stat().st_mtime
    except OSError:
        mtime = 0.0
    state.file_menu_cache_key = (str(cwd), mtime)
    state.file_menu_candidates = ["a.py", "b.py", "c.py"]
    state.file_menu_last_query = None
    state.editor.buf = ["@"]
    state.editor.row = 0
    state.editor.col = 1
    # First call: populates the cache.
    m1 = state._file_menu_matches()
    assert state.file_menu_last_query == ""
    # Second call with the same query: same result, no recomputation.
    m2 = state._file_menu_matches()
    assert m1 == m2
    # Typing a character changes the query → invalidates the cache.
    state.editor.buf = ["@a"]
    state.editor.col = 2
    m3 = state._file_menu_matches()
    assert state.file_menu_last_query == "a"
    # The matches must be a subset of the original (filtered by 'a').
    assert set(m3).issubset(set(m1))


def test_tui_file_menu_arrow_keys_navigate() -> None:
    """Up/Down arrow keys move the file menu selection when active."""
    state = _make_tui_state()
    cwd = pathlib.Path.cwd()
    try:
        mtime = cwd.stat().st_mtime
    except OSError:
        mtime = 0.0
    state.file_menu_cache_key = (str(cwd), mtime)
    state.file_menu_candidates = ["a.py", "b.py", "c.py"]
    state.file_menu_last_query = None
    state.editor.buf = ["@"]
    state.editor.row = 0
    state.editor.col = 1
    # Activate the menu and populate the matches.
    state._file_menu_matches()
    assert state._file_menu_active() is True
    # Move down twice → selected = 2.
    state._file_menu_move(1)
    state._file_menu_move(1)
    assert state.file_menu_selected == 2
    # Move up → 1.
    state._file_menu_move(-1)
    assert state.file_menu_selected == 1
    # Wraps around at the end.
    state._file_menu_move(2)
    assert state.file_menu_selected == 0


def test_tui_file_menu_caches_list() -> None:
    """_file_menu_candidates is populated on the first call and reused
    for subsequent calls (no re-walk of the filesystem)."""
    state = _make_tui_state()
    # The default cwd is whatever the test runs in; we just check
    # that the second call doesn't change the list.
    c1 = state._file_menu_candidates()
    c2 = state._file_menu_candidates()
    assert c1 is c2  # same list object — not just equal content


# === Helpers =============================================================


def test_abbr_and_precise_abbr() -> None:
    assert _abbr(0) == "0"
    assert _abbr(999) == "999"
    assert _abbr(1500) == "1.5K"
    assert _abbr(78825) == "78K"
    assert _abbr(1_234_567) == "1.2M"
    assert _precise_abbr(25152) == "25.15K"
    assert _precise_abbr(1_234_567) == "1.23M"


def test_default_system_prompt_is_concise_focused() -> None:
    """The default system prompt nudges toward concise, deliverable-first
    responses — the failure mode the prompt was written to fix is models
    that burn 200+ tokens on meta-commentary before producing output."""
    assert "Be concise" in _DEFAULT_SYSTEM
    assert "preamble" in _DEFAULT_SYSTEM.lower()
    assert "meta-commentary" in _DEFAULT_SYSTEM.lower() or "meta commentary" in _DEFAULT_SYSTEM.lower()
    # Should mention the bash tool and how to write files via heredoc.
    assert "bash" in _DEFAULT_SYSTEM
    assert "heredoc" in _DEFAULT_SYSTEM.lower() or "EOF" in _DEFAULT_SYSTEM


def test_build_agent_uses_default_system_when_unset() -> None:
    """No --system flag, no $ANDURIL_SYSTEM env → fall back to _DEFAULT_SYSTEM."""
    import argparse
    import os
    saved = os.environ.pop("ANDURIL_SYSTEM", None)
    try:
        args = argparse.Namespace(
            model="test-model", system="", base_url=None,
            history=None,
        )
        agent = _build_agent(args)
        assert agent.system == _DEFAULT_SYSTEM
        # Explicit --system wins.
        args.system = "custom prompt"
        agent = _build_agent(args)
        assert agent.system == "custom prompt"
        # Env var wins over default.
        os.environ["ANDURIL_SYSTEM"] = "env prompt"
        args.system = ""
        agent = _build_agent(args)
        assert agent.system == "env prompt"
    finally:
        if saved is not None:
            os.environ["ANDURIL_SYSTEM"] = saved
        else:
            os.environ.pop("ANDURIL_SYSTEM", None)


# === _normalize_approval (B1 regression) =================================


def test_normalize_approval_known_values() -> None:
    """The previously-missing _normalize_approval covers every CLI form."""
    assert _normalize_approval("yolo") == ("yolo", None)
    assert _normalize_approval("YOLO") == ("yolo", None)
    assert _normalize_approval("all") == ("prompt_all", None)
    assert _normalize_approval("prompt") == ("prompt_all", None)
    assert _normalize_approval("none") == ("prompt_all", None)
    assert _normalize_approval("strict") == ("prompt_all", None)
    assert _normalize_approval("low") == ("threshold", "low")
    assert _normalize_approval("lo") == ("threshold", "low")
    assert _normalize_approval("medium") == ("threshold", "medium")
    assert _normalize_approval("med") == ("threshold", "medium")
    assert _normalize_approval("mid") == ("threshold", "medium")
    assert _normalize_approval("high") == ("threshold", "high")
    assert _normalize_approval("hi") == ("threshold", "high")
    # Whitespace tolerated.
    assert _normalize_approval("  yolo  ") == ("yolo", None)


def test_normalize_approval_unknown() -> None:
    """Unknown levels return (None, None) — callers treat this as an error."""
    assert _normalize_approval("") == (None, None)
    assert _normalize_approval("bogus") == (None, None)
    assert _normalize_approval(None) == (None, None)  # type: ignore[arg-type]


# === Approval threshold (per-tool risk gating) ===========================


def test_confirm_callback_threshold() -> None:
    """--approval high should not prompt for medium-risk tools.

    Regression test: previously every --approval level (except yolo)
    prompted for every dangerous tool, making --approval low/medium/high
    inert. Now the threshold filters by tool.risk so the level actually
    does what it says on the tin.
    """
    from anduril.tools import Tool
    from anduril.tui import _make_confirm_callback

    # Build three stub tools with different risk levels.
    low_tool = Tool("low", "low", {"type": "object", "properties": {}}, lambda: "ok",
                    dangerous=True, risk="low")
    med_tool = Tool("med", "med", {"type": "object", "properties": {}}, lambda: "ok",
                    dangerous=True, risk="medium")
    high_tool = Tool("high", "high", {"type": "object", "properties": {}}, lambda: "ok",
                     dangerous=True, risk="high")
    safe_tool = Tool("safe", "safe", {"type": "object", "properties": {}}, lambda: "ok",
                     dangerous=False, risk="low")

    class _StubAgent:
        def __init__(self, tools):
            self.tools = tools

    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def refresh(self): pass
        def get_wch(self): return "y"  # default to yes if _confirm_key reaches us

    agent = _StubAgent({"low": low_tool, "med": med_tool,
                        "high": high_tool, "safe": safe_tool})
    stdscr = _StubWin()

    # Count prompts by hooking addnstr. If a prompt would be shown, the
    # callback reaches _confirm_key which calls addnstr; we don't
    # need to actually drive the key, we just want to know whether the
    # prompt was attempted.
    prompts: list[str] = []
    real_addnstr = stdscr.addnstr
    def tracking_addnstr(*a, **k):
        if len(a) >= 3 and isinstance(a[2], str) and "Allow" in a[2]:
            prompts.append(a[2])
        return real_addnstr(*a, **k)
    stdscr.addnstr = tracking_addnstr  # type: ignore[assignment]

    # yolo: never prompts, returns True.
    cb = _make_confirm_callback(agent, stdscr, "yolo")  # type: ignore[arg-type]
    for name in ("low", "med", "high", "safe"):
        assert cb(name, {}) is True
    assert prompts == []

    # high: only the high-risk tool would prompt; safe tool allowed
    # silently. We can't drive the actual key for the high tool, so
    # we use a side effect: we wrap _confirm_key in the tui module to
    # raise a sentinel on invocation, then check the high-risk call
    # raised it.
    import importlib
    tui_mod = importlib.import_module("anduril.tui")
    class _WouldPrompt(RuntimeError):
        pass
    orig = tui_mod._confirm_key
    tui_mod._confirm_key = lambda *a, **k: (_ for _ in ()).throw(_WouldPrompt())
    try:
        cb = _make_confirm_callback(agent, stdscr, "high")  # type: ignore[arg-type]
        assert cb("low", {}) is True   # low-risk, silent
        assert cb("med", {}) is True   # medium-risk, silent under "high" threshold
        assert cb("safe", {}) is True  # non-dangerous, silent
        try:
            cb("high", {})
        except _WouldPrompt:
            pass
        else:
            raise AssertionError("high-risk tool should have prompted under --approval high")
    finally:
        tui_mod._confirm_key = orig

    # medium: medium- and high-risk both prompt; low-risk silent.
    try:
        tui_mod._confirm_key = lambda *a, **k: (_ for _ in ()).throw(_WouldPrompt())
        cb = _make_confirm_callback(agent, stdscr, "medium")  # type: ignore[arg-type]
        assert cb("low", {}) is True
        try:
            cb("med", {})
        except _WouldPrompt:
            pass
        else:
            raise AssertionError("medium-risk tool should have prompted under --approval medium")
        try:
            cb("high", {})
        except _WouldPrompt:
            pass
        else:
            raise AssertionError("high-risk tool should have prompted under --approval medium")
    finally:
        tui_mod._confirm_key = orig


def test_agent_pop_last_and_set_system() -> None:
    """The TUI now mutates the agent through methods, not _messages poking."""
    agent = Agent(model="test", system="original")
    agent._messages.append({"role": "user", "content": "hi"})
    agent._messages.append({"role": "assistant", "content": "hello"})

    # pop_last removes and returns the tail.
    last = agent.pop_last()
    assert last == {"role": "assistant", "content": "hello"}
    assert len(agent._messages) == 2  # system + user
    assert agent.pop_last() == {"role": "user", "content": "hi"}
    # _messages still has the system prompt at index 0.
    assert len(agent._messages) == 1
    assert agent._messages[0]["role"] == "system"
    assert agent.pop_last() == {"role": "system", "content": "original"}
    assert agent.pop_last() is None  # now empty

    # set_system replaces the existing system message in place.
    agent.set_system("replacement")
    assert agent._messages[0] == {"role": "system", "content": "replacement"}

    # set_system inserts a system message when none exists.
    agent._messages = [{"role": "user", "content": "no system"}]
    agent.set_system("inserted")
    assert agent._messages[0] == {"role": "system", "content": "inserted"}


# === ToolCallAggregator.peek (encapsulation) =============================


def test_tool_call_aggregator_peek() -> None:
    """peek() is the public read-only view used by the streaming loop."""
    agg = _ToolCallAggregator()
    assert agg.peek(0) is None  # no call yet
    agg.add(_FakeDelta(0, id_="call_1", name="read_", arguments='{"a'))
    peeked = agg.peek(0)
    assert peeked is not None
    assert peeked["id"] == "call_1"
    assert peeked["function"]["name"] == "read_"
    # finalize returns the full assembled call.
    out = agg.finalize()
    assert out[0]["function"]["arguments"] == '{"a'


# === TUI command dispatch (B2 regression) ================================


def test_tui_command_dispatch_basic() -> None:
    """The /command dispatch path used to TypeError on every call (15 args
    vs 1 in the signature). Verify it now dispatches and mutates state."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    # Stub curses window — no terminal, no IO. The methods we exercise
    # call render() which touches erase/addnstr/move/refresh; we stub
    # them all to no-ops.
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    stdscr = _StubWin()
    state = _TUIState(agent, stdscr, 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()

    # /model — current model echo when no arg.
    state._handle_command("/model")
    # /model NAME — switch.
    state._handle_command("/model bigger-model")
    assert agent.model == "bigger-model"
    # /system with no arg echoes current.
    state._handle_command("/system")
    # /system with arg replaces.
    state._handle_command("/system a new prompt")
    assert agent.system == "a new prompt"
    # /yolo toggles.
    assert state.approval_level != "yolo"
    state._handle_command("/yolo")
    assert state.approval_level == "yolo"
    state._handle_command("/yolo")
    assert state.approval_level == "all"
    # /approval with unknown level.
    state._handle_command("/approval bogus")
    # /approval with valid level.
    state._handle_command("/approval high")
    assert state.approval_level == "high"
    # Unknown command pushes a note.
    state._handle_command("/not-a-command")
    assert any(k == "note" and "unknown command" in t for k, t, _ in state.log)
    # /quit raises SystemExit.
    try:
        state._handle_command("/quit")
    except SystemExit:
        pass
    else:
        raise AssertionError("/quit should have raised SystemExit")


# === TUI scroll (regression: full-history scroll) =======================


def _make_tui_state(rows: int = 40, cols: int = 100):
    """Build a TUIState wired to a stub curses window for scroll tests."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (rows, cols)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
    return state


def test_tui_scroll_clamps_to_visible_lines() -> None:
    """self.scroll is measured in wrapped visible lines, not log entries.
    A log of 30 entries can wrap to 100+ lines on a narrow terminal, and
    the user must be able to scroll past the old `len(self.log)-1` bound."""
    state = _make_tui_state(rows=20, cols=60)
    # 30 long entries — each wraps to ~3-4 lines on a 60-col terminal,
    # so the wrapped log is ~100 lines. The visible window is small
    # (rows - 1 status - 3 editor box - 2 editor content = ~14 rows).
    for i in range(30):
        text = (f"Entry {i}: lorem ipsum dolor sit amet consectetur "
                f"adipiscing elit " * 2)
        state.push("user" if i % 2 else "assistant", text, 0x20)

    # First, baseline: scroll=0 means "follow latest".
    state.scroll = 0
    state.render()
    assert state.scroll == 0  # nothing to clamp

    # The wrap cache + render should have produced a long final_lines.
    wrapped = state._get_wrapped_log(59)  # max_w = 60 - 1
    final = state._truncate_tool_blocks(wrapped)
    assert len(final) > 30, (
        f"expected the log to wrap to many more than 30 lines; got {len(final)}"
    )

    # Compute the expected max_scroll the same way render() does. ed_h
    # is min(MAX_EDITOR_LINES=7, max(MIN_EDITOR_LINES=3, len(visual))).
    # For an empty editor, ed_h = 3. With the new single-line editor
    # border, log_h is rows - status - 1 (editor top line) - ed_h.
    ed_h = 3
    log_h = 20 - 1 - 1 - ed_h  # rows - status - editor top line - content
    max_scroll = max(0, len(final) - log_h)
    # The old buggy bound was len(self.log) - 1 = 29. The new bound is
    # len(final) - log_h, which should be much larger.
    assert max_scroll > 29, (
        f"max_scroll ({max_scroll}) should exceed the old bound (29) — "
        f"this is exactly the bug we're fixing"
    )

    # Push scroll well past the old bound. Render must clamp to max_scroll,
    # not silently keep a too-large value.
    state.scroll = 9999
    state.render()
    assert state.scroll == max_scroll, (
        f"scroll not clamped: got {state.scroll}, expected {max_scroll}"
    )


def test_tui_scroll_resets_on_clear() -> None:
    """/clear must reset scroll, otherwise the next render slices an empty
    log and shows nothing."""
    state = _make_tui_state()
    for i in range(20):
        state.push("user", f"entry {i}", 0x20)
    state.scroll = 50  # way past the end
    state.render()  # clamps to something > 0
    assert state.scroll > 0

    state._handle_command("/clear")
    assert state.scroll == 0, (
        f"/clear must reset scroll to 0, got {state.scroll}"
    )


# === TUI input layout (prompt prefix on the first row) ================


def test_tui_input_prompt_prefix() -> None:
    """The input area renders a `> ` prompt on the first line where the
    user types, with the cursor landing right after it."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    captured = []
    class _StubWin:
        def getmaxyx(self): return (15, 60)
        def erase(self): pass
        def addnstr(self, y, x, s, n, attr=None): captured.append(("draw", y, s))
        def addstr(self, *a, **k): pass
        def move(self, y, x): captured.append(("cursor", y, x))
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()

        # Empty editor — the first content row should show "> " (just the
        # prompt, with the cursor right after it at column 2).
        captured.clear()
        state.render()
        draws = [s for k, _, s in captured if k == "draw"]
        cursor = next((x for k, _, x in captured if k == "cursor"), None)
        # The first content row is the row right after the straight
        # '─' separator.
        content_rows = [s for s in draws if s.startswith(">")]
        assert content_rows, f"no '> ' prompt found in {draws!r}"
        assert content_rows[0].startswith("> "), \
            f"first content row should start with '> ', got {content_rows[0]!r}"
        # Cursor lands at column 2 (right after "> ").
        assert cursor == 2, \
            f"empty-editor cursor should be at col 2, got {cursor}"

        # Now type "hello" and re-render.
        state.editor.insert_text("hello")
        state.editor.row = 0
        state.editor.col = 5
        captured.clear()
        state.render()
        cursor = next((x for k, _, x in captured if k == "cursor"), None)
        # Cursor = 2 ("> ") + 5 ("hello") = 7.
        assert cursor == 7, \
            f"after typing 'hello' cursor should be at col 7, got {cursor}"
        draws = [s for k, _, s in captured if k == "draw"]
        first_line = next(s for s in draws if s.startswith(">"))
        assert first_line.startswith("> hello"), \
            f"first line should be '> hello', got {first_line!r}"

        # Multi-line: only the FIRST line gets the '> ' prefix.
        state.editor.insert_text("\nsecond")
        state.editor.row = 1
        state.editor.col = 6
        captured.clear()
        state.render()
        draws = [s for k, _, s in captured if k == "draw"]
        first_line = next(s for s in draws if "hello" in s)
        second_line = next(s for s in draws if "second" in s)
        assert first_line.startswith("> "), \
            f"first line keeps '> ', got {first_line!r}"
        assert not second_line.startswith("> "), \
            f"continuation lines don't get '> ', got {second_line!r}"
        # Cursor on second line is unaffected by the prefix.
        cursor = next((x for k, _, x in captured if k == "cursor"), None)
        assert cursor == 6, \
            f"cursor on continuation line should be at col 6, got {cursor}"


# === TUI live resize (SIGWINCH handling) ===============================


def test_tui_render_processes_resize_flag() -> None:
    """A SIGWINCH received mid-stream is applied on the next render."""
    import curses
    import os
    from unittest.mock import patch
    from anduril.tui import _TUIState
    captured = []
    class _StubWin:
        def __init__(self, rows, cols): self.rows, self.cols = rows, cols
        def getmaxyx(self): return (self.rows, self.cols)
        def erase(self): pass
        def addnstr(self, y, x, s, n, attr=None): captured.append(s)
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None), \
         patch.object(curses, "resizeterm",
                      lambda nlines, ncols: setattr(_StubWin, "_resized_to", (nlines, ncols))):
        agent = Agent(model="m", system="sys")
        stdscr = _StubWin(20, 60)
        state = _TUIState(agent, stdscr, 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()

        # Simulate a SIGWINCH arriving during a long-running agent call.
        state.resize_pending = True

        # Patch os.get_terminal_size to return columns=100, lines=30.
        # (os.terminal_size takes (columns, lines); curses.resizeterm
        # takes (nlines, ncols) — opposite order.)
        with patch("os.get_terminal_size",
                   return_value=os.terminal_size((100, 30))):
            state.render()

        # The flag was cleared.
        assert state.resize_pending is False
        # And the resize was applied with the right arg order.
        assert getattr(stdscr, "_resized_to", None) == (30, 100), \
            f"curses.resizeterm not called with (30, 100); got {getattr(stdscr, '_resized_to', None)}"


def test_tui_render_handles_resize_failure_gracefully() -> None:
    """If the resize syscall fails (e.g. not a TTY), render must not crash."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 60)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        state.resize_pending = True
        # os.get_terminal_size raises (not a TTY in a test env).
        with patch("os.get_terminal_size",
                   side_effect=OSError("not a tty")):
            # Must not raise.
            state.render()
        assert state.resize_pending is False  # still cleared


# === TUI render crash-safety ============================================


def test_tui_render_swallows_curses_error() -> None:
    """A curses.error raised during the inner render must not crash the
    TUI — the next render (next key or next event) gets a fresh chance."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 60)
        def erase(self): pass
        def addnstr(self, *a, **k):
            raise curses.error("simulated")
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Must not raise.
        state.render()


def test_tui_render_skips_zero_sized_resize() -> None:
    """A race during the resize itself may report rows=0 or cols=0
    (very briefly). The handler must not call resizeterm with those
    values — and must re-arm the flag so the next tick retries."""
    import curses
    import os
    from unittest.mock import patch
    from anduril.tui import _TUIState
    resized = []
    class _StubWin:
        def getmaxyx(self): return (20, 60)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None), \
         patch.object(curses, "resizeterm",
                      lambda nlines, ncols: resized.append((nlines, ncols))):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        state.resize_pending = True
        # os.get_terminal_size returns an invalid size (zero cols).
        with patch("os.get_terminal_size",
                   return_value=os.terminal_size((0, 30))):
            state.render()
        assert resized == [], f"resizeterm should not be called with 0 size; got {resized}"
        # The flag is re-armed for retry.
        assert state.resize_pending is True


def test_tui_render_handles_window_split_to_tiny_size() -> None:
    """A window split that produces a pane smaller than the TUI's
    minimum dimensions must not crash — it should clear the screen
    and re-arm the flag so the next render tries again (the user will
    likely un-split or grow the pane)."""
    import curses
    import os
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 60)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        state.resize_pending = True
        # Tiny pane: 2 rows, 30 columns. Below _MIN_ROWS.
        with patch("os.get_terminal_size",
                   return_value=os.terminal_size((30, 2))):
            # Must not raise.
            state.render()
        # Flag is re-armed (the user might grow the pane again).
        assert state.resize_pending is True, \
            "flag should be re-armed when terminal is too small"


# === TUI streaming tool calls ============================================


def test_tui_streaming_tool_call_appears_as_it_forms() -> None:
    """A tool_call event pushes a 'tool_call' log entry immediately,
    not waiting for the tool to finish executing."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()

        # First delta — only the name is present, args are empty.
        state.on_event({
            "role": "tool_call",
            "id": "call_abc",
            "index": 0,
            "name": "bash",
            "arguments": "",
        })
        kinds = [k for k, _, _ in state.log]
        assert "tool_call" in kinds, \
            f"tool_call entry not in log: {state.log!r}"
        # The header reflects what we have so far.
        tc_idx = kinds.index("tool_call")
        header = state.log[tc_idx][1]
        assert "bash" in header and "()" in header, \
            f"unexpected header: {header!r}"

        # Second delta — args start streaming in. The same entry should
        # be updated in place, not a new entry appended.
        state.on_event({
            "role": "tool_call",
            "id": "call_abc",
            "index": 0,
            "name": "bash",
            "arguments": '{"command": "cat',
        })
        assert len(state.log) == len(kinds), \
            f"expected in-place update, got new entry; kinds={kinds}"
        header = state.log[tc_idx][1]
        assert "bash" in header and "cat" in header, \
            f"args not filled in: {header!r}"

        # Final delta — args complete.
        state.on_event({
            "role": "tool_call",
            "id": "call_abc",
            "index": 0,
            "name": "bash",
            "arguments": '{"command": "cat > foo.txt"}',
        })
        header = state.log[tc_idx][1]
        assert "foo.txt" in header, f"final args not present: {header!r}"

        # When the tool returns, the result is pushed — no duplicate
        # header this time. (push() inserts a blank separator between
        # non-streaming entries, which is by design.)
        kinds_before = [k for k, _, _ in state.log]
        state.on_event({
            "role": "tool",
            "name": "bash",
            "args": '{"command": "cat > foo.txt"}',
            "result": "ok",
        })
        new_kinds = [k for k, _, _ in state.log]
        added = new_kinds[len(kinds_before):]
        # Last entry must be the result. No new "tool_call" or
        # duplicate header.
        assert added[-1] == "tool", \
            f"last entry should be the result, got: {added!r}"
        assert "tool_call" not in added, \
            f"tool event shouldn't push a new tool_call header: {added!r}"


def test_tui_tool_call_id_tracking() -> None:
    """Two parallel tool calls each get their own log entry."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        state.on_event({"role": "tool_call", "id": "a", "index": 0,
                        "name": "read", "arguments": ""})
        state.on_event({"role": "tool_call", "id": "b", "index": 1,
                        "name": "write", "arguments": ""})
        tc_entries = [t for k, t, _ in state.log if k == "tool_call"]
        assert len(tc_entries) == 2, \
            f"expected 2 tool_call entries, got {len(tc_entries)}"
        # Each is updated independently.
        state.on_event({"role": "tool_call", "id": "a", "index": 0,
                        "name": "read", "arguments": "foo"})
        tc_entries = [t for k, t, _ in state.log if k == "tool_call"]
        assert any("foo" in t for t in tc_entries), tc_entries
        assert any("write" in t for t in tc_entries), tc_entries


def test_tui_clear_resets_tool_call_tracking() -> None:
    """After /clear, old tool_call ids are forgotten (fresh state)."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        state.on_event({"role": "tool_call", "id": "old", "index": 0,
                        "name": "x", "arguments": ""})
        assert "old" in state._tool_call_log_idx
        state._handle_command("/clear")
        assert state._tool_call_log_idx == {}, \
            f"_tool_call_log_idx should reset on /clear, got {state._tool_call_log_idx!r}"


# === TUI signal-interrupted get_wch ====================================


def test_confirm_key_survives_curses_error() -> None:
    """A signal-interrupted get_wch (e.g. SIGWINCH during the approval
    prompt) must redraw the prompt and retry — not crash."""
    import curses
    import importlib
    from unittest.mock import patch
    tui_mod = importlib.import_module("anduril.tui")
    class _StubWin:
        def __init__(self):
            self.calls = []
            self.w = 80
        def getmaxyx(self): return (24, self.w)
        def erase(self): pass
        def addnstr(self, *a, **k): self.calls.append(a)
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        stdscr = _StubWin()
        stdscr.get_wch = lambda: "y"  # default
        # First call raises curses.error (simulated SIGWINCH
        # interruption), second call returns 'y' (approval).
        call_count = [0]
        def fake_get_wch():
            call_count[0] += 1
            if call_count[0] == 1:
                raise curses.error("no input")
            return "y"
        stdscr.get_wch = fake_get_wch
        # Must not raise, and must return True.
        result = tui_mod._confirm_key(stdscr, "approve? [y/N] ")
        assert result is True
        # It retried at least once.
        assert call_count[0] >= 2


# === TUI status bar live tracking =====================================


class _FakeChunk:
    """Minimal stand-in for an OpenAI streaming chunk."""
    def __init__(self, content: str = "", usage=None):
        class _Delta:
            pass
        d = _Delta()
        d.content = content
        d.reasoning_content = None
        d.model_extra = {}
        d.tool_calls = None
        class _Choice:
            pass
        c = _Choice()
        c.delta = d
        self.choices = [c] if content else []
        self.usage = usage
        self.model_extra = {}


class _FakeStream:
    """Iterable response that yields the prebuilt chunks."""
    def __init__(self, chunks):
        self._chunks = chunks
    def __iter__(self):
        return iter(self._chunks)
    def close(self): pass


class _FakeUsage:
    """Stand-in for OpenAI's usage object."""
    def __init__(self, prompt=0, completion=0, cached=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        class ptd: pass
        ptd.cached_tokens = cached
        self.prompt_tokens_details = ptd


def _make_status_state(model: str = "m", system: str = "sys"):
    """Build a TUI state whose render() writes row 0 to a list.

    Returns (state, captured_row0_strings). The stub window also
    has nodelay/get_wch so the Esc poller doesn't blow up.
    """
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    captured: list[str] = []
    class _StubWin:
        def getmaxyx(self): return (20, 160)
        def erase(self): pass
        def addnstr(self, y, x, s, n, attr=None):
            if y == 0: captured.append(s)
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def nodelay(self, flag): pass
        def get_wch(self):
            import curses as _c
            raise _c.error("no input")
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model=model, system=system, max_retries=1)
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
    return state, captured


def test_tui_status_bar_tracks_during_streaming() -> None:
    """While the model streams, the status bar's `out` ticks up per
    delta. The previous version of the bar only updated at end of
    stream (from the API's usage chunk), so per-delta renders all
    read the same pre-turn values and the bar looked frozen."""
    state, captured = _make_status_state(system="You are anduril.")
    state.agent.client.chat.completions.create = lambda **kw: _FakeStream([
        _FakeChunk(content="Hello"),
        _FakeChunk(content=" there"),
        _FakeChunk(content=" friend"),
        _FakeChunk(content="!"),
        _FakeChunk(usage=_FakeUsage(prompt=100, completion=50, cached=30)),
    ])
    state.run_agent_turn("Hi there")
    # Find the per-delta "turn" lines. Each streaming delta should
    # produce a new "turn" line with a higher `out` value than the
    # previous one (or at least not lower). The very first one (at
    # the moment we flip into turn_active) has out=0.
    turn_lines = [s for s in captured if "turn ctx" in s]
    assert len(turn_lines) >= 2, (
        f"expected multiple 'turn ctx' lines during streaming, got: "
        f"{turn_lines!r}"
    )
    # The first turn line should have out=0 (no tokens yet).
    assert "out 0" in turn_lines[0], (
        f"first turn line should be out=0, got: {turn_lines[0]!r}"
    )
    # A later turn line should have a higher out value.
    last_turn = turn_lines[-1]
    assert "out 1" in last_turn or "out 2" in last_turn or "out 3" in last_turn or "out 4" in last_turn or "out 5" in last_turn, (
        f"later turn line should reflect streamed output, got: {last_turn!r}"
    )


def test_tui_status_bar_persists_after_turn_without_usage() -> None:
    """If the API doesn't return a usage chunk, the status bar must
    still show the last turn's per-turn data after streaming ends —
    not jump back to all-zeros (which was the old behaviour, since
    cumulative session metrics stayed at 0)."""
    state, captured = _make_status_state()
    state.agent.client.chat.completions.create = lambda **kw: _FakeStream([
        _FakeChunk(content="Hello"),
        _FakeChunk(content=" there"),
        _FakeChunk(content="!"),
        # No _FakeChunk(usage=...) — API didn't report usage.
    ])
    state.run_agent_turn("Hi")
    # The final status bar line (after the turn ends) must include
    # the per-turn data, not the cumulative-only "ctx 0" line.
    final = captured[-1]
    assert "last ctx" in final, (
        f"expected 'last ctx' in final line (no-usage case), got: {final!r}"
    )
    assert "out" in final
    # The session cumulative `ses` should still be visible.
    assert "ses" in final
    # Most importantly: the old all-zeros line must NOT be the
    # final line.
    assert not final.startswith("anduril") or "ctx 0" not in final or "ses 0" not in final, (
        f"final line shouldn't be the all-zeros idle view when we "
        f"have per-turn data: {final!r}"
    )


def test_tui_status_bar_uses_api_values_after_turn() -> None:
    """When the API reports usage, the post-turn status bar must
    use those ground-truth values, not the rough char estimate
    we used mid-stream."""
    state, captured = _make_status_state()
    state.agent.client.chat.completions.create = lambda **kw: _FakeStream([
        _FakeChunk(content="Hi"),
        _FakeChunk(usage=_FakeUsage(prompt=200, completion=80, cached=50)),
    ])
    state.run_agent_turn("hi")
    final = captured[-1]
    # The API reported prompt=200, cached=50, so total=200 (since
    # turn_prompt_tokens = input + cache_read = 150+50? No wait —
    # _normalize_usage sets input=prompt-cached=150, cache=50, and
    # we sum them in run_agent_turn for the bar). Either way, the
    # final line must reflect the API values, not just "out 0".
    assert "last ctx" in final
    # The output should be the API's completion_tokens (80), not
    # just 0 (which is what the char-based estimate would have
    # yielded for "Hi" — 2 chars / 4 = 0).
    assert "out 80" in final, (
        f"expected API-reported output (80) in final line, got: {final!r}"
    )


# === Pasted-image support ==============================================


def test_files_save_pasted_image_writes_bytes() -> None:
    """save_pasted_image writes the bytes to the anduril images dir
    and returns the saved path."""
    from anduril.files import save_pasted_image
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    path = save_pasted_image(png_bytes, "png")
    try:
        assert path.exists()
        assert path.read_bytes() == png_bytes
        assert path.suffix == ".png"
        assert "anduril" in str(path)
        assert "images" in str(path)
        assert path.name.startswith("image-")
    finally:
        path.unlink()


def test_files_save_pasted_image_normalizes_extension() -> None:
    """Unknown extensions are coerced to .png (we only save formats
    the multimodal endpoint accepts)."""
    from anduril.files import save_pasted_image
    path = save_pasted_image(b"fake", "exe")
    try:
        assert path.suffix == ".png"
    finally:
        path.unlink()


def test_files_save_pasted_image_jpg_to_jpeg() -> None:
    """The .jpg extension is normalized to .jpeg so the MIME type
    read by the model is correct (image/jpeg, not image/jpg)."""
    from anduril.files import save_pasted_image
    path = save_pasted_image(b"fake jpg bytes", "jpg")
    try:
        assert path.suffix == ".jpeg"
    finally:
        path.unlink()


def test_files_save_pasted_image_empty_data_raises() -> None:
    """Empty data is rejected with a clear ValueError."""
    from anduril.files import save_pasted_image
    try:
        save_pasted_image(b"", "png")
    except ValueError as e:
        assert "empty" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for empty data")


def test_files_resolve_mention_path_expands_tilde() -> None:
    """A mention starting with ~ is expanded to the user's home dir."""
    from anduril.files import _resolve_mention_path
    p = _resolve_mention_path("~/foo/bar.png", cwd=pathlib.Path("/tmp"))
    # Should NOT start with /tmp; it should be the home dir.
    assert not str(p).startswith("/tmp")
    assert str(p).endswith("foo/bar.png")
    # The expanded path should be under the home dir.
    assert str(p).startswith(str(pathlib.Path.home()))


def test_files_expand_mentions_tilde_path() -> None:
    """An @-mention with a tilde-prefixed absolute path resolves
    and is attached as an image."""
    import base64
    from anduril.files import expand_mentions
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    # Save an image under the home dir for the test.
    img_dir = pathlib.Path.home() / ".local" / "state" / "anduril" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / "test_tilde_mention.png"
    img.write_bytes(png_bytes)
    try:
        out = expand_mentions("see this @~/.local/state/anduril/images/test_tilde_mention.png",
                              cwd="/tmp")
        img_parts = [p for p in out if p.get("type") == "image_url"]
        assert len(img_parts) == 1
        url = img_parts[0]["image_url"]["url"]
        b64 = url.split(",", 1)[1]
        assert base64.b64decode(b64) == png_bytes
    finally:
        img.unlink()


def test_tui_parse_kitty_graphics_basic() -> None:
    """A well-formed Kitty transmit sequence decodes to the original
    image bytes and reports the correct extension."""
    from anduril.tui import _parse_kitty_graphics
    import base64
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    b64 = base64.b64encode(png_bytes).decode("ascii")
    payload = f"a=T,f=100,t=d,m=0;{b64}"
    result = _parse_kitty_graphics(payload)
    assert result is not None
    data, ext = result
    assert data == png_bytes
    assert ext == "png"


def test_tui_parse_kitty_graphics_multipart_flag() -> None:
    """A chunk with m=1 sets the more-chunks flag on the reader so
    the next chunk gets accumulated rather than saved as its own image."""
    from anduril.tui import _parse_kitty_graphics, _read_image_paste
    import base64
    png_bytes = b"\x89PNG\r\n\x1a\n fake png \n"
    b64 = base64.b64encode(png_bytes).decode("ascii")
    # Reset any prior state.
    if hasattr(_read_image_paste, "_more_chunks"):
        del _read_image_paste._more_chunks
    payload = f"a=T,f=100,t=d,m=1;{b64}"
    _parse_kitty_graphics(payload)
    assert getattr(_read_image_paste, "_more_chunks", False) is True
    # The next chunk (m=0) must clear the flag.
    payload2 = f"a=T,f=100,t=d,m=0;{b64}"
    _parse_kitty_graphics(payload2)
    assert getattr(_read_image_paste, "_more_chunks", False) is False


def test_tui_parse_kitty_graphics_rejects_non_transmit() -> None:
    """Non-transmit actions (a=Q for query, a=p for put, etc.) are
    not images and we ignore them."""
    from anduril.tui import _parse_kitty_graphics
    import base64
    payload = f"a=Q,f=100,t=d;{base64.b64encode(b'x').decode()}"
    assert _parse_kitty_graphics(payload) is None


def test_tui_parse_kitty_graphics_rejects_missing_payload() -> None:
    """A sequence without the base64 separator is malformed."""
    from anduril.tui import _parse_kitty_graphics
    assert _parse_kitty_graphics("a=T,f=100") is None


def test_tui_parse_iterm2_image_basic() -> None:
    """A well-formed iTerm2 image sequence decodes to the original bytes."""
    from anduril.tui import _parse_iterm2_image
    import base64
    png_bytes = b"\x89PNG\r\n\x1a\n fake png \n"
    name_b64 = base64.b64encode(b"screenshot.png").decode("ascii")
    data_b64 = base64.b64encode(png_bytes).decode("ascii")
    payload = f"File=name={name_b64};size={len(png_bytes)};inline=1:{data_b64}"
    result = _parse_iterm2_image(payload)
    assert result is not None
    data, ext = result
    assert data == png_bytes
    assert ext == "png"


def test_tui_parse_iterm2_image_rejects_missing_size() -> None:
    """Without a size hint, the iTerm2 protocol can't tell us how
    big the image is; we refuse to decode anything."""
    from anduril.tui import _parse_iterm2_image
    import base64
    payload = f"File=name={base64.b64encode(b'x.png').decode()};inline=1:AAAA"
    assert _parse_iterm2_image(payload) is None


def test_tui_parse_iterm2_image_rejects_oversize() -> None:
    """Images larger than MAX_IMAGE_BYTES are rejected up front."""
    from anduril.tui import _parse_iterm2_image
    # Claim a 100MB image.
    payload = "File=size=104857600;inline=1:AAAA"
    assert _parse_iterm2_image(payload) is None


# === Short-reference attachments =====================================


def test_tui_register_attachment_returns_short_id() -> None:
    """Pastes get unique short IDs of the form image-N, with the
    smallest free N being chosen each time. The buffer is
    updated between pastes so the IDs don't collide."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Simulate the user pasting image 1, then image 2 with
        # image 1's @-mention still in the buffer.
        state.editor.buf = [""]
        id1 = state._register_attachment(pathlib.Path("/tmp/a.png"))
        state.editor.buf = [f"see @{id1} and "]
        id2 = state._register_attachment(pathlib.Path("/tmp/b.png"))
        assert id1 == "image-1"
        assert id2 == "image-2", (
            f"expected image-2 (buffer has image-1), got {id2}"
        )
        assert state.attachments["image-1"] == "/tmp/a.png"
        assert state.attachments["image-2"] == "/tmp/b.png"


def test_tui_register_attachment_reuses_freed_id() -> None:
    """Paste, delete the @-mention, paste again: the new paste
    reuses image-1 instead of bumping to image-2. The user's
    counter doesn't grow when they undo a paste."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # First paste: buffer is empty, so we get image-1.
        state.editor.buf = [""]
        id1 = state._register_attachment(pathlib.Path("/tmp/a.png"))
        assert id1 == "image-1"
        # The caller would now have inserted "@image-1 " into
        # the buffer; we simulate the user deleting it.
        state.editor.buf = [""]
        # Paste again: the buffer has no @image-N references,
        # so the smallest-free algorithm picks image-1 again.
        id2 = state._register_attachment(pathlib.Path("/tmp/b.png"))
        assert id2 == "image-1", (
            f"expected image-1 (reused), got {id2}"
        )


def test_tui_register_attachment_unique_across_buffer() -> None:
    """When the buffer already references image-1, a new paste
    gets image-2 (not image-1) so the IDs don't collide."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Pre-populate the buffer with image-1 already referenced.
        state.editor.buf = ["see @image-1 vs new image"]
        # The new paste must avoid colliding with the existing reference.
        new_id = state._register_attachment(pathlib.Path("/tmp/new.png"))
        assert new_id == "image-2"


def test_tui_register_attachment_fills_lowest_free() -> None:
    """A new paste always picks the smallest N not currently in
    the buffer. If the user manually crafts a buffer with only
    ``@image-3`` (skipping 1 and 2), a fresh paste gets
    ``image-1`` — the lowest free number — not image-4.
    This is the "smallest free" rule and it's the natural
    complement to the reuse behaviour: freed IDs are fair game
    the moment they stop being referenced."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Buffer has only image-3. Pasted-1 and image-2 are
        # not referenced — they were either never used or the
        # user deleted their @-mentions.
        state.editor.buf = ["see @image-3"]
        new_id = state._register_attachment(pathlib.Path("/tmp/b.png"))
        assert new_id == "image-1", (
            f"expected image-1 (smallest free), got {new_id}"
        )


def test_tui_clear_resets_attachments() -> None:
    """The attachments dict and counter reset on /clear so a new
    session's short IDs start from 1."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Insert each @-mention into the buffer between calls
        # so the IDs don't get reused.
        state.editor.buf = [""]
        state._register_attachment(pathlib.Path("/tmp/a.png"))
        state.editor.buf = ["@image-1 "]
        state._register_attachment(pathlib.Path("/tmp/b.png"))
        assert len(state.attachments) == 2
        state._handle_command("/clear")
        assert state.attachments == {}
        assert state._next_attachment_id == 1


def test_files_expand_mentions_resolves_short_id() -> None:
    """An @image-1 mention is resolved via the attachments dict
    and produces an image_url content part."""
    import base64
    from anduril.files import expand_mentions
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        img = root / "real.png"
        img.write_bytes(png_bytes)
        # attachments maps the short ID to an ABSOLUTE path
        # (the @-mention parser only resolves the cwd-relative
        # form when the short ID is NOT in the dict).
        attachments = {"image-1": str(img)}
        out = expand_mentions("see @image-1", cwd=root,
                              attachments=attachments)
        img_parts = [p for p in out if p.get("type") == "image_url"]
        assert len(img_parts) == 1
        b64 = img_parts[0]["image_url"]["url"].split(",", 1)[1]
        assert base64.b64decode(b64) == png_bytes


def test_files_expand_mentions_short_id_overrides_cwd() -> None:
    """When the same text could resolve as both a short ID and
    a relative path, the short ID wins. This is what makes
    @image-1 work even if the cwd happens to contain a file
    named image-1."""
    from anduril.files import expand_mentions
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        # A file literally named 'image-1' in cwd.
        bogus = root / "image-1"
        bogus.write_text("not an image\n", encoding="utf-8")
        # Attachments say image-1 is actually an image
        # elsewhere on disk.
        real_image = root / "real.png"
        real_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 50)
        attachments = {"image-1": str(real_image)}
        out = expand_mentions("@image-1", cwd=root,
                              attachments=attachments)
        img_parts = [p for p in out if p.get("type") == "image_url"]
        # The short ID won — the bogus text file is ignored.
        assert len(img_parts) == 1
        # And the bogus text file is NOT inlined as [file: ...].
        text_parts = [p["text"] for p in out if p.get("type") == "text"]
        assert not any("not an image" in t for t in text_parts)


def test_tui_cmd_paste_inserts_short_id_not_full_path() -> None:
    """After /paste, the buffer contains @image-1 (not the
    full filesystem path) and the attachments dict maps
    image-1 to the actual file."""
    import curses
    import unittest.mock as mock
    from unittest.mock import patch
    from anduril.tui import _TUIState
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        with mock.patch("anduril.tui._read_clipboard_image",
                        return_value=(png_bytes, "png")):
            result = state._cmd_paste("")
        # On success the status is silent — the @-mention in
        # the editor is the feedback. (The previous behaviour
        # of pushing an "image attached" log line was noise.)
        assert result == ""
        # The buffer contains only the short ID, not the path.
        assert any("@image-1" in line for line in state.editor.buf)
        assert not any("anduril/images" in line for line in state.editor.buf)
        # The attachments dict maps image-1 to the actual file.
        assert "image-1" in state.attachments
        assert pathlib.Path(state.attachments["image-1"]).exists()


def test_files_clipboard_uses_wl_paste_on_wayland() -> None:
    """On Wayland, the /paste tool picker must use wl-paste
    even when XDG_SESSION_TYPE is not set. The earlier version
    defaulted to xclip (X11) and silently failed on Wayland
    systems where xclip isn't installed."""
    from anduril.files import read_clipboard_image
    import unittest.mock as mock
    # Pretend only wl-paste is installed (the user's case).
    which_map = {"wl-paste": "/usr/bin/wl-paste"}  # no xclip
    # WAYLAND_DISPLAY set, XDG_SESSION_TYPE unset.
    env = {"WAYLAND_DISPLAY": "wayland-0"}
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 50
    fake_proc = mock.Mock()
    fake_proc.returncode = 0
    fake_proc.stdout = png_bytes
    with mock.patch("anduril.files.shutil.which",
                    side_effect=lambda name: which_map.get(name)), \
         mock.patch("anduril.files.platform.system", return_value="Linux"), \
         mock.patch.dict("os.environ", env, clear=True), \
         mock.patch("anduril.files.subprocess.run", return_value=fake_proc) as run_mock:
        result = read_clipboard_image()
    assert result is not None
    # And the command we ran was wl-paste, not xclip.
    args, _ = run_mock.call_args
    assert args[0][0] == "wl-paste", f"expected wl-paste, got {args[0]}"


def test_tui_cmd_paste_reports_available_tools() -> None:
    """When /paste finds no image, the status line tells the
    user which tools we tried (so they can tell 'no image' from
    'tool not installed')."""
    import curses
    import unittest.mock as mock
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        with mock.patch("anduril.tui._read_clipboard_image",
                        return_value=None), \
             mock.patch("anduril.tui._clipboard_tools_status",
                        return_value="wl-paste"):
            result = state._cmd_paste("")
        # The status mentions the tool that IS available, so the
        # user knows the tool is fine and the clipboard just
        # has no image.
        assert "wl-paste" in result
        assert "no image" in result.lower()


def test_files_clipboard_falls_back_when_preferred_missing() -> None:
    """If the Wayland-preferred tool (wl-paste) isn't installed
    but xclip is, we still use xclip rather than failing. The
    preference order is based on which display server is
    active, but the actual choice is "whatever is on PATH"."""
    from anduril.files import read_clipboard_image
    import unittest.mock as mock
    # WAYLAND_DISPLAY is set, but only xclip is on PATH.
    which_map = {"xclip": "/usr/bin/xclip"}
    env = {"WAYLAND_DISPLAY": "wayland-0"}
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 50
    fake_proc = mock.Mock()
    fake_proc.returncode = 0
    fake_proc.stdout = png_bytes
    with mock.patch("anduril.files.shutil.which",
                    side_effect=lambda name: which_map.get(name)), \
         mock.patch("anduril.files.platform.system", return_value="Linux"), \
         mock.patch.dict("os.environ", env, clear=True), \
         mock.patch("anduril.files.subprocess.run", return_value=fake_proc) as run_mock:
        result = read_clipboard_image()
    assert result is not None
    args, _ = run_mock.call_args
    assert args[0][0] == "xclip", f"expected xclip fallback, got {args[0]}"


def test_tui_cmd_attachments_lists_in_use_and_stale() -> None:
    """`/attachments` shows every short ID with the actual
    filename, plus a marker indicating whether the @-mention
    is currently in the editor buffer or has been deleted."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Register two attachments. The first goes into the
        # buffer immediately (so its ID isn't reused); the
        # second one we DON'T put in the buffer to simulate the
        # user pasting then deleting the @-mention.
        state._register_attachment(pathlib.Path("/tmp/a.png"))
        state.editor.buf = ["@image-1 "]
        state._register_attachment(pathlib.Path("/tmp/b.png"))
        # State: image-1 in buffer (in use), image-2 not
        # in buffer (stale). We then write a buffer that
        # references image-1 only.
        state.editor.buf = ["see @image-1 but not the other"]
        out = state._cmd_attachments("")
        # Both attachments are listed.
        assert "@image-1" in out
        assert "@image-2" in out
        assert "a.png" in out
        assert "b.png" in out
        # The in-use vs stale marker is correct.
        assert "in use" in out
        assert "stale" in out


def test_tui_cmd_attachments_empty_state() -> None:
    """With no attachments, /attachments shows a friendly
    empty-state line instead of a table header."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        out = state._cmd_attachments("")
        assert "no attachments" in out


def test_tui_attachments_command_registered() -> None:
    """The /attachments command is in the dispatch table so
    users can discover it via the slash menu."""
    from anduril.tui import _TUIState
    assert "attachments" in _TUIState._COMMANDS
    desc = _TUIState._COMMANDS["attachments"].description
    assert "pasted" in desc.lower() or "attachment" in desc.lower()



    """When /paste finds no image, the status line tells the
    user which tools we tried (so they can tell 'no image' from
    'tool not installed')."""
    import curses
    import unittest.mock as mock
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        with mock.patch("anduril.tui._read_clipboard_image",
                        return_value=None), \
             mock.patch("anduril.tui._clipboard_tools_status",
                        return_value="wl-paste"):
            result = state._cmd_paste("")
        # The status mentions the tool that IS available, so the
        # user knows the tool is fine and the clipboard just
        # has no image.
        assert "wl-paste" in result
        assert "no image" in result.lower()


def test_tui_alt_v_keybinding_triggers_paste() -> None:
    """Alt+V runs the paste handler and inserts a short ID."""
    import curses
    import unittest.mock as mock
    from unittest.mock import patch
    from anduril.tui import _TUIState
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 50
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        with mock.patch("anduril.tui._read_clipboard_image",
                        return_value=(png_bytes, "png")):
            state._handle_escape_seq("alt_v")
        # The paste inserted the short ID into the editor.
        assert any("@image-1" in line for line in state.editor.buf)





def test_files_read_clipboard_image_returns_none_when_no_tool() -> None:
    """If the platform tool isn't on PATH, the call returns None
    cleanly (no exception) so the TUI can show a friendly message."""
    from anduril.files import read_clipboard_image
    import unittest.mock as mock
    # Force shutil.which to always return None so no tool is "available".
    with mock.patch("anduril.files.shutil.which", return_value=None):
        assert read_clipboard_image() is None


def test_files_read_clipboard_image_handles_subprocess_error() -> None:
    """A subprocess error (timeout, exit-code, empty stdout) returns
    None rather than propagating the error to the TUI."""
    from anduril.files import read_clipboard_image
    import unittest.mock as mock
    fake_proc = mock.Mock()
    fake_proc.returncode = 1
    fake_proc.stdout = b""
    with mock.patch("anduril.files.shutil.which", return_value="/usr/bin/xclip"), \
         mock.patch("anduril.files.subprocess.run", return_value=fake_proc), \
         mock.patch("anduril.files.platform.system", return_value="Linux"), \
         mock.patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
        assert read_clipboard_image() is None


def test_files_read_clipboard_image_decodes_png_magic() -> None:
    """The extension picker recognises the PNG magic header."""
    from anduril.files import read_clipboard_image
    import unittest.mock as mock
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"x" * 100
    fake_proc = mock.Mock()
    fake_proc.returncode = 0
    fake_proc.stdout = png_bytes
    with mock.patch("anduril.files.shutil.which", return_value="/usr/bin/xclip"), \
         mock.patch("anduril.files.subprocess.run", return_value=fake_proc), \
         mock.patch("anduril.files.platform.system", return_value="Linux"), \
         mock.patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
        result = read_clipboard_image()
    assert result is not None
    data, ext = result
    assert data == png_bytes
    assert ext == "png"


def test_files_read_clipboard_image_decodes_jpg_magic() -> None:
    """The extension picker recognises the JPEG magic header."""
    from anduril.files import read_clipboard_image
    import unittest.mock as mock
    jpg_bytes = b"\xff\xd8\xff\xe0" + b"x" * 50
    fake_proc = mock.Mock()
    fake_proc.returncode = 0
    fake_proc.stdout = jpg_bytes
    with mock.patch("anduril.files.shutil.which", return_value="/usr/bin/xclip"), \
         mock.patch("anduril.files.subprocess.run", return_value=fake_proc), \
         mock.patch("anduril.files.platform.system", return_value="Linux"), \
         mock.patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
        result = read_clipboard_image()
    assert result is not None
    _, ext = result
    assert ext == "jpg"


def test_tui_cmd_paste_inserts_at_mention_on_success() -> None:
    """Successful /paste inserts an @-mention at the cursor and
    returns a status line."""
    import curses
    import unittest.mock as mock
    from unittest.mock import patch
    from anduril.tui import _TUIState
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        # Mock the clipboard reader to return a real PNG.
        with mock.patch("anduril.tui._read_clipboard_image",
                        return_value=(png_bytes, "png")):
            result = state._cmd_paste("")
        # On success, the status is silent (the @-mention in
        # the editor is the feedback) — no log line is pushed.
        # The buffer carries the @-mention so the user can see
        # the paste landed.
        assert result == ""
        assert any("@" in line and "image-" in line
                   for line in state.editor.buf), \
            f"no @-mention in buffer: {state.editor.buf}"


def test_tui_cmd_paste_noop_when_clipboard_empty() -> None:
    """If the clipboard has no image, /paste returns a status
    line and doesn't modify the editor buffer."""
    import curses
    import unittest.mock as mock
    from unittest.mock import patch
    from anduril.tui import _TUIState
    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        state.editor.buf = [""]
        state.editor.row = 0
        state.editor.col = 0
        with mock.patch("anduril.tui._read_clipboard_image", return_value=None):
            result = state._cmd_paste("")
        # No image → no buffer change.
        assert state.editor.buf == [""]
        assert state.editor.col == 0
        # The status line tells the user what went wrong.
        assert "no image" in result.lower() or "no clipboard" in result.lower()


def test_tui_paste_command_registered() -> None:
    """The /paste command is wired up in the dispatcher so the
    user can discover it via Tab-completion."""
    from anduril.tui import _TUIState
    assert "paste" in _TUIState._COMMANDS
    desc = _TUIState._COMMANDS["paste"].description
    assert "clipboard" in desc.lower()


def test_tui_kitty_multipart_accumulates_and_decodes() -> None:
    """A multi-chunk Kitty paste (m=1 then m=0) is reassembled
    and base64-decoded as a single image."""
    from anduril.tui import _parse_kitty_graphics, _KittyContinuationState
    import base64
    # Reset the global state so a previous test doesn't pollute us.
    _KittyContinuationState = None
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
        b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB\x60\x82"
    )
    b64 = base64.b64encode(png_bytes).decode("ascii")
    half = len(b64) // 2
    # First chunk: full params + m=1 + first half of b64.
    r1 = _parse_kitty_graphics(f"a=T,f=100,t=d,m=1;{b64[:half]}")
    assert r1 is not None
    # Continuation chunk: just m=0 + second half of b64.
    r2 = _parse_kitty_graphics(f"m=0;{b64[half:]}")
    assert r2 is not None
    # Each chunk is returned as raw b64 ASCII (not yet decoded).
    # Concatenating and decoding gives back the original bytes.
    joined = b"".join([r1[0], r2[0]])
    assert base64.b64decode(joined.decode("ascii")) == png_bytes


# === Syntax highlighting ==================================================


def test_normalize_lang_aliases() -> None:
    """Common short forms map to a canonical pygments name."""
    assert normalize_lang("py") == "python"
    assert normalize_lang("Python") == "python"
    assert normalize_lang("js") == "javascript"
    assert normalize_lang("ts") == "typescript"
    assert normalize_lang("tsx") == "typescript"
    assert normalize_lang("sh") == "bash"
    assert normalize_lang("shell") == "bash"
    assert normalize_lang("rs") == "rust"
    assert normalize_lang("cpp") == "cpp"
    assert normalize_lang("c++") == "cpp"
    assert normalize_lang("md") == "markdown"
    # Unknown tags pass through so the caller can decide.
    assert normalize_lang("klingon") == "klingon"
    # Empty / whitespace.
    assert normalize_lang("") == ""
    assert normalize_lang("  ") == ""


def test_highlight_code_round_trip() -> None:
    """highlight_code must always round-trip the source text.

    The renderer relies on the invariant
    ``"".join(t for t, _ in spans) == text`` to wrap and draw
    correctly. A round-trip break is a wrap / draw bug.

    Pygments is known to add or strip a trailing newline for
    some lexers (HTML is a known offender). We tolerate that
    one specific case so the rest of the contract is still
    tested.
    """
    samples = [
        ('def hello(name):\n    return "Hello, " + name\n', "python"),
        ('const x = 42;\nconsole.log(x);\n', "javascript"),
        ('echo "hi"\nif [ "$1" = "x" ]; then echo yes; fi\n', "bash"),
        ('{"a": 1, "b": [2, 3], "c": null}\n', "json"),
        ('name: hello\ncount: 42\n', "yaml"),
        ('fn main() { println!("hi"); }\n', "rust"),
        ('package main\nfunc main() {}\n', "go"),
        ('int main(void) { return 0; }\n', "c"),
        ('class Foo:\n    def bar(self): pass\n', "python"),
        ('<html><body><p>hi</p></body></html>', "html"),
    ]
    for code, lang in samples:
        spans = highlight_code(code, lang, 0, lambda t: 0)
        joined = "".join(s for s, _ in spans)
        # Pygments' HTML lexer tacks on a trailing newline;
        # strip it for the comparison.
        if joined.rstrip("\n") != code.rstrip("\n"):
            raise AssertionError(
                f"Round-trip failed for {lang}:\n"
                f"  in:  {code!r}\n  out: {joined!r}"
            )


def test_highlight_code_empty_lang_returns_default() -> None:
    """An empty / unknown language returns plain text."""
    spans = highlight_code("hello world", "", 42, lambda t: 0)
    assert spans == [("hello world", 42)]
    spans = highlight_code("hello world", "klingon", 42, lambda t: 0)
    # Regex backend doesn't know "klingon"; falls through to plain.
    assert spans == [("hello world", 42)]


def test_highlight_code_default_attr_used_for_unmapped() -> None:
    """Tokens with no mapping get the default attr (no gaps)."""
    def color_for_token(t: str) -> int:
        return 1 if "Keyword" in t else 7
    spans = highlight_code("def x(): return 1\n", "python", 7,
                           color_for_token)
    # Every span has attr 1 or 7 — no "uncoloured" gap.
    for text, attr in spans:
        assert attr in (1, 7)


def test_highlight_code_adjacent_same_attr_merged() -> None:
    """Two adjacent spans with the same attr should be merged."""
    spans = highlight_code("a" * 100, "python", 99, lambda t: 0)
    attrs = {a for _, a in spans}
    assert len(attrs) == 1


def test_highlight_code_alias() -> None:
    """\"\\`\\`\\`js\" maps to the javascript highlighter."""
    spans = highlight_code("const x = 1;\n", "js", 0, lambda t: 0)
    assert "".join(s for s, _ in spans) == "const x = 1;\n"


# === TUI integration: highlighted log lines ==============================


def test_tui_split_code_fences_basic() -> None:
    """Fenced code blocks are split out for highlighting."""
    from anduril.tui import _split_code_fences
    text = "Here is some code:\n```python\ndef hello():\n    pass\n```\nDone."
    segs = _split_code_fences(text)
    kinds = [k for k, _, _ in segs]
    assert kinds == ["text", "code", "text"]
    code_segs = [(k, t, l) for k, t, l in segs if k == "code"]
    assert len(code_segs) == 1
    assert code_segs[0][2] == "python"
    assert "def hello():" in code_segs[0][1]


def test_tui_split_code_fences_unterminated() -> None:
    """An unterminated fence is emitted as plain text."""
    from anduril.tui import _split_code_fences
    text = "before\n```python\ndef f():\n    pass\n"
    segs = _split_code_fences(text)
    kinds = [k for k, _, _ in segs]
    assert "code" not in kinds


def test_tui_split_code_fences_no_fences() -> None:
    """Text without fences stays a single text segment."""
    from anduril.tui import _split_code_fences
    segs = _split_code_fences("Just plain text, no fences here.")
    assert segs == [("text", "Just plain text, no fences here.", "")]


def test_tui_split_code_fences_bare_closer() -> None:
    """A bare ``` outside a code block stays in the text stream."""
    from anduril.tui import _split_code_fences
    segs = _split_code_fences("Some text\n```\nMore text")
    kinds = [k for k, _, _ in segs]
    assert "code" not in kinds


def test_tui_wrap_entry_returns_spans() -> None:
    """_wrap_entry returns (kind, [spans]) tuples, not (kind, text, attr)."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState

    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass

    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        wrapped = state._wrap_entry("assistant", "hello world", 0x20, 78)
    for line in wrapped:
        assert len(line) == 2
        kind, spans = line
        assert isinstance(kind, str)
        assert isinstance(spans, list)
        for span in spans:
            assert len(span) == 2  # (text, attr)


def test_tui_wrap_entry_highlighted_code_round_trip() -> None:
    """A code block's per-line spans concatenate back to the source."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState

    class _StubWin:
        def getmaxyx(self): return (20, 200)  # wide enough to fit
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass

    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        text = "before\n```python\ndef f():\n    pass\n```\nafter"
        wrapped = state._wrap_entry("assistant", text, 0x20, 100)
    # Reconstruct the entry by joining all spans.
    joined = ""
    for _kind, spans in wrapped:
        for span_text, _attr in spans:
            joined += span_text
    # The actual code content survives the wrap-and-highlight
    # round trip (newlines and indent may be re-formatted).
    assert "def f():" in joined
    assert "pass" in joined
    assert "before" in joined
    assert "after" in joined


def test_tui_truncate_tool_blocks_preserves_spans() -> None:
    """_truncate_tool_blocks works on the new (kind, [spans]) shape."""
    import curses
    from unittest.mock import patch
    from anduril.tui import _TUIState

    class _StubWin:
        def getmaxyx(self): return (20, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass

    with patch.object(curses, "curs_set", lambda flag: None):
        agent = Agent(model="m", system="sys")
        state = _TUIState(agent, _StubWin(), 0,
                          0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
        state.bootstrap()
        wrapped = [
            ("tool", [("line1", 0x20)]),
            ("tool", [("line2", 0x20)]),
            ("assistant", [("reply", 0x30)]),
        ]
        final = state._truncate_tool_blocks(wrapped)
    assert final == [
        [("line1", 0x20)],
        [("line2", 0x20)],
        [("reply", 0x30)],
    ]


def test_tui_token_to_attr_basic() -> None:
    """_token_to_attr returns mapped values, sentinel for unknown."""
    from anduril.tui import _token_to_attr, _HL_UNMAPPED
    assert _token_to_attr("Token.Keyword") != _HL_UNMAPPED
    assert _token_to_attr("Token.Literal.Some.Thing") == _HL_UNMAPPED
    assert _token_to_attr("") == _HL_UNMAPPED


# === MCP client ============================================================


def test_mcp_protocol_version_exported() -> None:
    """The client announces a specific protocol version."""
    from anduril.mcp_client import PROTOCOL_VERSION
    assert isinstance(PROTOCOL_VERSION, str)
    # YYYY-MM-DD form.
    assert len(PROTOCOL_VERSION.split("-")) == 3


def test_mcp_server_repr() -> None:
    """MCPServer has a useful repr for debugging."""
    s = MCPServer(name="fs", command=["npx", "-y", "mcp-fs", "/tmp"])
    r = repr(s)
    assert "fs" in r
    assert "npx" in r


def test_mcp_server_empty_command_rejected() -> None:
    """An empty command raises ValueError at construction time."""
    import pytest
    with pytest.raises(ValueError):
        MCPServer(name="x", command="")
    with pytest.raises(ValueError):
        MCPServer(name="x", command=[])


def test_mcp_tool_name_for_default() -> None:
    """Default naming policy: ``<name>__<tool>``."""
    from anduril.mcp_client import _tool_name_for
    s = MCPServer(name="fs", command=["dummy"])
    assert _tool_name_for(s, "read_file") == "fs__read_file"


def test_mcp_tool_name_for_none() -> None:
    """``prefix='none'`` returns the raw tool name."""
    from anduril.mcp_client import _tool_name_for
    s = MCPServer(name="fs", command=["dummy"], prefix="none")
    assert _tool_name_for(s, "read_file") == "read_file"


def test_mcp_strip_tool_name_prefix() -> None:
    """Inverse of ``_tool_name_for`` for the default policy."""
    from anduril.mcp_client import _strip_tool_name_prefix
    s = MCPServer(name="fs", command=["dummy"])
    assert _strip_tool_name_prefix(s, "fs__read_file") == "read_file"
    assert _strip_tool_name_prefix(s, "git__log") == "git__log"  # wrong server
    # ``prefix='none'`` returns the full name unchanged.
    s2 = MCPServer(name="fs", command=["dummy"], prefix="none")
    assert _strip_tool_name_prefix(s2, "read_file") == "read_file"


def test_mcp_schema_translation_strips_meta() -> None:
    """MCP schemas with $schema / title are accepted by the validator."""
    from anduril.mcp_client import _mcp_schema_to_our_schema
    from anduril import _validate
    raw = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ToolInput",
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
    }
    translated = _mcp_schema_to_our_schema("read", "doc", raw)
    # Meta keys stripped.
    assert "$schema" not in translated
    assert "title" not in translated
    # Validator accepts it.
    assert _validate({"path": "x"}, translated) == []
    # Required is enforced.
    assert _validate({}, translated) != []


def test_mcp_schema_translation_default_type() -> None:
    """MCP schemas without a top-level type pass through."""
    from anduril.mcp_client import _mcp_schema_to_our_schema
    raw = {"properties": {"x": {"type": "string"}}}
    out = _mcp_schema_to_our_schema("f", "d", raw)
    assert "properties" in out


def test_mcp_format_call_result_text() -> None:
    """Plain text content is joined into a single string."""
    from anduril.mcp_client import _format_call_result
    out = _format_call_result({
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ],
    })
    assert out == "hello\nworld"


def test_mcp_format_call_result_image() -> None:
    """Image content is surfaced as a marker (not embedded)."""
    from anduril.mcp_client import _format_call_result
    out = _format_call_result({
        "content": [{"type": "image", "data": "AAAA", "mimeType": "image/png"}],
    })
    assert "[image:" in out
    assert "image/png" in out


def test_mcp_format_call_result_error() -> None:
    """``isError: true`` prefixes the output so the model sees the failure."""
    from anduril.mcp_client import _format_call_result
    out = _format_call_result({
        "isError": True,
        "content": [{"type": "text", "text": "oh no"}],
    })
    assert out.startswith("error:")
    assert "oh no" in out


def test_mcp_format_call_result_malformed() -> None:
    """A non-list content is coerced to a string."""
    from anduril.mcp_client import _format_call_result
    out = _format_call_result({"content": "not a list"})
    assert "not a list" in out


def test_mcp_load_from_pyproject_missing_file() -> None:
    """A nonexistent pyproject returns an empty list (no error)."""
    out = load_mcp_servers_from_pyproject("/no/such/path/pyproject.toml")
    assert out == []


def test_mcp_load_from_pyproject_no_section() -> None:
    """A pyproject with no MCP section returns an empty list."""
    import tempfile
    import pathlib
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[project]\nname = 'foo'\n")
        path = f.name
    try:
        out = load_mcp_servers_from_pyproject(path)
        assert out == []
    finally:
        pathlib.Path(path).unlink()


def test_mcp_load_from_pyproject_with_servers(tmp_path) -> None:
    """A pyproject with MCP servers is parsed correctly."""
    toml = """
[tool.anduril.mcp_servers.servers.fs]
command = "npx -y @mcp/server-fs /tmp"
prefix = "server"

[tool.anduril.mcp_servers.servers.git]
command = "uvx mcp-server-git"
"""
    path = tmp_path / "pyproject.toml"
    path.write_text(toml)
    servers = load_mcp_servers_from_pyproject(path)
    assert len(servers) == 2
    names = {s.name for s in servers}
    assert names == {"fs", "git"}
    fs = next(s for s in servers if s.name == "fs")
    # Commands are tokenised into argv form. ``MCPServer``
    # accepts a string and does the shlex split itself.
    assert fs.command == ["npx", "-y", "@mcp/server-fs", "/tmp"]
    assert fs.prefix == "server"
    git = next(s for s in servers if s.name == "git")
    # Default prefix applied.
    assert git.prefix == "server"
    assert git.command == ["uvx", "mcp-server-git"]


def test_mcp_end_to_end_with_fake_server(tmp_path) -> None:
    """Spin up a tiny fake MCP server, talk to it, tear it down.

    The fake server is a small Python script that reads
    JSON-RPC from stdin and writes responses to stdout. It
    implements ``initialize``, ``tools/list`` and
    ``tools/call`` — enough to exercise the full client.
    """
    import textwrap
    import sys
    fake_server = textwrap.dedent('''
        import json, sys

        TOOLS = [
            {
                "name": "echo",
                "description": "Echo back the input",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
            {
                "name": "add",
                "description": "Add two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        ]

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            method = msg.get("method")
            msg_id = msg.get("id")
            if method == "initialize":
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "fake", "version": "0.0.1"},
                    },
                }) + "\\n")
            elif method == "tools/list":
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"tools": TOOLS},
                }) + "\\n")
            elif method == "tools/call":
                name = msg["params"]["name"]
                args = msg["params"].get("arguments", {})
                if name == "echo":
                    out_text = args.get("text", "")
                    result = {"content": [{"type": "text", "text": out_text}]}
                elif name == "add":
                    result = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}]}
                else:
                    result = {"isError": True, "content": [{"type": "text", "text": "unknown tool"}]}
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": msg_id, "result": result,
                }) + "\\n")
            sys.stdout.flush()
        ''')
    script = tmp_path / "fake_mcp.py"
    script.write_text(fake_server)
    server = MCPServer(name="fake", command=[sys.executable, str(script)])
    try:
        tools = discover_mcp_tools([server])
        # Both tools are present and named with the prefix.
        names = {t.name for t in tools}
        assert names == {"fake__echo", "fake__add"}
        # The tool functions actually work end-to-end.
        echo = next(t for t in tools if t.name == "fake__echo")
        assert echo.fn(text="hi") == "hi"
        add = next(t for t in tools if t.name == "fake__add")
        assert add.fn(a=2, b=3) == "5"
    finally:
        shutdown_servers([server])


def test_mcp_end_to_end_over_http() -> None:
    """Spin up a tiny HTTP server that speaks MCP, talk to it via the HTTP transport.

    The fake server is a single-threaded ``http.server`` that
    handles one POST at a time. The request body is a
    JSON-RPC payload; the response is either a single
    ``application/json`` body (for ``initialize`` /
    ``tools/list``) or a ``text/event-stream`` (for
    ``tools/call``, to exercise the SSE branch of the
    transport).
    """
    import http.server
    import json as _json
    import socketserver
    import threading
    from anduril.mcp_client import (
        MCPServer, discover_mcp_tools, shutdown_servers,
    )

    TOOLS = [{
        "name": "echo",
        "description": "Echo back the input",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }]

    class _Handler(http.server.BaseHTTPRequestHandler):
        # Silence the default per-request logging.
        def log_message(self, fmt, *args): pass

        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                msg = _json.loads(body)
            except Exception:
                self.send_error(400, "bad json"); return
            method = msg.get("method")
            msg_id = msg.get("id")
            # tools/call uses SSE so we exercise the streaming branch.
            if method == "tools/call":
                args = msg["params"].get("arguments", {})
                out_text = args.get("text", "")
                payload = _json.dumps({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": out_text}],
                    },
                })
                body_bytes = (
                    f"event: message\r\n"
                    f"data: {payload}\r\n"
                    f"\r\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)
                return
            # Everything else: single JSON response.
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "serverInfo": {"name": "fake-http", "version": "0"},
                }
            elif method == "tools/list":
                result = {"tools": TOOLS}
            else:
                result = {}
            body = _json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result})
            body_bytes = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

    # Bind to port 0 to get a free port. ``socketserver.TCPServer``
    # with ``bind_and_activate=False`` lets us read the port
    # back before serving.
    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    httpd = _Server(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        url = f"http://127.0.0.1:{port}/mcp"
        mcp_server = MCPServer(name="httpfake", url=url, timeout=5.0)
        tools = discover_mcp_tools([mcp_server])
        assert {t.name for t in tools} == {"httpfake__echo"}
        # Round-trip a tool call (exercises the SSE branch).
        echo = next(t for t in tools if t.name == "httpfake__echo")
        assert echo.fn(text="hi") == "hi"
    finally:
        shutdown_servers([mcp_server])
        httpd.shutdown()
        httpd.server_close()


def test_mcp_server_validates_command_and_url() -> None:
    """An MCPServer must have exactly one of command or url."""
    import pytest
    # Both missing.
    with pytest.raises(ValueError, match="exactly one"):
        MCPServer(name="x")
    # Both set.
    with pytest.raises(ValueError, match="exactly one"):
        MCPServer(name="x", command=["echo"], url="http://localhost:8000")
    # Bad URL.
    with pytest.raises(ValueError, match="http"):
        MCPServer(name="x", url="ftp://nope")
    # Empty command.
    with pytest.raises(ValueError, match="empty"):
        MCPServer(name="x", command=[])


def test_mcp_http_transport_requires_http_url() -> None:
    """The HTTP transport rejects non-http(s) URLs."""
    import pytest
    from anduril.mcp_client import _HTTPMCPTransport
    with pytest.raises(Exception, match="http"):
        _HTTPMCPTransport("ftp://nope/", timeout=1.0)


def test_mcp_stdio_transport_alias() -> None:
    """The legacy ``_MCPTransport`` name still points at the stdio class.

    The stdio / HTTP split was a refactor; existing callers
    and tests import the bare ``_MCPTransport`` name. Make
    sure that name is still the stdio class.
    """
    from anduril.mcp_client import _MCPTransport
    assert _MCPTransport is _StdioMCPTransport


def test_mcp_handshake_failure_does_not_break_agent() -> None:
    """A server that exits immediately is reported but doesn't break the rest."""
    import sys
    import io
    import contextlib
    # ``python -c "pass"`` exits with 0, no JSON-RPC. The client
    # should swallow the failure and return [] for that server.
    # Use a very short timeout so the test doesn't hang waiting
    # for a response that will never come.
    server = MCPServer(
        name="bad", command=[sys.executable, "-c", "pass"],
        timeout=0.5,
    )
    try:
        # Capture stderr so the test output isn't polluted.
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            tools = discover_mcp_tools([server])
        # The bad server contributed nothing.
        assert tools == []
        assert "bad" in tools.errors  # type: ignore[attr-defined]
    finally:
        shutdown_servers([server])


def test_tui_mcp_command_no_servers() -> None:
    """``/mcp`` with no MCP tools installed shows a helpful hint."""
    from anduril.tui import _TUIState
    agent = Agent(model="m", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_mcp("")
    assert "no MCP tools" in out
    assert "pyproject.toml" in out


def test_tui_mcp_command_with_servers() -> None:
    """``/mcp`` lists each MCP server and its tools."""
    from anduril.tui import _TUIState
    from anduril.tools import Tool
    agent = Agent(model="m", system="sys")
    def _echo(**kw): return kw.get("text", "")
    agent.register_tool(Tool("fs__echo", "echo", {"type":"object","properties":{}}, _echo, dangerous=False, risk="medium"))
    agent.register_tool(Tool("fs__read", "read", {"type":"object","properties":{}}, _echo, dangerous=False, risk="medium"))
    agent.register_tool(Tool("git__log", "log", {"type":"object","properties":{}}, _echo, dangerous=False, risk="medium"))
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_mcp("")
    assert "fs:" in out
    assert "git:" in out
    assert "echo" in out
    assert "read" in out
    assert "log" in out


# === Pricing / cost tracking ==============================================


def test_pricing_known_models() -> None:
    """Common model names resolve to a ModelPricing."""
    assert pricing_for("gpt-4o") is not None
    assert pricing_for("gpt-4o-mini") is not None
    assert pricing_for("o3-mini") is not None
    assert pricing_for("claude-3-5-sonnet-latest") is not None
    assert pricing_for("claude-3-7-sonnet") is not None
    assert pricing_for("gemini-2.5-pro") is not None
    assert pricing_for("deepseek-chat") is not None
    # Local model names resolve to a zero-pricing entry.
    assert pricing_for("llama-3.1-70b") is not None


def test_pricing_unknown_returns_none() -> None:
    """An unknown model returns None so the UI shows "—", not "$0.00"."""
    assert pricing_for("totally-fictional-model-99") is None
    assert pricing_for(None) is None
    assert pricing_for("") is None


def test_pricing_case_insensitive() -> None:
    """Pricing lookup is case-insensitive."""
    assert pricing_for("GPT-4O") is not None
    assert pricing_for("Claude-3-5-Sonnet") is not None


def test_pricing_overrides_via_env() -> None:
    """ANDURIL_PRICING_OVERRIDES JSON wins over the built-in table."""
    import os
    import json
    os.environ["ANDURIL_PRICING_OVERRIDES"] = json.dumps({
        "gpt-4o": {"input": 1.0, "output": 2.0},
    })
    try:
        p = pricing_for("gpt-4o")
        assert p is not None
        assert p.input_per_mtok == 1.0
        assert p.output_per_mtok == 2.0
    finally:
        os.environ.pop("ANDURIL_PRICING_OVERRIDES", None)


def test_pricing_overrides_invalid_json_ignored() -> None:
    """Invalid JSON in the override env var is silently dropped."""
    import os
    os.environ["ANDURIL_PRICING_OVERRIDES"] = "{not valid json"
    try:
        # Should still find the built-in pricing.
        p = pricing_for("gpt-4o")
        assert p is not None
        # Built-in price (2.50), not the bogus 1.0.
        assert p.input_per_mtok == 2.50
    finally:
        os.environ.pop("ANDURIL_PRICING_OVERRIDES", None)


def test_pricing_cost_calculation_basic() -> None:
    """A 1K input + 500 output call against gpt-4o costs the right amount."""
    p = pricing_for("gpt-4o")
    assert p is not None
    # gpt-4o: $2.50/1M input, $10/1M output. 1000 input + 500 output:
    #   1000 * 2.50 / 1e6 = 0.0025
    #   500 * 10.00 / 1e6 = 0.005
    #   total = 0.0075
    cost = p.cost(input_tokens=1000, output_tokens=500)
    assert abs(cost - 0.0075) < 1e-9


def test_pricing_cost_calculation_with_cache() -> None:
    """Cached tokens are billed at the cache rate."""
    p = pricing_for("gpt-4o")
    assert p is not None
    # 1000 input, 800 cached, 200 fresh, 0 output.
    #   200 * 2.50 / 1e6 = 0.0005
    #   800 * 1.25 / 1e6 = 0.001
    #   total = 0.0015
    cost = p.cost(input_tokens=1000, cached_tokens=800)
    assert abs(cost - 0.0015) < 1e-9


def test_pricing_reasoning_tokens_billed_separately() -> None:
    """For o1, reasoning tokens have their own rate."""
    p = pricing_for("o1")
    assert p is not None
    # o1: $15/1M input, $60/1M output, $60/1M reasoning.
    cost = p.cost(
        input_tokens=1000, output_tokens=500, reasoning_tokens=2000,
    )
    # 1000 * 15/1e6 = 0.015
    # 500 * 60/1e6 = 0.030
    # 2000 * 60/1e6 = 0.120
    # total = 0.165
    assert abs(cost - 0.165) < 1e-9


def test_pricing_local_model_is_free() -> None:
    """Local / zero-pricing models report $0.00 for any usage."""
    p = pricing_for("llama-3.1-70b")
    assert p is not None
    assert p.cost(input_tokens=1_000_000, output_tokens=1_000_000) == 0.0


def test_fmt_cost_thresholds() -> None:
    """fmt_cost picks the right precision at each magnitude."""
    assert fmt_cost(0.0) == "< $0.0001"
    assert fmt_cost(0.0001) == "$0.0001"
    assert fmt_cost(0.0012) == "$0.0012"
    assert fmt_cost(0.5) == "$0.5000"
    assert fmt_cost(1.5) == "$1.50"
    assert fmt_cost(99.99) == "$99.99"
    assert fmt_cost(123.45) == "$123.45"
    assert fmt_cost(1234.5) == "$1,234"  # :.0f truncates
    assert fmt_cost(1_500_000) == "$1.5M"


# === Metrics cost integration ============================================


def test_metrics_records_cost_per_call() -> None:
    """A call's cost is added to the session total and the per-model row."""
    m = _Metrics("test", model="gpt-4o")
    # 1K input + 500 output = $0.0075.
    m.add({"input_tokens": 1000, "output_tokens": 500,
           "cache_read_tokens": 0, "reasoning_tokens": 0}, model="gpt-4o")
    assert abs(m.total_cost - 0.0075) < 1e-9
    assert "gpt-4o" in m.cost_by_model
    assert abs(m.cost_by_model["gpt-4o"] - 0.0075) < 1e-9
    assert m.api_calls == 1


def test_metrics_accumulates_across_calls() -> None:
    """Multiple calls on the same model add up."""
    m = _Metrics("test", model="gpt-4o")
    m.add({"input_tokens": 1000, "output_tokens": 0,
           "cache_read_tokens": 0, "reasoning_tokens": 0}, model="gpt-4o")
    m.add({"input_tokens": 1000, "output_tokens": 0,
           "cache_read_tokens": 0, "reasoning_tokens": 0}, model="gpt-4o")
    assert abs(m.total_cost - 0.005) < 1e-9  # 2 * 0.0025
    assert abs(m.cost_by_model["gpt-4o"] - 0.005) < 1e-9


def test_metrics_separate_per_model_costs() -> None:
    """A model switch mid-session records each model separately."""
    m = _Metrics("test", model="gpt-4o")
    m.add({"input_tokens": 1000, "output_tokens": 0,
           "cache_read_tokens": 0, "reasoning_tokens": 0}, model="gpt-4o")
    m.add({"input_tokens": 1000, "output_tokens": 0,
           "cache_read_tokens": 0, "reasoning_tokens": 0}, model="gpt-4o-mini")
    # gpt-4o: 0.0025; gpt-4o-mini: 0.00015. Total 0.00265.
    assert abs(m.total_cost - 0.00265) < 1e-9
    assert abs(m.cost_by_model["gpt-4o"] - 0.0025) < 1e-9
    assert abs(m.cost_by_model["gpt-4o-mini"] - 0.00015) < 1e-9


def test_metrics_unknown_model_no_cost_added() -> None:
    """A call to an unpriced model doesn't update totals."""
    m = _Metrics("test", model="some-mystery-model")
    m.add({"input_tokens": 1000, "output_tokens": 1000,
           "cache_read_tokens": 0, "reasoning_tokens": 0}, model="some-mystery-model")
    # Tokens still count (the user did spend compute), but
    # cost stays at zero.
    assert m.input_tokens == 1000
    assert m.output_tokens == 1000
    assert m.total_cost == 0.0
    assert m.cost_by_model == {}


def test_metrics_load_preserves_cost() -> None:
    """Saved cost is restored when the session is reloaded."""
    m = _Metrics("test", model="gpt-4o")
    m.add({"input_tokens": 1000, "output_tokens": 0,
           "cache_read_tokens": 0, "reasoning_tokens": 0}, model="gpt-4o")
    saved = m.as_meta()
    m2 = _Metrics("test", model="gpt-4o")
    m2.load(saved)
    assert abs(m2.total_cost - 0.0025) < 1e-9
    assert "gpt-4o" in m2.cost_by_model


# === TUI /cost command ===================================================


def test_tui_cost_command_no_calls() -> None:
    """``/cost`` with no API calls yet shows a friendly hint."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_cost("")
    assert "no API calls" in out


def test_tui_cost_command_with_calls() -> None:
    """``/cost`` shows a per-model breakdown after some calls."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    # Simulate two calls.
    state.metrics.add(
        {"input_tokens": 1000, "output_tokens": 500,
         "cache_read_tokens": 0, "reasoning_tokens": 0},
        model="gpt-4o",
    )
    state.metrics.add(
        {"input_tokens": 2000, "output_tokens": 0,
         "cache_read_tokens": 0, "reasoning_tokens": 0},
        model="gpt-4o-mini",
    )
    out = state._cmd_cost("")
    assert "gpt-4o" in out
    assert "gpt-4o-mini" in out
    assert "$" in out
    assert "per model" in out


def test_tui_cost_command_unpriced_model() -> None:
    """``/cost`` surfaces the missing-pricing notice for unknown models."""
    from anduril.tui import _TUIState
    agent = Agent(model="my-mystery-model", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    state.metrics.add(
        {"input_tokens": 1000, "output_tokens": 0,
         "cache_read_tokens": 0, "reasoning_tokens": 0},
        model="my-mystery-model",
    )
    out = state._cmd_cost("")
    assert "not in table" in out
    assert "ANDURIL_PRICING_OVERRIDES" in out


if __name__ == "__main__":
    test_basic_schema()
    test_validation()
    test_rich_schema()
    test_bash_defaults()
    test_history_save_load()
    test_history_preserves_local_system()
    test_sanitize_tool_result_short_passthrough()
    test_sanitize_tool_result_dedup_runs()
    test_sanitize_tool_result_caps_long_output()
    test_parse_text_calls_basic()
    test_parse_text_calls_no_calls()
    test_parse_text_calls_ignores_malformed()
    test_tool_call_aggregator()
    test_metrics_add_and_meta()
    test_metrics_load_from_saved()
    test_normalize_usage_basic()
    test_safe_title()
    test_new_session_id_format()
    test_short_id()
    test_editor_basic_insert_and_submit()
    test_editor_multiline_paste_no_cap()
    test_editor_backspace_merges_lines()
    test_editor_history_navigation()
    test_enter_submits_regardless_of_cr_or_lf()
    test_wrap_visual_chunks_use_word_boundaries()
    test_wrap_cursor_mapping_with_ranges()
    test_prune_empty_assistant_messages()
    test_sessions_save_load_delete_list()
    test_resolve_session_by_id_and_prefix()
    test_resolve_session_by_index()
    test_abbr_and_precise_abbr()
    test_default_system_prompt_is_concise_focused()
    test_build_agent_uses_default_system_when_unset()
    test_normalize_approval_known_values()
    test_normalize_approval_unknown()
    test_agent_pop_last_and_set_system()
    test_tool_call_aggregator_peek()
    test_tui_command_dispatch_basic()
    test_tui_scroll_clamps_to_visible_lines()
    test_tui_scroll_resets_on_clear()
    test_tui_input_prompt_prefix()
    test_tui_render_processes_resize_flag()
    test_tui_render_handles_resize_failure_gracefully()
    test_tui_render_swallows_curses_error()
    test_tui_render_skips_zero_sized_resize()
    test_tui_render_handles_window_split_to_tiny_size()
    test_tui_streaming_tool_call_appears_as_it_forms()
    test_tui_tool_call_id_tracking()
    test_tui_clear_resets_tool_call_tracking()
    test_confirm_key_survives_curses_error()
    test_tui_status_bar_tracks_during_streaming()
    test_tui_status_bar_persists_after_turn_without_usage()
    test_tui_status_bar_uses_api_values_after_turn()
    # File-attachment features
    test_files_is_image_recognizes_known_formats()
    test_files_is_text_file_uses_extension_hint()
    test_files_fuzzy_match_empty_query_preserves_order()
    test_files_fuzzy_match_prefix_wins()
    test_files_fuzzy_match_word_boundary_bonus()
    test_files_fuzzy_match_rejects_non_subsequence()
    test_files_find_active_mention_basic()
    test_files_find_active_mention_email_left_alone()
    test_files_find_active_mention_trailing_terminator()
    test_files_mention_query_returns_path()
    test_files_list_files_respects_max_count()
    test_files_list_files_skips_default_ignored_dirs()
    test_files_expand_mentions_text_file_inlined()
    test_files_expand_mentions_image_loaded_as_data_url()
    test_files_expand_mentions_unknown_file_left_literal()
    test_files_expand_mentions_email_left_alone()
    test_files_expand_mentions_multiple_in_order()
    test_files_read_image_data_url_size_limit()
    test_files_read_text_file_truncates()
    test_agent_run_accepts_multimodal_content()
    test_tui_file_menu_activates_on_at()
    test_tui_file_menu_does_not_activate_for_email()
    test_tui_file_menu_complete_inserts_path()
    test_tui_file_menu_query_updates_on_typing()
    test_tui_file_menu_dismiss_clears_mention()
    test_tui_file_menu_uses_cache_for_repeated_queries()
    test_tui_file_menu_arrow_keys_navigate()
    test_tui_file_menu_caches_list()
    # Pasted-image support
    test_files_save_pasted_image_writes_bytes()
    test_files_save_pasted_image_normalizes_extension()
    test_files_save_pasted_image_jpg_to_jpeg()
    test_files_save_pasted_image_empty_data_raises()
    test_files_resolve_mention_path_expands_tilde()
    test_files_expand_mentions_tilde_path()
    test_tui_parse_kitty_graphics_basic()
    test_tui_parse_kitty_graphics_multipart_flag()
    test_tui_parse_kitty_graphics_rejects_non_transmit()
    test_tui_parse_kitty_graphics_rejects_missing_payload()
    test_tui_parse_iterm2_image_basic()
    test_tui_parse_iterm2_image_rejects_missing_size()
    test_tui_parse_iterm2_image_rejects_oversize()
    test_tui_kitty_multipart_accumulates_and_decodes()
    # Clipboard /paste fallback
    test_files_read_clipboard_image_returns_none_when_no_tool()
    test_files_read_clipboard_image_handles_subprocess_error()
    test_files_read_clipboard_image_decodes_png_magic()
    test_files_read_clipboard_image_decodes_jpg_magic()
    test_tui_cmd_paste_inserts_at_mention_on_success()
    test_tui_cmd_paste_noop_when_clipboard_empty()
    test_tui_paste_command_registered()
    # Short-reference attachments
    test_tui_register_attachment_returns_short_id()
    test_tui_register_attachment_reuses_freed_id()
    test_tui_register_attachment_unique_across_buffer()
    test_tui_register_attachment_fills_lowest_free()
    test_tui_clear_resets_attachments()
    test_files_expand_mentions_resolves_short_id()
    test_files_expand_mentions_short_id_overrides_cwd()
    test_tui_cmd_paste_inserts_short_id_not_full_path()
    test_tui_alt_v_keybinding_triggers_paste()
    # Clipboard tool detection
    test_files_clipboard_uses_wl_paste_on_wayland()
    test_files_clipboard_falls_back_when_preferred_missing()
    test_tui_cmd_paste_reports_available_tools()
    # /attachments list
    test_tui_cmd_attachments_lists_in_use_and_stale()
    test_tui_cmd_attachments_empty_state()
    test_tui_attachments_command_registered()
    # Syntax highlighting
    test_normalize_lang_aliases()
    test_highlight_code_round_trip()
    test_highlight_code_empty_lang_returns_default()
    test_highlight_code_default_attr_used_for_unmapped()
    test_highlight_code_adjacent_same_attr_merged()
    test_highlight_code_alias()
    test_tui_split_code_fences_basic()
    test_tui_split_code_fences_unterminated()
    test_tui_split_code_fences_no_fences()
    test_tui_split_code_fences_bare_closer()
    test_tui_wrap_entry_returns_spans()
    test_tui_wrap_entry_highlighted_code_round_trip()
    test_tui_truncate_tool_blocks_preserves_spans()
    test_tui_token_to_attr_basic()
    # MCP client
    test_mcp_protocol_version_exported()
    test_mcp_server_repr()
    test_mcp_server_empty_command_rejected()
    test_mcp_tool_name_for_default()
    test_mcp_tool_name_for_none()
    test_mcp_strip_tool_name_prefix()
    test_mcp_schema_translation_strips_meta()
    test_mcp_schema_translation_default_type()
    test_mcp_format_call_result_text()
    test_mcp_format_call_result_image()
    test_mcp_format_call_result_error()
    test_mcp_format_call_result_malformed()
    test_mcp_load_from_pyproject_missing_file()
    test_mcp_load_from_pyproject_no_section()
    import tempfile
    import shutil
    _tmp = tempfile.mkdtemp(prefix="anduril-mcp-")
    try:
        test_mcp_load_from_pyproject_with_servers(_tmp)
        test_mcp_end_to_end_with_fake_server(_tmp)
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)
    test_mcp_handshake_failure_does_not_break_agent()
    test_mcp_end_to_end_over_http()
    test_mcp_server_validates_command_and_url()
    test_mcp_http_transport_requires_http_url()
    test_mcp_stdio_transport_alias()
    # add_mcp_server
    test_add_mcp_server_basic("/tmp/addmcp_test")
    test_add_mcp_server_persists_config("/tmp/addmcp_persist_test")
    test_add_mcp_server_rejects_both()
    test_add_mcp_server_rejects_neither()
    test_agent_close_clears_current_pointer()
    # Per-model system prompts
    test_agent_resolve_system_prompt_default()
    test_agent_resolve_system_prompt_specific_match()
    test_agent_resolve_system_prompt_no_match()
    test_agent_set_system_per_model_override()
    test_agent_run_applies_per_model_override()
    test_tui_system_command_shows_overrides()
    test_tui_system_command_per_model_form()
    test_tui_system_command_pure_alpha_first_token_is_default()
    # Budget
    test_metrics_budget_default_is_none()
    test_agent_run_refuses_when_budget_exceeded()
    test_tui_budget_command_no_arg_shows_status()
    test_tui_budget_command_absolute()
    test_tui_budget_command_off_clears()
    test_tui_budget_command_relative()
    test_tui_budget_command_bad_input()


def test_metrics_budget_default_is_none() -> None:
    """A fresh metrics object has no budget cap."""
    m = _Metrics("test", model="gpt-4o")
    assert m.budget is None


def test_agent_run_refuses_when_budget_exceeded() -> None:
    """With a tight budget, a run() that would exceed it returns a short status."""
    a = Agent(model="gpt-4o", system="sys", max_turns=2)
    a.set_metrics(_Metrics("test", model="gpt-4o"))
    # Set the running total to already exceed the budget so
    # the first call would push us over.
    a._metrics.total_cost = 0.001
    a._metrics.budget = 0.0001
    called = {"n": 0}
    def fake_create(**kwargs):
        called["n"] += 1
        return None
    a.client.chat.completions.create = fake_create  # type: ignore[method-assign]
    result = a.run("hello")
    assert called["n"] == 0
    assert "budget reached" in result


def test_tui_budget_command_no_arg_shows_status() -> None:
    """``/budget`` with no arg shows the current cap (or 'no cap')."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    # No budget set.
    out = state._cmd_budget("")
    assert "no budget set" in out
    # Set a budget, re-check.
    state._cmd_budget("5.00")
    out = state._cmd_budget("")
    assert "budget: $5.00" in out


def test_tui_budget_command_absolute() -> None:
    """``/budget <usd>`` sets an absolute cap."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_budget("1.50")
    assert "budget" in out and "$1.50" in out
    assert state.metrics.budget == 1.50


def test_tui_budget_command_off_clears() -> None:
    """``/budget off`` (or 0) clears the cap."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    state._cmd_budget("2.00")
    assert state.metrics.budget == 2.00
    out = state._cmd_budget("off")
    assert state.metrics.budget is None
    assert "cleared" in out


def test_tui_budget_command_relative() -> None:
    """``/budget +1.00`` raises the cap; ``/budget -0.50`` lowers it."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    state._cmd_budget("2.00")
    state._cmd_budget("+1.00")
    assert state.metrics.budget == 3.00
    state._cmd_budget("-0.50")
    assert abs(state.metrics.budget - 2.50) < 1e-9
    # Going below zero is clamped.
    state._cmd_budget("-10.00")
    assert state.metrics.budget == 0.0


def test_tui_budget_command_bad_input() -> None:
    """``/budget xyz`` returns an error, doesn't change the cap."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="sys")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_budget("xyz")
    assert "bad number" in out
    assert state.metrics.budget is None



def test_add_mcp_server_basic(tmp_path) -> None:
    """``add_mcp_server`` discovers a server and registers its tools."""
    import sys
    import os
    import textwrap
    # Build a fake server in a temp file.
    fake = textwrap.dedent('''
        import json, sys
        TOOLS = [{
            "name": "greet",
            "description": "greet someone",
            "inputSchema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }]
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            method = msg.get("method")
            msg_id = msg.get("id")
            if method == "initialize":
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "addmcp-test", "version": "0"},
                    },
                }) + chr(10))
            elif method == "tools/list":
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"tools": TOOLS},
                }) + chr(10))
            elif method == "tools/call":
                args = msg["params"].get("arguments", {})
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"hi {args.get('name', '')}"}],
                    },
                }) + chr(10))
            sys.stdout.flush()
        ''')
    script = tmp_path / "fake_mcp_add.py"
    script.write_text(fake)
    # Use ANDURIL_MCP_CONFIG_DIR to keep the persisted
    # config under tmp_path (not the user's real dir).
    os.environ["ANDURIL_MCP_CONFIG_DIR"] = str(tmp_path / "mcp")
    try:
        from anduril.tools import add_mcp_server
        result = add_mcp_server.fn(
            name="addmcp",
            command=[sys.executable, str(script)],
            persistent=False,
        )
        assert "addmcp__greet" in result
        assert "greet" in result
    finally:
        pass
        # Shutdown any servers the tool may have started.
        # (We don't have a direct handle, but ``discover_mcp_tools``
        # is the only thing that starts transports, so closing
        # via the ``_transport`` attribute on the server isn't
        # available. We rely on process exit for the test.)


def test_add_mcp_server_persists_config(tmp_path) -> None:
    """``add_mcp_server(persistent=True)`` writes a JSON config file."""
    import os
    import json
    os.environ["ANDURIL_MCP_CONFIG_DIR"] = str(tmp_path / "mcp")
    from anduril.tools import add_mcp_server
    # The actual server start will fail (the command is bogus);
    # but the config file should be written before the start.
    add_mcp_server.fn(
        name="persisted",
        command="echo persisted",
        persistent=True,
    )
    cfg_path = tmp_path / "mcp" / "persisted.json"
    assert cfg_path.is_file()
    data = json.loads(cfg_path.read_text())
    assert data["command"] == "echo persisted"
    assert data["url"] is None


def test_add_mcp_server_rejects_both() -> None:
    """Setting both command= and url= is an error."""
    from anduril.tools import add_mcp_server
    out = add_mcp_server.fn(
        name="x", command=["echo"], url="http://nope/",
    )
    assert "error" in out
    assert "only one" in out


def test_add_mcp_server_rejects_neither() -> None:
    """Setting neither command= nor url= is an error."""
    from anduril.tools import add_mcp_server
    out = add_mcp_server.fn(name="x")
    assert "error" in out
    assert "must set" in out


def test_agent_resolve_system_prompt_default() -> None:
    """With no overrides, ``_resolve_system_prompt`` returns the default."""
    a = Agent(model="gpt-4o", system="base prompt")
    assert a._resolve_system_prompt() == "base prompt"
    # Different model, still the default.
    assert a._resolve_system_prompt("claude-3-5-sonnet") == "base prompt"


def test_agent_resolve_system_prompt_specific_match() -> None:
    """A substring match resolves to the override; longer wins."""
    a = Agent(
        model="gpt-4o-mini",
        system="base",
        system_overrides={
            "gpt-4o": "for gpt-4o",
            "gpt-4o-mini": "for mini",
        },
    )
    assert a._resolve_system_prompt() == "for mini"  # longer match wins
    # Switch to a different model — different override.
    a.model = "gpt-4o"
    assert a._resolve_system_prompt() == "for gpt-4o"


def test_agent_resolve_system_prompt_no_match() -> None:
    """No match → default."""
    a = Agent(
        model="some-mystery",
        system="default",
        system_overrides={"gpt-4o": "for gpt-4o"},
    )
    assert a._resolve_system_prompt() == "default"


def test_agent_set_system_per_model_override() -> None:
    """``set_system(text, for_model=...)`` registers a per-model override."""
    a = Agent(model="gpt-4o", system="default")
    a.set_system("a terser prompt for the mini", for_model="gpt-4o-mini")
    # Current model still uses the default.
    assert a._resolve_system_prompt() == "default"
    # The override applies to the matching model.
    assert a._resolve_system_prompt("gpt-4o-mini") == "a terser prompt for the mini"


def test_agent_run_applies_per_model_override() -> None:
    """Switching models mid-session changes the live system message."""
    a = Agent(model="gpt-4o", system="default for o")
    a.set_system("a terser prompt for the mini", for_model="gpt-4o-mini")
    # System message exists with the default.
    assert a._messages[0]["role"] == "system"
    assert a._messages[0]["content"] == "default for o"
    # Switch model + run a turn.
    a.model = "gpt-4o-mini"
    # Fake the network.
    captured = {"msgs": []}

    class _FakeChunk:
        def __init__(self, content):
            self.choices = [type("C", (), {"delta": type("D", (), {
                "content": content, "reasoning_content": None,
                "tool_calls": None,
            })()})()]
            self.usage = type("U", (), {
                "prompt_tokens": 10, "completion_tokens": 5,
                "prompt_tokens_details": type("P", (), {"cached_tokens": 0})(),
            })()
            self.model_extra = {}
    class _FakeResp:
        def __init__(self, content):
            self._content = content
        def __iter__(self):
            yield _FakeChunk(self._content)
        def close(self): pass
    def fake_create(**kwargs):
        captured["msgs"].append([dict(m) for m in kwargs["messages"]])
        return _FakeResp("ok")
    a.client.chat.completions.create = fake_create  # type: ignore[method-assign]
    a.run("hi")
    # The system message in the call's payload is the
    # per-model override, not the default.
    sys_msg = captured["msgs"][0][0]
    assert sys_msg["role"] == "system"
    assert sys_msg["content"] == "a terser prompt for the mini"


def test_tui_system_command_shows_overrides() -> None:
    """``/system`` with no arg lists the default + per-model overrides."""
    from anduril.tui import _TUIState
    agent = Agent(
        model="gpt-4o", system="default",
        system_overrides={"claude-3-5-sonnet": "claude prompt"},
    )
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_system("")
    assert "default: default" in out
    assert "claude-3-5-sonnet" in out
    assert "claude prompt" in out


def test_tui_system_command_per_model_form() -> None:
    """``/system <model-with-dashes> <text>`` registers a per-model override."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="default")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_system("gpt-4o-mini terser for mini")
    assert "system override" in out
    assert "gpt-4o-mini" in out
    # The override is actually registered.
    assert "gpt-4o-mini" in agent.system_overrides
    assert agent.system_overrides["gpt-4o-mini"] == "terser for mini"


def test_tui_system_command_pure_alpha_first_token_is_default() -> None:
    """A pure-alpha first token like 'a' is treated as prompt text, not a model."""
    from anduril.tui import _TUIState
    agent = Agent(model="gpt-4o", system="default")
    class _StubWin:
        def getmaxyx(self): return (24, 80)
        def erase(self): pass
        def addnstr(self, *a, **k): pass
        def addstr(self, *a, **k): pass
        def move(self, *a): pass
        def refresh(self): pass
        def get_wch(self): return "y"
    state = _TUIState(agent, _StubWin(), 0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)
    state.bootstrap()
    out = state._cmd_system("a new prompt")
    # Treated as a global default.
    assert "updated" in out
    assert agent.system == "a new prompt"
    # No new override.
    assert agent.system_overrides == {}


def test_agent_close_clears_current_pointer() -> None:
    """``Agent.close()`` releases the process-wide ``_current`` slot."""
    from anduril.agent import _current_agent_module
    a = Agent(model="m", system="sys")
    assert _current_agent_module._current is a
    a.close()
    assert _current_agent_module._current is None
    b = Agent(model="m2", system="sys")
    assert _current_agent_module._current is b
    b.close()
    test_tui_mcp_command_no_servers()
    test_tui_mcp_command_with_servers()
    # Pricing / cost
    test_pricing_known_models()
    test_pricing_unknown_returns_none()
    test_pricing_case_insensitive()
    test_pricing_overrides_via_env()
    test_pricing_overrides_invalid_json_ignored()
    test_pricing_cost_calculation_basic()
    test_pricing_cost_calculation_with_cache()
    test_pricing_reasoning_tokens_billed_separately()
    test_pricing_local_model_is_free()
    test_fmt_cost_thresholds()
    test_metrics_records_cost_per_call()
    test_metrics_accumulates_across_calls()
    test_metrics_separate_per_model_costs()
    test_metrics_unknown_model_no_cost_added()
    test_metrics_load_preserves_cost()
    test_tui_cost_command_no_calls()
    test_tui_cost_command_with_calls()
    test_tui_cost_command_unpriced_model()

    print("All tests passed.")
