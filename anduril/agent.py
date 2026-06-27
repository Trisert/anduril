"""The agent: model turn loop, tool dispatch, retry nudges, and compression.

This is the part that actually talks to the model. The :class:`Agent`
class holds the message history and runs the streaming tool-calling
loop. :func:`compress` summarizes older turns to bound context size.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time
from types import SimpleNamespace
from typing import Any, Callable, Optional, Union

from openai import OpenAI

from anduril.context import (
    DEFAULT_AUTO_COMPRESS,
    DEFAULT_CONTEXT_FRACTION,
    should_auto_compress,
)
from anduril.env import _env_int, RED, RESET
from anduril.metrics import _Metrics, _normalize_usage
from anduril.sessions import _load_session
from anduril.tools import Tool, _validate


# A user message can be a plain string (the historical default) or the
# OpenAI multimodal ``content`` list — a sequence of text / image_url
# parts. The agent's streaming loop and the chat-completions endpoint
# both accept either form unchanged.
UserMessage = Union[str, list[dict[str, Any]]]


# === Tunables =============================================================

MAX_COMPLETION_TOKENS = _env_int("ANDURIL_MAX_TOKENS", 16000)
REASONING_ONLY_CHAR_LIMIT = _env_int("ANDURIL_REASONING_ONLY_CHARS", 36000)
TOOL_RESULT_CHARS = _env_int("ANDURIL_TOOL_RESULT_CHARS", 20000)
COMPRESS_KEEP = 2
MALFORMED_STREAM_RETRY_LIMIT = _env_int("ANDURIL_MALFORMED_STREAM_RETRIES", 2)
REASONING_ONLY_RETRY_LIMIT = _env_int("ANDURIL_REASONING_ONLY_RETRIES", 1)


# === Tool-result sanitization =============================================


def _sanitize_tool_result(text: str) -> str:
    """Head/tail cap and dedup ≥3 identical consecutive lines.

    Big, repetitive tool outputs (find/ls/grep path dumps) are the fuel for
    context-copying collapse on local models, so we collapse runs of ≥3
    identical lines and head/tail-cap to ``TOOL_RESULT_CHARS``. Short,
    non-repetitive results pass through untouched.
    """
    if not isinstance(text, str) or len(text) <= 1000:
        return text
    lines = text.split("\n")
    out_lines: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        run = j - i
        if run >= 3:
            out_lines.append(lines[i])
            out_lines.append(f"... [+{run - 1} identical lines elided]")
        else:
            out_lines.extend(lines[i:j])
        i = j
    result = "\n".join(out_lines)
    budget = TOOL_RESULT_CHARS
    if budget > 0 and len(result) > budget:
        head = budget * 2 // 3
        tail = budget - head
        elided = len(result) - head - tail
        result = (
            result[:head]
            + f"\n... [{elided} chars elided to bound context; "
              f"re-run more narrowly if you need the rest]\n"
            + result[-tail:]
        )
    return result


# === Text-fallback tool parsing ==========================================

TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_text_calls(content: str) -> list[tuple[str, dict]]:
    """Pull <tool_call>{...}</tool_call> blocks out of model text.

    Survives models whose native tool-calling isn't wired up by falling back
    to the convention most open-weight models (Hermes / Qwen / Nemotron)
    emit in plain text.
    """
    calls: list[tuple[str, dict]] = []
    for m in TOOL_TAG.finditer(content or ""):
        try:
            obj = json.loads(m.group(1))
            calls.append((obj["name"], obj.get("arguments", {})))
        except (json.JSONDecodeError, KeyError):
            pass
    return calls


# === Context compression =================================================


def compress(messages: list, keep: int = COMPRESS_KEEP,
             model: str | None = None, client: Any = None) -> tuple | None:
    """Summarize everything except the system prompt + last `keep` turns.

    Mutates `messages` in place on success. Returns (kept_n, summarized_n,
    summary_chars) or None on failure.
    """
    if client is None or not model:
        print(f"{RED}  ✗ compress needs model + client{RESET}")
        return None
    if len(messages) <= 1 + keep:
        return None
    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    body = messages[1:] if sys_msg else messages
    if len(body) <= keep:
        return None

    head, tail = body[:-keep], body[-keep:]
    summarized_n = len(head)

    # Tail must start on a turn the chat template can render. A `tool` turn
    # with no preceding assistant(tool_calls) parent — or an
    # assistant(tool_calls) turn whose result got cut off into `head` —
    # makes llama.cpp's Jinja template raise. Walk from the front of the tail
    # and drop any leading unsafe turns.
    while tail:
        first = tail[0]
        if first.get("role") == "tool":
            tail = tail[1:]
            summarized_n += 1
            continue
        if first.get("role") == "assistant" and first.get("tool_calls"):
            ids = {tc["id"] for tc in first["tool_calls"]}
            seen = set()
            for m in tail[1:]:
                tcid = m.get("tool_call_id")
                if m.get("role") == "tool" and tcid:
                    seen.add(tcid)
            if ids - seen:
                tail = tail[1:]
                summarized_n += 1
                continue
        break
    if not tail:
        return None

    def _render(msgs: list) -> str:
        out = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content")
            if content is None and m.get("tool_calls"):
                calls = ", ".join(
                    f"{c['function']['name']}({c['function']['arguments']})"
                    for c in m["tool_calls"]
                )
                out.append(f"[{role}] → {calls}")
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", "")[:2000])
                text = " ".join(parts)
                out.append(f"[{role}] {text}" if text else f"[{role}] (image)")
            else:
                out.append(f"[{role}] {(content or '')[:2000]}")
        return "\n\n".join(out)

    summary_prompt = (
        "Summarize the following conversation history for context retention. "
        "Preserve: the original user goal/task, key decisions made, file paths "
        "and identifiers touched, current state of any in-progress work, and "
        "any unresolved questions. Drop: raw tool outputs, full file contents, "
        "and verbose back-and-forth — keep it dense and information-rich. "
        "Write in the same language as the conversation. Output ONLY the "
        "summary, no preamble.\n\n"
        f"---\n{_render(head)}\n---"
    )

    payload = [{"role": "user", "content": summary_prompt}]
    try:
        resp = client.chat.completions.create(
            model=model, messages=payload, stream=False, timeout=60,
        )
    except Exception as e:
        print(f"{RED}  ✗ compress failed: {type(e).__name__}: {e}{RESET}")
        return None

    summary = (resp.choices[0].message.content or "").strip()
    if not summary:
        return None

    header = (
        f"[Compressed context — {summarized_n} earlier turns summarized; "
        f"last {keep} turns kept verbatim]"
    )
    new_mid = [{"role": "user", "content": f"{header}\n\n{summary}"}]
    messages[:] = ([sys_msg] if sys_msg else []) + new_mid + tail
    return len(tail), summarized_n, len(summary)


# === Message / tool-call helpers =========================================


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Normalize an OpenAI message object or dict into a plain dict."""
    if isinstance(msg, dict):
        return dict(msg)
    out: dict[str, Any] = {"role": msg.role}
    if getattr(msg, "content", None):
        out["content"] = msg.content
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [_tool_call_to_dict(tc) for tc in tool_calls]
    return out


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    if isinstance(tc, dict):
        return dict(tc)
    return {
        "id": tc.id,
        "type": getattr(tc, "type", "function"),
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }


class _ToolCallAggregator:
    """Collect streaming tool-call deltas into complete tool calls."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def add(self, delta: Any) -> None:
        idx = delta.index
        if idx not in self._calls:
            self._calls[idx] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        if delta.id:
            self._calls[idx]["id"] += delta.id
        func = delta.function
        if func:
            if func.name:
                self._calls[idx]["function"]["name"] += func.name
            if func.arguments:
                self._calls[idx]["function"]["arguments"] += func.arguments

    def peek(self, idx: int) -> dict[str, Any] | None:
        """Public read-only view used by the streaming loop."""
        return self._calls.get(idx)

    def finalize(self) -> list[dict[str, Any]]:
        return [self._calls[i] for i in sorted(self._calls.keys())]


# Turn status codes returned by Agent._model_turn.
TURN_DONE = "done"
TURN_TOOL = "tool"
TURN_STREAM_CUT = "stream_cut"
TURN_FORCE_FINAL = "force_final"
TURN_ESC = "esc"

# A nudge injected into the user's last turn when the model needs to be told
# to drop whatever it was doing and retry cleanly.
_RUNTIME_NOTE_RE = re.compile(r"\n\n\[Runtime note: .*?\]\s*$", re.DOTALL)
FORCED_FINAL_NUDGE = (
    "Your previous streamed response produced reasoning only. Do not continue "
    "private reasoning. Return a complete visible answer now in at most six "
    "short bullets or paragraphs. If you are blocked, say exactly what is "
    "blocking you and what input is needed."
)
MALFORMED_STREAM_NUDGE = (
    "Your previous streamed response became malformed before it completed. "
    "Discard it. Retry the same task from the current conversation state, but "
    "emit either valid tool calls or a concise final answer only."
)
MALFORMED_TOOL_CALL_NUDGE = (
    "Your previous native tool call had malformed or truncated JSON "
    "arguments, so it was discarded before execution. Retry the same task "
    "from the current conversation state, but emit a complete, valid tool "
    "call with all required arguments. For large file writes, emit only the "
    "tool call and skip any explanatory prose."
)
INTERRUPT_NOTE = (
    "[User interrupted your previous response with Esc. "
    "Acknowledge briefly and wait for their next message.]"
)
ESC_AT_APPROVAL_NOTE = (
    "[User pressed Esc at a tool approval prompt and returned to chat to "
    "add more input. Acknowledge briefly and wait for their next message.]"
)


def _nudge_current_user_turn(messages: list, nudge: str) -> None:
    note = f"[Runtime note: {nudge}]"
    for msg in reversed(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), str):
            continue
        content = _RUNTIME_NOTE_RE.sub("", msg["content"]).rstrip()
        msg["content"] = f"{content}\n\n{note}" if content else note
        return
    messages.append({"role": "user", "content": note})


# === Current-agent registry ===============================================
#
# A small module-level registry so tools that run inside an
# agent's tool-call loop can find the agent that called them
# without an env-var round-trip. The TUI / CLI instantiates
# exactly one agent, so the ``_current`` slot is unambiguous
# in the typical use case. Multi-agent setups would need to
# pass the target agent explicitly to the tool (not built
# yet).
#
# Using a dedicated module (``_current_agent_module``) rather
# than ``anduril.agent`` itself avoids an import cycle:
# ``anduril.tools.add_mcp_server`` imports this module to
# look up the current agent, and ``anduril.agent.Agent``
# assigns to it on construction.

_current_agent_module = SimpleNamespace()
_current_agent_module._current = None


# === Agent ===============================================================


class Agent:
    def __init__(
        self,
        model: str,
        system: str = "",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: tuple[Tool, ...] = (),
        max_turns: int = 25,
        max_retries: int = 3,
        history_path: Optional[pathlib.Path | str] = None,
        confirm_callback: Optional[Callable[[str, dict[str, Any]], bool]] = None,
        interrupt_check: Optional[Callable[[], bool]] = None,
        auto_compress: bool = DEFAULT_AUTO_COMPRESS,
        context_fraction: float = DEFAULT_CONTEXT_FRACTION,
        system_overrides: Optional[dict[str, str]] = None,
    ):
        # Per-model system-prompt overrides. Keys are
        # case-insensitive substrings of the model name; the
        # most specific (longest) match wins. A blank
        # string means "force the system prompt to empty
        # for this model" (some local models prefer no
        # system prompt at all).
        self.system_overrides: dict[str, str] = {
            k.lower(): v for k, v in (system_overrides or {}).items()
        }
        # Register this agent as the process-wide "current agent"
        # so the ``add_mcp_server`` tool (and any other tool that
        # wants to register new tools at runtime) can find the
        # agent without an env-var round-trip. The reference is
        # cleared by ``self.close()``; until then, only this
        # agent is "current". This is fine for the usual
        # single-agent CLI / TUI use case; a multi-agent setup
        # would have to pass the target agent explicitly to the
        # tool (not implemented yet).
        _current_agent_module._current = self
        self.model = model
        self.system = system
        self.goal: str | None = None
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.auto_compress = auto_compress
        self.context_fraction = context_fraction
        self.tools = {t.name: t for t in tools}
        self._tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        self.confirm_callback = confirm_callback
        self.user_input_callback: Callable[[str, list[str] | None], str] | None = None
        self.interrupt_check = interrupt_check
        self.client = OpenAI(
            base_url=base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("ANDURIL_BASE_URL")
            or "http://localhost:8080/v1",
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "no-key",
        )
        self.history_path = (
            pathlib.Path(history_path)
            if history_path
            else pathlib.Path.home() / ".local" / "state" / "anduril" / "history.jsonl"
        )
        self._messages: list[dict[str, Any]] = []
        if self.system:
            self._messages.append({"role": "system", "content": self.system})
        self._metrics: Optional[_Metrics] = None

    def fetch_model_from_server(self) -> None:
        """Query /v1/models to discover the real model name from the server.

        Silently does nothing on failure (server not reachable, etc.).
        """
        try:
            models = self.client.models.list()
            for m in models:
                if m.id and m.id != self.model:
                    self.model = m.id
                    break
        except Exception:
            pass
        # Per-turn usage for the most recent model call. The TUI
        # reads this after a turn completes to replace the rough
        # char-based live estimate with the API-reported ground
        # truth (input / cache_read / output deltas for the last
        # call). Updated right after _metrics.add() in both
        # streaming and non-streaming paths.
        self.last_turn_usage: dict[str, int] | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def set_metrics(self, metrics: _Metrics | None) -> None:
        self._metrics = metrics

    def _record_turn_usage(self, usage: Any, timings: dict | None) -> None:
        """Normalize the most recent call's usage and stash it on the
        agent so the TUI can display the final per-turn numbers after
        the stream ends. Also feeds the cumulative session metrics."""
        delta = _normalize_usage(usage, timings)
        if delta:
            self.last_turn_usage = delta
        if self._metrics is not None:
            self._metrics.add(delta, model=self.model)

    def clear(self) -> None:
        """Drop all in-memory messages except the system prompt (if any)."""
        self._messages = []
        if self.system:
            self._messages.append({"role": "system", "content": self.system})

    def pop_last(self) -> dict[str, Any] | None:
        """Remove and return the last message, or None if the list is empty."""
        if not self._messages:
            return None
        return self._messages.pop()

    def last_user_message(self) -> dict[str, Any] | None:
        """Return the most recent user-role message, or None.

        Used by ``/retry`` (re-run the same question) and
        ``/edit`` (load it into the buffer for re-submission).
        Skips over the system message at index 0.
        """
        for m in reversed(self._messages):
            if m.get("role") == "user":
                return m
        return None

    def undo_last_turn(self) -> bool:
        """Drop everything after the most recent user message.

        Returns True if anything was actually dropped. The
        post-user-message sequence is the assistant turn
        plus any tool-call chain that followed; undoing
        ``/undo`` lets the user re-submit a different
        message or simply erase a bad response.

        Walks from the end of ``self._messages`` backwards
        until it finds a user message (or runs out of
        messages), popping everything it finds. If the
        most recent message is *itself* a user message
        (i.e. the agent hasn't replied yet), we pop that
        too — the user effectively retracts their last
        submission.
        """
        if not self._messages:
            return False
        # Walk backwards; stop at the last user message
        # (inclusive, so we always remove at least the
        # last assistant turn if there is one).
        popped = 0
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            role = msg.get("role")
            if role == "system":
                # Never pop the system message.
                break
            if role == "user" and popped > 0:
                # We've already popped the assistant turn
                # + tool chain; stop here so the user's
                # most recent submission stays in the
                # history. (If popped == 0, we're looking
                # at the very first user message and we
                # should pop it too — the user is
                # retracting their input.)
                break
            self._messages.pop()
            popped += 1
        return popped > 0

    def replay_last_user(
        self,
        on_event: Optional[Callable[[dict], None]] = None,
        stream: bool = True,
    ) -> Optional[str]:
        """Re-run ``run()`` with the most recent user message.

        ``/retry`` calls this. The user's previous question
        is sent to the model again, with the same tool set
        and system prompt. Useful when the model produced
        a bad answer and the user wants a fresh attempt
        without re-typing the prompt.

        Returns the new assistant content, or ``None`` if
        there is no user message to replay (an empty
        history).
        """
        user_msg = self.last_user_message()
        if user_msg is None:
            return None
        content = user_msg.get("content")
        # ``undo_last_turn`` keeps the most recent user
        # message in place (so the user can edit-and-retry
        # without losing their text). For ``/retry`` we
        # need the history to end cleanly with the replayed
        # user message; ``run()`` will append it for us.
        # So pop the user message here too.
        self.undo_last_turn()
        if self._messages and self._messages[-1].get("role") == "user":
            self._messages.pop()
        assert content is not None  # last_user_message returns non-None iff user
        return self.run(content, on_event=on_event, stream=stream)

    def _resolve_system_prompt(self, model: str | None = None) -> str:
        """Pick the system prompt for ``model``.

        Longest matching override wins (so a more specific
        pattern overrides a more general one). The
        constructor's ``system=`` is the fallback when no
        override matches.
        """
        if not self.system_overrides:
            return self.system
        needle = (model or self.model).lower()
        best_key = None
        best_len = -1
        for key in self.system_overrides:
            if key in needle and len(key) > best_len:
                best_key = key
                best_len = len(key)
        if best_key is None:
            return self.system
        return self.system_overrides[best_key]

    def set_system(self, text: str | None = None, *,
                  for_model: str | None = None) -> None:
        """Insert or replace the system message at the start of the history.

        With no arguments, restores the resolved system
        prompt for the current model (re-applying any
        per-model override that may have been bypassed by
        a previous ``set_system(text=...)`` call).

        With ``text=``: updates the global default and the
        current message. With ``for_model=``: registers a
        per-model override; the override only applies to
        that model and doesn't change the default.

        The current message list is updated to match the
        new prompt so the model sees the change immediately.
        """
        if for_model is not None:
            # Per-model override path.
            self.system_overrides[for_model.lower()] = text or ""
            # The current message list keeps whatever the
            # current resolved prompt is; we don't reset
            # it because the user might be in the middle of
            # a turn with a different model. If ``self.model``
            # matches, we DO update the live message so the
            # next call sees it.
            if for_model.lower() in self.model.lower():
                self._replace_system_message(self._resolve_system_prompt())
            return
        # Global default path.
        self.system = text if text is not None else ""
        self._replace_system_message(self.system)

    def _replace_system_message(self, text: str) -> None:
        """Replace (or insert) the system message at index 0."""
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = text
        else:
            self._messages.insert(0, {"role": "system", "content": text})

    def load_history(self, path: Optional[pathlib.Path | str] = None) -> int:
        """Load a prior conversation from JSON/JSONL. Returns the count loaded."""
        file = pathlib.Path(path) if path else self.history_path
        if not file.exists():
            return 0
        try:
            text = file.read_text(encoding="utf-8")
            if text.strip().startswith("["):
                data = json.loads(text)
            else:
                data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except Exception as e:
            raise RuntimeError(f"failed to load history from {file}: {e}") from e
        if not isinstance(data, list):
            raise RuntimeError("history file must contain a JSON array or JSONL lines")
        # Preserve system message if already set; otherwise use the loaded one.
        original_system = next(
            (m for m in self._messages if m.get("role") == "system"), None
        )
        self._messages = []
        if original_system is not None:
            self._messages.append(original_system)
        for msg in data:
            if original_system is not None and msg.get("role") == "system":
                continue
            self._messages.append(msg)
        return len(data)

    def save_history(self, path: Optional[pathlib.Path | str] = None) -> pathlib.Path:
        """Save the current conversation as JSONL. Returns the file path."""
        file = pathlib.Path(path) if path else self.history_path
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w", encoding="utf-8") as f:
            for msg in self._messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return file

    def load_session(self, session_id: str) -> bool:
        """Load a session by id. Replaces messages; returns False if missing."""
        data = _load_session(session_id)
        if not data:
            return False
        msgs = [m for m in data.get("messages", []) if m.get("role") != "system"]
        self._messages = []
        if self.system:
            self._messages.append({"role": "system", "content": self.system})
        self._messages.extend(msgs)
        return True

    def run(
        self,
        user_message: UserMessage,
        on_event: Optional[Callable[[dict], None]] = None,
        stream: bool = True,
        tick_callback: Optional[Callable[[], None]] = None,
    ) -> str:
        """Run the agent loop for a single user message. Returns the final content.

        The loop:
          1. Add the user message.
          2. Stream a model response (one model "turn").
          3. If the turn produced tool calls, dispatch each, append results, loop.
          4. Otherwise return the content.

        Resilience built in: stream cuts and malformed tool calls are retried
        with a model nudge; reasoning-only stalls trigger a forced final-answer
        turn. ``self.interrupt_check()`` is polled between chunks to support
        Esc-to-interrupt.
        """
        # Apply any per-model system-prompt override at the
        # start of every turn. We do this just before
        # appending the user message so the very first
        # model call of the turn sees the right prompt,
        # even if the user switched models mid-session.
        # Idempotent: a no-op if the prompt is already
        # correct.
        resolved = self._resolve_system_prompt()
        if (self._messages and self._messages[0].get("role") == "system"
                and self._messages[0].get("content") != resolved):
            self._messages[0]["content"] = resolved

        self._messages.append({"role": "user", "content": user_message})

        steps = 0
        reasoning_loop_cuts = 0
        malformed_stream_cuts = 0
        force_final = False
        last_content = ""

        while steps < self.max_turns:
            # Budget check: if a cap is set and we've already
            # met it, refuse to continue. The check is on the
            # *running* total, not a per-call estimate, so
            # it's conservative (the next call might not have
            # spent much). Better to under-spend by a fraction
            # of a cent than to silently blow through the cap.
            if (self._metrics is not None
                    and self._metrics.budget is not None
                    and self._metrics.total_cost >= self._metrics.budget):
                return (
                    f"(budget reached: ${self._metrics.total_cost:.4f} / "
                    f"${self._metrics.budget:.4f})"
                )
            # Automatic context compression. Runs at the top of every
            # loop iteration (so it also fires after a long tool-call
            # chain, not just on the first turn). The check is cheap
            # — a single walk over ``self._messages`` — so we don't
            # gate it on a step counter.
            if self.auto_compress:
                should, est, window, threshold = should_auto_compress(
                    self._messages,
                    model=self.model,
                    system=self.system or "",
                    tool_schemas=self._tool_schemas,
                    fraction=self.context_fraction,
                )
                if should:
                    if on_event:
                        on_event({
                            "type": "auto_compress",
                            "est_tokens": est,
                            "window": window,
                            "threshold": threshold,
                        })
                    result = compress(
                        self._messages,
                        model=self.model,
                        client=self.client,
                    )
                    if result is not None and on_event:
                        kept_n, summarized_n, summary_chars = result
                        on_event({
                            "type": "auto_compress_done",
                            "kept": kept_n,
                            "summarized": summarized_n,
                            "summary_chars": summary_chars,
                        })

            try:
                status, content, tool_calls = self._model_turn(
                    stream=stream,
                    force_final=force_final,
                    on_event=on_event,
                    tick_callback=tick_callback,
                )
            except KeyboardInterrupt:
                # Let the caller (TUI) handle it — drop the user message we
                # just added so a retry doesn't double-submit.
                self._messages.pop()
                raise
            except Exception:
                self._messages.pop()
                raise

            force_final = False
            last_content = content

            if status == TURN_ESC:
                # User pressed Esc during a tool approval. Tool calls that
                # were about to run were cancelled; remaining ones get a
                # SKIPPED placeholder so the message history stays valid.
                return ""

            if status == TURN_STREAM_CUT:
                malformed_stream_cuts += 1
                if malformed_stream_cuts > MALFORMED_STREAM_RETRY_LIMIT:
                    if on_event:
                        on_event({"type": "error",
                                  "message": "stream kept failing; bailing out"})
                    return content
                # _model_turn has already nudged the user turn. Retry.
                steps += 1
                continue

            if status == TURN_FORCE_FINAL:
                reasoning_loop_cuts += 1
                if reasoning_loop_cuts > REASONING_ONLY_RETRY_LIMIT:
                    if on_event:
                        on_event({"type": "error",
                                  "message": "model kept reasoning without answering"})
                    return ""
                # _model_turn has already nudged. Retry, this time with
                # force_final=True (handled via the local var above).
                force_final = True
                steps += 1
                continue

            if status == TURN_DONE:
                return content

            if status == TURN_TOOL:
                # Dispatch tool calls. Each one becomes a `tool` role message
                # in the history; results are sanitized before they enter.
                if not tool_calls:
                    return content
                self._messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    name = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments", "")
                    try:
                        parsed = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError as e:
                        malformed_stream_cuts += 1
                        if malformed_stream_cuts > MALFORMED_STREAM_RETRY_LIMIT:
                            result = f"error: invalid JSON arguments: {e}"
                            self._messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": result,
                            })
                            if on_event:
                                on_event({
                                    "role": "tool",
                                    "name": name,
                                    "args": "{}",
                                    "result": result,
                                })
                            return ""
                        _nudge_current_user_turn(self._messages, MALFORMED_TOOL_CALL_NUDGE)
                        # Re-add the partial assistant tool_calls so the
                        # next model turn can see them when it re-issues.
                        self._messages.insert(-1, {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": name, "arguments": raw_args},
                            }],
                        })
                        break
                    result = self._call_tool(name, parsed)
                    result_sanitized = _sanitize_tool_result(str(result))
                    # Pick up any tools that the just-called tool queued
                    # (e.g. create_skill adds the new skill's tools here).
                    added = self.drain_pending_registrations()
                    if added:
                        result_sanitized = (
                            result_sanitized
                            + ("" if result_sanitized.endswith("\n") else "\n")
                            + f"\n[registered new tools: {', '.join(added)}]"
                        )
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_sanitized,
                    })
                    if on_event:
                        on_event({
                            "role": "tool",
                            "name": name,
                            "args": json.dumps(parsed, ensure_ascii=False,
                                               separators=(",", ":")),
                            "result": str(result),
                        })
                steps += 1
                reasoning_loop_cuts = 0
                malformed_stream_cuts = 0
                continue

        return last_content or "(max turns reached)"

    def _model_turn(
        self,
        stream: bool = True,
        force_final: bool = False,
        on_event: Optional[Callable[[dict], None]] = None,
        tick_callback: Optional[Callable[[], None]] = None,
    ) -> tuple[str, str, list[dict]]:
        """Run one model call and return (status, content, tool_calls)."""
        if not stream:
            return self._model_turn_non_streaming(force_final=force_final)

        # Build the request kwargs.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if MAX_COMPLETION_TOKENS > 0:
            kwargs["max_tokens"] = MAX_COMPLETION_TOKENS
        if self._tool_schemas and not force_final:
            kwargs["tools"] = self._tool_schemas
        if force_final:
            # Force a single short text answer by disabling tools + capping
            # the output. We don't define a final_answer tool because the
            # model might be one that doesn't honor tool_choice cleanly.
            kwargs["max_tokens"] = min(kwargs.get("max_tokens", 2048), 2048)

        # Open the stream with retries.
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        else:
            raise last_error  # type: ignore[misc]

        t0 = time.time()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_chars = 0
        tool_agg = _ToolCallAggregator()
        usage = None
        timings: dict | None = None
        t_first: float | None = None
        interrupted = False
        stream_error: Exception | None = None

        # Stream the response in a daemon thread so the main thread can
        # poll with a short timeout, calling ``tick_callback`` between
        # chunks.  Without this the spinner (and Esc-polling) freezes
        # while the model is thinking between tokens.
        import queue
        import threading
        _chunk_queue: queue.Queue = queue.Queue(maxsize=128)
        _reader_done = threading.Event()

        def _reader() -> None:
            try:
                for chunk in response:
                    _chunk_queue.put(("chunk", chunk))
                _chunk_queue.put(("done", None))
            except Exception as e:
                _chunk_queue.put(("error", e))
            finally:
                _reader_done.set()

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        try:
            while True:
                try:
                    kind, payload = _chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    if tick_callback:
                        tick_callback()
                    # Also check for interrupts during the wait.
                    if self.interrupt_check and self.interrupt_check():
                        interrupted = True
                        close = getattr(response, "close", None)
                        if close:
                            try:
                                close()
                            except Exception:
                                pass
                        # Drain the reader thread.
                        _reader_done.wait(timeout=5)
                        break
                    continue

                if kind == "done":
                    break
                if kind == "error":
                    stream_error = payload
                    close = getattr(response, "close", None)
                    if close:
                        try:
                            close()
                        except Exception:
                            pass
                    break

                chunk = payload

                extra = getattr(chunk, "model_extra", None) or {}
                if "timings" in extra:
                    timings = extra["timings"]
                if chunk.usage:
                    usage = chunk.usage

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Reasoning (modern OpenAI SDK + many local backends).
                rc = getattr(delta, "reasoning_content", None)
                if rc is None:
                    rc = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
                if rc:
                    reasoning_parts.append(rc)
                    if t_first is None:
                        t_first = time.time() - t0
                    if on_event:
                        on_event({
                            "role": "reasoning",
                            "content": "".join(reasoning_parts),
                            "delta": rc,
                        })
                    if not content_parts and not tool_agg.peek(0):
                        reasoning_chars += len(rc)
                        if (REASONING_ONLY_CHAR_LIMIT > 0
                                and reasoning_chars >= REASONING_ONLY_CHAR_LIMIT):
                            # Cut the stream — we'll force a final answer next.
                            close = getattr(response, "close", None)
                            if close:
                                try:
                                    close()
                                except Exception:
                                    pass
                            # Append the partial reasoning so the model has it
                            # for the next turn.
                            self._messages.append({
                                "role": "assistant",
                                "content": None,
                            })
                            self._messages.append({
                                "role": "user",
                                "content": (
                                    "[Reasoning so far, then cut:]\n"
                                    + "".join(reasoning_parts)[-REASONING_ONLY_CHAR_LIMIT:]
                                    + "\n\nDo not continue private reasoning. Return a complete "
                                      "visible answer now in at most six short bullets or "
                                      "paragraphs. If you are blocked, say exactly what is "
                                      "blocking you and what input is needed."
                                ),
                            })
                            return (TURN_FORCE_FINAL, "", [])

                # Visible content.
                if delta.content:
                    if t_first is None:
                        t_first = time.time() - t0
                    content_parts.append(delta.content)
                    if on_event:
                        on_event({
                            "role": "assistant",
                            "content": "".join(content_parts),
                            "delta": delta.content,
                        })

                # Native tool calls. Emit a "tool_call" event on every
                # delta so the TUI can show the call as it forms — name
                # and arguments filling in over time — rather than
                # waiting for the tool to finish executing before the
                # user sees anything.
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        tool_agg.add(tc)
                        if on_event:
                            current = tool_agg.peek(tc.index)
                            if current:
                                on_event({
                                    "role": "tool_call",
                                    "id": current["id"],
                                    "index": tc.index,
                                    "name": current["function"]["name"],
                                    "arguments": current["function"]["arguments"],
                                })
        except Exception as e:
            stream_error = e
            close = getattr(response, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        tool_calls = tool_agg.finalize()

        # Update metrics + record per-turn usage for the TUI's live
        # status bar (so the rough char-based live estimate gets
        # replaced with the API-reported ground truth at end of turn).
        self._record_turn_usage(usage, timings)

        if interrupted:
            self._messages.append({"role": "user", "content": INTERRUPT_NOTE})
            return (TURN_DONE, content, [])

        if stream_error is not None:
            _nudge_current_user_turn(self._messages, MALFORMED_STREAM_NUDGE)
            return (TURN_STREAM_CUT, content, [])

        # Reasoning-only with content/tool_calls: fine, normal turn.
        if not content.strip() and not tool_calls:
            # No content, no tool calls, no interruption — reasoning-only.
            if reasoning:
                _nudge_current_user_turn(self._messages, FORCED_FINAL_NUDGE)
                return (TURN_FORCE_FINAL, "", [])

        if tool_calls:
            return (TURN_TOOL, content, tool_calls)

        # No tool calls and we have visible content. Append the
        # assistant message so ``/retry`` and ``/undo`` can
        # find a clean snapshot to roll back to. (The non-
        # streaming path already does this; the streaming
        # path was missing it, which used to be fine because
        # the next user message was always appended
        # immediately — but with ``/retry`` we'd see two
        # user messages in a row.)
        if content:
            self._messages.append({
                "role": "assistant",
                "content": content,
            })

        # No tool calls. Try text-fallback for models that emit <tool_call>
        # tags in plain text. The streaming loop has already appended an
        # assistant turn containing the raw text — parse the tags, dispatch
        # each call, append a user observation, and return TURN_DONE so the
        # outer loop continues with the next model turn.
        text_calls = parse_text_calls(content)
        if text_calls:
            obs = []
            for name, args in text_calls:
                result = str(self._call_tool(name, args))
                result = _sanitize_tool_result(result)
                obs.append(f"Observation ({name}): {result}")
            self._messages.append({"role": "user", "content": "\n".join(obs)})
            return (TURN_DONE, content, [])

        return (TURN_DONE, content, [])

    def _model_turn_non_streaming(
        self,
        force_final: bool = False,
        on_event: Optional[Callable[[dict], None]] = None,
    ) -> tuple[str, str, list[dict]]:
        """Non-streaming fallback path. Used when stream=False is requested."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages,
            "stream": False,
        }
        if self._tool_schemas and not force_final:
            kwargs["tools"] = self._tool_schemas
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
        else:
            raise last_error  # type: ignore[misc]

        msg = resp.choices[0].message
        content = msg.content or ""
        tool_calls = [
            {
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in (msg.tool_calls or [])
        ]
        if content or tool_calls:
            self._messages.append({
                "role": "assistant",
                "content": content or None,
                **({"tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls
                ]} if tool_calls else {}),
            })
        # Update metrics + record per-turn usage for the TUI's live
        # status bar (see _record_turn_usage docstring).
        self._record_turn_usage(getattr(resp, "usage", None), None)
        # Emit a "tool_call" event for each tool call so the TUI can
        # show the call as soon as the model returns it (the streaming
        # path emits these per delta; the non-streaming path doesn't
        # have intermediate events so we do it here).
        if tool_calls and on_event:
            for i, tc in enumerate(tool_calls):
                on_event({
                    "role": "tool_call",
                    "id": tc["id"],
                    "index": i,
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
        if tool_calls:
            return (TURN_TOOL, content, tool_calls)
        return (TURN_DONE, content, [])

    def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self.tools:
            return f"error: unknown tool '{name}'"
        tool = self.tools[name]
        errors = _validate(args, tool.parameters)
        if errors:
            return "error: invalid arguments:\n" + "\n".join(errors)
        if name == "ask" and self.user_input_callback:
            return self.user_input_callback(args["question"], args.get("options"))
        if tool.dangerous and self.confirm_callback:
            if not self.confirm_callback(name, args):
                return "error: user declined to run tool"
        try:
            return tool.fn(**args)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"

    def register_tool(self, tool: "Tool") -> None:
        """Add a tool at runtime so it's available to subsequent turns.

        Used by :func:`anduril.tools.create_skill` to extend the agent
        with new capabilities without restarting. Also drains the
        pending-registrations queue (used by skills that register tools
        as a side effect of being loaded).
        """
        if tool.name in self.tools:
            return  # already registered — de-dupe silently
        self.tools[tool.name] = tool
        self._tool_schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        })

    def drain_pending_registrations(self) -> list[str]:
        """Drain any tools queued by ``anduril.skills.register_tool``.

        Returns the names of newly registered tools (may be empty).
        """
        # Imported here to avoid a top-level cycle (skills -> tools -> ...).
        from anduril.skills import drain_pending_registrations
        added: list[str] = []
        for tool in drain_pending_registrations():
            if tool.name in self.tools:
                continue
            self.register_tool(tool)
            added.append(tool.name)
        return added

    def close(self) -> None:
        """Release process-wide resources held by this agent.

        Clears the ``_current`` pointer in the agent-registry
        module so the next :class:`Agent` instance can claim
        it. Subclasses or callers that explicitly want the
        process to keep using this agent as the "current"
        one can call :meth:`__init__` of a new agent (which
        will overwrite the pointer).

        Idempotent.
        """
        if _current_agent_module._current is self:
            _current_agent_module._current = None
