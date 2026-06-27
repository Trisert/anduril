from __future__ import annotations

import asyncio
import functools
import json
import pathlib
import re
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.theme import Theme

from anduril.agent import Agent
from anduril.files import (
    expand_mentions as _expand_mentions,
    is_image as _is_image,
    read_clipboard_image as _read_clipboard_image,
    save_pasted_image as _save_pasted_image,
)
from anduril.tui_textual.widgets.agent_input import AgentInput
from anduril.tui_textual.widgets.agent_output import AgentOutput
from anduril.tui_textual.widgets.ask_modal import AskModal
from anduril.tui_textual.widgets.command_palette import CommandPalette
from anduril.tui_textual.widgets.file_picker import FilePicker
from anduril.tui_textual.widgets.status_bar import StatusBar
from anduril.tui_textual.widgets.thinking import ThinkingIndicator

_IMAGE_ID_RE = re.compile(r"^image-(\d+)$")


_ANDURIL_THEME = Theme(
    name="anduril",
    primary="#9ece6a",
    accent="#9ece6a",
    surface="#1a1b26",
    background="#11111b",
    foreground="#c0caf5",
    error="#f7768e",
    success="#9ece6a",
    warning="#e0af68",
    dark=True,
)


class AndurilApp(App):
    """Textual TUI for anduril."""

    CSS = """
    Screen {
        layout: vertical;
    }

    StatusBar {
        dock: top;
        height: 1;
        padding: 0 1;
    }

    AgentOutput {
        height: 1fr;
        padding: 0 1;
    }

    AgentInput {
        height: auto;
        min-height: 1;
        max-height: 7;
        padding: 0 1;
    }

    ThinkingIndicator {
        height: 1;
        padding: 0 1;
        display: none;
    }
    """

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+o", "file_picker", "Insert file", show=True),
        Binding("ctrl+s", "save_session", "Save session", show=True),
        Binding("ctrl+r", "resume_session", "Resume session", show=True),
    ]

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.register_theme(_ANDURIL_THEME)
        self.theme = "anduril"
        self.agent = agent
        self._submitting = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self.attachments: dict[str, str] = {}
        self._next_image_n: int = 1

    def compose(self) -> ComposeResult:
        yield AgentOutput()
        yield ThinkingIndicator()
        yield AgentInput()
        yield StatusBar()

    async def on_mount(self) -> None:
        self._loop = asyncio.get_running_loop()
        output = self.query_one(AgentOutput)
        await output.add_system_message(
            f"Model: {self.agent.model} · "
            f"Session: {self.agent.history_path or 'none'}"
        )
        self.query_one(AgentInput).focus()
        self.query_one(StatusBar).refresh_branch()
        # Deferred fetch: try to resolve "local" from the server in case
        # it wasn't ready when _build_agent ran.
        if self.agent.model == "local":
            self._loop.run_in_executor(None, self._resolve_model)

    async def on_agent_input_submit_message(self, message: AgentInput.SubmitMessage) -> None:
        if self._submitting:
            return
        self._submitting = True
        inp = self.query_one(AgentInput)
        out = self.query_one(AgentOutput)
        thinking = self.query_one(ThinkingIndicator)
        inp.disabled = True
        await out.add_user_message(message.text)
        await out.start_assistant_message()
        thinking.start()

        self.agent.confirm_callback = self._threadsafe_confirm
        self.agent.user_input_callback = self._threadsafe_ask

        user_message: str | list[dict[str, Any]] = message.text
        try:
            parts = _expand_mentions(
                message.text,
                cwd=pathlib.Path.cwd(),
                attachments=self.attachments,
            )
            multimodal = (
                len(parts) > 1
                or any(p.get("type") != "text" for p in parts)
            )
            if multimodal:
                user_message = parts
        except Exception:
            pass

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    self.agent.run,
                    user_message,
                    stream=True,
                    on_event=self._threadsafe_event,
                ),
            )
            await out.end_assistant_message(result)
            self._auto_save()
            self._update_status()
        except Exception as exc:
            await out.end_assistant_message()
            await out.add_error(f"Agent error: {exc}")
        finally:
            thinking.stop()
            self._submitting = False
            inp.disabled = False
            inp.focus()

    async def on_agent_input_interrupt_message(self, message: AgentInput.InterruptMessage) -> None:
        if self._submitting:
            self.agent.interrupt_check = lambda: True
            await self.query_one(AgentOutput).add_system_message("Interrupting…")
        else:
            self.exit()

    # ── Event callbacks (called from agent thread) ──────────────

    def _threadsafe_event(self, event: dict) -> None:
        async def _wrapped():
            with self._context():
                await self._handle_event(event)
        asyncio.run_coroutine_threadsafe(_wrapped(), loop=self._loop)

    async def _handle_event(self, event: dict) -> None:
        out = self.query_one(AgentOutput)
        role = event.get("role", event.get("type", ""))
        if role == "assistant":
            out.add_stream_chunk(event.get("delta", ""))
        elif role == "tool_call":
            raw_args = event.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {"raw": raw_args}
            await out.add_tool_call(event.get("name", "?"), raw_args)
        elif role == "tool":
            await out.add_tool_result(
                event.get("name", "?"), str(event.get("result", ""))
            )
        elif role == "error":
            await out.add_error(event.get("message", ""))
        elif role == "auto_compress":
            await out.add_system_message(
                f"Context compression ({event.get('est_tokens', 0)} tokens)…"
            )
        elif role == "auto_compress_done":
            await out.add_system_message(
                f"Compressed: kept {event.get('kept', 0)}, "
                f"summarized {event.get('summarized', 0)}"
            )

    # ── Confirm / Ask modals (called from agent thread) ─────────

    def _threadsafe_confirm(self, name: str, args: dict) -> bool:
        async def _wrapped():
            with self._context():
                return await self._push_confirm_inline(name, args)
        future = asyncio.run_coroutine_threadsafe(_wrapped(), loop=self._loop)
        return future.result(timeout=120)

    async def _push_confirm_inline(self, name: str, args: dict) -> bool:
        out = self.query_one(AgentOutput)
        future = await out.add_confirm_prompt(name, args)
        return await future

    def _threadsafe_ask(self, question: str, options: list[str] | None) -> str:
        async def _wrapped():
            with self._context():
                return await self._push_ask(question, options)
        future = asyncio.run_coroutine_threadsafe(_wrapped(), loop=self._loop)
        return future.result(timeout=120)

    async def _push_ask(self, question: str, options: list[str] | None) -> str:
        future = asyncio.get_running_loop().create_future()
        self.push_screen(AskModal(question, options), callback=lambda r: future.set_result(r or ""))
        return await future

    # ── Status bar ─────────────────────────────────────────────

    def _update_status(self) -> None:
        status = self.query_one(StatusBar)
        usage = self.agent.last_turn_usage
        tokens_in = usage.get("input_tokens", 0) + usage.get("cache_read_tokens", 0) if usage else 0
        tokens_out = usage.get("output_tokens", 0) if usage else 0
        status.update_status(
            model=self.agent.model,
            session=self.agent.history_path or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def _resolve_model(self) -> None:
        """Fetch model name from server and refresh status bar."""
        prev = self.agent.model
        self.agent.fetch_model_from_server()
        if self.agent.model != prev:
            self.call_from_thread(self._update_status)

            async def _notify() -> None:
                await self.query_one(AgentOutput).add_system_message(
                    f"Model resolved: {self.agent.model}"
                )
            self.call_from_thread(_notify)

    # ── Actions ──────────────────────────────────────────────────

    def action_command_palette(self) -> None:
        self.push_screen(CommandPalette(), self._on_command)

    async def _on_command(self, cmd: str) -> None:
        if not cmd:
            return
        out = self.query_one(AgentOutput)
        if cmd == "/goal show":
            goal = self._get_goal()
            await out.add_system_message(goal or "No goal set")
        elif cmd == "/goal clear":
            self.agent.goal = None
            await out.add_system_message("Goal cleared")
        elif cmd == "/paste":
            await self._cmd_paste()
        elif cmd == "/attachments":
            await self._cmd_attachments()
        elif cmd == "/help":
            await out.add_system_message(
                "Commands: /goal show, /goal clear, /paste, /attachments, /help | "
                "Keys: Ctrl+P commands, Ctrl+O file, Ctrl+S save, "
                "Ctrl+R resume, Ctrl+C interrupt, Enter send, Shift+Enter newline"
            )
        elif cmd == "/clear":
            await self._cmd_clear()

    def _get_goal(self) -> str | None:
        return getattr(self.agent, "goal", None)

    async def _cmd_paste(self) -> None:
        out = self.query_one(AgentOutput)
        try:
            data, ext = _read_clipboard_image()
        except Exception as e:
            await out.add_error(f"Clipboard image failed: {e}")
            return
        if not data:
            await out.add_system_message("No image in clipboard")
            return
        try:
            path = _save_pasted_image(data, ext)
        except Exception as e:
            await out.add_error(f"Save failed: {e}")
            return
        n = self._next_image_n
        self._next_image_n += 1
        short_id = f"image-{n}"
        self.attachments[short_id] = str(path)
        inp = self.query_one(AgentInput)
        sep = "" if inp.text.endswith((" ", "\n")) or not inp.text else " "
        inp.text += f"{sep}@{short_id}"
        inp.focus()
        await out.add_system_message(
            f"Image pasted → @{short_id} ({pathlib.Path(path).name})"
        )

    async def _cmd_attachments(self) -> None:
        out = self.query_one(AgentOutput)
        if not self.attachments:
            await out.add_system_message("No attachments")
            return
        lines = [f"  {k} → {v}" for k, v in self.attachments.items()]
        await out.add_system_message(
            "Attachments:\n" + "\n".join(lines)
        )

    async def _cmd_clear(self) -> None:
        self.agent._messages.clear()
        self.attachments.clear()
        out = self.query_one(AgentOutput)
        out.remove_children()
        await out.add_system_message("Conversation cleared")

    def action_file_picker(self) -> None:
        self.push_screen(FilePicker(), self._on_file_picked)

    def _on_file_picked(self, path: str) -> None:
        if path:
            inp = self.query_one(AgentInput)
            sep = "" if inp.text.endswith((" ", "\n")) or not inp.text else " "
            inp.text += f"{sep}@{path}"
            inp.focus()

    async def action_save_session(self) -> None:
        out = self.query_one(AgentOutput)
        if self.agent.history_path:
            self.agent.save_session()
            await out.add_system_message("Session saved")
        else:
            await out.add_system_message("No history path set")

    async def action_resume_session(self) -> None:
        from anduril.sessions import _list_sessions

        out = self.query_one(AgentOutput)
        sessions = _list_sessions(limit=50)
        if not sessions:
            await out.add_system_message("No saved sessions")
            return
        latest = sessions[0]
        self.agent.load_session(latest["id"])
        await out.add_system_message(
            f"Resumed session {latest['id']} ({latest.get('title', '')})"
        )
        self._update_status()

    def _auto_save(self) -> None:
        if self.agent.history_path:
            try:
                self.agent.save_session()
            except Exception:
                pass
