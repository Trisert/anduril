from __future__ import annotations

import asyncio
import threading
from typing import Any

from textual.app import App, ComposeResult

from anduril.tui_textual.widgets.ask_modal import AskModal


class _AskApp(App):
    """Minimal Textual app that shows AskModals on demand.

    Runs in its own thread + event loop.  ``prompt_user`` is synchronous
    (blocks the caller) and safe to call from any thread.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready_event = threading.Event()
        self._result: str = ""
        self._pending: threading.Event | None = None

    def compose(self) -> ComposeResult:
        return []

    async def on_mount(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ready_event.set()

    async def _show_modal(self, question: str, options: list[str] | None) -> None:
        result = await self.push_screen_wait(AskModal(question, options))
        self._result = result if result is not None else ""
        if self._pending:
            self._pending.set()

    def prompt_user(self, question: str, options: list[str] | None = None) -> str:
        """Synchronous bridge — blocks the calling thread."""
        self._pending = threading.Event()
        self._result = ""
        asyncio.run_coroutine_threadsafe(
            self._show_modal(question, options), self._loop  # type: ignore[arg-type]
        )
        self._pending.wait(timeout=120)
        return self._result


_app: _AskApp | None = None
_thread: threading.Thread | None = None


def prompt_user(question: str, options: list[str] | None = None) -> str:
    """Synchronous ask-textual prompt (safe for ``agent.user_input_callback``)."""
    global _app, _thread
    if _app is None:
        _app = _AskApp()
        _thread = threading.Thread(
            target=lambda: asyncio.run(_app.run_async()), daemon=True
        )
        _thread.start()
        _app._ready_event.wait()  # wait for on_mount (app is alive)
    return _app.prompt_user(question, options)
