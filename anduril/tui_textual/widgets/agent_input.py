from __future__ import annotations

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class AgentInput(TextArea):
    """Multi-line input with history, Ctrl+Enter submit, @ and / triggers."""

    class SubmitMessage(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class InterruptMessage(Message):
        pass

    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True),
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
        Binding("up", "history_back", "History back", show=False),
        Binding("down", "history_forward", "History forward", show=False),
    ]

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault(
            "placeholder",
            "Type a message… (Enter to send, Shift+Enter for newline, Ctrl+C to interrupt)",
        )
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1
        self._editing_text: str = ""

    def action_submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self._history.append(text)
        self._history_index = len(self._history)
        self.post_message(self.SubmitMessage(text))
        self.text = ""
        self._history_index = len(self._history)

    def action_newline(self) -> None:
        self.insert("\n")

    def action_interrupt(self) -> None:
        self.post_message(self.InterruptMessage())

    def action_history_back(self) -> None:
        if self._history_index <= 0:
            return
        if self._history_index == len(self._history):
            self._editing_text = self.text
        self._history_index -= 1
        self.text = self._history[self._history_index]
        self.move_cursor_to_end()

    def action_history_forward(self) -> None:
        if self._history_index >= len(self._history):
            return
        self._history_index += 1
        if self._history_index == len(self._history):
            self.text = self._editing_text
            self._editing_text = ""
        else:
            self.text = self._history[self._history_index]
        self.move_cursor_to_end()

    def move_cursor_to_end(self) -> None:
        lines = self.text.split("\n")
        last_line = len(lines)
        last_col = len(lines[-1]) if lines else 0
        self.cursor_location = (last_line - 1, last_col)
