from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView


class AskModal(ModalScreen[str]):
    """Modal that asks the user a question and returns their answer."""

    CSS = """
    AskModal > Vertical {
        align: center middle;
        padding: 2 4;
        border: thick $success;
        background: $surface;
        width: 60%;
    }

    #question {
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
    }

    #options-list {
        width: 100%;
        max-height: 10;
        margin-bottom: 1;
    }

    #answer-input {
        width: 100%;
        margin-top: 1;
    }
    """

    def __init__(self, question: str, options: list[str] | None = None) -> None:
        super().__init__()
        self._question = question
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-dialog"):
            yield Label(self._question, id="question")
            if self._options:
                items = [ListItem(Label(opt)) for opt in self._options]
                items.append(ListItem(Label("Type your answer…")))
                yield ListView(*items, id="options-list")
            yield Input(placeholder="Type your answer…", id="answer-input")

    def on_mount(self) -> None:
        if not self._options:
            self.query_one("#answer-input", Input).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        answer = event.item.children[0].renderable  # type: ignore[union-attr]
        if answer == "Type your answer…":
            self.query_one("#answer-input", Input).focus()
        else:
            self.dismiss(answer)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)
