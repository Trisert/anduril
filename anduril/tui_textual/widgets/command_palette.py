from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView


_COMMANDS = [
    ("/goal show", "Show current goal"),
    ("/goal clear", "Clear current goal"),
    ("/help", "Show help"),
    ("/resume", "Resume session"),
    ("/clear", "Clear conversation"),
]


class CommandPalette(ModalScreen[str]):
    """Slash-command palette shown when user types / at start of input."""

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type a command…", id="cmd-input")
        yield ListView(id="cmd-list", *[
            ListItem.from_values(f"{cmd} — {desc}")
            for cmd, desc in _COMMANDS
        ])

    def on_mount(self) -> None:
        self.query_one("#cmd-input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        lv = self.query_one("#cmd-list", ListView)
        lv.clear()
        for cmd, desc in _COMMANDS:
            if query in cmd.lower() or query in desc.lower():
                lv.append(ListItem.from_values(f"{cmd} — {desc}"))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item:
            text = str(event.item.children[0].renderable or "")
            cmd = text.split(" — ")[0]
            self.dismiss(cmd)

    def action_dismiss(self) -> None:
        self.dismiss("")
