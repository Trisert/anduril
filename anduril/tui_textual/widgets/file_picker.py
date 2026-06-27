from __future__ import annotations

import os
import pathlib

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView


class FilePicker(ModalScreen[str]):
    """File picker for @-file insertion."""

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel", show=False),
    ]

    def __init__(self, root: str = ".") -> None:
        super().__init__()
        self._root = pathlib.Path(root).resolve()
        self._current = self._root

    def compose(self) -> ComposeResult:
        yield Input(placeholder=f"Filter files in {self._current}…", id="file-input")
        yield ListView(id="file-list")

    def on_mount(self) -> None:
        self._refresh()
        self.query_one("#file-input", Input).focus()

    def _refresh(self, filter_text: str = "") -> None:
        lv = self.query_one("#file-list", ListView)
        lv.clear()
        try:
            entries = sorted(
                self._current.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            entries = []

        lv.append(ListItem.from_values(".."))

        for entry in entries:
            name = entry.name + "/" if entry.is_dir() else entry.name
            if filter_text and filter_text.lower() not in name.lower():
                continue
            lv.append(ListItem.from_values(name))

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refresh(event.value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None:
            return
        text = str(event.item.children[0].renderable or "")
        if text == "..":
            self._current = self._current.parent
            self._refresh()
            self.query_one("#file-input", Input).value = ""
            return
        if text.endswith("/"):
            self._current = self._current / text.rstrip("/")
            self._refresh()
            self.query_one("#file-input", Input).value = ""
            return
        path = str(self._current / text)
        self.dismiss(path)

    def action_dismiss(self) -> None:
        self.dismiss("")
