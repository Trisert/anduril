from __future__ import annotations

import random

from textual.widgets import Static

_SPINNER = ["·", "✢", "✳", "✶", "✻", "✽"]
_VERBS = ["Thinking", "Processing", "Analyzing", "Working", "Computing", "Reasoning"]


class ThinkingIndicator(Static):
    """Animated spinner shown while the agent is working."""

    def __init__(self) -> None:
        super().__init__("")
        self._idx = 0
        self._verb = random.choice(_VERBS)
        self._timer = None

    def on_mount(self) -> None:
        self.display = False

    def start(self) -> None:
        self._idx = 0
        self._verb = random.choice(_VERBS)
        self.display = True
        self._timer = self.set_interval(0.05, self._tick)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.display = False
        self.update("")

    def _tick(self) -> None:
        self._idx = (self._idx + 1) % len(_SPINNER)
        self.update(f"  {_SPINNER[self._idx]} {self._verb}…")
