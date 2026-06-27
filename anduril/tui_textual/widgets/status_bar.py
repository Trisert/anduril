from __future__ import annotations

import subprocess

from textual.widgets import Static


class StatusBar(Static):
    """Single-line status bar showing model, session, tokens, and git branch."""

    def __init__(self) -> None:
        super().__init__("  ")
        self._model = ""
        self._session = ""
        self._tokens_in = 0
        self._tokens_out = 0
        self._branch = ""

    def update_status(
        self,
        model: str = "",
        session: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        branch: str = "",
    ) -> None:
        self._model = model
        self._session = session
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._branch = branch
        self._render_bar()

    def _render_bar(self) -> None:
        parts = []
        if self._model:
            parts.append(self._model)
        if self._session:
            short = self._session.split("/")[-1][:12] if "/" in self._session else self._session[:12]
            parts.append(short)
        if self._tokens_in or self._tokens_out:
            parts.append(f"{self._tokens_in}→{self._tokens_out}")
        if self._branch:
            parts.append(self._branch)
        text = " · ".join(parts) if parts else ""
        self.update(f"  {text}")

    def refresh_branch(self) -> None:
        try:
            self._branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            self._branch = ""
        self._render_bar()
