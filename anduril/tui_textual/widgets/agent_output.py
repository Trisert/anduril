from __future__ import annotations

import asyncio
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Button, Collapsible, Static


COLOR_USER = "bold #9ece6a"
COLOR_ASSISTANT = "default"
COLOR_TOOL = "dim #9ece6a"
COLOR_TOOL_BORDER = "#9ece6a"
COLOR_TOOL_RESULT = "dim white"
COLOR_ERROR = "bold #f7768e"
COLOR_SYSTEM = "dim #565f89"


class InlineConfirm(Static):
    """Inline tool-approval prompt embedded in the conversation output."""

    def __init__(self, tool_name: str, args: dict, future: asyncio.Future) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args
        self._future = future

    def compose(self) -> None:
        args_text = ", ".join(
            f"{k}={v}" if not (isinstance(v, str) and len(v) > 60) else f"{k}={v[:57]}…"
            for k, v in self._args.items()
            if k != "__call_id"
        )
        yield Static(Text(f"  Approve tool: {self._tool_name}({args_text})", COLOR_TOOL))
        with Horizontal():
            yield Button("Approve", variant="primary", id="btn-approve")
            yield Button("Deny", variant="default", id="btn-deny")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        approved = event.button.id == "btn-approve"
        self.remove_children()
        label = "Approved" if approved else "Denied"
        color = "bold #9ece6a" if approved else "bold #f7768e"
        self.mount(Static(Text(f"  {self._tool_name}: {label}", color)))
        self._future.set_result(approved)


class AgentOutput(ScrollableContainer):
    """Scrollable conversation output with Markdown + code rendering."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._streaming_msg: Static | None = None
        self._streaming_text: str = ""
        self._tool_call_widgets: dict[str, Collapsible] = {}
        self._tool_call_names: dict[str, str] = {}
        self._tool_call_counter: int = 0

    async def add_user_message(self, content: str) -> None:
        await self.mount(
            Static(Text(f"  {content}", COLOR_USER), classes="msg user")
        )
        self.scroll_end(animate=False)

    async def start_assistant_message(self) -> None:
        self._streaming_text = ""
        self._streaming_msg = Static("", classes="msg assistant streaming")
        await self.mount(self._streaming_msg)
        self.scroll_end(animate=False)

    def add_stream_chunk(self, chunk: str) -> None:
        if self._streaming_msg is None:
            return
        self._streaming_text += chunk
        self._streaming_msg.update(self._streaming_text)
        self.scroll_end(animate=False)

    async def end_assistant_message(self, content: str | None = None) -> None:
        if self._streaming_msg is None:
            return
        body = content if content is not None else self._streaming_text
        await self._streaming_msg.remove()
        self._streaming_msg = None
        self._streaming_text = ""
        if body.strip():
            await self.mount(Static(
                Panel(Markdown(body), border_style=COLOR_TOOL_BORDER, padding=(0, 1)),
                classes="msg assistant",
            ))
            self.scroll_end(animate=False)

    async def add_assistant_message(self, content: str) -> None:
        if self._streaming_msg is not None:
            await self.end_assistant_message(content)
            return
        if content.strip():
            await self.mount(Static(
                Panel(Markdown(content), border_style=COLOR_TOOL_BORDER, padding=(0, 1)),
                classes="msg assistant",
            ))
            self.scroll_end(animate=False)

    async def add_tool_call(self, name: str, args: dict[str, Any]) -> None:
        import json as _json
        parts = []
        for k, v in args.items():
            if k == "__call_id":
                continue
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "…"
            parts.append(f"{k}={v}")
        args_str = ", ".join(parts)
        label = Text(f"  {name}({args_str})", COLOR_TOOL)
        self._tool_call_counter += 1
        uid = f"{name}-{self._tool_call_counter}"
        collapsible = Collapsible(
            Static(label),
            title=f"  {name}",
            collapsed=True,
            collapsed_symbol="▶",
            expanded_symbol="▼",
            classes="msg tool",
        )
        self._tool_call_widgets[uid] = collapsible
        self._tool_call_names[uid] = name
        await self.mount(collapsible)
        self.scroll_end(animate=False)

    async def add_tool_result(self, name: str, result: str) -> None:
        matching = [
            uid for uid, n in self._tool_call_names.items()
            if n == name
        ]
        uid = matching[-1] if matching else None
        widget = self._tool_call_widgets.get(uid) if uid else None
        truncated = result[:400] + "…" if len(result) > 400 else result
        if "\n" in truncated or len(truncated) > 80:
            body: Any = Syntax(truncated, "text", theme="monokai", word_wrap=True)
        else:
            body = Text(truncated, COLOR_TOOL_RESULT)
        panel = Panel(
            body,
            title=f"Result: {name}",
            border_style="dim white",
            padding=(0, 1),
        )
        if widget is not None:
            await widget.mount(Static(panel))
        else:
            await self.mount(Static(panel, classes="msg tool-result"))
        self.scroll_end(animate=False)

    async def add_error(self, message: str) -> None:
        await self.mount(
            Static(Text(f"  ✗ {message}", COLOR_ERROR), classes="msg error")
        )
        self.scroll_end(animate=False)

    async def add_system_message(self, message: str) -> None:
        await self.mount(
            Static(Text(f"  {message}", COLOR_SYSTEM), classes="msg system")
        )
        self.scroll_end(animate=False)

    async def add_confirm_prompt(self, name: str, args: dict) -> asyncio.Future:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        widget = InlineConfirm(name, args, future)
        await self.mount(widget)
        self.scroll_end(animate=False)
        return future
