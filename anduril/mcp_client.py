"""Minimal MCP (Model Context Protocol) client.

MCP is a JSON-RPC 2.0 protocol that lets language-model applications
discover and call tools exposed by an external server process. The
official spec is at https://modelcontextprotocol.io — this module
implements just the subset we need:

* The ``initialize`` / ``notifications/initialized`` handshake.
* ``tools/list`` to enumerate a server's tools.
* ``tools/call`` to invoke a tool.

The transport is **stdio only** (the server is a subprocess; we
talk to it over its stdin/stdout). HTTP/SSE transports exist
but they're uncommon for the local "run a server next to the
agent" use case, and supporting them would pull in
``anyio`` / ``httpx`` / ``sse-starlette``. Stdio covers the
overwhelming majority of MCP servers in the wild.

Why stdlib-only? ``anduril`` keeps ``openai`` as its single
runtime dependency, and the MCP official SDK is heavy. This
implementation is ~250 lines and works for any server that
speaks the stdio transport correctly.

Usage::

    from anduril.mcp_client import MCPServer, discover_mcp_tools
    server = MCPServer(command=["npx", "-y", "@mcp/server-filesystem", "/tmp"])
    tools = discover_mcp_tools([server])

The returned ``Tool`` namedtuples have their ``fn`` attribute
set to a closure that routes the call back through the MCP
client. The standard anduril tool pipeline (schema derivation,
validation, approval gating, sanitization) all works on these
tools unmodified.
"""

from __future__ import annotations

import http.client
import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from anduril.env import _env_int
from anduril.tools import Tool


# === Tunables =============================================================

#: Per-request timeout for an MCP tool call. Generous by default
#: because some servers (e.g. ones that shell out to a database)
#: can take a while. Override with ``ANDURIL_MCP_TIMEOUT``.
DEFAULT_TIMEOUT_S: float = _env_int("ANDURIL_MCP_TIMEOUT", 60) or 60

#: Maximum size (in bytes) of a single JSON-RPC response we'll
#: buffer. 16 MiB is enough for any reasonable tool output;
#: larger responses get truncated with a clear error.
MAX_RESPONSE_BYTES: int = 16 * 1024 * 1024

#: MCP protocol version we announce in the ``initialize`` handshake.
#: Pin to the version this client was tested against; newer
#: servers negotiate down. Update both sides together.
PROTOCOL_VERSION: str = "2024-11-05"

#: Client identity we send to the server. Some servers use this
#: for telemetry / logging.
CLIENT_INFO: dict[str, str] = {
    "name": "anduril",
    "version": "0.1.0",
}


# === Errors ===============================================================


class MCPError(Exception):
    """Base class for MCP client errors."""


class MCPProtocolError(MCPError):
    """The server returned malformed JSON or a non-JSON-RPC response."""


class MCPServerError(MCPError):
    """The server returned a JSON-RPC error response.

    Carries the original error code and message so the caller
    can decide how to surface it.
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"server error {code}: {message}")


class MCPTimeoutError(MCPError):
    """A tool call exceeded the configured timeout."""


# === Subprocess transport =================================================


class _MCPTransport:
    """A single MCP server connection (stdio or HTTP).

    Subclasses implement the wire-specific pieces
    (``_wire_send`` writes a single JSON-RPC payload to
    the transport, ``_wire_close`` shuts the transport
    down). The base class handles the rest: a monotonic
    request-id counter, a future-id → Event map for
    matching responses to requests, a daemon reader
    thread that demuxes incoming JSON-RPC messages.

    JSON-RPC notifications (messages with no ``id``) are
    silently discarded for now — the client only uses
    request/response, and MCP's notification types
    (``notifications/progress``, etc.) are not part of
    our use case.
    """

    def __init__(self, *, name: str = "transport") -> None:
        self._name = name
        self._lock = threading.Lock()
        # Future-id -> Event that fires when the response lands.
        self._waiters: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._closed = False
        # Stderr lines for diagnostic output when a server
        # fails to start. Subclasses populate this; the base
        # exposes ``stderr_text()`` for the caller.
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        # A reader thread that consumes the wire and
        # dispatches messages to the waiter map. Subclasses
        # set ``self._reader_thread`` after starting their
        # own I/O; the base class only owns the start.
        self._reader_thread: threading.Thread | None = None

    def _dispatch_message(self, raw: str) -> None:
        """Parse a single JSON-RPC message and route to waiters.

        Pulled out of ``_read_loop`` so both the stdio and
        HTTP transports can share it. On any non-fatal
        parse error (e.g. a stray line of stdout from a
        misbehaving server) we silently drop the line —
        JSON-RPC spec says implementations should be
        tolerant of noise.
        """
        raw = raw.strip()
        if not raw:
            return
        if len(raw) > MAX_RESPONSE_BYTES:
            self._fail_all(
                MCPProtocolError(
                    f"response line exceeds {MAX_RESPONSE_BYTES} bytes"
                )
            )
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return  # tolerate noise
        if not isinstance(msg, dict):
            return
        msg_id = msg.get("id")
        if not isinstance(msg_id, int):
            return  # notification; we don't use any
        event = self._waiters.get(msg_id)
        if event is None:
            return  # spurious response for an id we no longer care about
        self._responses[msg_id] = msg
        event.set()

    def _fail_all(self, err: Exception) -> None:
        """Wake every pending waiter with the given error.

        Used when the transport dies mid-stream so callers
        don't hang waiting for responses that will never
        arrive. We ``set()`` each event AND pre-populate
        ``_responses`` with a sentinel dict so the request's
        ``pop(msg_id)`` doesn't KeyError on the way out.
        """
        sentinel = {"error": {"code": -1, "message": f"transport closed: {err}"}}
        for msg_id, event in self._waiters.items():
            self._responses.setdefault(msg_id, sentinel)
            event.set()

    def stderr_text(self, *, timeout: float = 0.5) -> str:
        """Read whatever stderr / log lines the server has produced.

        Returns an empty string if the transport doesn't have
        a stderr concept (e.g. HTTP).
        """
        with self._stderr_lock:
            return "\n".join(self._stderr_lines)

    def _next_id(self) -> int:
        cur = getattr(self, "_id_counter", 0)
        self._id_counter = cur + 1
        return cur + 1

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response.

        Raises:
            MCPProtocolError: malformed response.
            MCPServerError: server returned ``{"error": ...}``.
            MCPTimeoutError: no response within ``timeout``.
            MCPError: transport closed / died.
        """
        if self._closed:
            raise MCPError("transport is closed")
        with self._lock:
            msg_id = self._next_id()
            payload = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": method,
            }
            if params is not None:
                payload["params"] = params
            event = threading.Event()
            self._waiters[msg_id] = event
            try:
                self._wire_send(json.dumps(payload).encode("utf-8"))
            except (BrokenPipeError, OSError) as e:
                self._waiters.pop(msg_id, None)
                raise MCPError(f"transport closed: {e}") from e
            if not event.wait(timeout=timeout):
                self._waiters.pop(msg_id, None)
                raise MCPTimeoutError(
                    f"timeout after {timeout}s waiting for {method!r}"
                )
            msg = self._responses.pop(msg_id)
            self._waiters.pop(msg_id, None)
        if "error" in msg:
            err = msg["error"]
            if not isinstance(err, dict):
                raise MCPProtocolError(f"malformed error: {err!r}")
            code = err.get("code", -1)
            message = err.get("message", "(no message)")
            data = err.get("data")
            raise MCPServerError(code, message, data)
        if "result" not in msg:
            raise MCPProtocolError(f"no result in response: {msg!r}")
        return msg["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        with self._lock:
            payload = {
                "jsonrpc": "2.0",
                "method": method,
            }
            if params is not None:
                payload["params"] = params
            try:
                self._wire_send(json.dumps(payload).encode("utf-8"))
            except (BrokenPipeError, OSError) as e:
                raise MCPError(f"transport closed: {e}") from e

    def close(self) -> None:
        """Shut the transport down. Idempotent. Safe to call from cleanup."""
        if self._closed:
            return
        self._closed = True
        try:
            self._wire_close()
        except Exception:
            pass
        # Wake any pending waiters so callers don't hang.
        self._fail_all(MCPError("transport closed"))


class _StdioMCPTransport(_MCPTransport):
    """MCP server via a local subprocess + JSON-RPC over stdio.

    This is the original / canonical transport for local MCP
    servers. The server reads one JSON object per line from
    stdin and writes responses to stdout.
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None,
                 cwd: str | pathlib.Path | None = None) -> None:
        super().__init__(name=command[0] if command else "stdio")
        self._command = list(command)
        # Merge the parent env with the caller's overrides. We
        # explicitly strip the agent-specific vars so a server
        # can't accidentally inherit ANDURIL_* state from a
        # parent process and behave differently in tests vs.
        # production.
        merged = {
            k: v for k, v in os.environ.items()
            if not k.startswith("ANDURIL_")
        }
        if env:
            merged.update(env)
        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged,
                cwd=str(cwd) if cwd else None,
                text=True,
                bufsize=1,  # line-buffered: each \n flushes immediately
                start_new_session=True,
            )
        except FileNotFoundError as e:
            raise MCPError(f"MCP server not found: {e}") from e
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True,
            name=f"mcp-stdio-reader-{self._name}",
        )
        self._reader_thread.start()

    @property
    def pid(self) -> int:
        return self._proc.pid

    def _read_loop(self) -> None:
        try:
            assert self._proc.stdout is not None
            for raw in self._proc.stdout:
                if not raw:
                    break
                self._dispatch_message(raw)
            self._fail_all(MCPError("server closed stdout"))
        except Exception as e:
            self._fail_all(e)

    def _wire_send(self, payload: bytes) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(payload.decode("utf-8") + "\n")
        self._proc.stdin.flush()

    def _wire_close(self) -> None:
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except Exception:
            pass

    # Backwards-compat alias: older code referenced the
    # transport by the bare ``_MCPTransport`` name. Keep the
    # alias so we don't have to rename every reference.
    stderr_text = _MCPTransport.stderr_text


class _HTTPMCPTransport(_MCPTransport):
    """MCP server over HTTP, using the Streamable HTTP transport.

    The 2025-03-26 MCP spec introduces a single HTTP endpoint
    that serves both single-response and streaming modes. The
    client POSTs a JSON-RPC request to the endpoint; the
    server responds with either a single JSON body (content-
    type ``application/json``) or an SSE stream (content-type
    ``text/event-stream``). The transport class below handles
    both: it reads the response headers, and if it's SSE it
    dispatches messages from the event stream; otherwise it
    reads the body as a single message.

    The HTTP transport uses the stdlib ``http.client`` —
    no extra dependencies, no SSL certificate fuss beyond
    the system trust store. Servers that need an
    Authorization header can pass one in ``headers=``;
    per-request headers (e.g. per-call auth) are
    deliberately not supported because the MCP spec models
    session state, not per-request credentials.

    Concurrent requests on the same transport are
    serialized through a lock; we don't support HTTP/2
    stream multiplexing in this minimal client. That's
    fine for the local-tool / one-server-per-process use
    case; a real production client would dispatch multiple
    streams.
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None,
                 timeout: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__(name=url)
        # Validate the URL up-front so ``MCPServer(url=...)`` with
        # a non-http URL fails fast rather than at the first
        # request.
        if not url.startswith(("http://", "https://")):
            raise MCPError(
                f"MCP HTTP URL must be http:// or https://, got {url!r}"
            )
        self._url = url
        self._headers = dict(headers or {})
        # The per-request HTTP timeout. We use the MCP
        # request timeout for both the connect and the
        # read; the SSE stream's per-event reads block
        # indefinitely until the server closes, so the
        # per-request timeout governs when we give up.
        self._http_timeout = timeout
        # The currently open response, if any. SSE
        # responses are streaming; we hold the connection
        # here so ``_read_loop`` can read line-by-line.
        self._conn = None  # http.client.HTTPConnection
        self._response = None  # http.client.HTTPResponse
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True,
            name=f"mcp-http-reader-{self._name}",
        )
        self._reader_thread.start()

    def _parse_url(self) -> tuple[str, int, str, bool]:
        """Split the URL into (host, port, path, https)."""
        from urllib.parse import urlparse
        u = urlparse(self._url)
        if u.scheme not in ("http", "https"):
            raise MCPError(
                f"MCP HTTP URL must be http:// or https://, got {u.scheme!r}"
            )
        return (
            u.hostname or "localhost",
            u.port or (443 if u.scheme == "https" else 80),
            u.path or "/",
            u.scheme == "https",
        )

    def _open_connection(self) -> None:
        """Open (or reuse) the HTTP connection to the server."""
        if self._conn is not None:
            return
        host, port, _, https = self._parse_url()
        if https:
            import ssl
            ctx = ssl.create_default_context()
            self._conn = http.client.HTTPSConnection(
                host, port, timeout=self._http_timeout, context=ctx,
            )
        else:
            self._conn = http.client.HTTPConnection(
                host, port, timeout=self._http_timeout,
            )

    def _wire_send(self, payload: bytes) -> None:
        """POST a single JSON-RPC request and store the response.

        The transport opens one HTTP request per call, holds
        the response open for ``_read_loop`` to consume, and
        blocks the caller (``request()``) on the per-id
        Event. The next call to ``_wire_send`` waits for
        the previous response to finish first; we hold the
        request lock across the whole send-and-wait to
        serialize concurrent callers.
        """
        if self._conn is None:
            self._open_connection()
        assert self._conn is not None
        host, _port, path, _https = self._parse_url()
        headers = dict(self._headers)
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json, text/event-stream"
        headers["Host"] = host
        # We hold the previous response's lock until the
        # caller has consumed the response. This is a
        # simplification: a production client would
        # multiplex, but the spec allows one in-flight
        # request per session, so the simplest correct
        # implementation is the right one.
        try:
            self._conn.request(
                "POST", path, body=payload, headers=headers,
            )
            self._response = self._conn.getresponse()
        except (BrokenPipeError, OSError) as e:
            # Connection died; try once more to recover.
            self._conn = None
            raise MCPError(f"HTTP request failed: {e}") from e
        if self._response.status >= 400:
            body = self._response.read().decode("utf-8", errors="replace")
            # The server might have closed the connection
            # anyway; let the next call reconnect.
            self._conn = None
            raise MCPError(
                f"MCP HTTP server returned HTTP {self._response.status}: "
                f"{body[:200]}"
            )

    def _read_loop(self) -> None:
        """Drain responses as they arrive.

        For each call we read either an SSE stream
        (text/event-stream) or a single JSON body
        (application/json). We dispatch each parsed
        message via ``_dispatch_message``. When a response
        ends, we close the connection so the next call
        opens a fresh one — this matches the simple "one
        in-flight request per session" usage.
        """
        try:
            while not self._closed:
                if self._response is None:
                    time.sleep(0.01)
                    continue
                ctype = self._response.getheader("Content-Type", "")
                if "text/event-stream" in ctype:
                    self._read_sse_stream(self._response)
                else:
                    body = self._response.read().decode("utf-8", errors="replace")
                    self._dispatch_message(body)
                # Response done; close the connection so the
                # next call gets a fresh one.
                try:
                    self._response.close()
                except Exception:
                    pass
                self._response = None
        except Exception as e:
            self._fail_all(e)
        finally:
            self._fail_all(MCPError("HTTP transport closed"))

    def _read_sse_stream(self, response) -> None:
        """Read an SSE stream: dispatch each ``data:`` line as a JSON-RPC msg.

        SSE wire format: lines of the form
        ``event: foo\\ndata: {json}\\n\\n``. The blank line
        terminates an event. Multiple ``data:`` lines
        accumulate into one event (we take the last).
        ``id:`` is ignored; ``retry:`` is ignored.

        ``http.client.HTTPResponse`` yields raw bytes when
        read via ``for line in response``. We decode each
        line as UTF-8 (SSE is always UTF-8) and strip
        ``\\r`` so the parsing below is line-oriented.
        """
        data_buf: list[str] = []
        for raw in response:
            if isinstance(raw, bytes):
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            else:
                line = raw
            # SSE line endings: a blank line separates events.
            if line in ("", "\r", "\n", "\r\n"):
                if data_buf:
                    self._dispatch_message("\n".join(data_buf))
                    data_buf = []
                continue
            line = line.rstrip("\n").rstrip("\r")
            if line.startswith("data:"):
                # Strip the "data:" prefix and at most one
                # leading space (per the SSE spec).
                value = line[len("data:"):]
                if value.startswith(" "):
                    value = value[1:]
                data_buf.append(value)
            # Other fields (event:, id:, retry:) ignored
        # End of stream without a trailing blank line.
        if data_buf:
            self._dispatch_message("\n".join(data_buf))

    def _wire_close(self) -> None:
        try:
            if self._response is not None:
                try:
                    self._response.close()
                except Exception:
                    pass
                self._response = None
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        except Exception:
            pass


# Keep the old name as a backwards-compat alias. Existing
# code (and tests) use ``_MCPTransport`` for the stdio
# transport. New code can use ``_StdioMCPTransport``
# directly; both names refer to the same class now.
_MCPTransport = _StdioMCPTransport


# === Server definition ====================================================


class MCPServer:
    """Configuration for one MCP server.

    The user-facing API: pass a list of these to
    :func:`discover_mcp_tools`. The transport is created
    lazily on the first call.

    Two transports are supported:

    * **stdio** (default): a local subprocess. Pass
      ``command`` (a list, or a string which is shlex-
      split). The subprocess is started on the first
      request and torn down on :meth:`shutdown`.
    * **HTTP**: a remote server. Pass ``url=`` (and
      optional ``headers=`` for auth). The client uses
      the Streamable HTTP transport from the 2025-03-26
      MCP spec (POST + optional SSE response).

    Exactly one of ``command`` or ``url`` must be set.
    Mixing them raises ``ValueError``.

    Attributes:
        name: A short identifier for the server (used in
            tool-name prefixes, so the model can tell which
            server a tool came from).
        command: The stdio command (None for HTTP).
        env: Optional env overrides (stdio only).
        cwd: Optional working directory (stdio only).
        url: The HTTP endpoint (None for stdio).
        headers: Optional HTTP headers (HTTP only).
        prefix: How to prefix tool names. ``"server"``
            (default) gives ``"<name>__<tool>"``; ``"none"``
            gives just the tool name.
    """

    def __init__(
        self,
        name: str,
        command: list[str] | str | None = None,
        *,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | pathlib.Path | None = None,
        prefix: str = "server",
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.name = name
        # Validate the (command, url) combo.
        if (command is None) == (url is None):
            raise ValueError(
                f"MCP server {name!r}: exactly one of "
                f"``command=`` or ``url=`` must be set"
            )
        # Stdio path.
        if command is not None:
            if isinstance(command, str):
                command = shlex.split(command)
            if not command:
                raise ValueError(
                    f"MCP server {name!r}: empty command"
                )
            self.command: list[str] | None = list(command)
            self.env = env
            self.cwd = cwd
            self.url: str | None = None
            self.headers = None
        # HTTP path.
        else:
            if not url or not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"MCP server {name!r}: url must be http(s)://..."
                )
            self.command = None
            self.env = None
            self.cwd = None
            self.url = url
            self.headers = dict(headers or {})
        self.prefix = prefix
        self.timeout = timeout
        self._transport: _MCPTransport | None = None

    def __repr__(self) -> str:
        if self.url is not None:
            return f"MCPServer(name={self.name!r}, url={self.url!r})"
        return f"MCPServer(name={self.name!r}, command={self.command!r})"

    def transport(self) -> _MCPTransport:
        if self._transport is None or self._transport._closed:
            if self.url is not None:
                self._transport = _HTTPMCPTransport(
                    self.url, headers=self.headers,
                    timeout=self.timeout,
                )
            else:
                assert self.command is not None
                self._transport = _StdioMCPTransport(
                    self.command, env=self.env, cwd=self.cwd,
                )
        return self._transport

    def shutdown(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None


# === Public API ===========================================================


def _mcp_schema_to_our_schema(
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Translate an MCP tool's JSON Schema into anduril's tool schema.

    MCP uses the standard JSON Schema with ``type: "object"`` at
    the top and ``properties`` / ``required``. The only common
    transformation needed: ``"$schema"`` and ``"title"`` keys
    that anduril's validator doesn't know about are stripped so
    it doesn't trip on them.
    """
    schema = dict(input_schema) if isinstance(input_schema, dict) else {}
    # Drop JSON-Schema-meta keys that the anduril validator
    # doesn't recognise. ``$schema`` is a URL string; ``title``
    # is just metadata. Anything starting with ``$`` is JSON
    # Schema vocabulary and not part of the validation.
    for k in ("$schema", "title", "$id", "$ref", "$comment"):
        schema.pop(k, None)
    if "type" not in schema:
        # MCP tools may omit ``type`` (defaulting to object).
        # Our validator is fine with no type, but the
        # ``required`` check needs a properties dict, so
        # ensure one exists.
        if "properties" not in schema:
            schema["properties"] = {}
    return schema


def _tool_name_for(server: MCPServer, raw_name: str) -> str:
    """Apply the configured naming policy to an MCP tool name."""
    if server.prefix == "none":
        return raw_name
    # Default: prefix with the server name. Double-underscore
    # is the conventional MCP separator and is unlikely to
    # appear in a real tool name.
    return f"{server.name}__{raw_name}"


def _strip_tool_name_prefix(server: MCPServer, full_name: str) -> str:
    """Inverse of :func:`_tool_name_for`.

    Strips the server's prefix so we can address the original
    tool name on the wire. The leading double-underscore is
    the separator; if the user picked a different prefix
    scheme (``prefix="none"``) this returns ``full_name``
    unchanged.
    """
    if server.prefix == "none":
        return full_name
    prefix = f"{server.name}__"
    if full_name.startswith(prefix):
        return full_name[len(prefix):]
    return full_name


def _make_tool_fn(
    server: MCPServer, raw_name: str,
) -> Callable[..., str]:
    """Build the Python function the anduril tool system will call.

    The wrapper serialises the kwargs as a JSON-RPC
    ``tools/call`` request and unwraps the result. The
    contract: the function returns a string (which the model
    sees as the tool's output). If the server returns a
    non-text content part we stringify it; if the result is
    an error we raise so the standard tool pipeline records
    it as a tool error.

    The wrapper captures ``server`` and ``raw_name`` in a
    closure; the agent calls ``fn(**kwargs)`` and gets back
    the tool's text output.
    """
    def _call(**kwargs: Any) -> str:
        transport = server.transport()
        result = transport.request(
            "tools/call",
            {
                "name": raw_name,
                "arguments": kwargs,
            },
            timeout=server.timeout,
        )
        # The result is a ``CallToolResult`` per the MCP spec:
        # ``{"content": [...], "isError": false}``. We flatten
        # the content list to a single string for the model.
        return _format_call_result(result)

    _call.__name__ = raw_name
    return _call


def _format_call_result(result: dict[str, Any]) -> str:
    """Flatten an MCP ``tools/call`` result into a single string.

    The result has a ``content`` array; each entry is one of:

    * ``{"type": "text", "text": "..."}`` — plain text.
    * ``{"type": "image", ...}`` — base64-encoded image. We
      surface a marker rather than try to embed the bytes
      (anduril's tool pipeline is text-only).
    * ``{"type": "resource", ...}`` — a link to an MCP
      resource. We surface the URI so the model can refer
      to it.

    If the result has ``isError: true``, the content is an
    error message; we still return it as a string, but
    prefixed so the model can see it's a tool failure.
    """
    is_error = bool(result.get("isError"))
    content = result.get("content")
    if not isinstance(content, list):
        # Older / non-conformant servers may return the
        # content as a plain string. Coerce.
        content = [str(content)]
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        ctype = item.get("type")
        if ctype == "text":
            parts.append(str(item.get("text", "")))
        elif ctype == "image":
            mime = item.get("mimeType", "image")
            parts.append(f"[image: {mime}, {len(item.get('data', ''))} b64 chars]")
        elif ctype == "resource":
            res = item.get("resource", {})
            uri = res.get("uri", "?")
            parts.append(f"[resource: {uri}]")
        else:
            parts.append(f"[{ctype}: {item!r}]")
    out = "\n".join(parts)
    if is_error:
        return f"error: {out}"
    return out


def _do_handshake(transport: _MCPTransport) -> dict[str, Any]:
    """Run the MCP ``initialize`` + ``initialized`` handshake.

    Returns the server's ``serverInfo`` dict (name, version).
    Raises :class:`MCPError` if the handshake fails — usually
    because the command isn't a real MCP server.
    """
    result = transport.request(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        },
    )
    # The spec says we must send ``notifications/initialized``
    # to acknowledge. We don't expect a response.
    transport.notify("notifications/initialized")
    if not isinstance(result, dict):
        raise MCPProtocolError(
            f"initialize returned non-dict: {result!r}"
        )
    return result


def _list_tools(transport: _MCPTransport) -> list[dict[str, Any]]:
    """Call ``tools/list`` and return the list of tool dicts."""
    result = transport.request("tools/list", {})
    if not isinstance(result, dict):
        raise MCPProtocolError(
            f"tools/list returned non-dict: {result!r}"
        )
    tools = result.get("tools")
    if not isinstance(tools, list):
        raise MCPProtocolError(
            f"tools/list returned non-list 'tools': {tools!r}"
        )
    return [t for t in tools if isinstance(t, dict)]


def _build_tool(server: MCPServer, raw: dict[str, Any]) -> Tool:
    """Convert one MCP tool dict to a :class:`anduril.tools.Tool`."""
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise MCPProtocolError(f"tool missing 'name': {raw!r}")
    description = raw.get("description", "") or ""
    input_schema = raw.get("inputSchema", {}) or {}
    schema = _mcp_schema_to_our_schema(name, description, input_schema)
    return Tool(
        name=_tool_name_for(server, name),
        description=description,
        parameters=schema,
        fn=_make_tool_fn(server, name),
        dangerous=False,  # MCP tools are sandboxed; the server is responsible for safety
        risk="medium",  # medium because the user didn't vet the server
    )


class _MCPDiscoverResult(list):
    """A list of MCP tools with a stashed ``errors`` dict.

    Plain Python lists don't accept attribute assignment, so
    we use a tiny subclass. The :mod:`anduril.tools` code
    only iterates the result, so the extra attribute is
    invisible to it. Callers (tests, the TUI's ``/mcp``
    command) can read ``result.errors`` to see what went
    wrong per server.
    """

    def __init__(self, tools: list[Tool], errors: dict[str, str] | None = None) -> None:
        super().__init__(tools)
        self.errors: dict[str, str] = errors or {}


def discover_mcp_tools(
    servers: list[MCPServer],
) -> _MCPDiscoverResult:
    """Connect to each server, list its tools, and return them.

    On any per-server failure (handshake error, non-conformant
    server, subprocess crash), the error is swallowed and a
    zero-tool result is returned for that server. The failure
    reason is written to stderr so the user can see it, and
    stashed in ``result.errors[server_name]`` for tests / the
    TUI's ``/mcp`` command.

    This "best-effort" behaviour is deliberate: a single broken
    MCP server should not break the entire agent.
    """
    errors: dict[str, str] = {}
    out: list[Tool] = []
    for server in servers:
        try:
            transport = server.transport()
            _do_handshake(transport)
            for raw in _list_tools(transport):
                try:
                    out.append(_build_tool(server, raw))
                except MCPProtocolError as e:
                    errors[f"{server.name}.{raw.get('name', '?')}"] = str(e)
        except (MCPError, OSError) as e:
            errors[server.name] = f"{type(e).__name__}: {e}"
            # Surface a hint on stderr so the user can debug.
            stderr = server._transport.stderr_text() if server._transport else ""
            msg = f"MCP server {server.name!r} failed: {e}"
            if stderr:
                msg += f"\n  stderr: {stderr[:500]}"
            print(msg, file=sys.stderr)
    return _MCPDiscoverResult(out, errors)  # type: ignore[return-value]


def shutdown_servers(servers: list[MCPServer]) -> None:
    """Shut every server down. Idempotent. Safe to call at exit."""
    for server in servers:
        try:
            server.shutdown()
        except Exception:
            pass


# === Config parsing =======================================================


def _parse_mcp_servers_from_config(config: dict[str, Any]) -> list[MCPServer]:
    """Parse the ``[tool.anduril.mcp_servers]`` table from pyproject.toml.

    The schema is::

        [tool.anduril.mcp_servers.servers.fs]
        command = "npx -y @mcp/server-filesystem /tmp"
        prefix = "server"  # optional, default

        [tool.anduril.mcp_servers.servers.remote]
        url = "https://my-mcp-server.example/mcp"
        headers = {Authorization = "Bearer ..."}

    The section name under ``servers`` becomes the server
    name. Each entry must have either ``command`` (stdio)
    or ``url`` (HTTP); exactly one of the two. ``prefix``,
    ``env``, and ``headers`` are optional.
    """
    servers_section = config.get("servers", {})
    if not isinstance(servers_section, dict):
        return []
    out: list[MCPServer] = []
    for name, spec in servers_section.items():
        if not isinstance(spec, dict):
            continue
        cmd = spec.get("command")
        url = spec.get("url")
        if not cmd and not url:
            continue  # need at least one
        if cmd and url:
            continue  # ambiguous; skip (rather than error)
        env = spec.get("env")
        cwd = spec.get("cwd")
        prefix = spec.get("prefix", "server")
        headers = spec.get("headers")
        if url:
            out.append(MCPServer(
                name=name, url=url, headers=headers, prefix=prefix,
            ))
        else:
            out.append(MCPServer(
                name=name, command=cmd, env=env, cwd=cwd, prefix=prefix,
            ))
    return out


def load_mcp_servers_from_pyproject(
    path: str | pathlib.Path | None = None,
) -> list[MCPServer]:
    """Load MCP server configs from a pyproject.toml.

    Returns an empty list if the file doesn't exist or has no
    ``[tool.anduril.mcp_servers]`` table. We deliberately don't
    pull in ``tomllib`` (3.11+) for this — a tiny hand-rolled
    regex parse of just the relevant section is enough.
    """
    if path is None:
        path = pathlib.Path.cwd() / "pyproject.toml"
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    # We only care about the ``[tool.anduril.mcp_servers]``
    # block. Extract it; everything else is ignored.
    # TOML has no real "end of section" marker — the next
    # ``[...]`` line starts a new section. We extract the
    # contiguous block between the first matching opener and
    # the next top-level section. Sub-section headers like
    # ``[servers.fs]`` are part of the same block, so we
    # don't end the block on them.
    in_section = False
    body_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("["):
            is_ours = stripped.startswith("[tool.anduril.mcp_servers")
            if in_section and not is_ours:
                # Hit a different section; stop collecting.
                in_section = False
                continue
            if is_ours:
                in_section = True
                # Drop the outer section header; keep sub-headers
                # (``[servers.fs]``) for the body parser.
                if not stripped.startswith("[tool.anduril.mcp_servers]"):
                    body_lines.append(stripped)
                continue
            # Some other section, we weren't in ours; skip.
            continue
        if in_section:
            body_lines.append(line)
    if not body_lines:
        return []
    # Hand-parse the body. Format is the same as pyproject's
    # standard tables:
    #
    #   [tool.anduril.mcp_servers.servers.fs]
    #   command = "..."
    #   prefix = "server"
    #
    #   [tool.anduril.mcp_servers.servers.git]
    #   command = "..."
    config: dict[str, Any] = {"servers": {}}
    current_server: dict[str, Any] | None = None
    for line in body_lines:
        if not line.strip():
            continue
        s = line.strip()
        if s.startswith("["):
            # Nested server section, e.g.
            # ``[tool.anduril.mcp_servers.servers.fs]``.
            inner = s.strip("[]").strip()
            # Strip our outer prefix to get the relative form
            # ``servers.fs``; then split off ``servers.`` to get
            # the server name.
            for prefix in ("tool.anduril.mcp_servers.", "tool.anduril.mcp_servers"):
                if inner.startswith(prefix):
                    inner = inner[len(prefix):]
                    break
            if inner.startswith("servers."):
                name = inner[len("servers."):].strip()
                current_server = {}
                config["servers"][name] = current_server
            else:
                current_server = None
        elif "=" in s and current_server is not None:
            key, _, val = s.partition("=")
            key = key.strip()
            val = val.strip()
            # Strip surrounding quotes.
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            current_server[key] = val
    return _parse_mcp_servers_from_config(config)


__all__ = [
    "CLIENT_INFO",
    "DEFAULT_TIMEOUT_S",
    "MCPError",
    "MCPServer",
    "MCPProtocolError",
    "MCPServerError",
    "MCPTimeoutError",
    "PROTOCOL_VERSION",
    "discover_mcp_tools",
    "load_mcp_servers_from_pyproject",
    "shutdown_servers",
]
