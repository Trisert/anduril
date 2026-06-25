"""web — anduril skill: free, no-API-key web search and page fetching.

Install dependencies with::

    pip install ddgs httpx trafilatura

Then enable the skill by symlinking or copying this directory into
``~/.local/share/anduril/skills/web/`` (or any directory listed in
``$ANDURIL_SKILLS_PATH``). The skill will be picked up on the next
``anduril`` startup.

Tools provided
--------------

* :func:`web_search` — multi-backend search via ``ddgs`` (DDG, Bing,
  Brave, Google, …). No API keys needed.
* :func:`fetch_content` — fetch a URL and extract readable text via
  ``httpx`` + ``trafilatura``. Handles redirects, normalizes HTML.
"""

from __future__ import annotations

try:
    import ddgs
    import httpx
    import trafilatura
except ImportError as e:  # pragma: no cover
    # Raising ImportError here lets the loader emit a clean one-line
    # install hint on stderr instead of a full traceback.
    raise ImportError(
        "missing required packages: 'ddgs', 'httpx', 'trafilatura' "
        "(install with: pip install ddgs httpx trafilatura)"
    ) from e

from anduril.tools import tool


# A realistic UA — trafilatura and some sites silently 403 on default
# Python UAs.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)


@tool
def web_search(query: str, n: int = 10, backend: str = "auto") -> str:
    """Search the web and return a formatted list of results.

    Uses the ``ddgs`` library to query multiple search backends
    (DuckDuckGo, Bing, Brave, Google, Mojeek, Wikipedia, …) — no API
    keys required. Each backend has different strengths; ``"auto"``
    lets ``ddgs`` pick across them.

    :param query: Search query.
    :param n: Maximum number of results to return (default 10).
    :param backend: Backend name (``"auto"``, ``"bing"``, ``"duckduckgo"``,
        ``"brave"``, ``"google"``, ``"wikipedia"``, …) or a comma-separated
        list of backends to try in order.
    """
    backends = [b.strip() for b in backend.split(",") if b.strip()]
    last_err: Exception | None = None
    results: list[dict] = []
    # If the user pinned a single backend, use it directly. Otherwise
    # try "auto" first (covers most cases) and fall back to per-backend
    # attempts if that returns nothing.
    attempts: list[list[str]] = []
    if len(backends) == 1:
        attempts.append(backends)
    else:
        attempts.append(backends or None)  # type: ignore[arg-type]
    with ddgs.DDGS() as client:
        if attempts == [[]]:
            results = list(client.text(query, max_results=n))
        else:
            for backend_list in attempts:
                for b in backend_list or [None]:
                    try:
                        kwargs: dict = {"max_results": n}
                        if b is not None:
                            kwargs["backend"] = b
                        results = list(client.text(query, **kwargs))
                        if results:
                            break
                    except Exception as e:  # backend may be down/empty
                        last_err = e
                        continue
                if results:
                    break
    if not results:
        msg = f"no results for {query!r}"
        if last_err is not None:
            msg += f" (last error: {type(last_err).__name__}: {last_err})"
        return msg
    lines = [f"Search results for {query!r} — {len(results)} hits:\n"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        href = (r.get("href") or r.get("url") or "").strip()
        body = (r.get("body") or r.get("snippet") or "").strip()
        lines.append(f"{i}. {title}")
        if href:
            lines.append(f"   {href}")
        if body:
            # Trim very long snippets to keep tool results bounded.
            if len(body) > 400:
                body = body[:397] + "…"
            lines.append(f"   {body}")
        lines.append("")
    return "\n".join(lines).rstrip()


@tool
def fetch_content(url: str, max_chars: int = 12000) -> str:
    """Fetch a URL and return the page content as readable text.

    Uses ``httpx`` to download and ``trafilatura`` to extract the main
    readable content (boilerplate stripped). Works on most blogs,
    documentation sites, and Wikipedia. For JavaScript-heavy SPAs, the
    extraction may be empty — fall back to a real headless browser
    in that case.

    :param url: URL to fetch.
    :param max_chars: Maximum number of characters to return
        (default 12000). The text is head-truncated past this limit.
    """
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    try:
        response = httpx.get(
            url, headers=headers, follow_redirects=True, timeout=30.0
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return f"error fetching {url}: {type(e).__name__}: {e}"
    html = response.text
    extracted = trafilatura.extract(
        html,
        include_links=True,
        include_formatting=True,
        favor_recall=True,
        with_metadata=True,
    )
    if not extracted:
        # Empty extraction usually means SPA / JS-rendered. Surface a
        # clear hint rather than an empty string.
        return (
            f"could not extract readable content from {url}\n"
            f"(page may be JavaScript-rendered; try a different URL or "
            f"a headless browser)"
        )
    if len(extracted) > max_chars:
        extracted = (
            extracted[:max_chars]
            + f"\n\n[truncated at {max_chars} chars — full content not shown]"
        )
    return extracted


tools = [web_search, fetch_content]
