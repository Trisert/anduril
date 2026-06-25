---
name: web
description: Free, no-API-key web search and page fetching. Use when you need to look something up on the web, find URLs, or read the contents of a public page.
---

# web — search and fetch

Free web capabilities for anduril. No API keys, no accounts, no rate-limit surprises.

## Setup

The Nix dev shell already includes `ddgs`, `httpx`, and `trafilatura`, so
if you use `nix develop` you're done. Otherwise install with:

```bash
pip install ddgs httpx trafilatura
```

Then enable the skill (one of):

```bash
# Option A: symlink (changes here reflect on next reload)
ln -s "$(pwd)/anduril/contrib/skills/web" ~/.local/share/anduril/skills/web

# Option B: copy
cp -r anduril/contrib/skills/web ~/.local/share/anduril/skills/
```

The skill is discovered from `~/.local/share/anduril/skills/` and any directories in `$ANDURIL_SKILLS_PATH` (colon-separated). Restart `anduril` to pick it up.

## Tools

### `web_search(query, n=10, backend="auto")`

Multi-backend search. `ddgs` aggregates results from DuckDuckGo, Bing, Brave, Google, Mojeek, Mullvad-Brave, Mullvad-Google, Wikipedia, and others. No keys required for any of them.

```python
web_search("python async tutorial")               # 10 results, auto backend
web_search("rust ownership", n=20, backend="bing") # 20 results from Bing only
web_search("...", backend="brave,duckduckgo")      # try Brave, fall back to DDG
```

Output is a plain-text list, one hit per block:

```
Search results for 'python async tutorial' — 8 hits:

1. Async IO in Python: A Complete Walkthrough
   https://realpython.com/async-io-python/
   Learn how to use async/await in Python with this comprehensive…

2. ...
```

### `fetch_content(url, max_chars=12000)`

Fetch a URL and return the page's main readable text, boilerplate stripped. Uses `httpx` for the request and `trafilatura` for extraction.

```python
fetch_content("https://en.wikipedia.org/wiki/Python_(programming_language)")
```

For JavaScript-heavy SPAs, extraction may come back empty — the tool surfaces a clear hint in that case. (`trafilatura` cannot run JS; for that you'd need a real headless browser.)

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Skill doesn't appear in tool list | `ddgs`/`httpx`/`trafilatura` not installed | `pip install ddgs httpx trafilatura` |
| `web_search` returns no results | Backend blocked the request or rate-limited | Try a different `backend` (e.g. `backend="brave"`) |
| `fetch_content` returns "could not extract readable content" | SPA / JavaScript-rendered page | Use a headless browser (out of scope for this skill) |
| `fetch_content` returns "error fetching" | Network issue, 4xx/5xx, redirect loop | Check the URL, retry, or use a different domain |

## Notes

- The skill is a thin wrapper — its value is the curated set of free, no-key backends. If you want higher-quality results, swap `ddgs` for Tavily, Brave Search API, or another provider (still no key changes for the tool signature).
- All requests use a realistic Firefox User-Agent by default. Some sites still block datacenter IPs; rotate or use a proxy if you hit that.
- No scraping of `robots.txt` is performed. Be a good citizen.
