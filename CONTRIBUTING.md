# Contributing

Thanks for your interest in anduril! This is a small project, so the
guidelines are short.

## Quick start

```bash
# Fork + clone, then:
make install   # create venv with uv + install dev + web extras
make test      # run the test suite
```

## Pull requests

1. **One logical change per PR.** Refactors + features + drive-by
   reformatting in the same PR make the diff unreviewable.
2. **Tests first.** If you're fixing a bug, write a failing test
   that demonstrates it. If you're adding a feature, write the
   test that proves it works. PRs without tests are usually
   bounced.
3. **Lint clean.** `make lint` must pass. If ruff format
   complains, `make format` fixes it.
4. **All tests pass.** `make test` (or `make ci` for the full
   loop including build) must be green on your local machine
   before you push. CI runs the same checks on Linux / Python
   3.11–3.13.

## Commit messages

A loose conventional-commits style:

```
short-verb noun: terse summary (50 chars or so)

Optional longer explanation of the *why* — what was the
motivation, what alternatives were considered, what
followup work this opens up.

Refs #1234 if there's an issue.
```

Examples:

* `sessions: add lightweight metadata index for sub-ms listing`
* `agent: don't double-submit user message on Ctrl-C`
* `tools: split file tools out of bash for safer edits`

## Code style

* `ruff` is the source of truth. Don't fight the formatter; let
  it do its thing.
* Imports are sorted by ruff (I001). The `from __future__ import
  annotations` header is the first line of every module.
* Public functions get a docstring. The docstring's first line
  is the one-line summary; the rest explains the *why*, not
  the *what* (the what is already in the signature).
* Use the existing patterns:
  * Tools: `@tool` decorator with a `risk="low|medium|high"`
    if `dangerous=True`.
  * Skills: drop a Python file with a `tools = [...]` list
    into `~/.local/share/anduril/skills/`.
  * New env vars: use the `_env_int` / `_env_float` / `_env_str`
    / `_env_bool` helpers from `anduril.env` for the defaults.

## Architecture

* `anduril/agent.py` is the model turn loop. The streaming
  loop, tool-call aggregation, retry nudges, and context
  compression all live here. Touch this file carefully — most
  regressions are subtle.
* `anduril/tools.py` is the tool system. The `@tool`
  decorator derives JSON Schema from type hints; the
  `Tool` namedtuple carries name + description + parameters
  + fn + dangerous + risk.
* `anduril/tui.py` is the curses TUI. It's the biggest file
  by far (~3000 lines). The render loop is hot — anything
  that runs per-token needs to be cheap.
* `anduril/sessions.py` is session persistence. Sessions are
  JSON files in `~/.local/state/anduril/sessions/` with a
  metadata index at `<sessions_dir>/_index.json` for fast
  listing.
* `anduril/context.py` is the context-window sizing and
  auto-compression trigger. Pure functions only — no I/O.
* `anduril/files.py` is the `@`-mention picker: file
  scanning, fuzzy matching, image / text file expansion.
* `anduril/skills.py` is the skill discovery and runtime
  registration.
* `anduril/metrics.py` is the cumulative token-usage
  tracker that the status bar reads.
* `anduril/env.py` is the env-var helpers. Pure stdlib.
* `anduril/cli.py` is the argparse CLI. `_build_agent`
  builds the default agent; `tui(agent)` is the REPL.

## Filing issues

For bugs: include the model name, the `anduril` version
(`pip show anduril` or `git rev-parse HEAD`), and a minimal
reproduction (the prompt + the unexpected output).

For feature requests: one paragraph on the use case is
usually enough. PRs with a working implementation beat PRs
with a long design doc.
