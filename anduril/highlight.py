"""Syntax highlighting for the TUI log.

A code block in the log (anything wrapped in triple backticks, with an
optional language tag) is highlighted by this module. Plain text is
returned as a single ``(text, attr)`` span with the default attr so
the caller doesn't need a special case for "no highlight".

Two backends:

* :class:`PygmentsHighlighter` — uses ``pygments`` if installed. High
  quality, slow, has many lexers. Best choice when the user is
  viewing real code.
* :class:`RegexHighlighter` — pure-stdlib fallback. Cheap, but the
  highlighting is approximate. Covers the most common tokens
  (keywords, strings, comments, numbers) for the languages we
  expect to see most: Python, JS/TS, shell/bash, JSON, YAML, Rust,
  Go, C/C++. Anything else falls back to "no highlight".

The dispatcher in :func:`highlight_code` picks the pygments backend
if available, else the regex one. ``pygments`` is *not* a runtime
dependency (we keep "one runtime dep: openai"); it's an optional
extra the user can install with ``pip install pygments``.

The output is a list of ``(text, attr)`` spans, where ``attr`` is a
curses attribute (the caller provides one — see
:func:`make_pygments_formatter`). Spans are *character-aligned* with
the source text: a span's ``text`` concatenates back to the
original. This is what makes wrapping with proper indent possible
downstream.
"""

from __future__ import annotations

import re
from typing import Callable


# === Public API ===========================================================

#: Aliases for the language tag on a fence. The model often writes
#: ``\`\`\`js`` or ``\`\`\`typescript`` or ``\`\`\`sh`` — we accept
#: all the common ones and map to a canonical name used by the
#: highlighter.
LANG_ALIASES: dict[str, str] = {
    "py": "python", "python": "python", "py3": "python",
    "js": "javascript", "javascript": "javascript",
    "jsx": "javascript", "ts": "typescript", "tsx": "typescript",
    "typescript": "typescript",
    "sh": "bash", "bash": "bash", "shell": "bash", "zsh": "bash",
    "console": "bash", "shellscript": "bash",
    "json": "json", "jsonc": "json",
    "yaml": "yaml", "yml": "yaml",
    "html": "html", "xml": "xml",
    "css": "css", "scss": "css",
    "md": "markdown", "markdown": "markdown",
    "rs": "rust", "rust": "rust",
    "go": "go", "golang": "go",
    "c": "c", "h": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp",
    "hpp": "cpp", "hxx": "cpp",
    "java": "java", "kt": "kotlin", "kotlin": "kotlin",
    "rb": "ruby", "ruby": "ruby",
    "sql": "sql",
    "diff": "diff", "patch": "diff",
    "toml": "toml", "ini": "ini",
}


def normalize_lang(tag: str) -> str:
    """Map a fence language tag to a canonical name.

    Unknown tags return ``""`` (the highlighter treats that as
    "no language known, no highlight"). The lookup is case
    insensitive; surrounding whitespace is stripped.
    """
    tag = (tag or "").strip().lower()
    if not tag:
        return ""
    return LANG_ALIASES.get(tag, tag)


def highlight_code(
    text: str,
    lang: str,
    default_attr: int,
    color_for_token: Callable[[str], int],
) -> list[tuple[str, int]]:
    """Return ``[(text, attr), ...]`` spans for ``text`` in language ``lang``.

    ``text`` is the raw code (without the surrounding backticks —
    the caller is responsible for stripping those). ``lang`` is
    already normalised via :func:`normalize_lang`. ``default_attr``
    is the curses attribute to use for "no specific token"; it
    becomes the attr of any span the highlighter can't classify.
    ``color_for_token`` is a callback the caller provides that
    maps a Pygments-style token type to a curses attr
    (``color_for_token("Token.Keyword")`` etc.).

    The returned list concatenates back to ``text``: ``"".join(t
    for t, _ in spans) == text``. This invariant is what lets the
    downstream wrap layer preserve spans across line breaks.

    The dispatcher tries Pygments first (best quality, slow),
    then the regex fallback (faster, lower quality). An empty
    result means "no highlighter claimed this language" — the
    caller should treat it as plain text.
    """
    canonical = normalize_lang(lang)
    if not canonical:
        return [(text, default_attr)]
    spans = _highlight_with_pygments(text, canonical, default_attr,
                                     color_for_token)
    if spans is not None:
        return spans
    spans = _highlight_with_regex(text, canonical, default_attr)
    if spans:
        return spans
    return [(text, default_attr)]


# === Pygments backend ====================================================

def _highlight_with_pygments(
    text: str,
    lang: str,
    default_attr: int,
    color_for_token: Callable[[str], int],
) -> list[tuple[str, int]] | None:
    """Pygments-based highlighter. Returns ``None`` if pygments is
    unavailable or the lexer doesn't load — the caller falls back
    to the regex path."""
    try:
        from pygments.lexers import get_lexer_by_name
    except ImportError:
        return None
    try:
        lexer = get_lexer_by_name(lang)
    except Exception:
        return None
    # Build a per-line mapping. Pygments yields (tokentype, value)
    # pairs in a flat stream; we use the value's text directly and
    # map each token to an attr via ``color_for_token``. Tokens
    # we don't recognise get the default attr (so the output
    # never has a "blank" span that would be invisible).
    out: list[tuple[str, int]] = []
    for tok, value in lexer.get_tokens(text):
        if not value:
            continue
        # ``tok`` is a Token tree, e.g. ``Token.Keyword.Namespace``.
        # We map by the deepest matching ``color_for_token`` that
        # returns a non-default attr, falling back to default for
        # unrecognised types. ``color_for_token`` is expected to be
        # a small mapping defined by the caller; see
        # :func:`make_pygments_formatter` for the default.
        attr = _resolve_token_attr(tok, color_for_token, default_attr)
        out.append((value, attr))
    if not out:
        return None
    return _merge_adjacent_spans(out)


def _resolve_token_attr(
    tok,
    color_for_token: Callable[[str], int],
    default_attr: int,
) -> int:
    """Walk up the Pygments token tree to find an attr.

    ``tok`` is the leaf token (e.g. ``Token.Keyword.Namespace``).
    We try the leaf first, then each ancestor, until
    ``color_for_token`` returns something other than the default
    attr. This lets a small mapping handle the whole family
    (e.g. a single "Keyword" entry covers all of Keyword.*).
    """
    while tok is not None:
        attr = color_for_token(str(tok))
        if attr != default_attr:
            return attr
        tok = tok.parent
    return default_attr


def _merge_adjacent_spans(spans: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Coalesce adjacent spans that share an attr.

    Pygments sometimes emits a single conceptual token as several
    small chunks (e.g. an interpolated string in Python 3.6+ f-strings
    splits into "f", "'", "hello", "'", "{", "name", "}"). Merging
    reduces the span count and the renderer's per-span overhead.
    """
    if not spans:
        return spans
    out: list[tuple[str, int]] = [(spans[0][0], spans[0][1])]
    for text, attr in spans[1:]:
        if out[-1][1] == attr:
            out[-1] = (out[-1][0] + text, attr)
        else:
            out.append((text, attr))
    return out


# === Regex fallback =======================================================

# The regex highlighter covers the common-case tokens for the
# languages the agent sees most. It's deliberately small: any
# misclassification just means the wrong colour for a few chars,
# not a syntax error or a crash. A span's category maps to one of
# these strings; the caller converts to a curses attr.

# Per-language token definitions. Each entry is a list of
# (category, regex) pairs; the highlighter tries them in order on
# each line and emits a span for the first match. Unmatched text
# is emitted as the "default" category. The regexes use the ``re``
# module's longest-match semantics via ``re.finditer`` plus a
# per-position scan.
_LANG_TOKENS: dict[str, list[tuple[str, str]]] = {
    "python": [
        ("comment", r"#[^\n]*"),
        ("string", r'(?:[fFbBrRuU]?(?:"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'))'),
        ("keyword", r"\b(?:and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield|True|False|None)\b"),
        ("number", r"\b\d+(?:\.\d+)?\b"),
        ("builtin", r"\b(?:print|len|range|list|dict|set|tuple|str|int|float|bool|open|isinstance|hasattr|getattr|setattr|type|super|self|cls|Exception|ValueError|TypeError|KeyError|IndexError|ImportError|OSError|IOError|RuntimeError)\b"),
    ],
    "javascript": [
        ("comment", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("string", r'(?:`(?:\\.|[^`\\])*`|\'(?:\\.|[^\'\\\n])*\'|"(?:\\.|[^"\\\n])*")'),
        ("keyword", r"\b(?:var|let|const|function|return|if|else|for|while|do|switch|case|break|continue|new|delete|typeof|instanceof|this|super|class|extends|import|export|from|as|async|await|yield|throw|try|catch|finally|of|in|void|null|undefined|true|false)\b"),
        ("number", r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"),
        ("builtin", r"\b(?:console|window|document|require|module|exports|process|setTimeout|setInterval|clearTimeout|clearInterval|fetch|Promise|Array|Object|String|Number|Boolean|JSON|Math|Date|RegExp|Map|Set|Symbol|Error)\b"),
    ],
    "typescript": [
        ("comment", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("string", r'(?:`(?:\\.|[^`\\])*`|\'(?:\\.|[^\'\\\n])*\'|"(?:\\.|[^"\\\n])*")'),
        ("keyword", r"\b(?:var|let|const|function|return|if|else|for|while|do|switch|case|break|continue|new|delete|typeof|instanceof|this|super|class|extends|import|export|from|as|async|await|yield|throw|try|catch|finally|of|in|void|null|undefined|true|false|interface|type|enum|public|private|protected|readonly|implements|namespace|declare|abstract|keyof|infer|is|asserts)\b"),
        ("number", r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"),
        ("type", r"\b(?:string|number|boolean|any|unknown|never|void|object|Array|Promise|Record|Partial|Required|Readonly|Pick|Omit|Exclude|Extract|ReturnType|InstanceType|this)\b"),
    ],
    "bash": [
        ("comment", r"#[^\n]*"),
        ("string", r'(?:"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\')'),
        ("keyword", r"\b(?:if|then|else|elif|fi|for|while|do|done|case|esac|in|function|return|exit|export|local|readonly|declare|set|unset|alias|source|trap|shift|break|continue)\b"),
        ("builtin", r"\b(?:echo|printf|cd|pwd|ls|cat|grep|sed|awk|find|xargs|sort|uniq|head|tail|wc|tr|cut|test|true|false|read|eval|exec|wait|kill|jobs|fg|bg|history|env|export|set)\b"),
        ("variable", r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?"),
    ],
    "json": [
        ("string", r'"(?:\\.|[^"\\\n])*"'),
        ("number", r"-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"),
        ("literal", r"\b(?:true|false|null)\b"),
    ],
    "yaml": [
        ("comment", r"#[^\n]*"),
        ("string", r"""(?:'(?:[^'\n]|'')*'|"(?:\\.|[^"\\])*")"""),
        ("literal", r"\b(?:true|false|yes|no|null|~)\b"),
        ("number", r"\b\d+(?:\.\d+)?\b"),
        ("keyword", r"^[ \t]*-[ \t]"),
    ],
    "rust": [
        ("comment", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("string", r'r#?(?:"(?:\\.|[^"\\\n])*")*#?'),
        ("keyword", r"\b(?:fn|let|mut|const|static|pub|use|mod|struct|enum|impl|trait|for|while|loop|if|else|match|return|break|continue|as|in|where|self|Self|super|crate|async|await|unsafe|move|true|false|None|Some|Ok|Err)\b"),
        ("number", r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|f32|f64)?\b"),
    ],
    "go": [
        ("comment", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("string", r'(?:`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\')'),
        ("keyword", r"\b(?:func|var|const|type|struct|interface|map|chan|package|import|return|if|else|for|range|switch|case|default|break|continue|go|defer|select|fallthrough|true|false|nil|iota)\b"),
        ("number", r"\b\d+(?:\.\d+)?\b"),
    ],
    "c": [
        ("comment", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("string", r'"(?:\\.|[^"\\\n])*"'),
        ("keyword", r"\b(?:if|else|for|while|do|switch|case|default|break|continue|return|goto|sizeof|typedef|struct|union|enum|static|extern|const|volatile|register|auto|signed|unsigned|short|long|int|char|float|double|void|inline|restrict|_Bool|_Complex|_Imaginary|NULL|true|false)\b"),
        ("number", r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:[fFlLuU]+)?\b"),
        ("type", r"\b(?:int8|int16|int32|int64|uint8|uint16|uint32|uint64|size_t|ssize_t|ptrdiff_t|intptr_t|uintptr_t|bool|true|false)\b"),
    ],
    "cpp": [],  # falls through to the c rules (good enough)
    "java": [
        ("comment", r"//[^\n]*|/\*[\s\S]*?\*/"),
        ("string", r'"(?:\\.|[^"\\\n])*"'),
        ("keyword", r"\b(?:if|else|for|while|do|switch|case|default|break|continue|return|class|interface|extends|implements|public|private|protected|static|final|abstract|void|int|long|short|byte|float|double|char|boolean|new|this|super|try|catch|finally|throw|throws|import|package|instanceof|null|true|false)\b"),
        ("number", r"\b\d+(?:\.\d+)?(?:[fFdDlL])?\b"),
    ],
    "diff": [
        ("diff_meta", r"^---[^\n]*|^\+\+\+[^\n]*|^\d+,\d+[^\n]*"),
        ("diff_added", r"^\+[^\n]*"),
        ("diff_removed", r"^-[^\n]*"),
    ],
    "toml": [
        ("comment", r"#[^\n]*"),
        ("string", r'''(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|"""[\s\S]*?""")'''),
        ("literal", r"\b(?:true|false)\b"),
        ("number", r"\b\d+(?:\.\d+)?\b"),
    ],
    "ini": [
        ("comment", r";[^\n]*|#[^\n]*"),
        ("section", r"^\[[^\]\n]+\]"),
        ("string", r'"(?:\\.|[^"\\\n])*"'),
        ("number", r"\b\d+(?:\.\d+)?\b"),
    ],
    "markdown": [
        ("comment", r"<!--[\s\S]*?-->"),
        ("string", r"```[^\n]*|~~~[^\n]*"),
        ("keyword", r"^#{1,6}[ \t].*$|^\s*[-*+][ \t]|^\s*\d+\.[ \t]|^>\s"),
    ],
    "html": [
        ("comment", r"<!--[\s\S]*?-->"),
        ("tag", r"</?[A-Za-z][A-Za-z0-9-]*(?:\s+[^>]*)?/?>"),
        ("string", r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''),
    ],
    "xml": [
        ("comment", r"<!--[\s\S]*?-->"),
        ("tag", r"</?[A-Za-z][A-Za-z0-9-]*(?:\s+[^>]*)?/?>|<\?[\s\S]*?\?>|<!\[CDATA\[[\s\S]*?\]\]>"),
        ("string", r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''),
    ],
    "css": [
        ("comment", r"/\*[\s\S]*?\*/"),
        ("tag", r"@[A-Za-z-]+|#[A-Za-z][A-Za-z0-9-]*|\.[A-Za-z][A-Za-z0-9-]*"),
        ("string", r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''),
        ("number", r"\b\d+(?:\.\d+)?(?:px|em|rem|%|vh|vw|s|ms|deg|rad)?\b"),
    ],
}


def _highlight_with_regex(
    text: str,
    lang: str,
    default_attr: int,
) -> list[tuple[str, int]]:
    """Tokenise ``text`` with the regex rules for ``lang``.

    Returns a list of ``(text, attr)`` spans. ``default_attr`` is
    used for any text we can't classify (we don't have a category
    for it). Returns ``[]`` for unknown languages so the caller
    can fall back to plain text.
    """
    rules = _LANG_TOKENS.get(lang) or _LANG_TOKENS.get(_canonical_to_regex_lang(lang))
    if not rules:
        return []
    compiled = [(cat, re.compile(pat, re.MULTILINE)) for cat, pat in rules]
    out: list[tuple[str, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        # Find the earliest match across all rules at this
        # position. A single forward pass per position is fine
        # because each rule's regex is anchored at ``pos`` via
        # ``match`` semantics (the rules use ``\A`` or character
        # classes that don't have an implicit ``^`` so they
        # match anywhere).
        best_end = -1
        best_cat: str | None = None
        best_start = pos
        for cat, regex in compiled:
            m = regex.match(text, pos)
            if m and m.end() > best_end:
                best_end = m.end()
                best_cat = cat
                best_start = m.start()
        if best_cat is None or best_end <= pos:
            # No match at this position — emit one char and move on.
            out.append((text[pos], default_attr))
            pos += 1
            continue
        # Emit any gap before the match as default.
        if best_start > pos:
            out.append((text[pos:best_start], default_attr))
        out.append((text[best_start:best_end], default_attr))
        pos = best_end
    return _merge_adjacent_spans(out)


def _canonical_to_regex_lang(canonical: str) -> str:
    """Map a canonical pygments-style name to a regex backend key.

    The regex backend is a small subset. We accept pygments-style
    names and silently fall back to plain text for anything we
    don't have rules for.
    """
    mapping = {
        "python": "python", "python3": "python",
        "javascript": "javascript", "typescript": "typescript",
        "bash": "bash", "sh": "bash", "shell": "bash",
        "json": "json", "json5": "json",
        "yaml": "yaml",
        "rust": "rust", "rs": "rust",
        "go": "go", "golang": "go",
        "c": "c", "h": "c",
        "cpp": "cpp", "c++": "cpp",
        "java": "java", "kotlin": "java",
        "ruby": "yaml", "rb": "yaml",  # crude: yaml rules cover most of ruby too
        "html": "html", "xml": "xml", "css": "css",
        "diff": "diff", "patch": "diff",
        "toml": "toml", "ini": "ini",
        "markdown": "markdown", "md": "markdown",
    }
    return mapping.get(canonical, "")


__all__ = [
    "LANG_ALIASES",
    "highlight_code",
    "normalize_lang",
]
