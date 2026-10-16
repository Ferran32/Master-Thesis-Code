"""A deliberately simple C/C++ lexer for near-duplicate detection.

Not a full preprocessor/parser -- just enough to turn normalised source into
a token stream where identifiers, keywords, literals and punctuation are
distinguishable, so that 3-gram shingles capture structure rather than raw
character runs. Comments/whitespace are assumed already stripped by
common.normalize() (the thesis's own SHA-1 normalisation function), but the
regex here also strips any that remain (e.g. in the train/val `text` column
which stores raw, non-normalised code in some rows).
"""
from __future__ import annotations

import re
from typing import List

_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)

_TOKEN_RE = re.compile(
    r"""
    (?P<string>  "(?:\\.|[^"\\])*"       )  |
    (?P<char>    '(?:\\.|[^'\\])*'       )  |
    (?P<number>  0[xX][0-9a-fA-F]+[uUlL]* | \d+\.\d+([eE][+-]?\d+)?[fFlL]? | \d+[uUlL]* )  |
    (?P<ident>   [A-Za-z_][A-Za-z0-9_]*  )  |
    (?P<op>      ->|\+\+|--|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||::|
                 [+\-*/%=<>!&|^~?:;,.(){}\[\]]
    )
    """,
    re.VERBOSE,
)


def tokenize(code: str) -> List[str]:
    """Return a flat token list: identifiers and keywords keep their literal
    text as separate tokens (so a renamed-variable copy is NOT automatically
    treated as identical, per the task spec listing identifiers/keywords as
    distinct categories); numeric and string/char literals are abstracted to
    `NUM`/`STR`/`CHAR` (literal values such as buffer sizes or format strings
    are the most common cosmetic diff between otherwise-identical functions
    across projects, and abstracting them is standard in clone detection);
    operators/punctuation are kept verbatim.
    """
    code = _COMMENT_RE.sub(" ", code)
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer(code):
        kind = m.lastgroup
        text = m.group()
        if kind == "ident":
            tokens.append(text)
        elif kind == "number":
            tokens.append("NUM")
        elif kind == "string":
            tokens.append("STR")
        elif kind == "char":
            tokens.append("CHAR")
        else:
            tokens.append(text)
    return tokens


def shingles(tokens: List[str], n: int = 3) -> set:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
