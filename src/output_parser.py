from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

_TRUTHY = {"true", "yes", "1", "vulnerable"}
_FALSY = {"false", "no", "0", "safe", "not vulnerable", "non-vulnerable"}

_BPE_ARTIFACTS = str.maketrans({
    "\u0120": " ",   # Ġ -> space
    "\u010a": "\n",  # Ċ -> newline
    "\u0109": "\t",  # ĉ -> tab
})


def _clean_bpe(text: str) -> str:
    return text.translate(_BPE_ARTIFACTS)


def extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text or not raw_text.strip():
        return None

    for text in (raw_text, _clean_bpe(raw_text)):
        # Fenced blocks: try last match first (CoT puts JSON at the end)
        fenced_matches = list(_FENCED_JSON_RE.finditer(text))
        for match in reversed(fenced_matches):
            parsed = _try_parse(match.group(1).strip())
            if parsed is not None:
                return parsed

        bare_matches = list(_BARE_JSON_RE.finditer(text))
        for match in reversed(bare_matches):
            parsed = _try_parse(match.group(0).strip())
            if parsed is not None:
                return parsed

        parsed = _try_parse(text.strip())
        if parsed is not None:
            return parsed

    return None


def _try_parse(text: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text).replace("'", '"')
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    normalized = re.sub(r'([{,]\s*)"?([a-zA-Z_]\w*)"?\s*:', r'\1"\2":', cleaned)
    try:
        obj = json.loads(normalized)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def parse_prediction(raw_text: str) -> Tuple[int, Optional[Dict[str, Any]]]:
    parsed = extract_json(raw_text)

    if parsed is not None:
        vuln_value = parsed.get("vulnerable")

        if isinstance(vuln_value, bool):
            return (1 if vuln_value else 0), parsed
        if isinstance(vuln_value, (int, float)):
            return (1 if vuln_value else 0), parsed
        if isinstance(vuln_value, str):
            lower = vuln_value.strip().lower()
            if lower in _TRUTHY:
                return 1, parsed
            if lower in _FALSY:
                return 0, parsed

    m = re.search(r'"?vulnerable"?\s*:\s*(true|false)', _clean_bpe(raw_text).lower())
    if m:
        return (1 if m.group(1) == "true" else 0), parsed

    return -1, parsed


def parse_cwe(parsed_json: Optional[Dict[str, Any]]) -> Optional[str]:
    if parsed_json is None:
        return None
    cwe = parsed_json.get("cwe_id") or parsed_json.get("cwe")
    if cwe is None:
        return None
    cwe = str(cwe).strip().upper()
    m = re.match(r"(CWE-?)?\s*(\d+)", cwe)
    if m:
        return f"CWE-{m.group(2)}"
    return cwe if cwe.startswith("CWE-") else None
