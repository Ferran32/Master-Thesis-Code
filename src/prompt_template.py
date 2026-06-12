from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from src.config import SEED

MAX_EXAMPLE_CHARS = 2000

TASK_INSTRUCTION = """\
You are a security vulnerability detector. Analyze the provided code and respond with ONLY a single JSON object — no markdown fences, no commentary, no code, no extra text.

You MUST always include every field below. Use empty strings for fields that do not apply.

Required JSON schema:
{"vulnerable": <bool>, "cwe_id": <string>, "vulnerable_code": <string>, "explanation": <string>, "exploit": <string>, "patch": <string>}

Example output for vulnerable code:
{"vulnerable": true, "cwe_id": "CWE-787", "vulnerable_code": "memcpy(buf, src, len);", "explanation": "Out-of-bounds write due to unchecked len.", "exploit": "Attacker supplies len > sizeof(buf) to overwrite adjacent memory.", "patch": "Add bounds check: if (len > sizeof(buf)) return -1;"}

Example output for safe code:
{"vulnerable": false, "cwe_id": "", "vulnerable_code": "", "explanation": "No vulnerability found.", "exploit": "", "patch": ""}

RULES:
- Output ONLY the JSON object, nothing else.
- Do NOT wrap the JSON in markdown code fences or backticks.
- Do NOT include any text before or after the JSON.
- The "vulnerable" field MUST be a JSON boolean (true or false)."""

COT_INSTRUCTION = """\
Before producing the JSON output, reason step-by-step:
1. Identify potentially unsafe operations (e.g., memory access, input handling, API misuse).
2. Trace data flow from untrusted sources to sensitive sinks.
3. Evaluate whether proper validation, bounds checking, or sanitization is in place.
4. Determine if the code can be exploited and describe the exploitation scenario.
5. Conclude whether the code is vulnerable or safe.

After your reasoning, output ONLY the JSON object on the final line. Do NOT wrap it in markdown fences."""


def _format_code_block(code: str) -> str:
    return f"```\n{code.strip()}\n```"


def _format_example(code: str, label_json: Dict[str, Any]) -> str:
    return (
        f"Code:\n{_format_code_block(code)}\n\n"
        f"Analysis:\n```json\n{json.dumps(label_json, indent=2)}\n```"
    )


def build_zero_shot_prompt(code: str) -> str:
    return f"{TASK_INSTRUCTION}\n\nCode:\n{_format_code_block(code)}"


def _filter_examples(
    examples: List[Tuple[str, Dict[str, Any]]],
    max_chars: int = MAX_EXAMPLE_CHARS,
) -> List[Tuple[str, Dict[str, Any]]]:
    return [(c, l) for c, l in examples if len(c) <= max_chars]


def build_few_shot_prompt(
    code: str,
    examples: List[Tuple[str, Dict[str, Any]]],
    n: int = 3,
) -> str:
    pool = _filter_examples(examples)
    rng = random.Random(SEED)
    selected = rng.sample(pool, min(n, len(pool)))
    examples_text = "\n\n---\n\n".join(
        f"Example {i + 1}:\n{_format_example(ex_code, ex_label)}"
        for i, (ex_code, ex_label) in enumerate(selected)
    )
    return (
        f"{TASK_INSTRUCTION}\n\n"
        f"Here are some examples of vulnerability analysis:\n\n"
        f"{examples_text}\n\n---\n\n"
        f"Now analyze the following code:\n{_format_code_block(code)}"
    )


def build_cot_prompt(code: str) -> str:
    return (
        f"{TASK_INSTRUCTION}\n\n{COT_INSTRUCTION}\n\n"
        f"Code:\n{_format_code_block(code)}"
    )


def build_sft_messages(
    code: str,
    label_json: Optional[Dict[str, Any]] = None,
    user_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a security expert specializing in source code "
                "vulnerability analysis. You analyze code and return "
                "structured JSON vulnerability reports."
            ),
        },
        {"role": "user", "content": user_prompt if user_prompt is not None else build_zero_shot_prompt(code)},
    ]
    if label_json is not None:
        messages.append({
            "role": "assistant",
            "content": json.dumps(label_json, indent=2),
        })
    return messages


def build_skill_prompt(code: str, skill_text: str) -> str:
    return (
        f"You are a security expert. Use the following systematic procedure "
        f"for your analysis:\n\n"
        f"<skill>\n{skill_text.strip()}\n</skill>\n\n"
        f"{TASK_INSTRUCTION}\n\n"
        f"Code:\n{_format_code_block(code)}"
    )


def build_skill_few_shot_prompt(
    code: str,
    skill_text: str,
    examples: List[Tuple[str, Dict[str, Any]]],
    n: int = 3,
) -> str:
    pool = _filter_examples(examples)
    rng = random.Random(SEED)
    selected = rng.sample(pool, min(n, len(pool)))
    examples_text = "\n\n---\n\n".join(
        f"Example {i + 1}:\n{_format_example(ex_code, ex_label)}"
        for i, (ex_code, ex_label) in enumerate(selected)
    )
    return (
        f"You are a security expert. Use the following systematic procedure "
        f"for your analysis:\n\n"
        f"<skill>\n{skill_text.strip()}\n</skill>\n\n"
        f"{TASK_INSTRUCTION}\n\n"
        f"Here are some examples of vulnerability analysis:\n\n"
        f"{examples_text}\n\n---\n\n"
        f"Now analyze the following code:\n{_format_code_block(code)}"
    )


