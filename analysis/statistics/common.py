"""Shared loading/alignment helpers for the examiner-revision analyses.

Read-only with respect to the rest of the repository: this module never
writes to results/, data/, or any other tracked file. It only loads the
existing MVFSC test set, train/val splits, and the 28 prediction files
that the thesis already produced, and aligns them on `idx`.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 42

MODELS = ["llama3-8b", "qwen2.5-7b", "deepseek-coder-6.7b", "codellama-7b"]
STRATEGIES = ["zero_shot", "few_shot_1shot", "few_shot_3shot", "cot"]

# The 6 "headline" configs used in Part 2, in a fixed order.
HEADLINE_CONFIGS = ["zero_shot", "few_shot_1shot", "few_shot_3shot", "cot", "sft", "skill_v9"]

MODEL_DISPLAY = {
    "llama3-8b": "Llama 3 8B Instruct",
    "qwen2.5-7b": "Qwen2.5 7B Instruct",
    "deepseek-coder-6.7b": "DeepSeek-Coder 6.7B Instruct",
    "codellama-7b": "Code Llama 7B Instruct",
}
CONFIG_DISPLAY = {
    "zero_shot": "zero-shot",
    "few_shot_1shot": "few-shot k=1",
    "few_shot_3shot": "few-shot k=3",
    "cot": "CoT",
    "sft": "SFT",
    "skill_v9": "skill v9",
    "skill_v6": "skill v6",
}

# --- Reproduced VERBATIM from src/data_utils.py (do not edit; "reuse as-is"). ---
_COMMENT_RE = re.compile(
    r"""
    //.*?$                           |
    /\*.*?\*/                        |
    ^\s*\#\s*(include|define|if|ifdef|ifndef|elif|else|endif).*?$
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)
_WS_RE = re.compile(r"\s+")


def normalize(code: str) -> str:
    """Identical to src.data_utils._normalize."""
    if not code:
        return ""
    code = _COMMENT_RE.sub("", code)
    return _WS_RE.sub(" ", code).strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def base_eval_path(model: str, strat: str) -> Path:
    return (
        REPO_ROOT
        / "results/base-eval"
        / model
        / strat
        / "base_eval"
        / f"base_eval_{model}_{strat}"
        / "predictions.jsonl"
    )


def sft_path(model: str) -> Path:
    return REPO_ROOT / "results/sft" / model / "predictions.jsonl"


def skill_path(version: str, model: str) -> Path:
    return REPO_ROOT / f"results/skill/{version}_test" / model / "predictions.jsonl"


def all_config_paths() -> Dict[Tuple[str, str], Path]:
    """Map (model, config) -> predictions.jsonl path for all 28 configurations."""
    paths: Dict[Tuple[str, str], Path] = {}
    for m in MODELS:
        for s in STRATEGIES:
            paths[(m, s)] = base_eval_path(m, s)
        paths[(m, "sft")] = sft_path(m)
        paths[(m, "skill_v9")] = skill_path("v9", m)
        paths[(m, "skill_v6")] = skill_path("v6", m)
    return paths


def load_predictions(path: Path) -> List[dict]:
    """Load a predictions.jsonl file, sorted by idx. Parse failures are kept
    as-is (parse_success=False); callers apply the thesis convention
    (parse failure -> predicted negative) explicitly, since different
    analyses need the raw flag as well as the collapsed prediction."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["idx"])
    return rows


def predicted_label(row: dict) -> int:
    """Thesis convention: parse failure -> negative (0)."""
    if not row.get("parse_success", True):
        return 0
    return int(row["predicted_label"])


def load_mvfsc_test():
    """The 872-row MVFSC test set (cached HF dataset at data/splits/mivia_test),
    row order == idx used by every predictions.jsonl file (verified in Step 0)."""
    return load_from_disk(str(REPO_ROOT / "data/splits/mivia_test"))


def load_train_val():
    train = load_from_disk(str(REPO_ROOT / "data/splits/train"))
    val = load_from_disk(str(REPO_ROOT / "data/splits/val"))
    return train, val
