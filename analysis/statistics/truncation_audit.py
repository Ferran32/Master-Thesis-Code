"""Task 5b: how many of the 872 MVFSC prompts exceed max_seq, per model
tokenizer, for each prompt strategy; and how many CoT outputs hit
max_new_tokens=512 before a parseable JSON verdict.

No model inference or forward pass: only tokenizer.encode() (counting),
using the locally cached HF tokenizers, and the already-saved
predictions.jsonl files for the CoT output-length check.

Run: .venv/bin/python3 analysis/statistics/truncation_audit.py
Output: analysis/statistics/results/truncation_audit.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_DISPLAY, MODELS, REPO_ROOT, all_config_paths, load_mvfsc_test, load_predictions  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from src.data_utils import load_few_shot_pool  # noqa: E402
from src.prompt_template import build_zero_shot_prompt, build_cot_prompt, build_few_shot_prompt  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HF_IDS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
}

MAX_SEQ = {"zero_shot": 4096, "cot": 4096, "few_shot_1shot": 8192, "few_shot_3shot": 8192}


def main():
    mvfsc = load_mvfsc_test()
    codes = list(mvfsc["text"])
    pool = load_few_shot_pool()

    prompts_by_strategy = {
        "zero_shot": [build_zero_shot_prompt(c) for c in codes],
        "cot": [build_cot_prompt(c) for c in codes],
        "few_shot_1shot": [build_few_shot_prompt(c, pool, n=1) for c in codes],
        "few_shot_3shot": [build_few_shot_prompt(c, pool, n=3) for c in codes],
    }

    rows = []
    for model in MODELS:
        print(f"Loading tokenizer for {model} ({HF_IDS[model]}) ...")
        tok = AutoTokenizer.from_pretrained(HF_IDS[model])
        for strat, prompts in prompts_by_strategy.items():
            limit = MAX_SEQ[strat]
            lens = [len(tok.encode(p, add_special_tokens=True)) for p in prompts]
            n_exceed = sum(1 for l in lens if l > limit)
            rows.append({
                "model": model, "model_display": MODEL_DISPLAY[model],
                "strategy": strat, "max_seq": limit,
                "n_exceeding": n_exceed, "share_exceeding": n_exceed / len(lens),
                "min_tokens": min(lens), "max_tokens": max(lens),
                "mean_tokens": sum(lens) / len(lens),
            })
            print(f"  {strat}: {n_exceed}/872 exceed {limit} tokens (max {max(lens)})")
        del tok

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "truncation_audit.csv", index=False)

    # CoT outputs: how many hit max_new_tokens=512 without a parseable JSON verdict.
    # No finish_reason/token-count field is stored in predictions.jsonl, so this
    # re-tokenizes the saved `raw_output` text with the same model's tokenizer as
    # a proxy: a parse failure whose raw_output re-tokenizes to >=505 tokens
    # (within 7 of the 512 cap) is consistent with truncation before a verdict,
    # though this is a re-tokenization of saved text, not the original generation
    # token stream, so it is an estimate, not a verified finish_reason.
    cot_rows = []
    paths = all_config_paths()
    for model in MODELS:
        print(f"Re-tokenizing CoT parse failures for {model} ...")
        tok = AutoTokenizer.from_pretrained(HF_IDS[model])
        preds = load_predictions(paths[(model, "cot")])
        fails = [r for r in preds if not r.get("parse_success", True)]
        lens = [len(tok.encode(r.get("raw_output", "") or "", add_special_tokens=False)) for r in fails]
        near_cap = sum(1 for l in lens if l >= 505)
        cot_rows.append({
            "model": model, "model_display": MODEL_DISPLAY[model],
            "n_parse_failures_cot": len(fails),
            "n_near_512_cap_proxy": near_cap,
            "max_raw_output_tokens": max(lens) if lens else None,
        })
        del tok
    df_cot = pd.DataFrame(cot_rows)
    df_cot.to_csv(OUT_DIR / "cot_parse_failure_audit.csv", index=False)
    print("\nCoT parse-failure audit (proxy for max_new_tokens=512 truncation; see docstring caveat):")
    print(df_cot)


if __name__ == "__main__":
    main()
