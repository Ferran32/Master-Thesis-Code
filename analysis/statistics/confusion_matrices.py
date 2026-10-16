"""Task 1: confusion matrices + full metrics for all 28 configurations
(4 models x {zero-shot, few-shot k=1, few-shot k=3, CoT, FT/SFT, skill v9,
skill v6}), cross-checked against the thesis's currently-published F2 values
where one exists (tolerance 0.001).

Run: .venv/bin/python3 analysis/statistics/confusion_matrices.py
Output: analysis/statistics/results/confusion_matrices.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG_DISPLAY, MODEL_DISPLAY, MODELS, all_config_paths, load_predictions, predicted_label  # noqa: E402
from metrics import confusion_from_pairs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CONFIGS = ["zero_shot", "few_shot_1shot", "few_shot_3shot", "cot", "sft", "skill_v9", "skill_v6"]

# Thesis-published F2 values this task gives us directly to check against
# (Prompt_for_code_agent.md CONFIGURATIONS block + CLAUDE.md "Results (final)").
# Only configs with a literal, currently-cited number are listed; base-eval
# (zero-shot/few-shot/CoT) per-config F2 values are not individually quoted
# anywhere in the thesis text available to this task (only "precision
# ceiling ~0.50-0.53" qualitatively), so those are left unchecked.
THESIS_F2 = {
    ("llama3-8b", "sft"): 0.325,
    ("qwen2.5-7b", "sft"): 0.384,
    ("deepseek-coder-6.7b", "sft"): 0.592,
    ("codellama-7b", "sft"): 0.535,
    # "best version per model" from CLAUDE.md; recorded against whichever of
    # v9/v6 CLAUDE.md names as that model's best.
    ("llama3-8b", "skill_v6"): 0.464,
    ("qwen2.5-7b", "skill_v9"): 0.042,
    ("deepseek-coder-6.7b", "skill_v9"): 0.431,
    ("codellama-7b", "skill_v6"): 0.449,
}
THESIS_PR = {
    ("llama3-8b", "sft"): (0.529, 0.296),
    ("qwen2.5-7b", "sft"): (0.508, 0.362),
    ("deepseek-coder-6.7b", "sft"): (0.496, 0.622),
    ("codellama-7b", "sft"): (0.508, 0.542),
}


def mcc_of(tp, fp, fn, tn):
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den else float("nan")


def main():
    paths = all_config_paths()
    rows = []
    for model in MODELS:
        for cfg in ALL_CONFIGS:
            path = paths[(model, cfg)]
            rows_raw = load_predictions(path)
            labels = [r["label"] for r in rows_raw]
            preds = [predicted_label(r) for r in rows_raw]
            parse_fail = sum(1 for r in rows_raw if not r.get("parse_success", True))
            m = confusion_from_pairs(labels, preds)
            mcc = mcc_of(m.tp, m.fp, m.fn, m.tn)
            f2_thesis = THESIS_F2.get((model, cfg))
            f2_flag = None
            if f2_thesis is not None:
                f2_flag = abs(m.f2 - f2_thesis) > 0.001
            pr_thesis = THESIS_PR.get((model, cfg))
            pr_flag = None
            if pr_thesis is not None:
                pr_flag = (abs(m.precision - pr_thesis[0]) > 0.001) or (abs(m.recall - pr_thesis[1]) > 0.001)
            rows.append({
                "model": model, "model_display": MODEL_DISPLAY[model],
                "config": cfg, "config_display": CONFIG_DISPLAY.get(cfg, cfg),
                "path": str(path.relative_to(Path(__file__).resolve().parents[2])),
                "n": m.n, "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                "parse_failures": parse_fail,
                "ppr": m.ppr, "precision": m.precision, "recall": m.recall,
                "f1": m.f1, "f2": m.f2, "accuracy": m.accuracy, "mcc": mcc,
                "thesis_f2": f2_thesis, "f2_mismatch_gt_0.001": f2_flag,
                "thesis_precision_recall": pr_thesis, "pr_mismatch_gt_0.001": pr_flag,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "confusion_matrices.csv", index=False)

    n_checked = df["f2_mismatch_gt_0.001"].notna().sum()
    n_mismatch = (df["f2_mismatch_gt_0.001"] == True).sum()  # noqa: E712
    print(f"28 configs written to {OUT_DIR / 'confusion_matrices.csv'}")
    print(f"F2 checked against a thesis-cited value for {n_checked}/28 configs; mismatches > 0.001: {n_mismatch}")
    if n_mismatch:
        print(df[df["f2_mismatch_gt_0.001"] == True][["model", "config", "f2", "thesis_f2"]])  # noqa: E712
    n_pr_checked = df["pr_mismatch_gt_0.001"].notna().sum()
    n_pr_mismatch = (df["pr_mismatch_gt_0.001"] == True).sum()  # noqa: E712
    print(f"P/R checked for {n_pr_checked}/28 configs; mismatches > 0.001: {n_pr_mismatch}")


if __name__ == "__main__":
    main()
