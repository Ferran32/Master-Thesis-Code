"""Step 0 validation gate: recompute the confusion matrix of all 28
configurations from the prediction files and check against the thesis
tables (n=872, 439 positives). Run once; result cached to
results/validation_gate.json and reproduced in SUMMARY.md.

Run: .venv/bin/python3 analysis/statistics/validate_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIG_DISPLAY, MODEL_DISPLAY, MODELS, STRATEGIES, all_config_paths, load_predictions  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    ("llama3-8b", "zero_shot"): (125, 132, 314, 301),
    ("llama3-8b", "few_shot_1shot"): (86, 78, 353, 355),
    ("llama3-8b", "few_shot_3shot"): (175, 167, 264, 266),
    ("llama3-8b", "cot"): (151, 144, 288, 289),
    ("qwen2.5-7b", "zero_shot"): (1, 1, 438, 432),
    ("qwen2.5-7b", "few_shot_1shot"): (16, 17, 423, 416),
    ("qwen2.5-7b", "few_shot_3shot"): (46, 50, 393, 383),
    ("qwen2.5-7b", "cot"): (5, 3, 434, 430),
    ("deepseek-coder-6.7b", "zero_shot"): (57, 48, 382, 385),
    ("deepseek-coder-6.7b", "few_shot_1shot"): (294, 277, 145, 156),
    ("deepseek-coder-6.7b", "few_shot_3shot"): (246, 239, 193, 194),
    ("deepseek-coder-6.7b", "cot"): (76, 83, 363, 350),
    ("codellama-7b", "zero_shot"): (224, 206, 215, 227),
    ("codellama-7b", "few_shot_1shot"): (287, 293, 152, 140),
    ("codellama-7b", "few_shot_3shot"): (226, 221, 213, 212),
    ("codellama-7b", "cot"): (253, 242, 186, 191),
    ("llama3-8b", "sft"): (130, 116, 309, 317),
    ("qwen2.5-7b", "sft"): (159, 154, 280, 279),
    ("deepseek-coder-6.7b", "sft"): (273, 277, 166, 156),
    ("codellama-7b", "sft"): (238, 231, 201, 202),
    ("llama3-8b", "skill_v9"): (199, 193, 240, 240),
    ("qwen2.5-7b", "skill_v9"): (15, 16, 424, 417),
    ("deepseek-coder-6.7b", "skill_v9"): (182, 173, 257, 260),
    ("codellama-7b", "skill_v9"): (183, 175, 256, 258),
    ("llama3-8b", "skill_v6"): (200, 198, 239, 235),
    ("qwen2.5-7b", "skill_v6"): (10, 11, 429, 422),
    ("deepseek-coder-6.7b", "skill_v6"): (137, 139, 302, 294),
    ("codellama-7b", "skill_v6"): (190, 169, 249, 264),
}


def main():
    paths = all_config_paths()
    results = []
    all_labels_by_cfg = {}
    all_ok = True
    for (model, cfg), path in paths.items():
        if not path.exists():
            results.append({"model": model, "config": cfg, "path": str(path), "error": "MISSING"})
            all_ok = False
            continue
        rows = load_predictions(path)
        tp = fp = fn = tn = parse_fail = 0
        labels = {}
        for r in rows:
            y = r["label"]
            ps = r.get("parse_success", True)
            if not ps:
                parse_fail += 1
                pred = 0
            else:
                pred = r["predicted_label"]
            labels[r["idx"]] = y
            if y == 1 and pred == 1:
                tp += 1
            elif y == 0 and pred == 1:
                fp += 1
            elif y == 1 and pred == 0:
                fn += 1
            else:
                tn += 1
        all_labels_by_cfg[(model, cfg)] = labels
        exp = EXPECTED.get((model, cfg))
        match = None
        if exp:
            match = all(abs(a - b) <= 1 for a, b in zip((tp, fp, fn, tn), exp))
            all_ok = all_ok and match
        results.append({
            "model": model, "model_display": MODEL_DISPLAY[model], "config": cfg,
            "config_display": CONFIG_DISPLAY.get(cfg, cfg), "path": str(path.relative_to(Path.cwd())),
            "n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "parse_failures": parse_fail,
            "expected": exp, "match": match,
        })

    ref_key = next(iter(all_labels_by_cfg))
    ref = all_labels_by_cfg[ref_key]
    labels_identical = all(labels == ref for labels in all_labels_by_cfg.values())

    out = {
        "n_configs": len(results),
        "all_match": all_ok,
        "ground_truth_identical_across_all_28": labels_identical,
        "rows": results,
    }
    with open(OUT_DIR / "validation_gate.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"All {len(results)} configs match expected confusion matrices (within +-1): {all_ok}")
    print(f"Ground-truth labels identical across all 28 configs (aligned on idx): {labels_identical}")


if __name__ == "__main__":
    main()
