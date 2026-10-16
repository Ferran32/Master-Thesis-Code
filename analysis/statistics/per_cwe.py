"""Task 5c: CWE-class distribution + per-CWE recall/FN for all 28 configs.

Run: .venv/bin/python3 analysis/statistics/per_cwe.py
Outputs (analysis/statistics/results/):
  per_cwe.csv              - n, tp, fn, recall per (model, config, cwe_class)
  cwe_distribution.csv     - n_all, n_vulnerable per CWE class over the 872 MVFSC functions
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONFIG_DISPLAY,
    MODEL_DISPLAY,
    MODELS,
    all_config_paths,
    load_mvfsc_test,
    load_predictions,
    predicted_label,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALL_CONFIGS = ["zero_shot", "few_shot_1shot", "few_shot_3shot", "cot", "sft", "skill_v9", "skill_v6"]
PHASE1_CONFIGS = ["zero_shot", "few_shot_1shot", "few_shot_3shot", "cot"]


def main():
    mvfsc = load_mvfsc_test()
    cwe_by_idx = {i: c for i, c in enumerate(mvfsc["cwe_id"])}
    label_by_idx = {i: l for i, l in enumerate(mvfsc["label"])}

    dist_rows = []
    cwe_order = sorted(set(cwe_by_idx.values()))
    for c in cwe_order:
        n_all = sum(1 for i, cc in cwe_by_idx.items() if cc == c)
        n_vuln = sum(1 for i, cc in cwe_by_idx.items() if cc == c and label_by_idx[i] == 1)
        dist_rows.append({"cwe_class": c, "n_functions": n_all, "n_vulnerable": n_vuln})
    pd.DataFrame(dist_rows).to_csv(OUT_DIR / "cwe_distribution.csv", index=False)

    paths = all_config_paths()
    per_cwe_rows = []
    fn_phase1_accum = {c: 0 for c in cwe_order}
    for model in MODELS:
        for cfg in ALL_CONFIGS:
            rows = load_predictions(paths[(model, cfg)])
            by_idx = {r["idx"]: r for r in rows}
            for c in cwe_order:
                idxs = [i for i, cc in cwe_by_idx.items() if cc == c and label_by_idx[i] == 1]
                n = len(idxs)
                tp = 0
                fn = 0
                for i in idxs:
                    pred = predicted_label(by_idx[i])
                    if pred == 1:
                        tp += 1
                    else:
                        fn += 1
                recall = tp / n if n else float("nan")
                per_cwe_rows.append({
                    "model": model, "model_display": MODEL_DISPLAY[model],
                    "config": cfg, "config_display": CONFIG_DISPLAY.get(cfg, cfg),
                    "cwe_class": c, "n": n, "tp": tp, "fn": fn, "recall": recall,
                })
                if cfg in PHASE1_CONFIGS:
                    fn_phase1_accum[c] += fn

    df = pd.DataFrame(per_cwe_rows)
    df.to_csv(OUT_DIR / "per_cwe.csv", index=False)

    print("CWE distribution (872 functions / 439 vulnerable):")
    print(pd.DataFrame(dist_rows).to_string(index=False))
    print("\nTotal FN per CWE class, summed across the 16 Phase 1 configs (4 models x 4 strategies):")
    for c, v in sorted(fn_phase1_accum.items()):
        print(f"  {c}: {v}")
    pd.DataFrame(
        [{"cwe_class": c, "total_fn_across_16_phase1_configs": v} for c, v in fn_phase1_accum.items()]
    ).to_csv(OUT_DIR / "per_cwe_fn_phase1_totals.csv", index=False)


if __name__ == "__main__":
    main()
