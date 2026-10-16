"""Part 2: paired significance tests between strategies, per model, on the
shared 872-row MVFSC test set.

Run: .venv/bin/python3 analysis/statistics/paired_tests.py

Outputs (analysis/statistics/results/):
  pt_pairwise_60.csv         - all 60 within-model headline-config comparisons
  pt_v9_vs_v6.csv            - the 4 skill-version-selection comparisons
  pt_highlight.json          - the two thesis-cited comparisons, paired vs unpaired
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONFIG_DISPLAY,
    HEADLINE_CONFIGS,
    MODEL_DISPLAY,
    MODELS,
    SEED,
    all_config_paths,
    load_predictions,
    predicted_label,
)
from metrics import confusion_from_pairs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
RNG = np.random.default_rng(SEED)


def load_all():
    paths = all_config_paths()
    data = {}
    for (model, cfg), path in paths.items():
        rows = load_predictions(path)
        labels = np.array([r["label"] for r in rows])
        preds = np.array([predicted_label(r) for r in rows])
        correct = (preds == labels).astype(int)
        data[(model, cfg)] = {"labels": labels, "preds": preds, "correct": correct}
    return data


def f2_precision(labels: np.ndarray, preds: np.ndarray):
    """Vectorised confusion -> (F2, precision, ppr); matches metrics.Metrics exactly."""
    n = len(labels)
    tp = np.sum((labels == 1) & (preds == 1))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    ppr = (tp + fp) / n if n else float("nan")
    if precision != precision or recall != recall:
        f2 = float("nan")
    else:
        beta2 = 4.0
        denom = beta2 * precision + recall
        f2 = (1 + beta2) * precision * recall / denom if denom else float("nan")
    return f2, precision, ppr


def f2_precision_batch(labels_b: np.ndarray, preds_b: np.ndarray):
    """Same as f2_precision but on (n_boot, n) arrays -> (n_boot,) arrays."""
    tp = np.sum((labels_b == 1) & (preds_b == 1), axis=1).astype(float)
    fp = np.sum((labels_b == 0) & (preds_b == 1), axis=1).astype(float)
    fn = np.sum((labels_b == 1) & (preds_b == 0), axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        denom = 4.0 * precision + recall
        f2 = np.where(denom > 0, 5.0 * precision * recall / denom, np.nan)
    return f2, precision


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray):
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))  # A right, B wrong
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))  # A wrong, B right
    n = b + c
    if n == 0:
        p = 1.0
    else:
        res = stats.binomtest(b, n, 0.5, alternative="two-sided")
        p = res.pvalue
    return b, c, p


def paired_bootstrap_diff(labels, preds_a, preds_b, n_boot=N_BOOT, rng=RNG):
    """Resample the 872 sample indices jointly (same resample applied to both
    configs), vectorised across all n_boot resamples at once."""
    n = len(labels)
    idx = rng.integers(0, n, size=(n_boot, n))
    la = labels[idx]
    pa = preds_a[idx]
    pb = preds_b[idx]
    f2a, preca = f2_precision_batch(la, pa)
    f2b, precb = f2_precision_batch(la, pb)
    f2_diffs = f2a - f2b
    prec_diffs = preca - precb
    f2_lo, f2_hi = np.nanpercentile(f2_diffs, [2.5, 97.5])
    p_lo, p_hi = np.nanpercentile(prec_diffs, [2.5, 97.5])
    return (f2_lo, f2_hi), (p_lo, p_hi)


def holm_bonferroni(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def run_pairwise(data):
    rows = []
    for model in MODELS:
        pairs = list(itertools.combinations(HEADLINE_CONFIGS, 2))
        pvals_this_model = []
        row_idx_this_model = []
        for cfg_a, cfg_b in pairs:
            da = data[(model, cfg_a)]
            db = data[(model, cfg_b)]
            labels = da["labels"]
            assert np.array_equal(labels, db["labels"]), "labels must align"

            b, c, p = mcnemar_exact(da["correct"], db["correct"])
            (f2_lo, f2_hi), (p_lo, p_hi) = paired_bootstrap_diff(labels, da["preds"], db["preds"])
            f2_a, prec_a, ppr_a = f2_precision(labels, da["preds"])
            f2_b, prec_b, ppr_b = f2_precision(labels, db["preds"])

            row = {
                "model": model, "model_display": MODEL_DISPLAY[model],
                "config_a": cfg_a, "config_b": cfg_b,
                "config_a_display": CONFIG_DISPLAY[cfg_a], "config_b_display": CONFIG_DISPLAY[cfg_b],
                "f2_a": f2_a, "f2_b": f2_b, "f2_diff": f2_a - f2_b,
                "f2_diff_ci_lo": f2_lo, "f2_diff_ci_hi": f2_hi,
                "f2_diff_ci_excludes_zero": bool(f2_lo > 0 or f2_hi < 0),
                "precision_a": prec_a, "precision_b": prec_b, "precision_diff": prec_a - prec_b,
                "precision_diff_ci_lo": p_lo, "precision_diff_ci_hi": p_hi,
                "ppr_a": ppr_a, "ppr_b": ppr_b, "ppr_diff": ppr_a - ppr_b,
                "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
            }
            rows.append(row)
            pvals_this_model.append(p)
            row_idx_this_model.append(len(rows) - 1)

        adjusted = holm_bonferroni(np.array(pvals_this_model))
        for ridx, adj_p in zip(row_idx_this_model, adjusted):
            rows[ridx]["mcnemar_p_holm"] = adj_p

    return pd.DataFrame(rows)


def run_v9_vs_v6(data):
    """Kept for backward compatibility: the 4 skill_v9-vs-skill_v6 rows only
    (a subset of run_v6_supplement's 24 rows)."""
    rows = []
    for model in MODELS:
        da = data[(model, "skill_v9")]
        db = data[(model, "skill_v6")]
        labels = da["labels"]
        b, c, p = mcnemar_exact(da["correct"], db["correct"])
        (f2_lo, f2_hi), (p_lo, p_hi) = paired_bootstrap_diff(labels, da["preds"], db["preds"])
        f2_a, prec_a, ppr_a = f2_precision(labels, da["preds"])
        f2_b, prec_b, ppr_b = f2_precision(labels, db["preds"])
        rows.append({
            "model": model, "model_display": MODEL_DISPLAY[model],
            "config_a": "skill_v9", "config_b": "skill_v6",
            "f2_v9": f2_a, "f2_v6": f2_b, "f2_diff": f2_a - f2_b,
            "f2_diff_ci_lo": f2_lo, "f2_diff_ci_hi": f2_hi,
            "precision_v9": prec_a, "precision_v6": prec_b, "precision_diff": prec_a - prec_b,
            "precision_diff_ci_lo": p_lo, "precision_diff_ci_hi": p_hi,
            "ppr_v9": ppr_a, "ppr_v6": ppr_b, "ppr_diff": ppr_a - ppr_b,
            "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
        })
    df = pd.DataFrame(rows)
    df["mcnemar_p_holm_within_v9v6"] = holm_bonferroni(df["mcnemar_p"].to_numpy())
    return df


def run_v6_supplement(data):
    """skill_v6 against each of the other 6 headline configs (incl. skill_v9),
    per model: 4 models x 6 = 24 comparisons (config_a is always skill_v6)."""
    rows = []
    for model in MODELS:
        others = [c for c in HEADLINE_CONFIGS if c != "skill_v6"] + ["skill_v9"]
        # HEADLINE_CONFIGS already ends with skill_v9, not skill_v6, so the
        # "other 6" are exactly HEADLINE_CONFIGS (zero_shot..skill_v9).
        others = list(HEADLINE_CONFIGS)
        da = data[(model, "skill_v6")]
        labels = da["labels"]
        pvals_this_model = []
        row_idx_this_model = []
        for cfg_b in others:
            db = data[(model, cfg_b)]
            assert np.array_equal(labels, db["labels"])
            b, c, p = mcnemar_exact(da["correct"], db["correct"])
            (f2_lo, f2_hi), (p_lo, p_hi) = paired_bootstrap_diff(labels, da["preds"], db["preds"])
            f2_a, prec_a, ppr_a = f2_precision(labels, da["preds"])
            f2_b, prec_b, ppr_b = f2_precision(labels, db["preds"])
            rows.append({
                "model": model, "model_display": MODEL_DISPLAY[model],
                "config_a": "skill_v6", "config_b": cfg_b,
                "config_b_display": CONFIG_DISPLAY[cfg_b],
                "f2_a": f2_a, "f2_b": f2_b, "f2_diff": f2_a - f2_b,
                "f2_diff_ci_lo": f2_lo, "f2_diff_ci_hi": f2_hi,
                "f2_diff_ci_excludes_zero": bool(f2_lo > 0 or f2_hi < 0),
                "precision_a": prec_a, "precision_b": prec_b, "precision_diff": prec_a - prec_b,
                "precision_diff_ci_lo": p_lo, "precision_diff_ci_hi": p_hi,
                "ppr_a": ppr_a, "ppr_b": ppr_b, "ppr_diff": ppr_a - ppr_b,
                "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
            })
            pvals_this_model.append(p)
            row_idx_this_model.append(len(rows) - 1)
        adjusted = holm_bonferroni(np.array(pvals_this_model))
        for ridx, adj_p in zip(row_idx_this_model, adjusted):
            rows[ridx]["mcnemar_p_holm_within_v6_supplement"] = adj_p
    return pd.DataFrame(rows)


def main():
    print("Loading 28 prediction files (28 of which supply the 24 headline cells)...")
    data = load_all()

    print(f"Running {len(MODELS)} x C(6,2)=15 = {len(MODELS)*15} pairwise comparisons "
          f"(McNemar exact + paired bootstrap, B={N_BOOT}, seed={SEED}) ...")
    df_pairwise = run_pairwise(data)
    df_pairwise.to_csv(OUT_DIR / "pt_pairwise_60.csv", index=False)

    print("Running v9-vs-v6 comparisons (4) ...")
    df_v9v6 = run_v9_vs_v6(data)
    df_v9v6.to_csv(OUT_DIR / "pt_v9_vs_v6.csv", index=False)

    print("Running v6-supplement comparisons (24: v6 vs each of the other 6 headline configs) ...")
    df_v6supp = run_v6_supplement(data)
    df_v6supp.to_csv(OUT_DIR / "pt_v6_supplement.csv", index=False)

    # Highlight: Llama3 skill_v9 vs few_shot_3shot, and skill_v9 vs sft
    highlight = {}
    for cfg_b, label in [("few_shot_3shot", "v9_vs_fewshot3"), ("sft", "v9_vs_sft")]:
        row = df_pairwise[
            (df_pairwise["model"] == "llama3-8b")
            & (df_pairwise["config_a"] == "skill_v9")
            & (df_pairwise["config_b"] == cfg_b)
        ]
        if row.empty:
            row = df_pairwise[
                (df_pairwise["model"] == "llama3-8b")
                & (df_pairwise["config_b"] == "skill_v9")
                & (df_pairwise["config_a"] == cfg_b)
            ]
            sign = -1
        else:
            sign = 1
        r = row.iloc[0].to_dict()
        highlight[label] = {
            "paired_f2_diff": sign * r["f2_diff"],
            "paired_f2_ci": [sign * r["f2_diff_ci_hi"], sign * r["f2_diff_ci_lo"]] if sign == -1 else [r["f2_diff_ci_lo"], r["f2_diff_ci_hi"]],
            "mcnemar_b": r["mcnemar_b"], "mcnemar_c": r["mcnemar_c"],
            "mcnemar_p": r["mcnemar_p"], "mcnemar_p_holm": r["mcnemar_p_holm"],
        }
    import json
    with open(OUT_DIR / "pt_highlight.json", "w") as f:
        json.dump(highlight, f, indent=2, default=str)

    print("Done. See analysis/statistics/results/pt_*.csv and pt_highlight.json")


if __name__ == "__main__":
    main()
