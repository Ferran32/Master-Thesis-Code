"""Task 3: project-disjoint re-scoring (no retraining).

Splits the 872 MVFSC functions into a DISJOINT subset (13 projects whose
name does not appear, case-insensitively, among train+val project names)
and a SHARED subset (18 projects that do appear), replicating the matching
rule already used in near_duplicates.py (dynamic intersection, not a
hardcoded list). Recomputes all 28 configs' metrics on each subset, with
bootstrap CIs, a binomial precision test against the subset's own base
rate, F2_rand, and per-model ROC convex hull area.

Run: .venv/bin/python3 analysis/statistics/project_disjoint.py
Outputs (analysis/statistics/results/):
  pd_project_list.csv        - all 31 projects, n functions, n vulnerable, disjoint/shared
  pd_subset_totals.csv       - totals for disjoint vs shared subset
  pd_metrics.csv             - 28 configs x {disjoint, shared}: full metrics + CIs + tests
  pd_roc_hull.csv            - per-model convex hull area (disjoint subset)
  pd_near_dup_in_disjoint.json - near-dup audit restricted to the disjoint subset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONFIG_DISPLAY,
    MODEL_DISPLAY,
    MODELS,
    SEED,
    STRATEGIES,
    all_config_paths,
    load_mvfsc_test,
    load_predictions,
    load_train_val,
    predicted_label,
)
from metrics import confusion_from_pairs  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000
ALL_CONFIGS = STRATEGIES + ["sft", "skill_v9", "skill_v6"]


def f2_of(p, r):
    if p != p or r != r:
        return float("nan")
    denom = 4.0 * p + r
    return 5.0 * p * r / denom if denom else float("nan")


def mcc_of(tp, fp, fn, tn):
    num = tp * tn - fp * fn
    denom = np.sqrt(float((tp + fp)) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / denom if denom else float("nan")


def main():
    print("[1/6] Loading MVFSC test set + train/val project names ...")
    mvfsc = load_mvfsc_test()
    train, val = load_train_val()

    mvfsc_projects = [str(p) for p in mvfsc["project"]]
    corpus_projects_lower = set(
        str(p).lower() for p in list(train["project"]) + list(val["project"]) if p
    )

    # Per-MVFSC-project stats.
    df_m = pd.DataFrame({
        "idx": range(len(mvfsc)),
        "project": mvfsc_projects,
        "label": list(mvfsc["label"]),
    })
    proj_stats = (
        df_m.groupby("project")
        .agg(n_functions=("label", "size"), n_vulnerable=("label", "sum"))
        .reset_index()
    )
    proj_stats["in_training_corpus"] = proj_stats["project"].str.lower().isin(corpus_projects_lower)
    proj_stats = proj_stats.sort_values("project").reset_index(drop=True)
    proj_stats.to_csv(OUT_DIR / "pd_project_list.csv", index=False)

    disjoint_projects = set(proj_stats.loc[~proj_stats["in_training_corpus"], "project"])
    shared_projects = set(proj_stats.loc[proj_stats["in_training_corpus"], "project"])
    print(f"  {len(shared_projects)} shared projects (by name), {len(disjoint_projects)} disjoint projects")
    print(f"  shared: {sorted(shared_projects)}")
    print(f"  disjoint: {sorted(disjoint_projects)}")

    disjoint_idx = set(df_m.loc[df_m["project"].isin(disjoint_projects), "idx"])
    shared_idx = set(df_m.loc[df_m["project"].isin(shared_projects), "idx"])
    assert disjoint_idx | shared_idx == set(range(len(mvfsc)))
    assert not (disjoint_idx & shared_idx)

    def subset_totals(idx_set, name):
        sub = df_m[df_m["idx"].isin(idx_set)]
        n = len(sub)
        nv = int(sub["label"].sum())
        return {"subset": name, "n_functions": n, "n_vulnerable": nv,
                "n_nonvulnerable": n - nv, "base_rate": nv / n if n else float("nan")}

    totals = pd.DataFrame([subset_totals(disjoint_idx, "disjoint"), subset_totals(shared_idx, "shared")])
    totals.to_csv(OUT_DIR / "pd_subset_totals.csv", index=False)
    print(totals)

    print("[2/6] Loading 28 prediction files ...")
    paths = all_config_paths()
    preds_by_cfg = {}
    for (model, cfg), path in paths.items():
        rows = load_predictions(path)
        labels = np.array([r["label"] for r in rows])
        preds = np.array([predicted_label(r) for r in rows])
        idxs = np.array([r["idx"] for r in rows])
        order = np.argsort(idxs)
        preds_by_cfg[(model, cfg)] = (labels[order], preds[order], idxs[order])

    disjoint_mask_full = np.array([i in disjoint_idx for i in range(len(mvfsc))])
    shared_mask_full = ~disjoint_mask_full

    def boot_ci(labels, preds, n_boot=N_BOOT, seed=SEED):
        rng = np.random.default_rng(seed)
        n = len(labels)
        idx = rng.integers(0, n, size=(n_boot, n))
        lb = labels[idx]
        pb = preds[idx]
        tp = np.sum((lb == 1) & (pb == 1), axis=1).astype(float)
        fp = np.sum((lb == 0) & (pb == 1), axis=1).astype(float)
        fn = np.sum((lb == 1) & (pb == 0), axis=1).astype(float)
        tn = np.sum((lb == 0) & (pb == 0), axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
            rec = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
            denom = 4.0 * prec + rec
            f2 = np.where(denom > 0, 5.0 * prec * rec / denom, np.nan)
            mcc_num = tp * tn - fp * fn
            mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = np.where(mcc_den > 0, mcc_num / mcc_den, np.nan)
        out = {}
        for name, arr in [("precision", prec), ("f2", f2), ("mcc", mcc)]:
            lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            out[f"{name}_ci_lo"] = lo
            out[f"{name}_ci_hi"] = hi
        return out

    print("[3/6] Computing per-config metrics on disjoint and shared subsets ...")
    rows_out = []
    roc_points = {m: [] for m in MODELS}  # model -> list of (fpr, tpr) on disjoint subset
    for model in MODELS:
        for cfg in ALL_CONFIGS:
            labels, preds, idxs = preds_by_cfg[(model, cfg)]
            assert np.array_equal(idxs, np.arange(len(mvfsc)))
            for subset_name, mask in [("disjoint", disjoint_mask_full), ("shared", shared_mask_full)]:
                lab_s = labels[mask]
                pred_s = preds[mask]
                m = confusion_from_pairs(lab_s, pred_s)
                n = m.n
                pi = m.base_rate
                q = m.ppr
                f2 = m.f2
                f2_rand = (5.0 * pi * q) / (4.0 * pi + q) if (4 * pi + q) else float("nan")
                mcc = mcc_of(m.tp, m.fp, m.fn, m.tn)
                if (m.tp + m.fp) > 0:
                    binom_p = stats.binomtest(m.tp, m.tp + m.fp, pi, alternative="two-sided").pvalue
                else:
                    binom_p = float("nan")
                cis = boot_ci(lab_s, pred_s)
                row = {
                    "model": model, "model_display": MODEL_DISPLAY[model],
                    "config": cfg, "config_display": CONFIG_DISPLAY.get(cfg, cfg),
                    "subset": subset_name, "n": n,
                    "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                    "ppr": q, "precision": m.precision, "recall": m.recall,
                    "f1": m.f1, "f2": f2, "accuracy": m.accuracy, "mcc": mcc,
                    "base_rate": pi, "f2_rand": f2_rand, "f2_minus_f2rand": f2 - f2_rand,
                    "binom_p_precision_vs_base_rate": binom_p,
                    **cis,
                }
                rows_out.append(row)
                if subset_name == "disjoint":
                    fpr = m.fp / (m.fp + m.tn) if (m.fp + m.tn) else float("nan")
                    tpr = m.recall
                    roc_points[model].append((fpr, tpr))

    df_metrics = pd.DataFrame(rows_out)
    df_metrics.to_csv(OUT_DIR / "pd_metrics.csv", index=False)

    print("[4/6] Computing ROC convex hull area (disjoint subset) per model ...")

    def hull_area(points):
        pts = set(points)
        pts.add((0.0, 0.0))
        pts.add((1.0, 1.0))
        pts.add((1.0, 0.0))  # chance corner so hull is well-defined even with few points
        pts = [p for p in pts if p == p and p[0] == p[0]]  # drop any nan
        pts_arr = np.array(sorted(pts))
        # Upper-left convex hull (ROC convex hull) via monotone chain on upper boundary.
        # Standard approach: compute full convex hull, then take the upper hull from (0,0) to (1,1).
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_arr)
        hull_pts = pts_arr[hull.vertices]
        # Area under the upper envelope of the hull (the ROC convex hull itself),
        # computed as hull polygon area intersected with the region above y=x
        # Simpler and standard: ROCCH area = area of convex hull of {points, (0,0), (1,1), (1,0)}
        # restricted to being the hull polygon itself (since (1,0) pulls hull to worst case,
        # but ROCCH convention uses only achievable/chance-dominating points). We instead
        # compute the hull of {(0,0),(1,1)} U points on the upper side, standard way:
        pts_sorted = pts_arr[np.argsort(pts_arr[:, 0])]
        # Build upper hull (maximizing TPR for given FPR) over (0,0)..(1,1) plus points,
        # excluding (1,0) which only served to make ConvexHull well-posed above; redo hull
        # without (1,0) for area calc.
        core_pts = [p for p in pts if p != (1.0, 0.0)]
        core_pts = sorted(set(core_pts))
        # Upper convex hull via Andrew's monotone chain (upper part only)
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        upper = []
        for p in core_pts:
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) >= 0:
                upper.pop()
            upper.append(p)
        # trapezoidal area under upper hull
        area = 0.0
        for i in range(len(upper) - 1):
            x0, y0 = upper[i]
            x1, y1 = upper[i + 1]
            area += (x1 - x0) * (y0 + y1) / 2.0
        return area, upper

    hull_rows = []
    for model in MODELS:
        area, upper = hull_area(roc_points[model])
        hull_rows.append({
            "model": model, "model_display": MODEL_DISPLAY[model],
            "roc_hull_area_disjoint": area,
            "points": str(sorted(set(roc_points[model]))),
            "hull_vertices": str(upper),
        })
    df_hull = pd.DataFrame(hull_rows)
    df_hull.to_csv(OUT_DIR / "pd_roc_hull.csv", index=False)
    print(df_hull[["model", "roc_hull_area_disjoint"]])

    print("[5/6] Near-duplicate audit restricted to disjoint subset ...")
    nd_path = OUT_DIR / "nd_best_match_per_row.csv"
    nd_summary = {}
    if nd_path.exists():
        nd = pd.read_csv(nd_path)
        nd_disjoint = nd[nd["mvfsc_idx"].isin(disjoint_idx)]
        for t in [0.7, 0.8, 0.9]:
            flagged = nd_disjoint[nd_disjoint["jaccard"] >= t]
            nd_summary[f"n_disjoint_flagged_t{t}"] = int(len(flagged))
        nd_summary["n_disjoint_total"] = int(len(nd_disjoint))
        nd_summary["source"] = str(nd_path.relative_to(Path(__file__).resolve().parents[2]))
    else:
        nd_summary["error"] = "nd_best_match_per_row.csv not found; run near_duplicates.py first"
    with open(OUT_DIR / "pd_near_dup_in_disjoint.json", "w") as f:
        json.dump(nd_summary, f, indent=2)
    print(nd_summary)

    print("[6/6] Done.")


if __name__ == "__main__":
    main()
