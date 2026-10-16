"""Part 1: near-duplicate analysis between MVFSC (test) and train/val.

Run: .venv/bin/python3 analysis/statistics/near_duplicates.py

Outputs (all under analysis/statistics/results/):
  nd_counts_by_threshold.csv       - counts broken down by split/label/same-label/project
  nd_jaccard_distribution.csv      - cumulative share of MVFSC functions at each sim cutoff
  nd_jaccard_histogram.png
  nd_filtered_metrics_t0.7.csv, _t0.8.csv, _t0.9.csv   - 28-config metrics on filtered subsets
  nd_example_pairs.csv
  nd_sanity_check.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CONFIG_DISPLAY,
    HEADLINE_CONFIGS,
    MODEL_DISPLAY,
    MODELS,
    REPO_ROOT,
    SEED,
    STRATEGIES,
    all_config_paths,
    load_mvfsc_test,
    load_predictions,
    load_train_val,
    normalize,
    predicted_label,
)
from lexer import shingles, tokenize  # noqa: E402
from metrics import binomial_precision_test, confusion_from_pairs  # noqa: E402

random.seed(SEED)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TOKENS = 10
NUM_PERM = 128
THRESHOLDS = [0.7, 0.8, 0.9]
LSH_THRESHOLD = 0.5  # conservative: candidate-generation threshold, real filtering uses exact Jaccard
SHINGLE_N = 3

# The 18 projects the thesis says overlap BigVul/MegaVul/CVEfixes by name
# (CLAUDE.md: "18 overlap training corpus by name but 0 SHA1 snippet matches").
# We derive this set from the actual intersection of project names rather
# than hardcoding a list, so it stays correct if the underlying data changes.


def build_shingle_sets(df: pd.DataFrame, text_col: str = "text"):
    """Return dict: row_pos -> (token_count, shingle_set). Rows with fewer
    than MIN_TOKENS tokens after lexing are skipped (not included in dict)."""
    out = {}
    skipped = 0
    for pos, text in enumerate(df[text_col]):
        norm = normalize(text)
        toks = tokenize(norm)
        if len(toks) < MIN_TOKENS:
            skipped += 1
            continue
        out[pos] = (len(toks), shingles(toks, SHINGLE_N))
    return out, skipped


def minhash_for(shingle_set) -> MinHash:
    mh = MinHash(num_perm=NUM_PERM, seed=SEED)
    for sh in shingle_set:
        mh.update(" ".join(sh).encode("utf-8"))
    return mh


def exact_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main():
    print("[1/7] Loading MVFSC test set, train/val splits ...")
    mvfsc = load_mvfsc_test()
    mvfsc_df = mvfsc.to_pandas()
    train_ds, val_ds = load_train_val()
    train_df = train_ds.to_pandas()
    val_df = val_ds.to_pandas()

    with open(Path(__file__).resolve().parent / "data" / "mvfsc_pairs.json") as f:
        pair_rows = json.load(f)
    pair_by_idx = {r["idx"]: r for r in pair_rows}
    assert len(pair_by_idx) == len(mvfsc_df), "pair map must cover every MVFSC row"

    print(f"  MVFSC: {len(mvfsc_df)} rows, train: {len(train_df)} rows, val: {len(val_df)} rows")

    # 18-overlap-project set, derived from name intersection (reported, not assumed).
    mvfsc_projects = set(mvfsc_df["project"].str.lower())
    corpus_projects = set(train_df["project"].str.lower()) | set(val_df["project"].str.lower())
    overlap_projects = mvfsc_projects & corpus_projects
    print(f"  MVFSC projects: {len(mvfsc_projects)}; name-overlap with train/val corpus: {len(overlap_projects)}")

    print("[2/7] Building shingle sets for train/val corpus ...")
    train_sets, train_skipped = build_shingle_sets(train_df)
    val_sets, val_skipped = build_shingle_sets(val_df)
    print(f"  train: {len(train_sets)} kept, {train_skipped} skipped (<{MIN_TOKENS} tokens)")
    print(f"  val:   {len(val_sets)} kept, {val_skipped} skipped (<{MIN_TOKENS} tokens)")

    print("[3/7] Building shingle sets for MVFSC + LSH index over corpus ...")
    mvfsc_sets, mvfsc_skipped = build_shingle_sets(mvfsc_df)
    print(f"  MVFSC: {len(mvfsc_sets)} kept, {mvfsc_skipped} skipped (<{MIN_TOKENS} tokens)")

    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    corpus_minhash = {}
    for pos, (_, sset) in train_sets.items():
        key = ("train", pos)
        mh = minhash_for(sset)
        corpus_minhash[key] = mh
        lsh.insert(f"train:{pos}", mh)
    for pos, (_, sset) in val_sets.items():
        key = ("val", pos)
        mh = minhash_for(sset)
        corpus_minhash[key] = mh
        lsh.insert(f"val:{pos}", mh)
    print(f"  LSH index size: {len(corpus_minhash)} (threshold={LSH_THRESHOLD}, num_perm={NUM_PERM})")

    print("[4/7] Querying best match per MVFSC function (LSH candidates -> exact Jaccard) ...")
    best_match = {}  # mvfsc_pos -> dict(split, corpus_pos, jaccard, same_label, same_sha1)
    for pos, (_, sset) in mvfsc_sets.items():
        mh = minhash_for(sset)
        candidates = lsh.query(mh)
        best = None
        for cand_key in candidates:
            split, cpos_str = cand_key.split(":")
            cpos = int(cpos_str)
            csset = train_sets[cpos][1] if split == "train" else val_sets[cpos][1]
            j = exact_jaccard(sset, csset)
            if best is None or j > best[2]:
                best = (split, cpos, j)
        if best is not None:
            split, cpos, j = best
            corpus_row = train_df.iloc[cpos] if split == "train" else val_df.iloc[cpos]
            mv_row = mvfsc_df.iloc[pos]
            same_label = int(corpus_row["label"]) == int(mv_row["label"])
            same_sha1 = corpus_row["code_sha1"] == mv_row["code_sha1"]
            best_match[pos] = {
                "split": split,
                "corpus_pos": cpos,
                "jaccard": j,
                "same_label": same_label,
                "same_sha1": same_sha1,
                "mvfsc_label": int(mv_row["label"]),
                "mvfsc_project": mv_row["project"],
                "corpus_label": int(corpus_row["label"]),
                "corpus_project": corpus_row["project"],
            }
    print(f"  best-match found for {len(best_match)}/{len(mvfsc_sets)} MVFSC functions with >=1 LSH candidate")

    # --- sanity check: exact SHA-1 dup to an identical *normalised* function should be 0 ---
    exact_dup_cases = [
        {"mvfsc_pos": pos, **v} for pos, v in best_match.items() if v["jaccard"] >= 1.0 - 1e-9
    ]
    sanity = {
        "n_exact_jaccard_1.0": len(exact_dup_cases),
        "cases": exact_dup_cases[:20],
        "expected": 0,
        "pass": len(exact_dup_cases) == 0,
    }
    with open(OUT_DIR / "nd_sanity_check.json", "w") as f:
        json.dump(sanity, f, indent=2, default=str)
    print(f"[5/7] Sanity check (Jaccard=1.0 count should be 0): {len(exact_dup_cases)} -> pass={sanity['pass']}")

    # --- distribution of best-match Jaccard over all 872 (functions with no candidate => 0.0) ---
    all_jaccards = []
    for pos in range(len(mvfsc_df)):
        if pos in best_match:
            all_jaccards.append(best_match[pos]["jaccard"])
        elif pos in mvfsc_sets:
            all_jaccards.append(0.0)  # had shingles, but zero LSH candidates above threshold
        else:
            all_jaccards.append(float("nan"))  # skipped (<10 tokens)

    dist_rows = []
    valid = [j for j in all_jaccards if j == j]
    for cutoff in [0.5, 0.6, 0.7, 0.8, 0.9]:
        n_at_or_above = sum(1 for j in valid if j >= cutoff)
        dist_rows.append({
            "cutoff": cutoff,
            "n_at_or_above": n_at_or_above,
            "share_of_872": n_at_or_above / 872,
            "share_of_scored": n_at_or_above / len(valid) if valid else float("nan"),
        })
    pd.DataFrame(dist_rows).to_csv(OUT_DIR / "nd_jaccard_distribution.csv", index=False)

    # full per-row best-match table (used for the LSH-recall spot check and auditing)
    per_row_rows = []
    for pos in range(len(mvfsc_df)):
        if pos in best_match:
            v = best_match[pos]
            per_row_rows.append({
                "mvfsc_idx": pos, "n_tokens_ge_min": True, "split": v["split"],
                "corpus_pos": v["corpus_pos"], "jaccard": v["jaccard"],
                "same_label": v["same_label"], "same_sha1": v["same_sha1"],
            })
        else:
            per_row_rows.append({
                "mvfsc_idx": pos, "n_tokens_ge_min": pos in mvfsc_sets, "split": None,
                "corpus_pos": None, "jaccard": 0.0 if pos in mvfsc_sets else float("nan"),
                "same_label": None, "same_sha1": None,
            })
    pd.DataFrame(per_row_rows).to_csv(OUT_DIR / "nd_best_match_per_row.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.hist([j for j in valid], bins=40, range=(0, 1), color="#4C72B0", edgecolor="white")
    plt.axvline(0.7, color="#C44E52", linestyle="--", label="0.7")
    plt.axvline(0.8, color="#DD8452", linestyle="--", label="0.8")
    plt.axvline(0.9, color="#55A868", linestyle="--", label="0.9")
    plt.xlabel("Best-match token-3-gram Jaccard similarity (MVFSC function vs. train/val)")
    plt.ylabel("Number of MVFSC functions")
    plt.title("Distribution of best-match similarity, MVFSC vs. train+val corpus (n=872)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "nd_jaccard_histogram.png", dpi=150)
    plt.close()

    # --- counts by threshold, broken down ---
    print("[6/7] Counting near-duplicates by threshold/split/label/same-label/project ...")
    count_rows = []
    flagged_idx_by_threshold = {}
    for t in THRESHOLDS:
        flagged = [pos for pos, j in enumerate(all_jaccards) if j == j and j >= t]
        flagged_idx_by_threshold[t] = set(flagged)

        n_total = len(flagged)
        n_train = sum(1 for p in flagged if best_match[p]["split"] == "train")
        n_val = sum(1 for p in flagged if best_match[p]["split"] == "val")
        n_vuln = sum(1 for p in flagged if best_match[p]["mvfsc_label"] == 1)
        n_nonvuln = sum(1 for p in flagged if best_match[p]["mvfsc_label"] == 0)
        n_same_label = sum(1 for p in flagged if best_match[p]["same_label"])
        n_opp_label = n_total - n_same_label
        n_overlap_proj = sum(
            1 for p in flagged if str(best_match[p]["mvfsc_project"]).lower() in overlap_projects
        )
        count_rows.append({
            "threshold": t,
            "n_flagged": n_total,
            "share_of_872": n_total / 872,
            "n_from_train": n_train,
            "n_from_val": n_val,
            "n_mvfsc_label_vuln": n_vuln,
            "n_mvfsc_label_nonvuln": n_nonvuln,
            "n_same_label_as_match": n_same_label,
            "n_opposite_label_as_match": n_opp_label,
            "n_in_18_overlap_projects": n_overlap_proj,
        })

        # per-project breakdown for overlap projects, this threshold
        per_proj = defaultdict(int)
        for p in flagged:
            proj = str(best_match[p]["mvfsc_project"])
            if proj.lower() in overlap_projects:
                per_proj[proj] += 1
        pd.DataFrame(
            [{"threshold": t, "project": k, "n_flagged": v} for k, v in sorted(per_proj.items())]
        ).to_csv(OUT_DIR / f"nd_per_project_t{t}.csv", index=False)

    pd.DataFrame(count_rows).to_csv(OUT_DIR / "nd_counts_by_threshold.csv", index=False)

    # --- example pairs at highest similarity ---
    ranked = sorted(best_match.items(), key=lambda kv: -kv[1]["jaccard"])
    examples = []
    for pos, v in ranked[:5]:
        mv_text = mvfsc_df.iloc[pos]["text"]
        corpus_df = train_df if v["split"] == "train" else val_df
        corpus_text = corpus_df.iloc[v["corpus_pos"]]["text"]
        examples.append({
            "mvfsc_idx": pos,
            "mvfsc_project": v["mvfsc_project"],
            "mvfsc_label": v["mvfsc_label"],
            "corpus_split": v["split"],
            "corpus_project": v["corpus_project"],
            "corpus_label": v["corpus_label"],
            "jaccard": round(v["jaccard"], 4),
            "mvfsc_snippet": mv_text[:300].replace("\n", " \\n "),
            "corpus_snippet": corpus_text[:300].replace("\n", " \\n "),
        })
    pd.DataFrame(examples).to_csv(OUT_DIR / "nd_example_pairs.csv", index=False)

    # --- effect on results: recompute 28-config metrics on filtered subsets ---
    print("[7/7] Recomputing metrics on near-duplicate-filtered subsets (variants a/b, 3 thresholds) ...")
    config_paths = all_config_paths()
    # cache loaded predictions once
    preds_by_config = {}
    for key, path in config_paths.items():
        rows = load_predictions(path)
        labels = [r["label"] for r in rows]
        preds = [predicted_label(r) for r in rows]
        preds_by_config[key] = (labels, preds)

    full_metrics_rows = []
    for (model, cfg), (labels, preds) in preds_by_config.items():
        m = confusion_from_pairs(labels, preds)
        full_metrics_rows.append({
            "model": model, "config": cfg, "variant": "full", "threshold": None,
            "n": m.n, "precision": m.precision, "recall": m.recall, "f1": m.f1, "f2": m.f2,
            "accuracy": m.accuracy, "ppr": m.ppr, "base_rate": m.base_rate,
            "binom_p_precision_vs_base_rate": binomial_precision_test(m),
        })

    filtered_rows = []
    for t in THRESHOLDS:
        flagged_a = flagged_idx_by_threshold[t]
        # variant b: remove whole pair if either half flagged
        flagged_pairs = {pair_by_idx[p]["pair_id"] for p in flagged_a}
        flagged_b = {idx for idx, pr in pair_by_idx.items() if pr["pair_id"] in flagged_pairs}

        for variant_name, flagged_set in [("a_flagged_only", flagged_a), ("b_whole_pair", flagged_b)]:
            keep_idx = [i for i in range(872) if i not in flagged_set]
            for (model, cfg), (labels, preds) in preds_by_config.items():
                sub_labels = [labels[i] for i in keep_idx]
                sub_preds = [preds[i] for i in keep_idx]
                m = confusion_from_pairs(sub_labels, sub_preds)
                full_m = confusion_from_pairs(labels, preds)
                row = {
                    "model": model, "config": cfg, "variant": variant_name, "threshold": t,
                    "n": m.n, "precision": m.precision, "recall": m.recall, "f1": m.f1, "f2": m.f2,
                    "accuracy": m.accuracy, "ppr": m.ppr, "base_rate": m.base_rate,
                    "binom_p_precision_vs_base_rate": binomial_precision_test(m),
                    "f2_delta_vs_full": (m.f2 - full_m.f2) if m.f2 == m.f2 and full_m.f2 == full_m.f2 else float("nan"),
                }
                filtered_rows.append(row)

    df_filtered = pd.DataFrame(filtered_rows)
    df_full = pd.DataFrame(full_metrics_rows)
    df_full.to_csv(OUT_DIR / "nd_filtered_metrics_full.csv", index=False)
    for t in THRESHOLDS:
        df_filtered[df_filtered["threshold"] == t].to_csv(
            OUT_DIR / f"nd_filtered_metrics_t{t}.csv", index=False
        )
    df_filtered.to_csv(OUT_DIR / "nd_filtered_metrics_all.csv", index=False)

    # bundle a small machine-readable summary for the SUMMARY.md generator
    summary = {
        "mvfsc_skipped_short": mvfsc_skipped,
        "train_skipped_short": train_skipped,
        "val_skipped_short": val_skipped,
        "n_mvfsc_scored": len(mvfsc_sets),
        "n_overlap_projects": len(overlap_projects),
        "overlap_projects": sorted(overlap_projects),
        "sanity_check": sanity,
        "counts_by_threshold": count_rows,
        "jaccard_distribution": dist_rows,
    }
    with open(OUT_DIR / "nd_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("Done. See analysis/statistics/results/nd_*.csv and nd_run_summary.json")


if __name__ == "__main__":
    main()
