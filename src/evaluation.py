from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    parse_failures = int((y_pred_arr == -1).sum())
    y_pred_clean = np.where(y_pred_arr == -1, 0, y_pred_arr)

    return {
        "precision": round(float(precision_score(y_true_arr, y_pred_clean, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true_arr, y_pred_clean, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true_arr, y_pred_clean, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true_arr, y_pred_clean, beta=2, zero_division=0)), 4),
        "accuracy": round(float((y_true_arr == y_pred_clean).mean()), 4),
        "total": len(y_true_arr),
        "parse_failures": parse_failures,
    }


def error_analysis(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    cwe_ids: Optional[Sequence[Optional[str]]] = None,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    parse_failures = int((y_pred_arr == -1).sum())
    y_pred_clean = np.where(y_pred_arr == -1, 0, y_pred_arr)

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr, y_pred_clean, labels=[0, 1]
    ).ravel()

    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    result: Dict[str, Any] = {
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "fp_rate": round(float(fp_rate), 4),
        "fn_rate": round(float(fn_rate), 4),
        "parse_failures": parse_failures,
    }

    if cwe_ids is not None:
        per_cwe: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )
        for true, pred, cwe in zip(y_true_arr, y_pred_clean, cwe_ids):
            cwe_key = cwe if cwe else "UNKNOWN"
            if true == 1 and pred == 1:
                per_cwe[cwe_key]["tp"] += 1
            elif true == 0 and pred == 1:
                per_cwe[cwe_key]["fp"] += 1
            elif true == 1 and pred == 0:
                per_cwe[cwe_key]["fn"] += 1
            else:
                per_cwe[cwe_key]["tn"] += 1

        per_cwe_metrics = {}
        for cwe_key, counts in per_cwe.items():
            tp_c, fp_c, fn_c = counts["tp"], counts["fp"], counts["fn"]
            p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_cwe_metrics[cwe_key] = {
                **counts, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1_c, 4),
            }
        result["per_cwe"] = per_cwe_metrics

    return result


def load_results(path: Path) -> List[Dict[str, Any]]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_result_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_results_file(path: Path) -> Dict[str, Any]:
    results = load_results(path)
    y_true = [r["label"] for r in results]
    y_pred = [r["predicted_label"] for r in results]
    cwe_ids = [r.get("cwe_id") for r in results]
    return {
        "metrics": compute_metrics(y_true, y_pred),
        "error_analysis": error_analysis(y_true, y_pred, cwe_ids=cwe_ids),
    }


def print_metrics(metrics: Dict[str, float], title: str = "") -> None:
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  F2-score  : {metrics['f2']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Total     : {metrics['total']}")
    if metrics.get("parse_failures", 0) > 0:
        print(f"  Parse fail: {metrics['parse_failures']}")
    print()
