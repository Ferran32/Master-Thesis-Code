
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation import (
    compute_metrics,
    error_analysis,
)


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    cwe_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    metrics = compute_metrics(y_true, y_pred)
    errors = error_analysis(y_true, y_pred, cwe_ids=cwe_ids)
    return {"metrics": metrics, "error_analysis": errors}


def write_run_outputs(
    run_dir: Path,
    predictions: List[Dict[str, Any]],
    metrics: Dict[str, float],
    errors: Dict[str, Any],
    config_snapshot: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> Dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    pred_path = run_dir / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths["predictions"] = pred_path

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics, "error_analysis": errors}, f, indent=2)
    paths["metrics"] = metrics_path

    cfg_path = run_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(config_snapshot, f, indent=2)
    paths["config"] = cfg_path

    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    paths["run_meta"] = meta_path

    summary_path = run_dir / "summary.md"
    summary_path.write_text(_build_summary_md(metrics, errors, config_snapshot, run_meta))
    paths["summary"] = summary_path

    csv_path = run_dir / "metrics_table.csv"
    _write_metrics_csv(csv_path, metrics, config_snapshot, run_meta)
    paths["csv"] = csv_path

    return paths


def _build_summary_md(
    metrics: Dict[str, float],
    errors: Dict[str, Any],
    config_snapshot: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> str:
    conf = errors.get("confusion", {})
    lines = [
        f"# Run Summary",
        "",
        f"- **Model**: {config_snapshot.get('model_name', '?')}",
        f"- **Approach**: {config_snapshot.get('approach', '?')}",
        f"- **Strategy**: {config_snapshot.get('prompt_strategy', '?')}",
        f"- **Split**: {config_snapshot.get('dataset_split', '?')}",
        f"- **Samples**: {metrics.get('total', '?')}",
        f"- **Duration**: {run_meta.get('elapsed_seconds', '?')}s",
        "",
        "## Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Precision | {metrics.get('precision', 0):.4f} |",
        f"| Recall | {metrics.get('recall', 0):.4f} |",
        f"| F1 | {metrics.get('f1', 0):.4f} |",
        f"| F2 | {metrics.get('f2', 0):.4f} |",
        f"| Accuracy | {metrics.get('accuracy', 0):.4f} |",
        f"| Parse failures | {metrics.get('parse_failures', 0)} |",
        "",
        "## Confusion Matrix",
        "",
        f"| | Predicted Vuln | Predicted Safe |",
        f"|---|---|---|",
        f"| **Actual Vuln** | TP={conf.get('tp', 0)} | FN={conf.get('fn', 0)} |",
        f"| **Actual Safe** | FP={conf.get('fp', 0)} | TN={conf.get('tn', 0)} |",
        "",
    ]

    per_cwe = errors.get("per_cwe", {})
    if per_cwe:
        lines.extend([
            "## Per-CWE Performance",
            "",
            "| CWE | TP | FP | FN | Precision | Recall | F1 |",
            "|-----|----|----|----|-----------|---------|----|",
        ])
        for cwe_key, c in sorted(per_cwe.items(), key=lambda x: x[1].get("fn", 0), reverse=True):
            lines.append(
                f"| {cwe_key} | {c['tp']} | {c['fp']} | {c['fn']} | "
                f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _write_metrics_csv(
    path: Path,
    metrics: Dict[str, float],
    config_snapshot: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> None:
    fieldnames = [
        "model", "approach", "strategy", "split", "samples",
        "precision", "recall", "f1", "f2", "accuracy", "parse_failures",
        "elapsed_seconds",
    ]
    row = {
        "model": config_snapshot.get("model_name", ""),
        "approach": config_snapshot.get("approach", ""),
        "strategy": config_snapshot.get("prompt_strategy", ""),
        "split": config_snapshot.get("dataset_split", ""),
        "samples": metrics.get("total", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1": metrics.get("f1", 0),
        "f2": metrics.get("f2", 0),
        "accuracy": metrics.get("accuracy", 0),
        "parse_failures": metrics.get("parse_failures", 0),
        "elapsed_seconds": run_meta.get("elapsed_seconds", 0),
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
