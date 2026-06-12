from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def collect_run_metrics(runs_dir: Path) -> List[Dict[str, Any]]:
    records = []
    if not runs_dir.exists():
        return records

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        errors = data.get("error_analysis", {})
        conf = errors.get("confusion", {})

        cfg: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)

        records.append({
            "run_id": run_dir.name,
            "model": cfg.get("model_name", ""),
            "approach": cfg.get("approach", ""),
            "strategy": cfg.get("prompt_strategy", ""),
            "split": cfg.get("dataset_split", ""),
            "samples": metrics.get("total", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1", 0),
            "f2": metrics.get("f2", 0),
            "accuracy": metrics.get("accuracy", 0),
            "parse_failures": metrics.get("parse_failures", 0),
            "tp": conf.get("tp", 0),
            "fp": conf.get("fp", 0),
            "fn": conf.get("fn", 0),
            "tn": conf.get("tn", 0),
        })

    return records


def export_comparison_csv(records: List[Dict[str, Any]], output_path: Path) -> Path:
    if not records:
        output_path.write_text("No data\n")
        return output_path

    fieldnames = [
        "model", "approach", "strategy", "split", "samples",
        "precision", "recall", "f1", "f2", "accuracy",
        "parse_failures", "tp", "fp", "fn", "tn",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(records, key=lambda x: x.get("f1", 0), reverse=True):
            writer.writerow(r)

    return output_path


def export_comparison_markdown(records: List[Dict[str, Any]], output_path: Path) -> Path:
    if not records:
        output_path.write_text("No results to export.\n")
        return output_path

    sorted_records = sorted(records, key=lambda x: x.get("f1", 0), reverse=True)

    lines = [
        "# Model & Approach Comparison",
        "",
        "| # | Model | Approach | Strategy | F1 | F2 | Precision | Recall | Accuracy | FP | FN | Parse Fail |",
        "|---|-------|----------|----------|----|----|-----------|--------|----------|----|----|------------|",
    ]
    for i, r in enumerate(sorted_records, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['approach']} | {r['strategy']} | "
            f"{r['f1']:.4f} | {r['f2']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['accuracy']:.4f} | {r['fp']} | {r['fn']} | {r['parse_failures']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def export_model_rankings(records: List[Dict[str, Any]], output_path: Path, metric: str = "f1") -> Path:
    best_per_model: Dict[str, Dict[str, Any]] = {}
    for r in records:
        model = r["model"]
        if model not in best_per_model or r.get(metric, 0) > best_per_model[model].get(metric, 0):
            best_per_model[model] = r

    ranked = sorted(best_per_model.values(), key=lambda x: x.get(metric, 0), reverse=True)

    lines = [
        f"# Model Rankings (by best {metric.upper()})",
        "",
        f"| Rank | Model | Best Strategy | {metric.upper()} | F1 | Precision | Recall |",
        f"|------|-------|---------------|{'-' * len(metric)}|----|-----------|--------|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['strategy']} | "
            f"{r.get(metric, 0):.4f} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def export_fp_fn_summary(records: List[Dict[str, Any]], output_path: Path) -> Path:
    lines = [
        "# False Positive / False Negative Summary",
        "",
        "| Model | Approach | Strategy | TP | FP | FN | TN | FP Rate | FN Rate |",
        "|-------|----------|----------|----|----|----|----|---------|---------|",
    ]
    for r in sorted(records, key=lambda x: x.get("fn", 0), reverse=True):
        tp, fp, fn, tn = r["tp"], r["fp"], r["fn"], r["tn"]
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        lines.append(
            f"| {r['model']} | {r['approach']} | {r['strategy']} | "
            f"{tp} | {fp} | {fn} | {tn} | {fp_rate:.4f} | {fn_rate:.4f} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def export_all(results_base: Path, exports_dir: Optional[Path] = None) -> Dict[str, Path]:
    if exports_dir is None:
        exports_dir = results_base / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    for phase_dir in sorted(results_base.iterdir()):
        if phase_dir.is_dir() and phase_dir.name != "exports":
            all_records.extend(collect_run_metrics(phase_dir))

    paths = {}
    if all_records:
        paths["comparison_csv"] = export_comparison_csv(
            all_records, exports_dir / "comparison.csv"
        )
        paths["comparison_md"] = export_comparison_markdown(
            all_records, exports_dir / "comparison.md"
        )
        paths["rankings"] = export_model_rankings(
            all_records, exports_dir / "model_rankings.md"
        )
        paths["fp_fn"] = export_fp_fn_summary(
            all_records, exports_dir / "fp_fn_summary.md"
        )

    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export thesis tables from results")
    parser.add_argument("--results-dir", default="results/runs", help="Root results directory")
    parser.add_argument("--exports-dir", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    exports_dir = Path(args.exports_dir) if args.exports_dir else None
    paths = export_all(results_dir, exports_dir)

    for name, path in paths.items():
        print(f"  {name}: {path}")
