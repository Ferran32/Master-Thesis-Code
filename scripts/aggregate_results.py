#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR

VALID_PHASES = ["base-eval", "sft", "skill", "skill_eval", "combo"]

PHASE_DIR_MAP = {
    "base-eval": "base-eval",
}

RESULTS_RUNS = RESULTS_DIR / "runs"


def phase_to_dir(phase: str) -> str:
    return PHASE_DIR_MAP.get(phase, phase)


def load_run_metadata(run_dir: Path) -> Dict[str, str]:
    config_path = run_dir / "config.json"
    meta_path = run_dir / "run_meta.json"

    model = "unknown"
    strategy = "unknown"
    approach = "unknown"
    skill_version = ""

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model = cfg.get("model_name", model)
        strategy = cfg.get("prompt_strategy", strategy)
        approach = cfg.get("approach", approach)
        if "few_shot" in strategy:
            k = cfg.get("few_shot_k", cfg.get("n_few_shot", 3))
            strategy = f"{strategy}_{k}shot"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if model == "unknown":
            model = meta.get("model", model)
        if strategy == "unknown":
            strategy = meta.get("strategy", strategy)
            if "few_shot" in strategy:
                k = meta.get("few_shot_k", meta.get("n_few_shot", 3))
                strategy = f"{strategy}_{k}shot"
        if approach == "unknown":
            approach = meta.get("approach", approach)
        skill_version = meta.get("skill_version", "")

    return {"model": model, "strategy": strategy, "approach": approach, "skill_version": skill_version}


def discover_runs(phase_dir: Path) -> List[Tuple[Dict[str, str], Dict[str, Any], Path]]:
    seen: Dict[Tuple[str, str, str, str], Tuple[Dict[str, str], Dict[str, Any], Path]] = {}

    for metrics_path in sorted(phase_dir.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        try:
            with open(metrics_path) as f:
                metrics_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        metadata = load_run_metadata(run_dir)
        key = (metadata["model"], metadata["approach"], metadata["strategy"], metadata.get("skill_version", ""))

        if key in seen:
            existing_dir = seen[key][2].parent.name
            if run_dir.name > existing_dir:
                seen[key] = (metadata, metrics_data, metrics_path)
        else:
            seen[key] = (metadata, metrics_data, metrics_path)

    entries = list(seen.values())
    entries.sort(key=lambda e: (e[0]["model"], e[0]["strategy"]))
    return entries


def print_summary_table(
    phase: str,
    entries: List[Tuple[Dict[str, str], Dict[str, Any], Path]],
) -> None:
    header = (
        f"{'Model':<24} {'Approach':<8} {'Strategy':<26} "
        f"{'F2':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} "
        f"{'Acc':>8} {'FP':>5} {'FN':>5} {'Parse':>6}"
    )
    title = f"  {phase.upper()} RESULTS"
    print("=" * len(header))
    print(title)
    print("=" * len(header))
    print()
    print(header)
    print("-" * len(header))

    current_model = None
    for metadata, data, _ in entries:
        model = metadata["model"]
        strategy = metadata["strategy"]
        approach = metadata.get("approach", "")
        skill_ver = metadata.get("skill_version", "")
        display_strategy = f"{strategy} ({skill_ver})" if skill_ver else strategy
        m = data.get("metrics", {})
        ea = data.get("error_analysis", {})
        conf = ea.get("confusion", {})

        if model != current_model:
            if current_model is not None:
                print()
            current_model = model

        print(
            f"{model:<24} {approach:<8} {display_strategy:<26} "
            f"{m.get('f2', 0.0):>8.4f} "
            f"{m.get('f1', 0.0):>8.4f} "
            f"{m.get('precision', 0.0):>8.4f} "
            f"{m.get('recall', 0.0):>8.4f} "
            f"{m.get('accuracy', 0.0):>8.4f} "
            f"{conf.get('fp', 0):>5} "
            f"{conf.get('fn', 0):>5} "
            f"{m.get('parse_failures', 0):>6}"
        )

    print()


def print_best_per_model(
    entries: List[Tuple[Dict[str, str], Dict[str, Any], Path]],
) -> None:
    best: Dict[str, Tuple[str, float]] = {}
    for metadata, data, _ in entries:
        model = metadata["model"]
        strategy = metadata["strategy"]
        f2 = data.get("metrics", {}).get("f2", 0.0)
        if model not in best or f2 > best[model][1]:
            best[model] = (strategy, f2)

    if not best:
        return

    print("=" * 60)
    print("  BEST STRATEGY PER MODEL (by F2)")
    print("=" * 60)
    print()
    print(f"{'Model':<24} {'Best Strategy':<20} {'F2':>8}")
    print("-" * 54)
    for model in sorted(best):
        strat, f2 = best[model]
        print(f"{model:<24} {strat:<20} {f2:>8.4f}")
    print()


def print_cwe_miss_report(
    entries: List[Tuple[Dict[str, str], Dict[str, Any], Path]],
) -> None:
    cwe_totals: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    model_cwe_fn: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for metadata, data, _ in entries:
        model = metadata["model"]
        per_cwe = data.get("error_analysis", {}).get("per_cwe", {})
        for cwe, counts in per_cwe.items():
            cwe_totals[cwe]["tp"] += counts.get("tp", 0)
            cwe_totals[cwe]["fp"] += counts.get("fp", 0)
            cwe_totals[cwe]["fn"] += counts.get("fn", 0)
            cwe_totals[cwe]["tn"] += counts.get("tn", 0)
            model_cwe_fn[model][cwe] += counts.get("fn", 0)

    sorted_cwes = sorted(
        cwe_totals.items(),
        key=lambda x: x[1]["fn"],
        reverse=True,
    )

    has_fn = any(counts["fn"] > 0 for _, counts in sorted_cwes)
    if not has_fn:
        return

    print("=" * 70)
    print("  CWEs MOST FREQUENTLY MISSED (sorted by total FN)")
    print("=" * 70)
    print()
    print(f"  {'CWE':<14} {'FN':>6} {'TP':>6} {'Recall':>8}  Models that miss it")
    print("  " + "-" * 62)

    for cwe, counts in sorted_cwes:
        if counts["fn"] == 0:
            continue
        tp = counts["tp"]
        fn = counts["fn"]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        missing_models = [
            m for m in sorted(model_cwe_fn)
            if model_cwe_fn[m].get(cwe, 0) > 0
        ]
        models_str = ", ".join(missing_models) if missing_models else "\u2014"

        print(
            f"  {cwe:<14} {fn:>6} {tp:>6} {recall:>8.4f}  {models_str}"
        )

    print()

    print("=" * 70)
    print("  PER-MODEL TOP 5 MISSED CWEs")
    print("=" * 70)
    print()
    for model in sorted(model_cwe_fn):
        top_missed = sorted(
            model_cwe_fn[model].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        if not top_missed or top_missed[0][1] == 0:
            continue
        print(f"  {model}:")
        for cwe, fn_count in top_missed:
            if fn_count > 0:
                print(f"    {cwe:<14} FN={fn_count}")
        print()


def build_json_output(
    phase: str,
    entries: List[Tuple[Dict[str, str], Dict[str, Any], Path]],
) -> Dict[str, Any]:
    experiments = []
    for metadata, data, metrics_path in entries:
        experiments.append({
            "model": metadata["model"],
            "strategy": metadata["strategy"],
            "approach": metadata["approach"],
            "skill_version": metadata.get("skill_version", ""),
            "run_dir": str(metrics_path.parent),
            "metrics": data.get("metrics", {}),
            "error_analysis": data.get("error_analysis", {}),
        })

    return {
        "phase": phase,
        "n_runs": len(experiments),
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results for a given phase.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=VALID_PHASES,
        help="Experiment phase to aggregate.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_RUNS,
        help=f"Root directory containing phase subdirectories (default: {RESULTS_RUNS})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for structured JSON output (default: <results_dir>/<phase>/aggregate_summary.json).",
    )
    args = parser.parse_args()

    phase_dir_name = phase_to_dir(args.phase)
    phase_dir = args.results_dir / phase_dir_name

    if not phase_dir.exists():
        print(f"ERROR: Phase directory not found: {phase_dir}")
        print(f"Available phases under {args.results_dir}:")
        for d in sorted(args.results_dir.iterdir()):
            if d.is_dir():
                print(f"  - {d.name}")
        return

    entries = discover_runs(phase_dir)
    if not entries:
        print(f"ERROR: No metrics.json files found under {phase_dir}")
        return

    print(f"Phase: {args.phase}")
    print(f"Found {len(entries)} run(s) in {phase_dir}")
    print()

    print_summary_table(args.phase, entries)
    print_best_per_model(entries)
    print_cwe_miss_report(entries)

    json_output = build_json_output(args.phase, entries)
    out_json = args.output_json or (phase_dir / "aggregate_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Structured summary saved to: {out_json}")


if __name__ == "__main__":
    main()
