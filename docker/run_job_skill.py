#!/usr/bin/env python3
"""Skill evaluation job: downloads skill from GCS, runs inference, uploads results."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "/app")

from src.config import get_model_id, model_short_name
from src.legacy.run_agent_skill import generate_diagnostic_report, load_skill, skill_version_from_path
from src.pipeline.config_loader import load_config
from src.pipeline.dataset import load_dataset_split, save_index_manifest
from src.pipeline.evaluation import evaluate_predictions, write_run_outputs
from src.pipeline.inference import generate, load_model, unload_model
from src.pipeline.parsing import parse_sample
from src.pipeline.prompting import build_prompt_from_config
from src.pipeline.run_state import append_prediction, load_completed_indices
from src.pipeline.tracking import RunTimer, build_run_meta

MODEL_NAME = os.environ["MODEL_NAME"]
GCS_BUCKET = os.environ["GCS_BUCKET"]
SKILL_GCS_PATH = os.environ["SKILL_GCS_PATH"]
GCS_PREFIX = os.environ.get("GCS_PREFIX", "skill")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/configs/phases/skill_vertexai.yaml")
DIAGNOSE = os.environ.get("DIAGNOSE", "false").lower() == "true"
DATASET_SPLIT_OVERRIDE = os.environ.get("DATASET_SPLIT", "")
LIMIT_OVERRIDE = os.environ.get("LIMIT", "")
SKILL_VERSION_SUFFIX = os.environ.get("SKILL_VERSION_SUFFIX", "")

LOCAL_OUT = Path("/tmp/results")
LOCAL_SKILL_DIR = Path("/tmp/skills")


def download_skill_from_gcs() -> Path:
    from google.cloud import storage as gcs
    LOCAL_SKILL_DIR.mkdir(parents=True, exist_ok=True)
    skill_filename = Path(SKILL_GCS_PATH).name
    local_path = LOCAL_SKILL_DIR / skill_filename
    print(f"Downloading skill from gs://{GCS_BUCKET}/{SKILL_GCS_PATH} → {local_path}")
    client = gcs.Client()
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(SKILL_GCS_PATH).download_to_filename(str(local_path))
    return local_path


def upload_results_to_gcs(skill_ver: str, run_dir: Path) -> None:
    from google.cloud import storage as gcs
    prefix = f"{GCS_PREFIX}/{skill_ver}/{MODEL_NAME}"
    print(f"\nUploading results to gs://{GCS_BUCKET}/{prefix}/")
    client = gcs.Client()
    bucket = client.bucket(GCS_BUCKET)
    uploaded = 0
    for f in run_dir.rglob("*"):
        if f.is_file():
            blob_name = f"{prefix}/{f.relative_to(run_dir)}"
            bucket.blob(blob_name).upload_from_filename(str(f))
            print(f"  Uploaded: {blob_name}")
            uploaded += 1
    print(f"Upload complete: {uploaded} file(s)")


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    skill_path = download_skill_from_gcs()
    skill_text = load_skill(str(skill_path))
    skill_ver = skill_version_from_path(str(skill_path))

    full_model_id = get_model_id(MODEL_NAME)
    short = model_short_name(full_model_id)

    cfg.model_name = MODEL_NAME
    cfg.approach = "skill"
    cfg.prompt_strategy = "skill"

    if DATASET_SPLIT_OVERRIDE:
        cfg.dataset_split = DATASET_SPLIT_OVERRIDE
    if LIMIT_OVERRIDE:
        cfg.limit = int(LIMIT_OVERRIDE)

    run_id = f"skill_{short}_{skill_ver}"
    base_output = LOCAL_OUT / cfg.phase
    run_dir = base_output / run_id

    print(f"[run_job_skill] {short} / {skill_ver} / {cfg.dataset_split}")

    ds = load_dataset_split(
        cfg.dataset_split,
        subset_fraction=cfg.subset_fraction,
        max_samples=cfg.effective_limit(),
        seed=cfg.seed,
        source=getattr(cfg, "dataset_source", "local"),
    )
    save_index_manifest(run_dir, ds, cfg.dataset_split, cfg.seed)

    model, tokenizer = load_model(
        full_model_id,
        max_seq_len=cfg.max_seq_length,
        quantization=cfg.quantization,
        device_map=cfg.device_map,
    )

    pred_path = run_dir / "predictions.jsonl"
    completed = load_completed_indices(pred_path) if cfg.resume else set()
    if completed:
        print(f"  Resuming: {len(completed)} samples already done")

    y_true, y_pred, cwe_ids = [], [], []
    predictions: List[Dict[str, Any]] = []

    with RunTimer() as timer:
        for idx, sample in enumerate(ds):
            if idx in completed:
                continue
            prompt = build_prompt_from_config(
                code=sample["text"], strategy="skill", skill_text=skill_text, seed=cfg.seed,
            )
            raw_output = generate(
                model, tokenizer, prompt, full_model_id,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
            )
            parsed = parse_sample(raw_output)
            row = {
                "idx": idx, "label": sample["label"], "cwe_id": sample.get("cwe_id", ""),
                "raw_output": raw_output if cfg.save_raw_outputs else "",
                **parsed,
            }
            append_prediction(pred_path, row)
            y_true.append(sample["label"])
            y_pred.append(parsed["predicted_label"])
            cwe_ids.append(sample.get("cwe_id", ""))
            predictions.append(row)

            if (idx + 1) % cfg.save_every_n == 0:
                print(f"  [{idx + 1}/{len(ds)}] {timer.elapsed_so_far:.0f}s")

    unload_model()

    result = evaluate_predictions(y_true, y_pred, cwe_ids)
    metrics = result["metrics"]
    errors = result["error_analysis"]

    run_meta = build_run_meta(
        cfg, run_id,
        elapsed_seconds=timer.elapsed,
        n_samples=len(y_true),
        n_errors=metrics.get("parse_failures", 0),
        extra={"skill_version": skill_ver, "skill_gcs_path": SKILL_GCS_PATH},
    )
    write_run_outputs(run_dir, predictions, metrics, errors, cfg.to_dict(), run_meta)

    print(f"\n  F1={metrics['f1']:.4f}  F2={metrics['f2']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    if DIAGNOSE:
        diag_path = run_dir / "diagnostic.txt"
        report = generate_diagnostic_report(pred_path, output_path=diag_path)
        print(f"\n{report}")

    summary_path = run_dir / "skill_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "skill_version": skill_ver,
            "metrics": metrics,
            "error_analysis": errors,
        }, f, indent=2)

    upload_results_to_gcs(skill_ver + SKILL_VERSION_SUFFIX, run_dir)


if __name__ == "__main__":
    main()
