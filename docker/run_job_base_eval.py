#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "/app")

from src.config import get_model_id, model_short_name
from src.data_utils import load_few_shot_pool, prepare_splits
from src.legacy.run_agent_skill import load_skill, skill_version_from_path
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
GCS_PREFIX = os.environ.get("GCS_PREFIX", "base-eval")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/configs/phases/base_eval_vertexai.yaml")
PROMPT_STRATEGY = os.environ.get("PROMPT_STRATEGY")
FEW_SHOT_K = int(os.environ.get("FEW_SHOT_K", "0"))
SKILL_GCS_PATH = os.environ.get("SKILL_GCS_PATH", "")

LOCAL_OUT = Path("/tmp/results")
LOCAL_SKILL_DIR = Path("/tmp/skills")


def download_skill_from_gcs() -> Path:
    from google.cloud import storage as gcs
    LOCAL_SKILL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_SKILL_DIR / Path(SKILL_GCS_PATH).name
    print(f"Downloading skill from gs://{GCS_BUCKET}/{SKILL_GCS_PATH} → {local_path}")
    client = gcs.Client()
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(SKILL_GCS_PATH).download_to_filename(str(local_path))
    return local_path


def run_single_config(
    model_name: str,
    strategy: str,
    few_shot_k: int,
    cfg,
    base_output: Path,
    few_shot_pool=None,
    skill_text=None,
    skill_ver: str = "",
) -> Dict[str, Any]:
    full_model_id = get_model_id(model_name)
    short = model_short_name(full_model_id)

    cfg.model_name = model_name
    cfg.prompt_strategy = strategy
    cfg.few_shot_k = few_shot_k

    suffix = strategy
    if "few_shot" in strategy:
        suffix += f"_{few_shot_k}shot"

    run_id = f"{cfg.phase}_{short}_{suffix}"
    run_dir = base_output / run_id

    print(f"\n  [{short} / {suffix}]")

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
        print(f"    Resuming: {len(completed)} samples already done")

    y_true, y_pred, cwe_ids = [], [], []
    predictions: List[Dict[str, Any]] = []

    with RunTimer() as timer:
        for idx, sample in enumerate(ds):
            if idx in completed:
                continue

            prompt = build_prompt_from_config(
                code=sample["text"],
                strategy=strategy,
                few_shot_pool=few_shot_pool,
                few_shot_k=few_shot_k,
                skill_text=skill_text,
                seed=cfg.seed,
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
                "idx": idx,
                "label": sample["label"],
                "cwe_id": sample.get("cwe_id", ""),
                "raw_output": raw_output if cfg.save_raw_outputs else "",
                **parsed,
            }
            append_prediction(pred_path, row)
            y_true.append(sample["label"])
            y_pred.append(parsed["predicted_label"])
            cwe_ids.append(sample.get("cwe_id", ""))
            predictions.append(row)

            if (idx + 1) % cfg.save_every_n == 0:
                print(f"    [{idx + 1}/{len(ds)}] {timer.elapsed_so_far:.0f}s")

    result = evaluate_predictions(y_true, y_pred, cwe_ids)
    metrics = result["metrics"]
    errors  = result["error_analysis"]

    meta_extra = {"skill_version": skill_ver, "skill_gcs_path": SKILL_GCS_PATH} if skill_ver else None
    run_meta = build_run_meta(
        cfg, run_id,
        elapsed_seconds=timer.elapsed,
        n_samples=len(y_true),
        n_errors=metrics.get("parse_failures", 0),
        extra=meta_extra,
    )
    write_run_outputs(run_dir, predictions, metrics, errors, cfg.to_dict(), run_meta)
    print(f"    F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  ({timer.elapsed:.0f}s)")

    return {"model": model_name, "strategy": suffix, "metrics": metrics, "run_dir": str(run_dir)}


def _gcs_prefix() -> str:
    if PROMPT_STRATEGY:
        suffix = PROMPT_STRATEGY
        if "few_shot" in PROMPT_STRATEGY:
            suffix = f"{PROMPT_STRATEGY}_{FEW_SHOT_K}shot"
        return f"{GCS_PREFIX}/{MODEL_NAME}/{suffix}"
    return f"{GCS_PREFIX}/{MODEL_NAME}"


def upload_results_to_gcs() -> None:
    from google.cloud import storage as gcs
    prefix = _gcs_prefix()
    print(f"\nUploading results to gs://{GCS_BUCKET}/{prefix}/")
    client = gcs.Client()
    bucket = client.bucket(GCS_BUCKET)
    uploaded = 0
    for f in LOCAL_OUT.rglob("*"):
        if f.is_file():
            blob_name = f"{prefix}/{f.relative_to(LOCAL_OUT)}"
            bucket.blob(blob_name).upload_from_filename(str(f))
            print(f"  Uploaded: {blob_name}")
            uploaded += 1
    print(f"Upload complete: {uploaded} file(s)")


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    dataset_split_override = os.environ.get("DATASET_SPLIT", "")
    limit_override = os.environ.get("LIMIT", "")
    if dataset_split_override:
        cfg.dataset_split = dataset_split_override
    if limit_override:
        cfg.limit = int(limit_override)

    prepare_splits()  # no-op if splits already exist; generates them otherwise

    base_output = LOCAL_OUT / cfg.phase

    full_model_id = get_model_id(MODEL_NAME)
    short = model_short_name(full_model_id)
    print(f"[run_job_base_eval] model: {short}")

    skill_text = None
    skill_ver = ""
    if SKILL_GCS_PATH:
        skill_path = download_skill_from_gcs()
        skill_text = load_skill(str(skill_path))
        skill_ver = skill_version_from_path(str(skill_path))
        print(f"Loaded skill: {skill_ver}")

    if PROMPT_STRATEGY:
        # per-strategy mode: single job per strategy
        few_shot_pool = None
        if "few_shot" in PROMPT_STRATEGY:
            few_shot_pool = load_few_shot_pool()
            print(f"Loaded few-shot pool: {len(few_shot_pool)} examples")

        result = run_single_config(
            MODEL_NAME, PROMPT_STRATEGY, FEW_SHOT_K, cfg, base_output, few_shot_pool,
            skill_text=skill_text, skill_ver=skill_ver,
        )

        unload_model()

        suffix = PROMPT_STRATEGY
        if "few_shot" in PROMPT_STRATEGY:
            suffix = f"{PROMPT_STRATEGY}_{FEW_SHOT_K}shot"
        summary_path = base_output / f"base_eval_{MODEL_NAME}_{suffix}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump([result], f, indent=2)

    else:
        strategies = getattr(cfg, "prompt_strategies", None) or [
            "zero_shot", "few_shot", "cot"
        ]

        few_shot_pool = None
        if any("few_shot" in s for s in strategies):
            few_shot_pool = load_few_shot_pool()
            print(f"Loaded few-shot pool: {len(few_shot_pool)} examples")

        all_results: List[Dict[str, Any]] = []
        for strategy in strategies:
            if "few_shot" in strategy:
                for k in cfg.few_shot_counts:
                    r = run_single_config(MODEL_NAME, strategy, k, cfg, base_output, few_shot_pool)
                    all_results.append(r)
            else:
                r = run_single_config(MODEL_NAME, strategy, 0, cfg, base_output, few_shot_pool)
                all_results.append(r)

        unload_model()

        summary_path = base_output / f"base_eval_{MODEL_NAME}_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

    upload_results_to_gcs()


if __name__ == "__main__":
    main()
