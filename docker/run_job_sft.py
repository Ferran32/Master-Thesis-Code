#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "/app")

from src.config import adapter_path, get_model_id, model_short_name
from src.data_utils import load_few_shot_pool
from src.legacy.run_agent_skill import load_skill, skill_version_from_path
from src.pipeline.config_loader import load_config
from src.pipeline.dataset import load_dataset_split, save_index_manifest
from src.pipeline.evaluation import evaluate_predictions, write_run_outputs
from src.pipeline.inference import generate, load_model, unload_model
from src.pipeline.parsing import parse_sample
from src.pipeline.prompting import build_prompt_from_config
from src.prompt_template import build_sft_messages
from src.pipeline.run_state import append_prediction, load_completed_indices
from src.pipeline.tracking import RunTimer, build_run_meta

MODEL_NAME = os.environ["MODEL_NAME"]
GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_PREFIX = os.environ.get("GCS_PREFIX", "sft")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/configs/phases/sft_vertexai.yaml")
EVAL_ONLY = os.environ.get("EVAL_ONLY", "false").lower() == "true"
GCS_ADAPTER_PREFIX = os.environ.get("GCS_ADAPTER_PREFIX", f"sft/{MODEL_NAME}/adapter/final_adapter")
MAX_SEQ_LENGTH_OVERRIDE = int(os.environ["MAX_SEQ_LENGTH"]) if "MAX_SEQ_LENGTH" in os.environ else None

# Combined-method eval: apply a few-shot and/or skill prompt to the fine-tuned model.
SFT_STRATEGY = os.environ.get("SFT_STRATEGY", "zero_shot")
FEW_SHOT_K = int(os.environ.get("FEW_SHOT_K", "0"))
SKILL_GCS_PATH = os.environ.get("SKILL_GCS_PATH", "")
DATASET_SPLIT_OVERRIDE = os.environ.get("DATASET_SPLIT", "")
LIMIT_OVERRIDE = os.environ.get("LIMIT", "")
IS_COMBO = SFT_STRATEGY != "zero_shot"

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


def _combo_suffix() -> str:
    suffix = SFT_STRATEGY
    if "few_shot" in SFT_STRATEGY:
        suffix += f"_{FEW_SHOT_K}shot"
    return suffix


def download_adapter_from_gcs(local_dir: Path) -> None:
    from google.cloud import storage as gcs
    print(f"Downloading adapter from gs://{GCS_BUCKET}/{GCS_ADAPTER_PREFIX}/ → {local_dir}")
    client = gcs.Client()
    bucket = client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix=GCS_ADAPTER_PREFIX))
    if not blobs:
        raise RuntimeError(f"No adapter blobs found at gs://{GCS_BUCKET}/{GCS_ADAPTER_PREFIX}/")
    for blob in blobs:
        rel = blob.name[len(GCS_ADAPTER_PREFIX):].lstrip("/")
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
    print(f"Downloaded {len(blobs)} adapter file(s)")


def upload_to_gcs(local_root: Path, gcs_prefix_override: str = "") -> None:
    from google.cloud import storage as gcs
    prefix = gcs_prefix_override or f"{GCS_PREFIX}/{MODEL_NAME}"
    print(f"\nUploading to gs://{GCS_BUCKET}/{prefix}/")
    client = gcs.Client()
    bucket = client.bucket(GCS_BUCKET)
    uploaded = 0
    for f in local_root.rglob("*"):
        if f.is_file():
            blob_name = f"{prefix}/{f.relative_to(local_root)}"
            bucket.blob(blob_name).upload_from_filename(str(f))
    print(f"Upload complete: {uploaded} file(s)")


def run_training(cfg) -> Path:
    from src.legacy.run_sft_train import train
    max_seq = cfg.max_seq_length

    resume_path = None
    if cfg.resume:
        full_model_id = get_model_id(MODEL_NAME)
        checkpoint_dir = adapter_path(full_model_id)
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            resume_path = str(checkpoints[-1])
            print(f"Resuming from checkpoint: {resume_path}")

    return train(model_id=MODEL_NAME, max_seq_length=max_seq, resume_from_checkpoint=resume_path)


def run_evaluation(cfg, base_output: Path) -> Dict[str, Any]:
    full_model_id = get_model_id(MODEL_NAME)
    short = model_short_name(full_model_id)

    adapter_dir = adapter_path(full_model_id) / "final_adapter"
    if EVAL_ONLY and not adapter_dir.exists():
        download_adapter_from_gcs(adapter_dir)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_dir}")

    cfg.model_name = MODEL_NAME
    cfg.approach = "sft"
    cfg.prompt_strategy = SFT_STRATEGY
    cfg.few_shot_k = FEW_SHOT_K

    if DATASET_SPLIT_OVERRIDE:
        cfg.dataset_split = DATASET_SPLIT_OVERRIDE
    if LIMIT_OVERRIDE:
        cfg.limit = int(LIMIT_OVERRIDE)

    # Load combo prompt ingredients (few-shot pool / skill) when applicable.
    few_shot_pool = load_few_shot_pool() if "few_shot" in SFT_STRATEGY else None
    skill_text = None
    skill_ver = ""
    if "skill" in SFT_STRATEGY:
        skill_path = download_skill_from_gcs()
        skill_text = load_skill(str(skill_path))
        skill_ver = skill_version_from_path(str(skill_path))

    run_id = f"sft_{short}_{_combo_suffix()}" if IS_COMBO else f"sft_{short}"
    run_dir = base_output / run_id

    print(f"\n  Evaluating SFT model: {short}  (strategy={SFT_STRATEGY}, k={FEW_SHOT_K})")
    print(f"  Adapter: {adapter_dir}")

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
        adapter_path=str(adapter_dir),
    )

    pred_path = run_dir / "predictions.jsonl"
    completed = load_completed_indices(pred_path) if cfg.resume else set()

    y_true, y_pred, cwe_ids = [], [], []
    predictions: List[Dict[str, Any]] = []

    with RunTimer() as timer:
        for idx, sample in enumerate(ds):
            if idx in completed:
                continue
            prompt = build_prompt_from_config(
                code=sample["text"],
                strategy=SFT_STRATEGY,
                few_shot_pool=few_shot_pool,
                few_shot_k=FEW_SHOT_K,
                skill_text=skill_text,
                seed=cfg.seed,
            )
            messages = build_sft_messages(sample["text"], user_prompt=prompt)
            raw_output = generate(
                model, tokenizer, prompt, full_model_id,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                messages=messages,
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
                print(f"    [{idx + 1}/{len(ds)}] {timer.elapsed_so_far:.0f}s")

    unload_model()

    result = evaluate_predictions(y_true, y_pred, cwe_ids)
    metrics = result["metrics"]
    errors = result["error_analysis"]

    meta_extra = {"skill_version": skill_ver, "skill_gcs_path": SKILL_GCS_PATH} if skill_ver else None
    run_meta = build_run_meta(
        cfg, run_id,
        elapsed_seconds=timer.elapsed,
        n_samples=len(y_true),
        n_errors=metrics.get("parse_failures", 0),
        extra=meta_extra,
    )
    write_run_outputs(run_dir, predictions, metrics, errors, cfg.to_dict(), run_meta)

    print(f"  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  ({timer.elapsed:.0f}s)")

    return {"model": MODEL_NAME, "metrics": metrics, "run_dir": str(run_dir)}


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    if MAX_SEQ_LENGTH_OVERRIDE is not None:
        cfg.max_seq_length = MAX_SEQ_LENGTH_OVERRIDE

    if not EVAL_ONLY:
        print(f"[run_job_sft] training: {MODEL_NAME}")
        run_training(cfg)

        # Upload adapter to GCS immediately after training so it's persisted
        full_model_id = get_model_id(MODEL_NAME)
        adapter_dir = adapter_path(full_model_id)
        upload_to_gcs(adapter_dir, gcs_prefix_override=f"{GCS_PREFIX}/{MODEL_NAME}/adapter")

    print(f"[run_job_sft] evaluation: {MODEL_NAME}")
    base_output = LOCAL_OUT / cfg.phase
    result = run_evaluation(cfg, base_output)

    summary_path = base_output / f"sft_{MODEL_NAME}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    # Upload evaluation results
    results_prefix = (
        f"{GCS_PREFIX}/{MODEL_NAME}/{_combo_suffix()}" if IS_COMBO
        else f"{GCS_PREFIX}/{MODEL_NAME}/results"
    )
    upload_to_gcs(LOCAL_OUT, gcs_prefix_override=results_prefix)


if __name__ == "__main__":
    main()
