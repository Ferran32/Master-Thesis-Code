#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from google.cloud import aiplatform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS, machine_spec_for

# GCP target is read from the environment so no project or bucket name is committed.
# Required: GCP_PROJECT, GCS_RESULTS_BUCKET. Optional: GCP_REGION (defaults to europe-west4).
PROJECT_ID = os.environ.get("GCP_PROJECT", "")
LOCATION = os.environ.get("GCP_REGION", "europe-west4")
IMAGE_URI = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/docker-image/base-eval:latest"
QWEN36_IMAGE_URI = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/docker-image/qwen36:latest"
GCS_BUCKET = os.environ.get("GCS_RESULTS_BUCKET", "")

ALL_MODELS: List[str] = list(MODELS.keys())


def image_for(model_name: str) -> str:
    return QWEN36_IMAGE_URI if model_name == "qwen3.6-27b" else IMAGE_URI

# Qwen2.5-7B OOMs on a 22 GB L4 at 4096; the 32B models QLoRA-train at 2048 on
# the A100 40 GB. All other models are fine at 4096.
MODEL_MAX_SEQ: dict = {
    "qwen2.5-7b": 2048,
    "qwen3.6-27b": 2048,
    "gemma4-31b": 2048,
}


def submit_jobs(models: List[str], hf_token: str = "",
                eval_only: bool = False, adapter_prefix: str = "") -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=f"gs://{GCS_BUCKET}")

    for model_name in models:
        env = [
            {"name": "MODEL_NAME", "value": model_name},
            {"name": "GCS_BUCKET", "value": GCS_BUCKET},
            {"name": "GCS_PREFIX", "value": "sft"},
            {"name": "MAX_SEQ_LENGTH", "value": str(MODEL_MAX_SEQ.get(model_name, 4096))},
        ]
        if hf_token:
            env.append({"name": "HF_TOKEN", "value": hf_token})
        if eval_only:
            env.append({"name": "EVAL_ONLY", "value": "true"})
            prefix = adapter_prefix or f"sft/{model_name}/adapter/final_adapter"
            env.append({"name": "GCS_ADAPTER_PREFIX", "value": prefix})

        job = aiplatform.CustomJob(
            display_name=f"sft-{'eval' if eval_only else 'train'}-{model_name}",
            worker_pool_specs=[{
                "machine_spec": machine_spec_for(model_name, sft=True),
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_for(model_name),
                    "command": ["python3", "run_job_sft.py"],
                    "env": env,
                },
            }],
        )
        job.submit()
        print(f"Submitted: {model_name:30s} → {job.resource_name}")

    print(f"\nAll {len(models)} job(s) submitted.")
    print("\nMonitor at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"\nRetrieve adapter after completion:")
    for m in models:
        print(f"  gsutil -m cp -r gs://{GCS_BUCKET}/sft/{m}/adapter/ models/sft/{m}/")
        print(f"\nRetrieve results:")
        print(f"  gsutil -m cp -r gs://{GCS_BUCKET}/sft/{m}/results/ results/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Vertex AI Phase 3 (SFT) jobs")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, metavar="MODEL")
    hf_token = os.getenv("HF_TOKEN")
    parser.add_argument(
        "--hf-token", default=hf_token, 
        help="HuggingFace token. Defaults to HF_TOKEN environment variable."
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; download existing adapter from GCS and evaluate only",
    )
    parser.add_argument(
        "--adapter-prefix", default="",
        help="GCS prefix of the existing adapter (used with --eval-only)",
    )
    args = parser.parse_args()

    if not PROJECT_ID or not GCS_BUCKET:
        sys.exit(
            "Set GCP_PROJECT and GCS_RESULTS_BUCKET environment variables before submitting "
            "(see README). GCP_REGION is optional and defaults to europe-west4."
        )

    invalid = [m for m in args.models if m not in ALL_MODELS]
    if invalid:
        print(f"ERROR: Unknown model(s): {invalid}")
        print(f"Valid choices: {ALL_MODELS}")
        sys.exit(1)

    submit_jobs(
        args.models,
        hf_token=args.hf_token,
        eval_only=args.eval_only,
        adapter_prefix=args.adapter_prefix,
    )


if __name__ == "__main__":
    main()
